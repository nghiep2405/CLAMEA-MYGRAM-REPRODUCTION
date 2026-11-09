def _cell12_controlled_self_training():
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys
    import textwrap
    import time
    import uuid
    from pathlib import Path

    # ========================================================
    # 1. PREFLIGHT
    # ========================================================

    assert (
        PRECHECK["status"]
        == "PASS_MOLAB_PREFLIGHT_TECHNICAL"
    )

    assert source_dir.is_dir()
    assert data_dir.is_dir()
    assert torch.cuda.is_available()

    def digest(path):
        h = hashlib.sha256()

        with open(path, "rb") as f:
            for b in iter(
                lambda: f.read(8 * 1024 * 1024),
                b"",
            ):
                h.update(b)

        return h.hexdigest()

    # Verify existing v1.2.4 patched source.

    for name, key in (
        ("main.py", "patched_main_sha256"),
        ("model.py", "patched_model_sha256"),
        ("utils.py", "patched_utils_sha256"),
    ):
        assert (
            digest(source_dir / name)
            == PATCH_INFO[key]
        ), name

    # ========================================================
    # 2. CREATE ISOLATED EXPERIMENT
    # ========================================================

    job_id = (
        time.strftime("cell12_%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:6]
    )

    root = WORK / "diagnostics" / job_id

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    fixed_source = root / "source"

    shutil.copytree(
        source_dir,
        fixed_source,
    )

    main_path = fixed_source / "main.py"

    code = main_path.read_text(
        encoding="utf-8"
    )

    # ========================================================
    # 3. APPLY CELL 11 DETERMINISTIC PATCH
    # ========================================================
    #
    # Keep the original source unchanged.
    #
    # Negative sampling:
    #   - deterministic per epoch and step
    #   - ordered optimizer updates
    #   - independent of multiprocessing arrival order
    #
    # ========================================================

    anchor = """        for steps_task in relation_step_tasks:
            mp.Process(target=generate_relation_triple_batch_queue,
                       args=(kgs.relation_triples_list1, kgs.relation_triples_list2,
                             kgs.relation_triples_set1, kgs.relation_triples_set2,
                             kgs.kg1_entities_list, kgs.kg2_entities_list,
                             args.batch_size, steps_task,
                             relation_batch_queue, args.neg_triple_num)).start()
        for _ in range(relation_triple_steps):
            optimizer.zero_grad()
            batch_pos, batch_neg = relation_batch_queue.get()"""

    replacement = """        # CELL12: same deterministic batching as Cell11.
        for _det_step in range(relation_triple_steps):
            optimizer.zero_grad()

            _old_py_rng = random.getstate()
            _old_np_rng = np.random.get_state()

            try:
                _det_seed = (
                    int(args.random_seed) * 1000003
                    + i * 9176
                    + _det_step
                ) % (2**32)

                random.seed(_det_seed)
                np.random.seed(_det_seed)

                batch_pos, batch_neg = generate_relation_triple_batch(
                    kgs.relation_triples_list1,
                    kgs.relation_triples_list2,
                    kgs.relation_triples_set1,
                    kgs.relation_triples_set2,
                    kgs.kg1_entities_list,
                    kgs.kg2_entities_list,
                    args.batch_size,
                    _det_step,
                    args.neg_triple_num
                )

            finally:
                random.setstate(_old_py_rng)
                np.random.set_state(_old_np_rng)"""

    assert code.count(anchor) == 1, (
        "Cell 11 deterministic patch anchor not found"
    )

    code = code.replace(
        anchor,
        replacement,
        1,
    )

    # ========================================================
    # 4. DISABLE OBSERVATIONAL TEST MONITORING
    # ========================================================
    #
    # This experiment compares Validation only.
    #
    # Test entity IDs can still occur in the transductive
    # candidate pool, as in the original CLAMEA protocol.
    #
    # Gold Test labels are NOT used for scoring or selection.
    #
    # ========================================================

    test_call = (
        "            test_hits1 = "
        "monitor_test_hits1(i, model, kgs, args)"
    )

    test_json = (
        "'test_raw_hits1_observational': "
        "float(test_hits1),"
    )

    assert code.count(test_call) == 1
    assert code.count(test_json) == 1

    code = code.replace(
        test_call,
        "            test_hits1 = None  "
        "# Cell12: Test monitoring disabled",
        1,
    )

    code = code.replace(
        test_json,
        "'test_raw_hits1_observational': None,",
        1,
    )

    compile(
        code,
        str(main_path),
        "exec",
    )

    main_path.write_text(
        code,
        encoding="utf-8",
    )

    # ========================================================
    # 5. EXPERIMENT CONFIGURATION
    # ========================================================

    cfg = dict(base_config)

    cfg.update({
        "random_seed": 66,
        "start_valid": 70,
        "eval_freq": 70,
        "early_stop": False,
        "visual_dim": 4096,
        "training_data": str(data_dir) + "/",
    })

    assert cfg["dataset_division"] == "721"

    assert cfg["stop_metric"] == "mrr"
    assert cfg["eval_norm"] is True

    assert cfg["batch_size"] == 5000
    assert cfg["neg_triple_num"] == 1

    # Verify parent baseline configuration.
    # Individual phase max_epoch is set inside the worker.

    assert cfg["max_epoch"] == 300

    job = {
        "id": job_id,

        "root": str(root),
        "source": str(fixed_source),

        "base_config": cfg,

        "branch_epoch": 69,
        "end_epoch": 100,

        "patched_main_sha256": digest(main_path),

        "original_main_sha256": digest(
            source_dir / "main.py"
        ),

        "dataset_sha256": DATA_SHA256,

        "policy": (
            "Validation-only; NO Test monitoring; "
            "same model/optimizer/RNG at epoch69"
        ),
    }

    job_path = root / "job.json"

    job_path.write_text(
        json.dumps(job, indent=2),
        encoding="utf-8",
    )

    # ========================================================
    # 6. INDEPENDENT TRAINING WORKER
    # ========================================================
    #
    # Three phases:
    #
    #   shared: epochs 1–69, iterative=False
    #   on:     resume epoch69, iterative=True
    #   off:    resume epoch69, iterative=False
    #
    # All phases use the same deterministic patched source.
    #
    # ========================================================

    worker = textwrap.dedent(r'''
        import hashlib
        import json
        import os
        import random
        import sys

        from pathlib import Path

        import numpy as np
        import torch

        # ====================================================
        # A. LOAD JOB
        # ====================================================

        job = json.loads(
            Path(sys.argv[1]).read_text()
        )

        phase = sys.argv[2]

        assert phase in (
            "shared",
            "on",
            "off",
        )

        root = Path(job["root"])

        cfg = dict(job["base_config"])

        is_shared = phase == "shared"

        cfg.update({
            "iterative": phase == "on",

            "max_epoch": (
                69
                if is_shared
                else job["end_epoch"]
            ),

            "output": (
                str(root / phase / "results") + "/"
            ),
        })

        # ====================================================
        # B. IMPORT PATCHED SOURCE
        # ====================================================

        sys.path.insert(
            0,
            job["source"],
        )

        # Keep original CLAMEA import order.

        import kgs
        import main as official

        from kgs import KGs

        def sha(path):
            h = hashlib.sha256()

            with open(path, "rb") as f:
                for part in iter(
                    lambda: f.read(8 * 1024 * 1024),
                    b"",
                ):
                    h.update(part)

            return h.hexdigest()

        # ====================================================
        # C. DETERMINISTIC EXECUTION
        # ====================================================

        assert os.environ["PYTHONHASHSEED"] == "0"

        assert (
            os.environ["CUBLAS_WORKSPACE_CONFIG"]
            == ":4096:8"
        )

        assert torch.cuda.is_available()

        seed = int(
            cfg["random_seed"]
        )

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)

        args = official.ARGs(cfg)

        # ====================================================
        # D. INITIALIZE DATA AND MODEL
        # ====================================================

        run_dir = root / phase

        run_dir.mkdir(
            exist_ok=False,
        )

        kg = KGs(
            args.training_data,
            args.dataset_division,
            args.random_seed,

            ordered=True,
            modality="ria",
        )

        model = official.CLAMEA(
            kg,
            args,
        ).cuda()

        assert model.fc_i.in_features == 4096

        output = official.generate_out_folder(
            args.output,
            args.training_data,
            args.dataset_division,
            "CLAMEA_cell12",
        )

        (
            run_dir / "out_folder.txt"
        ).write_text(
            str(output)
        )

        # ====================================================
        # E. CHECKPOINT CONFIGURATION
        # ====================================================

        os.environ["CLAMEA_WANDB_ENABLED"] = "0"

        os.environ["CLAMEA_RESUME_SNAPSHOTS"] = "1"

        os.environ["CLAMEA_RESUME_INTERVAL"] = (
            "69"
            if is_shared
            else "100"
        )

        # ====================================================
        # F. SHARED OR BRANCH INITIALIZATION
        # ====================================================

        if is_shared:

            os.environ.pop(
                "CLAMEA_RESUME_STATE",
                None,
            )

        else:

            # Both branches MUST load the exact same
            # end-of-epoch69 snapshot.

            ckpt = root / "shared_epoch69.pt"

            assert ckpt.is_file(), (
                "Missing shared epoch69 checkpoint"
            )

            state = torch.load(
                ckpt,
                map_location="cpu",
                weights_only=False,
            )

            assert (
                state["format"]
                == "CLAMEA_METHOD_V12_STATE"
            )

            assert state["epoch"] == 69
            assert state["random_seed"] == 66

            assert not state["new_e1"]
            assert not state["new_e2"]

            assert state["best_model_state"] is None

            del state

            os.environ["CLAMEA_RESUME_STATE"] = (
                str(ckpt)
            )

            print(
                "[FORK]",
                phase,
                "from epoch=69 sha256=",
                sha(ckpt),
                flush=True,
            )

        # ====================================================
        # G. TRAIN
        # ====================================================

        print(
            "[PHASE]",
            phase,

            "iterative=",
            args.iterative,

            "epochs=",
            args.max_epoch,

            flush=True,
        )

        official.train(
            model,
            kg,
            args,
            output,
        )

        # ====================================================
        # H. VERIFY FINAL SNAPSHOT
        # ====================================================

        saved = (
            Path(output) / "resume_state.pt"
        )

        assert saved.is_file(), (
            "Missing end-of-phase state"
        )

        state = torch.load(
            saved,
            map_location="cpu",
            weights_only=False,
        )

        expected_epoch = (
            69
            if is_shared
            else job["end_epoch"]
        )

        assert int(state["epoch"]) == expected_epoch

        assert int(state["random_seed"]) == 66

        # ====================================================
        # I. SHARED STATE
        # ====================================================

        if is_shared:

            assert len(state["new_e1"]) == 0

            shared_path = (
                root / "shared_epoch69.pt"
            )

            import shutil

            shutil.copy2(
                saved,
                shared_path,
            )

            print(
                "[SHARED READY]",
                sha(shared_path),
                flush=True,
            )

        # ====================================================
        # J. BRANCH VALIDATION
        # ====================================================

        else:

            if phase == "off":

                assert not state["new_e1"], (
                    "OFF branch unexpectedly "
                    "has pseudo-pairs"
                )

            else:

                assert len(state["new_e1"]) > 0, (
                    "ON branch produced no pseudo-pairs"
                )

        pseudo_count = len(
            state["new_e1"]
        )

        del state

        # ====================================================
        # K. VALIDATION HISTORY
        # ====================================================

        history = []

        curve = (
            Path(output) / "epoch_eval.jsonl"
        )

        if not is_shared:

            assert curve.exists(), (
                "Missing Validation curve"
            )

            history = [
                json.loads(line)

                for line in curve.read_text().splitlines()

                if line.strip()
            ]

            expected = list(
                range(70, 101)
            )

            observed = [
                int(row["epoch"])
                for row in history
            ]

            assert observed == expected, (
                "Incomplete Validation history"
            )

            # No Test metrics in this experiment.

            assert all(
                row["test_raw_hits1_observational"]
                is None

                for row in history
            )

        # ====================================================
        # L. SAVE PHASE REPORT
        # ====================================================

        info = {
            "phase": phase,

            "iterative": cfg["iterative"],

            "end_epoch": expected_epoch,

            "pseudo_pairs": pseudo_count,

            "validation": history,

            "end_snapshot_sha256": sha(saved),
        }

        report_path = (
            root / (phase + "_report.json")
        )

        report_path.write_text(
            json.dumps(
                info,
                indent=2,
            )
        )

        print(
            "[PHASE DONE]",
            phase,

            "pseudo_pairs=",
            pseudo_count,

            flush=True,
        )
    ''').lstrip()

    # ========================================================
    # 7. WRITE AND COMPILE WORKER
    # ========================================================

    worker_path = root / "worker.py"

    compile(
        worker,
        str(worker_path),
        "exec",
    )

    worker_path.write_text(
        worker,
        encoding="utf-8",
    )

    # ========================================================
    # 8. SUBPROCESS ENVIRONMENT
    # ========================================================

    env = os.environ.copy()

    env.update({
        "PYTHONHASHSEED": "0",

        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",

        "WANDB_MODE": "disabled",

        "CLAMEA_WANDB_ENABLED": "0",
    })

    env.pop(
        "CLAMEA_RESUME_STATE",
        None,
    )

    # ========================================================
    # 9. EXECUTE ONE PHASE
    # ========================================================

    def execute(phase):

        log_path = (
            root / (phase + ".log")
        )

        print(
            "\n[START]",
            phase.upper(),
            flush=True,
        )

        cmd = [
            sys.executable,
            "-u",

            str(worker_path),
            str(job_path),

            phase,
        ]

        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log:

            proc = subprocess.Popen(
                cmd,

                cwd=str(fixed_source),
                env=env,

                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,

                text=True,
                bufsize=1,
            )

            try:

                for line in proc.stdout:

                    log.write(line)

                    msg = line.strip()

                    relevant = msg.startswith((
                        "[PHASE]",
                        "[FORK]",
                        "[SHARED READY]",
                        "[PHASE DONE]",
                        "[VALID_MONITOR]",

                        "generate new semi-pairs:",
                        "SAVED RESUME STATE",

                        "Traceback",
                        "RuntimeError",
                        "AssertionError",
                    ))

                    epoch_marker = False

                    if msg.startswith("epoch "):

                        epoch = int(
                            msg.split()[1].rstrip(":")
                        )

                        epoch_marker = (
                            epoch % 5 == 0
                            or epoch in (69, 70, 71)
                        )

                    if relevant or epoch_marker:

                        print(
                            "[" + phase.upper() + "]",
                            msg,
                            flush=True,
                        )

                returncode = proc.wait()

            except BaseException:

                proc.terminate()

                try:
                    proc.wait(timeout=10)

                except subprocess.TimeoutExpired:
                    proc.kill()

                raise

        if returncode != 0:

            raise RuntimeError(
                f"Phase {phase} failed "
                f"(exit {returncode}); "
                f"log: {log_path}"
            )

        result_path = (
            root / (phase + "_report.json")
        )

        return json.loads(
            result_path.read_text()
        )

    # ========================================================
    # 10. START EXPERIMENT
    # ========================================================

    print("=" * 65)

    print(
        "CELL 12 — CONTROLLED SELF-TRAINING ABLATION"
    )

    print(
        "Seed=66 | Shared epochs 1–69 | ON/OFF 70–100"
    )

    print(
        "Deterministic batching | "
        "Validation-only | Test disabled"
    )

    print(
        "Job:",
        job_id,
    )

    print("=" * 65, flush=True)

    # ========================================================
    # 11. TRAIN SHARED MODEL — EPOCHS 1 TO 69
    # ========================================================

    shared = execute("shared")

    # ========================================================
    # 12. BACKUP SHARED CHECKPOINT
    # ========================================================

    # This checkpoint contains:
    #   model
    #   optimizer
    #   RNG
    #   relation triples
    #   pseudo-pair state
    #
    # Both branches load exactly this file.

    remote = (
        "paper_method_runs/"
        "paper_method_20261023_160901_s66/"
        "diagnostics/"
        + job_id
    )

    try:

        fs_drive.makedirs(
            remote,
            exist_ok=True,
        )

        for path in (
            root / "job.json",
            root / "shared.log",
            root / "shared_epoch69.pt",
        ):

            fs_drive.put_file(
                str(path),
                remote + "/" + path.name,
            )

        print(
            "[DRIVE] Shared state backed up",
            flush=True,
        )

    except Exception as exc:

        print(
            "[DRIVE WARNING] "
            "Shared state backup failed:",
            repr(exc),
            flush=True,
        )

    # ========================================================
    # 13. BRANCH A — SELF-TRAINING ON
    # ========================================================

    on = execute("on")

    # ========================================================
    # 14. BRANCH B — SELF-TRAINING OFF
    # ========================================================

    off = execute("off")

    # ========================================================
    # 15. CHECK COMMON STARTING CONDITIONS
    # ========================================================

    left = {
        int(row["epoch"]): row

        for row in on["validation"]
    }

    right = {
        int(row["epoch"]): row

        for row in off["validation"]
    }

    expected_epochs = list(
        range(70, 101)
    )

    assert sorted(left) == expected_epochs
    assert sorted(right) == expected_epochs

    # Epoch70 evaluation occurs before pseudo-label
    # generation at the END of epoch70.
    #
    # Therefore ON and OFF should match.

    at70_mrr_diff = abs(
        float(left[70]["validation_raw_mrr"])
        - float(right[70]["validation_raw_mrr"])
    )

    at70_hits_diff = abs(
        float(left[70]["validation_raw_hits1"])
        - float(right[70]["validation_raw_hits1"])
    )

    assert (
        at70_mrr_diff < 1e-8
        and at70_hits_diff < 1e-8
    ), (
        "Epoch70 differs BEFORE pseudo training: "
        f"dMRR={at70_mrr_diff}, "
        f"dHits1={at70_hits_diff}"
    )

    # ========================================================
    # 16. COMPARE VALIDATION METRICS
    # ========================================================

    comparison = []

    for epoch in expected_epochs:

        a = left[epoch]
        b = right[epoch]

        on_mrr = float(
            a["validation_raw_mrr"]
        )

        off_mrr = float(
            b["validation_raw_mrr"]
        )

        comparison.append({
            "epoch": epoch,

            "on_mrr": on_mrr,

            "off_mrr": off_mrr,

            "delta_mrr_on_minus_off": (
                on_mrr - off_mrr
            ),

            "on_hits1": float(
                a["validation_raw_hits1"]
            ),

            "off_hits1": float(
                b["validation_raw_hits1"]
            ),
        })

    # ========================================================
    # 17. BUILD FINAL REPORT
    # ========================================================

    report = {
        "status": "PASS_SHARED_BRANCH_CONTROL",

        "job_id": job_id,

        "initial_shared_epoch": 69,

        "shared_snapshot_sha256": digest(
            root / "shared_epoch69.pt"
        ),

        "self_training_pairs_on": (
            on["pseudo_pairs"]
        ),

        "self_training_pairs_off": (
            off["pseudo_pairs"]
        ),

        "epoch70_delta_mrr": (
            at70_mrr_diff
        ),

        "epoch70_delta_hits1": (
            at70_hits_diff
        ),

        "comparison": comparison,

        "interpretation": (
            "Overall effect of pseudo-pair supervision "
            "AND expanded relation triples; "
            "one seed, deterministic variant; "
            "no causal attribution to either "
            "component alone."
        ),
    }

    result_path = (
        root / "cell12_report.json"
    )

    result_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),

        encoding="utf-8",
    )

    # ========================================================
    # 18. DISPLAY COMPARISON
    # ========================================================

    print()
    print("=" * 65)

    print(
        "CELL 12 — CONTROLLED COMPARISON (Validation only)"
    )

    print("=" * 65)

    print(
        "[CONTROL] epoch70 ON == OFF: PASS"
    )

    print(
        "[PSEUDO] ON:",
        on["pseudo_pairs"],

        "| OFF:",
        off["pseudo_pairs"],
    )

    for row in comparison:

        if row["epoch"] in (
            70, 71, 75, 80,
            85, 90, 95, 100,
        ):

            print(
                "epoch {epoch:3d} | "
                "ON MRR={on_mrr:.6f} | "
                "OFF MRR={off_mrr:.6f} | "
                "delta={delta_mrr_on_minus_off:+.6f}"
                .format(**row)
            )

    print(
        "[LOCAL]",
        result_path,
    )

    # ========================================================
    # 19. BACKUP FINAL REPORTS
    # ========================================================

    for path in (
        result_path,

        root / "on.log",
        root / "off.log",

        root / "on_report.json",
        root / "off_report.json",

        fixed_source / "main.py",
    ):

        try:

            fs_drive.put_file(
                str(path),
                remote + "/" + path.name,
            )

        except Exception as exc:

            print(
                "[DRIVE WARNING]",
                path.name,
                repr(exc),
            )

    print(
        "[DONE] Cell 12 completed; "
        "prior runs and checkpoints unchanged."
    )

    return report


# ============================================================
# EXECUTE CELL 12
# ============================================================

_cell12_controlled_self_training()

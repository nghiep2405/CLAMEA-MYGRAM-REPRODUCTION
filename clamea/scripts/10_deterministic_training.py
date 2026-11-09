def _cell11_deterministic_training():
    import hashlib
    import json
    import os
    import re
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

    assert source_dir.is_dir()
    assert data_dir.is_dir()

    assert (
        PRECHECK["status"]
        == "PASS_MOLAB_PREFLIGHT_TECHNICAL"
    )

    assert base_config["batch_threads_num"] == 2

    def sha256(path):
        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(
                lambda: f.read(4 * 1024 * 1024),
                b"",
            ):
                h.update(chunk)

        return h.hexdigest()

    # Verify original patched source.

    for filename, key in (
        ("main.py", "patched_main_sha256"),
        ("model.py", "patched_model_sha256"),
        ("utils.py", "patched_utils_sha256"),
    ):
        assert (
            sha256(source_dir / filename)
            == PATCH_INFO[key]
        ), filename

    # ========================================================
    # 2. CREATE ISOLATED EXPERIMENT
    # ========================================================

    job_id = (
        time.strftime("cell11_%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:6]
    )

    root = WORK / "diagnostics" / job_id

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    fixed_source = root / "deterministic_source"

    shutil.copytree(
        source_dir,
        fixed_source,
    )

    # ========================================================
    # 3. PATCH NEGATIVE SAMPLING
    # ========================================================
    #
    # Patch ONLY the isolated source copy.
    #
    # Original:
    #   multiprocessing workers -> shared queue
    #
    # New:
    #   deterministic sampling per (seed, epoch, step)
    #   -> ordered optimizer updates
    #
    # ========================================================

    path = fixed_source / "main.py"

    original_text = path.read_text(
        encoding="utf-8"
    )

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

    replacement = """        # CELL11: Deterministic negative sampling.
        for _det_step in range(relation_triple_steps):
            optimizer.zero_grad()

            _old_py_rng = random.getstate()
            _old_np_rng = np.random.get_state()

            try:
                _det_seed = (
                    int(args.random_seed) * 1000003
                    + i * 9176
                    + _det_step
                ) % (2 ** 32)

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

    assert original_text.count(anchor) == 1, (
        "Unexpected main.py batching implementation. "
        "Check CLAMEA v1.2.4 source."
    )

    fixed_text = original_text.replace(
        anchor,
        replacement,
        1,
    )

    compile(
        fixed_text,
        str(path),
        "exec",
    )

    path.write_text(
        fixed_text,
        encoding="utf-8",
    )

    # ========================================================
    # 4. EXPERIMENT CONFIGURATION
    # ========================================================

    config = dict(base_config)

    config.update({
        "training_data": str(data_dir) + "/",
        "random_seed": 66,
        "iterative": False,
        "max_epoch": 5,
        "start_valid": 70,
        "early_stop": False,
        "visual_dim": 4096,
        "output": str(root / "unused") + "/",
    })

    job = {
        "config": config,
        "source_dir": str(fixed_source.resolve()),
        "root": str(root.resolve()),

        "patched_main_sha256": sha256(path),

        "original_main_sha256": sha256(
            source_dir / "main.py"
        ),

        "seed_formula": (
            "(seed*1000003 + epoch*9176 + step) % 2**32"
        ),

        "source_change": (
            "Deterministic negative sampling "
            "with ordered optimizer steps"
        ),
    }

    job_file = root / "job.json"

    job_file.write_text(
        json.dumps(job, indent=2),
        encoding="utf-8",
    )

    # ========================================================
    # 5. INDEPENDENT TRAINING WORKER
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

        label = sys.argv[2]

        root = Path(job["root"])

        cfg = dict(job["config"])

        out = root / ("run_" + label)

        out.mkdir(
            parents=True,
            exist_ok=False,
        )

        cfg["output"] = (
            str(out / "results") + "/"
        )

        sys.path.insert(
            0,
            job["source_dir"],
        )

        # CLAMEA circular import order.

        import kgs
        import main as official
        import utils

        from kgs import KGs

        # ====================================================
        # B. RANDOM SEED
        # ====================================================

        def seed_all(s):

            random.seed(s)
            np.random.seed(s)

            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # ====================================================
        # C. HASH FUNCTIONS
        # ====================================================

        def array_hash(items):

            h = hashlib.sha256()

            for item in items:

                h.update(
                    repr(
                        tuple(map(int, item))
                    ).encode()
                    + b"\n"
                )

            return h.hexdigest()

        def model_hash(model):

            h = hashlib.sha256()

            for name, val in sorted(
                model.state_dict().items()
            ):

                arr = (
                    val.detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                )

                h.update(name.encode())

                h.update(
                    str(arr.shape).encode()
                )

                h.update(
                    str(arr.dtype).encode()
                )

                h.update(
                    arr.tobytes()
                )

            return h.hexdigest()

        # ====================================================
        # D. DETERMINISM SETTINGS
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

        seed_all(seed)

        torch.use_deterministic_algorithms(True)

        args = official.ARGs(cfg)

        # ====================================================
        # E. LOAD DATASET
        # ====================================================

        kg = KGs(
            args.training_data,
            args.dataset_division,
            args.random_seed,

            ordered=True,
            modality="ria",
        )

        # ====================================================
        # F. NEGATIVE SAMPLING PROBE
        # ====================================================

        from math import ceil

        nsteps = ceil(
            (
                len(kg.relation_triples_list1)
                + len(kg.relation_triples_list2)
            )
            / args.batch_size
        )

        probes = {}

        for step in (0, nsteps // 2):

            batch_seed = (
                seed * 1000003
                + 1 * 9176
                + step
            ) % (2 ** 32)

            state_py = random.getstate()
            state_np = np.random.get_state()

            try:

                random.seed(batch_seed)
                np.random.seed(batch_seed)

                pos, neg = (
                    utils.generate_relation_triple_batch(
                        kg.relation_triples_list1,
                        kg.relation_triples_list2,

                        kg.relation_triples_set1,
                        kg.relation_triples_set2,

                        kg.kg1_entities_list,
                        kg.kg2_entities_list,

                        args.batch_size,
                        step,
                        args.neg_triple_num,
                    )
                )

            finally:

                random.setstate(state_py)
                np.random.set_state(state_np)

            probes[str(step)] = {
                "positive_sha256": array_hash(pos),
                "negative_sha256": array_hash(neg),
                "n": len(pos),
            }

        # ====================================================
        # G. MODEL INITIALIZATION
        # ====================================================

        model = official.CLAMEA(
            kg,
            args,
        ).cuda()

        assert model.fc_i.in_features == 4096

        initial = model_hash(model)

        print(
            "[INIT]",
            label,
            initial,
            flush=True,
        )

        # ====================================================
        # H. TRAINING — 5 EPOCHS
        # ====================================================

        output_folder = official.generate_out_folder(
            args.output,
            args.training_data,
            args.dataset_division,
            "CLAMEA_cell11",
        )

        official.train(
            model,
            kg,
            args,
            output_folder,
        )

        final = model_hash(model)

        # ====================================================
        # I. VALIDATION
        # ====================================================

        mrr, h1 = official.valid(
            5,
            model,
            kg,
            args,
        )

        result = {
            "label": label,

            "initial_hash": initial,
            "final_hash": final,

            "validation_mrr": float(mrr),
            "validation_hits1_pct": float(h1),

            "probe": probes,
        }

        result_path = (
            root / ("run_" + label + ".json")
        )

        result_path.write_text(
            json.dumps(
                result,
                indent=2,
            )
        )

        print(
            "[FINAL]",
            label,
            final,
            flush=True,
        )

        print(
            "[VALID]",
            label,
            result["validation_mrr"],
            result["validation_hits1_pct"],
            flush=True,
        )

        print(
            "[DONE]",
            label,
            flush=True,
        )
    ''').lstrip()

    worker_file = root / "worker.py"

    compile(
        worker,
        str(worker_file),
        "exec",
    )

    worker_file.write_text(
        worker,
        encoding="utf-8",
    )

    # ========================================================
    # 6. SUBPROCESS ENVIRONMENT
    # ========================================================

    env = os.environ.copy()

    env["PYTHONHASHSEED"] = "0"

    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    env["WANDB_MODE"] = "disabled"

    env["CLAMEA_WANDB_ENABLED"] = "0"

    env["CLAMEA_RESUME_SNAPSHOTS"] = "0"

    env.pop(
        "CLAMEA_RESUME_STATE",
        None,
    )

    # ========================================================
    # 7. LAUNCH TWO INDEPENDENT RUNS
    # ========================================================

    def launch(label):

        log_file = root / ("run_" + label + ".log")

        print(
            f"\n[START] 5 epochs, replicate {label}",
            flush=True,
        )

        cmd = [
            sys.executable,
            "-u",
            str(worker_file),
            str(job_file),
            label,
        ]

        with log_file.open(
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

                    if (
                        line.startswith((
                            "[INIT]",
                            "[FINAL]",
                            "[VALID]",
                            "[DONE]",
                            "epoch ",
                            "loss:",
                            "Training ends.",
                        ))

                        or "Traceback" in line
                        or "Error:" in line
                    ):

                        print(
                            f"[{label}] {line.rstrip()}",
                            flush=True,
                        )

                rc = proc.wait()

            except BaseException:

                proc.terminate()

                try:
                    proc.wait(timeout=10)

                except subprocess.TimeoutExpired:
                    proc.kill()

                raise

        if rc != 0:

            raise RuntimeError(
                f"Worker {label} failed "
                f"(exit={rc}). "
                f"Inspect {log_file}"
            )

        result_path = (
            root / ("run_" + label + ".json")
        )

        return (
            json.loads(
                result_path.read_text()
            ),
            log_file,
        )

    # ========================================================
    # 8. EXPERIMENT START
    # ========================================================

    print()
    print("=" * 65)

    print(
        "CELL 11 — DETERMINISTIC TRAINING REPRODUCIBILITY"
    )

    print("=" * 65)

    print(
        "[CONFIG] Seed=66 | "
        "2 independent runs × 5 epochs"
    )

    print(
        "[CONFIG] Self-training=False | "
        "Validation only"
    )

    print(
        "[CONFIG] Original source/checkpoints: UNCHANGED"
    )

    print(
        "[CONFIG] Fixed source SHA256:",
        job["patched_main_sha256"],
    )

    # Sequential execution.

    a, log_a = launch("A")
    b, log_b = launch("B")

    # ========================================================
    # 9. EXTRACT TRAINING LOSSES
    # ========================================================

    def losses(log_file):

        values = []

        for line in log_file.read_text().splitlines():

            if line.startswith("loss:"):

                match = re.match(
                    r"loss:\s*([0-9.eE+\-]+)",
                    line,
                )

                assert match, (
                    "Malformed training loss line"
                )

                values.append(
                    float(match.group(1))
                )

        assert len(values) == 5, (
            f"Expected 5 loss values, "
            f"got {len(values)}"
        )

        return values

    loss_a = losses(log_a)
    loss_b = losses(log_b)

    differences = [
        abs(x - y)
        for x, y in zip(loss_a, loss_b)
    ]

    # ========================================================
    # 10. COMPARE NEGATIVE SAMPLES
    # ========================================================

    negative_equal = {
        key: (
            a["probe"][key]["negative_sha256"]
            ==
            b["probe"][key]["negative_sha256"]
        )

        for key in a["probe"]
    }

    # ========================================================
    # 11. BUILD REPORT
    # ========================================================

    report = {
        "status": "COMPLETED",

        "job_id": job_id,

        "seed": 66,

        "epochs_per_run": 5,

        "source_isolated": True,

        "original_main_sha256": (
            job["original_main_sha256"]
        ),

        "deterministic_main_sha256": (
            job["patched_main_sha256"]
        ),

        "negative_batches_match": negative_equal,

        "initial_weights_match": (
            a["initial_hash"]
            == b["initial_hash"]
        ),

        "final_weights_bitwise_match": (
            a["final_hash"]
            == b["final_hash"]
        ),

        "loss_a": loss_a,
        "loss_b": loss_b,

        "max_loss_abs_diff": max(differences),

        "loss_match_within_1e_6": all(
            d <= 1e-6
            for d in differences
        ),

        "validation_a": {
            "mrr": a["validation_mrr"],
            "hits1_pct": a["validation_hits1_pct"],
        },

        "validation_b": {
            "mrr": b["validation_mrr"],
            "hits1_pct": b["validation_hits1_pct"],
        },
    }

    # ========================================================
    # 12. PASS / FAIL
    # ========================================================

    report["pass"] = (
        all(negative_equal.values())

        and report["initial_weights_match"]

        and report["final_weights_bitwise_match"]

        and report["loss_match_within_1e_6"]

        and (
            a["validation_mrr"]
            == b["validation_mrr"]
        )

        and (
            a["validation_hits1_pct"]
            == b["validation_hits1_pct"]
        )
    )

    # ========================================================
    # 13. SAVE REPORT
    # ========================================================

    report_file = root / "cell11_report.json"

    report_file.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 14. PRINT RESULTS
    # ========================================================

    print()
    print("=" * 65)
    print("CELL 11 — REPORT")
    print("=" * 65)

    print(
        "[P1] Negative batches identical:",
        negative_equal,
    )

    print(
        "[P2] Initial model identical:",
        report["initial_weights_match"],
    )

    print(
        "[P3] Max loss difference:",
        report["max_loss_abs_diff"],
    )

    print(
        "[P4] Final model bitwise identical:",
        report["final_weights_bitwise_match"],
    )

    print(
        "[P5] Validation MRR:",
        a["validation_mrr"],
        "vs",
        b["validation_mrr"],
    )

    print(
        "[RESULT]",
        "PASS"
        if report["pass"]
        else "NOT YET DETERMINISTIC",
    )

    print(
        "[LOCAL]",
        report_file,
    )

    # ========================================================
    # 15. BACKUP REPORT TO GOOGLE DRIVE
    # ========================================================

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

        for item in (
            report_file,
            job_file,
            log_a,
            log_b,
        ):

            fs_drive.put_file(
                str(item),
                remote + "/" + item.name,
            )

        print(
            "[DRIVE]",
            remote,
        )

    except Exception as exc:

        print(
            "[DRIVE WARNING]",
            repr(exc),
        )

    return report


# ============================================================
# EXECUTE CELL 11
# ============================================================

_cell11_deterministic_training()

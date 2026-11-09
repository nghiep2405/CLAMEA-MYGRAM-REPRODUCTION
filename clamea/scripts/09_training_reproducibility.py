def _cell10_training_reproducibility():
    import hashlib
    import json
    import os
    import re
    import subprocess
    import sys
    import textwrap
    import time
    from pathlib import Path

    # ========================================================
    # 1. PREFLIGHT
    # ========================================================

    assert source_dir.is_dir(), "Missing Cell 03 patched source."
    assert data_dir.is_dir(), "Missing Cell 02 dataset."

    assert (
        PRECHECK["status"]
        == "PASS_MOLAB_PREFLIGHT_TECHNICAL"
    )

    def sha256(path):
        h = hashlib.sha256()

        with open(path, "rb") as f:
            for block in iter(
                lambda: f.read(8 * 1024 * 1024),
                b"",
            ):
                h.update(block)

        return h.hexdigest()

    # Verify that the patched source has not changed.

    for filename, manifest_key in (
        ("main.py", "patched_main_sha256"),
        ("model.py", "patched_model_sha256"),
        ("utils.py", "patched_utils_sha256"),
    ):
        assert (
            sha256(source_dir / filename)
            == PATCH_INFO[manifest_key]
        ), "Source mismatch: " + filename

    # ========================================================
    # 2. EXPERIMENT CONFIGURATION
    # ========================================================

    # Same source, seed, and dataset.
    # Disable iterative self-training.
    #
    # Only 5 epochs for reproducibility diagnostics.
    # No validation/pseudo-labeling during training.
    #
    # Validation is executed once AFTER epoch 5.

    cfg = dict(base_config)

    cfg.update({
        "random_seed": 66,
        "iterative": False,
        "max_epoch": 5,
        "start_valid": 70,
        "early_stop": False,
    })

    assert (
        cfg["training_data"].rstrip("/")
        == str(data_dir).rstrip("/")
    )

    assert cfg["batch_threads_num"] == 2
    assert cfg["max_epoch"] == 5
    assert cfg["start_valid"] == 70

    # ========================================================
    # 3. ISOLATED DIAGNOSTIC DIRECTORY
    # ========================================================

    job_id = time.strftime(
        "cell10_%Y%m%d_%H%M%S"
    )

    root = (
        WORK / "diagnostics" / job_id
    )

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    worker_py = root / "cell10_worker.py"
    job_cfg = root / "job.json"

    report_path = (
        root / "reproducibility_report.json"
    )

    job_cfg.write_text(
        json.dumps({
            "source_dir": str(source_dir.resolve()),
            "config": cfg,
            "runs_root": str(root),
            "probe_batch_steps": 2,
        }, indent=2),
        encoding="utf-8",
    )

    # ========================================================
    # 4. INDEPENDENT PYTHON WORKER
    # ========================================================
    #
    # The worker runs in a separate Python process.
    #
    # This avoids:
    #   - Marimo global variable conflicts.
    #   - Reusing model objects from other cells.
    #   - Reusing random states from previous experiments.
    #
    # ========================================================

    worker = textwrap.dedent(r'''
        import hashlib
        import json
        import math
        import multiprocessing as mp
        import os
        import random
        import sys
        import time

        from pathlib import Path

        import numpy as np
        import torch

        # ====================================================
        # A. LOAD CONFIGURATION
        # ====================================================

        job = json.loads(
            Path(sys.argv[1]).read_text()
        )

        mode, label = sys.argv[2:4]

        root = Path(job["runs_root"])

        cfg = dict(job["config"])

        sys.path.insert(
            0,
            job["source_dir"],
        )

        # Same import order as the training runner.

        import kgs
        import main as official
        import utils

        from kgs import KGs

        # ====================================================
        # B. STABLE HASH FUNCTIONS
        # ====================================================

        def digest_records(records):
            h = hashlib.sha256()

            for item in records:
                h.update(
                    repr(
                        tuple(map(int, item))
                    ).encode("ascii")
                )

                h.update(b"\n")

            return h.hexdigest()

        def digest_weights(model):
            h = hashlib.sha256()

            for name, tensor in sorted(
                model.state_dict().items()
            ):
                a = (
                    tensor.detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                )

                h.update(
                    name.encode("utf-8")
                )

                h.update(
                    str(a.shape).encode("ascii")
                )

                h.update(
                    str(a.dtype).encode("ascii")
                )

                h.update(
                    a.tobytes()
                )

            return h.hexdigest()

        # ====================================================
        # C. RANDOM SEED
        # ====================================================

        def setup_seed(seed):
            random.seed(seed)
            np.random.seed(seed)

            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # ====================================================
        # D. NEGATIVE SAMPLING WORKER
        # ====================================================

        def fork_batch(step, kg, cfg, q):

            # Use the official CLAMEA batch generator.
            # No modification to negative-sampling logic.

            pos, neg = (
                utils.generate_relation_triple_batch(
                    kg.relation_triples_list1,
                    kg.relation_triples_list2,

                    kg.relation_triples_set1,
                    kg.relation_triples_set2,

                    kg.kg1_entities_list,
                    kg.kg2_entities_list,

                    int(cfg["batch_size"]),
                    int(step),
                    int(cfg["neg_triple_num"]),
                )
            )

            q.put((
                int(step),

                digest_records(pos),
                digest_records(neg),

                len(pos),
                len(neg),
            ))

        # ====================================================
        # E. MULTIPROCESSING PROBE
        # ====================================================

        def batch_probe(kg, cfg):

            assert (
                "fork" in mp.get_all_start_methods()
            ), "Linux fork context is required."

            ctx = mp.get_context("fork")

            total_steps = math.ceil(
                (
                    len(kg.relation_triples_list1)
                    + len(kg.relation_triples_list2)
                )
                / int(cfg["batch_size"])
            )

            # Sample two representative batch positions.

            steps = [
                0,
                total_steps // 2,
            ]

            q = ctx.Queue()

            ps = [
                ctx.Process(
                    target=fork_batch,
                    args=(step, kg, cfg, q),
                )
                for step in steps
            ]

            for p in ps:
                p.start()

            try:
                arrivals = [
                    q.get(timeout=120)
                    for _ in ps
                ]

            finally:
                for p in ps:
                    p.join(timeout=15)

                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)

            assert all(
                p.exitcode == 0
                for p in ps
            ), "Negative-sampling worker failed."

            q.close()

            return {
                "arrival_order": [
                    x[0]
                    for x in arrivals
                ],

                "by_step": {
                    str(x[0]): {
                        "positive_sha256": x[1],
                        "negative_sha256": x[2],

                        "n_positive": x[3],
                        "n_negative": x[4],
                    }
                    for x in arrivals
                },
            }

        # ====================================================
        # F. ISOLATE EXPERIMENT
        # ====================================================

        os.environ.pop(
            "CLAMEA_RESUME_STATE",
            None,
        )

        os.environ["CLAMEA_WANDB_ENABLED"] = "0"
        os.environ["CLAMEA_RESUME_SNAPSHOTS"] = "0"
        os.environ["WANDB_MODE"] = "disabled"

        run_dir = (
            root / (mode + "_" + label)
        )

        run_dir.mkdir(
            exist_ok=False,
        )

        cfg["output"] = (
            str(run_dir / "results") + "/"
        )

        seed = int(
            cfg["random_seed"]
        )

        setup_seed(seed)

        args = official.ARGs(cfg)

        # ====================================================
        # G. LOAD SAME DATASET
        # ====================================================

        kg = KGs(
            args.training_data,
            args.dataset_division,
            args.random_seed,

            ordered=True,
            modality="ria",
        )

        observations = {
            "mode": mode,
            "run": label,

            "seed": seed,

            "python_hash_seed": os.environ.get(
                "PYTHONHASHSEED"
            ),

            "relation_triples_count": [
                len(kg.relation_triples_list1),
                len(kg.relation_triples_list2),
            ],

            "relation_order_sha256": [
                digest_records(
                    kg.relation_triples_list1[:1000]
                ),

                digest_records(
                    kg.relation_triples_list2[:1000]
                ),
            ],
        }

        # ====================================================
        # H. NEGATIVE SAMPLING TEST
        # ====================================================

        if mode == "probe":

            observations["negative_sampling"] = (
                batch_probe(kg, cfg)
            )

            print(
                "[PROBE]",
                label,

                observations["negative_sampling"],

                flush=True,
            )

        # ====================================================
        # I. FIVE-EPOCH TRAINING TEST
        # ====================================================

        elif mode == "train":

            assert torch.cuda.is_available(), (
                "Molab GPU is required."
            )

            # -----------------------------------------------
            # I1. Initialize model.
            # -----------------------------------------------

            model = official.CLAMEA(
                kg,
                args,
            ).cuda()

            assert (
                model.fc_i.in_features == 4096
            )

            # -----------------------------------------------
            # I2. Hash INITIAL model weights.
            # -----------------------------------------------

            observations["initial_model_sha256"] = (
                digest_weights(model)
            )

            print(
                "[INIT]",
                label,

                observations["initial_model_sha256"],

                flush=True,
            )

            # -----------------------------------------------
            # I3. Create isolated output folder.
            # -----------------------------------------------

            out = official.generate_out_folder(
                args.output,
                args.training_data,
                args.dataset_division,

                "CLAMEA_cell10_repro_smoke",
            )

            # -----------------------------------------------
            # I4. Run original patched train() — 5 epochs.
            # -----------------------------------------------

            print(
                "[TRAIN_START]",
                label,
                flush=True,
            )

            official.train(
                model,
                kg,
                args,
                out,
            )

            # -----------------------------------------------
            # I5. Hash FINAL model weights.
            # -----------------------------------------------

            observations["final_model_sha256"] = (
                digest_weights(model)
            )

            # -----------------------------------------------
            # I6. Evaluate Validation only.
            # -----------------------------------------------

            val_mrr, val_hits1 = (
                official.valid(
                    5,
                    model,
                    kg,
                    args,
                )
            )

            observations["validation_after_epoch5"] = {
                "mrr": float(val_mrr),

                "hits1_pct": float(val_hits1),
            }

            print(
                "[FINAL_MODEL]",
                label,

                observations["final_model_sha256"],

                flush=True,
            )

            print(
                "[VAL_EPOCH5]",
                label,

                observations["validation_after_epoch5"],

                flush=True,
            )

        else:
            raise ValueError(
                "Invalid worker mode."
            )

        # ====================================================
        # J. SAVE WORKER REPORT
        # ====================================================

        out_json = (
            root
            / (mode + "_" + label + ".json")
        )

        out_json.write_text(
            json.dumps(
                observations,
                indent=2,
            ),

            encoding="utf-8",
        )

        print(
            "[WORKER_DONE]",
            mode,
            label,

            flush=True,
        )
    ''').lstrip()

    compile(
        worker,
        str(worker_py),
        "exec",
    )

    worker_py.write_text(
        worker,
        encoding="utf-8",
    )

    # ========================================================
    # 5. SUBPROCESS ENVIRONMENT
    # ========================================================

    # Same Python hash seed for both replicates.
    # This controls hash-dependent ordering.

    child_env = os.environ.copy()

    child_env["PYTHONHASHSEED"] = "0"
    child_env["WANDB_MODE"] = "disabled"

    child_env.pop(
        "CLAMEA_RESUME_STATE",
        None,
    )

    child_env["CLAMEA_WANDB_ENABLED"] = "0"
    child_env["CLAMEA_RESUME_SNAPSHOTS"] = "0"

    # ========================================================
    # 6. RUN INDEPENDENT EXPERIMENT
    # ========================================================

    def launch(mode, label):

        print(
            f"\n[START] {mode.upper()} {label} "
            "| seed=66 | PYTHONHASHSEED=0",
            flush=True,
        )

        log_path = (
            root / (mode + "_" + label + ".log")
        )

        cmd = [
            sys.executable,
            "-u",

            str(worker_py),
            str(job_cfg),

            mode,
            label,
        ]

        with log_path.open(
            "w",
            encoding="utf-8",
        ) as output:

            proc = subprocess.Popen(
                cmd,

                cwd=str(source_dir),
                env=child_env,

                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,

                text=True,
                bufsize=1,
            )

            try:

                for line in proc.stdout:

                    output.write(line)

                    s = line.strip()

                    if (
                        s.startswith((
                            "[PROBE]",
                            "[INIT]",
                            "[TRAIN_START]",
                            "[FINAL_MODEL]",
                            "[VAL_EPOCH5]",
                            "[WORKER_DONE]",

                            "epoch ",
                            "loss:",
                            "Training ends.",
                        ))

                        or "Traceback" in s
                        or "Error:" in s
                    ):
                        print(
                            f"[{mode.upper()} {label}] {s}",
                            flush=True,
                        )

                code = proc.wait()

            except BaseException:

                proc.terminate()

                try:
                    proc.wait(timeout=15)

                except subprocess.TimeoutExpired:
                    proc.kill()

                raise

        if code != 0:
            raise RuntimeError(
                f"Cell 10 {mode}/{label} "
                f"exited {code}. Log: {log_path}"
            )

        return (
            json.loads(
                (
                    root
                    / (mode + "_" + label + ".json")
                ).read_text()
            ),

            log_path,
        )

    # ========================================================
    # 7. NEGATIVE SAMPLING — TWO INDEPENDENT PROBES
    # ========================================================

    print()
    print("=" * 65)
    print("CELL 10 — TRAINING REPRODUCIBILITY")
    print("=" * 65)

    print("[CONFIG] Model: CLAMEA v1.2.4")
    print("[CONFIG] Seed: 66")
    print("[CONFIG] Training epochs: 5 × 2 runs")
    print("[CONFIG] Self-training: DISABLED")
    print("[CONFIG] PYTHONHASHSEED: 0")
    print("[CONFIG] Test gold labels: NOT USED")

    print("=" * 65)

    probe_a, log_probe_a = launch(
        "probe",
        "A",
    )

    probe_b, log_probe_b = launch(
        "probe",
        "B",
    )

    # ========================================================
    # 8. TRAINING — TWO INDEPENDENT RUNS
    # ========================================================

    # Execute sequentially, not in parallel.

    train_a, log_train_a = launch(
        "train",
        "A",
    )

    train_b, log_train_b = launch(
        "train",
        "B",
    )

    # ========================================================
    # 9. EXTRACT TRAINING LOSSES
    # ========================================================

    def losses(log_file):

        matches = []

        for line in log_file.read_text(
            encoding="utf-8"
        ).splitlines():

            if line.startswith("loss:"):

                m = re.search(
                    r"^loss:\s*([0-9.+\-eE]+)",
                    line,
                )

                if m:
                    matches.append(
                        float(
                            m.group(1).rstrip(",")
                        )
                    )

        assert len(matches) == 5, (
            f"Expected exactly 5 epochs: "
            f"{log_file} -> {matches}"
        )

        return matches

    loss_a = losses(log_train_a)
    loss_b = losses(log_train_b)

    differences = [
        abs(a - b)

        for a, b in zip(
            loss_a,
            loss_b,
        )
    ]

    # ========================================================
    # 10. COMPARE NEGATIVE SAMPLING
    # ========================================================

    probe_step_equal = {
        step: (
            probe_a[
                "negative_sampling"
            ]["by_step"][step]["negative_sha256"]

            ==

            probe_b[
                "negative_sampling"
            ]["by_step"][step]["negative_sha256"]
        )

        for step in probe_a[
            "negative_sampling"
        ]["by_step"]
    }

    # ========================================================
    # 11. BUILD REPRODUCIBILITY REPORT
    # ========================================================

    report = {
        "status": "COMPLETED",

        "purpose": (
            "Training reproducibility diagnostic, "
            "NOT paper performance"
        ),

        "seed": 66,

        "epochs_per_train_run": 5,

        "iterative": False,
        "start_valid": 70,

        "python_hash_seed": "0",

        "source_commit": PINNED_COMMIT,

        "patched_main_sha256": (
            PATCH_INFO["patched_main_sha256"]
        ),

        "probe": {
            "kg_relation_order_identical": (
                probe_a["relation_order_sha256"]
                == probe_b["relation_order_sha256"]
            ),

            "negative_batch_equal_by_step": (
                probe_step_equal
            ),

            "queue_arrival_order_a": (
                probe_a[
                    "negative_sampling"
                ]["arrival_order"]
            ),

            "queue_arrival_order_b": (
                probe_b[
                    "negative_sampling"
                ]["arrival_order"]
            ),
        },

        "training": {
            "initial_weights_identical": (
                train_a["initial_model_sha256"]
                == train_b["initial_model_sha256"]
            ),

            "final_weights_identical": (
                train_a["final_model_sha256"]
                == train_b["final_model_sha256"]
            ),

            "loss_a": loss_a,
            "loss_b": loss_b,

            "loss_abs_diff": differences,

            "loss_all_within_1e-6": all(
                d <= 1e-6
                for d in differences
            ),

            "val_epoch5_a": (
                train_a["validation_after_epoch5"]
            ),

            "val_epoch5_b": (
                train_b["validation_after_epoch5"]
            ),
        },

        "interpretation": (
            "Same Python hash seed was forced in both "
            "child processes. Batch probe samples only "
            "two steps, not the full training queue. "
            "Same initial weights and loss curves do "
            "not guarantee bitwise CUDA determinism."
        ),
    }

    # ========================================================
    # 12. SAVE LOCAL REPORT
    # ========================================================

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),

        encoding="utf-8",
    )

    # ========================================================
    # 13. PRINT COMPARISON
    # ========================================================

    print()
    print("=" * 66)
    print("CELL 10 — TRAINING REPRODUCIBILITY REPORT")
    print("=" * 66)

    print(
        "[P0] KG order identical:",
        report["probe"]["kg_relation_order_identical"],
    )

    print(
        "[P1] Negative batch equality by step:",
        probe_step_equal,
    )

    print(
        "[P1] Queue arrival order:",
        report["probe"]["queue_arrival_order_a"],
        "vs",
        report["probe"]["queue_arrival_order_b"],
    )

    print(
        "[P2] Initial model identical:",
        report["training"]["initial_weights_identical"],
    )

    print(
        "[P3] 5-epoch loss A:",
        [round(x, 8) for x in loss_a],
    )

    print(
        "[P3] 5-epoch loss B:",
        [round(x, 8) for x in loss_b],
    )

    print(
        "[P3] Max absolute loss difference:",
        max(differences),
    )

    print(
        "[P4] Final model bitwise identical:",
        report["training"]["final_weights_identical"],
    )

    print(
        "[P5] Validation MRR epoch5:",
        train_a["validation_after_epoch5"]["mrr"],
        "vs",
        train_b["validation_after_epoch5"]["mrr"],
    )

    # ========================================================
    # 14. BACKUP DIAGNOSTIC REPORT TO DRIVE
    # ========================================================

    # Only upload small reports/logs.
    # Do NOT upload additional large model checkpoints.

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
            report_path,
            job_cfg,
            log_probe_a,
            log_probe_b,
            log_train_a,
            log_train_b,
        ):

            fs_drive.put_file(
                str(item),
                remote + "/" + item.name,
            )

        print(
            "[DRIVE] Uploaded:",
            remote,
        )

    except Exception as exc:

        print(
            "[DRIVE WARNING] "
            "Local report saved but Drive sync failed:",
            repr(exc),
        )

    print()
    print(
        "[DONE] Cell 10 completed. "
        "Full-training checkpoints unchanged."
    )

    return report


# ============================================================
# EXECUTE CELL 10
# ============================================================

_cell10_training_reproducibility()

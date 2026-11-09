def _cell15_final_reproduction_preflight():
    import csv
    import hashlib
    import json
    import os
    import platform
    import shutil
    import sys
    import time
    import uuid
    from pathlib import Path

    import torch

    # ========================================================
    # 1. FROZEN EXPERIMENT IDENTIFIERS
    # ========================================================

    C12 = "cell12_20261025_113643_30cbdd"
    C13 = "cell13_20261025_125049_c45e6c"
    C14 = "cell14_20261025_142206_e81c2f"

    REMOTE = (
        "paper_method_runs/"
        "paper_method_20261023_160901_s66/diagnostics"
    )

    SEEDS = (66, 190, 1234)
    EPOCHS = 300

    assert (
        PRECHECK["status"]
        == "PASS_MOLAB_PREFLIGHT_TECHNICAL"
    )

    assert Path(source_dir).is_dir()
    assert Path(data_dir).is_dir()
    assert len(set(SEEDS)) == 3

    root = Path(WORK) / "final_protocols" / (
        "cell15_"
        + time.strftime("%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:6]
    )

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ========================================================
    # 2. SHA256 HELPER
    # ========================================================

    def sha(path):
        h = hashlib.sha256()

        with Path(path).open("rb") as f:
            for chunk in iter(
                lambda: f.read(8 * 1024 * 1024),
                b"",
            ):
                h.update(chunk)

        return h.hexdigest()

    # ========================================================
    # 3. RECOVER PREVIOUS ARTIFACTS
    # ========================================================

    def get_previous(
        run_id,
        filename,
        expected=None,
    ):

        local = (
            Path(WORK)
            / "diagnostics"
            / run_id
            / filename
        )

        if local.is_file():
            if (
                expected is None
                or sha(local) == expected
            ):
                return local

        fetched = (
            root
            / "recovered"
            / run_id
            / filename
        )

        fetched.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fs_drive.get_file(
            f"{REMOTE}/{run_id}/{filename}",
            str(fetched),
        )

        assert fetched.is_file(), (
            f"Missing artifact: {run_id}/{filename}"
        )

        if expected is not None:
            assert sha(fetched) == expected, (
                f"SHA mismatch: {run_id}/{filename}"
            )

        return fetched

    # ========================================================
    # 4. LOAD CELL 12–14 REPORTS
    # ========================================================

    c12_job = json.loads(
        get_previous(
            C12,
            "job.json",
        ).read_text()
    )

    c12_report = json.loads(
        get_previous(
            C12,
            "cell12_report.json",
        ).read_text()
    )

    c13_report = json.loads(
        get_previous(
            C13,
            "cell13_report.json",
        ).read_text()
    )

    c14_report = json.loads(
        get_previous(
            C14,
            "cell14_report.json",
        ).read_text()
    )

    assert c12_job["id"] == C12

    assert (
        c12_report["status"]
        == "PASS_SHARED_BRANCH_CONTROL"
    )

    assert (
        c13_report["status"]
        == "PASS_CONTROLLED_COMPONENT_ISOLATION"
    )

    assert (
        c14_report["status"]
        == "CONTROLLED_GRAPH_STEPS_COMPLETED"
    )

    assert c13_report["parent_cell12"] == C12
    assert c14_report["parent_cell12"] == C12
    assert c14_report["parent_cell13"] == C13

    assert (
        c12_job["dataset_sha256"]
        == DATA_SHA256
    )

    assert (
        c12_report["shared_snapshot_sha256"]
        == c13_report["shared_snapshot_sha256"]
    )

    assert (
        c12_report["shared_snapshot_sha256"]
        == c14_report["shared_snapshot_sha256"]
    )

    assert (
        c13_report["pseudo_pair_sha256"]
        == c14_report["pseudo_pair_sha256"]
    )

    assert (
        c14_report["validation_epoch70_identical"]
        is True
    )

    assert (
        c13_report["epoch70_mrr_identical"]
        is True
    )

    assert c13_report["pseudo_pair_count"] == 3872
    assert c14_report["pseudo_pair_count"] == 3872

    # ========================================================
    # 5. VERIFY CELL 14 CONCLUSIONS
    # ========================================================

    s14 = c14_report["summary"]

    assert (
        s14["graph_preserve_fixed"]
        ["mean_mrr_71_100"]
        >
        s14["graph_only"]
        ["mean_mrr_71_100"]
    )

    assert (
        s14["graph_preserve_fixed"]
        ["mean_mrr_71_100"]
        >
        s14["graph_preserve_dynamic"]
        ["mean_mrr_71_100"]
    )

    assert (
        s14["full_preserve_gold"]
        ["mean_mrr_71_100"]
        <
        s14["baseline_off"]
        ["mean_mrr_71_100"]
    )

    print(
        "[P1] Cell12–14 reports and provenance: PASS",
        flush=True,
    )

    # ========================================================
    # 6. VERIFY SHARED EPOCH69 CHECKPOINT
    # ========================================================

    # No training or model restoration is performed here.

    shared = get_previous(
        C12,
        "shared_epoch69.pt",
        c12_report["shared_snapshot_sha256"],
    )

    shared_digest = sha(shared)

    assert (
        shared_digest
        == c12_report["shared_snapshot_sha256"]
    )

    print(
        "[P2] Shared epoch69 checkpoint SHA256: PASS",
        flush=True,
    )

    # ========================================================
    # 7. CREATE SOURCE SNAPSHOT DIRECTORIES
    # ========================================================

    sources = root / "sources"

    original_reference = (
        sources / "method_compat_reference"
    )

    deterministic = (
        sources / "deterministic_method"
    )

    corrected = (
        sources / "deterministic_preserve_gold"
    )

    def copy_source(destination):

        shutil.copytree(
            source_dir,
            destination,

            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".git",
                "wandb",
                "runs",
                ".env",
                ".env.*",
                "*.pem",
                "*.key",
                "*.token",
                "credentials*",
            ),
        )

    # ========================================================
    # 8. VERIFY ORIGINAL SOURCE
    # ========================================================

    assert (
        sha(Path(source_dir) / "main.py")
        == c12_job["original_main_sha256"]
    )

    # Exact deterministic source used in Cell 12.

    c12_main = get_previous(
        C12,
        "main.py",
        c12_job["patched_main_sha256"],
    )

    # Exact component-isolation source used in Cell 13.

    c13_main = get_previous(
        C13,
        "main.py",
        c13_report["modified_main_sha256"],
    )

    # Verify Cell 14 source provenance.

    c14_main = get_previous(
        C14,
        "main.py",
        c14_report["cell14_main_sha256"],
    )

    assert (
        sha(c13_main)
        == c14_report["cell13_main_sha256"]
    )

    assert (
        sha(c14_main)
        == c14_report["cell14_main_sha256"]
    )

    # ========================================================
    # 9. FREEZE THREE SOURCE VARIANTS
    # ========================================================

    for destination in (
        original_reference,
        deterministic,
        corrected,
    ):
        copy_source(destination)

    shutil.copy2(
        c12_main,
        deterministic / "main.py",
    )

    shutil.copy2(
        c13_main,
        corrected / "main.py",
    )

    # ========================================================
    # 10. REMOVE SEED66-ONLY ASSERTION
    # ========================================================

    # Cell 13 validates against a hardcoded epoch70
    # result for seed 66.
    #
    # This must not be used for seeds 190 and 1234.
    #
    # Modify ONLY the frozen final-run source.

    corrected_file = corrected / "main.py"

    code = corrected_file.read_text(
        encoding="utf-8"
    )

    old_guard = """            if i == 70:

                assert abs(
                    float(flag)
                    - float(os.environ['CELL13_REF_E70_MRR'])
                ) < 1e-8, 'Cell13 epoch70 MRR mismatch'

                assert abs(
                    float(val_hits1)
                    - float(os.environ['CELL13_REF_E70_H1'])
                ) < 1e-8, 'Cell13 epoch70 Hits1 mismatch'

                print(
                    '[CELL13_CONTROL] epoch70 MATCH',
                    flush=True
                )

"""

    assert code.count(old_guard) == 1, (
        "Cell13 epoch70 diagnostic guard changed"
    )

    code = code.replace(
        old_guard,

        (
            "            # CELL15: seed-independent validation; "
            "no seed66-specific guard.\n"
        ),

        1,
    )

    assert "CELL13_REF_E70_MRR" not in code
    assert "CELL13_REF_E70_H1" not in code

    # Keep Cell13 control switches.

    assert (
        "os.environ['CELL13_GRAPH_MODE']"
        in code
    )

    assert (
        "os.environ.get('CELL13_PSEUDO_LOSS') == '1'"
        in code
    )

    assert (
        "os.environ.get('CELL13_SAVE_BEST') == '1'"
        in code
    )

    compile(
        code,
        str(corrected_file),
        "exec",
    )

    corrected_file.write_text(
        code,
        encoding="utf-8",
    )

    # ========================================================
    # 11. VERIFY FINAL TRAINING SOURCE POLICIES
    # ========================================================

    for branch, main in (
        (
            "deterministic_method",
            deterministic / "main.py",
        ),
        (
            "deterministic_preserve_gold",
            corrected_file,
        ),
    ):

        txt = main.read_text(
            encoding="utf-8"
        )

        compile(
            txt,
            str(main),
            "exec",
        )

        # No observational Test monitoring
        # during training.

        assert (
            "test_hits1 = None"
            in txt
        )

        assert (
            "= monitor_test_hits1(i, model, kgs, args)"
            not in txt
        )

        assert (
            "'test_raw_hits1_observational': None"
            in txt
        )

        # Deterministic negative sampling.

        assert (
            "for _det_step in range(relation_triple_steps)"
            in txt
        )

        # Checkpoint support.

        assert (
            "CLAMEA_RESUME_STATE"
            in txt
        )

        assert (
            "model_best.pkl"
            in txt
        )

    assert (
        sha(deterministic / "main.py")
        == c12_job["patched_main_sha256"]
    )

    assert (
        sha(original_reference / "main.py")
        == c12_job["original_main_sha256"]
    )

    print(
        "[P3] Source snapshots and final-run patch: PASS",
        flush=True,
    )

    # ========================================================
    # 12. DATASET FILE HASH MANIFEST
    # ========================================================

    # Hash each actual dataset file.
    #
    # This is separate from the previous DATA_SHA256
    # fingerprint and provides file-level provenance.

    data_files = []
    total_bytes = 0

    for p in sorted(
        Path(data_dir).rglob("*")
    ):

        if not p.is_file():
            continue

        rel = p.relative_to(
            data_dir
        ).as_posix()

        size = p.stat().st_size

        data_files.append({
            "path": rel,
            "bytes": size,
            "sha256": sha(p),
        })

        total_bytes += size

    assert data_files, (
        "Dataset directory is empty"
    )

    dataset_lock = {
        "root_at_freeze": str(
            Path(data_dir).resolve()
        ),

        "prior_dataset_sha256": DATA_SHA256,

        "file_count": len(data_files),

        "total_bytes": total_bytes,

        "files": data_files,
    }

    (
        root / "dataset_lock.json"
    ).write_text(
        json.dumps(
            dataset_lock,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "[P4] Dataset files hashed:",
        len(data_files),
        "files",
        flush=True,
    )

    # ========================================================
    # 13. SOURCE FILE HASH MANIFEST
    # ========================================================

    def source_hashes(folder):

        return {
            p.relative_to(folder).as_posix(): sha(p)

            for p in sorted(
                folder.rglob("*")
            )

            if p.is_file()
        }

    # ========================================================
    # 14. FREEZE BASE CONFIGURATION
    # ========================================================

    base_cfg = dict(
        c12_job["base_config"]
    )

    assert base_cfg["random_seed"] == 66
    assert base_cfg["dataset_division"] == "721"
    assert base_cfg["visual_dim"] == 4096
    assert base_cfg["dim"] == 100
    assert base_cfg["stop_metric"] == "mrr"
    assert base_cfg["eval_norm"] is True
    assert base_cfg["early_stop"] is False
    assert base_cfg["max_epoch"] == 300

    # ========================================================
    # 15. DEFINE FINAL EXPERIMENTAL ARMS
    # ========================================================

    # Distinguish original-method behavior,
    # deterministic implementation,
    # and corrected implementation.
    #
    # The original compatibility source is preserved
    # as a reference, NOT declared an exact upstream
    # paper implementation.
    #
    # Final controlled runs use the verified
    # deterministic sources.

    arms = [
        {
            "name": "deterministic_method_on",

            "source": "deterministic_method",

            "iterative": True,

            "required": True,

            "env": {},

            "description": (
                "Cell12 deterministic batching; "
                "original CLAMEA self-training policy"
            ),
        },

        {
            "name": "deterministic_method_off",

            "source": "deterministic_method",

            "iterative": False,

            "required": True,

            "env": {},

            "description": (
                "No-self-training controlled ablation"
            ),
        },

        {
            "name": "deterministic_preserve_gold_on",

            "source": "deterministic_preserve_gold",

            "iterative": True,

            "required": True,

            "env": {
                "CELL13_GRAPH_MODE": "preserve",

                "CELL13_PSEUDO_LOSS": "1",

                "CELL13_SAVE_BEST": "1",
            },

            "description": (
                "Cell13 graph-preserve, fixed steps, "
                "pseudo loss ON; modified method"
            ),
        },
    ]

    # ========================================================
    # 16. GENERATE FINAL RUN MATRIX
    # ========================================================

    rows = []

    for arm in arms:

        for seed in SEEDS:

            cfg = dict(base_cfg)

            cfg.update({
                "random_seed": seed,

                "max_epoch": EPOCHS,

                "iterative": arm["iterative"],

                "start_valid": 70,

                "eval_freq": 70,

                "early_stop": False,

                "training_data": (
                    str(Path(data_dir).resolve())
                    + "/"
                ),

                "output": (
                    str(
                        root
                        / "runs"
                        / arm["name"]
                        / f"s{seed}"
                    )
                    + "/"
                ),
            })

            rows.append({
                "run_id": (
                    f"{arm['name']}_s{seed}"
                ),

                "arm": arm["name"],

                "seed": seed,

                "epochs": EPOCHS,

                "source": arm["source"],

                "config": cfg,

                # These values must be supplied to
                # the child process by the Stage5 runner.

                "env": {
                    "PYTHONHASHSEED": "0",

                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",

                    "CLAMEA_WANDB_ENABLED": "0",

                    "CLAMEA_RESUME_SNAPSHOTS": "1",

                    "CLAMEA_RESUME_INTERVAL": "70",

                    **arm["env"],
                },

                # Final independent runs start
                # from epoch 1, NOT checkpoint69.

                "unset_env": [
                    "CLAMEA_RESUME_STATE"
                ],

                "status": "PLANNED_NOT_RUN",

                "selection": (
                    "best checkpoint by Validation MRR only"
                ),

                "test": (
                    "evaluate once after selection; "
                    "do not use for tuning"
                ),
            })

    # ========================================================
    # 17. REFERENCE-SOURCE LIMITATION
    # ========================================================

    method_reference = {
        "label": "method_compat_reference",

        "folder": (
            "sources/method_compat_reference"
        ),

        "status": "SOURCE_REFERENCE_ONLY",

        "warning": (
            "This v1.2.4 patched source is NOT "
            "unmodified official paper code. "
            "Its multiprocessing and Test monitoring "
            "need separate audit before any claimed "
            "original-method final run."
        ),
    }

    # ========================================================
    # 18. BUILD FINAL REPRODUCTION MANIFEST
    # ========================================================

    manifest = {
        "schema": "CLAMEA_STAGE5_PREFLIGHT_V1",

        "status": "PASS_PROTOCOL_FREEZE_NO_TRAINING",

        "job_id": root.name,

        "created_local": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        ),

        "dataset": {
            "name": "CLAMEA_FBDB15K_protocolB",

            "split": "721",

            "image_features": (
                "VGG16 4096d "
                "(not paper ResNet 2048d)"
            ),

            "provenance_sha256": DATA_SHA256,

            "file_lock": "dataset_lock.json",

            "file_count": len(data_files),
        },

        "upstream": {
            "repo": "Quanquan429/CLAMEA",

            "commit": str(PINNED_COMMIT),

            "note": (
                "Pinned upstream provenance; "
                "no unmodified-paper-result claim"
            ),
        },

        "diagnostics": {
            "cell12": C12,

            "cell13": C13,

            "cell14": C14,

            "epoch69_state_sha256": shared_digest,

            "cell13_pseudo_pair_sha256": (
                c13_report["pseudo_pair_sha256"]
            ),

            "cell14_mrr_avg_71_100": {
                x: y["mean_mrr_71_100"]

                for x, y in s14.items()
            },

            "preferred_graph_step_policy": (
                "fixed (from Cell14 Validation only)"
            ),

            "no_claim_of_statistical_generalization": True,
        },

        "reference": method_reference,

        "training_arms": arms,

        "seeds": list(SEEDS),

        "epochs": EPOCHS,

        "planned_runs": len(rows),

        "source_hashes": {
            "method_compat_reference": (
                source_hashes(original_reference)
            ),

            "deterministic_method": (
                source_hashes(deterministic)
            ),

            "deterministic_preserve_gold": (
                source_hashes(corrected)
            ),
        },

        "runtime": {
            "python": sys.version.split()[0],

            "torch": torch.__version__,

            "cuda_available": (
                torch.cuda.is_available()
            ),

            "cuda_runtime": torch.version.cuda,

            "gpu": (
                torch.cuda.get_device_name(0)

                if torch.cuda.is_available()

                else None
            ),

            "platform": platform.platform(),
        },

        "evaluation_policy": {
            "model_selection": (
                "Validation MRR (raw), best checkpoint"
            ),

            "report": [
                "Test raw Hits@1",
                "Test raw MRR",
                "CSLS metrics if protocol allows",
            ],

            "test_during_training": False,

            "test_used_for_hyperparameter_tuning": False,

            "report_aggregation": (
                "mean +/- sample std "
                "across 3 independent seeds"
            ),

            "transductive_note": (
                "Unlabeled validation/test entity IDs "
                "can be present in self-training "
                "candidate pool; gold test labels "
                "must not be used."
            ),

            "source_scope": (
                "Protocol B, VGG16 4096d, "
                "not direct paper-number reproduction"
            ),
        },

        "further_actions": [
            (
                "Stage5 launches each run as a fresh "
                "process; PYTHONHASHSEED must be set "
                "before interpreter start"
            ),

            (
                "Stage5 must enable deterministic "
                "PyTorch flags in child process"
            ),

            (
                "Stage5 corrected arm must set "
                "CELL13_SAVE_BEST=1"
            ),

            (
                "Stage5 must not set CLAMEA_RESUME_STATE "
                "for fresh final runs"
            ),

            (
                "Stage5 evaluates Test only after all "
                "training/config selection is frozen"
            ),

            (
                "Any additional original-method run "
                "requires audit of reference source "
                "and monitoring"
            ),
        ],
    }

    # ========================================================
    # 19. SAVE STAGE5 MANIFEST
    # ========================================================

    manifest_file = (
        root / "stage5_manifest.json"
    )

    manifest_file.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 20. SAVE RUN MATRIX
    # ========================================================

    run_matrix_file = (
        root / "run_matrix.json"
    )

    run_matrix_file.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # CSV for convenient inspection.

    csv_file = (
        root / "run_matrix.csv"
    )

    with csv_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "run_id",
            "arm",
            "seed",
            "epochs",
            "source",
            "status",
        ])

        for row in rows:

            writer.writerow([
                row[k]

                for k in (
                    "run_id",
                    "arm",
                    "seed",
                    "epochs",
                    "source",
                    "status",
                )
            ])

    # ========================================================
    # 21. ARCHIVE FROZEN SOURCE CODE
    # ========================================================

    archive = Path(
        shutil.make_archive(
            str(root / "frozen_sources"),

            "zip",

            root_dir=root,

            base_dir="sources",
        )
    )

    assert archive.is_file()
    assert archive.stat().st_size > 0

    archive_sha = sha(archive)

    (
        root / "source_archive_sha256.txt"
    ).write_text(
        archive_sha + "  frozen_sources.zip\n",
        encoding="utf-8",
    )

    assert len(rows) == 9

    print(
        "[P5] Final protocol: "
        "3 arms × 3 seeds = 9 planned runs",
        flush=True,
    )

    print(
        "[P6] Sources archived with SHA256: PASS",
        flush=True,
    )

    # ========================================================
    # 22. BACKUP PREFLIGHT ARTIFACTS
    # ========================================================

    # No checkpoint is reuploaded.
    # Only manifest, metadata and source snapshots.

    remote15 = (
        f"{REMOTE}/{root.name}"
    )

    uploads = [
        manifest_file,

        run_matrix_file,

        csv_file,

        root / "dataset_lock.json",

        archive,

        root / "source_archive_sha256.txt",
    ]

    uploaded = []

    try:

        fs_drive.makedirs(
            remote15,
            exist_ok=True,
        )

        for p in uploads:

            fs_drive.put_file(
                str(p),
                f"{remote15}/{p.name}",
            )

            uploaded.append(
                p.name
            )

    except Exception as exc:

        print(
            "[DRIVE WARNING] "
            "Local artifacts remain intact:",
            repr(exc),
            flush=True,
        )

    # ========================================================
    # 23. FINAL PREFLIGHT REPORT
    # ========================================================

    print()
    print("=" * 68)

    print(
        "CELL 15 — FINAL REPRODUCTION PROTOCOL & PREFLIGHT"
    )

    print("=" * 68)

    print(
        "[RESULT] PASS — protocol and source snapshots "
        "frozen; NO TRAINING"
    )

    print(
        "[NOTE] Patched Protocol B + VGG16, "
        "not exact paper replication"
    )

    print(
        "[NOTE] Original/method-compat source is "
        "REFERENCE ONLY (audit required)"
    )

    print(
        "[CONTROL] Cell12–14 hashes, "
        "epoch69 checkpoint, pseudo pairs: PASS"
    )

    print(
        "[SEEDS]",
        list(SEEDS),
    )

    print(
        "[ARMS]",
        [
            arm["name"]

            for arm in arms
        ],
    )

    print(
        "[RUNS]",
        len(rows),
        "planned; 0 executed",
    )

    print(
        "[SOURCE SHA] corrected:",
        sha(corrected_file),
    )

    print(
        "[LOCAL]",
        manifest_file,
    )

    print(
        "[DRIVE]",
        remote15,
        "| uploaded",
        len(uploaded),
        "/",
        len(uploads),
    )

    print(
        "[NEXT] Stage 5 runner: fresh training + "
        "Validation selection + final Test"
    )

    return manifest


# ============================================================
# EXECUTE CELL 15
# ============================================================

_cell15_final_reproduction_preflight()

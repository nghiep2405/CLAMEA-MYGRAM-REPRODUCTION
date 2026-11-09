def _cell14_graph_augmentation_diagnosis():
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
    # 1. CONFIGURATION
    # ========================================================

    C12 = "cell12_20261025_113643_30cbdd"
    C13 = "cell13_20261025_125049_c45e6c"

    REMOTE_BASE = (
        "paper_method_runs/"
        "paper_method_20261023_160901_s66/diagnostics"
    )

    assert source_dir.is_dir()
    assert data_dir.is_dir()

    assert (
        PRECHECK["status"]
        == "PASS_MOLAB_PREFLIGHT_TECHNICAL"
    )

    assert torch.cuda.is_available()

    # ========================================================
    # 2. ISOLATED WORKSPACE
    # ========================================================

    root = WORK / "diagnostics" / (
        "cell14_"
        + time.strftime("%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:6]
    )

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ========================================================
    # 3. SHA256
    # ========================================================

    def sha(path):
        h = hashlib.sha256()

        with Path(path).open("rb") as f:
            for block in iter(
                lambda: f.read(8 * 1024 * 1024),
                b"",
            ):
                h.update(block)

        return h.hexdigest()

    # ========================================================
    # 4. RECOVER PREVIOUS ARTIFACTS
    # ========================================================

    def get_previous(
        cell_id,
        filename,
        expected_sha=None,
    ):

        local = (
            WORK
            / "diagnostics"
            / cell_id
            / filename
        )

        if local.is_file():

            if (
                expected_sha is None
                or sha(local) == expected_sha
            ):
                return local

        destination = (
            root / (cell_id + "__" + filename)
        )

        print(
            "[RECOVER]",
            cell_id,
            filename,
            flush=True,
        )

        fs_drive.get_file(
            f"{REMOTE_BASE}/{cell_id}/{filename}",
            str(destination),
        )

        assert destination.is_file(), (
            f"Missing artifact: {filename}"
        )

        if expected_sha:
            assert (
                sha(destination)
                == expected_sha
            ), (
                f"SHA256 mismatch: {filename}"
            )

        return destination

    # ========================================================
    # 5. LOAD CELL 12 AND CELL 13 METADATA
    # ========================================================

    c12_job = json.loads(
        get_previous(
            C12,
            "job.json",
        ).read_text()
    )

    c13_report = json.loads(
        get_previous(
            C13,
            "cell13_report.json",
        ).read_text()
    )

    assert c12_job["id"] == C12

    assert (
        c12_job["dataset_sha256"]
        == DATA_SHA256
    )

    assert (
        c13_report["status"]
        == "PASS_CONTROLLED_COMPONENT_ISOLATION"
    )

    assert (
        c13_report["parent_cell12"]
        == C12
    )

    assert (
        c13_report["pseudo_pair_count"]
        == 3872
    )

    assert (
        c13_report["epoch70_mrr_identical"]
        is True
    )

    # ========================================================
    # 6. LOAD SHARED CHECKPOINT
    # ========================================================

    shared = get_previous(
        C12,
        "shared_epoch69.pt",
        c13_report["shared_snapshot_sha256"],
    )

    print(
        "[VERIFY] Shared checkpoint epoch69: PASS",
        flush=True,
    )

    # ========================================================
    # 7. LOAD EXACT CELL 13 SOURCE
    # ========================================================

    cell13_main = get_previous(
        C13,
        "main.py",
        c13_report["modified_main_sha256"],
    )

    assert (
        sha(source_dir / "main.py")
        == c12_job["original_main_sha256"]
    )

    # Copy original source into isolated directory.

    source_copy = root / "source"

    shutil.copytree(
        source_dir,
        source_copy,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
        ),
    )

    # Replace ONLY the copied main.py
    # with Cell 13's already validated source.

    shutil.copy2(
        cell13_main,
        source_copy / "main.py",
    )

    main_file = (
        source_copy / "main.py"
    )

    code = main_file.read_text(
        encoding="utf-8"
    )

    # ========================================================
    # 8. PATCH DYNAMIC RELATION STEPS
    # ========================================================
    #
    # Cell 13 computes relation_triple_steps
    # only once before the epoch loop.
    #
    # Cell 14 provides two modes:
    #
    # FIXED:
    #   Keep original number of steps.
    #
    # DYNAMIC:
    #   Recompute based on actual graph size
    #   at the beginning of every epoch.
    #
    # Both modes preserve gold-derived triples.
    #
    # ========================================================

    old = (
        "        model.train()\n"
        "        # CELL12: same deterministic "
        "batching as Cell11.\n"
    )

    new = """        model.train()

        # CELL14: graph-size-aware epoch diagnostics.
        _c14_n1 = len(kgs.relation_triples_list1)
        _c14_n2 = len(kgs.relation_triples_list2)

        _c14_total = _c14_n1 + _c14_n2

        _c14_needed = int(
            math.ceil(_c14_total / args.batch_size)
        )

        if os.environ.get('CELL14_DYNAMIC_STEPS') == '1':
            relation_triple_steps = _c14_needed

        # Approximate training-triple coverage.
        _c14_b1 = int(
            _c14_n1 / _c14_total * args.batch_size
        )

        _c14_b2 = args.batch_size - _c14_b1

        _c14_covered = (
            min(
                _c14_n1,
                relation_triple_steps * _c14_b1
            )
            + min(
                _c14_n2,
                relation_triple_steps * _c14_b2
            )
        ) / _c14_total

        print(
            '[CELL14_STEPS] epoch={} n1={} n2={} needed={} used={} coverage={:.8f}'.format(
                i,
                _c14_n1,
                _c14_n2,
                _c14_needed,
                relation_triple_steps,
                _c14_covered
            ),
            flush=True
        )

        # CELL12: same deterministic batching as Cell11.
"""

    assert code.count(old) == 1, (
        "Cell13 source marker mismatch."
    )

    code = code.replace(
        old,
        new,
        1,
    )

    compile(
        code,
        str(main_file),
        "exec",
    )

    main_file.write_text(
        code,
        encoding="utf-8",
    )

    print(
        "[VERIFY] Cell14 source patch: PASS",
        flush=True,
    )

    # ========================================================
    # 9. EXPERIMENT CONFIGURATION
    # ========================================================

    reference = (
        c13_report["validation_by_epoch"]
        ["70"]["baseline_off"]
    )

    cfg = dict(
        c12_job["base_config"]
    )

    cfg.update(
        iterative=True,
        max_epoch=100,
        start_valid=70,
        eval_freq=70,
        early_stop=False,
        training_data=str(data_dir) + "/",
    )

    assert cfg["random_seed"] == 66
    assert cfg["visual_dim"] == 4096
    assert cfg["dataset_division"] == "721"

    job = {
        "config": cfg,

        "source": str(
            source_copy.resolve()
        ),

        "shared": str(
            shared.resolve()
        ),

        "root": str(
            root.resolve()
        ),

        "epoch70_mrr": float(
            reference["mrr"]
        ),

        "epoch70_hits1": float(
            reference["hits1_pct"]
        ),
    }

    job_file = root / "job.json"

    job_file.write_text(
        json.dumps(
            job,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 10. INDEPENDENT TRAINING WORKER
    # ========================================================

    worker_code = textwrap.dedent(r'''
        import json
        import os
        import random
        import sys

        from pathlib import Path

        import numpy as np
        import torch

        # ================================================
        # A. LOAD CONFIGURATION
        # ================================================

        job = json.loads(
            Path(sys.argv[1]).read_text()
        )

        arm = sys.argv[2]

        assert arm in (
            'graph_preserve_fixed',
            'graph_preserve_dynamic',
        )

        cfg = dict(job['config'])

        output_dir = (
            Path(job['root']) / arm
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        cfg['output'] = (
            str(output_dir / 'results') + '/'
        )

        # ================================================
        # B. LOAD CELL 14 SOURCE
        # ================================================

        sys.path.insert(
            0,
            job['source'],
        )

        import kgs
        import main as official

        from kgs import KGs

        # ================================================
        # C. VERIFY EXPERIMENT CONDITIONS
        # ================================================

        assert (
            os.environ['PYTHONHASHSEED']
            == '0'
        )

        assert (
            os.environ['CUBLAS_WORKSPACE_CONFIG']
            == ':4096:8'
        )

        assert (
            os.environ['CELL13_GRAPH_MODE']
            == 'preserve'
        )

        assert (
            os.environ['CELL13_PSEUDO_LOSS']
            == '0'
        )

        assert (
            os.environ['CLAMEA_RESUME_STATE']
            == job['shared']
        )

        assert torch.cuda.is_available()

        # ================================================
        # D. DETERMINISTIC TRAINING
        # ================================================

        random.seed(66)
        np.random.seed(66)

        torch.manual_seed(66)
        torch.cuda.manual_seed_all(66)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)

        # ================================================
        # E. INITIALIZE MODEL AND DATA
        # ================================================

        args = official.ARGs(cfg)

        kg = KGs(
            args.training_data,
            args.dataset_division,
            args.random_seed,
            ordered=True,
            modality='ria',
        )

        model = official.CLAMEA(
            kg,
            args,
        ).cuda()

        assert (
            model.fc_i.in_features
            == 4096
        )

        out = official.generate_out_folder(
            args.output,
            args.training_data,
            args.dataset_division,
            'CLAMEA_cell14',
        )

        print(
            '[CELL14_START]',
            arm,
            flush=True,
        )

        # ================================================
        # F. RESUME EPOCH 69 → 100
        # ================================================

        # The patched train() restores:
        #   model
        #   optimizer
        #   RNG
        #   relation triples
        #   pseudo-label state

        official.train(
            model,
            kg,
            args,
            out,
        )

        # ================================================
        # G. VALIDATION HISTORY
        # ================================================

        history = (
            Path(out) / 'epoch_eval.jsonl'
        )

        assert history.is_file(), (
            'Missing Validation history'
        )

        rows = [
            json.loads(s)

            for s in history.read_text().splitlines()

            if s.strip()
        ]

        assert (
            [int(x['epoch']) for x in rows]
            == list(range(70, 101))
        )

        assert all(
            x['test_raw_hits1_observational']
            is None

            for x in rows
        )

        # ================================================
        # H. SAVE BRANCH REPORT
        # ================================================

        report = {
            'arm': arm,

            'dynamic_steps': (
                os.environ['CELL14_DYNAMIC_STEPS']
                == '1'
            ),

            'validation': rows,
        }

        report_file = (
            Path(job['root'])
            / (arm + '_report.json')
        )

        report_file.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding='utf-8',
        )

        print(
            '[CELL14_DONE]',
            arm,
            flush=True,
        )
    ''').lstrip()

    compile(
        worker_code,
        "cell14_worker.py",
        "exec",
    )

    worker_file = (
        root / "worker.py"
    )

    worker_file.write_text(
        worker_code,
        encoding="utf-8",
    )

    # ========================================================
    # 11. SUBPROCESS ENVIRONMENT
    # ========================================================

    env_base = os.environ.copy()

    env_base.update({
        "PYTHONHASHSEED": "0",

        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",

        "WANDB_MODE": "disabled",

        "CLAMEA_WANDB_ENABLED": "0",

        "CLAMEA_RESUME_SNAPSHOTS": "0",

        "CLAMEA_RESUME_STATE": str(
            shared.resolve()
        ),

        "CELL13_SAVE_BEST": "0",

        "CELL13_GRAPH_MODE": "preserve",

        "CELL13_PSEUDO_LOSS": "0",

        "CELL13_REF_E70_MRR": str(
            job["epoch70_mrr"]
        ),

        "CELL13_REF_E70_H1": str(
            job["epoch70_hits1"]
        ),
    })

    # ========================================================
    # 12. EXPERIMENT OVERVIEW
    # ========================================================

    print()
    print("=" * 70)

    print(
        "CELL 14 — GRAPH AUGMENTATION DIAGNOSIS"
    )

    print("=" * 70)

    print(
        "[CONFIG] Reusing shared checkpoint epoch69"
    )

    print(
        "[CONFIG] Reusing Cell13 comparison arms"
    )

    print(
        "[CONFIG] Two new branches × 31 epochs"
    )

    print(
        "[CONFIG] Graph mode: PRESERVE"
    )

    print(
        "[CONFIG] Pseudo-label loss: OFF"
    )

    print(
        "[CONFIG] Validation only | Test disabled"
    )

    print(
        "[SOURCE]",
        sha(main_file),
    )

    print(
        "[CHECKPOINT]",
        sha(shared),
    )

    print("=" * 70, flush=True)

    # ========================================================
    # 13. RUN ADDITIONAL BRANCHES
    # ========================================================

    new_results = {}
    steps = {}
    fingerprints = {}

    branches = (
        ("graph_preserve_fixed", "0"),
        ("graph_preserve_dynamic", "1"),
    )

    for arm, dynamic in branches:

        env = dict(
            env_base,
            CELL14_DYNAMIC_STEPS=dynamic,
        )

        log_path = (
            root / (arm + ".log")
        )

        print(
            f"\n[START] {arm} dynamic={dynamic}",
            flush=True,
        )

        with log_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            p = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(worker_file),
                    str(job_file),
                    arm,
                ],

                cwd=str(source_copy),
                env=env,

                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,

                text=True,
                bufsize=1,
            )

            try:

                for line in p.stdout:

                    f.write(line)

                    msg = line.strip()

                    if msg.startswith((
                        "[CELL14_",
                        "[CELL13_PSEUDO]",
                        "[CELL13_GRAPH]",
                        "[CELL13_CONTROL]",
                        "Traceback",
                        "AssertionError",
                        "RuntimeError",
                    )):

                        # Print selected epoch step diagnostics.

                        if (
                            not msg.startswith(
                                "[CELL14_STEPS]"
                            )

                            or any(
                                f"epoch={x} " in msg

                                for x in (
                                    70, 71, 72,
                                    80, 90, 100
                                )
                            )
                        ):

                            print(
                                f"[{arm}] {msg}",
                                flush=True,
                            )

                rc = p.wait()

            except BaseException:

                p.terminate()

                try:
                    p.wait(timeout=15)

                except subprocess.TimeoutExpired:
                    p.kill()

                raise

        if rc != 0:

            raise RuntimeError(
                f"{arm} failed (exit={rc}); "
                f"log: {log_path}"
            )

        # ====================================================
        # 14. VERIFY PSEUDO-LABEL CONTROL
        # ====================================================

        raw = log_path.read_text(
            encoding="utf-8"
        )

        assert (
            "[CELL13_CONTROL] epoch70 MATCH"
            in raw
        )

        fp = re.findall(
            (
                r"\[CELL13_PSEUDO\] "
                r"epoch=70 "
                r"count=(\d+) "
                r"sha256=([0-9a-f]{64})"
            ),
            raw,
        )

        assert len(fp) == 1

        assert (
            int(fp[0][0])
            == 3872
        )

        assert (
            fp[0][1]
            == c13_report["pseudo_pair_sha256"]
        ), (
            f"Pseudo-pairs differ from Cell13: {arm}"
        )

        fingerprints[arm] = fp[0][1]

        # ====================================================
        # 15. VERIFY GOLD TRIPLES PRESERVED
        # ====================================================

        graph_lines = re.findall(
            (
                r"\[CELL13_GRAPH\] "
                r"mode=preserve "
                r"before=\((\d+), (\d+)\) "
                r"after=\((\d+), (\d+)\) "
                r"removed_preexisting=\((\d+), (\d+)\)"
            ),
            raw,
        )

        assert len(graph_lines) == 1

        graph_data = graph_lines[0]

        assert (
            graph_data[-2:]
            == ("0", "0")
        ), (
            f"Gold triples removed in {arm}"
        )

        assert (
            sum(map(int, graph_data[2:4]))
            >
            sum(map(int, graph_data[:2]))
        )

        # ====================================================
        # 16. EXTRACT RELATION STEP DIAGNOSTICS
        # ====================================================

        step_rows = re.findall(
            (
                r"\[CELL14_STEPS\] "
                r"epoch=(\d+) "
                r"n1=(\d+) "
                r"n2=(\d+) "
                r"needed=(\d+) "
                r"used=(\d+) "
                r"coverage=([0-9.]+)"
            ),
            raw,
        )

        assert len(step_rows) == 31, (
            f"Expected 31 step records, "
            f"got {len(step_rows)}"
        )

        parsed = {
            int(epoch): {
                "kg1_triples": int(n1),
                "kg2_triples": int(n2),
                "needed": int(needed),
                "used": int(used),
                "estimated_coverage": float(coverage),
            }

            for (
                epoch,
                n1,
                n2,
                needed,
                used,
                coverage,
            ) in step_rows
        }

        assert (
            sorted(parsed)
            == list(range(70, 101))
        )

        # Epoch 70: graph has not yet expanded.

        assert (
            parsed[70]["needed"]
            == parsed[70]["used"]
        )

        # Epoch 71: pseudo-derived triples were added.

        assert (
            parsed[71]["needed"]
            > parsed[70]["needed"]
        )

        if dynamic == "1":

            # Dynamic mode must use updated steps.

            assert all(
                v["needed"] == v["used"]

                for v in parsed.values()
            )

        else:

            # Fixed mode must preserve original steps.

            assert all(
                v["used"]
                == parsed[70]["used"]

                for v in parsed.values()
            )

        steps[arm] = parsed

        # ====================================================
        # 17. VERIFY VALIDATION RESULTS
        # ====================================================

        result = json.loads(
            (
                root
                / (arm + "_report.json")
            ).read_text()
        )

        assert (
            result["dynamic_steps"]
            == (dynamic == "1")
        )

        assert abs(
            result["validation"][0]["validation_raw_mrr"]
            - job["epoch70_mrr"]
        ) < 1e-8

        assert abs(
            result["validation"][0]["validation_raw_hits1"]
            - job["epoch70_hits1"]
        ) < 1e-8

        new_results[arm] = result

        print(
            "[VERIFIED]",
            arm,
            "31 epochs; gold preserved",
            flush=True,
        )

    # ========================================================
    # 18. VERIFY IDENTICAL CONDITIONS
    # ========================================================

    assert (
        fingerprints["graph_preserve_fixed"]
        ==
        fingerprints["graph_preserve_dynamic"]
    )

    fixed_steps = steps[
        "graph_preserve_fixed"
    ]

    dynamic_steps = steps[
        "graph_preserve_dynamic"
    ]

    assert (
        fixed_steps[71]["needed"]
        ==
        dynamic_steps[71]["needed"]
    )

    assert (
        fixed_steps[71]["kg1_triples"]
        ==
        dynamic_steps[71]["kg1_triples"]
    )

    assert (
        fixed_steps[71]["kg2_triples"]
        ==
        dynamic_steps[71]["kg2_triples"]
    )

    # ========================================================
    # 19. COMBINE CELL13 AND CELL14 ARMS
    # ========================================================

    old_results = (
        c13_report["validation_by_epoch"]
    )

    arm_names = (
        "baseline_off",
        "graph_only",
        "graph_preserve_fixed",
        "graph_preserve_dynamic",
        "full_preserve_gold",
    )

    curves = {
        "baseline_off": [
            old_results[str(e)]["baseline_off"]

            for e in range(70, 101)
        ],

        "graph_only": [
            old_results[str(e)]["graph_only"]

            for e in range(70, 101)
        ],

        "full_preserve_gold": [
            old_results[str(e)]["full_preserve_gold"]

            for e in range(70, 101)
        ],
    }

    # Add Cell14 experimental arms.

    for arm, result in new_results.items():

        curves[arm] = [
            {
                "mrr": float(
                    row["validation_raw_mrr"]
                ),

                "hits1_pct": float(
                    row["validation_raw_hits1"]
                ),
            }

            for row in result["validation"]
        ]

    assert all(
        len(curves[arm]) == 31

        for arm in arm_names
    )

    # All arms must have identical epoch70 MRR.

    assert (
        max(
            curves[arm][0]["mrr"]
            for arm in arm_names
        )
        -
        min(
            curves[arm][0]["mrr"]
            for arm in arm_names
        )
        < 1e-8
    )

    # ========================================================
    # 20. COMPUTE BASELINE
    # ========================================================

    base_mrr = (
        sum(
            x["mrr"]

            for x in curves["baseline_off"][1:]
        )

        / 30
    )

    # ========================================================
    # 21. COMPUTE SUMMARY
    # ========================================================

    summary = {}

    for arm in arm_names:

        seq = curves[arm][1:]

        mrr_mean = (
            sum(
                v["mrr"] for v in seq
            ) / 30
        )

        hits1_mean = (
            sum(
                v["hits1_pct"] for v in seq
            ) / 30
        )

        summary[arm] = {
            "mean_mrr_71_100": mrr_mean,

            "delta_mrr_vs_baseline": (
                mrr_mean - base_mrr
            ),

            "mean_hits1_pct_71_100": hits1_mean,

            "mrr_epoch100": (
                curves[arm][-1]["mrr"]
            ),

            "best_mrr_71_100": max(
                v["mrr"] for v in seq
            ),
        }

    # ========================================================
    # 22. ISOLATE COMPONENT EFFECTS
    # ========================================================

    comparisons = {
        "preservation_effect_without_pseudo_loss": (
            summary["graph_preserve_fixed"]
            ["mean_mrr_71_100"]

            -

            summary["graph_only"]
            ["mean_mrr_71_100"]
        ),

        "dynamic_step_effect_without_pseudo_loss": (
            summary["graph_preserve_dynamic"]
            ["mean_mrr_71_100"]

            -

            summary["graph_preserve_fixed"]
            ["mean_mrr_71_100"]
        ),

        "pseudo_loss_effect_with_fixed_preserved_graph": (
            summary["full_preserve_gold"]
            ["mean_mrr_71_100"]

            -

            summary["graph_preserve_fixed"]
            ["mean_mrr_71_100"]
        ),
    }

    # ========================================================
    # 23. FINAL REPORT
    # ========================================================

    report = {
        "status": (
            "CONTROLLED_GRAPH_STEPS_COMPLETED"
        ),

        "parent_cell12": C12,
        "parent_cell13": C13,

        "shared_snapshot_sha256": sha(shared),

        "cell13_main_sha256": sha(cell13_main),

        "cell14_main_sha256": sha(main_file),

        "pseudo_pair_count": 3872,

        "pseudo_pair_sha256": (
            c13_report["pseudo_pair_sha256"]
        ),

        "validation_epoch70_identical": True,

        "summary": summary,

        "comparisons": comparisons,

        "steps": {
            arm: {
                str(epoch): row

                for epoch, row in records.items()
            }

            for arm, records in steps.items()
        },

        "curves": {
            arm: [
                {
                    "epoch": e,
                    **curves[arm][e - 70],
                }

                for e in range(70, 101)
            ]

            for arm in arm_names
        },

        "caveat": (
            "One seed; dynamic steps increases "
            "optimizer updates and changes subsequent "
            "RNG behavior. Coverage is an estimate "
            "based on fixed-ratio batching. "
            "Graph preservation also changes triple "
            "ordering. Validation only."
        ),
    }

    report_file = (
        root / "cell14_report.json"
    )

    report_file.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 24. PRINT COMPARISON
    # ========================================================

    print()
    print("=" * 76)

    print(
        "CELL 14 — GRAPH AUGMENTATION DIAGNOSIS "
        "(Validation epochs 71–100)"
    )

    print("=" * 76)

    print(
        f"{'BRANCH':25s} "
        f"{'MRR AVG':>10s} "
        f"{'Δ vs OFF':>11s} "
        f"{'H@1 AVG':>10s} "
        f"{'MRR@100':>10s}"
    )

    for arm in arm_names:

        s = summary[arm]

        print(
            f"{arm:25s} "

            f"{s['mean_mrr_71_100']:10.6f} "

            f"{s['delta_mrr_vs_baseline']:+11.6f} "

            f"{s['mean_hits1_pct_71_100']:10.3f} "

            f"{s['mrr_epoch100']:10.6f}"
        )

    print(
        "[STEPS] epoch71 fixed:",
        fixed_steps[71],
    )

    print(
        "[STEPS] epoch71 dynamic:",
        dynamic_steps[71],
    )

    for title, delta in comparisons.items():

        print(
            f"[DELTA] {title}: {delta:+.8f}"
        )

    print(
        "[CONTROL] Same checkpoint, pseudo-pairs "
        "and epoch70: PASS"
    )

    print(
        "[LOCAL]",
        report_file,
    )

    # ========================================================
    # 25. BACKUP DIAGNOSTICS TO GOOGLE DRIVE
    # ========================================================

    remote = (
        f"{REMOTE_BASE}/{root.name}"
    )

    try:

        fs_drive.makedirs(
            remote,
            exist_ok=True,
        )

        upload_files = [
            job_file,
            main_file,

            *root.glob("*.log"),
            *root.glob("*_report.json"),
        ]

        for p in upload_files:

            if p.is_file():

                fs_drive.put_file(
                    str(p),
                    remote + "/" + p.name,
                )

        print(
            "[DRIVE]",
            remote,
        )

    except Exception as exc:

        print(
            "[DRIVE WARNING] "
            "Local results preserved:",
            repr(exc),
        )

    print(
        "[DONE] Cell 14 completed; "
        "previous runs unchanged"
    )

    return report


# ============================================================
# EXECUTE CELL 14
# ============================================================

_cell14_graph_augmentation_diagnosis()

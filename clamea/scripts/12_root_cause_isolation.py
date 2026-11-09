def _cell13_root_cause_isolation():
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
    # 1. REUSE CELL 12
    # ========================================================

    C12_ID = "cell12_20261025_113643_30cbdd"

    RUN_PREFIX = (
        "paper_method_runs/"
        "paper_method_20261023_160901_s66/"
        "diagnostics"
    )

    remote12 = f"{RUN_PREFIX}/{C12_ID}"

    previous = (
        WORK / "diagnostics" / C12_ID
    )

    root = (
        WORK
        / "diagnostics"
        / (
            "cell13_"
            + time.strftime("%Y%m%d_%H%M%S_")
            + uuid.uuid4().hex[:6]
        )
    )

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ========================================================
    # 2. FILE INTEGRITY
    # ========================================================

    def sha256(path):
        h = hashlib.sha256()

        with Path(path).open("rb") as f:
            for chunk in iter(
                lambda: f.read(8 * 1024 * 1024),
                b"",
            ):
                h.update(chunk)

        return h.hexdigest()

    def recover(name):
        old = previous / name

        if old.is_file():
            return old

        local = root / ("cell12_" + name)

        fs_drive.get_file(
            f"{remote12}/{name}",
            str(local),
        )

        assert local.is_file(), (
            f"Missing Cell 12 artifact: {name}"
        )

        return local

    # ========================================================
    # 3. LOAD CELL 12 EXPERIMENT
    # ========================================================

    c12_job = json.loads(
        recover("job.json").read_text()
    )

    c12_report = json.loads(
        recover("cell12_report.json").read_text()
    )

    previous_on = json.loads(
        recover("on_report.json").read_text()
    )

    previous_off = json.loads(
        recover("off_report.json").read_text()
    )

    shared = recover(
        "shared_epoch69.pt"
    )

    assert c12_job["id"] == C12_ID

    assert (
        c12_report["status"]
        == "PASS_SHARED_BRANCH_CONTROL"
    )

    assert (
        c12_job["dataset_sha256"]
        == DATA_SHA256
    )

    assert (
        sha256(shared)
        == c12_report["shared_snapshot_sha256"]
    )

    assert (
        c12_report["self_training_pairs_on"]
        == 3872
    )

    assert previous_on["pseudo_pairs"] == 3872
    assert previous_off["pseudo_pairs"] == 0

    print(
        "[VERIFY] Cell12 shared checkpoint: PASS",
        flush=True,
    )

    # ========================================================
    # 4. COPY EXACT CELL 12 SOURCE
    # ========================================================

    assert source_dir.is_dir()

    assert (
        sha256(source_dir / "main.py")
        == c12_job["original_main_sha256"]
    )

    fixed = root / "source"

    shutil.copytree(
        source_dir,
        fixed,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
        ),
    )

    old_main = (
        previous / "source" / "main.py"
    )

    if not old_main.is_file():
        old_main = recover("main.py")

    assert (
        sha256(old_main)
        == c12_job["patched_main_sha256"]
    )

    shutil.copy2(
        old_main,
        fixed / "main.py",
    )

    main_file = fixed / "main.py"

    code = main_file.read_text(
        encoding="utf-8"
    )

    # ========================================================
    # 5. SAFE PATCH HELPER
    # ========================================================

    def patch_once(old, new, label):
        nonlocal code

        assert code.count(old) == 1, (
            f"Unexpected Cell12 source: {label}"
        )

        code = code.replace(
            old,
            new,
            1,
        )

    # ========================================================
    # 6. PATCH PSEUDO-LOSS SWITCH
    # ========================================================
    #
    # Controls auxiliary pseudo-label supervision.
    #
    # CELL13_PSEUDO_LOSS:
    #   "1" = enabled
    #   "0" = disabled
    #
    # Gold alignment loss is always retained.
    # ========================================================

    patch_once(
        (
            "        if args.iterative and new_e1:\n"
            "            sup_align_steps = "
            "int(math.ceil(len(new_e1) / args.e_batch_size))"
        ),
        (
            "        if args.iterative and new_e1 "
            "and os.environ.get('CELL13_PSEUDO_LOSS') == '1':\n"
            "            sup_align_steps = "
            "int(math.ceil(len(new_e1) / args.e_batch_size))"
        ),
        "auxiliary pseudo loss gate",
    )

    # ========================================================
    # 7. TRACK PSEUDO-PAIR FINGERPRINT
    # ========================================================
    #
    # The 3 new branches must generate identical
    # pseudo-pairs at epoch 70.
    #
    # A count of 3872 alone is insufficient.
    # We also verify SHA256 of the generated pairs.
    # ========================================================

    insertion = (
        "            new_e2 += "
        "[x[1] for x in new_pair]\n"
    )

    patch_once(
        insertion,
        insertion + """
            import hashlib as _c13_hashlib

            _c13_blob = '\\n'.join(
                '{}\\t{}'.format(a, b)
                for a, b in new_pair
            )

            print(
                '[CELL13_PSEUDO] epoch={} count={} sha256={}'.format(
                    i,
                    len(new_pair),
                    _c13_hashlib.sha256(
                        _c13_blob.encode('ascii')
                    ).hexdigest()
                ),
                flush=True
            )
""",
        "pseudo fingerprint",
    )

    # ========================================================
    # 8. PATCH GRAPH AUGMENTATION SWITCH
    # ========================================================
    #
    # CELL13_GRAPH_MODE:
    #
    # off:
    #   No graph augmentation.
    #
    # replace:
    #   Original CLAMEA behavior.
    #   Base triples + pseudo-derived triples.
    #
    # preserve:
    #   Preserve existing gold-derived triples,
    #   then union with pseudo-derived triples.
    #
    # ========================================================

    old_graph = """            sup_triples1_set, sup_triples2_set = kgs.generate_sup_relation_triples(new_pair,
                                                                                   kgs.rt_dict1, kgs.hr_dict1,
                                                                                   kgs.rt_dict2, kgs.hr_dict2)
            kgs.relation_triples_list1, kgs.relation_triples_list2 = kgs.add_sup_relation_triples(sup_triples1_set,
                                                                                                  sup_triples2_set)
            kgs.relation_triples_set1 = set(kgs.relation_triples_list1)
            kgs.relation_triples_set2 = set(kgs.relation_triples_list2)"""

    new_graph = """            _c13_mode = os.environ['CELL13_GRAPH_MODE']

            _c13_before1 = set(
                kgs.relation_triples_list1
            )

            _c13_before2 = set(
                kgs.relation_triples_list2
            )

            if _c13_mode != 'off':

                sup_triples1_set, sup_triples2_set = kgs.generate_sup_relation_triples(
                    new_pair,
                    kgs.rt_dict1,
                    kgs.hr_dict1,
                    kgs.rt_dict2,
                    kgs.hr_dict2
                )

                kgs.relation_triples_list1, kgs.relation_triples_list2 = kgs.add_sup_relation_triples(
                    sup_triples1_set,
                    sup_triples2_set
                )

                if _c13_mode == 'preserve':

                    kgs.relation_triples_list1 = sorted(
                        _c13_before1
                        | set(kgs.relation_triples_list1)
                    )

                    kgs.relation_triples_list2 = sorted(
                        _c13_before2
                        | set(kgs.relation_triples_list2)
                    )

                kgs.relation_triples_set1 = set(
                    kgs.relation_triples_list1
                )

                kgs.relation_triples_set2 = set(
                    kgs.relation_triples_list2
                )

            _c13_after1 = set(
                kgs.relation_triples_list1
            )

            _c13_after2 = set(
                kgs.relation_triples_list2
            )

            print(
                '[CELL13_GRAPH] mode={} before={} after={} removed_preexisting={}'.format(
                    _c13_mode,
                    (
                        len(_c13_before1),
                        len(_c13_before2)
                    ),
                    (
                        len(_c13_after1),
                        len(_c13_after2)
                    ),
                    (
                        len(_c13_before1 - _c13_after1),
                        len(_c13_before2 - _c13_after2)
                    )
                ),
                flush=True
            )"""

    patch_once(
        old_graph,
        new_graph,
        "graph augmentation modes",
    )

    # ========================================================
    # 9. VERIFY EPOCH 70 BEFORE SELF-TRAINING
    # ========================================================

    e70 = c12_report["comparison"][0]

    assert int(e70["epoch"]) == 70

    assert (
        float(e70["on_mrr"])
        == float(e70["off_mrr"])
    )

    patch_once(
        (
            "            test_hits1 = None  "
            "# Cell12: Test monitoring disabled"
        ),
        """            if i == 70:

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

            test_hits1 = None  # Cell12: Test monitoring disabled""",
        "epoch70 equality guard",
    )

    # ========================================================
    # 10. DISABLE LARGE CHECKPOINT OUTPUTS
    # ========================================================
    #
    # We only need Validation curves.
    #
    # Existing Cell12 checkpoints are not modified.
    # ========================================================

    patch_once(
        (
            "                torch.save("
            "model.state_dict(), "
            "out_folder + 'model_best.pkl')"
        ),
        (
            "                if os.environ.get("
            "'CELL13_SAVE_BEST') == '1':\n"
            "                    torch.save("
            "model.state_dict(), "
            "out_folder + 'model_best.pkl')"
        ),
        "avoid unnecessary best-checkpoint storage",
    )

    # ========================================================
    # 11. VALIDATE MODIFIED SOURCE
    # ========================================================

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
        "[VERIFY] Cell13 source patch: PASS",
        flush=True,
    )

    # ========================================================
    # 12. THREE ADDITIONAL BRANCHES
    # ========================================================
    #
    # Baseline OFF and Full Original are reused
    # directly from Cell12.
    #
    # Only these branches require new GPU training.
    # ========================================================

    branches = {
        "pseudo_only": ("1", "off"),
        "graph_only": ("0", "replace"),
        "full_preserve_gold": ("1", "preserve"),
    }

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

    job = {
        "source": str(
            fixed.resolve()
        ),
        "shared": str(
            shared.resolve()
        ),
        "root": str(
            root.resolve()
        ),
        "config": cfg,
        "ref_epoch70_mrr": float(
            e70["off_mrr"]
        ),
        "ref_epoch70_hits1": float(
            e70["off_hits1"]
        ),
    }

    job_file = root / "job.json"

    job_file.write_text(
        json.dumps(
            job,
            indent=2,
        )
    )

    # ========================================================
    # 13. INDEPENDENT TRAINING WORKER
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

        # ================================================
        # A. LOAD JOB
        # ================================================

        job = json.loads(
            Path(sys.argv[1]).read_text()
        )

        label = sys.argv[2]

        cfg = dict(job['config'])

        folder = (
            Path(job['root']) / label
        )

        folder.mkdir(
            parents=True,
            exist_ok=False
        )

        cfg['output'] = (
            str(folder / 'results') + '/'
        )

        sys.path.insert(
            0,
            job['source']
        )

        # Preserve CLAMEA import order.

        import kgs
        import main as official

        from kgs import KGs

        # ================================================
        # B. DETERMINISTIC CONFIGURATION
        # ================================================

        assert (
            os.environ['PYTHONHASHSEED']
            == '0'
        )

        assert (
            os.environ['CUBLAS_WORKSPACE_CONFIG']
            == ':4096:8'
        )

        assert torch.cuda.is_available()

        assert cfg['iterative']
        assert cfg['max_epoch'] == 100

        random.seed(66)
        np.random.seed(66)

        torch.manual_seed(66)
        torch.cuda.manual_seed_all(66)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)

        # ================================================
        # C. CREATE MODEL
        # ================================================

        args = official.ARGs(cfg)

        kg = KGs(
            args.training_data,
            args.dataset_division,
            args.random_seed,
            ordered=True,
            modality='ria'
        )

        model = official.CLAMEA(
            kg,
            args
        ).cuda()

        assert (
            model.fc_i.in_features
            == 4096
        )

        out = official.generate_out_folder(
            args.output,
            args.training_data,
            args.dataset_division,
            'CLAMEA_cell13'
        )

        print(
            '[CELL13_START]',
            label,
            'pseudo_loss=',
            os.environ['CELL13_PSEUDO_LOSS'],
            'graph=',
            os.environ['CELL13_GRAPH_MODE'],
            flush=True
        )

        # ================================================
        # D. RESUME FROM CELL12 EPOCH69
        # ================================================

        # CLAMEA_RESUME_STATE is supplied by the
        # parent cell as an environment variable.
        #
        # The patched train() function restores:
        #   model
        #   optimizer
        #   RNG
        #   relation triples
        #   pseudo-pair state
        #
        # No retraining from epoch 1.

        official.train(
            model,
            kg,
            args,
            out
        )

        # ================================================
        # E. LOAD VALIDATION HISTORY
        # ================================================

        curve = (
            Path(out) / 'epoch_eval.jsonl'
        )

        rows = [
            json.loads(x)

            for x in curve.read_text().splitlines()

            if x.strip()
        ]

        assert (
            [x['epoch'] for x in rows]
            == list(range(70, 101))
        )

        assert all(
            x['test_raw_hits1_observational']
            is None

            for x in rows
        )

        # ================================================
        # F. SAVE BRANCH REPORT
        # ================================================

        report = {
            'branch': label,

            'pseudo_loss': (
                os.environ['CELL13_PSEUDO_LOSS']
                == '1'
            ),

            'graph_mode': (
                os.environ['CELL13_GRAPH_MODE']
            ),

            'validation': rows,
        }

        report_path = (
            Path(job['root'])
            / (label + '_report.json')
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2
            )
        )

        print(
            '[CELL13_DONE]',
            label,
            flush=True
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
    # 14. SUBPROCESS ENVIRONMENT
    # ========================================================

    env = os.environ.copy()

    env.update(
        PYTHONHASHSEED="0",

        CUBLAS_WORKSPACE_CONFIG=":4096:8",

        WANDB_MODE="disabled",

        CLAMEA_WANDB_ENABLED="0",

        CLAMEA_RESUME_SNAPSHOTS="0",

        CELL13_SAVE_BEST="0",

        CELL13_REF_E70_MRR=str(
            job["ref_epoch70_mrr"]
        ),

        CELL13_REF_E70_H1=str(
            job["ref_epoch70_hits1"]
        ),

        CLAMEA_RESUME_STATE=str(
            shared.resolve()
        ),
    )

    # ========================================================
    # 15. PRINT EXPERIMENT CONFIGURATION
    # ========================================================

    print()
    print("=" * 65)
    print("CELL 13 — ROOT CAUSE ISOLATION")
    print("=" * 65)

    print(
        "[CONFIG] Using Cell12 shared checkpoint epoch69"
    )

    print(
        "[CONFIG] No training epochs 1–69"
    )

    print(
        "[CONFIG] Reusing Cell12 OFF and FULL branches"
    )

    print(
        "[CONFIG] Training 3 additional branches"
    )

    print(
        "[CONFIG] Validation only | Test disabled"
    )

    print(
        "[CONFIG] Shared checkpoint SHA256:",
        sha256(shared)
    )

    print("=" * 65, flush=True)

    # ========================================================
    # 16. RUN THREE ADDITIONAL BRANCHES
    # ========================================================

    extra = {}
    fingerprints = {}

    for name, (
        loss_enabled,
        graph_mode,
    ) in branches.items():

        log = (
            root / (name + ".log")
        )

        child_env = dict(
            env,

            CELL13_PSEUDO_LOSS=loss_enabled,

            CELL13_GRAPH_MODE=graph_mode,
        )

        print()
        print(
            f"[START] {name} | "
            f"pseudo_loss={loss_enabled} "
            f"graph={graph_mode}",
            flush=True,
        )

        with log.open(
            "w",
            encoding="utf-8",
        ) as f:

            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",

                    str(worker_file),
                    str(job_file),

                    name,
                ],

                cwd=str(fixed),

                env=child_env,

                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,

                text=True,
                bufsize=1,
            )

            try:

                for line in process.stdout:

                    f.write(line)

                    msg = line.strip()

                    if (
                        msg.startswith((
                            "[CELL13_",
                            "[VALID_MONITOR]",
                            "[PHASE]",
                            "Traceback",
                            "AssertionError",
                            "RuntimeError",
                        ))

                        or msg in (
                            "epoch 70:",
                            "epoch 71:",
                            "epoch 80:",
                            "epoch 90:",
                            "epoch 100:",
                        )
                    ):

                        print(
                            f"[{name}] {msg}",
                            flush=True,
                        )

                rc = process.wait()

            except BaseException:

                process.terminate()

                try:
                    process.wait(timeout=15)

                except subprocess.TimeoutExpired:
                    process.kill()

                raise

        if rc != 0:

            raise RuntimeError(
                f"Cell13 branch {name} failed "
                f"(exit={rc}); inspect {log}"
            )

        # ====================================================
        # 17. VALIDATE BRANCH OUTPUT
        # ====================================================

        result = json.loads(
            (
                root / (name + "_report.json")
            ).read_text()
        )

        log_text = log.read_text()

        pseudo_lines = re.findall(
            (
                r"\[CELL13_PSEUDO\] "
                r"epoch=70 "
                r"count=(\d+) "
                r"sha256=([0-9a-f]{64})"
            ),
            log_text,
        )

        assert len(pseudo_lines) == 1, (
            f"Missing pseudo-pair fingerprint: {name}"
        )

        assert (
            int(pseudo_lines[0][0])
            == 3872
        )

        assert (
            "[CELL13_CONTROL] epoch70 MATCH"
            in log_text
        )

        fingerprints[name] = (
            pseudo_lines[0][1]
        )

        extra[name] = result

        print(
            "[BRANCH VERIFIED]",
            name,
            flush=True,
        )

    # ========================================================
    # 18. VERIFY PSEUDO-PAIR IDENTITY
    # ========================================================

    assert (
        len(set(fingerprints.values()))
        == 1
    ), (
        "Generated pseudo-pairs differ between "
        "branches; cannot attribute effects cleanly"
    )

    print(
        "[CONTROL] Same pseudo-pairs "
        "in all 3 new branches: PASS",
        flush=True,
    )

    # ========================================================
    # 19. COMBINE FIVE EXPERIMENTAL ARMS
    # ========================================================

    arms = {
        "baseline_off": (
            previous_off["validation"]
        ),

        "pseudo_only": (
            extra["pseudo_only"]["validation"]
        ),

        "graph_only": (
            extra["graph_only"]["validation"]
        ),

        "full_original": (
            previous_on["validation"]
        ),

        "full_preserve_gold": (
            extra["full_preserve_gold"]["validation"]
        ),
    }

    table = {
        name: {
            int(row["epoch"]): row

            for row in rows
        }

        for name, rows in arms.items()
    }

    assert all(
        sorted(t) == list(range(70, 101))

        for t in table.values()
    )

    # ========================================================
    # 20. VERIFY SHARED EPOCH70 RESULTS
    # ========================================================

    epoch70 = [
        t[70]["validation_raw_mrr"]

        for t in table.values()
    ]

    assert (
        max(epoch70) - min(epoch70)
        < 1e-8
    )

    # ========================================================
    # 21. COMPUTE BASELINE MEAN
    # ========================================================

    baseline_mean = (
        sum(
            table["baseline_off"][i][
                "validation_raw_mrr"
            ]

            for i in range(71, 101)
        )

        / 30
    )

    # ========================================================
    # 22. COMPUTE BRANCH METRICS
    # ========================================================

    summary = {}

    for name, rows in table.items():

        vals = [
            float(
                rows[i]["validation_raw_mrr"]
            )

            for i in range(71, 101)
        ]

        hits = [
            float(
                rows[i]["validation_raw_hits1"]
            )

            for i in range(71, 101)
        ]

        summary[name] = {
            "mean_mrr_71_100": (
                sum(vals) / 30
            ),

            "mean_hits1_pct_71_100": (
                sum(hits) / 30
            ),

            "delta_mrr_vs_off": (
                sum(vals) / 30
                - baseline_mean
            ),

            "mrr_epoch_71": vals[0],

            "mrr_epoch_100": vals[-1],
        }

    # ========================================================
    # 23. BUILD FINAL REPORT
    # ========================================================

    report = {
        "status": (
            "PASS_CONTROLLED_COMPONENT_ISOLATION"
        ),

        "parent_cell12": C12_ID,

        "shared_snapshot_sha256": (
            sha256(shared)
        ),

        "modified_main_sha256": (
            sha256(main_file)
        ),

        "pseudo_pair_count": 3872,

        "pseudo_pair_sha256": (
            next(iter(fingerprints.values()))
        ),

        "epoch70_mrr_identical": True,

        "branch_summary": summary,

        "validation_by_epoch": {
            str(i): {
                name: {
                    "mrr": float(
                        rows[i]["validation_raw_mrr"]
                    ),

                    "hits1_pct": float(
                        rows[i]["validation_raw_hits1"]
                    ),
                }

                for name, rows in table.items()
            }

            for i in range(70, 101)
        },

        "caveat": (
            "One seed. Graph-replace includes removal "
            "of existing gold-derived triples. "
            "Graph-preserve changes both graph composition "
            "and triple ordering. "
            "Only Validation behavior is compared; "
            "no Test selection."
        ),
    }

    # ========================================================
    # 24. SAVE REPORT
    # ========================================================

    result_file = (
        root / "cell13_report.json"
    )

    result_file.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    # ========================================================
    # 25. PRINT RESULTS
    # ========================================================

    print()
    print("=" * 76)

    print(
        "CELL 13 — ROOT CAUSE ISOLATION REPORT "
        "(Validation epochs 71–100)"
    )

    print("=" * 76)

    print(
        f"{'BRANCH':23s} "
        f"{'MRR AVG':>10s} "
        f"{'Δ vs OFF':>11s} "
        f"{'H@1 AVG':>10s} "
        f"{'MRR@100':>10s}"
    )

    for name, v in summary.items():

        print(
            f"{name:23s} "

            f"{v['mean_mrr_71_100']:10.6f} "

            f"{v['delta_mrr_vs_off']:+11.6f} "

            f"{v['mean_hits1_pct_71_100']:10.3f} "

            f"{v['mrr_epoch_100']:10.6f}"
        )

    print(
        "[CONTROL] All branches identical "
        "at epoch70: PASS"
    )

    print(
        "[PSEUDO] Same 3872 pseudo-pairs "
        "in 3 new branches: PASS"
    )

    print(
        "[RESULT] Components isolated within "
        "seed66 deterministic variant"
    )

    print(
        "[LOCAL]",
        result_file,
    )

    # ========================================================
    # 26. BACKUP SMALL ARTIFACTS TO DRIVE
    # ========================================================

    remote13 = (
        f"{RUN_PREFIX}/{root.name}"
    )

    try:

        fs_drive.makedirs(
            remote13,
            exist_ok=True,
        )

        for p in [
            result_file,
            job_file,
            main_file,

            *root.glob("*_report.json"),
            *root.glob("*.log"),
        ]:

            if p.is_file():

                fs_drive.put_file(
                    str(p),
                    remote13 + "/" + p.name,
                )

        print(
            "[DRIVE]",
            remote13,
        )

    except Exception as exc:

        print(
            "[DRIVE WARNING] Local results preserved:",
            repr(exc),
        )

    print(
        "[DONE] Cell13 completed; "
        "Cell12 checkpoints untouched"
    )

    return report


# ============================================================
# EXECUTE CELL 13
# ============================================================

_cell13_root_cause_isolation()

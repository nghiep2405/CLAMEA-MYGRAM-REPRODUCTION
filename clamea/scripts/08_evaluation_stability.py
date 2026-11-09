def _cell09_evaluation_stability():
    import hashlib
    import json
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    # ========================================================
    # 1. CONFIGURATION
    # ========================================================

    assert source_dir.is_dir(), "Missing patched source (Cell 03)."
    assert data_dir.is_dir(), "Missing dataset (Cell 02)."

    baseline_ids = (
        "paper_method_20261023_160901_s66",
        "ablation_no_iter_20261024_030021_s66",
    )

    workdir = WORK / "diagnostics" / "cell09_eval_stability"
    workdir.mkdir(parents=True, exist_ok=True)

    NATURAL_REPEATS = 10
    RNG_REPLAY_REPEATS = 3
    INFERENCE_SEED = 2026

    # ========================================================
    # 2. SHA256 VERIFICATION
    # ========================================================

    def sha256_file(path):
        h = hashlib.sha256()

        with Path(path).open("rb") as f:
            for chunk in iter(
                lambda: f.read(8 * 1024 * 1024),
                b"",
            ):
                h.update(chunk)

        return h.hexdigest()

    # ========================================================
    # 3. LOCATE CHECKPOINTS
    # ========================================================

    runs = []

    for run_id in baseline_ids:

        run_dir_local = WORK / "runs" / run_id
        run_dir_local.mkdir(parents=True, exist_ok=True)

        remote_root = "paper_method_runs/" + run_id

        # Load configuration and training metadata.

        for filename in (
            "config.json",
            "manifest.json",
            "training_summary.json",
        ):

            local = run_dir_local / filename

            if not local.is_file():

                assert fs_drive.exists(
                    remote_root + "/" + filename
                ), filename

                fs_drive.get_file(
                    remote_root + "/" + filename,
                    str(local),
                )

        cfg = json.loads(
            (run_dir_local / "config.json").read_text()
        )

        manifest = json.loads(
            (run_dir_local / "manifest.json").read_text()
        )

        summary = json.loads(
            (run_dir_local / "training_summary.json").read_text()
        )

        # Verify completed training.

        assert summary["status"] == "COMPLETED", run_id
        assert int(cfg["random_seed"]) == 66
        assert cfg["stop_metric"] == "mrr"
        assert int(cfg["visual_dim"]) == 4096
        assert manifest["run_id"] == run_id

        assert (
            manifest["dataset_sha256"] == DATA_SHA256
        )

        # Verify that patched source has not changed.

        for source_filename, key in (
            ("main.py", "patched_main_sha256"),
            ("model.py", "patched_model_sha256"),
            ("utils.py", "patched_utils_sha256"),
        ):

            assert (
                sha256_file(source_dir / source_filename)
                == manifest[key]
            ), (
                "Patched source mismatch: " + source_filename
            )

        # ----------------------------------------------------
        # Find best checkpoint locally.
        # ----------------------------------------------------

        ckpt = None

        pointer = run_dir_local / "out_folder.txt"

        if pointer.is_file():

            candidate = (
                Path(pointer.read_text().strip())
                / "model_best.pkl"
            )

            if candidate.is_file():
                ckpt = candidate

        # ----------------------------------------------------
        # Download from Google Drive if necessary.
        # ----------------------------------------------------

        if ckpt is None:

            candidate = (
                run_dir_local / "model_best_stability.pkl"
            )

            if not candidate.is_file():

                print(
                    "[DOWNLOAD]",
                    run_id,
                    "best checkpoint (~564 MB)",
                    flush=True,
                )

                assert fs_drive.exists(
                    remote_root + "/model_best.pkl"
                )

                fs_drive.get_file(
                    remote_root + "/model_best.pkl",
                    str(candidate),
                )

            ckpt = candidate

        # ----------------------------------------------------
        # Verify best checkpoint.
        # ----------------------------------------------------

        assert (
            sha256_file(ckpt)
            == summary["checkpoint_sha256"]
        ), (
            "Best-checkpoint SHA256 mismatch for " + run_id
        )

        print(
            "[VERIFY]",
            run_id,
            "checkpoint SHA256 OK",
            flush=True,
        )

        runs.append({
            "run_id": run_id,
            "config_path": str(
                run_dir_local / "config.json"
            ),
            "checkpoint_path": str(ckpt),
            "iterative": bool(cfg["iterative"]),
        })

    # ========================================================
    # 4. VERIFY EVALUATION CONFIGURATION
    # ========================================================

    c0 = json.loads(
        Path(runs[0]["config_path"]).read_text()
    )

    c1 = json.loads(
        Path(runs[1]["config_path"]).read_text()
    )

    # Training epochs and iterative mode may differ.
    # Evaluation settings must be identical.

    for field in (
        "training_data",
        "dataset_division",
        "eval_metric",
        "eval_norm",
        "stop_metric",
        "visual_dim",
        "dim",
        "random_seed",
        "test_threads_num",
        "top_k",
    ):

        assert c0[field] == c1[field], (
            "Config mismatch: " + field
        )

    assert runs[0]["iterative"] is True
    assert runs[1]["iterative"] is False

    print(
        "[CONFIG] Two checkpoint configurations verified",
        flush=True,
    )

    # ========================================================
    # 5. CREATE EVALUATION JOB
    # ========================================================

    job_path = workdir / "job.json"

    result_path = (
        workdir / "stability_report.json"
    )

    worker_path = (
        workdir / "worker.py"
    )

    job_path.write_text(
        json.dumps({
            "source_dir": str(source_dir),
            "runs": runs,
            "natural_repeats": NATURAL_REPEATS,
            "rng_replay_repeats": RNG_REPLAY_REPEATS,
            "inference_seed": INFERENCE_SEED,
        }, indent=2)
    )

    # ========================================================
    # 6. INDEPENDENT EVALUATION PROCESS
    # ========================================================
    #
    # Running evaluation in a separate Python process:
    #
    # - avoids Marimo global variable conflicts;
    # - avoids reusing models imported in previous cells;
    # - does not modify training checkpoints;
    # - does not start model training.
    #
    # ========================================================

    worker_code = textwrap.dedent(r'''
import contextlib
import gc
import io
import json
import random
import statistics
import sys

from pathlib import Path

import numpy as np
import torch


# ============================================================
# A. LOAD JOB
# ============================================================

job = json.loads(
    Path(sys.argv[1]).read_text()
)

report_path = Path(sys.argv[2])

assert torch.cuda.is_available(), (
    "GPU is required for CLAMEA evaluation."
)

sys.path.insert(
    0,
    job["source_dir"],
)

# Keep the import order used by the training runner.

import kgs
import main as official

from kgs import KGs


# ============================================================
# B. REPRODUCIBILITY SETTINGS
# ============================================================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def reseed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# C. VALIDATION EVALUATION
# ============================================================

def metrics(model, kg, args):

    captured = io.StringIO()

    with (
        contextlib.redirect_stdout(captured),
        torch.no_grad(),
    ):

        returned = official.valid(
            -1,
            model,
            kg,
            args,
        )

    # Our patched valid() returns:
    # (Validation MRR, Validation Hits@1)

    assert (
        isinstance(returned, tuple)
        and len(returned) == 2
    ), (
        "Cell 03 F7 + Hits@1 patch is missing."
    )

    mrr, h1_pct = map(
        float,
        returned,
    )

    assert np.isfinite(mrr)
    assert np.isfinite(h1_pct)

    assert 0.0 <= mrr <= 1.0
    assert 0.0 <= h1_pct <= 100.0

    return {
        "mrr": mrr,
        "hits1_pct": h1_pct,
    }


# ============================================================
# D. SUMMARY STATISTICS
# ============================================================

def describe(rows, key):

    values = [
        float(row[key])
        for row in rows
    ]

    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


# ============================================================
# E. INITIALIZE DATASET
# ============================================================

cfg0 = json.loads(
    Path(
        job["runs"][0]["config_path"]
    ).read_text()
)

reseed(
    job["inference_seed"]
)

kg = KGs(
    cfg0["training_data"],
    cfg0["dataset_division"],
    int(cfg0["random_seed"]),
    ordered=True,
    modality="ria",
)

assert len(kg.valid_links) == 1284

print(
    "[DATA] Validation pairs:",
    len(kg.valid_links),
    flush=True,
)

print(
    "[DATA] Gold Test labels are NOT used in evaluation.",
    flush=True,
)


# ============================================================
# F. PREPARE REPORT
# ============================================================

report = {
    "status": "COMPLETED",
    "scope": "validation_only",

    "natural_repeats": job["natural_repeats"],
    "rng_replay_repeats": job["rng_replay_repeats"],

    "note": (
        "Inference repeats are not independent "
        "training seeds."
    ),

    "results": [],
}


# ============================================================
# G. EVALUATE BOTH CHECKPOINTS
# ============================================================

for item in job["runs"]:

    cfg = json.loads(
        Path(
            item["config_path"]
        ).read_text()
    )

    args = official.ARGs(cfg)

    reseed(
        job["inference_seed"]
    )

    # --------------------------------------------------------
    # G1. Load frozen best checkpoint.
    # --------------------------------------------------------

    model = official.CLAMEA(
        kg,
        args,
    ).cuda()

    assert model.fc_i.in_features == 4096

    checkpoint = torch.load(
        item["checkpoint_path"],
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    del checkpoint

    model.eval()

    assert not any(
        p.grad is not None
        for p in model.parameters()
    )

    print(
        "\n[RUN]",
        item["run_id"],
        "| iterative =",
        item["iterative"],
        flush=True,
    )

    # ========================================================
    # G2. NATURAL EVALUATION — 10 REPEATS
    # ========================================================
    #
    # Set seed ONCE.
    #
    # Do not reset RNG between repetitions.
    # This measures ordinary repeated inference variation.
    #
    # ========================================================

    reseed(
        job["inference_seed"]
    )

    natural = []

    for i in range(
        job["natural_repeats"]
    ):

        row = {
            "repeat": i + 1,
            **metrics(model, kg, args),
        }

        natural.append(row)

        print(
            f"[NATURAL {i+1:02d}/10] "
            f"Hits@1={row['hits1_pct']:.3f}% "
            f"MRR={row['mrr']:.8f}",
            flush=True,
        )

    # ========================================================
    # G3. RNG REPLAY — 3 REPEATS
    # ========================================================
    #
    # Restore exactly the same RNG state before each call.
    #
    # If the results are identical here but vary during
    # natural evaluation, stochastic sampling is a
    # likely contributor to the variation.
    #
    # ========================================================

    rng_py = random.getstate()
    rng_np = np.random.get_state()

    rng_cpu = (
        torch.get_rng_state().clone()
    )

    rng_cuda = [
        state.clone()
        for state in torch.cuda.get_rng_state_all()
    ]

    replay = []

    for i in range(
        job["rng_replay_repeats"]
    ):

        random.setstate(rng_py)
        np.random.set_state(rng_np)

        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state_all(rng_cuda)

        row = {
            "repeat": i + 1,
            **metrics(model, kg, args),
        }

        replay.append(row)

        print(
            f"[RNG REPLAY {i+1:02d}/3] "
            f"Hits@1={row['hits1_pct']:.3f}% "
            f"MRR={row['mrr']:.8f}",
            flush=True,
        )

    # ========================================================
    # G4. COMPUTE STATISTICS
    # ========================================================

    summary = {
        "run_id": item["run_id"],
        "iterative": item["iterative"],

        "natural": natural,

        "natural_stats": {
            "hits1_pct": describe(
                natural,
                "hits1_pct",
            ),
            "mrr": describe(
                natural,
                "mrr",
            ),
        },

        "rng_replay": replay,

        "rng_replay_stats": {
            "hits1_pct": describe(
                replay,
                "hits1_pct",
            ),
            "mrr": describe(
                replay,
                "mrr",
            ),
        },
    }

    report["results"].append(summary)

    # --------------------------------------------------------
    # Print natural evaluation statistics.
    # --------------------------------------------------------

    print(
        "\n[STATS] NATURAL EVALUATION",
        flush=True,
    )

    for metric_name in (
        "hits1_pct",
        "mrr",
    ):

        stat = summary[
            "natural_stats"
        ][metric_name]

        print(
            f"  {metric_name}: "
            f"mean={stat['mean']:.8f}, "
            f"std={stat['std']:.8f}, "
            f"min={stat['min']:.8f}, "
            f"max={stat['max']:.8f}",
            flush=True,
        )

    # --------------------------------------------------------
    # Print RNG replay statistics.
    # --------------------------------------------------------

    print(
        "\n[STATS] RNG REPLAY",
        flush=True,
    )

    for metric_name in (
        "hits1_pct",
        "mrr",
    ):

        stat = summary[
            "rng_replay_stats"
        ][metric_name]

        print(
            f"  {metric_name}: "
            f"mean={stat['mean']:.8f}, "
            f"std={stat['std']:.8f}, "
            f"range={stat['range']:.8f}",
            flush=True,
        )

    # --------------------------------------------------------
    # Release GPU memory before loading next checkpoint.
    # --------------------------------------------------------

    del model

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# H. SAVE REPORT
# ============================================================

report_path.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\n[PASS] Evaluation stability report saved.",
    flush=True,
)
''')

    # ========================================================
    # 7. VERIFY AND WRITE WORKER
    # ========================================================

    compile(
        worker_code,
        str(worker_path),
        "exec",
    )

    worker_path.write_text(
        worker_code,
        encoding="utf-8",
    )

    # ========================================================
    # 8. RUN INDEPENDENT EVALUATION
    # ========================================================

    print()
    print("=" * 60)
    print("CLAMEA — EVALUATION STABILITY TEST")
    print("=" * 60)

    print(
        "[CONFIG] Checkpoints:",
        len(runs),
    )

    print(
        "[CONFIG] Natural repetitions:",
        NATURAL_REPEATS,
    )

    print(
        "[CONFIG] RNG replay repetitions:",
        RNG_REPLAY_REPEATS,
    )

    print(
        "[CONFIG] Inference seed:",
        INFERENCE_SEED,
    )

    print(
        "[CONFIG] Training: DISABLED"
    )

    print("=" * 60)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(worker_path),
            str(job_path),
            str(result_path),
        ],

        cwd=str(source_dir),

        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,

        text=True,
        bufsize=1,
    )

    try:

        for line in proc.stdout:
            print(
                line,
                end="",
                flush=True,
            )

        returncode = proc.wait()

    except BaseException:

        proc.terminate()
        raise

    assert returncode == 0, (
        f"Cell 09 evaluation process failed "
        f"(exit={returncode})."
    )

    assert result_path.is_file(), (
        "Missing Cell 09 report."
    )

    # ========================================================
    # 9. LOAD FINAL REPORT
    # ========================================================

    result = json.loads(
        result_path.read_text()
    )

    # ========================================================
    # 10. UPLOAD TO GOOGLE DRIVE
    # ========================================================

    for run_id in baseline_ids:

        remote = (
            "paper_method_runs/"
            + run_id
            + "/diagnostics"
        )

        fs_drive.makedirs(
            remote,
            exist_ok=True,
        )

        fs_drive.put_file(
            str(result_path),
            remote + "/evaluation_stability_10x.json",
        )

    print(
        "[DRIVE] Report uploaded to both run "
        "diagnostics folders.",
        flush=True,
    )

    print()
    print("=" * 60)
    print("[DONE] EVALUATION STABILITY TEST COMPLETED")
    print("=" * 60)

    print(
        "No training performed. "
        "Original checkpoints remain unchanged.",
        flush=True,
    )

    return result


# ============================================================
# EXECUTE CELL 09
# ============================================================

_cell09_evaluation_stability()

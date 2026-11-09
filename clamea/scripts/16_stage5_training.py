# CELL 16 — CLAMEA Stage 5 final training runner
# Paste this entire cell into Molab/Marimo.

def cell16_stage5_runner():
    import hashlib
    import json
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time
    import uuid
    import zipfile
    from pathlib import Path

    import torch

    # ========================================================
    # 0. CONFIG — run one new training job per invocation
    # ========================================================

    MAX_NEW_RUNS = 9  # Set to 9 after confirming run 1.

    CELL15_ID = "cell15_20261025_150915_b52701"

    REMOTE_BASE = (
        "paper_method_runs/"
        "paper_method_20261023_160901_s66/diagnostics"
    )

    REMOTE15 = f"{REMOTE_BASE}/{CELL15_ID}"

    REMOTE16 = (
        f"{REMOTE_BASE}/cell16_stage5_{CELL15_ID}"
    )

    local15 = (
        Path(WORK)
        / "final_protocols"
        / CELL15_ID
    )

    root = (
        Path(WORK)
        / "final_training"
        / f"cell16_stage5_{CELL15_ID}"
    )

    local15.mkdir(
        parents=True,
        exist_ok=True,
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. HELPERS
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

    def json_read(path):
        return json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

    def json_write(path, value):
        p = Path(path)

        p.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp = p.with_name(
            p.name + ".tmp"
        )

        temp.write_text(
            json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        os.replace(
            temp,
            p,
        )

    # ========================================================
    # 2. GOOGLE DRIVE UTILITIES
    # ========================================================

    def fetch(
        remote,
        local,
        expected_sha=None,
    ):
        local = Path(local)

        if local.is_file() and (
            expected_sha is None
            or sha256(local) == expected_sha
        ):
            return local

        local.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = local.with_name(
            local.name + ".part"
        )

        if tmp.exists():
            tmp.unlink()

        fs_drive.get_file(
            remote,
            str(tmp),
        )

        assert tmp.is_file(), (
            f"Download failed: {remote}"
        )

        if expected_sha:
            assert (
                sha256(tmp) == expected_sha
            ), (
                f"SHA mismatch: {remote}"
            )

        os.replace(
            tmp,
            local,
        )

        return local

    def upload(
        local,
        remote,
    ):
        fs_drive.makedirs(
            remote.rsplit("/", 1)[0],
            exist_ok=True,
        )

        for attempt in range(1, 4):
            try:
                fs_drive.put_file(
                    str(local),
                    remote,
                )

                assert fs_drive.exists(remote), (
                    f"Remote not visible: {remote}"
                )

                return

            except Exception:
                if attempt == 3:
                    raise

                time.sleep(
                    3 * attempt
                )

    def remote_run(run_id):
        return (
            f"{REMOTE16}/runs/{run_id}"
        )

    # ========================================================
    # 3. RECOVER AND VERIFY CELL15 PROTOCOL
    # ========================================================

    manifest_p = fetch(
        f"{REMOTE15}/stage5_manifest.json",
        local15 / "stage5_manifest.json",
    )

    matrix_p = fetch(
        f"{REMOTE15}/run_matrix.json",
        local15 / "run_matrix.json",
    )

    archive_hash_p = fetch(
        f"{REMOTE15}/source_archive_sha256.txt",
        local15 / "source_archive_sha256.txt",
    )

    lock_p = fetch(
        f"{REMOTE15}/dataset_lock.json",
        local15 / "dataset_lock.json",
    )

    zip_sha = (
        archive_hash_p.read_text()
        .split()[0]
    )

    archive_p = fetch(
        f"{REMOTE15}/frozen_sources.zip",
        local15 / "frozen_sources.zip",
        expected_sha=zip_sha,
    )

    manifest = json_read(
        manifest_p
    )

    matrix = json_read(
        matrix_p
    )

    dataset_lock = json_read(
        lock_p
    )

    assert (
        manifest["status"]
        == "PASS_PROTOCOL_FREEZE_NO_TRAINING"
    )

    assert (
        manifest["job_id"]
        == CELL15_ID
    )

    assert (
        manifest["planned_runs"]
        == len(matrix)
        == 9
    )

    assert (
        [int(x["seed"]) for x in matrix]
        == [66, 190, 1234] * 3
    )

    assert (
        len({
            x["run_id"]
            for x in matrix
        }) == 9
    )

    assert (
        dataset_lock["file_count"]
        == len(dataset_lock["files"])
        == 16
    )

    assert (
        Path(
            dataset_lock["root_at_freeze"]
        ).resolve()
        ==
        Path(data_dir).resolve()
    )

    # ========================================================
    # 4. VERIFY DATASET SHA256
    # ========================================================

    for item in dataset_lock["files"]:

        data_file = (
            Path(data_dir)
            / item["path"]
        )

        assert data_file.is_file(), (
            f"Missing file: {data_file}"
        )

        assert (
            data_file.stat().st_size
            == item["bytes"]
        )

        assert (
            sha256(data_file)
            == item["sha256"]
        ), (
            f"Dataset mutated: {data_file}"
        )

    print(
        "[C16] Dataset lock: PASS (16 files)",
        flush=True,
    )

    # ========================================================
    # 5. RESTORE AND VERIFY FROZEN SOURCE
    # ========================================================

    frozen = (
        root / "frozen"
    )

    sources = (
        frozen / "sources"
    )

    if not sources.is_dir():

        frozen.mkdir(
            parents=True,
            exist_ok=True,
        )

        with zipfile.ZipFile(archive_p) as z:

            assert z.testzip() is None, (
                "Corrupt source ZIP"
            )

            for entry in z.infolist():

                dest = (
                    frozen / entry.filename
                ).resolve()

                assert (
                    dest == frozen.resolve()
                    or frozen.resolve() in dest.parents
                )

                assert (
                    (
                        entry.external_attr >> 16
                    ) & 0o170000
                ) != 0o120000

            z.extractall(frozen)

    for kind, files in (
        manifest["source_hashes"].items()
    ):

        for name, expected in files.items():

            p = (
                sources
                / kind
                / name
            )

            assert (
                p.is_file()
                and sha256(p) == expected
            ), (
                f"Source modified: {p}"
            )

    print(
        "[C16] Frozen source SHA256: PASS",
        flush=True,
    )

    # ========================================================
    # 6. CREATE INDEPENDENT WORKER
    # ========================================================

    worker_p = (
        root / "worker.py"
    )

    worker_code = r'''
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ========================================================
# LOAD JOB
# ========================================================

job = json.loads(
    Path(sys.argv[1]).read_text()
)

source = Path(
    job["source"]
).resolve()

sys.path.insert(
    0,
    str(source),
)

os.chdir(source)

# ========================================================
# CRITICAL FIX — CIRCULAR IMPORT
# ========================================================
#
# kgs.py imports:
#
#     from main import set_seed
#
# Therefore import kgs FIRST, then main.
#
# DO NOT reverse this order.
#
# ========================================================

import kgs
import main as official

from kgs import KGs

# ========================================================
# LOAD CONFIG
# ========================================================

cfg = dict(
    job["config"]
)

seed = int(
    cfg["random_seed"]
)

assert (
    os.environ["PYTHONHASHSEED"]
    == "0"
)

assert (
    os.environ["CUBLAS_WORKSPACE_CONFIG"]
    == ":4096:8"
)

assert cfg["max_epoch"] == 300
assert cfg["start_valid"] == 70

assert torch.cuda.is_available(), (
    "GPU unavailable"
)

# ========================================================
# DETERMINISTIC TRAINING
# ========================================================

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

torch.set_float32_matmul_precision(
    "high"
)

# ========================================================
# INITIALIZE MODEL AND DATASET
# ========================================================

args = official.ARGs(cfg)

kgs_obj = KGs(
    args.training_data,
    args.dataset_division,
    args.random_seed,
    ordered=True,
    modality="ria",
)

model = official.CLAMEA(
    kgs_obj,
    args,
).cuda()

# ========================================================
# CREATE OUTPUT DIRECTORY
# ========================================================

out = official.generate_out_folder(
    args.output,
    args.training_data,
    args.dataset_division,
    "CLAMEA_final",
)

print(
    "[C16_OUTPUT] " + out,
    flush=True,
)

# ========================================================
# RESTORE VALIDATION HISTORY IF RESUMING
# ========================================================

old_history = os.environ.get(
    "CELL16_PREVIOUS_HISTORY"
)

if old_history:

    (
        Path(out) / "epoch_eval.jsonl"
    ).write_text(
        Path(old_history).read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

# ========================================================
# TRAIN / RESUME
# ========================================================

# official.train() restores from
# CLAMEA_RESUME_STATE when specified.
#
# Source was frozen at Cell 15.

official.train(
    model,
    kgs_obj,
    args,
    out,
)

# IMPORTANT:
# We deliberately NEVER call official.test()
# inside Cell 16.

print(
    "[C16_TRAIN_FINISHED] " + out,
    flush=True,
)
'''.lstrip()

    compile(
        worker_code,
        str(worker_p),
        "exec",
    )

    worker_p.write_text(
        worker_code,
        encoding="utf-8",
    )

    # ========================================================
    # 7. CHECKPOINT RECOVERY
    # ========================================================

    def recover_checkpoint(
        run_dir,
        run_id,
    ):
        latest = (
            run_dir / "latest.json"
        )

        remote = remote_run(
            run_id
        )

        if (
            not latest.exists()
            and fs_drive.exists(
                remote + "/latest.json"
            )
        ):
            fetch(
                remote + "/latest.json",
                latest,
            )

        if not latest.exists():
            return None

        info = json_read(
            latest
        )

        assert (
            info["run_id"]
            == run_id
        )

        assert (
            1 <= int(info["epoch"]) <= 300
        )

        checkpoint = (
            run_dir
            / "checkpoints"
            / info["checkpoint"]
        )

        history = (
            run_dir
            / "checkpoints"
            / info["history"]
        )

        fetch(
            remote
            + "/checkpoints/"
            + info["checkpoint"],

            checkpoint,

            info["sha256"],
        )

        fetch(
            remote
            + "/checkpoints/"
            + info["history"],

            history,

            info["history_sha256"],
        )

        assert (
            len([
                line

                for line in (
                    history.read_text()
                    .splitlines()
                )

                if line.strip()
            ])

            == info["validation_rows"]
        )

        return (
            info,
            checkpoint,
            history,
        )

    # ========================================================
    # 8. ATOMIC CHECKPOINT BACKUP
    # ========================================================

    def backup(
        run_dir,
        run_id,
        epoch,
        output_dir,
    ):
        snapshot = (
            Path(output_dir)
            / "resume_state.pt"
        )

        history_src = (
            Path(output_dir)
            / "epoch_eval.jsonl"
        )

        assert snapshot.is_file(), (
            f"Missing resume_state.pt "
            f"at epoch {epoch}"
        )

        assert (
            snapshot.stat().st_size
            > 100_000_000
        )

        assert (
            1 <= epoch <= 300
        )

        cp_dir = (
            run_dir / "checkpoints"
        )

        cp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cp_name = (
            f"epoch_{epoch:03d}.pt"
        )

        hist_name = (
            f"epoch_{epoch:03d}_eval.jsonl"
        )

        cp = (
            cp_dir / cp_name
        )

        hist = (
            cp_dir / hist_name
        )

        temp = (
            cp_dir / (cp_name + ".tmp")
        )

        # Copy before original snapshot is overwritten.

        shutil.copyfile(
            snapshot,
            temp,
        )

        os.replace(
            temp,
            cp,
        )

        if history_src.is_file():

            shutil.copy2(
                history_src,
                hist,
            )

        else:

            hist.write_text(
                "",
                encoding="utf-8",
            )

        records = [
            json.loads(s)

            for s in (
                hist.read_text()
                .splitlines()
            )

            if s.strip()
        ]

        assert all(
            int(r["epoch"]) <= epoch

            for r in records
        )

        info = {
            "run_id": run_id,

            "epoch": epoch,

            "checkpoint": cp_name,

            "sha256": sha256(cp),

            "history": hist_name,

            "history_sha256": sha256(hist),

            "validation_rows": len(records),

            "saved_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
        }

        remote = remote_run(
            run_id
        )

        # Upload files FIRST.
        # Publish latest.json LAST.

        upload(
            cp,
            f"{remote}/checkpoints/{cp_name}",
        )

        upload(
            hist,
            f"{remote}/checkpoints/{hist_name}",
        )

        json_write(
            run_dir / "latest.json",
            info,
        )

        upload(
            run_dir / "latest.json",
            remote + "/latest.json",
        )

        # Keep only latest local checkpoint.
        # Previous checkpoints remain on Drive.

        for old in cp_dir.glob(
            "epoch_*.pt"
        ):

            if old != cp:
                old.unlink()

        for old in cp_dir.glob(
            "epoch_*_eval.jsonl"
        ):

            if old != hist:
                old.unlink()

        print(
            f"[C16 BACKUP] {run_id} "
            f"epoch={epoch} "
            f"val_rows={len(records)} "
            f"sha256={info['sha256'][:12]}",
            flush=True,
        )

    # ========================================================
    # 9. FINALIZE TRAINING — NO TEST
    # ========================================================

    def finalize(
        run,
        run_dir,
        output_dir=None,
        recovered=None,
    ):
        rid = (
            run["run_id"]
        )

        best = (
            run_dir / "model_best.pkl"
        )

        # ====================================================
        # RECOVER FINAL MODEL FROM EPOCH300 CHECKPOINT
        # ====================================================

        if recovered is not None:

            info, cp, history = recovered

            assert (
                int(info["epoch"]) == 300
            )

            state = torch.load(
                cp,
                map_location="cpu",
                weights_only=False,
            )

            assert (
                int(state["epoch"]) == 300
            )

            assert (
                state["best_model_state"]
                is not None
            ), (
                "Missing best state"
            )

            torch.save(
                state["best_model_state"],
                best,
            )

            del state

            curve = history

        else:

            assert (
                output_dir is not None
            )

            source_best = (
                Path(output_dir)
                / "model_best.pkl"
            )

            assert source_best.is_file(), (
                f"Missing best model: {rid}"
            )

            shutil.copy2(
                source_best,
                best,
            )

            curve = (
                Path(output_dir)
                / "epoch_eval.jsonl"
            )

        # ====================================================
        # VALIDATION HISTORY
        # ====================================================

        rows = [
            json.loads(s)

            for s in (
                curve.read_text()
                .splitlines()
            )

            if s.strip()
        ]

        epochs = [
            int(x["epoch"])

            for x in rows
        ]

        assert (
            epochs == list(range(70, 301))
        ), (
            f"Validation history incomplete "
            f"for {rid}: {len(epochs)} rows"
        )

        assert all(
            r["test_raw_hits1_observational"]
            is None

            for r in rows
        )

        # ====================================================
        # BEST CHECKPOINT SELECTED BY VALIDATION MRR
        # ====================================================

        best_row = max(
            rows,

            key=lambda r: float(
                r["validation_raw_mrr"]
            ),
        )

        curve_dst = (
            run_dir / "epoch_eval.jsonl"
        )

        if (
            curve.resolve()
            != curve_dst.resolve()
        ):

            shutil.copy2(
                curve,
                curve_dst,
            )

        # ====================================================
        # COMPLETION METADATA
        # ====================================================

        completed = {
            "status": (
                "COMPLETED_TRAINING_TEST_NOT_EVALUATED"
            ),

            "run_id": rid,

            "seed": run["seed"],

            "arm": run["arm"],

            "epochs": 300,

            "best_epoch": int(
                best_row["epoch"]
            ),

            "best_val_mrr": float(
                best_row["validation_raw_mrr"]
            ),

            "best_val_hits1_pct": float(
                best_row["validation_raw_hits1"]
            ),

            "model_best_sha256": sha256(best),

            "history_sha256": sha256(curve_dst),

            "source_main_sha256": (
                manifest["source_hashes"]
                [run["source"]]["main.py"]
            ),

            "config_sha256": hashlib.sha256(
                json.dumps(
                    run["config"],
                    sort_keys=True,
                ).encode()
            ).hexdigest(),

            "test_evaluated": False,
        }

        remote = remote_run(
            rid
        )

        # Upload best model and validation history.

        upload(
            best,
            remote + "/model_best.pkl",
        )

        upload(
            curve_dst,
            remote + "/epoch_eval.jsonl",
        )

        # Completion marker uploaded LAST.

        json_write(
            run_dir / "completed.json",
            completed,
        )

        upload(
            run_dir / "completed.json",
            remote + "/completed.json",
        )

        print(
            f"[C16 COMPLETE] {rid} "
            f"best_epoch={completed['best_epoch']} "
            f"val_mrr={completed['best_val_mrr']:.6f}",
            flush=True,
        )

        return completed

    # ========================================================
    # 10. TRAINING OVERVIEW
    # ========================================================

    print("=" * 72)

    print(
        "CELL 16 — STAGE 5 FINAL TRAINING | NO TEST"
    )

    print(
        "[C15]",
        CELL15_ID,
    )

    print(
        "[RUN MATRIX]",
        len(matrix),
        "runs × up to 300 epochs",
    )

    print(
        "[MAX NEW RUNS]",
        MAX_NEW_RUNS,
    )

    print(
        "[DRIVE]",
        REMOTE16,
    )

    print(
        "=" * 72,
        flush=True,
    )

    # ========================================================
    # 11. RUN FINAL TRAINING MATRIX
    # ========================================================

    completed_runs = []
    started = 0

    for run in matrix:

        rid = run["run_id"]

        run_dir = (
            root / "runs" / rid
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        remote = remote_run(
            rid
        )

        completed_p = (
            run_dir / "completed.json"
        )

        # ====================================================
        # SKIP PREVIOUSLY COMPLETED RUN
        # ====================================================

        if (
            not completed_p.is_file()
            and fs_drive.exists(
                remote + "/completed.json"
            )
        ):

            fetch(
                remote + "/completed.json",
                completed_p,
            )

        if completed_p.is_file():

            result = json_read(
                completed_p
            )

            assert (
                result["run_id"]
                == rid
            )

            assert (
                result["status"]
                == "COMPLETED_TRAINING_TEST_NOT_EVALUATED"
            )

            assert fs_drive.exists(
                remote + "/model_best.pkl"
            )

            assert fs_drive.exists(
                remote + "/epoch_eval.jsonl"
            )

            print(
                "[C16 SKIP COMPLETED]",
                rid,
                flush=True,
            )

            completed_runs.append(
                result
            )

            continue

        # ====================================================
        # RUN LIMIT
        # ====================================================

        if (
            MAX_NEW_RUNS is not None
            and started >= MAX_NEW_RUNS
        ):

            print(
                "[C16 PAUSE] Reached MAX_NEW_RUNS",
                flush=True,
            )

            break

        started += 1

        # ====================================================
        # RECOVER CHECKPOINT
        # ====================================================

        recovered = recover_checkpoint(
            run_dir,
            rid,
        )

        if (
            recovered is not None
            and int(
                recovered[0]["epoch"]
            ) == 300
        ):

            print(
                "[C16 FINALIZE RESUMED]",
                rid,
                flush=True,
            )

            completed_runs.append(
                finalize(
                    run,
                    run_dir,
                    recovered=recovered,
                )
            )

            continue

        # ====================================================
        # CREATE NEW TRAINING ATTEMPT
        # ====================================================

        attempt = (
            run_dir
            / "attempts"
            / (
                time.strftime(
                    "%Y%m%d_%H%M%S_"
                )
                + uuid.uuid4().hex[:8]
            )
        )

        attempt.mkdir(
            parents=True,
            exist_ok=False,
        )

        cfg = dict(
            run["config"]
        )

        cfg["training_data"] = (
            str(Path(data_dir).resolve())
            + "/"
        )

        cfg["output"] = (
            str(attempt / "results")
            + "/"
        )

        job_p = (
            attempt / "job.json"
        )

        json_write(
            job_p,
            {
                "source": str(
                    (
                        sources / run["source"]
                    ).resolve()
                ),

                "config": cfg,
            },
        )

        # ====================================================
        # CLEAN SUBPROCESS ENVIRONMENT
        # ====================================================

        env = os.environ.copy()

        env.update({
            k: str(v)

            for k, v in (
                run["env"].items()
            )
        })

        env["WANDB_MODE"] = "disabled"

        env["TOKENIZERS_PARALLELISM"] = "false"

        env.pop(
            "CLAMEA_RESUME_STATE",
            None,
        )

        env.pop(
            "CELL16_PREVIOUS_HISTORY",
            None,
        )

        # Prevent flags from other notebook
        # experiments leaking into final training.

        for flag in (
            "CELL13_GRAPH_MODE",
            "CELL13_PSEUDO_LOSS",
            "CELL13_SAVE_BEST",
            "CELL13_REF_E70_MRR",
            "CELL13_REF_E70_H1",
            "CELL14_DYNAMIC_STEPS",
        ):

            if flag not in run["env"]:

                env.pop(
                    flag,
                    None,
                )

        # ====================================================
        # RESUME OR FRESH
        # ====================================================

        if recovered:

            env["CLAMEA_RESUME_STATE"] = str(
                recovered[1].resolve()
            )

            env["CELL16_PREVIOUS_HISTORY"] = str(
                recovered[2].resolve()
            )

            print(
                f"[C16 RESUME] {rid} "
                f"from epoch={recovered[0]['epoch']}",
                flush=True,
            )

        else:

            print(
                f"[C16 FRESH] {rid} epoch=1",
                flush=True,
            )

        assert (
            env["PYTHONHASHSEED"] == "0"
        )

        assert (
            env["CUBLAS_WORKSPACE_CONFIG"]
            == ":4096:8"
        )

        assert (
            sources / run["source"]
        ).is_dir()

        # ====================================================
        # START INDEPENDENT TRAINING SUBPROCESS
        # ====================================================

        log_p = (
            attempt / "train.log"
        )

        output = None

        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(worker_p),
                str(job_p),
            ],

            cwd=str(
                sources / run["source"]
            ),

            env=env,

            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,

            text=True,
            bufsize=1,
        )

        try:

            with log_p.open(
                "w",
                encoding="utf-8",
            ) as logfile:

                for raw in proc.stdout:

                    logfile.write(raw)
                    logfile.flush()

                    line = raw.strip()

                    # ========================================
                    # OUTPUT DIRECTORY
                    # ========================================

                    if line.startswith(
                        "[C16_OUTPUT] "
                    ):

                        output = (
                            line.split(
                                " ",
                                1,
                            )[1].strip()
                        )

                    # ========================================
                    # CHECKPOINT EVENT
                    # ========================================

                    m = re.search(
                        r"SAVED RESUME STATE epoch\s+(\d+)",
                        line,
                    )

                    if m:

                        assert output is not None, (
                            "Checkpoint before output path"
                        )

                        backup(
                            run_dir,
                            rid,
                            int(m.group(1)),
                            output,
                        )

                        continue

                    # ========================================
                    # TRAINING PROGRESS
                    # ========================================

                    m = re.match(
                        r"^epoch (\d+):$",
                        line,
                    )

                    if (
                        m
                        and int(m.group(1)) % 10 == 0
                    ):

                        print(
                            f"[C16 TRAIN] {rid} "
                            f"epoch={m.group(1)}/300",
                            flush=True,
                        )

                    # ========================================
                    # VALIDATION PROGRESS
                    # ========================================

                    m = re.match(
                        (
                            r"\[VALID_MONITOR\] "
                            r"epoch=(\d+) "
                            r".*raw_mrr=([0-9.eE+-]+)"
                        ),
                        line,
                    )

                    if (
                        m
                        and int(m.group(1)) % 10 == 0
                    ):

                        print(
                            f"[C16 VAL] {rid} "
                            f"epoch={m.group(1)} "
                            f"MRR={m.group(2)}",
                            flush=True,
                        )

                    # ========================================
                    # ERRORS AND WORKER EVENTS
                    # ========================================

                    if line.startswith((
                        "[C16_",
                        "Traceback",
                        "ImportError",
                        "RuntimeError",
                        "AssertionError",
                        "CUDA out of memory",
                    )):

                        print(
                            f"[{rid}] {line}",
                            flush=True,
                        )

                code = proc.wait()

            # ================================================
            # UPLOAD TRAINING LOG
            # ================================================

            upload(
                log_p,
                remote
                + f"/logs/{attempt.name}.log",
            )

            # ================================================
            # ERROR HANDLING
            # ================================================

            if code != 0:

                tail = "\n".join(
                    log_p.read_text()
                    .splitlines()[-25:]
                )

                raise RuntimeError(
                    f"{rid} failed (exit={code}).\n"
                    f"Log: {log_p}\n"
                    f"Last 25 log lines:\n{tail}"
                )

            assert output is not None, (
                "Worker returned without output folder"
            )

            # ================================================
            # FINALIZE RUN
            # ================================================

            result = finalize(
                run,
                run_dir,
                output_dir=output,
            )

            completed_runs.append(
                result
            )

            # ================================================
            # CLEAN LARGE TEMPORARY FILES
            # ================================================

            # Only clean after successful Drive uploads.

            shutil.rmtree(
                run_dir / "attempts",
                ignore_errors=True,
            )

            shutil.rmtree(
                run_dir / "checkpoints",
                ignore_errors=True,
            )

        except BaseException:

            if proc.poll() is None:

                proc.terminate()

                try:
                    proc.wait(
                        timeout=20
                    )

                except subprocess.TimeoutExpired:

                    proc.kill()
                    proc.wait()

            print(
                f"[C16 STOPPED] {rid} "
                f"| log: {log_p}",
                flush=True,
            )

            print(
                "[C16 NOTE] Uploaded checkpoints "
                "are retained.",
                flush=True,
            )

            raise

    # ========================================================
    # 12. SUMMARY — NO TEST
    # ========================================================

    summary = {
        "schema": "CLAMEA_STAGE5_RUNNER",

        "cell15": CELL15_ID,

        "planned_runs": len(matrix),

        "completed_runs": len(completed_runs),

        "remaining": (
            len(matrix)
            - len(completed_runs)
        ),

        "results": [
            {
                k: d[k]

                for k in (
                    "run_id",
                    "best_epoch",
                    "best_val_mrr",
                    "best_val_hits1_pct",
                )
            }

            for d in completed_runs
        ],

        "test_evaluated": False,
    }

    out_summary = (
        root / "stage5_training_summary.json"
    )

    json_write(
        out_summary,
        summary,
    )

    upload(
        out_summary,
        REMOTE16 + "/stage5_training_summary.json",
    )

    print(
        "[CELL16 SUMMARY]",
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )

    print(
        "[DONE] No Test evaluated. "
        "Rerun to continue.",
        flush=True,
    )

    return summary


# ============================================================
# EXECUTE CELL 16
# ============================================================

cell16_stage5_runner()

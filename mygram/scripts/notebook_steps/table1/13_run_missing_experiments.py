import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

WORK_ROOT = Path("/content/mygram_work/runs")
STORE_ROOT = DRIVE_ROOT / "runs/experiments"
MATRIX_PATH = Path("configs/experiments.json")
DRIVER_LOG_ROOT = Path("/content/mygram_work/driver_logs")

# Tăng từ 25 -> 100 để giảm số lần publish/verify Drive.
CHUNK_EPOCHS = 100
MAX_WALL_MINUTES = 600
# Notebook chỉ báo trạng thái mỗi 60 giây.
HEARTBEAT_SECONDS = 60
# Thử chạy nhanh trước, không gradient checkpointing.
# Nếu CUDA OOM -> tự retry experiment đó với gradient checkpointing.
FAST_MODE_FIRST = False

WORK_ROOT.mkdir(parents=True, exist_ok=True)
STORE_ROOT.mkdir(parents=True, exist_ok=True)
DRIVER_LOG_ROOT.mkdir(parents=True, exist_ok=True)

STATUS_PATH = STORE_ROOT / "matrix_status.json"
LOCK_PATH = STORE_ROOT / "runner.lock"

# ============================================================
# HELPERS
# ============================================================
def load_json(path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except Exception:
        return {} if default is None else default
def load_matrix_status():
    return load_json(
        STATUS_PATH,
        {
            "counts": {},
            "runs": {},
        },
    )
def get_run_state(experiment_id):
    matrix_status = load_matrix_status()

    return (
        matrix_status
        .get("runs", {})
        .get(experiment_id, {})
    )
def get_next_experiment(experiments):
    """
    Lấy experiment đầu tiên chưa completed.

    Experiment completed sẽ được bỏ qua,
    nên khi chạy lại cell sẽ tiếp tục từ experiment
    đang paused/failed/planned.
    """
    matrix_status = load_matrix_status()
    runs = matrix_status.get("runs", {})

    for exp in experiments:
        state = runs.get(exp["id"], {})

        if state.get("status") != "completed":
            return exp
    return None
def count_completed(experiments):
    matrix_status = load_matrix_status()
    runs = matrix_status.get("runs", {})

    return sum(
        1
        for exp in experiments
        if runs.get(exp["id"], {}).get("status") == "completed"
    )
def print_counts():
    status = load_matrix_status()

    print(
        "Matrix:",
        json.dumps(
            status.get("counts", {}),
            ensure_ascii=False,
        ),
    )
def is_cuda_oom(state, log_path=None):
    """
    Kiểm tra lỗi CUDA OOM từ matrix_status và log.
    """
    text_parts = [
        str(state.get("error", "")),
        str(state.get("training_error", "")),
        str(state.get("traceback", "")),
    ]
    if log_path is not None and log_path.exists():
        try:
            lines = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

            text_parts.append(
                "\n".join(lines[-100:])
            )

        except Exception:
            pass

    text = " ".join(text_parts).lower()

    tokens = [
        "cuda out of memory",
        "cuda oom",
        "out of memory",
        "cublas_status_alloc_failed",
    ]

    return any(token in text for token in tokens)


# ============================================================
# RUNNER LOCK
# ============================================================

def check_runner_lock():
    """
    Chỉ tự xóa stale lock nếu lock thuộc đúng runtime hiện tại
    và PID không còn tồn tại.
    """

    if not LOCK_PATH.exists():
        return

    owner = load_json(LOCK_PATH)

    lock_host = owner.get("host")
    lock_pid = owner.get("pid")

    current_host = socket.gethostname()

    # --------------------------------------------------------
    # Lock thuộc runtime hiện tại
    # --------------------------------------------------------

    if lock_host == current_host and lock_pid:

        try:
            os.kill(int(lock_pid), 0)
            alive = True

        except (ProcessLookupError, ValueError):
            alive = False

        except PermissionError:
            alive = True

        if alive:
            raise RuntimeError(
                f"Runner khác vẫn đang chạy: PID={lock_pid}"
            )

        print("⚠️ Phát hiện stale runner.lock.")
        LOCK_PATH.unlink()
        print("✅ Đã xóa stale runner.lock.")

        return

    # --------------------------------------------------------
    # Lock của runtime khác
    # --------------------------------------------------------

    raise RuntimeError(
        "runner.lock thuộc Colab/runtime khác.\n"
        f"{json.dumps(owner, indent=2)}\n\n"
        "Nếu chắc chắn không còn session MyGram nào khác đang chạy, "
        "hãy xóa runner.lock rồi chạy lại cell."
    )


# ============================================================
# COMMAND
# ============================================================

def build_cmd(
    target_id,
    gradient_checkpointing=False,
):
    cmd = [
        sys.executable,
        "-u",

        "scripts/run_experiments.py",

        "--matrix",
        str(MATRIX_PATH),

        # Chạy trực tiếp experiment cần thiết.
        # Không quét lại toàn bộ completed runs.
        "--run-id",
        target_id,

        "--resume",

        "--max-runs",
        "1",

        "--chunk-epochs",
        str(CHUNK_EPOCHS),

        "--max-wall-minutes",
        str(MAX_WALL_MINUTES),

        "--data-root",
        "/content/mygram_data",

        "--work-root",
        str(WORK_ROOT),

        "--store-root",
        str(STORE_ROOT),

        "--device",
        "cuda:0",
    ]

    if gradient_checkpointing:
        cmd.append("--gradient-checkpointing")

    return cmd


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    target,
    gradient_checkpointing,
):
    target_id = target["id"]

    cmd = build_cmd(
        target_id,
        gradient_checkpointing=gradient_checkpointing,
    )

    mode = (
        "VRAM-SAFE / gradient checkpointing"
        if gradient_checkpointing
        else "FAST / no gradient checkpointing"
    )

    print(f"  Mode: {mode}")

    log_path = DRIVER_LOG_ROOT / f"{target_id}.log"

    env = os.environ.copy()

    # Child src.run cũng kế thừa biến này.
    env["PYTHONUNBUFFERED"] = "1"

    started = time.monotonic()

    with log_path.open(
        "a",
        encoding="utf-8",
        errors="backslashreplace",
    ) as log:

        log.write("\n\n")
        log.write("=" * 80 + "\n")

        log.write(
            f"START "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        log.write(
            f"MODE {mode}\n"
        )

        log.write(
            " ".join(map(str, cmd)) + "\n"
        )

        log.flush()

        process = subprocess.Popen(
            cmd,

            # Full log ghi vào file, không spam notebook.
            stdout=log,
            stderr=subprocess.STDOUT,

            text=True,

            env=env,

            # Runner + src.run cùng process group.
            start_new_session=True,
        )

        try:

            while True:
                returncode = process.poll()

                if returncode is not None:
                    break

                time.sleep(HEARTBEAT_SECONDS)

                elapsed = (
                    time.monotonic() - started
                ) / 60

                state = get_run_state(target_id)

                epoch_next = state.get("epoch_next")
                status_name = state.get(
                    "status",
                    "running",
                )

                if epoch_next is not None:

                    print(
                        f"  ⏳ {elapsed:.1f} min"
                        f" | status={status_name}"
                        f" | published epoch={epoch_next}"
                    )

                else:

                    print(
                        f"  ⏳ {elapsed:.1f} min"
                        f" | status={status_name}"
                    )

            returncode = process.wait()

        # ====================================================
        # USER STOP CELL
        # ====================================================

        except KeyboardInterrupt:

            print(
                "\n⚠️ Đang yêu cầu runner dừng an toàn..."
            )

            try:
                # Cho Python cơ hội chạy finally.
                os.killpg(
                    os.getpgid(process.pid),
                    signal.SIGINT,
                )

                process.wait(timeout=30)

            except subprocess.TimeoutExpired:

                print(
                    "⚠️ Runner chưa dừng, gửi SIGTERM..."
                )

                os.killpg(
                    os.getpgid(process.pid),
                    signal.SIGTERM,
                )

                try:
                    process.wait(timeout=10)

                except subprocess.TimeoutExpired:

                    print(
                        "⚠️ Force kill process group..."
                    )

                    os.killpg(
                        os.getpgid(process.pid),
                        signal.SIGKILL,
                    )

                    process.wait()

            print(
                "⏸️ Đã dừng."
            )

            print(
                "Các snapshot đã publish trên Drive vẫn được giữ."
            )

            raise

    elapsed = (
        time.monotonic() - started
    ) / 60

    return (
        returncode,
        elapsed,
        log_path,
    )


# ============================================================
# LOAD MATRIX
# ============================================================

matrix = json.loads(
    MATRIX_PATH.read_text(
        encoding="utf-8"
    )
)

# Chỉ chạy hai cấu hình còn thiếu của Table 1.
TARGET_EXPERIMENT_IDS = [
    "table1_fbyg_r050",
    "table1_fbyg_r080",
]

experiment_by_id = {
    experiment["id"]: experiment
    for experiment in matrix["experiments"]
}

missing_ids = [
    experiment_id
    for experiment_id in TARGET_EXPERIMENT_IDS
    if experiment_id not in experiment_by_id
]

if missing_ids:
    raise KeyError(
        f"Không tìm thấy experiment ID trong matrix: {missing_ids}"
    )

experiments = [
    experiment_by_id[experiment_id]
    for experiment_id in TARGET_EXPERIMENT_IDS
]

total = len(experiments)

print(
    "Selected Table 1 experiments:",
    [experiment["id"] for experiment in experiments],
)


print("=" * 70)
print("MyGram experiment driver")
print(f"Total experiments : {total}")
print(f"Chunk epochs      : {CHUNK_EPOCHS}")
print(
    f"Fast mode first   : {FAST_MODE_FIRST}"
)
print("=" * 70)

print_counts()


# ============================================================
# MODE OVERRIDE
#
# Nếu FAST mode OOM ở experiment hiện tại,
# experiment đó sẽ được retry với gradient checkpointing.
#
# Sau khi completed, experiment kế tiếp lại thử FAST mode.
# ============================================================

force_safe_for_target = set()


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

while True:

    # --------------------------------------------------------
    # 1. Tìm experiment tiếp theo
    # --------------------------------------------------------

    target = get_next_experiment(
        experiments
    )

    if target is None:

        print("\n" + "=" * 70)

        print(
            "✅ SELECTED TABLE 1 EXPERIMENTS COMPLETED"
        )

        print_counts()

        print("=" * 70)

        break


    target_id = target["id"]

    completed_count = count_completed(
        experiments
    )


    print("\n" + "=" * 70)

    print(
        f"▶ [{completed_count + 1}/{total}] "
        f"START {target_id}"
    )

    print(
        f"  {target['dataset']}"
        f" | rate={target['rate']}"
        f" | group={target['group']}"
    )


    old_state = get_run_state(
        target_id
    )

    if old_state.get("epoch_next") is not None:

        print(
            f"  Resume from epoch_next="
            f"{old_state.get('epoch_next')}"
        )


    # --------------------------------------------------------
    # 2. Check stale lock
    # --------------------------------------------------------

    check_runner_lock()


    # --------------------------------------------------------
    # 3. Chọn FAST / SAFE mode
    # --------------------------------------------------------
    use_gradient_checkpointing = (
        not FAST_MODE_FIRST
        or target_id in force_safe_for_target
    )
    # --------------------------------------------------------
    # 4. Run experiment
    # --------------------------------------------------------
    (
        returncode,
        elapsed,
        log_path,
    ) = run_experiment(
        target,
        gradient_checkpointing=
            use_gradient_checkpointing,
    )


    # --------------------------------------------------------
    # 5. Đọc status sau process
    # --------------------------------------------------------

    state = get_run_state(
        target_id
    )

    state_name = state.get(
        "status"
    )


    # ========================================================
    # PROCESS ERROR
    # ========================================================

    if returncode != 0:

        # ----------------------------------------------------
        # CUDA OOM khi FAST mode
        # -> retry cùng experiment với gradient checkpointing
        # ----------------------------------------------------

        cuda_oom = is_cuda_oom(
            state,
            log_path,
        )

        if (
            cuda_oom
            and not use_gradient_checkpointing
        ):

            print(
                "⚠️ FAST mode bị CUDA OOM."
            )

            print(
                "↻ Retry cùng experiment với "
                "gradient checkpointing."
            )

            force_safe_for_target.add(
                target_id
            )

            # Không chuyển experiment.
            # while loop sẽ chọn lại target hiện tại.
            continue


        # ----------------------------------------------------
        # Lỗi thật
        # ----------------------------------------------------

        print(
            f"\n❌ FAILED {target_id}"
        )

        print(
            f"returncode={returncode}"
        )

        print(
            f"status={state_name}"
        )

        print(
            f"Full log: {log_path}"
        )


        # In 30 dòng cuối để debug nhanh.
        try:

            lines = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

            print(
                "\n--- Last 30 log lines ---"
            )

            print(
                "\n".join(
                    lines[-30:]
                )
            )

        except Exception:
            pass


        raise RuntimeError(
            f"{target_id} failed "
            f"(returncode={returncode})"
        )


    # ========================================================
    # COMPLETED
    # ========================================================

    if state_name == "completed":

        print(
            f"✅ DONE {target_id} "
            f"({elapsed:.1f} min)"
        )

        print_counts()

        # Experiment sau sẽ lại thử FAST mode.
        force_safe_for_target.discard(
            target_id
        )

        # Tự động chuyển experiment tiếp theo.
        continue


    # ========================================================
    # PAUSED
    # ========================================================

    if state_name == "paused":

        print(
            f"⏸️ {target_id} paused "
            f"at epoch_next="
            f"{state.get('epoch_next')}."
        )

        print(
            "↻ Resume cùng experiment..."
        )

        # while loop chọn lại cùng target.
        continue


    # ========================================================
    # UNEXPECTED STATE
    # ========================================================

    print(
        f"⚠️ Unexpected state "
        f"for {target_id}:"
    )

    print(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        )
    )

    raise RuntimeError(
        f"Unexpected state: {state_name}"
    )

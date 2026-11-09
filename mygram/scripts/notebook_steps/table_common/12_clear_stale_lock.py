# Xóa stale lock của runtime cũ.
# Điều kiện sử dụng notebook: mọi session MyGram cũ đã được dừng.
import json
import os
import socket
from pathlib import Path

LOCK_PATH = DRIVE_ROOT / "runs" / "experiments" / "runner.lock"

if LOCK_PATH.exists():
    try:
        owner = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except Exception:
        owner = {"raw": LOCK_PATH.read_text(encoding="utf-8", errors="replace")}

    print("Found runner.lock:")
    print(json.dumps(owner, indent=2, ensure_ascii=False))

    # Không xóa lock nếu nó thuộc chính runtime này và PID còn sống.
    same_host = owner.get("host") == socket.gethostname()
    pid = owner.get("pid")
    alive = False

    if same_host and pid is not None:
        try:
            os.kill(int(pid), 0)
            alive = True
        except (ProcessLookupError, ValueError):
            alive = False
        except PermissionError:
            alive = True

    if alive:
        raise RuntimeError(
            f"Một MyGram runner vẫn đang chạy trong runtime này: PID={pid}"
        )

    LOCK_PATH.unlink()
    print("✅ Đã xóa stale runner.lock:", LOCK_PATH)
else:
    print("✅ Không có runner.lock cũ")

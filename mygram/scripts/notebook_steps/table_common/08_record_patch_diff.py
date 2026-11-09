# Lưu diff của thay đổi checkpoint để dùng trong phần provenance/limitations.
import difflib

PATCH_REPORT_DIR = DRIVE_ROOT / "reports" / "provenance"
PATCH_REPORT_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIFF_PATH = PATCH_REPORT_DIR / "run_py_checkpoint_patch.diff"

if BACKUP.exists() and RUN_PY.exists():
    before = BACKUP.read_text(encoding="utf-8").splitlines(keepends=True)
    after = RUN_PY.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile="src/run.py (source bundle)",
            tofile="src/run.py (checkpoint optimization)",
        )
    )
    PATCH_DIFF_PATH.write_text(diff, encoding="utf-8")
    print("✅ Saved patch provenance:", PATCH_DIFF_PATH)
else:
    raise FileNotFoundError("Không tìm thấy run.py hoặc backup để lưu diff")

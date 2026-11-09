# Kiểm thử lại SAU KHI patch run.py.
# Bộ test trước đây chạy trước patch nên không đủ làm bằng chứng cho source hiện tại.
import subprocess
import sys

POST_PATCH_TEST_LOG = (
    DRIVE_ROOT / "environment" / "post_patch_unittest.log"
)
POST_PATCH_TEST_LOG.parent.mkdir(parents=True, exist_ok=True)

result = subprocess.run(
    [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    ],
    text=True,
    capture_output=True,
)

test_output = result.stdout + "\n" + result.stderr
POST_PATCH_TEST_LOG.write_text(test_output, encoding="utf-8")
print(test_output)

if result.returncode != 0:
    raise RuntimeError(
        f"Post-patch unit tests failed: {POST_PATCH_TEST_LOG}"
    )

print("✅ Post-patch unit tests passed")
print("Log:", POST_PATCH_TEST_LOG)

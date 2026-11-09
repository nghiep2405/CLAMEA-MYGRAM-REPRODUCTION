
# 1. Kiểm tra FAISS trong đúng Python environment
check = subprocess.run(
    [
        sys.executable,
        "-c",
        "import faiss; print('FAISS:', faiss.__version__)",
    ],
    capture_output=True,
    text=True,
)

# 2. Chỉ cài đặt nếu chưa có
if check.returncode != 0:
    print("[SETUP] Installing faiss-cpu...", flush=True)

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "faiss-cpu",
    ])

# 3. Kiểm tra import toàn bộ CLAMEA source
source = (
    Path(WORK)
    / "final_training"
    / "cell16_stage5_cell15_20261025_150915_b52701"
    / "frozen"
    / "sources"
    / "deterministic_method"
)

assert source.is_dir(), f"Source not found: {source}"

subprocess.run(
    [
        sys.executable,
        "-c",
        (
            "import faiss; "
            "import kgs; "
            "import main; "
            "print('[C16 PREFLIGHT] FAISS + CLAMEA imports: PASS')"
        ),
    ],
    cwd=str(source),
    check=True,
)

print("[DONE] Environment ready for Cell 16")

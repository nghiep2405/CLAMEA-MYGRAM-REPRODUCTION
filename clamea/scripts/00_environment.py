import os
import sys
import json
import time
import random
import shutil
import hashlib
import zipfile
import subprocess
from pathlib import Path

import numpy as np
import torch
import marimo as mo

PROJECT_FOLDER_ID = "1wCvI_vGxmCBCYytGXl7CWctdPQ6qfhsb"

DATA_NAME = "CLAMEA_FBDB15K_protocolB.zip"

DATA_SHA256 = (
    "23458cde8d73943b04aceaec191a9e116fedcca7c86a4c4e4c00c3404431f5cb"
)

PINNED_COMMIT = "30affb5c1cbfd1db29857358a1e3b7cdf84d82d9"

EXPECTED_SPLITS = {
    "721train_links": (
        "79eb263b4efda7128eeffd10a703947844ed706d4900ec27fc2092e704af701c",
        2569
    ),
    "721valid_links": (
        "b4ff951d6228838b6b54f9da27c58ee9436fbec825895f00f1d8227edb0383f5",
        1284
    ),
    "721test_links": (
        "f0beaa573df183e95694a1c2236b77dcd33a4331859b683cb1ef53e0dd78d833",
        8993
    ),
}

WORK = Path("/tmp/clamea_paper_method_molab")
WORK.mkdir(parents=True, exist_ok=True)


def file_sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as stream:
        for piece in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b""
        ):
            h.update(piece)

    return h.hexdigest()


def run_checked(argv, *, cwd=None):
    done = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        capture_output=True
    )

    if done.returncode:
        raise RuntimeError(
            "COMMAND FAILED: " + repr(argv)
            + "\nSTDOUT:\n" + done.stdout[-5000:]
            + "\nSTDERR:\n" + done.stderr[-5000:]
        )

    return done.stdout.strip()


ENV_INFO = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": bool(torch.cuda.is_available()),
    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
}

print("CLAMEA PAPER-METHOD ADAPTATION · PREFLIGHT ONLY")
print(json.dumps(ENV_INFO, indent=2))
print("Full training is LOCKED in Cell 07.")

"""Fail-fast inventory for a MyGram Google Colab CUDA runtime."""
import argparse
import datetime
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_registry import ARCHIVE_BYTES, ARCHIVE_SHA256, sha256, write_json


PACKAGES = ('torch', 'numpy', 'scipy', 'transformers', 'tqdm', 'psutil')


def gib(value):
    return value / 2**30


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors='backslashreplace')
        sys.stderr.reconfigure(errors='backslashreplace')
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'environment.json')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--drive-root', type=Path)
    parser.add_argument('--local-path', type=Path, default=Path('/content'))
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--min-local-free-gib', type=float, default=8.0)
    parser.add_argument('--min-drive-free-gib', type=float, default=22.0)
    args = parser.parse_args(argv)

    import torch
    failures = []
    report = {
        'schema_version': 1,
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'platform': platform.platform(),
        'python': sys.version,
        'executable': sys.executable,
        'packages': {},
    }
    for name in PACKAGES:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
            failures.append(f'missing package: {name}')

    report['torch'] = {
        'version': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'cudnn': torch.backends.cudnn.version(),
        'devices': [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report['torch']['devices'].append({
                'index': index, 'name': props.name,
                'total_memory': props.total_memory,
                'compute_capability': [props.major, props.minor],
            })
        try:
            x = torch.randn(128, 128, device='cuda', requires_grad=True)
            (x @ x.T).square().mean().backward()
            torch.cuda.synchronize()
            report['cuda_smoke'] = 'passed'
        except Exception as exc:
            report['cuda_smoke'] = repr(exc)
            failures.append(f'CUDA forward/backward failed: {exc!r}')
    elif args.require_cuda:
        failures.append('CUDA was required but torch.cuda.is_available() is false')

    nvidia = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free',
         '--format=csv,noheader'], capture_output=True, text=True, check=False,
    ) if shutil.which('nvidia-smi') else None
    report['nvidia_smi'] = None if nvidia is None else {
        'returncode': nvidia.returncode,
        'stdout': nvidia.stdout.strip(), 'stderr': nvidia.stderr.strip(),
    }

    local_usage = shutil.disk_usage(args.local_path)
    report['local_disk'] = {
        'path': str(args.local_path), 'free_gib': gib(local_usage.free),
        'required_free_gib': args.min_local_free_gib,
    }
    if gib(local_usage.free) < args.min_local_free_gib:
        failures.append('insufficient Colab local disk')
    if args.drive_root:
        drive_usage = shutil.disk_usage(args.drive_root)
        report['drive_disk'] = {
            'path': str(args.drive_root), 'free_gib': gib(drive_usage.free),
            'required_free_gib': args.min_drive_free_gib,
        }
        if gib(drive_usage.free) < args.min_drive_free_gib:
            failures.append('insufficient Google Drive free space')
    if args.archive:
        archive = args.archive
        ok = (
            archive.is_file() and archive.stat().st_size == ARCHIVE_BYTES
            and sha256(archive) == ARCHIVE_SHA256
        )
        report['archive'] = {
            'path': str(archive), 'expected_bytes': ARCHIVE_BYTES,
            'expected_sha256': ARCHIVE_SHA256, 'verified': ok,
        }
        if not ok:
            failures.append('PMF archive size/SHA-256 mismatch')
    report['failures'] = failures
    report['status'] = 'passed' if not failures else 'failed'
    write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())

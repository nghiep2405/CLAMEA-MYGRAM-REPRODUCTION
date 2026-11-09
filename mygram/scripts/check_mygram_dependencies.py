"""Validate MyGram's dependency slice without failing on unrelated Colab packages."""
import argparse
import datetime
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_registry import write_json


IMPORTS = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'transformers': 'transformers',
    'tqdm': 'tqdm',
    'psutil': 'psutil',
    'torch': 'torch',
}


def locked_versions(path):
    values = {}
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '==' not in line:
            raise ValueError(f'Expected an exact version in Colab lock: {line}')
        name, version = line.split('==', 1)
        values[name.lower()] = version
    return values


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors='backslashreplace')
        sys.stderr.reconfigure(errors='backslashreplace')
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=ROOT/'requirements/colab.lock')
    parser.add_argument('--report', type=Path, default=ROOT/'mygram_dependency_check.json')
    args = parser.parse_args(argv)
    locked = locked_versions(args.lock)
    failures = []
    packages = {}
    imports = {}
    for package, module in IMPORTS.items():
        try:
            version = importlib.metadata.version(package)
            packages[package] = version
            importlib.import_module(module)
            imports[module] = 'passed'
            if package in locked and version != locked[package]:
                failures.append(
                    f'{package} version {version} does not match lock {locked[package]}'
                )
        except Exception as exc:
            packages[package] = None
            imports[module] = repr(exc)
            failures.append(f'{package} import failed: {exc!r}')

    try:
        import numpy as np
        import scipy.sparse as sp
        import torch
        from transformers import get_cosine_schedule_with_warmup
        matrix = sp.coo_matrix(np.eye(3, dtype=np.float32)).tocsr()
        tensor = torch.tensor(matrix.toarray(), dtype=torch.float32, requires_grad=True)
        tensor.square().sum().backward()
        if not torch.isfinite(tensor.grad).all():
            raise RuntimeError('nonfinite numerical smoke gradient')
        if not callable(get_cosine_schedule_with_warmup):
            raise TypeError('transformers scheduler symbol is not callable')
        numerical_smoke = 'passed'
    except Exception as exc:
        numerical_smoke = repr(exc)
        failures.append(f'MyGram numerical/import smoke failed: {exc!r}')

    check = subprocess.run(
        [sys.executable, '-m', 'pip', 'check'], capture_output=True,
        text=True, encoding='utf-8', errors='backslashreplace', check=False,
    )
    pip_lines = [line for line in (check.stdout + '\n' + check.stderr).splitlines() if line.strip()]
    direct_names = set(locked) | {'torch'}
    relevant_conflicts = [
        line for line in pip_lines
        if line.lower().split(maxsplit=1)[0].replace('_', '-') in direct_names
    ]
    if relevant_conflicts:
        failures.extend(f'MyGram dependency conflict: {line}' for line in relevant_conflicts)
    report = {
        'schema_version': 1,
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'passed' if not failures else 'failed',
        'lock': str(args.lock.resolve()),
        'locked_versions': locked,
        'installed_versions': packages,
        'imports': imports,
        'numerical_smoke': numerical_smoke,
        'pip_check': {
            'returncode': check.returncode,
            'all_output': pip_lines,
            'mygram_relevant_conflicts': relevant_conflicts,
            'policy': 'Unrelated preinstalled Colab conflicts are recorded but do not block MyGram.',
        },
        'failures': failures,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())

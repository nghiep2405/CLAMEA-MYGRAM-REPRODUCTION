"""Verify and prepare MyGram data on a fresh Colab local SSD."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_registry import ARCHIVE_BYTES, ARCHIVE_SHA256, sha256, write_json


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    if source.stat().st_size != ARCHIVE_BYTES or sha256(source) != ARCHIVE_SHA256:
        raise ValueError('Drive PMF archive size/SHA-256 mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != ARCHIVE_BYTES or sha256(destination) != ARCHIVE_SHA256:
        temporary = destination.with_name(destination.name + '.copying')
        temporary.unlink(missing_ok=True)
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != ARCHIVE_BYTES or sha256(temporary) != ARCHIVE_SHA256:
            temporary.unlink(missing_ok=True)
            raise OSError('Local archive copy failed verification')
        temporary.replace(destination)
    return destination


def run(command):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='PMF ZIP on Drive')
    parser.add_argument('--cache-root', type=Path, default=Path('/content/mygram_cache'))
    parser.add_argument('--data-root', type=Path, default=Path('/content/mygram_data'))
    parser.add_argument('--validation-root', type=Path, default=Path('/content/mygram_validation'))
    parser.add_argument('--report', type=Path, default=Path('/content/mygram_bootstrap.json'))
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args(argv)
    local_archive = copy_verified(args.archive, args.cache_root/'pmf_mmkg.zip')
    run([sys.executable, ROOT/'scripts/prepare_pmf_data.py', '--archive', local_archive,
         '--data-root', args.data_root])
    if not args.skip_validation:
        run([sys.executable, ROOT/'scripts/validate_data.py', '--data-root', args.data_root,
             '--all', '--output-root', args.validation_root, '--report',
             args.validation_root/'stage_b_validation.json'])
    result = {
        'status': 'passed', 'archive': str(local_archive),
        'archive_sha256': ARCHIVE_SHA256, 'data_root': str(args.data_root.resolve()),
        'validation_root': str(args.validation_root.resolve()),
        'validated': not args.skip_validation,
    }
    write_json(args.report, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

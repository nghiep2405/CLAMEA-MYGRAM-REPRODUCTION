"""Verify a MyGram source bundle before upload or after Drive download."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


BANNED_PARTS = {'.git', '.venv', 'data', 'outputs', 'dist', 'deps', 'sources', '__pycache__'}


def stream_hash(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return stream_hash(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    expected_bundle = manifest['bundle']
    if args.bundle.stat().st_size != expected_bundle['bytes'] or file_hash(args.bundle) != expected_bundle['sha256']:
        raise ValueError('Bundle size/SHA-256 mismatch')
    expected = manifest['files']
    with zipfile.ZipFile(args.bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('ZIP members differ from source bundle manifest')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or any(part in BANNED_PARTS for part in path.parts):
                raise ValueError(f'Unsafe or excluded ZIP member: {name}')
            info = archive.getinfo(name)
            with archive.open(info) as stream:
                digest = stream_hash(stream)
            if info.file_size != expected[name]['bytes'] or digest != expected[name]['sha256']:
                raise ValueError(f'ZIP member hash mismatch: {name}')
    print(json.dumps({
        'status': 'passed', 'bundle': str(args.bundle.resolve()),
        'sha256': expected_bundle['sha256'], 'file_count': len(expected),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

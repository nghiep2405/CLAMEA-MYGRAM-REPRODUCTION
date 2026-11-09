"""Build a clean, hashed source ZIP for upload to Google Drive."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    '.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache',
    'data', 'outputs', 'dist', 'deps', 'sources',
}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.tmp'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def source_files():
    found = []
    for current, directories, filenames in os.walk(ROOT, topdown=True, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in EXCLUDED_PARTS)
        current_path = Path(current)
        for filename in sorted(filenames):
            path = current_path/filename
            relative = path.relative_to(ROOT)
            if (path.suffix in EXCLUDED_SUFFIXES or path.suffix.lower() == '.zip'
                    or path.name.endswith('.uploading')):
                continue
            found.append((relative, path))
    yield from sorted(found, key=lambda item: item[0].as_posix())


def git_output(*args):
    result = subprocess.run(
        ['git', *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/mygram-source.zip')
    parser.add_argument('--manifest', type=Path, default=ROOT/'dist/source_bundle_manifest.json')
    parser.add_argument('--notebook-copy', type=Path, default=ROOT/'dist/MyGram_Colab.ipynb')
    parser.add_argument('--instructions', type=Path, default=ROOT/'dist/UPLOAD_INSTRUCTIONS.txt')
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in source_files():
            name = relative.as_posix()
            archive.write(path, name)
            records[name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    manifest = {
        'schema_version': 1,
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_commit': git_output('rev-parse', 'HEAD'),
        'git_status_porcelain': git_output('status', '--porcelain'),
        'file_count': len(records),
        'files': records,
        'bundle': {
            'name': args.output.name, 'bytes': args.output.stat().st_size,
            'sha256': digest(args.output),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    notebook = ROOT/'notebooks/MyGram_Colab.ipynb'
    if notebook.exists():
        shutil.copy2(notebook, args.notebook_copy)
    args.instructions.write_text(
        '1. Upload mygram-source.zip to MyDrive/MyGramRepro/source/mygram-source.zip\n'
        '2. Upload source_bundle_manifest.json to MyDrive/MyGramRepro/source/source_bundle_manifest.json\n'
        '3. Upload audit/sources/pmf_mmkg.zip to MyDrive/MyGramRepro/sources/pmf_mmkg.zip\n'
        '4. Open MyGram_Colab.ipynb in Google Colab and select a GPU runtime.\n'
        f'5. Expected PMF SHA-256: 2477b8af38d108fe4a66ebfaf460f4176b5b13b00c4082ffeb0637123f3059c1\n'
        f'6. Expected source bundle SHA-256: {manifest["bundle"]["sha256"]}\n',
        encoding='utf-8',
    )
    print(json.dumps(manifest['bundle'] | {'file_count': len(records)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

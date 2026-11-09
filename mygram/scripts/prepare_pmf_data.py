"""Verify the known PMF ZIP, then publish only 40 required files without overwrite."""
import argparse
import datetime
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_registry import (ARCHIVE_BYTES, ARCHIVE_SHA256, SOURCE_URL, REGISTRY,
                               dataset_files, safe_path, sha256, stream_hash, write_json)


def prepare(archive, root, reuse_root=None):
    archive, root = Path(archive).resolve(), Path(root).resolve()
    if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Archive size/SHA-256 mismatch. Stop: audit this archive separately before using baseline statistics.')
    names = [name for key in REGISTRY for name in dataset_files(key)]
    files, sources = {}, {}
    with zipfile.ZipFile(archive) as z:
        all_names = z.namelist()
        for name in names:
            if all_names.count(name) != 1:
                raise ValueError(f'Missing/ambiguous ZIP entry: {name}')
            info = z.getinfo(name)
            with z.open(info) as stream:
                digest = stream_hash(stream)
            files[name] = {'bytes': info.file_size, 'sha256': digest, 'archive_entry': name}
            target = safe_path(root, name)
            if target.exists():
                if target.stat().st_size != info.file_size or sha256(target) != digest:
                    raise FileExistsError(f'Conflict; preserving existing file: {target}')
                sources[name] = 'existing'
            else:
                prior = safe_path(reuse_root, name) if reuse_root else None
                sources[name] = prior if prior and prior.is_file() and sha256(prior) == digest else None
        manifest_path = safe_path(root, 'data_manifest.json')
        previous = None
        if manifest_path.exists():
            import json
            previous = json.loads(manifest_path.read_text(encoding='utf-8'))
            if previous.get('archive', {}).get('sha256') != ARCHIVE_SHA256 or previous.get('files') != files:
                raise FileExistsError(f'Manifest conflict; preserving {manifest_path}')
        # Preflight every conflict and disk requirement before writing any data.
        parent = root
        while not parent.exists():
            parent = parent.parent
        needed = sum(files[n]['bytes'] for n in names if sources[n] != 'existing')
        if shutil.disk_usage(parent).free < needed + 2 * 2**30:
            raise OSError(f'Insufficient free space at {parent}: require {needed} new bytes + 2 GiB headroom')
        for name in names:
            if sources[name] == 'existing':
                continue
            target = safe_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                temporary = Path(out.name)
                try:
                    source = sources[name].open('rb') if sources[name] else z.open(name)
                    with source:
                        shutil.copyfileobj(source, out, 8 * 1024 * 1024)
                except BaseException:
                    out.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                if sha256(temporary) != files[name]['sha256']:
                    raise ValueError(f'Extracted checksum mismatch: {name}')
                # Hard-link publishes atomically and refuses to overwrite a racing writer.
                import os
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            print(f'prepared {name}', flush=True)
    if previous is None:
        write_json(manifest_path, {
            'schema_version': 1, 'prepared_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_url': SOURCE_URL, 'source_repository': 'https://github.com/ninibymilk/PMF-MMEA',
            'archive': {'sha256': ARCHIVE_SHA256, 'bytes': ARCHIVE_BYTES},
            'preparation_script_sha256': sha256(__file__), 'files': files,
            'datasets': {key: {'status': 'prepared_not_validated', 'image_dim_expected': REGISTRY[key][2]} for key in REGISTRY},
            'limitations': ['PMF preprocessing is not confirmed identical to MyGram.', 'DBP15K image vectors have 2048 dimensions; backbone not inferred.', 'FBDB15K duplicate targets are preserved.']})
    else:
        previous['last_verified_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        previous['verification_script_sha256'] = sha256(__file__)
        write_json(manifest_path, previous)
    print(f'Verified {len(files)} files; reused {sum(v == "existing" for v in sources.values())} existing files. Manifest: {manifest_path}', flush=True)


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--reuse-root', type=Path, help='Optional verified text source; never modified')
    args = parser.parse_args()
    prepare(args.archive, args.data_root, args.reuse_root)


if __name__ == '__main__':
    main()

"""Verified local-to-persistent artifact mirroring for preemptible runtimes."""
from __future__ import annotations

import datetime as _datetime
import json
import os
from pathlib import Path
import shutil
import tempfile

from .checkpoint import checkpoint_epoch_next, reconcile_run_to_checkpoint
from .data_registry import sha256, write_json


TRANSIENT_SUFFIXES = ('.tmp', '.uploading')
STORE_METADATA = {'CURRENT.json', 'snapshot_manifest.json'}


def _files(root):
    root = Path(root)
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.name in STORE_METADATA or path.name.endswith(TRANSIENT_SUFFIXES):
            continue
        if relative == 'checkpoints/last.prev.pt':
            continue
        yield relative, path


def _atomic_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=destination.name + '.', suffix='.uploading', dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size or sha256(temporary) != sha256(source):
            raise OSError(f'Copied artifact failed verification: {destination}')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _record(path):
    path = Path(path)
    return {'bytes': path.stat().st_size, 'sha256': sha256(path)}


def validate_snapshot(store_run):
    store_run = Path(store_run)
    current_path = store_run/'CURRENT.json'
    if not current_path.exists():
        raise FileNotFoundError(f'Persistent run has no CURRENT.json: {store_run}')
    current = json.loads(current_path.read_text(encoding='utf-8'))
    for relative, expected in current.get('files', {}).items():
        path = store_run/relative
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Persistent snapshot artifact mismatch: {relative}')
    checkpoint = store_run/'checkpoints'/'last.pt'
    if not checkpoint.exists():
        raise ValueError('Persistent snapshot lacks checkpoints/last.pt')
    actual_epoch = checkpoint_epoch_next(checkpoint)
    if actual_epoch != int(current['epoch_next']):
        raise ValueError('CURRENT epoch does not match last checkpoint')
    return current


def publish_run(local_run, store_run):
    """Publish a run file-by-file, then atomically publish its verified manifest."""
    local_run, store_run = Path(local_run).resolve(), Path(store_run).resolve()
    checkpoint = local_run/'checkpoints'/'last.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Cannot publish without last checkpoint: {checkpoint}')
    store_run.mkdir(parents=True, exist_ok=True)

    # Keep one known-good last checkpoint.  If a VM dies during the remainder of
    # publication, restore_run can roll metrics back to this epoch.
    stored_last = store_run/'checkpoints'/'last.pt'
    source_last_hash = sha256(checkpoint)
    if stored_last.exists() and sha256(stored_last) != source_last_hash:
        _atomic_copy(stored_last, store_run/'checkpoints'/'last.prev.pt')

    for relative, source in _files(local_run):
        destination = store_run/relative
        if destination.exists() and destination.stat().st_size == source.stat().st_size:
            if sha256(destination) == sha256(source):
                continue
        _atomic_copy(source, destination)

    files = {relative: _record(store_run/relative) for relative, _ in _files(local_run)}
    status_path = local_run/'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    manifest = {
        'schema_version': 1,
        'published_at_utc': _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        'run_id': local_run.name,
        'status': status.get('status'),
        'epoch_next': checkpoint_epoch_next(checkpoint),
        'files': files,
    }
    write_json(store_run/'snapshot_manifest.json', manifest)
    write_json(store_run/'CURRENT.json', manifest)
    validate_snapshot(store_run)
    return manifest


def restore_run(store_run, local_run, allow_previous=True):
    """Restore a verified snapshot, falling back to last.prev after a torn publish."""
    store_run, local_run = Path(store_run).resolve(), Path(local_run).resolve()
    recovered_previous = False
    try:
        current = validate_snapshot(store_run)
    except (FileNotFoundError, ValueError):
        previous = store_run/'checkpoints'/'last.prev.pt'
        if not allow_previous or not previous.is_file():
            raise
        current = json.loads((store_run/'CURRENT.json').read_text(encoding='utf-8'))
        recovered_previous = True

    if local_run.exists():
        shutil.rmtree(local_run)
    local_run.mkdir(parents=True)
    for relative, source in _files(store_run):
        destination = local_run/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if recovered_previous:
        shutil.copy2(store_run/'checkpoints'/'last.prev.pt', local_run/'checkpoints'/'last.pt')
    committed_epoch = reconcile_run_to_checkpoint(local_run)
    return {
        'run_id': local_run.name,
        'epoch_next': committed_epoch + 1,
        'recovered_previous': recovered_previous,
        'published_epoch_next': current.get('epoch_next'),
    }

"""Run the 38-run MyGram matrix sequentially with Drive-backed resume."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import shutil
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_store import publish_run, restore_run, validate_snapshot
from src.checkpoint import checkpoint_epoch_next, make_run_id, reconcile_run_to_checkpoint
from src.config import parse_args as parse_training_args, resolved_dict
from src.data_registry import REGISTRY, sha256, write_json


BOOLEAN_FLAGS = {
    'disable_image': '--disable-image',
    'disable_attribute': '--disable-attribute',
    'disable_relation': '--disable-relation',
    'disable_mgd': '--disable-mgd',
    'disable_gram': '--disable-gram',
}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_matrix(path):
    matrix = json.loads(Path(path).read_text(encoding='utf-8'))
    experiments = matrix.get('experiments', [])
    expected = matrix.get('expected', {})
    if len(experiments) != expected.get('unique_long_runs', 38):
        raise ValueError(f'Expected 38 experiments, found {len(experiments)}')
    ids = [item['id'] for item in experiments]
    if len(set(ids)) != len(ids):
        raise ValueError('Experiment IDs are not unique')
    allowed = set(expected['datasets'])
    if allowed != set(REGISTRY) or any(item['dataset'] not in allowed for item in experiments):
        raise ValueError('Experiment matrix dataset set differs from the five-item registry')
    profiles = [item['profile'] for item in experiments]
    if profiles.count('repaired_pmf') != 37 or profiles.count('author_code_compat') != 1:
        raise ValueError('Matrix must contain 37 repaired_pmf and one author_code_compat run')
    sources = set(ids)
    if len(matrix.get('reuse', [])) != 2 or any(item['source'] not in sources for item in matrix['reuse']):
        raise ValueError('Figure 4 must contain exactly two valid H1 reuse references')
    return matrix


def training_argv(experiment, cli, include_process_limits=True):
    profile = experiment['profile']
    config = ROOT/'configs'/f'{profile}.json'
    values = [
        '--config', str(config), '--device', cli.device,
        '--data-root', str(cli.data_root), '--output-root', str(cli.work_root),
        '--dataset', experiment['dataset'], '--rate', str(experiment['rate']),
        '--epochs', str(cli.epochs),
    ]
    for key, value in experiment.get('overrides', {}).items():
        if key in BOOLEAN_FLAGS:
            if value:
                values.append(BOOLEAN_FLAGS[key])
        else:
            values.extend(['--' + key.replace('_', '-'), str(value)])
    if include_process_limits:
        values.extend(['--max-epochs-this-process', str(cli.chunk_epochs)])
    if getattr(cli, 'allow_hardware_change', False):
        values.append('--allow-hardware-change')
    if getattr(cli, 'gradient_checkpointing', False):
        values.append('--gradient-checkpointing')
    return values


def describe_experiments(matrix, cli):
    descriptions = []
    hashes = set()
    for experiment in matrix['experiments']:
        args = parse_training_args(training_argv(experiment, cli, False))
        config = resolved_dict(args)
        if config['experiment_sha256'] in hashes:
            raise ValueError(f'Duplicate semantic experiment: {experiment["id"]}')
        hashes.add(config['experiment_sha256'])
        descriptions.append({
            **experiment, 'run_id': make_run_id(config),
            'experiment_sha256': config['experiment_sha256'],
        })
    return descriptions


def read_status(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def current_source_hashes():
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted((ROOT/'src').glob('*.py'))
    }


def reset_incompatible_local_checkpoint(local_run, work_root):
    """Discard only a transient checkpoint whose recorded source is obsolete."""
    local_run = Path(local_run)
    checkpoint = local_run/'checkpoints'/'last.pt'
    if not checkpoint.exists():
        return False
    revision = read_status(local_run/'code_revision.json')
    if revision.get('source_sha256') == current_source_hashes():
        return False
    resolved_local = local_run.resolve()
    if resolved_local.parent != Path(work_root).resolve():
        raise ValueError(f'Refusing to reset run outside work root: {resolved_local}')
    shutil.rmtree(resolved_local)
    return True


def completed_and_valid(store_run, epochs, experiment_sha256=None):
    try:
        current = validate_snapshot(store_run)
        status = read_status(Path(store_run)/'status.json')
        summary = read_status(Path(store_run)/'summary.json')
        completed = read_status(Path(store_run)/'COMPLETED.json')
        config = read_status(Path(store_run)/'config.resolved.json')
        revision = read_status(Path(store_run)/'code_revision.json')
        return (
            current.get('status') == 'completed'
            and int(current['epoch_next']) == epochs
            and status.get('status') == 'completed'
            and int(summary.get('completed_epochs', -1)) == epochs
            and int(completed.get('epoch_next', -1)) == epochs
            and (experiment_sha256 is None
                 or config.get('experiment_sha256') == experiment_sha256)
            and revision.get('source_sha256') == current_source_hashes()
        )
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        return False


def acquire_lock(path, stale_minutes, force=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        age_minutes = (time.time() - path.stat().st_mtime) / 60
        if not force and age_minutes < stale_minutes:
            owner = path.read_text(encoding='utf-8', errors='replace')
            raise RuntimeError(f'Another runner lock is active ({age_minutes:.1f} min old): {owner}')
        path.unlink()
    owner = {
        'session_id': str(uuid.uuid4()), 'host': socket.gethostname(),
        'pid': os.getpid(), 'created_at_utc': utc_now(),
    }
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump(owner, stream, indent=2)
    return owner


def tee_process(command, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', errors='backslashreplace') as log:
        log.write(f'\n=== {utc_now()} ===\n$ {" ".join(command)}\n')
        log.flush()
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='backslashreplace', bufsize=1,
        )
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        return process.wait()


def save_matrix_status(path, entries, session):
    counts = {}
    for value in entries.values():
        status = value.get('status', 'planned')
        counts[status] = counts.get(status, 0) + 1
    write_json(path, {
        'schema_version': 1, 'updated_at_utc': utc_now(),
        'session': session, 'counts': counts, 'runs': entries,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=ROOT/'configs/experiments.json')
    parser.add_argument('--data-root', type=Path, default=Path('/content/mygram_data'))
    parser.add_argument('--work-root', type=Path, default=Path('/content/mygram_work/runs'))
    parser.add_argument('--store-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume valid local/Drive checkpoints (the safe default; retained for explicit notebook commands)',
    )
    parser.add_argument('--allow-hardware-change', action='store_true')
    parser.add_argument(
        '--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True,
        help='Use exact activation recomputation to fit full graphs in Colab VRAM',
    )
    parser.add_argument('--group', default='all', help='all or comma-separated group names')
    parser.add_argument('--run-id', action='append', dest='experiment_ids')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--chunk-epochs', type=int, default=5)
    parser.add_argument('--max-runs', type=int, default=1)
    parser.add_argument('--max-wall-minutes', type=float, default=600)
    parser.add_argument('--lock-stale-minutes', type=float, default=720)
    parser.add_argument('--force-stale-lock', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args(argv)
    if cli.epochs <= 0 or cli.chunk_epochs <= 0 or cli.max_runs <= 0:
        parser.error('epochs, chunk-epochs, and max-runs must be positive')

    matrix = load_matrix(cli.matrix)
    required_epochs = int(matrix['defaults']['epochs'])
    if cli.epochs != required_epochs:
        parser.error(
            f'This is the long-run runner: --epochs must remain {required_epochs}. '
            'Use scripts/run_colab_gates.py for reduced pilot budgets.'
        )
    descriptions = describe_experiments(matrix, cli)
    groups = None if cli.group == 'all' else set(cli.group.split(','))
    selected = [item for item in descriptions if groups is None or item['group'] in groups]
    if cli.experiment_ids:
        wanted = set(cli.experiment_ids)
        selected = [item for item in selected if item['id'] in wanted]
        missing = wanted - {item['id'] for item in selected}
        if missing:
            raise ValueError(f'Unknown or filtered experiment IDs: {sorted(missing)}')
    if cli.dry_run:
        print(json.dumps({
            'unique_long_runs': len(descriptions), 'selected': selected,
            'reuse': matrix['reuse'],
        }, indent=2, ensure_ascii=False))
        return 0

    cli.work_root.mkdir(parents=True, exist_ok=True)
    cli.store_root.mkdir(parents=True, exist_ok=True)
    lock_path = cli.store_root/'runner.lock'
    session = acquire_lock(lock_path, cli.lock_stale_minutes, cli.force_stale_lock)
    status_path = cli.store_root/'matrix_status.json'
    prior = read_status(status_path).get('runs', {})
    entries = {item['id']: prior.get(item['id'], {
        'status': 'planned', 'run_id': item['run_id'], 'group': item['group'],
    }) for item in descriptions}
    save_matrix_status(status_path, entries, session)

    started = time.monotonic()
    processed = 0
    try:
        for experiment in selected:
            if processed >= cli.max_runs:
                break
            run_id = experiment['run_id']
            local_run = cli.work_root/run_id
            store_run = cli.store_root/run_id
            if completed_and_valid(store_run, cli.epochs, experiment['experiment_sha256']):
                entries[experiment['id']].update({'status': 'completed', 'verified_at_utc': utc_now()})
                save_matrix_status(status_path, entries, session)
                continue
            processed += 1
            entries[experiment['id']].update({'status': 'running', 'started_at_utc': utc_now()})
            save_matrix_status(status_path, entries, session)
            try:
                if not local_run.exists() and (store_run/'CURRENT.json').exists():
                    restored = restore_run(store_run, local_run)
                    entries[experiment['id']]['restore'] = restored
                if reset_incompatible_local_checkpoint(local_run, cli.work_root):
                    entries[experiment['id']]['restart_reason'] = 'source_changed'
                    save_matrix_status(status_path, entries, session)
                checkpoint = local_run/'checkpoints'/'last.pt'
                while True:
                    elapsed_minutes = (time.monotonic() - started) / 60
                    if elapsed_minutes >= cli.max_wall_minutes:
                        if checkpoint.exists():
                            publish_run(local_run, store_run)
                        entries[experiment['id']].update({
                            'status': 'paused', 'reason': 'runner_max_wall_minutes',
                            'epoch_next': checkpoint_epoch_next(checkpoint) if checkpoint.exists() else 0,
                        })
                        save_matrix_status(status_path, entries, session)
                        return 0
                    if checkpoint.exists():
                        reconcile_run_to_checkpoint(local_run, checkpoint)
                    command = [sys.executable, '-m', 'src.run', *training_argv(experiment, cli)]
                    if checkpoint.exists():
                        command.extend(['--resume', str(checkpoint)])
                    return_code = tee_process(command, ROOT, local_run/'runner.log')
                    if checkpoint.exists():
                        snapshot = publish_run(local_run, store_run)
                        entries[experiment['id']]['epoch_next'] = snapshot['epoch_next']
                        entries[experiment['id']]['last_snapshot_utc'] = utc_now()
                        save_matrix_status(status_path, entries, session)
                    if return_code == 0 and completed_and_valid(
                            store_run, cli.epochs, experiment['experiment_sha256']):
                        entries[experiment['id']].update({'status': 'completed', 'completed_at_utc': utc_now()})
                        save_matrix_status(status_path, entries, session)
                        break
                    if return_code != 75:
                        raise RuntimeError(f'training subprocess failed with return code {return_code}')
            except Exception as exc:
                local_error = read_status(local_run/'status.json').get('error', '')
                error_text = (repr(exc) + ' ' + str(local_error)).lower()
                blocked = any(token in error_text for token in (
                    'out of memory', 'cuda oom', 'insufficient free space',
                    'no space left', 'disk quota',
                ))
                entries[experiment['id']].update({
                    'status': 'blocked' if blocked else 'failed',
                    'error': repr(exc), 'training_error': local_error or None,
                    'traceback': traceback.format_exc(), 'failed_at_utc': utc_now(),
                })
                save_matrix_status(status_path, entries, session)
                raise
    finally:
        if lock_path.exists():
            lock_path.unlink()
    save_matrix_status(status_path, entries, session)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

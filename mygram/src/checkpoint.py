"""Atomic MyGram run artifacts and deterministic checkpoint/resume."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch

from .data_registry import sha256, write_json


def ablation_name(config):
    names = []
    for key in ('image', 'relation', 'attribute'):
        if config.get(f'disable_{key}', False):
            names.append(f'no_{key}')
    if config.get('disable_mgd', False):
        names.append('no_mgd')
    if config.get('disable_gram', False):
        names.append('no_gram')
    return '+'.join(names) if names else 'full'


def make_run_id(config):
    required = ('dataset', 'rate', 'seed', 'profile')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'Run config lacks keys: {missing}')
    experiment_hash = config.get('experiment_sha256', config.get('config_sha256'))
    if not experiment_hash:
        raise ValueError('Run config lacks experiment_sha256/config_sha256')
    rate = f'{float(config["rate"]):g}'
    return (
        f'{config["dataset"]}-r{rate}-s{config["seed"]}-{config["profile"]}-'
        f'{ablation_name(config)}-{experiment_hash[:12]}'
    )


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    # load_checkpoint maps model/optimizer tensors directly to the requested
    # training device.  That map also affects serialized RNG byte tensors, while
    # both CPU and CUDA generator APIs require their state as CPU ByteTensors.
    torch.set_rng_state(state['torch_cpu'].detach().cpu())
    if state.get('torch_cuda') is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('Checkpoint contains CUDA RNG state but CUDA is unavailable')
        torch.cuda.set_rng_state_all([
            value.detach().cpu() for value in state['torch_cuda']
        ])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(handle)
    temp = Path(temp_name)
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_checkpoint(path, model, optimizer, scheduler, training_state,
                    resolved_config, artifact_hashes):
    payload = {
        'schema_version': 2,
        'experiment_sha256': resolved_config.get(
            'experiment_sha256', resolved_config['config_sha256']
        ),
        'runtime_sha256': resolved_config.get('runtime_sha256'),
        'resolved_config': resolved_config,
        'artifact_hashes': artifact_hashes,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'training_state': training_state,
        'rng_state': capture_rng_state(),
    }
    atomic_torch_save(payload, path)
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}


def load_checkpoint(path, model, optimizer, scheduler, resolved_config,
                    artifact_hashes, map_location='cpu', allow_hardware_change=False):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get('schema_version') != 2:
        raise ValueError('Unsupported checkpoint schema')
    expected = resolved_config.get(
        'experiment_sha256', resolved_config.get('config_sha256')
    )
    if payload.get('experiment_sha256') != expected:
        raise ValueError('Checkpoint experiment hash mismatch')
    checkpoint_hashes = payload.get('artifact_hashes')
    if checkpoint_hashes != artifact_hashes:
        checkpoint_comparable = dict(checkpoint_hashes or {})
        current_comparable = dict(artifact_hashes)
        checkpoint_runtime = checkpoint_comparable.pop('runtime_environment_sha256', None)
        current_runtime = current_comparable.pop('runtime_environment_sha256', None)
        runtime_only_change = (
            checkpoint_comparable == current_comparable
            and checkpoint_runtime != current_runtime
        )
        if not (allow_hardware_change and runtime_only_change):
            raise ValueError('Checkpoint data/code/split/runtime hash mismatch')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    if payload['scheduler'] is not None:
        if scheduler is None:
            raise ValueError('Checkpoint has scheduler state but current run has no scheduler')
        scheduler.load_state_dict(payload['scheduler'])
    restore_rng_state(payload['rng_state'])
    return payload['training_state']


def read_jsonl(path):
    """Read complete JSONL records and ignore a torn final append."""
    path = Path(path)
    if not path.exists():
        return []
    values = []
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return values


def write_jsonl_atomic(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(handle)
    temp = Path(temp_name)
    try:
        with temp.open('w', encoding='utf-8', newline='\n') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def checkpoint_epoch_next(path):
    """Read only training progress from a checkpoint for validation/supervision."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return int(payload['training_state']['epoch_next'])


def reconcile_run_to_checkpoint(run_dir, checkpoint_path=None):
    """Trim JSONL records newer than the last durable checkpoint."""
    run_dir = Path(run_dir)
    checkpoint_path = Path(checkpoint_path or run_dir/'checkpoints'/'last.pt')
    if not checkpoint_path.exists():
        return -1
    committed_epoch = checkpoint_epoch_next(checkpoint_path) - 1
    for filename in ('metrics.jsonl', 'events.jsonl'):
        path = run_dir/filename
        original = read_jsonl(path)
        eligible = [
            item for item in original
            if item.get('epoch') is None or int(item['epoch']) <= committed_epoch
        ]
        last_index = {}
        for index, item in enumerate(eligible):
            phase = item.get('phase')
            if phase in ('train', 'evaluation') and item.get('epoch') is not None:
                last_index[(phase, int(item['epoch']))] = index
        reconciled = [
            item for index, item in enumerate(eligible)
            if item.get('phase') not in ('train', 'evaluation')
            or item.get('epoch') is None
            or last_index[(item['phase'], int(item['epoch']))] == index
        ]
        if path.exists() and reconciled != original:
            write_jsonl_atomic(path, reconciled)
    for directory in (run_dir/'checkpoints', run_dir/'predictions'):
        if directory.exists():
            for temporary in directory.glob('*.tmp'):
                temporary.unlink(missing_ok=True)
    return committed_epoch


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(handle)
    temp = Path(temp_name)
    fields = [
        'direction', 'query_position', 'query_id', 'ground_truth_position',
        'ground_truth_id', 'rank_1based', 'top_candidate_positions',
        'top_candidate_ids', 'top_scores',
    ]
    try:
        with temp.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            for row in predictions:
                writer.writerow({
                    **row,
                    'top_candidate_positions': json.dumps(row['top_candidate_positions']),
                    'top_candidate_ids': json.dumps(row['top_candidate_ids']),
                    'top_scores': json.dumps(row['top_scores']),
                })
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _copy_exact(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise FileExistsError(f'Artifact conflict: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def initialize_run_directory(output_root, resolved_config, environment,
                             data_manifest, code_revision, split_path):
    run_id = make_run_id(resolved_config)
    run_dir = Path(output_root) / run_id
    status_path = run_dir/'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if status.get('status') == 'completed':
            raise FileExistsError(f'Completed run already exists: {run_dir}')
    for directory in ('checkpoints', 'predictions'):
        (run_dir/directory).mkdir(parents=True, exist_ok=True)
    write_json(run_dir/'config.resolved.json', resolved_config)
    write_json(run_dir/'environment.json', environment)
    write_json(run_dir/'data_manifest.json', data_manifest)
    write_json(run_dir/'code_revision.json', code_revision)
    _copy_exact(split_path, run_dir/'split.npz')
    for filename in ('metrics.jsonl', 'events.jsonl'):
        (run_dir/filename).touch(exist_ok=True)
    write_json(status_path, {'status': 'initialized', 'run_id': run_id})
    return run_id, run_dir

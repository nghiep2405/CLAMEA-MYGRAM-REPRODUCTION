"""Run an exact Stage G pilot with heartbeat logging and checkpoint resume."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import make_run_id
from src.config import parse_args, resolved_dict
from src.data_registry import write_json


def completed_epoch(run_dir):
    path = run_dir/'metrics.jsonl'
    if not path.exists():
        return -1
    last = -1
    for line in path.read_text(encoding='utf-8').splitlines():
        value = json.loads(line)
        if value.get('phase') == 'train':
            last = max(last, int(value['epoch']))
    return last


def status_value(run_dir):
    path = run_dir/'status.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8')).get('status')


def _read_jsonl(path):
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            # A power loss can leave only the final append incomplete.
            break
    return values


def checkpoint_epoch(run_dir):
    epochs = [
        int(value['epoch_next']) - 1
        for value in _read_jsonl(run_dir/'events.jsonl')
        if value.get('event') == 'checkpoint_saved'
        and value.get('kind') == 'last'
        and value.get('epoch_next') is not None
    ]
    return max(epochs, default=-1)


def _write_jsonl_atomic(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.supervisor.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
        stream.flush()
    temporary.replace(path)


def reconcile_to_checkpoint(run_dir):
    """Remove only uncommitted JSONL tail records before a checkpoint resume."""
    committed_epoch = checkpoint_epoch(run_dir)
    metrics_path = run_dir/'metrics.jsonl'
    metrics = _read_jsonl(metrics_path)
    eligible = [
        value for value in metrics
        if value.get('epoch') is None or int(value['epoch']) <= committed_epoch
    ]
    last_index = {}
    for index, value in enumerate(eligible):
        if value.get('phase') in ('train', 'evaluation'):
            last_index[(value['phase'], int(value['epoch']))] = index
    reconciled = [
        value for index, value in enumerate(eligible)
        if value.get('phase') not in ('train', 'evaluation')
        or last_index[(value['phase'], int(value['epoch']))] == index
    ]
    if metrics_path.exists() and reconciled != metrics:
        _write_jsonl_atomic(metrics_path, reconciled)
    for directory in (run_dir/'checkpoints', run_dir/'predictions'):
        if directory.exists():
            for temporary in directory.glob('*.tmp'):
                temporary.unlink(missing_ok=True)
    return committed_epoch


def main():
    try:
        sys.stdout.reconfigure(errors='backslashreplace')
        sys.stderr.reconfigure(errors='backslashreplace')
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('FBDB15K', 'zh_en'), required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/mmkg')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/stage_g_pilots')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--max-restarts', type=int, default=20)
    parser.add_argument('--max-epochs-per-process', type=int, default=5)
    parser.add_argument('--critical-available-mib', type=int, default=512)
    parser.add_argument('--report', type=Path)
    cli = parser.parse_args()
    rate = 0.2 if cli.dataset == 'FBDB15K' else 0.3
    canonical = parse_args([
        '--config', str(ROOT/'configs/repaired_pmf.json'), '--device', 'cpu',
        '--data-root', str(cli.data_root), '--output-root', str(cli.output_root),
        '--dataset', cli.dataset, '--rate', str(rate), '--epochs', str(cli.epochs),
        '--eval-every', '50',
    ])
    run_dir = cli.output_root.resolve()/make_run_id(resolved_dict(canonical))
    checkpoint = run_dir/'checkpoints'/'last.pt'
    log_path = run_dir/'pilot_supervisor.log'
    report_path = cli.report or ROOT/f'audit/stage_g_pilot_{cli.dataset}.json'
    base_command = [
        sys.executable, '-m', 'src.run', '--config', str(ROOT/'configs/repaired_pmf.json'),
        '--device', 'cpu', '--data-root', str(cli.data_root),
        '--output-root', str(cli.output_root), '--dataset', cli.dataset,
        '--rate', str(rate), '--epochs', str(cli.epochs), '--eval-every', '50',
    ]
    started = time.perf_counter()
    restarts = 0
    attempts = []
    while status_value(run_dir) != 'completed':
        if restarts > cli.max_restarts:
            raise RuntimeError(f'Pilot exceeded {cli.max_restarts} restart attempts')
        before = reconcile_to_checkpoint(run_dir)
        command = list(base_command)
        if checkpoint.exists():
            command.extend(['--resume', str(checkpoint)])
        run_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8', errors='backslashreplace') as log:
            log.write(f'\n=== attempt {restarts + 1} at {datetime.datetime.now().isoformat()} ===\n')
            log.flush()
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            termination_reason = None
            last_heartbeat = 0.0
            while True:
                try:
                    return_code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    current_checkpoint = checkpoint_epoch(run_dir)
                    available_mib = psutil.virtual_memory().available / (1024 ** 2)
                    now = time.monotonic()
                    if now - last_heartbeat >= 30:
                        print(
                            f'{cli.dataset}: heartbeat checkpoint_epoch={current_checkpoint} '
                            f'available_mib={available_mib:.0f} attempt={restarts + 1}',
                            flush=True,
                        )
                        last_heartbeat = now
                    if available_mib < cli.critical_available_mib:
                        termination_reason = 'critical_available_memory'
                    elif (cli.max_epochs_per_process > 0
                            and current_checkpoint - before >= cli.max_epochs_per_process):
                        termination_reason = 'planned_process_rotation'
                    if termination_reason is not None:
                        process.terminate()
                        try:
                            return_code = process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            return_code = process.wait()
                        break
        after = checkpoint_epoch(run_dir)
        attempts.append({
            'attempt': restarts + 1, 'return_code': return_code,
            'checkpoint_epoch_before': before, 'checkpoint_epoch_after': after,
            'termination_reason': termination_reason,
        })
        if return_code == 0 and status_value(run_dir) == 'completed':
            break
        if after <= before:
            raise RuntimeError(
                f'Pilot failed without checkpoint progress: rc={return_code}, epoch={after}'
            )
        restarts += 1
        print(
            f'{cli.dataset}: child stopped rc={return_code} reason={termination_reason}; '
            f'resuming from epoch {after + 1}',
            flush=True,
        )
    result = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'passed',
        'label': 'stage_g_pilot_not_research_result',
        'dataset': cli.dataset,
        'rate': rate,
        'epochs': cli.epochs,
        'device': 'cpu',
        'profile': 'repaired_pmf',
        'run_id': run_dir.name,
        'run_dir': str(run_dir),
        'checkpoint': str(checkpoint),
        'completed_epoch': completed_epoch(run_dir),
        'restart_count': restarts,
        'max_epochs_per_process': cli.max_epochs_per_process,
        'critical_available_mib': cli.critical_available_mib,
        'attempts': attempts,
        'runtime_seconds_supervised': time.perf_counter() - started,
        'log_path': str(log_path),
        'cuda': 'not_run; installed torch build is CPU-only',
    }
    write_json(report_path, result)
    print(f'{cli.dataset}: pilot passed; report: {report_path.resolve()}', flush=True)


if __name__ == '__main__':
    main()

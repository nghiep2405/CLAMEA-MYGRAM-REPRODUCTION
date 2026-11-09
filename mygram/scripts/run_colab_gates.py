"""Run mandatory MyGram CUDA gates with the same Drive snapshot contract as H."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiments import (
    completed_and_valid,
    read_status,
    reset_incompatible_local_checkpoint,
    tee_process,
    training_argv,
)
from src.artifact_store import publish_run, restore_run
from src.checkpoint import make_run_id, reconcile_run_to_checkpoint
from src.config import parse_args as parse_training_args, resolved_dict
from src.data_registry import write_json


GATES = {
    'g3': [
        {'id': f'g3_{dataset}', 'dataset': dataset, 'rate': .05, 'epochs': 1}
        for dataset in ('FBDB15K', 'FBYG15K', 'zh_en', 'ja_en', 'fr_en')
    ],
    'g4': [
        {'id': 'g4_fbdb_full_batch', 'dataset': 'FBDB15K', 'rate': .8, 'epochs': 1},
        {'id': 'g4_zh_full_batch', 'dataset': 'zh_en', 'rate': .3, 'epochs': 1},
    ],
    'g6': [
        {'id': 'g6_fbdb_r020', 'dataset': 'FBDB15K', 'rate': .2},
        {'id': 'g6_zh_r030', 'dataset': 'zh_en', 'rate': .3},
    ],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', choices=GATES, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/content/mygram_data'))
    parser.add_argument('--work-root', type=Path, default=Path('/content/mygram_work/gates'))
    parser.add_argument('--store-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-hardware-change', action='store_true')
    parser.add_argument(
        '--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument('--pilot-epochs', type=int, default=60)
    parser.add_argument('--chunk-epochs', type=int, default=5)
    args = parser.parse_args(argv)
    report = {
        'gate': args.gate,
        'status': 'running',
        'gradient_checkpointing': args.gradient_checkpointing,
        'events': [],
        'runs': [],
    }
    report_path = args.store_root/f'{args.gate}_report.json'
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.store_root.mkdir(parents=True, exist_ok=True)

    for raw in GATES[args.gate]:
        experiment = {
            **raw, 'profile': 'repaired_pmf', 'group': f'colab_{args.gate}',
        }
        epochs = raw.get('epochs', args.pilot_epochs)
        cli = SimpleNamespace(
            device=args.device, data_root=args.data_root, output_root=args.work_root,
            work_root=args.work_root, epochs=epochs, chunk_epochs=args.chunk_epochs,
            allow_hardware_change=args.allow_hardware_change,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        config_args = parse_training_args(training_argv(experiment, cli, False))
        config = resolved_dict(config_args)
        run_id = make_run_id(config)
        local_run, store_run = args.work_root/run_id, args.store_root/run_id
        if completed_and_valid(store_run, epochs, config['experiment_sha256']):
            report['runs'].append({'id': raw['id'], 'run_id': run_id, 'status': 'completed_reused'})
            continue
        if not local_run.exists() and (store_run/'CURRENT.json').exists():
            restore_run(store_run, local_run)

        # A bundle update changes the source identity.  Never feed a checkpoint
        # created by older source to the new trainer: src.checkpoint would reject
        # it after the expensive model/data initialization anyway.  Removing this
        # transient Colab copy leaves the old Drive snapshot intact for audit.
        if reset_incompatible_local_checkpoint(local_run, args.work_root):
            report['events'].append({
                'id': raw['id'], 'run_id': run_id,
                'status': 'restarted_after_source_change',
            })
            write_json(report_path, report)

        while not completed_and_valid(store_run, epochs, config['experiment_sha256']):
            checkpoint = local_run/'checkpoints'/'last.pt'
            if checkpoint.exists():
                reconcile_run_to_checkpoint(local_run, checkpoint)
            command = [sys.executable, '-m', 'src.run', *training_argv(experiment, cli)]
            if checkpoint.exists():
                command.extend(['--resume', str(checkpoint)])
            return_code = tee_process(command, ROOT, local_run/'gate_runner.log')
            if checkpoint.exists():
                publish_run(local_run, store_run)
            if return_code not in (0, 75):
                failed_status = read_status(local_run/'status.json')
                report['runs'].append({
                    'id': raw['id'], 'run_id': run_id, 'status': 'failed',
                    'return_code': return_code,
                    'training_error': failed_status.get('error'),
                    'local_log': str((local_run/'gate_runner.log').resolve()),
                })
                report['status'] = 'failed'
                write_json(report_path, report)
                return return_code
        report['runs'].append({'id': raw['id'], 'run_id': run_id, 'status': 'completed'})
        write_json(report_path, report)
    report['status'] = 'passed'
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

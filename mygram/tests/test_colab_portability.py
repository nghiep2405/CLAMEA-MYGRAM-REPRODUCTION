import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.artifact_store import publish_run, restore_run, validate_snapshot
from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import parse_args, resolved_dict
from scripts.run_experiments import reset_incompatible_local_checkpoint


class ColabPortabilityTests(unittest.TestCase):
    def test_obsolete_local_checkpoint_is_reset_but_drive_copy_is_untouched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local = root/'work'/'run-a'
            store = root/'store'/'run-a'
            (local/'checkpoints').mkdir(parents=True)
            (store/'checkpoints').mkdir(parents=True)
            (local/'checkpoints'/'last.pt').write_bytes(b'obsolete-local')
            (store/'checkpoints'/'last.pt').write_bytes(b'audit-copy')
            (local/'code_revision.json').write_text(
                json.dumps({'source_sha256': {'src/run.py': 'obsolete'}}),
                encoding='utf-8',
            )

            self.assertTrue(reset_incompatible_local_checkpoint(local, root/'work'))
            self.assertFalse(local.exists())
            self.assertEqual((store/'checkpoints'/'last.pt').read_bytes(), b'audit-copy')

    def test_experiment_hash_ignores_runtime_paths_device_and_chunking(self):
        first = resolved_dict(parse_args([
            '--device', 'cpu', '--data-root', 'data/mmkg', '--output-root', 'outputs/a',
            '--max-epochs-this-process', '1', '--max-wall-minutes', '2',
        ]))
        second = resolved_dict(parse_args([
            '--device', 'cuda:0', '--data-root', 'other/data', '--output-root', 'other/out',
            '--max-epochs-this-process', '10', '--max-wall-minutes', '30',
            '--gradient-checkpointing',
        ]))
        self.assertEqual(first['experiment_sha256'], second['experiment_sha256'])
        self.assertNotEqual(first['runtime_sha256'], second['runtime_sha256'])

    def test_experiment_hash_changes_for_semantic_options(self):
        baseline = resolved_dict(parse_args(['--device', 'cpu']))
        for values in (
            ['--rate', '0.5'], ['--seed', '2024'], ['--disable-gram'],
            ['--profile', 'author_code_compat'],
        ):
            changed = resolved_dict(parse_args(['--device', 'cpu', *values]))
            self.assertNotEqual(baseline['experiment_sha256'], changed['experiment_sha256'])

    def test_schema2_checkpoint_can_move_to_another_runtime_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = resolved_dict(parse_args([
                '--device', 'cpu', '--data-root', str(root/'data-a'),
                '--output-root', str(root/'out-a'),
            ]))
            second = resolved_dict(parse_args([
                '--device', 'cpu', '--data-root', str(root/'data-b'),
                '--output-root', str(root/'out-b'),
            ]))
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            expected = copy.deepcopy(model.state_dict())
            checkpoint = root/'last.pt'
            hashes = {'data': 'same', 'split': 'same', 'source': 'same'}
            save_checkpoint(checkpoint, model, optimizer, None, {'epoch_next': 3}, first, hashes)
            with torch.no_grad():
                model.weight.add_(9)
            state = load_checkpoint(checkpoint, model, optimizer, None, second, hashes)
            self.assertEqual(state['epoch_next'], 3)
            for key in expected:
                self.assertTrue(torch.equal(model.state_dict()[key], expected[key]))

    @staticmethod
    def _write_minimal_run(run_dir, epoch_next):
        (run_dir/'checkpoints').mkdir(parents=True, exist_ok=True)
        torch.save({'training_state': {'epoch_next': epoch_next}}, run_dir/'checkpoints'/'last.pt')
        (run_dir/'metrics.jsonl').write_text(
            ''.join(json.dumps({'phase': 'train', 'epoch': i}) + '\n' for i in range(epoch_next)),
            encoding='utf-8',
        )
        (run_dir/'events.jsonl').write_text('', encoding='utf-8')
        (run_dir/'status.json').write_text(
            json.dumps({'status': 'paused', 'epoch_next': epoch_next}), encoding='utf-8'
        )

    def test_publish_restore_and_previous_checkpoint_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local, store, restored = root/'local'/'run-a', root/'store'/'run-a', root/'restored'/'run-a'
            self._write_minimal_run(local, 1)
            publish_run(local, store)
            self._write_minimal_run(local, 2)
            publish_run(local, store)
            self.assertEqual(validate_snapshot(store)['epoch_next'], 2)
            (store/'checkpoints'/'last.pt').write_bytes(b'torn')
            result = restore_run(store, restored)
            self.assertTrue(result['recovered_previous'])
            self.assertEqual(result['epoch_next'], 1)
            metrics = [json.loads(line) for line in (restored/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([item['epoch'] for item in metrics], [0])


if __name__ == '__main__':
    unittest.main()

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from src.checkpoint import (
    initialize_run_directory,
    load_checkpoint,
    make_run_id,
    restore_rng_state,
    save_checkpoint,
    write_predictions,
)
from src.config import parse_args, resolved_dict
from src.evaluation import evaluate_embeddings
from src.iterative_learning import (
    IterativeLearningState,
    diagnostic_precision,
    mutual_nearest_pairs,
    ordered_il_state,
    update_il_state,
)
from src.run import apply_optimizer_policy


def dense_metrics(embeddings, left_ids, right_ids, csls_k, top_k=(1, 10, 50)):
    normalized = F.normalize(embeddings, dim=1)
    left = normalized[torch.tensor(left_ids)]
    right = normalized[torch.tensor(right_ids)]
    distance = (
        left.square().sum(1, keepdim=True) + right.square().sum(1)[None, :]
        - 2 * left @ right.T
    ).clamp_min(0)
    scores = 1 - distance
    if csls_k is not None:
        k = min(csls_k, len(left_ids))
        scores = (
            2 * scores
            - torch.topk(scores, k, dim=1).values.mean(1)[:, None]
            - torch.topk(scores.T, k, dim=1).values.mean(1)[None, :]
        )

    def direction(matrix):
        positions = torch.arange(matrix.shape[0])
        ground_truth = matrix[positions, positions]
        ranks = 1 + (matrix > ground_truth[:, None]).sum(1)
        ranks += ((matrix == ground_truth[:, None]) &
                  (positions[None, :] < positions[:, None])).sum(1)
        metrics = {f'hits@{k}': float((ranks <= k).float().mean()) for k in top_k}
        metrics.update({'mr': float(ranks.float().mean()),
                        'mrr': float(ranks.float().reciprocal().mean()),
                        'count': len(ranks)})
        return metrics, ranks

    l2r, lrank = direction(scores)
    r2l, rrank = direction(scores.T)
    return {'l2r': l2r, 'r2l': r2l}, lrank, rrank


def assert_nested_equal(test, left, right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            assert_nested_equal(test, first, second)
    else:
        test.assertEqual(left, right)


class EvaluationTests(unittest.TestCase):
    def test_block_evaluator_matches_dense_with_and_without_csls(self):
        torch.manual_seed(101)
        embeddings = torch.randn(12, 7)
        left_ids = [0, 1, 2, 3, 4]
        right_ids = [6, 7, 7, 9, 10]
        for csls_k in (None, 3):
            expected, lrank, rrank = dense_metrics(embeddings, left_ids, right_ids, csls_k)
            for block_size in (1, 2, 4):
                actual = evaluate_embeddings(
                    embeddings, left_ids, right_ids, csls_k=csls_k,
                    block_size=block_size,
                )
                self.assertEqual(actual['metrics'], expected)
                self.assertTrue(torch.equal(actual['ranks']['l2r'], lrank))
                self.assertTrue(torch.equal(actual['ranks']['r2l'], rrank))
                self.assertEqual(actual['protocol']['duplicate_target_ids'], {'7': 2})
                expected_scope = 'disabled' if csls_k is None else 'entire_candidate_set_two_pass'
                self.assertEqual(actual['protocol']['csls_scope'], expected_scope)

    def test_tie_rule_and_duplicate_candidate_positions(self):
        embeddings = torch.ones(6, 4)
        result = evaluate_embeddings(
            embeddings, [0, 1, 2], [3, 3, 5], csls_k=2, block_size=1
        )
        expected = torch.tensor([1, 2, 3])
        self.assertTrue(torch.equal(result['ranks']['l2r'], expected))
        self.assertTrue(torch.equal(result['ranks']['r2l'], expected))
        self.assertEqual(result['protocol']['duplicate_target_ids'], {'3': 2})
        self.assertEqual(result['predictions']['l2r'][1]['ground_truth_position'], 1)


class IterativeLearningTests(unittest.TestCase):
    def test_mutual_nearest_is_blockwise_ordered_and_handles_empty(self):
        embeddings = torch.tensor([
            [1., 0.], [0., 1.], [-1., 0.],
            [.9, .1], [.1, .9], [-.9, -.1],
        ])
        expected = [(0, 3), (1, 4), (2, 5)]
        for block_size in (1, 2, 8):
            pairs, diagnostics = mutual_nearest_pairs(
                embeddings, [0, 1, 2], [3, 4, 5], block_size
            )
            self.assertEqual(pairs, expected)
            self.assertFalse(diagnostics['empty_pool'])
        pairs, diagnostics = mutual_nearest_pairs(embeddings, [], [3], 2)
        self.assertEqual(pairs, [])
        self.assertTrue(diagnostics['empty_pool'])

    def test_schedule_intersection_addition_and_fixed_test_preservation(self):
        state = ordered_il_state(
            [(0, 10)], [(1, 11), (2, 12)], [0, 1, 2, 3], [10, 11, 12, 13]
        )
        fixed = copy.deepcopy(state.fixed_test_links)
        event = update_il_state(state, [(1, 11), (2, 12)], 0, 0, 1)
        self.assertEqual(event['action'], 'candidate_start')
        self.assertEqual(state.new_links, [(1, 11), (2, 12)])
        update_il_state(state, [(2, 12), (3, 13)], 1, 0, 1)
        self.assertEqual(state.new_links, [(2, 12)])
        for epoch in range(2, 9):
            update_il_state(state, [(2, 12)], epoch, 0, 1)
        event = update_il_state(state, [(2, 12)], 9, 0, 1)
        self.assertEqual(event['action'], 'pairs_added')
        self.assertEqual(event['added_pairs'], 1)
        self.assertIn((2, 12), state.train_links)
        self.assertNotIn(2, state.left_pool)
        self.assertNotIn(12, state.right_pool)
        self.assertEqual(state.fixed_test_links, fixed)
        diagnostic = diagnostic_precision([(2, 12)], state.fixed_test_links)
        self.assertFalse(diagnostic['used_for_selection'])

    def test_mid_cycle_state_resume_replays_the_same_addition(self):
        state = ordered_il_state(
            [(0, 10)], [(1, 11)], [0, 1, 2], [10, 11, 12]
        )
        candidates = [(1, 11), (2, 12)]
        for epoch in range(5):
            update_il_state(state, candidates, epoch, 0, 1)
        resumed = IterativeLearningState.from_dict(copy.deepcopy(state.to_dict()))
        for epoch in range(5, 10):
            update_il_state(state, candidates, epoch, 0, 1)
            update_il_state(resumed, candidates, epoch, 0, 1)
        self.assertEqual(resumed, state)
        self.assertEqual(state.train_links, [(0, 10), (1, 11), (2, 12)])
        self.assertEqual(state.new_links, [])


class CheckpointTests(unittest.TestCase):
    def _resolved(self, root, profile='repaired_pmf'):
        args = parse_args([
            '--profile', profile, '--device', 'cpu', '--output-root', str(root),
        ])
        return args, resolved_dict(args)

    def test_rng_restore_moves_device_mapped_byte_states_back_to_cpu(self):
        cpu_serialized = mock.Mock()
        cuda_serialized = mock.Mock()
        expected_cpu = torch.get_rng_state()
        expected_cuda = torch.arange(8, dtype=torch.uint8)
        cpu_serialized.detach.return_value.cpu.return_value = expected_cpu
        cuda_serialized.detach.return_value.cpu.return_value = expected_cuda
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': cpu_serialized,
            'torch_cuda': [cuda_serialized],
        }

        with mock.patch('src.checkpoint.torch.cuda.is_available', return_value=True), \
                mock.patch('src.checkpoint.torch.cuda.set_rng_state_all') as set_cuda:
            restore_rng_state(state)

        cpu_serialized.detach.return_value.cpu.assert_called_once_with()
        cuda_serialized.detach.return_value.cpu.assert_called_once_with()
        restored_cuda = set_cuda.call_args.args[0]
        self.assertEqual(len(restored_cuda), 1)
        self.assertTrue(torch.equal(restored_cuda[0], expected_cuda))

    def test_atomic_checkpoint_restores_rng_state_and_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args, config = self._resolved(root)
            random.seed(9)
            np.random.seed(9)
            torch.manual_seed(9)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            before = copy.deepcopy(model.state_dict())
            path = root/'last.pt'
            hashes = {'code': 'abc', 'data': 'def', 'split': 'ghi'}
            training_state = {'epoch_next': 50, 'new_links': [(1, 4)]}
            save_checkpoint(path, model, optimizer, None, training_state, config, hashes)
            expected_random = (random.random(), float(np.random.rand()), torch.rand(3))
            with torch.no_grad():
                model.weight.add_(10)
            loaded = load_checkpoint(path, model, optimizer, None, config, hashes)
            self.assertEqual(loaded, training_state)
            assert_nested_equal(self, model.state_dict(), before)
            actual_random = (random.random(), float(np.random.rand()), torch.rand(3))
            self.assertEqual(actual_random[0], expected_random[0])
            self.assertEqual(actual_random[1], expected_random[1])
            self.assertTrue(torch.equal(actual_random[2], expected_random[2]))
            self.assertFalse(list(root.glob('*.tmp')))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_checkpoint(path, model, optimizer, None, config, {**hashes, 'split': 'wrong'})

    def test_continuous_and_resume_match_at_a_normal_epoch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args, config = self._resolved(root)
            random.seed(23)
            np.random.seed(23)
            torch.manual_seed(23)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

            def step(current_model, current_optimizer):
                current_optimizer.zero_grad()
                inputs = torch.randn(3, 2)
                scale = random.random() + float(np.random.rand())
                (current_model(inputs).square().mean() * scale).backward()
                current_optimizer.step()

            for _ in range(3):
                step(model, optimizer)
            checkpoint = root/'normal.pt'
            hashes = {'code': 'same', 'data': 'same', 'split': 'same'}
            save_checkpoint(
                checkpoint, model, optimizer, None, {'epoch_next': 3}, config, hashes
            )
            for _ in range(2):
                step(model, optimizer)
            expected_model = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())

            resumed_model = torch.nn.Linear(2, 2)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=args.lr)
            state = load_checkpoint(
                checkpoint, resumed_model, resumed_optimizer, None, config, hashes
            )
            self.assertEqual(state['epoch_next'], 3)
            for _ in range(2):
                step(resumed_model, resumed_optimizer)
            assert_nested_equal(self, resumed_model.state_dict(), expected_model)
            assert_nested_equal(self, resumed_optimizer.state_dict(), expected_optimizer)

    def test_continuous_and_resume_match_across_lr_boundary_for_both_profiles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for profile in ('repaired_pmf', 'author_code_compat'):
                args, config = self._resolved(root/profile, profile)
                random.seed(13)
                np.random.seed(13)
                torch.manual_seed(13)
                model = torch.nn.Linear(2, 2)
                params = [{'params': list(model.parameters())}]
                optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

                def step(epoch, current_model, current_optimizer, current_params):
                    current_optimizer, _ = apply_optimizer_policy(
                        current_optimizer, current_params, args, epoch
                    )
                    current_optimizer.zero_grad()
                    inputs = torch.randn(4, 2)
                    scale = random.random() + float(np.random.rand())
                    (current_model(inputs).square().mean() * scale).backward()
                    current_optimizer.step()
                    return current_optimizer

                for epoch in range(50):
                    optimizer = step(epoch, model, optimizer, params)
                checkpoint = root/f'{profile}.pt'
                hashes = {'code': 'same', 'data': 'same', 'split': 'same'}
                il_state = IterativeLearningState(
                    [(0, 2)], [(1, 3)], [1], [3], [(1, 3)]
                ).to_dict()
                save_checkpoint(
                    checkpoint, model, optimizer, None,
                    {'epoch_next': 50, 'il_state': il_state}, config, hashes,
                )
                for epoch in range(50, 53):
                    optimizer = step(epoch, model, optimizer, params)
                continuous_model = copy.deepcopy(model.state_dict())
                continuous_optimizer = copy.deepcopy(optimizer.state_dict())

                resumed_model = torch.nn.Linear(2, 2)
                resumed_params = [{'params': list(resumed_model.parameters())}]
                resumed_optimizer = torch.optim.AdamW(
                    resumed_params, lr=args.lr, weight_decay=args.weight_decay
                )
                state = load_checkpoint(
                    checkpoint, resumed_model, resumed_optimizer, None,
                    config, hashes,
                )
                self.assertEqual(state['epoch_next'], 50)
                self.assertEqual(state['il_state']['new_links'], [(1, 3)])
                for epoch in range(50, 53):
                    resumed_optimizer = step(
                        epoch, resumed_model, resumed_optimizer, resumed_params
                    )
                assert_nested_equal(self, resumed_model.state_dict(), continuous_model)
                assert_nested_equal(self, resumed_optimizer.state_dict(), continuous_optimizer)

    def test_run_directory_contract_and_completed_collision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, config = self._resolved(root)
            split = root/'source_split.npz'
            np.savez(split, train=np.array([[0, 1]]))
            run_id, run_dir = initialize_run_directory(
                root/'outputs', config, {'python': 'test'}, {'archive': 'test'},
                {'commit': 'test'}, split,
            )
            self.assertEqual(run_id, make_run_id(config))
            expected = {
                'config.resolved.json', 'environment.json', 'data_manifest.json',
                'code_revision.json', 'split.npz', 'metrics.jsonl', 'events.jsonl',
                'status.json', 'checkpoints', 'predictions',
            }
            self.assertEqual({path.name for path in run_dir.iterdir()}, expected)
            prediction = {
                'direction': 'left_to_right', 'query_position': 0, 'query_id': 1,
                'ground_truth_position': 0, 'ground_truth_id': 2,
                'rank_1based': 1, 'top_candidate_positions': [0],
                'top_candidate_ids': [2], 'top_scores': [1.0],
            }
            write_predictions(run_dir/'predictions'/'last_l2r.tsv', [prediction])
            text = (run_dir/'predictions'/'last_l2r.tsv').read_text(encoding='utf-8')
            self.assertIn('left_to_right', text)
            self.assertFalse(list((run_dir/'predictions').glob('*.tmp')))
            (run_dir/'status.json').write_text(
                json.dumps({'status': 'completed'}), encoding='utf-8'
            )
            with self.assertRaisesRegex(FileExistsError, 'Completed run'):
                initialize_run_directory(
                    root/'outputs', config, {}, {}, {}, split,
                )


if __name__ == '__main__':
    unittest.main()

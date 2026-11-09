import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.run import MyGram
from src.config import parse_args, resolved_dict
from src.graph import build_graph_inputs
from src.iterative_learning import (
    IterativeLearningState,
    diagnostic_precision,
    mutual_nearest_pairs,
    ordered_il_state,
    update_il_state,
)
from src.loss import GRAM_Loss, InfoNCE_loss, alignment_objective
from src.models import MultiModalEncoderMrFusion


ROOT = Path(__file__).resolve().parents[1]
TRIPLES = [
    (0, 0, 1), (0, 1, 1), (1, 2, 0), (1, 1, 2),
    (2, 0, 3), (3, 2, 4), (4, 1, 5), (5, 0, 0),
]
LINKS = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.long)


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        if left.layout == torch.sparse_coo or right.layout == torch.sparse_coo:
            if left.layout != right.layout or left.shape != right.shape:
                raise AssertionError('Sparse tensor layout/shape differs after resume')
            left_coalesced = left.coalesce()
            right_coalesced = right.coalesce()
            if (not torch.equal(left_coalesced.indices(), right_coalesced.indices())
                    or not torch.equal(left_coalesced.values(), right_coalesced.values())):
                raise AssertionError('Sparse tensor state differs after resume')
        elif not torch.equal(left, right):
            raise AssertionError('Tensor state differs after resume')
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError('State keys differ after resume')
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError('State sequence lengths differ after resume')
        for first, second in zip(left, right):
            _assert_nested_equal(first, second)
    elif left != right:
        raise AssertionError('Scalar state differs after resume')


def _tiny_components(args):
    graph = build_graph_inputs(8, TRIPLES, mcd_self_loops=args.mcd_self_loops)
    model = MultiModalEncoderMrFusion(
        args=args,
        ent_num=8,
        rel_size=graph.relation_size,
        triple_size=graph.triple_size,
        img_feature_dim=12,
        adj_matrix=graph.edge_index,
        r_index=graph.relation_index,
        r_val=graph.relation_value,
        rel_matrix=torch.empty((2, 0), dtype=torch.long),
        ent_matrix=graph.entity_average,
        mcd_operator=graph.mcd_operator,
        left_ents=[0, 1, 2, 3],
        right_ents=[4, 5, 6, 7],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed + 91)
    features = (
        torch.randn(8, 12, generator=generator),
        torch.randn(8, args.bow_size, generator=generator),
        torch.randn(8, args.bow_size, generator=generator),
    )
    return model, optimizer, features


def _tiny_step(model, optimizer, features, args):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.arange(8), *features)
    loss, diagnostics = alignment_objective(
        outputs[-1], outputs[0], outputs[2], outputs[4], outputs[3], LINKS,
        InfoNCE_loss(temperature=args.nce_temperature),
        GRAM_Loss(
            temperature=args.gram_temperature,
            K=args.gram_k,
            direction_scale=args.gram_direction_scale,
            gram_modalities=args.gram_modalities,
        ),
        args.gram_weight,
    )
    if loss is None or not torch.isfinite(loss):
        raise AssertionError('Tiny training loss is not finite')
    loss.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise AssertionError('Tiny training gradients are missing or nonfinite')
    optimizer.step()
    return float(loss.detach()), diagnostics, len(gradients)


def tiny_training_resume_smoke():
    args = parse_args([
        '--config', str(ROOT/'configs/stage_g_smoke.json'), '--device', 'cpu'
    ])
    config = resolved_dict(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, optimizer, features = _tiny_components(args)
    initial = model.img_fc.weight.detach().clone()
    first_loss, first_diagnostics, first_gradients = _tiny_step(
        model, optimizer, features, args
    )
    if torch.equal(initial, model.img_fc.weight):
        raise AssertionError('Tiny optimizer step did not update img_fc')

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir)/'mid.pt'
        hashes = {'fixture': 'stage_g_tiny_v1'}
        il_state = ordered_il_state(
            LINKS.tolist(), [(0, 4)], [0, 1, 2, 3], [4, 5, 6, 7]
        ).to_dict()
        save_checkpoint(
            checkpoint, model, optimizer, None,
            {'epoch_next': 1, 'il_state': il_state}, config, hashes,
        )
        second_loss, second_diagnostics, second_gradients = _tiny_step(
            model, optimizer, features, args
        )
        continuous_model = copy.deepcopy(model.state_dict())
        continuous_optimizer = copy.deepcopy(optimizer.state_dict())

        resumed_model, resumed_optimizer, resumed_features = _tiny_components(args)
        state = load_checkpoint(
            checkpoint, resumed_model, resumed_optimizer, None, config, hashes
        )
        resumed_loss, resumed_diagnostics, resumed_gradients = _tiny_step(
            resumed_model, resumed_optimizer, resumed_features, args
        )
        _assert_nested_equal(resumed_model.state_dict(), continuous_model)
        _assert_nested_equal(resumed_optimizer.state_dict(), continuous_optimizer)

    if second_loss != resumed_loss or second_diagnostics != resumed_diagnostics:
        raise AssertionError('Tiny resumed loss/diagnostics differ from continuous run')
    return {
        'status': 'passed',
        'device': 'cpu',
        'epochs': 2,
        'losses': [first_loss, second_loss],
        'parameters_with_gradient': [first_gradients, second_gradients],
        'parameter_updated': True,
        'checkpoint_epoch_next': state['epoch_next'],
        'model_state_bitwise_equal': True,
        'optimizer_state_bitwise_equal': True,
        'resumed_gradients': resumed_gradients,
    }


def il_cycle_resume_smoke():
    embeddings = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
        [0.98, 0.02], [0.02, 0.98], [-0.98, -0.02],
    ])
    state = ordered_il_state(
        [], [(0, 3), (1, 4), (2, 5)], [0, 1, 2], [3, 4, 5]
    )
    events = []
    diagnostics = []
    for epoch in range(5):
        pairs, mining = mutual_nearest_pairs(
            embeddings, state.left_pool, state.right_pool, block_size=2
        )
        label_diagnostic = diagnostic_precision(pairs, state.fixed_test_links)
        events.append(update_il_state(state, pairs, epoch, 0, 1))
        diagnostics.append({'mining': mining, 'labels': label_diagnostic})

    restored = IterativeLearningState.from_dict(copy.deepcopy(state.to_dict()))
    for epoch in range(5, 10):
        pairs, mining = mutual_nearest_pairs(
            embeddings, state.left_pool, state.right_pool, block_size=2
        )
        restored_pairs, _ = mutual_nearest_pairs(
            embeddings, restored.left_pool, restored.right_pool, block_size=2
        )
        if pairs != restored_pairs:
            raise AssertionError('IL mining changed after mid-cycle resume')
        events.append(update_il_state(state, pairs, epoch, 0, 1))
        update_il_state(restored, restored_pairs, epoch, 0, 1)
        diagnostics.append({
            'mining': mining,
            'labels': diagnostic_precision(pairs, state.fixed_test_links),
        })
    if state != restored:
        raise AssertionError('IL state differs after mid-cycle resume')
    if state.train_links != [(0, 3), (1, 4), (2, 5)]:
        raise AssertionError('IL smoke did not add the mined mutual pairs')
    if state.fixed_test_links != [(0, 3), (1, 4), (2, 5)]:
        raise AssertionError('IL smoke changed the fixed test set')
    return {
        'status': 'passed',
        'label': 'smoke_fixture_not_research_metrics',
        'schedule': {'il_start': 0, 'il_step': 1, 'checks_per_cycle': 10},
        'actions': [event['action'] for event in events],
        'added_pairs': events[-1]['added_pairs'],
        'train_links': state.train_links,
        'fixed_test_links_unchanged': True,
        'ordered_pools_after_add': {
            'left': state.left_pool, 'right': state.right_pool,
        },
        'mid_cycle_resume_equal': True,
        'labels_used_for_selection': any(
            item['labels']['used_for_selection'] for item in diagnostics
        ),
    }


class StageGLayerTests(unittest.TestCase):
    def test_activation_checkpoint_outputs_and_gradients_match_direct_encoder(self):
        args = parse_args([
            '--config', str(ROOT/'configs/stage_g_smoke.json'), '--device', 'cpu'
        ])
        torch.manual_seed(707)
        direct, _, direct_features = _tiny_components(args)
        checkpointed, _, checkpointed_features = _tiny_components(args)
        checkpointed.load_state_dict(copy.deepcopy(direct.state_dict()))
        direct.train()
        checkpointed.train()
        entity_ids = torch.arange(8)

        torch.manual_seed(919)
        direct_outputs = direct(entity_ids, *direct_features)
        direct_loss = sum(
            value.square().mean() for value in direct_outputs if value is not None
        )
        direct_loss.backward()

        runner = MyGram.__new__(MyGram)
        runner.args = copy.copy(args)
        runner.args.gradient_checkpointing = True
        runner.multimodal_encoder = checkpointed
        runner.input_idx = entity_ids
        (
            runner.img_features,
            runner.rel_features,
            runner.att_features,
        ) = checkpointed_features
        runner.name_features = None
        runner.char_features = None

        torch.manual_seed(919)
        checkpointed_outputs = runner.forward_encoder()
        checkpointed_loss = sum(
            value.square().mean() for value in checkpointed_outputs if value is not None
        )
        checkpointed_loss.backward()

        torch.testing.assert_close(direct_loss, checkpointed_loss, atol=1e-6, rtol=1e-5)
        for direct_value, checkpointed_value in zip(direct_outputs, checkpointed_outputs):
            if direct_value is None:
                self.assertIsNone(checkpointed_value)
            else:
                torch.testing.assert_close(
                    direct_value, checkpointed_value, atol=1e-6, rtol=1e-5
                )
        direct_gradients = dict(direct.named_parameters())
        checkpointed_gradients = dict(checkpointed.named_parameters())
        self.assertEqual(set(direct_gradients), set(checkpointed_gradients))
        for name, parameter in direct_gradients.items():
            other = checkpointed_gradients[name]
            if parameter.grad is None:
                self.assertIsNone(other.grad, name)
            else:
                torch.testing.assert_close(
                    parameter.grad, other.grad, atol=1e-6, rtol=1e-5,
                    msg=lambda message, name=name: f'{name}: {message}',
                )

    def test_g2_tiny_training_and_resume(self):
        result = tiny_training_resume_smoke()
        self.assertEqual(result['status'], 'passed')

    def test_g5_shortened_il_cycle_and_resume(self):
        result = il_cycle_resume_smoke()
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['labels_used_for_selection'])


if __name__ == '__main__':
    unittest.main()

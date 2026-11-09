import copy
import unittest

import torch
import torch.nn.functional as F

from src.config import parse_args
from src.loss import (
    GRAM_Loss,
    InfoNCE_loss,
    alignment_objective,
    sparse_gram_loss,
)
from src.run import apply_optimizer_policy


ATOL = 1e-6
RTOL = 1e-5


def legacy_full_gram(struct, image, attribute, relation, links, K, temperature, scale):
    left, right = links[:, 0], links[:, 1]
    structure = F.normalize(struct[left], dim=1)
    visual = F.normalize(image[right], dim=1)
    attribute = F.normalize(attribute[right], dim=1)
    relation = F.normalize(relation[right], dim=1)
    similarity = structure @ visual.T
    _, topk_idx = torch.topk(similarity, K, dim=1)
    cur_structure = structure.unsqueeze(1).expand(-1, K, -1)
    all_modals = torch.stack([
        cur_structure, visual[topk_idx], attribute[topk_idx], relation[topk_idx]
    ], dim=2)
    gram = torch.einsum('bkif,bkjf->bkij', all_modals, all_modals)
    volumes = torch.sqrt(torch.clamp(torch.abs(torch.linalg.det(gram)), min=1e-8))
    targets = torch.arange(links.shape[0])
    mask = topk_idx == targets[:, None]
    valid_rows = mask.any(1).nonzero(as_tuple=False).squeeze(-1)
    valid_volumes = volumes[valid_rows] / temperature
    positive = mask[valid_rows].float().argmax(1)
    raw = -F.log_softmax(-valid_volumes, dim=1)[
        torch.arange(valid_rows.numel()), positive
    ].mean()
    return scale * raw


class LossOptimizerAblationTests(unittest.TestCase):
    def test_full_four_vector_gram_matches_published_reference_and_gradient(self):
        torch.manual_seed(31)
        links = torch.tensor([[0, 4], [1, 5], [2, 5], [3, 7]])
        actual_inputs = [torch.randn(8, 6, requires_grad=True) for _ in range(4)]
        reference_inputs = [value.detach().clone().requires_grad_(True) for value in actual_inputs]
        gram = GRAM_Loss(temperature=.05, K=10, direction_scale=.5)
        actual = gram(*actual_inputs, links)
        expected = legacy_full_gram(*reference_inputs, links, 4, .05, .5)
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        actual.backward()
        expected.backward()
        for value, reference in zip(actual_inputs, reference_inputs):
            torch.testing.assert_close(value.grad, reference.grad, atol=ATOL, rtol=RTOL)
        self.assertEqual(gram.last_diagnostics['K_eff'], 4)
        self.assertEqual(gram.last_diagnostics['gram_matrix_size'], 4)
        self.assertEqual(gram.last_diagnostics['gram_candidate_modality'], 'image')

    def test_b_smaller_than_k_duplicate_targets_and_near_singular_are_finite(self):
        links = torch.tensor([[0, 3], [1, 4], [2, 4]])
        base = torch.randn(6, 5)
        values = [
            (base + index * 1e-6).detach().clone().requires_grad_(True)
            for index in range(4)
        ]
        gram = GRAM_Loss(temperature=.05, K=10, direction_scale=.5)
        loss = gram(*values, links)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(gram.last_diagnostics['K_eff'], 3)
        loss.backward()
        for value in values:
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_zero_valid_positive_returns_graph_connected_zero(self):
        indices = torch.tensor([[1], [0]])
        volumes = torch.tensor([[.2], [.4]], requires_grad=True)
        loss, diagnostics = sparse_gram_loss(
            indices, volumes, torch.tensor([0, 1]), return_diagnostics=True
        )
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(diagnostics['gram_valid_rows'], 0)
        self.assertTrue(diagnostics['gram_zero_positive_batch'])
        loss.backward()
        self.assertTrue(torch.equal(volumes.grad, torch.zeros_like(volumes)))

    def test_partial_valid_rows_and_singleton_are_finite(self):
        volumes = torch.tensor([[.1, .2], [.3, .4]], requires_grad=True)
        loss, diagnostics = sparse_gram_loss(
            torch.tensor([[0, 1], [0, 0]]), volumes, torch.tensor([0, 1]),
            return_diagnostics=True,
        )
        expected = -F.log_softmax(-volumes[:1] / .05, dim=1)[0, 0]
        torch.testing.assert_close(loss, expected)
        self.assertEqual(diagnostics['gram_valid_rows'], 1)
        singleton = torch.tensor([[.7]], requires_grad=True)
        one, one_diag = sparse_gram_loss(
            torch.tensor([[0]]), singleton, torch.tensor([0]), return_diagnostics=True
        )
        self.assertTrue(torch.isfinite(one))
        self.assertFalse(one_diag['gram_has_useful_negatives'])
        one.backward()
        self.assertTrue(torch.isfinite(singleton.grad).all())
        singleton_embedding = torch.randn(2, 4, requires_grad=True)
        nce = InfoNCE_loss(temperature=.1)
        nce_one = nce(singleton_embedding, torch.tensor([[0, 1]]))
        self.assertTrue(torch.isfinite(nce_one))
        self.assertFalse(nce.last_diagnostics['nce_has_useful_negatives'])
        nce_one.backward()
        self.assertTrue(torch.isfinite(singleton_embedding.grad).all())

    def test_zero_valid_gram_does_not_block_nce_backward(self):
        structure = torch.tensor([
            [1., 0.], [0., 1.], [0., 0.], [0., 0.]
        ], requires_grad=True)
        image = torch.tensor([
            [0., 0.], [0., 0.], [0., 1.], [1., 0.]
        ], requires_grad=True)
        joint = torch.randn(4, 5, requires_grad=True)
        links = torch.tensor([[0, 2], [1, 3]])
        objective, diagnostics = alignment_objective(
            joint, structure, image, None, None, links,
            InfoNCE_loss(temperature=.1), GRAM_Loss(K=1), .1,
        )
        self.assertEqual(diagnostics['gram_valid_rows'], 0)
        self.assertEqual(diagnostics['gram_zero_positive_batches'], 1)
        self.assertTrue(torch.isfinite(objective))
        objective.backward()
        self.assertIsNotNone(joint.grad)
        self.assertTrue(torch.isfinite(joint.grad).all())

    def test_half_precision_inputs_promote_gram_math_to_float32(self):
        structure = torch.randn(3, 4, dtype=torch.float16, requires_grad=True)
        image = torch.randn(3, 4, dtype=torch.float16, requires_grad=True)
        from src.loss import volume_computation
        _, volumes = volume_computation(structure, {'image': image}, 'image', K=2)
        self.assertEqual(volumes.dtype, torch.float32)
        volumes.sum().backward()
        self.assertTrue(torch.isfinite(structure.grad).all())

    def test_available_modality_candidate_fallback(self):
        torch.manual_seed(37)
        links = torch.tensor([[0, 3], [1, 4], [2, 5]])
        structure = torch.randn(6, 5, requires_grad=True)
        relation = torch.randn(6, 5, requires_grad=True)
        attribute = torch.randn(6, 5, requires_grad=True)
        gram = GRAM_Loss(K=2)
        loss = gram(structure, None, attribute, relation, links)
        self.assertEqual(gram.last_diagnostics['gram_candidate_modality'], 'relation')
        self.assertEqual(gram.last_diagnostics['gram_matrix_size'], 3)
        loss.backward()
        gram(structure, None, attribute, None, links)
        self.assertEqual(gram.last_diagnostics['gram_candidate_modality'], 'attribute')
        with self.assertRaisesRegex(ValueError, 'at least one'):
            gram(structure, None, None, None, links)

    def test_alignment_diagnostics_disable_gram_and_empty_batch(self):
        torch.manual_seed(41)
        joint = torch.randn(6, 8, requires_grad=True)
        structure = torch.randn(6, 5, requires_grad=True)
        links = torch.tensor([[0, 3], [1, 4], [2, 5]])
        objective, diagnostics = alignment_objective(
            joint, structure, None, None, None, links,
            InfoNCE_loss(temperature=.1), GRAM_Loss(), .1, disable_gram=True,
        )
        self.assertTrue(torch.isfinite(objective))
        self.assertEqual(diagnostics['loss_gram_scaled'], 0.0)
        self.assertEqual(diagnostics['nce_direction'], 'left_to_right')
        empty, empty_diagnostics = alignment_objective(
            joint, structure, None, None, None, torch.empty((0, 2), dtype=torch.long),
            InfoNCE_loss(temperature=.1), GRAM_Loss(), .1, disable_gram=True,
        )
        self.assertIsNone(empty)
        self.assertTrue(empty_diagnostics['skipped_empty_batch'])

    def test_optimizer_profile_resume_semantics(self):
        def populated_optimizer(parameter):
            optimizer = torch.optim.AdamW([parameter], lr=.0005, weight_decay=.01)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            return optimizer

        repaired_args = parse_args(['--device', 'cpu'])
        repaired_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        repaired = populated_optimizer(repaired_parameter)
        state_size = len(repaired.state)
        repaired, event = apply_optimizer_policy(
            repaired, [{'params': [repaired_parameter]}], repaired_args, 50
        )
        self.assertEqual(event, 'learning_rate_changed')
        self.assertEqual(len(repaired.state), state_size)
        saved = copy.deepcopy(repaired.state_dict())
        resumed_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        resumed = torch.optim.AdamW([resumed_parameter], lr=.0005, weight_decay=.01)
        resumed.load_state_dict(saved)
        same, event = apply_optimizer_policy(
            resumed, [{'params': [resumed_parameter]}], repaired_args, 51
        )
        self.assertIs(same, resumed)
        self.assertIsNone(event)
        self.assertTrue(same.state)

        compat_args = parse_args(['--profile', 'author_code_compat'])
        compat_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        compat = populated_optimizer(compat_parameter)
        reset, event = apply_optimizer_policy(
            compat, [{'params': [compat_parameter]}], compat_args, 50
        )
        self.assertEqual(event, 'optimizer_reset')
        self.assertFalse(reset.state)
        self.assertEqual(reset.param_groups[0]['lr'], .0001)
        compat_parameter.grad = torch.ones_like(compat_parameter)
        reset.step()
        compat_saved = copy.deepcopy(reset.state_dict())
        compat_resumed_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        compat_resumed = torch.optim.AdamW([compat_resumed_parameter], lr=.0001, weight_decay=.01)
        compat_resumed.load_state_dict(compat_saved)
        reset_again, event = apply_optimizer_policy(
            compat_resumed, [{'params': [compat_resumed_parameter]}], compat_args, 51
        )
        self.assertEqual(event, 'optimizer_reset')
        self.assertIsNot(reset_again, compat_resumed)
        self.assertFalse(reset_again.state)
        self.assertEqual(reset_again.param_groups[0]['lr'], .0001)


if __name__ == '__main__':
    unittest.main()

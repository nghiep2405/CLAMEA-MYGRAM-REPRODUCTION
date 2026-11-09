import unittest

import numpy as np
import torch

from src.Load import get_matrix
from src.Mfusion import MCD
from src.graph import build_graph_inputs
from src.models import RRGAT


ATOL = 1e-6
RTOL = 1e-5


def dense_relation_incidence(index, value, edge_count, relation_count):
    result = torch.zeros(edge_count, relation_count, dtype=value.dtype)
    if value.numel():
        result.index_put_((index[0], index[1]), value, accumulate=True)
    return result


def dense_mcd(operator, inputs, k, alpha, beta):
    matrix = operator.to_dense()
    h0 = inputs
    h = h0
    for _ in range(k):
        h = beta * (matrix @ h) + alpha * h0
    gamma = beta**k + alpha * sum(beta**i for i in range(k))
    return h / gamma


def dense_rrgat(features, relation_embedding, edge_index, relation_index,
                 relation_value, kernels, node_count):
    relation_incidence = dense_relation_incidence(
        relation_index, relation_value, edge_index.shape[1], relation_embedding.shape[0]
    )
    tri_relation = relation_incidence @ relation_embedding
    tri_relation = torch.nn.functional.normalize(tri_relation, dim=1, p=2)
    current = torch.tanh(features)
    outputs = [current]
    for kernel in kernels:
        neighbours = current[edge_index[1]]
        reflected = neighbours - 2 * (neighbours * tri_relation).sum(1, keepdim=True) * tri_relation
        logits = (tri_relation @ kernel).squeeze(-1)
        weights = torch.empty_like(logits)
        for destination in range(node_count):
            mask = edge_index[0] == destination
            if mask.any():
                weights[mask] = torch.softmax(logits[mask], dim=0)
        aggregated = torch.zeros(node_count, reflected.shape[1], dtype=reflected.dtype)
        aggregated.index_add_(0, edge_index[0], reflected * weights[:, None])
        current = torch.tanh(aggregated)
        outputs.append(current)
    return torch.cat(outputs, dim=-1)


class SparseGraphTests(unittest.TestCase):
    def setUp(self):
        # Duplicate triple, a second relation on the same pair, an explicit reverse,
        # a self triple, and node 4 isolated from all input triples.
        self.triples = [
            (0, 0, 1), (0, 0, 1), (0, 1, 1), (1, 2, 0),
            (1, 1, 2), (2, 0, 3), (3, 2, 3),
        ]
        self.graph = build_graph_inputs(5, self.triples, mcd_self_loops=2)

    def test_rrgat_graph_matches_tiny_scipy_oracle(self):
        shifted = [(h, r + 1, t) for h, r, t in self.triples]
        entity = set(range(5))
        relation = {0, *(r for _, r, _ in shifted)}
        old_adj, old_r_index, old_r_value, old_entity, old_relation = get_matrix(
            shifted, entity, relation
        )

        old_edges = torch.from_numpy(np.stack(old_adj.nonzero(), axis=0)).long()
        self.assertTrue(torch.equal(self.graph.edge_index, old_edges))
        old_index = torch.tensor(old_r_index, dtype=torch.long).T
        old_value = torch.tensor(old_r_value, dtype=torch.float32)
        old_incidence = dense_relation_incidence(
            old_index, old_value, old_edges.shape[1], old_relation.shape[1]
        )
        new_incidence = dense_relation_incidence(
            self.graph.relation_index, self.graph.relation_value,
            self.graph.triple_size, self.graph.relation_size,
        )
        torch.testing.assert_close(new_incidence, old_incidence, atol=ATOL, rtol=RTOL)

        old_support = torch.from_numpy((old_entity != 0).toarray())
        new_support = self.graph.entity_average.to_dense() != 0
        self.assertTrue(torch.equal(new_support, old_support))
        expected_average = old_support.float() / old_support.sum(1, keepdim=True)
        torch.testing.assert_close(
            self.graph.entity_average.to_dense(), expected_average, atol=ATOL, rtol=RTOL
        )

    def test_mcd_operator_semantics_and_dense_output_gradient(self):
        operator = self.graph.mcd_operator.coalesce()
        self.assertEqual(operator.layout, torch.sparse_coo)
        self.assertEqual(operator._nnz(), self.graph.diagnostics['mcd_nonself_directed_edges'] + 5)
        self.assertEqual(self.graph.diagnostics['mcd_self_loops_per_node'], 2)
        self.assertEqual(self.graph.diagnostics['dense_n_by_n_tensors'], 0)

        # Independently form raw A with loop multiplicity two, degree, normalization,
        # and [destination, source] orientation.
        raw = torch.zeros(5, 5)
        for head, _, tail in self.triples:
            if head != tail:
                raw[tail, head] = 1
                raw[head, tail] = 1
        raw += 2 * torch.eye(5)
        degree = raw.sum(1)
        expected = degree.rsqrt()[:, None] * raw * degree.rsqrt()[None, :]
        torch.testing.assert_close(operator.to_dense(), expected, atol=ATOL, rtol=RTOL)

        generator = torch.Generator().manual_seed(7)
        sparse_x = torch.randn(5, 4, generator=generator, requires_grad=True)
        dense_x = sparse_x.detach().clone().requires_grad_(True)
        mcd = MCD(k=2, alpha=.7, beta=.3, x_drop_rate=0, edge_drop_rate=0, z_drop_rate=0)
        sparse_out = mcd(operator, {'x': sparse_x})['x']
        dense_out = dense_mcd(operator, dense_x, 2, .7, .3)
        torch.testing.assert_close(sparse_out, dense_out, atol=ATOL, rtol=RTOL)
        sparse_out.square().sum().backward()
        dense_out.square().sum().backward()
        torch.testing.assert_close(sparse_x.grad, dense_x.grad, atol=ATOL, rtol=RTOL)

    def test_rrgat_dense_output_and_gradients(self):
        torch.manual_seed(11)
        args = object()
        sparse_layer = RRGAT(
            args=args, node_size=5, rel_size=self.graph.relation_size,
            triple_size=self.graph.triple_size, node_dim=6, depth=2,
        )
        sparse_features = torch.randn(5, 6, requires_grad=True)
        sparse_relations = torch.randn(self.graph.relation_size, 6, requires_grad=True)
        dense_features = sparse_features.detach().clone().requires_grad_(True)
        dense_relations = sparse_relations.detach().clone().requires_grad_(True)
        dense_kernels = [p.detach().clone().requires_grad_(True) for p in sparse_layer.attn_kernels]

        sparse_out = sparse_layer([
            sparse_features, sparse_relations, self.graph.edge_index,
            self.graph.relation_index, self.graph.relation_value,
        ])
        dense_out = dense_rrgat(
            dense_features, dense_relations, self.graph.edge_index,
            self.graph.relation_index, self.graph.relation_value, dense_kernels, 5,
        )
        torch.testing.assert_close(sparse_out, dense_out, atol=ATOL, rtol=RTOL)
        sparse_out.square().sum().backward()
        dense_out.square().sum().backward()
        torch.testing.assert_close(sparse_features.grad, dense_features.grad, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(sparse_relations.grad, dense_relations.grad, atol=ATOL, rtol=RTOL)
        for actual, expected in zip(sparse_layer.attn_kernels, dense_kernels):
            torch.testing.assert_close(actual.grad, expected.grad, atol=ATOL, rtol=RTOL)
        # Isolated node receives no RRGAT message; every propagated block is zero.
        self.assertTrue(torch.equal(sparse_out[4, 6:], torch.zeros(12)))

    def test_one_node_and_configurable_loop_multiplicity(self):
        for loops in (1, 2):
            graph = build_graph_inputs(1, [(0, 3, 0), (0, 3, 0)], mcd_self_loops=loops)
            self.assertEqual(graph.mcd_operator._nnz(), 1)
            torch.testing.assert_close(graph.mcd_operator.to_dense(), torch.ones(1, 1))
            self.assertEqual(graph.diagnostics['mcd_self_loops_per_node'], loops)
            self.assertEqual(graph.diagnostics['mcd_nonself_directed_edges'], 0)

    def test_dense_reference_is_size_guarded(self):
        with self.assertRaisesRegex(ValueError, 'tiny reference'):
            get_matrix([(0, 1, 4096)], {0, 4096}, {0, 1})


if __name__ == '__main__':
    unittest.main()

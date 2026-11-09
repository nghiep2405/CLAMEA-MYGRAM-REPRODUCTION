"""Sparse graph construction for the repaired MyGram pipeline."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch


def tensor_nbytes(tensor):
    return tensor.element_size() * tensor.numel()


def sparse_storage_nbytes(tensor):
    tensor = tensor.coalesce()
    return tensor_nbytes(tensor.indices()) + tensor_nbytes(tensor.values())


def sparse_fingerprint(tensor):
    tensor = tensor.coalesce().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.indices().numpy().tobytes())
    digest.update(tensor.values().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class GraphInputs:
    node_size: int
    relation_size: int
    triple_size: int
    edge_index: torch.Tensor
    relation_index: torch.Tensor
    relation_value: torch.Tensor
    entity_average: torch.Tensor
    mcd_operator: torch.Tensor
    diagnostics: dict


def _canonical_directed_edges(triples_array):
    if triples_array.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    forward = triples_array[:, [0, 2]]
    reverse = triples_array[:, [2, 0]]
    directed = np.concatenate((forward, reverse), axis=0)
    unique_edges, inverse = np.unique(directed, axis=0, return_inverse=True)
    return unique_edges, inverse, directed


def build_graph_inputs(node_size, triples, mcd_self_loops=2):
    """Build RRGAT structures and the normalized MCD [destination, source] operator.

    RRGAT relation records preserve duplicate triples and multiple relations. MCD
    follows published execution: non-self directed edges are deduplicated and the
    dense diagonal plus DGL add_self_loop become two loops per node by default.
    """
    if node_size <= 0:
        raise ValueError('node_size must be positive')
    if mcd_self_loops not in (1, 2):
        raise ValueError('mcd_self_loops must be 1 or 2')
    triples_array = np.asarray(triples, dtype=np.int64)
    if triples_array.size == 0:
        triples_array = np.empty((0, 3), dtype=np.int64)
    if triples_array.ndim != 2 or triples_array.shape[1] != 3:
        raise ValueError('triples must have shape [T,3]')
    if triples_array.size and (
        triples_array[:, [0, 2]].min() < 0
        or triples_array[:, [0, 2]].max() >= node_size
        or triples_array[:, 1].min() < 0
    ):
        raise ValueError('triple ID outside graph range')

    raw_max_relation = int(triples_array[:, 1].max()) if len(triples_array) else -1
    base_relation_size = raw_max_relation + 2  # Upstream reserves relation ID zero.
    relation_size = 2 * base_relation_size
    edges, inverse, directed = _canonical_directed_edges(triples_array)
    edge_count = len(edges)

    if len(triples_array):
        shifted = triples_array[:, 1] + 1
        directed_relations = np.concatenate((shifted, shifted + base_relation_size))
        counts = np.bincount(inverse, minlength=edge_count)
        relation_values = (1.0 / counts[inverse]).astype(np.float32)
        relation_index = np.stack((inverse, directed_relations), axis=0)
    else:
        counts = np.empty(0, dtype=np.int64)
        relation_values = np.empty(0, dtype=np.float32)
        relation_index = np.empty((2, 0), dtype=np.int64)

    edge_index = torch.from_numpy(edges.T.copy()).long()
    relation_index_t = torch.from_numpy(relation_index.copy()).long()
    relation_value = torch.from_numpy(relation_values.copy())

    # Upstream entity averaging ignores scipy normalized values and applies sparse
    # softmax to zero logits: every unique neighbor/self receives 1/out-degree.
    diagonal = np.arange(node_size, dtype=np.int64)
    entity_edges = np.concatenate((edges, np.stack((diagonal, diagonal), axis=1)), axis=0)
    entity_edges = np.unique(entity_edges, axis=0)
    entity_degree = np.bincount(entity_edges[:, 0], minlength=node_size)
    entity_values = (1.0 / entity_degree[entity_edges[:, 0]]).astype(np.float32)
    entity_average = torch.sparse_coo_tensor(
        torch.from_numpy(entity_edges.T.copy()).long(),
        torch.from_numpy(entity_values),
        (node_size, node_size),
    ).coalesce()

    # MCD skips self-triples, deduplicates all non-self directed edges, then adds
    # the configured loop multiplicity. Matrix orientation is [destination, source].
    nonself = edges[edges[:, 0] != edges[:, 1]]
    in_degree = np.bincount(nonself[:, 1], minlength=node_size).astype(np.float64)
    in_degree += mcd_self_loops
    norm = np.power(in_degree, -0.5)
    mcd_indices = np.concatenate(
        (nonself[:, [1, 0]], np.stack((diagonal, diagonal), axis=1)), axis=0
    )
    nonself_values = norm[nonself[:, 0]] * norm[nonself[:, 1]]
    loop_values = mcd_self_loops * norm * norm
    mcd_values = np.concatenate((nonself_values, loop_values)).astype(np.float32)
    mcd_operator = torch.sparse_coo_tensor(
        torch.from_numpy(mcd_indices.T.copy()).long(),
        torch.from_numpy(mcd_values),
        (node_size, node_size),
    ).coalesce()

    diagnostics = {
        'node_size': node_size,
        'input_triples': len(triples_array),
        'rrgat_unique_directed_edges': edge_count,
        'rrgat_relation_contributions': len(relation_values),
        'rrgat_duplicate_relation_coordinates': len(relation_values) - np.unique(relation_index, axis=1).shape[1] if len(relation_values) else 0,
        'rrgat_isolated_nodes': int(np.sum(entity_degree == 1)),
        'entity_average_nnz': entity_average._nnz(),
        'mcd_nonself_directed_edges': len(nonself),
        'mcd_self_loops_per_node': mcd_self_loops,
        'mcd_nnz_after_coalesce': mcd_operator._nnz(),
        'mcd_min_degree': float(in_degree.min()),
        'mcd_max_degree': float(in_degree.max()),
        'relation_size': relation_size,
        'dense_n_by_n_tensors': 0,
        'storage_bytes': {
            'edge_index': tensor_nbytes(edge_index),
            'relation_index': tensor_nbytes(relation_index_t),
            'relation_value': tensor_nbytes(relation_value),
            'entity_average_sparse': sparse_storage_nbytes(entity_average),
            'mcd_operator_sparse': sparse_storage_nbytes(mcd_operator),
        },
        'fingerprints': {
            'entity_average': sparse_fingerprint(entity_average),
            'mcd_operator': sparse_fingerprint(mcd_operator),
        },
    }
    return GraphInputs(
        node_size=node_size,
        relation_size=relation_size,
        triple_size=edge_count,
        edge_index=edge_index,
        relation_index=relation_index_t,
        relation_value=relation_value,
        entity_average=entity_average,
        mcd_operator=mcd_operator,
        diagnostics=diagnostics,
    )

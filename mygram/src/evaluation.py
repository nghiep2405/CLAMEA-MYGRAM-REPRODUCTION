"""Memory-bounded evaluation over the complete ordered candidate set."""
from collections import Counter

import torch
import torch.nn.functional as F


def _similarity(left, right):
    # Upstream evaluates normalized squared L2 distance and converts via 1-distance.
    distance = (
        left.square().sum(1, keepdim=True)
        + right.square().sum(1).unsqueeze(0)
        - 2.0 * left @ right.T
    ).clamp_min(0.0)
    return 1.0 - distance


def _merge_topk(current, values, k):
    return torch.topk(torch.cat((current, values), dim=1), k=k, dim=1).values


def csls_corrections(left, right, k, block_size):
    """Compute both CSLS neighborhood means against the entire candidate set."""
    if k <= 0 or block_size <= 0:
        raise ValueError('CSLS k and block size must be positive')
    k_eff = min(k, right.shape[0], left.shape[0])
    row_top = torch.full((left.shape[0], k_eff), -torch.inf, device=left.device)
    column_top = torch.full((right.shape[0], k_eff), -torch.inf, device=left.device)
    for left_start in range(0, left.shape[0], block_size):
        left_stop = min(left_start + block_size, left.shape[0])
        for right_start in range(0, right.shape[0], block_size):
            right_stop = min(right_start + block_size, right.shape[0])
            similarities = _similarity(
                left[left_start:left_stop], right[right_start:right_stop]
            )
            row_top[left_start:left_stop] = _merge_topk(
                row_top[left_start:left_stop], similarities, k_eff
            )
            column_top[right_start:right_stop] = _merge_topk(
                column_top[right_start:right_stop], similarities.T, k_eff
            )
    return row_top.mean(1), column_top.mean(1), k_eff


def _direction_metrics(query, candidates, query_ids, candidate_ids, row_correction,
                       column_correction, block_size, top_k, direction):
    total = query.shape[0]
    ranks = torch.empty(total, dtype=torch.long)
    predictions = []
    candidate_positions = torch.arange(total, device=query.device)
    prediction_count = min(3, total)
    for start in range(0, total, block_size):
        stop = min(start + block_size, total)
        scores = _similarity(query[start:stop], candidates)
        if row_correction is not None:
            scores = 2.0 * scores - row_correction[start:stop, None] - column_correction[None, :]
        local_positions = torch.arange(start, stop, device=query.device)
        ground_truth = scores[
            torch.arange(stop - start, device=query.device), local_positions
        ]
        better = (scores > ground_truth[:, None]).sum(1)
        tied_before = (
            (scores == ground_truth[:, None])
            & (candidate_positions[None, :] < local_positions[:, None])
        ).sum(1)
        block_ranks = 1 + better + tied_before
        ranks[start:stop] = block_ranks.cpu()
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :prediction_count]
        ordered_scores = scores.gather(1, order)
        order_cpu = order.cpu().tolist()
        candidate_ids_cpu = candidate_ids.cpu().tolist()
        for offset in range(stop - start):
            position = start + offset
            predictions.append({
                'direction': direction,
                'query_position': position,
                'query_id': int(query_ids[position]),
                'ground_truth_position': position,
                'ground_truth_id': int(candidate_ids[position]),
                'rank_1based': int(block_ranks[offset]),
                'top_candidate_positions': order_cpu[offset],
                'top_candidate_ids': [candidate_ids_cpu[value] for value in order_cpu[offset]],
                'top_scores': [float(value) for value in ordered_scores[offset].cpu()],
            })
    metrics = {
        f'hits@{k}': float((ranks <= k).float().mean()) for k in top_k
    }
    metrics.update({
        'mr': float(ranks.float().mean()),
        'mrr': float(ranks.float().reciprocal().mean()),
        'count': total,
    })
    return metrics, predictions, ranks


def evaluate_embeddings(embeddings, left_ids, right_ids, *, csls_k=3,
                        block_size=1024, top_k=(1, 10, 50)):
    """Evaluate positional test-pair candidates without materializing full distance."""
    if block_size <= 0:
        raise ValueError('block_size must be positive')
    if len(left_ids) != len(right_ids) or len(left_ids) == 0:
        raise ValueError('Ordered left/right candidates must have equal nonzero length')
    device = embeddings.device
    left_ids = torch.as_tensor(left_ids, dtype=torch.long, device=device)
    right_ids = torch.as_tensor(right_ids, dtype=torch.long, device=device)
    normalized = F.normalize(embeddings, dim=1)
    left = normalized[left_ids]
    right = normalized[right_ids]

    if csls_k is None:
        left_correction = right_correction = None
        k_eff = None
        scope = 'disabled'
    else:
        left_correction, right_correction, k_eff = csls_corrections(
            left, right, csls_k, block_size
        )
        scope = 'entire_candidate_set_two_pass'
    l2r, l2r_predictions, l2r_ranks = _direction_metrics(
        left, right, left_ids, right_ids, left_correction, right_correction,
        block_size, top_k, 'left_to_right',
    )
    r2l, r2l_predictions, r2l_ranks = _direction_metrics(
        right, left, right_ids, left_ids, right_correction, left_correction,
        block_size, top_k, 'right_to_left',
    )
    duplicate_targets = {
        str(key): count for key, count in Counter(right_ids.cpu().tolist()).items()
        if count > 1
    }
    duplicate_sources = {
        str(key): count for key, count in Counter(left_ids.cpu().tolist()).items()
        if count > 1
    }
    return {
        'metrics': {'l2r': l2r, 'r2l': r2l},
        'predictions': {'l2r': l2r_predictions, 'r2l': r2l_predictions},
        'ranks': {'l2r': l2r_ranks, 'r2l': r2l_ranks},
        'protocol': {
            'candidate_protocol': 'test_pairs',
            'candidate_semantics': 'ordered positional rows; duplicate entity IDs retained',
            'selection_rule': 'best_test_hits@1_l2r',
            'tie_rule': 'higher score first; exact ties use lower candidate position',
            'csls_k': csls_k,
            'csls_k_eff': k_eff,
            'csls_scope': scope,
            'block_size': block_size,
            'normalization': 'l2',
            'base_score': '1 - squared_l2_distance',
            'duplicate_target_ids': duplicate_targets,
            'duplicate_candidate_ids': {
                'left_to_right': duplicate_targets,
                'right_to_left': duplicate_sources,
            },
        },
    }

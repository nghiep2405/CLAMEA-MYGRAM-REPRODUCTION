"""Ordered, blockwise iterative-learning state transitions."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass
class IterativeLearningState:
    train_links: list
    fixed_test_links: list
    left_pool: list
    right_pool: list
    new_links: list

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{key: [tuple(row) if isinstance(row, (list, tuple)) and len(row) == 2 else row
                            for row in rows]
                     for key, rows in value.items()})


def ordered_il_state(train_links, test_links, left_entities, right_entities):
    train_links = [tuple(map(int, pair)) for pair in train_links]
    fixed_test_links = [tuple(map(int, pair)) for pair in test_links]
    left_train = {left for left, _ in train_links}
    right_train = {right for _, right in train_links}
    return IterativeLearningState(
        train_links=train_links,
        fixed_test_links=fixed_test_links,
        left_pool=[int(value) for value in left_entities if value not in left_train],
        right_pool=[int(value) for value in right_entities if value not in right_train],
        new_links=[],
    )


def _nearest_positions(query, candidates, block_size):
    if block_size <= 0:
        raise ValueError('IL block size must be positive')
    if candidates.shape[0] == 0:
        return torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    result = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    for query_start in range(0, query.shape[0], block_size):
        query_stop = min(query_start + block_size, query.shape[0])
        best_distance = torch.full((query_stop - query_start,), torch.inf, device=query.device)
        best_position = torch.zeros(query_stop - query_start, dtype=torch.long, device=query.device)
        query_block = query[query_start:query_stop]
        for candidate_start in range(0, candidates.shape[0], block_size):
            candidate_stop = min(candidate_start + block_size, candidates.shape[0])
            candidate_block = candidates[candidate_start:candidate_stop]
            distance = (
                query_block.square().sum(1, keepdim=True)
                + candidate_block.square().sum(1).unsqueeze(0)
                - 2.0 * query_block @ candidate_block.T
            ).clamp_min(0.0)
            local_distance, local_position = distance.min(1)
            global_position = local_position + candidate_start
            replace = (local_distance < best_distance) | (
                (local_distance == best_distance) & (global_position < best_position)
            )
            best_distance = torch.where(replace, local_distance, best_distance)
            best_position = torch.where(replace, global_position, best_position)
        result[query_start:query_stop] = best_position
    return result


def mutual_nearest_pairs(embeddings, left_pool, right_pool, block_size=1024):
    """Return mutual pairs using only ordered pools and model embeddings."""
    if not left_pool or not right_pool:
        return [], {
            'left_pool_size': len(left_pool), 'right_pool_size': len(right_pool),
            'mutual_pairs': 0, 'empty_pool': True, 'block_size': block_size,
        }
    device = embeddings.device
    left_ids = torch.as_tensor(left_pool, dtype=torch.long, device=device)
    right_ids = torch.as_tensor(right_pool, dtype=torch.long, device=device)
    normalized = F.normalize(embeddings, dim=1)
    left = normalized[left_ids]
    right = normalized[right_ids]
    left_to_right = _nearest_positions(left, right, block_size)
    right_to_left = _nearest_positions(right, left, block_size)
    pairs = []
    for left_position, right_position in enumerate(left_to_right.tolist()):
        if int(right_to_left[right_position]) == left_position:
            pairs.append((int(left_pool[left_position]), int(right_pool[right_position])))
    return pairs, {
        'left_pool_size': len(left_pool),
        'right_pool_size': len(right_pool),
        'mutual_pairs': len(pairs),
        'empty_pool': False,
        'block_size': block_size,
        'tie_rule': 'lower ordered pool position',
    }


def update_il_state(state, mutual_pairs, epoch, il_start, il_step):
    """Apply the upstream 10-check intersection/add schedule without test labels."""
    if il_step <= 0:
        raise ValueError('il_step must be positive')
    event = {
        'epoch': epoch,
        'scheduled': False,
        'action': 'none',
        'mutual_pairs': len(mutual_pairs),
        'candidate_pairs': len(state.new_links),
        'added_pairs': 0,
    }
    if epoch < il_start or (epoch + 1) % il_step != 0:
        return event
    event['scheduled'] = True
    train_set = set(state.train_links)
    allowed_left = set(state.left_pool)
    allowed_right = set(state.right_pool)
    current = []
    seen = set()
    for pair in mutual_pairs:
        pair = tuple(map(int, pair))
        if (pair not in seen and pair not in train_set
                and pair[0] in allowed_left and pair[1] in allowed_right):
            current.append(pair)
            seen.add(pair)
    cycle_position = (epoch + 1) % (il_step * 10)
    if cycle_position == il_step:
        state.new_links = current
        event['action'] = 'candidate_start'
    else:
        current_set = set(current)
        state.new_links = [pair for pair in state.new_links if pair in current_set]
        event['action'] = 'candidate_intersection'

    if cycle_position == 0:
        additions = [pair for pair in state.new_links if pair not in train_set]
        state.train_links.extend(additions)
        added_left = {left for left, _ in additions}
        added_right = {right for _, right in additions}
        state.left_pool = [value for value in state.left_pool if value not in added_left]
        state.right_pool = [value for value in state.right_pool if value not in added_right]
        state.new_links = []
        event['action'] = 'pairs_added'
        event['added_pairs'] = len(additions)
    event['candidate_pairs'] = len(state.new_links)
    event['train_pairs'] = len(state.train_links)
    event['left_pool_size'] = len(state.left_pool)
    event['right_pool_size'] = len(state.right_pool)
    return event


def diagnostic_precision(pairs, fixed_test_links):
    """Held-out labels are accepted only by this post-selection diagnostic."""
    truth = set(fixed_test_links)
    return {
        'selected_pairs': len(pairs),
        'true_pairs': sum(pair in truth for pair in pairs),
        'precision': (sum(pair in truth for pair in pairs) / len(pairs)) if pairs else None,
        'used_for_selection': False,
    }

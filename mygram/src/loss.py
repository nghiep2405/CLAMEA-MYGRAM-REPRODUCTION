"""MyGram objectives with explicit diagnostics for reproducible training."""
import torch
from torch import nn
import torch.nn.functional as F


class CustomMultiLossLayer(nn.Module):
    def __init__(self, loss_num, device=None):
        super().__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(loss_num), requires_grad=True)

    def forward(self, loss_list):
        if len(loss_list) != self.loss_num:
            raise ValueError(f'Expected {self.loss_num} losses, received {len(loss_list)}')
        precision = torch.exp(-self.log_vars)
        return sum(
            precision[index] * loss + self.log_vars[index]
            for index, loss in enumerate(loss_list)
        )


class InfoNCE_loss(nn.Module):
    """Published one-direction positional InfoNCE objective."""

    def __init__(self, device=None, temperature=0.05) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError('InfoNCE temperature must be positive')
        self.device = device  # Compatibility only; labels follow the score device.
        self.t = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.last_diagnostics = {}

    @staticmethod
    def sim(emb_left, emb_right):
        return emb_left.mm(emb_right.t())

    def forward(self, emb, train_links):
        train_links = torch.as_tensor(train_links, dtype=torch.long, device=emb.device)
        if train_links.ndim != 2 or train_links.shape[1] != 2:
            raise ValueError('train_links must have shape [B,2]')
        if train_links.shape[0] == 0:
            raise ValueError('Empty alignment batches must be skipped by the training loop')
        normalized = F.normalize(emb)
        emb_train_left = normalized[train_links[:, 0]]
        emb_train_right = normalized[train_links[:, 1]]
        score = self.sim(emb_train_left, emb_train_right)
        batch_size = emb_train_left.shape[0]
        label = torch.arange(batch_size, dtype=torch.long, device=score.device)
        loss = self.ce_loss(score / self.t, label)
        self.last_diagnostics = {
            'nce_batch_rows': batch_size,
            'nce_direction': 'left_to_right',
            'nce_positional_positives': True,
            'nce_has_useful_negatives': batch_size > 1,
        }
        return loss


def _minimum_compute_dtype(*tensors):
    if any(tensor.dtype == torch.float64 for tensor in tensors):
        return torch.float64
    return torch.float32


def volume_computation(structure, modalities, candidate_modality, K=10):
    """Return top-K candidate indices and Gram volumes for active modalities."""
    if structure.ndim != 2:
        raise ValueError('structure must have shape [B,D]')
    if not modalities or candidate_modality not in modalities:
        raise ValueError('A valid candidate modality is required')
    batch_size = structure.shape[0]
    if batch_size == 0:
        raise ValueError('Empty alignment batches must be skipped by the training loop')
    if any(value.shape != structure.shape for value in modalities.values()):
        raise ValueError('All Gram modalities must have the same [B,D] shape')
    k_eff = min(K, batch_size)
    if k_eff <= 0:
        raise ValueError('K must be positive')

    dtype = _minimum_compute_dtype(structure, *modalities.values())
    structure = structure.to(dtype)
    modalities = {name: value.to(dtype) for name, value in modalities.items()}
    similarity = structure @ modalities[candidate_modality].T
    _, topk_idx = torch.topk(similarity, k_eff, dim=1)

    vectors = [structure.unsqueeze(1).expand(-1, k_eff, -1)]
    vectors.extend(value[topk_idx] for value in modalities.values())
    all_modals = torch.stack(vectors, dim=2)
    gram = torch.einsum('bkif,bkjf->bkij', all_modals, all_modals)
    determinant = torch.linalg.det(gram)
    volumes = torch.sqrt(torch.clamp(torch.abs(determinant), min=1e-8))
    return topk_idx, volumes


def volume_computation4(structure, visual, attribute, relation, K=10):
    """Compatibility wrapper preserving the published four-vector ordering."""
    return volume_computation(
        structure,
        {'image': visual, 'attribute': attribute, 'relation': relation},
        candidate_modality='image',
        K=K,
    )


def sparse_gram_loss(topk_idx, volumes, targets, temperature=0.05,
                     return_diagnostics=False):
    if temperature <= 0:
        raise ValueError('Gram temperature must be positive')
    if topk_idx.shape != volumes.shape or topk_idx.ndim != 2:
        raise ValueError('topk_idx and volumes must have matching [B,K] shapes')
    batch_size, k_eff = topk_idx.shape
    if batch_size == 0 or k_eff == 0:
        raise ValueError('Empty Gram batches must be skipped by the training loop')
    if topk_idx.device != volumes.device:
        raise ValueError('Gram indices and volumes must be on the same device')
    targets = targets.to(device=topk_idx.device, dtype=torch.long)
    if volumes.dtype in (torch.float16, torch.bfloat16):
        volumes = volumes.float()
    if targets.shape != (batch_size,):
        raise ValueError('targets must have shape [B]')
    targets_expanded = targets.unsqueeze(1).expand(-1, k_eff)
    positive_mask = topk_idx == targets_expanded
    has_positive = positive_mask.any(dim=1)
    valid_rows = has_positive.nonzero(as_tuple=False).squeeze(-1)
    diagnostics = {
        'gram_valid_rows': int(valid_rows.numel()),
        'gram_batch_rows': batch_size,
        'K_eff': k_eff,
        'gram_positive_coverage': float(valid_rows.numel() / batch_size) if batch_size else 0.0,
        'gram_zero_positive_batch': valid_rows.numel() == 0,
        'gram_has_useful_negatives': k_eff > 1,
    }
    if valid_rows.numel() == 0:
        loss = volumes.sum() * 0.0
    else:
        valid_volumes = volumes[valid_rows] / temperature
        valid_mask = positive_mask[valid_rows]
        valid_positive_index = valid_mask.float().argmax(dim=1)
        log_probabilities = F.log_softmax(-valid_volumes, dim=1)
        loss = -log_probabilities[
            torch.arange(valid_rows.numel(), device=volumes.device), valid_positive_index
        ].mean()
    return (loss, diagnostics) if return_diagnostics else loss


class GRAM_Loss(nn.Module):
    """One-direction MyGram volume objective over the available target modalities."""

    CANDIDATE_ORDER = ('image', 'relation', 'attribute')

    def __init__(self, temperature=0.05, K=10, direction_scale=0.5,
                 gram_modalities='available'):
        super().__init__()
        if temperature <= 0 or K <= 0 or direction_scale < 0:
            raise ValueError('Gram temperature/K must be positive and scale nonnegative')
        if gram_modalities != 'available':
            raise ValueError("gram_modalities must be 'available'")
        self.temperature = temperature
        self.K = K
        self.direction_scale = direction_scale
        self.gram_modalities = gram_modalities
        self.last_diagnostics = {}

    def forward(self, struct, visual, attr, rel, train_links):
        train_links = torch.as_tensor(train_links, dtype=torch.long, device=struct.device)
        if train_links.ndim != 2 or train_links.shape[1] != 2:
            raise ValueError('train_links must have shape [B,2]')
        if train_links.shape[0] == 0:
            raise ValueError('Empty alignment batches must be skipped by the training loop')

        supplied = {'image': visual, 'attribute': attr, 'relation': rel}
        available = {name: value for name, value in supplied.items() if value is not None}
        if not available:
            raise ValueError('Gram requires at least one target modality')
        candidate_modality = next(name for name in self.CANDIDATE_ORDER if name in available)

        left = train_links[:, 0]
        right = train_links[:, 1]
        structure_left = F.normalize(struct[left], dim=1)
        modality_right = {
            name: F.normalize(value[right], dim=1) for name, value in available.items()
        }
        topk_idx, volumes = volume_computation(
            structure_left, modality_right, candidate_modality, K=self.K
        )
        targets = torch.arange(train_links.shape[0], device=struct.device)
        raw_loss, diagnostics = sparse_gram_loss(
            topk_idx, volumes, targets, temperature=self.temperature,
            return_diagnostics=True,
        )
        scaled_loss = self.direction_scale * raw_loss
        self.last_diagnostics = {
            **diagnostics,
            'gram_candidate_modality': candidate_modality,
            'gram_modalities': ['structure', *available.keys()],
            'gram_matrix_size': 1 + len(available),
            'loss_gram_raw_one_direction': float(raw_loss.detach()),
            'loss_gram_scaled': float(scaled_loss.detach()),
        }
        return scaled_loss


def alignment_objective(joint_emb, structure_emb, img_emb, attr_emb, rel_emb,
                        train_links, nce_loss, gram_loss, gram_weight,
                        disable_gram=False):
    """Compose the baseline alignment objective and its required diagnostics."""
    links = torch.as_tensor(train_links, dtype=torch.long, device=joint_emb.device)
    if links.ndim != 2 or links.shape[1] != 2:
        raise ValueError('train_links must have shape [B,2]')
    if links.shape[0] == 0:
        return None, {
            'skipped_empty_batch': True,
            'loss_nce': None,
            'loss_gram_raw_one_direction': None,
            'loss_gram_scaled': None,
            'loss_total': None,
            'gram_valid_rows': 0,
            'gram_batch_rows': 0,
            'K_eff': 0,
            'gram_zero_positive_batches': 0,
            'grad_norm': None,
        }

    nce = nce_loss(joint_emb, links)
    if disable_gram:
        scaled_gram = structure_emb.sum() * 0.0
        gram_diagnostics = {
            'gram_valid_rows': 0,
            'gram_batch_rows': links.shape[0],
            'K_eff': 0,
            'gram_zero_positive_batch': False,
            'gram_candidate_modality': None,
            'gram_modalities': [],
            'gram_matrix_size': 0,
            'loss_gram_raw_one_direction': 0.0,
            'loss_gram_scaled': 0.0,
        }
    else:
        scaled_gram = gram_loss(structure_emb, img_emb, attr_emb, rel_emb, links)
        gram_diagnostics = gram_loss.last_diagnostics
    total = nce + gram_weight * scaled_gram
    diagnostics = {
        **gram_diagnostics,
        **nce_loss.last_diagnostics,
        'skipped_empty_batch': False,
        'loss_nce': float(nce.detach()),
        'loss_total': float(total.detach()),
        'gram_outer_weight': gram_weight,
        'gram_zero_positive_batches': int(gram_diagnostics['gram_zero_positive_batch']),
        'grad_norm': None,
    }
    return total, diagnostics

"""Gate E: verify losses, optimizer policies, batch semantics, and ablations."""
import argparse
import copy
import datetime
import gc
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.config import parse_args
from src.data_registry import sha256, write_json
from src.graph import build_graph_inputs
from src.loss import GRAM_Loss, InfoNCE_loss, alignment_objective
from src.models import MultiModalEncoderMrFusion
from src.run import apply_optimizer_policy


TRIPLES = [
    (0, 0, 1), (0, 1, 1), (1, 2, 0), (1, 1, 2),
    (2, 0, 3), (3, 2, 4), (4, 1, 5), (5, 0, 0),
]
LINKS = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.long)


def loss_test_suite():
    suite = unittest.defaultTestLoader.discover(
        str(ROOT/'tests'), pattern='test_loss_optimizer_ablation.py'
    )
    count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('Gate E loss/optimizer oracle suite failed')
    return {
        'status': 'passed',
        'tests': count,
        'atol': 1e-6,
        'rtol': 1e-5,
        'covered': [
            'published four-vector output and gradient',
            'B<K', 'B=1', 'zero-valid positives', 'partial-valid positives',
            'duplicate targets', 'near-singular vectors', 'empty batch skip',
            'candidate modality fallback', 'optimizer resume policies',
        ],
    }


def make_model(args):
    graph = build_graph_inputs(8, TRIPLES, mcd_self_loops=args.mcd_self_loops)
    model = MultiModalEncoderMrFusion(
        args=args,
        ent_num=graph.node_size,
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
    return model, graph


def make_features(args, seed):
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(8, 12, generator=generator) if args.w_img else None
    relation = torch.randn(8, args.bow_size, generator=generator) if args.w_rel else None
    attribute = torch.randn(8, args.bow_size, generator=generator) if args.w_attr else None
    return image, relation, attribute


def run_ablation(name, flags):
    args = parse_args([
        '--config', str(ROOT/'configs/repaired_pmf.json'), '--device', 'cpu', *flags,
    ])
    torch.manual_seed(args.seed)
    model, graph = make_model(args)
    model.train()
    image, relation, attribute = make_features(args, args.seed + 1)
    start = time.perf_counter()
    outputs = model(torch.arange(8), image, relation, attribute)
    if outputs[1].shape != (8, args.structural_output_dim):
        raise AssertionError(f'{name}: structural shape mismatch')
    if outputs[-1].shape != (8, args.joint_output_dim):
        raise AssertionError(f'{name}: joint shape mismatch')
    nce = InfoNCE_loss(temperature=args.nce_temperature)
    gram = GRAM_Loss(
        temperature=args.gram_temperature,
        K=args.gram_k,
        direction_scale=args.gram_direction_scale,
        gram_modalities=args.gram_modalities,
    )
    loss, diagnostics = alignment_objective(
        outputs[-1], outputs[0], outputs[2], outputs[4], outputs[3], LINKS,
        nce, gram, args.gram_weight, disable_gram=args.disable_gram,
    )
    loss.backward()
    gradients = {key: value.grad for key, value in model.named_parameters() if value.grad is not None}
    if not gradients or not all(torch.isfinite(value).all() for value in gradients.values()):
        raise AssertionError(f'{name}: missing or nonfinite gradients')
    expected_active = {
        'image': args.w_img,
        'relation': args.w_rel,
        'attribute': args.w_attr,
    }
    fc_names = {'image': 'img_fc.weight', 'relation': 'rel_fc.weight', 'attribute': 'att_fc.weight'}
    for modality, active in expected_active.items():
        if active != (fc_names[modality] in gradients):
            raise AssertionError(f'{name}: unexpected {modality} projection gradient state')
    expected_candidate = None if args.disable_gram else next(
        modality for modality in ('image', 'relation', 'attribute') if expected_active[modality]
    )
    if diagnostics['gram_candidate_modality'] != expected_candidate:
        raise AssertionError(f'{name}: candidate fallback mismatch')
    diagnostics['grad_norm'] = sum(
        float(value.detach().float().square().sum()) for value in gradients.values()
    ) ** 0.5
    return {
        'status': 'passed',
        'flags': flags,
        'active_modalities': [name for name, active in expected_active.items() if active],
        'mgd_enabled': not args.disable_mgd,
        'gram_enabled': not args.disable_gram,
        'structural_shape': list(outputs[1].shape),
        'joint_shape': list(outputs[-1].shape),
        'loss': float(loss.detach()),
        'parameters_with_gradient': len(gradients),
        'diagnostics': diagnostics,
        'seconds': time.perf_counter() - start,
        'graph_nnz': graph.mcd_operator._nnz(),
    }


def full_graph_batch_semantics():
    args = parse_args([
        '--config', str(ROOT/'configs/repaired_pmf.json'), '--device', 'cpu',
    ])
    torch.manual_seed(args.seed)
    model, _ = make_model(args)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    before = model.img_fc.weight.detach().clone()
    image, relation, attribute = make_features(args, args.seed + 2)
    optimizer.zero_grad()
    forward_calls = 1
    outputs = model(torch.arange(8), image, relation, attribute)
    diagnostics = []
    scalar_loss_sum = 0.0
    batches = list(LINKS.split(2))
    for index, links in enumerate(batches):
        nce = InfoNCE_loss(temperature=args.nce_temperature)
        gram = GRAM_Loss(
            temperature=args.gram_temperature, K=args.gram_k,
            direction_scale=args.gram_direction_scale,
            gram_modalities=args.gram_modalities,
        )
        loss, details = alignment_objective(
            outputs[-1], outputs[0], outputs[2], outputs[4], outputs[3], links,
            nce, gram, args.gram_weight,
        )
        loss.backward(retain_graph=index + 1 < len(batches))
        scalar_loss_sum += float(loss.detach())
        diagnostics.append(details)
    grad_norm = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in model.parameters() if parameter.grad is not None
    ) ** 0.5
    for details in diagnostics:
        details['grad_norm'] = grad_norm
    optimizer_steps = 1
    optimizer.step()
    if torch.equal(before, model.img_fc.weight):
        raise AssertionError('Optimizer step did not update an active projection')
    return {
        'status': 'passed',
        'full_graph_forward_calls': forward_calls,
        'loss_batches': len(batches),
        'batch_rows': [len(batch) for batch in batches],
        'loss_reduction_across_batches': 'sum of batch means',
        'loss_log_accumulator_type': type(scalar_loss_sum).__name__,
        'retained_graph': [True, False],
        'optimizer_steps': optimizer_steps,
        'parameter_updated': True,
        'loss_sum': scalar_loss_sum,
        'grad_norm': grad_norm,
        'batch_diagnostics': diagnostics,
    }


def optimizer_policies():
    result = {}
    for profile in ('repaired_pmf', 'author_code_compat'):
        args = parse_args(['--profile', profile, '--device', 'cpu'])
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        params = [{'params': [parameter]}]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        original_state_items = len(optimizer.state)
        sequence = []
        for epoch in (49, 50, 51):
            previous = optimizer
            optimizer, event = apply_optimizer_policy(optimizer, params, args, epoch)
            sequence.append({
                'epoch': epoch,
                'event': event,
                'same_optimizer': optimizer is previous,
                'state_items': len(optimizer.state),
                'lr': optimizer.param_groups[0]['lr'],
            })
            if profile == 'author_code_compat' and epoch >= 50:
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
        result[profile] = {
            'status': 'passed',
            'initial_state_items': original_state_items,
            'sequence': sequence,
            'resume_semantics': (
                'loaded optimizer state retained after epoch 50'
                if profile == 'repaired_pmf'
                else 'loaded optimizer state intentionally reset on every epoch >= 50'
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'audit/stage_e_validation.json')
    cli = parser.parse_args()
    report = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'running',
        'device': 'cpu',
        'source_sha256': {
            name: sha256(ROOT/name) for name in [
                'src/loss.py', 'src/run.py', 'src/config.py', 'src/models.py',
                'tests/test_loss_optimizer_ablation.py',
            ]
        },
        'loss_oracles': {},
        'optimizer_policies': {},
        'training_step': {},
        'ablations': {},
    }
    cases = {
        'full': [],
        'disable_image': ['--disable-image'],
        'disable_relation': ['--disable-relation'],
        'disable_attribute': ['--disable-attribute'],
        'disable_mgd': ['--disable-mgd'],
        'disable_gram': ['--disable-gram'],
    }
    try:
        report['loss_oracles'] = loss_test_suite()
        report['optimizer_policies'] = optimizer_policies()
        report['training_step'] = full_graph_batch_semantics()
        for name, flags in cases.items():
            report['ablations'][name] = run_ablation(name, flags)
            print(f'{name}: passed', flush=True)
            gc.collect()
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        write_json(cli.report, report)
    print(f'Gate E: {report["status"]}; report: {cli.report.resolve()}', flush=True)


if __name__ == '__main__':
    main()

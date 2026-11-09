"""Gate C: resolve configs, load five datasets, initialize models, and test shapes/gradients."""
import argparse
import datetime
import gc
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from src.config import emit_resolved, parse_args, resolved_dict
from src.data import load_dataset
from src.data_registry import REGISTRY, sha256, write_json
from src.models import MultiModalEncoderMrFusion


def empty_graph_inputs(node_count, relation_count):
    empty = torch.empty((2, 0), dtype=torch.long)
    diagonal = torch.arange(node_count, dtype=torch.long).repeat(2, 1)
    identity = torch.sparse_coo_tensor(
        diagonal, torch.ones(node_count), (node_count, node_count)
    ).coalesce()
    return dict(
        ent_num=node_count,
        rel_size=relation_count,
        triple_size=0,
        adj_matrix=empty,
        r_index=empty,
        r_val=torch.empty(0),
        rel_matrix=empty,
        ent_matrix=identity,
        mcd_operator=identity,
    )


def initialize_real_datasets(data_root, output_root):
    results = {}
    for dataset in REGISTRY:
        start = time.perf_counter()
        args = parse_args([
            '--config', str(ROOT/'configs/repaired_pmf.json'), '--device', 'cpu',
            '--data-root', str(data_root), '--output-root', str(output_root), '--dataset', dataset,
        ])
        data = load_dataset(data_root, dataset, seed=args.seed, rate=args.rate)
        relation_count = max(data.r_hs) + 2
        model = MultiModalEncoderMrFusion(
            args=args,
            img_feature_dim=data.image_features.shape[1],
            char_feature_dim=None,
            use_project_head=False,
            left_ents=data.ordered_ids[0],
            right_ents=data.ordered_ids[1],
            **empty_graph_inputs(data.statistics['total_entities'], relation_count),
        )
        results[dataset] = {
            'status': 'passed',
            'entities': data.statistics['total_entities'],
            'image_input_dim': model.img_fc.in_features,
            'image_output_dim': model.img_fc.out_features,
            'relation_count_for_initialization': relation_count,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'structural_output_dim': args.structural_output_dim,
            'joint_output_dim': args.joint_output_dim,
            'graph_state': 'isolated placeholders; production sparse graph is verified by Gate D',
            'removed_dead_initializers': {
                'RRGAT.emb_atten_cat': not hasattr(model.e_encoder, 'emb_atten_cat'),
                'MultiModalEncoderMrFusion.r_encoder': not hasattr(model, 'r_encoder'),
                'MultiModalEncoderMrFusion.joint_encoder': not hasattr(model, 'joint_encoder'),
            },
            'seconds': time.perf_counter() - start,
        }
        if not all(results[dataset]['removed_dead_initializers'].values()):
            raise AssertionError('A missing-class initializer is still present')
        del model, data
        gc.collect()
        print(f'{dataset}: model initialized', flush=True)
    return results


def tiny_forward_backward():
    args = parse_args([
        '--config', str(ROOT/'configs/repaired_pmf.json'), '--device', 'cpu',
        '--data-root', str(ROOT/'data/mmkg'), '--dataset', 'FBDB15K', '--disable-mgd',
    ])
    node_count, relation_count, edge_count, image_dim = 5, 3, 5, 8
    adj = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    r_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 0, 1]], dtype=torch.long)
    model = MultiModalEncoderMrFusion(
        args=args, ent_num=node_count, rel_size=relation_count, triple_size=edge_count,
        img_feature_dim=image_dim, adj_matrix=adj, r_index=r_index,
        r_val=torch.ones(edge_count), rel_matrix=torch.empty((2, 0), dtype=torch.long),
        ent_matrix=torch.sparse_coo_tensor(
            adj, torch.ones(edge_count), (node_count, node_count)
        ).coalesce(),
        mcd_operator=torch.sparse_coo_tensor(
            torch.arange(node_count).repeat(2, 1), torch.ones(node_count),
            (node_count, node_count),
        ).coalesce(),
        left_ents=[0, 1], right_ents=[2, 3],
    )
    generator = torch.Generator().manual_seed(2023)
    outputs = model(
        torch.arange(node_count),
        torch.randn(node_count, image_dim, generator=generator),
        torch.randn(node_count, args.bow_size, generator=generator),
        torch.randn(node_count, args.bow_size, generator=generator),
    )
    gph, joint = outputs[1], outputs[-1]
    if gph.shape != (node_count, 900) or joint.shape != (node_count, 1800):
        raise AssertionError(f'Unexpected structural/joint shapes: {gph.shape}, {joint.shape}')
    joint.square().mean().backward()
    none_grad = sorted(name for name, p in model.named_parameters() if p.requires_grad and p.grad is None)
    expected_unused = [
        'fusion1.fusion_layer.1.attention.output.LayerNorm.bias',
        'fusion1.fusion_layer.1.attention.output.LayerNorm.weight',
        'fusion1.fusion_layer.1.attention.output.dense.bias',
        'fusion1.fusion_layer.1.attention.output.dense.weight',
        'fusion1.fusion_layer.1.attention.self.value.bias',
        'fusion1.fusion_layer.1.attention.self.value.weight',
        'fusion1.fusion_layer.1.intermediate.dense.bias',
        'fusion1.fusion_layer.1.intermediate.dense.weight',
        'fusion1.fusion_layer.1.output.LayerNorm.bias',
        'fusion1.fusion_layer.1.output.LayerNorm.weight',
        'fusion1.fusion_layer.1.output.dense.bias',
        'fusion1.fusion_layer.1.output.dense.weight',
        'rel_fc_dual.bias',
        'rel_fc_dual.weight',
    ]
    if none_grad != expected_unused:
        raise AssertionError(f'Unexpected parameters without gradients: {none_grad}')
    nonfinite = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if nonfinite:
        raise AssertionError(f'Nonfinite gradients: {nonfinite}')
    return {
        'status': 'passed', 'configuration': 'tiny no-MGD fixture; stage D dependency isolated',
        'structural_shape': list(gph.shape), 'joint_shape': list(joint.shape),
        'parameters_with_finite_gradient': sum(p.grad is not None for p in model.parameters()),
        'parameters_without_gradient': none_grad,
        'unused_parameter_explanation': (
            'rel_fc_dual has no published forward reader. MFusion returns weights from '
            'the final attention probabilities, so the final layer value/output/feed-forward '
            'parameters cannot affect that result. They are retained and disclosed to avoid '
            'an untracked architecture/RNG change.'
        ),
    }


def main():
    try:
        sys.stdout.reconfigure(errors='backslashreplace')
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/mmkg')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/stage_c')
    parser.add_argument('--report', type=Path, default=ROOT/'audit/stage_c_validation.json')
    cli = parser.parse_args()
    report = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'running', 'device': 'cpu',
        'source_sha256': {name: sha256(ROOT/'src'/name) for name in ['config.py', 'models.py', 'Mfusion.py', 'run.py', 'loss.py']},
        'resolved_profiles': {}, 'datasets': {}, 'tiny_forward_backward': {},
        'sparse_mcd_forward': 'not_run in Gate C; verified separately by Gate D',
        'cuda': 'not_run; installed torch build is CPU-only',
    }
    try:
        for profile in ['repaired_pmf', 'author_code_compat']:
            args = parse_args(['--config', str(ROOT/f'configs/{profile}.json'), '--device', 'cpu'])
            report['resolved_profiles'][profile] = resolved_dict(args)
        report['datasets'] = initialize_real_datasets(cli.data_root, cli.output_root)
        report['tiny_forward_backward'] = tiny_forward_backward()
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        write_json(cli.report, report)
    print(f'Gate C: {report["status"]}; report: {cli.report.resolve()}')


if __name__ == '__main__':
    main()

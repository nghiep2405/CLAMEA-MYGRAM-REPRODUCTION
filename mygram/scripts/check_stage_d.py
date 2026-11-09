"""Gate D: inventory sparse graphs and run one real full-graph forward/backward."""
import argparse
import datetime
import gc
from pathlib import Path
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil
import torch

from src.Load import load_data_dual
from src.config import parse_args
from src.data import load_dataset
from src.data_registry import REGISTRY, read_manifest, resolve_data_paths, sha256, write_json
from src.models import MultiModalEncoderMrFusion


class RssSampler:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.wait(0.05):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop_event.set()
        self.thread.join()


def tiny_equivalence_suite():
    suite = unittest.defaultTestLoader.discover(
        str(ROOT/'tests'), pattern='test_sparse_graph.py'
    )
    count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('Tiny sparse/dense equivalence suite failed')
    return {
        'status': 'passed',
        'tests': count,
        'fixtures': [
            'duplicate triples', 'multiple relations on one edge', 'explicit reverse edge',
            'self triple', 'isolated node', 'one-node graph',
        ],
        'comparisons': ['RRGAT graph/reference', 'MCD output/gradient', 'RRGAT output/gradient'],
        'atol': 1e-6,
        'rtol': 1e-5,
        'dgl_cross_check': 'not_run; DGL is not installed and is not required by the sparse runtime',
    }


def graph_inventory(data_root):
    inventories = {}
    for dataset in REGISTRY:
        start = time.perf_counter()
        read_manifest(data_root, dataset)
        directory, _, _ = resolve_data_paths(data_root, dataset)
        graph = load_data_dual(directory, mcd_self_loops=2)
        if graph.entity_average.layout != torch.sparse_coo or graph.mcd_operator.layout != torch.sparse_coo:
            raise AssertionError(f'{dataset}: graph operator is not sparse COO')
        if graph.entity_average.shape != (graph.node_size, graph.node_size):
            raise AssertionError(f'{dataset}: invalid entity operator shape')
        if graph.mcd_operator.shape != (graph.node_size, graph.node_size):
            raise AssertionError(f'{dataset}: invalid MCD operator shape')
        inventories[dataset] = {
            **graph.diagnostics,
            'status': 'passed',
            'seconds': time.perf_counter() - start,
            'layouts': {
                'entity_average': str(graph.entity_average.layout),
                'mcd_operator': str(graph.mcd_operator.layout),
            },
        }
        print(
            f'{dataset}: N={graph.node_size}, RRGAT edges={graph.triple_size}, '
            f'MCD nnz={graph.mcd_operator._nnz()}', flush=True,
        )
        del graph
        gc.collect()
    return inventories


def full_graph_forward_backward(data_root, output_root, dataset):
    args = parse_args([
        '--config', str(ROOT/'configs/repaired_pmf.json'), '--device', 'cpu',
        '--data-root', str(data_root), '--output-root', str(output_root),
        '--dataset', dataset,
    ])
    torch.manual_seed(args.seed)
    directory, _, _ = resolve_data_paths(data_root, dataset)
    phase = {}
    with RssSampler() as sampler:
        phase['rss_before_bytes'] = psutil.Process().memory_info().rss
        start = time.perf_counter()
        data = load_dataset(data_root, dataset, seed=args.seed, rate=args.rate)
        graph = load_data_dual(directory, mcd_self_loops=args.mcd_self_loops)
        phase['load_seconds'] = time.perf_counter() - start
        phase['rss_after_load_bytes'] = psutil.Process().memory_info().rss

        image_dim = data.image_features.shape[1]
        image = torch.as_tensor(data.image_features, dtype=torch.float32)
        relation = torch.as_tensor(data.relation_features, dtype=torch.float32)
        attribute = torch.as_tensor(data.attribute_features, dtype=torch.float32)
        entity_ids = torch.arange(graph.node_size)
        left_ents, right_ents = data.ordered_ids
        del data
        gc.collect()

        start = time.perf_counter()
        model = MultiModalEncoderMrFusion(
            args=args,
            ent_num=graph.node_size,
            rel_size=graph.relation_size,
            triple_size=graph.triple_size,
            img_feature_dim=image_dim,
            adj_matrix=graph.edge_index,
            r_index=graph.relation_index,
            r_val=graph.relation_value,
            rel_matrix=torch.empty((2, 0), dtype=torch.long),
            ent_matrix=graph.entity_average,
            mcd_operator=graph.mcd_operator,
            char_feature_dim=None,
            use_project_head=False,
            left_ents=left_ents,
            right_ents=right_ents,
        )
        model.train()
        phase['model_init_seconds'] = time.perf_counter() - start

        graph_buffers = {
            name: value for name, value in model.named_buffers()
            if name in {'adj_list', 'r_index', 'r_val', 'ent_adj', 'mcd_operator'}
        }
        dense_square = [
            name for name, value in graph_buffers.items()
            if value.layout == torch.strided and tuple(value.shape) == (graph.node_size, graph.node_size)
        ]
        if dense_square:
            raise AssertionError(f'Dense N x N graph buffers detected: {dense_square}')

        start = time.perf_counter()
        outputs = model(entity_ids, image, relation, attribute)
        if outputs[1].shape != (graph.node_size, args.structural_output_dim):
            raise AssertionError(f'Unexpected structural shape: {outputs[1].shape}')
        if outputs[-1].shape != (graph.node_size, args.joint_output_dim):
            raise AssertionError(f'Unexpected joint shape: {outputs[-1].shape}')
        if not all(value is None or torch.isfinite(value).all() for value in outputs):
            raise AssertionError('Nonfinite full-graph output')
        phase['forward_seconds'] = time.perf_counter() - start
        phase['rss_after_forward_bytes'] = psutil.Process().memory_info().rss

        start = time.perf_counter()
        loss = outputs[-1].square().mean()
        loss.backward()
        phase['backward_seconds'] = time.perf_counter() - start
        phase['rss_after_backward_bytes'] = psutil.Process().memory_info().rss
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise AssertionError('Missing or nonfinite full-graph gradients')

    return {
        'status': 'passed',
        'dataset': dataset,
        'device': 'cpu',
        'nodes': graph.node_size,
        'rrgat_edges': graph.triple_size,
        'mcd_nnz': graph.mcd_operator._nnz(),
        'mcd_self_loops_per_node': args.mcd_self_loops,
        'graph_dense_n_by_n_buffers': dense_square,
        'structural_shape': list(outputs[1].shape),
        'joint_shape': list(outputs[-1].shape),
        'loss': float(loss.detach()),
        'parameters_with_gradient': len(gradients),
        'all_gradients_finite': True,
        'peak_process_rss_bytes': sampler.peak,
        'peak_process_rss_delta_bytes': sampler.peak - phase['rss_before_bytes'],
        'timings_and_rss': phase,
        'cuda_metrics': {
            'status': 'not_run',
            'reason': 'Installed torch build is CPU-only; CUDA was not exercised by Gate D.',
            'peak_allocated_bytes': None,
            'peak_reserved_bytes': None,
            'forward': None,
            'backward': None,
            'evaluation': None,
            'iterative_learning': None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/mmkg')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/stage_d')
    parser.add_argument('--full-dataset', choices=REGISTRY, default=None)
    parser.add_argument('--report', type=Path, default=ROOT/'audit/stage_d_validation.json')
    cli = parser.parse_args()
    report = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'running',
        'tolerances': {'atol': 1e-6, 'rtol': 1e-5},
        'source_sha256': {
            name: sha256(ROOT/name) for name in [
                'src/graph.py', 'src/Load.py', 'src/Mfusion.py', 'src/models.py',
                'src/run.py', 'tests/test_sparse_graph.py',
            ]
        },
        'tiny_equivalence': {},
        'graphs': {},
        'full_graph_forward_backward': {},
    }
    try:
        report['tiny_equivalence'] = tiny_equivalence_suite()
        report['graphs'] = graph_inventory(cli.data_root)
        selected = cli.full_dataset or min(
            report['graphs'], key=lambda key: report['graphs'][key]['rrgat_relation_contributions']
        )
        print(f'Full-graph forward/backward dataset: {selected}', flush=True)
        report['full_graph_forward_backward'] = full_graph_forward_backward(
            cli.data_root, cli.output_root, selected
        )
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        write_json(cli.report, report)
    print(f'Gate D: {report["status"]}; report: {cli.report.resolve()}', flush=True)


if __name__ == '__main__':
    main()

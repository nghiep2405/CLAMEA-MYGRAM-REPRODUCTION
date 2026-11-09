"""Layered Gate G validation and full-graph profiling for MyGram."""
import argparse
import datetime
import gc
import json
import math
from pathlib import Path
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))

import numpy as np
import psutil
import torch

from src.config import parse_args, resolved_dict
from src.data import load_dataset
from src.data_registry import REGISTRY, read_manifest, resolve_data_paths, sha256, write_json
from src.evaluation import evaluate_embeddings
from src.Load import load_data_dual
from src.loss import GRAM_Loss, InfoNCE_loss, alignment_objective
from src.models import MultiModalEncoderMrFusion
from test_stage_g import il_cycle_resume_smoke, tiny_training_resume_smoke


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


def run_test_patterns(patterns):
    suite = unittest.TestSuite()
    for pattern in patterns:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern))
    count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(f'Unit suite failed: {patterns}')
    return {'status': 'passed', 'tests': count, 'patterns': patterns}


def g0_parse_import_data(data_root):
    tests = run_test_patterns(['test_config_model.py', 'test_data.py'])
    profiles = {}
    for profile in ('repaired_pmf', 'author_code_compat'):
        args = parse_args([
            '--config', str(ROOT/f'configs/{profile}.json'), '--device', 'cpu',
            '--data-root', str(data_root), '--dataset', 'zh_en', '--print-config',
        ])
        profiles[profile] = {
            'config_sha256': resolved_dict(args)['config_sha256'],
            'device': args.device,
            'optimizer_reset_policy': args.optimizer_reset_policy,
        }
    datasets = {}
    for dataset in REGISTRY:
        started = time.perf_counter()
        manifest = read_manifest(data_root, dataset)
        rate = 0.2 if dataset in ('FBDB15K', 'FBYG15K') else 0.3
        data = load_dataset(
            data_root, dataset, seed=2023, rate=rate,
            rng=np.random.RandomState(2023),
        )
        datasets[dataset] = {
            'status': 'passed',
            'entities': len(data.ent2id),
            'triples': len(data.triples),
            'alignment_pairs': len(data.split['train_pairs']) + len(data.split['test_pairs']),
            'train_pairs': len(data.split['train_pairs']),
            'test_pairs': len(data.split['test_pairs']),
            'image_shape': list(data.image_features.shape),
            'relation_shape': list(data.relation_features.shape),
            'attribute_shape': list(data.attribute_features.shape),
            'split_sha256': data.split_metadata['split_sha256'],
            'archive_sha256': manifest['archive']['sha256'],
            'seconds': time.perf_counter() - started,
        }
        del data
        gc.collect()
    return {
        'status': 'passed',
        'criterion': 'imports/config/paths valid and five datasets match locked manifests',
        'unit_tests': tests,
        'profiles': profiles,
        'datasets': datasets,
    }


def g1_tiny_oracles():
    result = run_test_patterns([
        'test_sparse_graph.py', 'test_loss_optimizer_ablation.py',
        'test_evaluation_il_checkpoint.py',
    ])
    result['criterion'] = (
        'sparse output/gradient oracles, Gram edge cases, ablations, evaluator and checkpoint'
    )
    return result


def build_real_components(data_root, dataset):
    args = parse_args([
        '--config', str(ROOT/'configs/repaired_pmf.json'), '--device', 'cpu',
        '--data-root', str(data_root), '--dataset', dataset,
        '--output-root', str(ROOT/'outputs/stage_g'),
    ])
    torch.manual_seed(args.seed)
    data = load_dataset(
        data_root, dataset, seed=args.seed, rate=args.rate,
        rng=np.random.RandomState(args.seed),
    )
    directory, _, _ = resolve_data_paths(data_root, dataset)
    graph = load_data_dual(directory, mcd_self_loops=args.mcd_self_loops)
    image = torch.as_tensor(data.image_features, dtype=torch.float32)
    relation = torch.as_tensor(data.relation_features, dtype=torch.float32)
    attribute = torch.as_tensor(data.attribute_features, dtype=torch.float32)
    model = MultiModalEncoderMrFusion(
        args=args,
        ent_num=graph.node_size,
        rel_size=graph.relation_size,
        triple_size=graph.triple_size,
        img_feature_dim=image.shape[1],
        adj_matrix=graph.edge_index,
        r_index=graph.relation_index,
        r_val=graph.relation_value,
        rel_matrix=torch.empty((2, 0), dtype=torch.long),
        ent_matrix=graph.entity_average,
        mcd_operator=graph.mcd_operator,
        left_ents=data.ordered_ids[0],
        right_ents=data.ordered_ids[1],
    )
    return args, data, graph, model, (image, relation, attribute)


def _finite_metrics(metrics):
    return all(
        math.isfinite(float(value))
        for direction in metrics.values()
        for value in direction.values()
    )


def full_graph_profile(data_root, dataset, layer):
    if layer not in ('G3', 'G4'):
        raise ValueError(f'Unsupported full profile layer: {layer}')
    timings = {}
    process = psutil.Process()
    rss_before = process.memory_info().rss
    with RssSampler() as sampler:
        print(f'{layer} {dataset}: load/model init', flush=True)
        started = time.perf_counter()
        args, data, graph, model, features = build_real_components(data_root, dataset)
        timings['load_and_model_init_seconds'] = time.perf_counter() - started
        entity_ids = torch.arange(graph.node_size)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        all_train = torch.as_tensor(data.split['train_pairs'], dtype=torch.long)
        all_test = torch.as_tensor(data.split['test_pairs'], dtype=torch.long)
        if layer == 'G3':
            links = all_train[:min(8, len(all_train))]
            batches = [links]
            evaluation_rows = 64
        else:
            links = all_train
            batches = list(links.split(args.loss_batch_size))
            evaluation_rows = 128
        if not len(links):
            raise AssertionError(f'{dataset}: no training links available')
        del data
        gc.collect()

        model.train()
        optimizer.zero_grad()
        print(f'{layer} {dataset}: training forward', flush=True)
        started = time.perf_counter()
        outputs = model(entity_ids, *features)
        timings['train_forward_seconds'] = time.perf_counter() - started
        if not all(value is None or torch.isfinite(value).all() for value in outputs):
            raise AssertionError(f'{dataset}: nonfinite forward output')

        losses = []
        batch_diagnostics = []
        print(f'{layer} {dataset}: {len(batches)} loss/backward batch(es)', flush=True)
        started = time.perf_counter()
        for index, batch in enumerate(batches):
            loss, diagnostics = alignment_objective(
                outputs[-1], outputs[0], outputs[2], outputs[4], outputs[3], batch,
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
                raise AssertionError(f'{dataset}: nonfinite batch loss')
            loss.backward(retain_graph=index + 1 < len(batches))
            losses.append(float(loss.detach()))
            batch_diagnostics.append(diagnostics)
        timings['loss_and_backward_seconds'] = time.perf_counter() - started
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise AssertionError(f'{dataset}: missing/nonfinite gradients')
        grad_norm = sum(
            float(value.detach().float().square().sum()) for value in gradients
        ) ** 0.5
        gradient_parameter_count = len(gradients)
        del outputs, loss, gradients
        gc.collect()

        model.eval()
        print(f'{layer} {dataset}: evaluation forward', flush=True)
        started = time.perf_counter()
        with torch.no_grad():
            evaluated_outputs = model(entity_ids, *features)
        timings['evaluation_forward_seconds'] = time.perf_counter() - started
        test_pairs = all_test[:evaluation_rows]
        started = time.perf_counter()
        evaluation = evaluate_embeddings(
            evaluated_outputs[-1], test_pairs[:, 0], test_pairs[:, 1],
            csls_k=args.csls_k, block_size=min(64, len(test_pairs)),
        )
        timings['block_evaluation_seconds'] = time.perf_counter() - started
        if not _finite_metrics(evaluation['metrics']):
            raise AssertionError(f'{dataset}: nonfinite evaluation metric')
        prediction_scores = [
            score for direction in evaluation['predictions'].values()
            for row in direction for score in row['top_scores']
        ]
        if not prediction_scores or not all(math.isfinite(score) for score in prediction_scores):
            raise AssertionError(f'{dataset}: missing/nonfinite predictions')
        evaluation_metrics = evaluation['metrics']
        evaluation_protocol = evaluation['protocol']
        del evaluated_outputs, evaluation, prediction_scores
        gc.collect()

        print(f'{layer} {dataset}: optimizer step', flush=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    train_seconds = timings['train_forward_seconds'] + timings['loss_and_backward_seconds']
    return {
        'status': 'passed',
        'layer': layer,
        'dataset': dataset,
        'device': 'cpu',
        'profile': args.profile,
        'rate': args.rate,
        'nodes': graph.node_size,
        'rrgat_edges': graph.triple_size,
        'mcd_nnz': graph.mcd_operator._nnz(),
        'train_pairs_available': len(all_train),
        'train_pairs_exercised': len(links),
        'loss_batch_size': args.loss_batch_size,
        'batch_rows': [len(batch) for batch in batches],
        'one_full_graph_forward_per_training_epoch': True,
        'loss_sum_of_batch_means': sum(losses),
        'losses_finite': True,
        'grad_norm': grad_norm,
        'gradients_finite': True,
        'parameters_with_gradient': gradient_parameter_count,
        'optimizer_steps': 1,
        'evaluation_rows': len(test_pairs),
        'evaluation_metrics': evaluation_metrics,
        'prediction_scores_finite': True,
        'candidate_protocol': evaluation_protocol,
        'timings': timings,
        'measured_train_epoch_seconds': train_seconds,
        'peak_process_rss_bytes': sampler.peak,
        'peak_process_rss_delta_bytes': sampler.peak - rss_before,
        'cuda_peak_allocated_bytes': None,
        'cuda_status': 'not_run; installed torch build is CPU-only',
        'batch_diagnostics': batch_diagnostics,
    }


def base_report(data_root):
    return {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'passed',
        'device': 'cpu',
        'G0': g0_parse_import_data(data_root),
        'G1': g1_tiny_oracles(),
        'G2': tiny_training_resume_smoke(),
        'G5': il_cycle_resume_smoke(),
        'limitations': {
            'cuda': 'not_run; installed torch build is CPU-only',
            'research_metrics': 'none; all outputs in this report are validation/smoke',
        },
    }


def assemble_report(report_dir):
    def read(name):
        path = report_dir/name
        if not path.exists():
            return {'status': 'not_run', 'reason': f'missing {path.name}'}
        return json.loads(path.read_text(encoding='utf-8'))

    base = read('stage_g_base.json')
    g3 = {dataset: read(f'stage_g_g3_{dataset}.json') for dataset in REGISTRY}
    g4 = read('stage_g_g4_zh_en.json')
    required = [base.get('status'), *(item.get('status') for item in g3.values()), g4.get('status')]
    feasible_passed = all(value == 'passed' for value in required)
    g6 = {
        'status': 'blocked',
        'required_runs': [
            {'dataset': 'FBDB15K', 'rate': 0.2, 'epochs': 60},
            {'dataset': 'zh_en', 'rate': 0.3, 'epochs': 60},
        ],
        'reason': (
            'The locked environment has torch 2.7.1+cpu and no CUDA backend. '
            'The 60-epoch pilots were not silently replaced by reduced-model or reduced-budget runs.'
        ),
        'resume_condition': 'install and validate a separate CUDA torch environment, then run both exact pilots',
    }
    return {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'partial_blocked' if feasible_passed else 'failed_or_incomplete',
        'implementation_status': 'G0-G5 passed' if feasible_passed else 'one or more G0-G5 layers incomplete',
        'G0': base.get('G0', base),
        'G1': base.get('G1', {'status': 'not_run'}),
        'G2': base.get('G2', {'status': 'not_run'}),
        'G3': g3,
        'G4': g4,
        'G5': base.get('G5', {'status': 'not_run'}),
        'G6': g6,
        'source_sha256': {
            name: sha256(ROOT/name) for name in [
                'scripts/check_stage_g.py', 'tests/test_stage_g.py',
                'configs/stage_g_smoke.json', 'src/run.py', 'src/evaluation.py',
                'src/iterative_learning.py', 'src/checkpoint.py',
            ]
        },
        'classification': {
            'G0_G2': 'validation',
            'G3_G4': 'full-graph smoke/profile, not research metrics',
            'G5': 'shortened-schedule fixture, not research metrics',
            'G6': 'required pilot, blocked and not substituted',
        },
    }


def main():
    try:
        sys.stdout.reconfigure(errors='backslashreplace')
        sys.stderr.reconfigure(errors='backslashreplace')
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', choices=('base', 'g3', 'g4', 'assemble'), required=True)
    parser.add_argument('--dataset', choices=REGISTRY)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/mmkg')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'audit')
    parser.add_argument('--report', type=Path)
    cli = parser.parse_args()
    cli.report_dir.mkdir(parents=True, exist_ok=True)
    if cli.layer == 'base':
        default_path = cli.report_dir/'stage_g_base.json'
        action = lambda: base_report(cli.data_root)
    elif cli.layer == 'g3':
        if cli.dataset is None:
            parser.error('--dataset is required for g3')
        default_path = cli.report_dir/f'stage_g_g3_{cli.dataset}.json'
        action = lambda: full_graph_profile(cli.data_root, cli.dataset, 'G3')
    elif cli.layer == 'g4':
        dataset = cli.dataset or 'zh_en'
        default_path = cli.report_dir/f'stage_g_g4_{dataset}.json'
        action = lambda: full_graph_profile(cli.data_root, dataset, 'G4')
    else:
        default_path = cli.report_dir/'stage_g_validation.json'
        action = lambda: assemble_report(cli.report_dir)
    report_path = cli.report or default_path
    try:
        result = action()
    except Exception as exc:
        result = {
            'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status': 'failed', 'layer': cli.layer, 'dataset': cli.dataset,
            'error': repr(exc),
        }
        write_json(report_path, result)
        raise
    write_json(report_path, result)
    print(f'{cli.layer}: {result["status"]}; report: {report_path.resolve()}', flush=True)


if __name__ == '__main__':
    main()

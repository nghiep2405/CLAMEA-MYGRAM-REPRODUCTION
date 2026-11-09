"""Gate F: validate block evaluation, ordered IL, and checkpoint/resume."""
import argparse
import datetime
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_dataset
from src.data_registry import sha256, write_json
from src.evaluation import evaluate_embeddings


def gate_f_test_suite():
    suite = unittest.defaultTestLoader.discover(
        str(ROOT/'tests'), pattern='test_evaluation_il_checkpoint.py'
    )
    count = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('Gate F unit suite failed')
    return {
        'status': 'passed',
        'tests': count,
        'covered': [
            'dense/block rank equivalence with CSLS on/off',
            'deterministic tie rule and duplicate candidate positions',
            'ordered block mutual nearest neighbors and empty pools',
            'IL candidate start/intersection/add schedule',
            'fixed test preservation and post-selection label diagnostics',
            'atomic artifact/checkpoint contract and hash rejection',
            'Python/NumPy/torch RNG restoration',
            'bitwise continuous/resume at a normal CPU epoch',
            'bitwise continuous/resume across epoch 50 for both optimizer profiles',
            'mid-cycle IL new_links serialization',
        ],
    }


def dense_reference(embeddings, left_ids, right_ids, csls_k):
    normalized = F.normalize(embeddings, dim=1)
    left = normalized[left_ids]
    right = normalized[right_ids]
    distance = (
        left.square().sum(1, keepdim=True) + right.square().sum(1)[None, :]
        - 2 * left @ right.T
    ).clamp_min(0)
    scores = 1 - distance
    if csls_k is not None:
        k_eff = min(csls_k, len(left_ids))
        scores = (
            2 * scores
            - torch.topk(scores, k_eff, dim=1).values.mean(1)[:, None]
            - torch.topk(scores.T, k_eff, dim=1).values.mean(1)[None, :]
        )

    def ranks(matrix):
        positions = torch.arange(len(left_ids))
        ground_truth = matrix[positions, positions]
        return 1 + (matrix > ground_truth[:, None]).sum(1) + (
            (matrix == ground_truth[:, None]) & (positions[None, :] < positions[:, None])
        ).sum(1)
    return ranks(scores), ranks(scores.T)


def real_subset_check(data_root, dataset='zh_en', rows=96):
    start = time.perf_counter()
    data = load_dataset(
        data_root, dataset, seed=2023, rate=.3,
        rng=np.random.RandomState(2023),
    )
    pairs = torch.as_tensor(data.split['test_pairs'][:rows], dtype=torch.long)
    embeddings = torch.as_tensor(data.image_features, dtype=torch.float32)
    results = {}
    for csls_k in (None, 3):
        actual = evaluate_embeddings(
            embeddings, pairs[:, 0], pairs[:, 1],
            csls_k=csls_k, block_size=13,
        )
        expected_l2r, expected_r2l = dense_reference(
            embeddings, pairs[:, 0], pairs[:, 1], csls_k
        )
        if not torch.equal(actual['ranks']['l2r'], expected_l2r):
            raise AssertionError(f'{dataset}: real-subset l2r ranks differ')
        if not torch.equal(actual['ranks']['r2l'], expected_r2l):
            raise AssertionError(f'{dataset}: real-subset r2l ranks differ')
        results['csls' if csls_k else 'no_csls'] = {
            'status': 'passed',
            'metrics': actual['metrics'],
            'protocol': actual['protocol'],
            'dense_rank_match': True,
        }
    return {
        'status': 'passed',
        'dataset': dataset,
        'rows': len(pairs),
        'embedding_source': 'validated real PMF image features',
        'results': results,
        'seconds': time.perf_counter() - start,
    }


def main():
    try:
        sys.stdout.reconfigure(errors='backslashreplace')
        sys.stderr.reconfigure(errors='backslashreplace')
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/mmkg')
    parser.add_argument('--report', type=Path, default=ROOT/'audit/stage_f_validation.json')
    cli = parser.parse_args()
    report = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'running',
        'device': 'cpu',
        'source_sha256': {
            name: sha256(ROOT/name) for name in [
                'src/evaluation.py', 'src/iterative_learning.py', 'src/checkpoint.py',
                'src/run.py', 'src/config.py',
                'tests/test_evaluation_il_checkpoint.py',
            ]
        },
        'unit_oracles': {},
        'real_subset_evaluation': {},
        'limitations': {
            'cuda': 'not_run; installed torch build is CPU-only',
            'full_dataset_evaluation': 'not_run; Gate G',
            'research_metrics': 'not_run',
        },
    }
    try:
        report['unit_oracles'] = gate_f_test_suite()
        report['real_subset_evaluation'] = real_subset_check(cli.data_root)
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        write_json(cli.report, report)
    print(f'Gate F: {report["status"]}; report: {cli.report.resolve()}', flush=True)


if __name__ == '__main__':
    main()

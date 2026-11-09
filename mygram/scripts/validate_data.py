"""Validate all real PMF loaders, audited statistics and fixed ordered splits on CPU."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.data import load_dataset, make_split, save_split
from src.data_registry import REGISTRY, sha256, write_json


def validate(args):
    baseline = json.loads((ROOT/'audit/processed_dataset_report.json').read_text(encoding='utf-8'))
    provenance = json.loads((ROOT/'audit/crosskg_provenance_report.json').read_text(encoding='utf-8'))
    report = {'status': 'running', 'datasets': {}, 'seed': args.seed, 'profile': args.profile,
              'device': 'cpu', 'validator_sha256': sha256(__file__), 'model_smoke': 'not_run',
              'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'loader_sha256': {name: sha256(ROOT/'src'/name) for name in ['data.py', 'data_registry.py', 'Load.py', 'utils.py']}}
    keys = list(REGISTRY) if args.all else [args.dataset]
    for key in keys:
        start = time.perf_counter()
        try:
            rate = args.rate if args.rate is not None else (.2 if key in ('FBDB15K', 'FBYG15K') else .3)
            data = load_dataset(args.data_root, key, seed=args.seed, rate=rate, file_dir=args.file_dir)
            stats, old = data.statistics, baseline[key]
            for field in ['entities', 'total_entities', 'line_counts', 'link_count', 'duplicate_links', 'unique_left_links', 'unique_right_links']:
                if stats[field] != old[field]:
                    raise ValueError(f'{key}: audit mismatch for {field}')
            for field in ['attribute_shape', 'relation_shape', 'zero_attribute_rows', 'zero_relation_rows']:
                if stats[field] != old['upstream_text_loader_smoke'][field]:
                    raise ValueError(f'{key}: upstream BoW mismatch for {field}')
            if stats['image_entries'] != old['image_features']['entries'] or stats['image_coverage_per_graph'] != old['image_features']['coverage_per_graph']:
                raise ValueError(f'{key}: image coverage mismatch')
            # Independent legacy shuffle over tuple list, not reusing make_split's index implementation.
            legacy = list(data.pairs)
            np.random.RandomState(args.seed).shuffle(legacy)
            rates = sorted(set([rate] + ([.05, .1, .15, .2, .25, .3, .5, .8] if key in provenance else [.3])))
            stats['saved_splits'] = {}
            for fraction in rates:
                arrays, meta = make_split(data.pairs, fraction, np.random.RandomState(args.seed))
                cut = int(len(legacy)*fraction)
                if not np.array_equal(arrays['train_pairs'], legacy[:cut]) or not np.array_equal(arrays['test_pairs'], legacy[cut:]):
                    raise ValueError('Ordered split differs from upstream list shuffle')
                if args.seed == 2023 and key in provenance and str(fraction) in provenance[key]['split_checks_seed2023']:
                    expected = provenance[key]['split_checks_seed2023'][str(fraction)]
                    if any(meta[field] != value for field, value in expected.items()):
                        raise ValueError('Split differs from provenance audit')
                stats['saved_splits'][str(fraction)] = {'path': save_split(args.output_root, key, args.seed, fraction, arrays, meta), **meta}
            masks_path = Path(args.output_root) / 'coverage' / f'{key}.npz'
            masks_path.parent.mkdir(parents=True, exist_ok=True)
            if masks_path.exists():
                with np.load(masks_path, allow_pickle=False) as existing:
                    if any(not np.array_equal(existing[k], data.image_masks[k]) for k in ['original_mask', 'imputed_mask']):
                        raise FileExistsError(f'Coverage conflict: {masks_path}')
            else:
                np.savez_compressed(masks_path, **{k: data.image_masks[k] for k in ['original_mask', 'imputed_mask']})
            stats['coverage_file'] = str(masks_path.resolve())
            stats['audit_statistics_match'] = True
            stats['ordered_split_upstream_match'] = True
            stats['seconds'] = time.perf_counter()-start
            report['datasets'][key] = stats
            # Only validated datasets get a passed status in the prepared manifest.
            manifest = json.loads((args.data_root/'data_manifest.json').read_text(encoding='utf-8'))
            manifest['datasets'][key] = stats
            manifest['validation_code_sha256'] = report['loader_sha256']
            write_json(args.data_root/'data_manifest.json', manifest)
            del data
            gc.collect()
            print(f'{key}: passed, {stats["image_shape"]}, {stats["seconds"]:.1f}s', flush=True)
        except Exception as exc:
            report['datasets'][key] = {'status': 'failed', 'error': repr(exc)}
            manifest_path = args.data_root/'data_manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                manifest.setdefault('datasets', {})[key] = report['datasets'][key]
                write_json(manifest_path, manifest)
            print(f'{key}: FAILED: {exc}', flush=True)
        write_json(args.report, report)
    report['status'] = 'passed' if all(v['status']=='passed' for v in report['datasets'].values()) else 'failed'
    write_json(args.report, report)
    return 0 if report['status'] == 'passed' else 1


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true')
    group.add_argument('--dataset', choices=REGISTRY)
    parser.add_argument('--file_dir', help='Deprecated exact directory alias; must agree with registry')
    parser.add_argument('--device', choices=['cpu'], default='cpu', help='Data validation is CPU only; no model is run')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/data_validation')
    parser.add_argument('--profile', choices=['repaired_pmf', 'author_code_compat'], default='repaired_pmf', help='Data/split policy identical in both profiles')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--rate', type=float)
    parser.add_argument('--report', type=Path, default=ROOT/'audit/stage_b_validation.json')
    args = parser.parse_args()
    if args.all and args.file_dir:
        parser.error('--file_dir cannot be combined with --all')
    return validate(args)


if __name__ == '__main__':
    raise SystemExit(main())

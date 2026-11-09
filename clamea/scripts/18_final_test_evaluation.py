# CELL 17 — FINAL TEST EVALUATION (9 FROZEN BEST CHECKPOINTS)
# Paste this entire script into ONE Molab/Marimo Python cell.
# Prerequisites: Cell16.1 PASS; `fs_drive` connection; frozen dataset present.
# This does not train or select checkpoints using Test.

def cell17_final_test():
    import csv
    import hashlib
    import json
    import math
    import os
    import statistics
    import subprocess
    import sys
    import time
    import zipfile
    from pathlib import Path

    ID = 'cell15_20261025_150915_b52701'
    AUDIT_ID = 'audit_20261027_071659'  # Independently verified Cell16.1 PASS
    HOME = Path('/tmp/clamea_paper_method_molab')
    P15 = HOME / 'final_protocols' / ID
    P16 = HOME / 'final_training' / f'cell16_stage5_{ID}'
    DATA = HOME / 'datasets/CLAMEA_FBDB15K_protocolB'
    DROOT = 'paper_method_runs/paper_method_20261023_160901_s66/diagnostics'
    D15 = f'{DROOT}/{ID}'
    D16 = f'{DROOT}/cell16_stage5_{ID}'
    D17 = f'{D16}/final_test_evaluation'
    OUT = P16 / 'final_test_evaluation'
    OUT.mkdir(parents=True, exist_ok=True)

    # Locked before looking at any Test results.
    TOP_K = [1, 5, 10]
    PRIMARY = 'raw_no_csls'
    INCLUDE_CSLS10 = True  # Paper's official test() also reports CSLS as a second metric.
    TEST_THREADS = 1  # Same rank calculation, no multiprocessing / large matrix copies.
    KEEP_DOWNLOADED_MODELS = False  # Avoid ~5 GB of retained local downloads.

    try:
        drive = fs_drive
    except NameError as exc:
        raise RuntimeError('Missing fs_drive. Run the Drive connection cell first.') from exc

    def sha256(p):
        h = hashlib.sha256()
        with Path(p).open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    def read_json(p):
        return json.loads(Path(p).read_text(encoding='utf-8'))

    def write_json(p, value):
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + '.tmp')
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, p)

    def fetch(remote, local, expected=None):
        local = Path(local)
        if local.exists() and (expected is None or sha256(local) == expected):
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + '.part')
        if tmp.exists():
            tmp.unlink()
        try:
            drive.get_file(remote, str(tmp))
            assert tmp.is_file(), f'Drive download failed: {remote}'
            if expected:
                assert sha256(tmp) == expected, f'SHA256 mismatch: {remote}'
            os.replace(tmp, local)
        finally:
            if tmp.exists():
                tmp.unlink()
        return local

    def upload(local, remote):
        drive.makedirs(remote.rsplit('/', 1)[0], exist_ok=True)
        for attempt in range(1, 4):
            try:
                drive.put_file(str(local), remote)
                assert drive.exists(remote)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * attempt)

    def fail_if_mismatch(x, msg):
        if not x:
            raise AssertionError(msg)

    def check_result(res, run, completed):
        fail_if_mismatch(res.get('schema') == 'CLAMEA_CELL17_FINAL_TEST_V1', 'Result schema mismatch')
        fail_if_mismatch(res.get('run_id') == run['run_id'], 'Wrong run_id')
        fail_if_mismatch(res.get('arm') == run['arm'] and int(res.get('seed', -1)) == run['seed'], 'Wrong arm/seed')
        fail_if_mismatch(res.get('best_model_sha256') == completed['model_best_sha256'], 'Checkpoint SHA differs')
        fail_if_mismatch(res.get('best_epoch') == completed['best_epoch'], 'Best epoch differs')
        fail_if_mismatch(res.get('primary_metric') == PRIMARY and res.get('top_k') == TOP_K, 'Evaluation protocol changed')
        fail_if_mismatch(res.get('csls10_enabled') is INCLUDE_CSLS10, 'CSLS protocol changed')
        fail_if_mismatch(res.get('test_prediction_seed') == int(run['seed']), 'Test prediction RNG seed changed')
        fail_if_mismatch(res.get('test_selection_used') is False, 'Test selection policy violated')
        fail_if_mismatch(res.get('test_count', 0) > 0, 'Empty Test set')
        for key in ('hits_at_1_pct', 'hits_at_5_pct', 'hits_at_10_pct'):
            fail_if_mismatch(math.isfinite(float(res['raw_no_csls'][key])) and 0 <= float(res['raw_no_csls'][key]) <= 100,
                             f'Invalid raw {key}')
        fail_if_mismatch(0 <= float(res['raw_no_csls']['mrr']) <= 1, 'Invalid raw MRR')
        if INCLUDE_CSLS10:
            fail_if_mismatch(isinstance(res.get('csls10'), dict), 'CSLS result missing')
            fail_if_mismatch(0 <= float(res['csls10']['mrr']) <= 1, 'Invalid CSLS MRR')

    print('=' * 74)
    print('CELL 17 — FINAL TEST | FROZEN BEST VALIDATION CHECKPOINTS')
    print('[METRIC] primary=RAW (CSLS=0) | secondary=CSLS-10' if INCLUDE_CSLS10
          else '[METRIC] primary=RAW (CSLS=0)')
    print('[POLICY] Test is NEVER used for checkpoint selection or tuning')
    print('=' * 74, flush=True)

    audit_p = fetch(f'{D16}/audits/{AUDIT_ID}/audit_report.json',
                    P16 / 'audits' / AUDIT_ID / 'audit_report.json')
    audit = read_json(audit_p)
    fail_if_mismatch(audit['status'] == 'PASS' and audit['passed_runs'] == 9 and audit['errors'] == []
                     and audit['test_evaluated'] is False, 'Cell16.1 audit must PASS 9/9')
    manifest = read_json(fetch(f'{D15}/stage5_manifest.json', P15 / 'stage5_manifest.json'))
    matrix = read_json(fetch(f'{D15}/run_matrix.json', P15 / 'run_matrix.json'))
    lock = read_json(fetch(f'{D15}/dataset_lock.json', P15 / 'dataset_lock.json'))
    fail_if_mismatch(len(matrix) == 9 and len(audit['runs']) == 9, 'Wrong run count')
    fail_if_mismatch({r['run_id'] for r in matrix} == {r['run_id'] for r in audit['runs']},
                     'Audit-matrix identity mismatch')
    fail_if_mismatch(lock['file_count'] == 16, 'Dataset lock mismatch')
    for item in lock['files']:
        p = DATA / item['path']
        fail_if_mismatch(p.is_file() and p.stat().st_size == item['bytes'],
                         f'Missing/changed dataset file: {p}')
    print('[C17 PREFLIGHT] Audit PASS (9/9), dataset file sizes PASS (16/16)')

    archive_sha = fetch(f'{D15}/source_archive_sha256.txt',
                        P15 / 'source_archive_sha256.txt').read_text().split()[0]
    sourcezip = fetch(f'{D15}/frozen_sources.zip', P15 / 'frozen_sources.zip', archive_sha)
    frozen = P16 / 'frozen'
    sources = frozen / 'sources'
    if not sources.is_dir():
        frozen.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(sourcezip) as z:
            fail_if_mismatch(z.testzip() is None, 'Frozen source archive is corrupted')
            for ent in z.infolist():
                dst = (frozen / ent.filename).resolve()
                fail_if_mismatch(dst == frozen.resolve() or frozen.resolve() in dst.parents,
                                 'Unsafe frozen ZIP entry')
                fail_if_mismatch(((ent.external_attr >> 16) & 0o170000) != 0o120000,
                                 'Unsafe ZIP symlink')
            z.extractall(frozen)
    for src, files in manifest['source_hashes'].items():
        for filename, expected in files.items():
            p = sources / src / filename
            fail_if_mismatch(p.is_file() and sha256(p) == expected,
                             f'Frozen source mismatch: {p}')
    print('[C17 PREFLIGHT] Frozen source SHA256 PASS', flush=True)

    # Run the actual frozen CLAMEA inference/evaluation in an isolated subprocess.
    # The sample generated by stochastic VAE is reproducible from a locked seed.
    worker = OUT / 'cell17_worker.py'
    worker_code = r'''
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch

job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
source = Path(job['source']).resolve()
sys.path.insert(0, str(source))
os.chdir(source)

# Important: kgs.py imports main.set_seed. The order prevents circular import.
import kgs
import main as official
from kgs import KGs

assert torch.cuda.is_available(), 'Molab GPU is required for this evaluation'
assert os.environ['PYTHONHASHSEED'] == '0'
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('high')

args = official.ARGs(job['config'])
assert args.eval_metric == 'inner' and args.eval_norm is True
assert list(args.top_k) == [1, 5, 10]
assert int(args.csls) == 10
assert str(args.dataset_division) == '721'
assert int(args.random_seed) == int(job['seed'])
assert int(args.visual_dim) == 4096

# Model initialization and KG reading follow Cell16's exact source/config.
random.seed(job['seed'])
np.random.seed(job['seed'])
torch.manual_seed(job['seed'])
torch.cuda.manual_seed_all(job['seed'])
kgs_obj = KGs(args.training_data, args.dataset_division,
              args.random_seed, ordered=True, modality='ria')
model = official.CLAMEA(kgs_obj, args).cuda()
state = torch.load(job['model_path'], map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=True)
del state
model.eval()

# The frozen CLAMEA VAE can sample in model.eval(); use one fixed draw
# per run (same draw for ON/OFF/Preserve Gold with the same seed).
random.seed(job['seed'])
np.random.seed(job['seed'])
torch.manual_seed(job['seed'])
torch.cuda.manual_seed_all(job['seed'])

with torch.no_grad():
    e1 = torch.tensor(kgs_obj.test_entities1, dtype=torch.long, device='cuda')
    e2 = torch.tensor(kgs_obj.test_entities2, dtype=torch.long, device='cuda')
    at1 = torch.tensor([kgs_obj.attr_mask[x] for x in kgs_obj.test_entities1],
                       dtype=torch.uint8, device='cuda')
    at2 = torch.tensor([kgs_obj.attr_mask[x] for x in kgs_obj.test_entities2],
                       dtype=torch.uint8, device='cuda')
    im1 = torch.tensor([kgs_obj.image_mask[x] for x in kgs_obj.test_entities1],
                       dtype=torch.uint8, device='cuda')
    im2 = torch.tensor([kgs_obj.image_mask[x] for x in kgs_obj.test_entities2],
                       dtype=torch.uint8, device='cuda')
    embeddings1, embeddings2, *_ = model.predict(e1, e2, at1, im1, at2, im2)

assert isinstance(embeddings1, np.ndarray) and isinstance(embeddings2, np.ndarray)
assert embeddings1.shape[0] == embeddings2.shape[0] == len(kgs_obj.test_links)
assert len(kgs_obj.test_links) == int(job['expected_test_n'])
assert np.isfinite(embeddings1).all() and np.isfinite(embeddings2).all()
print('[C17 INFERENCE] test_pairs={} embed_dim={}'.format(
      len(kgs_obj.test_links), embeddings1.shape[1]), flush=True)

def metrics(csls_k):
    _, hits, mr, mrr = official.greedy_alignment(
        embeddings1, embeddings2, job['top_k'], job['test_threads'],
        metric=args.eval_metric, normalize=args.eval_norm,
        csls_k=csls_k, accurate=True)
    values = {
        'hits_at_1_pct': float(hits[0]),
        'hits_at_5_pct': float(hits[1]),
        'hits_at_10_pct': float(hits[2]),
        'mr': float(mr),
        'mrr': float(mrr),
    }
    assert all(np.isfinite(float(v)) for v in values.values())
    return values

raw = metrics(0)
print('[C17 RAW] Hits@1={:.3f}% Hits@5={:.3f}% Hits@10={:.3f}% MRR={:.6f}'.format(
      raw['hits_at_1_pct'], raw['hits_at_5_pct'], raw['hits_at_10_pct'], raw['mrr']),
      flush=True)
csls = metrics(10) if job['include_csls10'] else None
if csls:
    print('[C17 CSLS10] Hits@1={:.3f}% Hits@5={:.3f}% Hits@10={:.3f}% MRR={:.6f}'.format(
          csls['hits_at_1_pct'], csls['hits_at_5_pct'], csls['hits_at_10_pct'], csls['mrr']),
          flush=True)

result = {
    'schema': 'CLAMEA_CELL17_FINAL_TEST_V1',
    'run_id': job['run_id'], 'arm': job['arm'], 'seed': job['seed'],
    'best_epoch': job['best_epoch'], 'best_val_mrr': job['best_val_mrr'],
    'best_model_sha256': job['model_sha256'],
    'test_prediction_seed': job['seed'],
    'test_count': len(kgs_obj.test_links),
    'top_k': job['top_k'],
    'primary_metric': 'raw_no_csls',
    'csls10_enabled': bool(job['include_csls10']),
    'raw_no_csls': raw, 'csls10': csls,
    'source': job['source_kind'],
    'eval_metric': args.eval_metric,
    'eval_norm': args.eval_norm,
    'eval_threads': job['test_threads'],
    'test_selection_used': False,
    'protocol_notes': ('Test is evaluated from the best Validation-selected weights only; '
                       'VAE sampling seeded once per run. CSLS secondary uses the same embeddings.'),
}
out = Path(job['result_path'])
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_name(out.name + '.tmp')
tmp.write_text(json.dumps(result, indent=2), encoding='utf-8')
os.replace(tmp, out)
print('[C17 COMPLETE]', job['run_id'], str(out), flush=True)
'''.lstrip()
    compile(worker_code, str(worker), 'exec')
    worker.write_text(worker_code, encoding='utf-8')

    # This protocol uses the 8,993 ordered Test links, not the 1,284 Validation links.
    expected_test_n = 8993
    fail_if_mismatch('top_k' in matrix[0]['config'], 'Missing evaluation config')
    results = []
    for index, run in enumerate(matrix, 1):
        rid = run['run_id']
        run_dir = P16 / 'runs' / rid
        remote_run = f'{D16}/runs/{rid}'
        completed = read_json(fetch(f'{remote_run}/completed.json', run_dir / 'completed.json'))
        fail_if_mismatch(completed['status'] == 'COMPLETED_TRAINING_TEST_NOT_EVALUATED' and
                         completed['test_evaluated'] is False, 'Unexpected training status: ' + rid)
        fail_if_mismatch(completed['source_main_sha256'] ==
                         manifest['source_hashes'][run['source']]['main.py'], 'Source mismatch: ' + rid)
        config_hash = hashlib.sha256(json.dumps(run['config'], sort_keys=True).encode()).hexdigest()
        fail_if_mismatch(completed['config_sha256'] == config_hash, 'Config hash mismatch: ' + rid)

        local_result = OUT / 'runs' / rid / 'test_metrics.json'
        remote_result = f'{D17}/runs/{rid}/test_metrics.json'
        if not local_result.is_file() and drive.exists(remote_result):
            fetch(remote_result, local_result)
        if local_result.is_file():
            try:
                previous = read_json(local_result)
                check_result(previous, run, completed)
                # Ensure remote result is available, even after partial upload failure.
                if not drive.exists(remote_result):
                    upload(local_result, remote_result)
                print(f'[C17 SKIP TESTED {index}/9] {rid} | RAW MRR={previous[PRIMARY]["mrr"]:.6f}', flush=True)
                results.append(previous)
                continue
            except Exception as exc:
                raise RuntimeError(f'Existing Test result for {rid} has a protocol conflict. '
                                   f'DO NOT silently recompute Test: {exc}') from exc

        source = sources / run['source']
        weights = run_dir / 'model_best.pkl'
        reuse = weights.is_file() and sha256(weights) == completed['model_best_sha256']
        downloaded_for_test = not reuse
        if downloaded_for_test:
            weights = OUT / '_temporary_model_best.pkl'
        try:
            fetch(f'{remote_run}/model_best.pkl', weights, completed['model_best_sha256'])
            local_result.parent.mkdir(parents=True, exist_ok=True)
            job = {
                'run_id': rid, 'arm': run['arm'], 'seed': run['seed'],
                'source_kind': run['source'], 'source': str(source.resolve()),
                'model_path': str(weights.resolve()),
                'model_sha256': completed['model_best_sha256'],
                'best_epoch': completed['best_epoch'],
                'best_val_mrr': completed['best_val_mrr'],
                'config': dict(run['config']),
                'top_k': TOP_K, 'test_threads': TEST_THREADS,
                'include_csls10': INCLUDE_CSLS10,
                'expected_test_n': expected_test_n,
                'result_path': str(local_result.resolve()),
            }
            job['config']['training_data'] = str(DATA.resolve()) + '/'
            jobfile = local_result.parent / 'test_job.json'
            write_json(jobfile, job)
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = '0'
            env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            env['WANDB_MODE'] = 'disabled'
            env['TOKENIZERS_PARALLELISM'] = 'false'
            for key in ('CELL13_GRAPH_MODE', 'CELL13_PSEUDO_LOSS', 'CELL13_SAVE_BEST',
                        'CELL13_REF_E70_MRR', 'CELL13_REF_E70_H1',
                        'CELL14_DYNAMIC_STEPS', 'CLAMEA_RESUME_STATE',
                        'CELL16_PREVIOUS_HISTORY'):
                env.pop(key, None)
            env.update({str(k): str(v) for k, v in run['env'].items()})
            print(f'[C17 EVAL {index}/9] {rid} best_epoch={completed["best_epoch"]}', flush=True)
            log = local_result.parent / 'test.log'
            with log.open('w', encoding='utf-8') as f:
                proc = subprocess.Popen([sys.executable, '-u', str(worker), str(jobfile)],
                                        cwd=str(source), env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    f.write(line)
                    f.flush()
                    if ('[C17 INFERENCE]' in line or '[C17 RAW]' in line or
                            '[C17 CSLS10]' in line or '[C17 COMPLETE]' in line):
                        print(f'  {line.rstrip()}', flush=True)
                ret = proc.wait()
            if ret != 0:
                last = '\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-25:])
                raise RuntimeError(f'{rid} Test evaluation failed (exit={ret}). Log: {log}\n{last}')
            fail_if_mismatch(local_result.is_file(), f'Worker did not save Test result: {rid}')
            res = read_json(local_result)
            check_result(res, run, completed)
            upload(local_result, remote_result)
            upload(log, f'{D17}/runs/{rid}/test.log')
            print(f'[C17 BACKUP {index}/9] {rid} | RAW MRR={res[PRIMARY]["mrr"]:.6f}', flush=True)
            results.append(res)
        finally:
            if downloaded_for_test and weights.exists():
                weights.unlink()
            if downloaded_for_test and not KEEP_DOWNLOADED_MODELS:
                pass

    fail_if_mismatch(len(results) == 9, 'All 9 Test experiments must complete')
    fail_if_mismatch(len({r['run_id'] for r in results}) == 9, 'Duplicate Test result')
    aggregate = {}
    for arm in sorted({r['arm'] for r in results}):
        rr = [r for r in results if r['arm'] == arm]
        fail_if_mismatch(len(rr) == 3, f'Missing seed in {arm}')
        agg = {}
        for view in ([PRIMARY, 'csls10'] if INCLUDE_CSLS10 else [PRIMARY]):
            agg[view] = {}
            for key in ('hits_at_1_pct', 'hits_at_5_pct', 'hits_at_10_pct', 'mr', 'mrr'):
                vals = [float(r[view][key]) for r in rr]
                agg[view][key] = {'mean': statistics.mean(vals),
                                  'std_sample': statistics.stdev(vals),
                                  'values_by_seed': {str(r['seed']): r[view][key] for r in rr}}
        aggregate[arm] = agg

    summary = {
        'schema': 'CLAMEA_CELL17_FINAL_TEST_SUMMARY_V1',
        'cell15': ID, 'cell16_audit': AUDIT_ID, 'status': 'PASS',
        'completed_test_runs': 9, 'test_selection_used': False,
        'primary_metric': PRIMARY, 'csls10_enabled': INCLUDE_CSLS10,
        'results': results, 'aggregate': aggregate,
        'limitations': [
            'This is a Protocol B / VGG16 4096d method-level reproduction, not exact paper-data reproduction.',
            'Best checkpoint selected using Validation only; Test never used to tune or select.',
            'CLAMEA VAE stochastic sampling during evaluation is fixed using the run seed for this final Test draw.',
            'Metrics use frozen CLAMEA greedy_alignment and ordered Test links; Test queries rank all Test candidates.',
        ],
    }
    summary_path = OUT / 'final_test_summary.json'
    write_json(summary_path, summary)
    csv_path = OUT / 'final_test_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'arm', 'seed', 'best_epoch', 'best_val_mrr',
                         'raw_hits1_pct', 'raw_hits5_pct', 'raw_hits10_pct', 'raw_mrr',
                         'csls10_hits1_pct', 'csls10_hits5_pct', 'csls10_hits10_pct', 'csls10_mrr'])
        for r in results:
            raw, csls = r[PRIMARY], r.get('csls10') or {}
            writer.writerow([r['run_id'], r['arm'], r['seed'], r['best_epoch'], r['best_val_mrr'],
                             raw['hits_at_1_pct'], raw['hits_at_5_pct'], raw['hits_at_10_pct'], raw['mrr'],
                             csls.get('hits_at_1_pct'), csls.get('hits_at_5_pct'),
                             csls.get('hits_at_10_pct'), csls.get('mrr')])
    upload(csv_path, D17 + '/final_test_metrics.csv')
    # Publish the summary LAST: it is the 9/9 completion marker.
    upload(summary_path, D17 + '/final_test_summary.json')
    print('=' * 74)
    print('[CELL 17] PASS | 9/9 FINAL TEST | NO TEST-BASED SELECTION')
    for arm, agg in aggregate.items():
        raw = agg[PRIMARY]
        print(f'  {arm}: RAW Hits@1={raw["hits_at_1_pct"]["mean"]:.3f}% '
              f'± {raw["hits_at_1_pct"]["std_sample"]:.3f}%; '
              f'MRR={raw["mrr"]["mean"]:.6f} ± {raw["mrr"]["std_sample"]:.6f}', flush=True)
        if INCLUDE_CSLS10:
            c = agg['csls10']
            print(f'    CSLS10 Hits@1={c["hits_at_1_pct"]["mean"]:.3f}% '
                  f'± {c["hits_at_1_pct"]["std_sample"]:.3f}%; '
                  f'MRR={c["mrr"]["mean"]:.6f} ± {c["mrr"]["std_sample"]:.6f}', flush=True)
    print('[REPORT]', summary_path)
    print('[DRIVE]', D17)
    return summary

cell17_final_test()

# CELL 16.1 — Integrity audit of 9 completed runs. NO TRAINING / NO TEST.
def cell16_1_audit():
    import csv, gc, hashlib, json, math, os, statistics, time, zipfile
    from pathlib import Path
    import torch

    ID = 'cell15_20261025_150915_b52701'
    HOME = Path('/tmp/clamea_paper_method_molab')
    P15 = HOME / 'final_protocols' / ID
    P16 = HOME / 'final_training' / f'cell16_stage5_{ID}'
    DATA = HOME / 'datasets/CLAMEA_FBDB15K_protocolB'
    DRIVE_ROOT = 'paper_method_runs/paper_method_20261023_160901_s66/diagnostics'
    D15, D16 = f'{DRIVE_ROOT}/{ID}', f'{DRIVE_ROOT}/cell16_stage5_{ID}'
    CHECK_DATA_SHA = True
    CHECK_ALL_MODEL_WEIGHTS = True  # Downloads 9 models if missing locally (~5 GB).
    SAVE_TO_DRIVE = True

    try:
        drive = fs_drive  # Reuse Cell 16 connection.
    except NameError as exc:
        raise RuntimeError('fs_drive is missing. Run your Drive setup cell.') from exc

    audit_id = 'audit_' + time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    out = P16 / 'audits' / audit_id
    out.mkdir(parents=True, exist_ok=True)

    def digest(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(b)
        return h.hexdigest()

    def js(p):
        return json.loads(Path(p).read_text(encoding='utf-8'))

    def fetch(remote, local, expected=None):
        local = Path(local)
        if local.is_file() and (expected is None or digest(local) == expected):
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + '.part')
        if partial.exists():
            partial.unlink()
        try:
            drive.get_file(remote, str(partial))
            assert partial.is_file(), f'Drive download failed: {remote}'
            if expected:
                assert digest(partial) == expected, f'SHA256 mismatch: {remote}'
            os.replace(partial, local)
        finally:
            if partial.exists():
                partial.unlink()
        return local

    def check(ok, message):
        if not ok:
            raise AssertionError(message)

    print('=' * 70)
    print('CELL 16.1 — STAGE 5 INTEGRITY AUDIT | NO TRAINING / NO TEST')
    manifest = js(fetch(f'{D15}/stage5_manifest.json', P15 / 'stage5_manifest.json'))
    matrix = js(fetch(f'{D15}/run_matrix.json', P15 / 'run_matrix.json'))
    lock = js(fetch(f'{D15}/dataset_lock.json', P15 / 'dataset_lock.json'))
    zip_sha = fetch(f'{D15}/source_archive_sha256.txt',
                    P15 / 'source_archive_sha256.txt').read_text().split()[0]
    archive = fetch(f'{D15}/frozen_sources.zip', P15 / 'frozen_sources.zip', zip_sha)
    check(manifest['job_id'] == ID, 'Wrong Cell15 ID')
    check(manifest['status'] == 'PASS_PROTOCOL_FREEZE_NO_TRAINING', 'Freeze failed')
    check(len(matrix) == manifest['planned_runs'] == 9, 'Run matrix != 9')
    check(len({r['run_id'] for r in matrix}) == 9, 'Duplicated run ID')
    check({r['seed'] for r in matrix} == {66, 190, 1234}, 'Wrong seeds')
    with zipfile.ZipFile(archive) as z:
        check(z.testzip() is None, 'Source archive corrupt')
        for src, files in manifest['source_hashes'].items():
            for name, sha in files.items():
                actual = hashlib.sha256(z.read(f'sources/{src}/{name}')).hexdigest()
                check(actual == sha, f'Frozen source changed: {src}/{name}')
    print('[PASS] Cell15 protocol and frozen source checksums')

    check(lock['file_count'] == len(lock['files']) == 16, 'Wrong dataset lock')
    for item in lock['files']:
        p = DATA / item['path']
        check(p.is_file() and p.stat().st_size == item['bytes'],
              f'Missing/changed dataset file: {p}')
        if CHECK_DATA_SHA:
            check(digest(p) == item['sha256'], f'Dataset SHA mismatch: {p}')
    print('[PASS] Dataset lock: 16 files')

    results, errors = [], []
    tmp_weights = out / '_temporary_model_best.pkl'
    for i, run in enumerate(matrix, 1):
        rid = run['run_id']
        rlocal = P16 / 'runs' / rid
        rdrive = f'{D16}/runs/{rid}'
        try:
            completed = js(fetch(f'{rdrive}/completed.json',
                                 rlocal / 'completed.json'))
            check(completed['status'] == 'COMPLETED_TRAINING_TEST_NOT_EVALUATED',
                  f'Incomplete: {rid}')
            check(completed['run_id'] == rid and completed['arm'] == run['arm']
                  and int(completed['seed']) == run['seed'], f'Run identity mismatch: {rid}')
            check(completed['epochs'] == 300 and completed['test_evaluated'] is False,
                  f'Unexpected completion policy: {rid}')
            check(completed['source_main_sha256'] ==
                  manifest['source_hashes'][run['source']]['main.py'],
                  f'Source mismatch: {rid}')
            cfg_hash = hashlib.sha256(json.dumps(
                run['config'], sort_keys=True).encode()).hexdigest()
            check(completed['config_sha256'] == cfg_hash, f'Config mismatch: {rid}')

            history = fetch(f'{rdrive}/epoch_eval.jsonl',
                            rlocal / 'epoch_eval.jsonl', completed['history_sha256'])
            rows = [json.loads(s) for s in history.read_text().splitlines() if s.strip()]
            check(len(rows) == 231 and [x['epoch'] for x in rows] ==
                  list(range(70, 301)), f'Incomplete Validation curve: {rid}')
            for x in rows:
                check('test_raw_hits1_observational' in x and
                      x['test_raw_hits1_observational'] is None,
                      f'Test field populated/missing: {rid}')
                check(x['checkpoint_selection_metric'] == 'validation_raw_mrr',
                      f'Invalid best selection metric: {rid}')
                mrr, h1 = float(x['validation_raw_mrr']), float(x['validation_raw_hits1'])
                check(math.isfinite(mrr) and 0 <= mrr <= 1 and
                      math.isfinite(h1) and 0 <= h1 <= 100,
                      f'Invalid Validation metrics: {rid}')
            best = max(rows, key=lambda x: x['validation_raw_mrr'])
            check(best['epoch'] == completed['best_epoch'] and
                  abs(best['validation_raw_mrr'] - completed['best_val_mrr']) < 1e-10 and
                  abs(best['validation_raw_hits1'] - completed['best_val_hits1_pct']) < 1e-8,
                  f'Best metrics mismatch: {rid}')

            remote_model = f'{rdrive}/model_best.pkl'
            check(drive.exists(remote_model), f'Best checkpoint missing: {rid}')
            weights_status = 'SKIPPED'
            if CHECK_ALL_MODEL_WEIGHTS:
                local_model = rlocal / 'model_best.pkl'
                reuse = (local_model.is_file() and
                         digest(local_model) == completed['model_best_sha256'])
                path = local_model if reuse else tmp_weights
                try:
                    fetch(remote_model, path, completed['model_best_sha256'])
                    state = torch.load(path, map_location='cpu', weights_only=True)
                    check(isinstance(state, dict) and len(state) > 0,
                          f'Invalid state_dict: {rid}')
                    for key in ('ent_embed.weight', 'rel_embed.weight',
                                'fc_i.weight', 'fc_a.weight'):
                        check(key in state and isinstance(state[key], torch.Tensor),
                              f'Missing tensor {key}: {rid}')
                    check(state['fc_i.weight'].shape[1] == 4096,
                          f'Image dimensionality != 4096: {rid}')
                    for key, v in state.items():
                        check(isinstance(v, torch.Tensor), f'Non-tensor state: {rid}/{key}')
                        if v.is_floating_point() or v.is_complex():
                            check(bool(torch.isfinite(v).all()), f'NaN/Inf: {rid}/{key}')
                    weights_status = 'PASS'
                    del state
                    gc.collect()
                finally:
                    if not reuse and tmp_weights.exists():
                        tmp_weights.unlink()
            result = {
                'run_id': rid, 'arm': run['arm'], 'seed': run['seed'],
                'best_epoch': completed['best_epoch'],
                'best_val_mrr': completed['best_val_mrr'],
                'best_val_hits1_pct': completed['best_val_hits1_pct'],
                'epoch300_val_mrr': rows[-1]['validation_raw_mrr'],
                'weights_status': weights_status, 'status': 'PASS',
            }
            results.append(result)
            print(f"[PASS {i}/9] {rid} best={result['best_epoch']} "
                  f"MRR={result['best_val_mrr']:.6f} weights={weights_status}",
                  flush=True)
        except Exception as exc:
            msg = f'{rid}: {type(exc).__name__}: {exc}'
            errors.append(msg)
            print(f'[FAIL {i}/9] {msg}', flush=True)

    aggregate = {}
    for arm in sorted(set(r['arm'] for r in matrix)):
        arr = [r for r in results if r['arm'] == arm]
        if len(arr) == 3:
            aggregate[arm] = {
                'mean_val_mrr': statistics.mean(r['best_val_mrr'] for r in arr),
                'sample_std_val_mrr': statistics.stdev(r['best_val_mrr'] for r in arr),
                'mean_val_hits1_pct': statistics.mean(r['best_val_hits1_pct'] for r in arr),
            }
    status = 'PASS' if len(results) == 9 and not errors else 'FAIL'
    report = {
        'schema': 'CLAMEA_CELL16_1_INTEGRITY_AUDIT_V1',
        'cell15': ID, 'status': status, 'passed_runs': len(results),
        'errors': errors, 'runs': results, 'aggregate': aggregate,
        'test_evaluated': False, 'independent_validation_recomputed': False,
        'warning': ('This checks the saved weights, frozen source/config, '
                    'history and best metrics. It does NOT independently '
                    'recompute Validation MRR. CLAMEA VAE samples random '
                    'noise during eval, so exact metric equality on rerun '
                    'requires the original RNG state.'),
    }
    report_path = out / 'audit_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    csv_path = out / 'audit_runs.csv'
    with csv_path.open('w', newline='') as f:
        keys = ['run_id', 'arm', 'seed', 'status', 'best_epoch',
                'best_val_mrr', 'best_val_hits1_pct', 'epoch300_val_mrr']
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    if SAVE_TO_DRIVE:
        dst = f'{D16}/audits/{audit_id}'
        try:
            drive.makedirs(dst, exist_ok=True)
            drive.put_file(str(report_path), f'{dst}/audit_report.json')
            drive.put_file(str(csv_path), f'{dst}/audit_runs.csv')
            check(drive.exists(f'{dst}/audit_report.json'), 'Report upload failed')
            print('[DRIVE AUDIT]', dst)
        except Exception as exc:
            print('[WARNING] Drive upload failed; report saved locally:', exc)
    print('=' * 70)
    print(f'[CELL 16.1] {status} | {len(results)}/9 PASS | {len(errors)} errors')
    for arm, m in aggregate.items():
        print(f"  {arm}: MRR={m['mean_val_mrr']:.6f} ± "
              f"{m['sample_std_val_mrr']:.6f}")
    print('[REPORT]', report_path)
    print('[NO TEST] No Test was evaluated')
    if errors:
        raise RuntimeError('Audit failed. Check audit_report.json:\n' + '\n'.join(errors))
    return report

cell16_1_audit()

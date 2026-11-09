# CELL 19 — UPLOAD HISTORICAL CLAMEA RESULTS FROM MOLAB TO W&B
# Paste this entire script into ONE NEW Molab/marimo cell *after Drive setup*.
# Only publishes EXISTING frozen results; does not train or evaluate models.
# First set PUBLISH=False (preflight), then change to True to upload.

def cell19_push_clamea_to_wandb():
    import os, sys, json, csv, hashlib, statistics, math, subprocess
    from pathlib import Path

    PUBLISH = True             # <--- Change to True only after CHECK PASS
    INCLUDE_VALIDATION_HISTORY = True  # Import real epoch_eval.jsonl (70–300), if available
    WANDB_HOST = 'https://api.forge.coreweave.com'
    ENTITY = 'khnhg05'
    PROJECT = 'MLG-Seminar-CLAMEA'
    GROUP = 'CLAMEA-ProtocolB-Final-9Runs'

    CELL15 = 'cell15_20261025_150915_b52701'
    HOME = Path(os.environ.get('CLAMEA_WORK_HOME', '/tmp/clamea_paper_method_molab'))
    P16 = HOME / 'final_training' / f'cell16_stage5_{CELL15}'
    DROOT = 'paper_method_runs/paper_method_20261023_160901_s66/diagnostics'
    D16 = f'{DROOT}/cell16_stage5_{CELL15}'
    D17 = f'{D16}/final_test_evaluation'
    WORK = P16 / 'wandb_retro_import'
    WORK.mkdir(parents=True, exist_ok=True)
    ARMS = {
        'deterministic_method_on': 'Self-training ON',
        'deterministic_method_off': 'Self-training OFF',
        'deterministic_preserve_gold_on': 'Preserve Gold + ON',
    }
    SEEDS = {66, 190, 1234}

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for part in iter(lambda: f.read(1024 * 1024), b''):
                h.update(part)
        return h.hexdigest()

    def fetch(remote, local, expected=None, optional=False):
        local = Path(local)
        if local.is_file() and (expected is None or sha(local) == expected):
            return local
        try:
            drive = fs_drive  # The Drive connection set up in previous Molab cells.
        except NameError:
            if optional:
                return None
            raise RuntimeError('fs_drive is missing. Re-run the Google Drive connection cell.')
        local.parent.mkdir(parents=True, exist_ok=True)
        temp = local.with_suffix(local.suffix + '.downloading')
        try:
            if temp.exists(): temp.unlink()
            drive.get_file(remote, str(temp))
            if not temp.is_file(): raise FileNotFoundError(remote)
            if expected is not None and sha(temp) != expected:
                raise ValueError('SHA256 mismatch: ' + remote)
            temp.replace(local)
            return local
        except Exception as exc:
            if temp.exists(): temp.unlink()
            if optional:
                print('[HISTORY SKIPPED]', remote, type(exc).__name__)
                return None
            raise

    final_path = fetch(f'{D17}/final_test_summary.json', P16 / 'final_test_evaluation' / 'final_test_summary.json')
    final = json.loads(final_path.read_text(encoding='utf-8'))
    assert final.get('schema') == 'CLAMEA_CELL17_FINAL_TEST_SUMMARY_V1'
    assert final.get('status') == 'PASS' and final.get('completed_test_runs') == 9
    assert final.get('cell15') == CELL15 and final.get('test_selection_used') is False
    assert final.get('primary_metric') == 'raw_no_csls' and final.get('csls10_enabled') is True
    rows = final['results']
    assert len(rows) == 9 and len({r['run_id'] for r in rows}) == 9
    assert {(r['arm'], int(r['seed'])) for r in rows} == {(a,s) for a in ARMS for s in SEEDS}
    histories = {}
    records = []

    for r in rows:
        rid, arm, seed = r['run_id'], r['arm'], int(r['seed'])
        assert rid == f'{arm}_s{seed}'
        assert r.get('test_selection_used') is False
        assert r['test_count'] == 8993 and r['test_prediction_seed'] == seed
        assert r['primary_metric'] == 'raw_no_csls' and r['csls10_enabled'] is True
        assert 1 <= r['best_epoch'] <= 300
        for typ in ['raw_no_csls','csls10']:
            obj = r[typ]
            for name in ['hits_at_1_pct','hits_at_5_pct','hits_at_10_pct']:
                assert math.isfinite(obj[name]) and 0 <= obj[name] <= 100
            assert math.isfinite(obj['mrr']) and 0 <= obj['mrr'] <= 1
        records.append({
            'run_id':rid, 'arm':arm, 'seed':seed,
            'best_epoch':r['best_epoch'],'best_val_mrr':r['best_val_mrr'],
            'raw_hits1_pct':r['raw_no_csls']['hits_at_1_pct'],
            'raw_mrr':r['raw_no_csls']['mrr'],
            'csls_hits1_pct':r['csls10']['hits_at_1_pct'],
            'csls_mrr':r['csls10']['mrr']
        })
        histories[rid] = (None, None)
        if INCLUDE_VALIDATION_HISTORY:
            remote = f'{D16}/runs/{rid}'
            done_path = fetch(f'{remote}/completed.json', P16 / 'runs' / rid / 'completed.json', optional=True)
            if done_path:
                done = json.loads(done_path.read_text(encoding='utf-8'))
                assert done['run_id'] == rid and done['seed'] == seed and done['arm'] == arm
                assert done['best_epoch'] == r['best_epoch'] and done['model_best_sha256'] == r['best_model_sha256']
                hp = fetch(f'{remote}/epoch_eval.jsonl', P16 / 'runs' / rid / 'epoch_eval.jsonl',
                           expected=done['history_sha256'], optional=False)
                hist = [json.loads(s) for s in hp.read_text(encoding='utf-8').splitlines() if s.strip()]
                assert [int(v['epoch']) for v in hist] == list(range(70,301)), rid
                assert all(v.get('checkpoint_selection_metric') == 'validation_raw_mrr' for v in hist)
                best = max(hist, key=lambda v: float(v['validation_raw_mrr']))
                assert int(best['epoch']) == r['best_epoch']
                assert math.isclose(best['validation_raw_mrr'],r['best_val_mrr'],abs_tol=1e-9)
                histories[rid] = (hp, hist)

    csv_path = WORK / 'cell19_verified_run_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    for arm, label in ARMS.items():
        rr = [x for x in records if x['arm'] == arm]
        h1 = [x['raw_hits1_pct'] for x in rr]
        mrr = [x['raw_mrr'] for x in rr]
        print(f'{label:22s} RAW H@1={statistics.mean(h1):.3f} ± {statistics.stdev(h1):.3f}% | '
              f'RAW MRR={statistics.mean(mrr):.6f}')
    n_hist = sum(int(hist is not None) for _,hist in histories.values())
    print(f'CHECK PASS: 9/9 runs; Final JSON SHA256={sha(final_path)}; '
          f'validated history files={n_hist}/9')
    if not PUBLISH:
        print('DRY RUN ONLY. Set PUBLISH=True in this cell to upload. No W&B changes made.')
        return

    try:
        import wandb
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'wandb'])
        import wandb
    os.environ['WANDB_BASE_URL'] = WANDB_HOST
    assert wandb.login(host=WANDB_HOST), 'W&B login unsuccessful; set WANDB_API_KEY in Molab Secrets.'

    urls = []
    for j,r in enumerate(rows):
        rid, arm, seed = r['run_id'], r['arm'], int(r['seed'])
        wb_id = hashlib.sha256(('clamea-final-cell17:' + rid).encode()).hexdigest()[:8]
        hp, history = histories[rid]
        with wandb.init(
            entity=ENTITY, project=PROJECT, group=GROUP,
            id=wb_id, resume='never', name=rid,
            job_type='retrospective-import', mode='online',
            tags=['CLAMEA','Protocol-B','final-9runs','posthoc','not-paper-faithful'],
            notes=('Retrospective import from frozen Cell17. No training/re-evaluation. '
                   'Selected by Validation RAW MRR, not by Test. '
                   'Final Test uses one seeded stochastic VAE draw. '
                   'Seed 1234 selected identical checkpoint across arms.'),
            config={
                'source_run_id':rid,'arm':arm,'training_seed':seed,
                'dataset':'FB15K-DB15K reconstructed','split':'721 / Protocol B',
                'test_pairs':8993,'visual_features':'VGG16 4096d',
                'cell15':CELL15, 'audit_id':final['cell16_audit'],
                'checkpoint_sha256':r['best_model_sha256'],
                'historical_import':True, 'training_replayed':False,
                'validation_history_imported': history is not None,
            },
        ) as run:
            if history:
                run.define_metric('epoch')
                run.define_metric('validation/raw_mrr',step_metric='epoch')
                for point in history:
                    run.log({'epoch':int(point['epoch']),
                             'validation/raw_mrr':float(point['validation_raw_mrr'])})
            results = {
                'validation/best_raw_mrr':float(r['best_val_mrr']),
                'validation/best_epoch':int(r['best_epoch']),
                'final_test/test_pairs':int(r['test_count']),
            }
            for src,prefix in [('raw_no_csls','final_test/raw'),('csls10','final_test/csls10')]:
                for k in ['hits_at_1_pct','hits_at_5_pct','hits_at_10_pct','mrr','mr']:
                    results[f'{prefix}/{k}'] = float(r[src][k])
            run.summary.update(results)
            artifact = wandb.Artifact('clamea-cell17-' + wb_id, type='evaluation-evidence',
                                      metadata={'source_run_id':rid,'final_json_sha256':sha(final_path)})
            artifact.add_file(str(final_path),name='final_test_summary.json')
            artifact.add_file(str(csv_path),name='cell19_verified_run_metrics.csv')
            if hp is not None: artifact.add_file(str(hp),name='epoch_eval.jsonl')
            run.log_artifact(artifact)
            urls.append(run.url)
            print(f'UPLOAD {j+1}/9: {rid} -> {run.url}')
    print('UPLOAD COMPLETE — 9 runs:')
    for url in urls: print(url)

cell19_push_clamea_to_wandb()

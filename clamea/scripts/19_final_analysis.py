# CELL 18 — FINAL REPRODUCTION ANALYSIS
# One Molab/Marimo cell. READ-ONLY with respect to Cell15–17 artifacts.
# Does NOT train, reload model weights, evaluate Test, or select checkpoints.

def cell18_final_analysis():
    import csv
    import hashlib
    import json
    import math
    import os
    import statistics
    import time
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Frozen experiment identities — do not change after viewing Test results.
    CELL15 = 'cell15_20261025_150915_b52701'
    HOME = Path(os.environ.get('CLAMEA_WORK_HOME', '/tmp/clamea_paper_method_molab'))
    P15 = HOME / 'final_protocols' / CELL15
    P16 = HOME / 'final_training' / f'cell16_stage5_{CELL15}'
    DROOT = 'paper_method_runs/paper_method_20261023_160901_s66/diagnostics'
    D15 = f'{DROOT}/{CELL15}'
    D16 = f'{DROOT}/cell16_stage5_{CELL15}'
    D17 = f'{D16}/final_test_evaluation'
    OUT = P16 / 'final_reproduction_analysis' / ('analysis_' + time.strftime('%Y%m%d_%H%M%S', time.gmtime()))
    OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = [66, 190, 1234]
    ARMS = ['deterministic_method_on', 'deterministic_method_off',
            'deterministic_preserve_gold_on']
    LABELS = {'deterministic_method_on': 'Self-training ON',
              'deterministic_method_off': 'Self-training OFF',
              'deterministic_preserve_gold_on': 'Preserve Gold + ON'}
    COLORS = {'deterministic_method_on': '#334155',
              'deterministic_method_off': '#17617a',
              'deterministic_preserve_gold_on': '#b06a40'}
    MARKERS = (70, 140, 210, 280)
    UPLOAD_TO_DRIVE = True
    # No paper score is hard-coded: this uses Protocol B / VGG16 features.

    try:
        drive = fs_drive  # Reuse the authenticated Cell16/17 connection.
    except NameError as exc:
        raise RuntimeError('fs_drive is missing; run the Google Drive setup cell.') from exc

    def ensure(condition, message):
        if not condition:
            raise AssertionError(message)

    def hash_file(p):
        h = hashlib.sha256()
        with Path(p).open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    def fetch(remote, local, expected_hash=None):
        local = Path(local)
        if local.is_file() and (not expected_hash or hash_file(local) == expected_hash):
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + '.part')
        if partial.exists():
            partial.unlink()
        try:
            drive.get_file(remote, str(partial))
            ensure(partial.is_file(), f'Download failed: {remote}')
            if expected_hash:
                ensure(hash_file(partial) == expected_hash, f'Hash mismatch: {remote}')
            os.replace(partial, local)
        finally:
            if partial.exists():
                partial.unlink()
        return local

    def read_json(p):
        return json.loads(Path(p).read_text(encoding='utf-8'))

    def save_json(p, obj):
        Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

    def csv_write(p, fields, data):
        with Path(p).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)

    def mean_sd(items):
        a = [float(x) for x in items]
        return {'mean': statistics.mean(a),
                'std_sample': statistics.stdev(a) if len(a) > 1 else 0.0}

    def metric(r, name, view='raw_no_csls'):
        value = float(r[view][name])
        ensure(math.isfinite(value), f'Non-finite {view}/{name}: {r["run_id"]}')
        return value

    def upload(local, remote):
        drive.makedirs(remote.rsplit('/', 1)[0], exist_ok=True)
        for attempt in range(1, 4):
            try:
                drive.put_file(str(local), remote)
                ensure(drive.exists(remote), f'Upload verification failed: {remote}')
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)

    print('=' * 74)
    print('CELL 18 — FINAL REPRODUCTION ANALYSIS | READ-ONLY, NO TEST RE-EVALUATION')
    print('[PRIMARY] RAW (no CSLS) | [SECONDARY] CSLS-10')
    print('=' * 74, flush=True)

    # Re-read frozen matrix, audit, and Cell17 Test summary.
    matrix = read_json(fetch(f'{D15}/run_matrix.json', P15 / 'run_matrix.json'))
    final = read_json(fetch(f'{D17}/final_test_summary.json',
                            P16 / 'final_test_evaluation' / 'final_test_summary.json'))
    audit_id = final['cell16_audit']
    audit = read_json(fetch(f'{D16}/audits/{audit_id}/audit_report.json',
                            P16 / 'audits' / audit_id / 'audit_report.json'))
    ensure(len(matrix) == 9 and len(final['results']) == 9, 'Wrong number of runs')
    ensure(audit['status'] == 'PASS' and audit['passed_runs'] == 9 and not audit['errors'],
           'Cell16.1 audit is not PASS')
    ensure(final['schema'] == 'CLAMEA_CELL17_FINAL_TEST_SUMMARY_V1' and
           final['status'] == 'PASS' and final['completed_test_runs'] == 9,
           'Cell17 final Test summary is not PASS')
    ensure(final['cell15'] == CELL15 and audit['cell15'] == CELL15,
           'Frozen Cell15 identity mismatch')
    ensure(final['test_selection_used'] is False and
           final['primary_metric'] == 'raw_no_csls' and final['csls10_enabled'] is True,
           'Test evaluation protocol changed')

    by_test = {r['run_id']: r for r in final['results']}
    by_audit = {r['run_id']: r for r in audit['runs']}
    ensure(len(by_test) == len(by_audit) == 9 and
           set(by_test) == set(by_audit) == {r['run_id'] for r in matrix},
           'Audit / matrix / Test run identities do not match')
    ensure({r['arm'] for r in matrix} == set(ARMS) and
           {int(r['seed']) for r in matrix} == set(SEEDS), 'Unexpected arms or seeds')

    all_histories, rows, per_run = {}, [], {}
    for j, run in enumerate(matrix, 1):
        rid, arm, seed = run['run_id'], run['arm'], int(run['seed'])
        t, a = by_test[rid], by_audit[rid]
        local = P16 / 'runs' / rid
        remote = f'{D16}/runs/{rid}'
        done = read_json(fetch(f'{remote}/completed.json', local / 'completed.json'))
        ensure(done['status'] == 'COMPLETED_TRAINING_TEST_NOT_EVALUATED',
               f'Training not complete: {rid}')
        ensure(done['run_id'] == rid and done['seed'] == seed and done['arm'] == arm,
               f'Identity mismatch: {rid}')
        ensure(t['run_id'] == a['run_id'] == rid and t['arm'] == a['arm'] == arm and
               int(t['seed']) == int(a['seed']) == seed, f'Mismatched arm/seed: {rid}')
        ensure(t['best_epoch'] == a['best_epoch'] == done['best_epoch'] and
               abs(float(t['best_val_mrr']) - float(a['best_val_mrr'])) < 1e-10,
               f'Best checkpoint mismatch: {rid}')
        ensure(t['best_model_sha256'] == done['model_best_sha256'],
               f'Model checkpoint SHA mismatch: {rid}')
        ensure(t['test_selection_used'] is False and
               t['primary_metric'] == 'raw_no_csls' and t['csls10_enabled'] is True and
               int(t['test_count']) == 8993, f'Test protocol mismatch: {rid}')
        ensure(t['test_prediction_seed'] == seed, f'Test seed mismatch: {rid}')
        ensure(a['weights_status'] == 'PASS' and a['status'] == 'PASS',
               f'Weights were not audited: {rid}')

        hp = fetch(f'{remote}/epoch_eval.jsonl', local / 'epoch_eval.jsonl',
                   done['history_sha256'])
        history = [json.loads(line) for line in hp.read_text(encoding='utf-8').splitlines()
                   if line.strip()]
        ensure([int(x['epoch']) for x in history] == list(range(70, 301)),
               f'Validation history incomplete: {rid}')
        ensure(all(x['test_raw_hits1_observational'] is None and
                   x['checkpoint_selection_metric'] == 'validation_raw_mrr'
                   for x in history), f'Test leakage or selection policy mismatch: {rid}')
        val = [float(x['validation_raw_mrr']) for x in history]
        ensure(all(math.isfinite(v) and 0 <= v <= 1 for v in val),
               f'Invalid Validation MRR: {rid}')
        best_row = max(history, key=lambda r: float(r['validation_raw_mrr']))
        ensure(int(best_row['epoch']) == int(done['best_epoch']) and
               abs(float(best_row['validation_raw_mrr']) - float(done['best_val_mrr'])) < 1e-10,
               f'Best Validation is inconsistent: {rid}')
        all_histories[(arm, seed)] = {'epoch': list(range(70, 301)), 'mrr': val}
        r = {'run_id': rid, 'arm': arm, 'label': LABELS[arm], 'seed': seed,
             'best_epoch': int(done['best_epoch']),
             'best_val_mrr': float(done['best_val_mrr']),
             'best_val_hits1_pct': float(done['best_val_hits1_pct']),
             'epoch300_val_mrr': val[-1],
             'raw_hits1_pct': metric(t, 'hits_at_1_pct'),
             'raw_hits5_pct': metric(t, 'hits_at_5_pct'),
             'raw_hits10_pct': metric(t, 'hits_at_10_pct'),
             'raw_mrr': metric(t, 'mrr'),
             'csls_hits1_pct': metric(t, 'hits_at_1_pct', 'csls10'),
             'csls_hits5_pct': metric(t, 'hits_at_5_pct', 'csls10'),
             'csls_hits10_pct': metric(t, 'hits_at_10_pct', 'csls10'),
             'csls_mrr': metric(t, 'mrr', 'csls10'),
             'best_model_sha256': done['model_best_sha256'],
             'test_count': int(t['test_count'])}
        per_run[(arm, seed)] = r
        rows.append(r)
        print(f'[C18 LOAD {j}/9] {rid} | best_val={r["best_val_mrr"]:.6f} '
              f'raw_test={r["raw_mrr"]:.6f}', flush=True)

    # Make the 9-run and aggregate tables; use SAMPLE standard deviation.
    metric_keys = ['best_val_mrr', 'best_val_hits1_pct', 'epoch300_val_mrr',
                   'raw_hits1_pct', 'raw_hits5_pct', 'raw_hits10_pct', 'raw_mrr',
                   'csls_hits1_pct', 'csls_hits5_pct', 'csls_hits10_pct', 'csls_mrr']
    agg = {}
    aggregate_rows = []
    for arm in ARMS:
        rr = [per_run[(arm, s)] for s in SEEDS]
        agg[arm] = {key: mean_sd(r[key] for r in rr) for key in metric_keys}
        aggregate_rows.append({'arm': arm, 'label': LABELS[arm], 'seeds': 3,
                               **{f'{k}_mean': agg[arm][k]['mean'] for k in metric_keys},
                               **{f'{k}_std_sample': agg[arm][k]['std_sample'] for k in metric_keys}})
    # Check against published Cell17 aggregate (independent bookkeeping).
    for arm in ARMS:
        for key, original in [('raw_mrr', final['aggregate'][arm]['raw_no_csls']['mrr']),
                              ('raw_hits1_pct', final['aggregate'][arm]['raw_no_csls']['hits_at_1_pct']),
                              ('csls_mrr', final['aggregate'][arm]['csls10']['mrr'])]:
            ensure(abs(agg[arm][key]['mean'] - original['mean']) < 1e-10,
                   f'Aggregate mismatch: {arm}/{key}')

    comparisons = []
    for target in ['deterministic_method_off', 'deterministic_preserve_gold_on']:
        for seed in SEEDS:
            base, other = per_run[('deterministic_method_on', seed)], per_run[(target, seed)]
            comparisons.append({'comparison': f'{LABELS[target]} minus Self-training ON',
                                'seed': seed,
                                'delta_raw_mrr': other['raw_mrr'] - base['raw_mrr'],
                                'delta_raw_hits1_pp': other['raw_hits1_pct'] - base['raw_hits1_pct'],
                                'delta_csls_mrr': other['csls_mrr'] - base['csls_mrr'],
                                'delta_val_mrr': other['best_val_mrr'] - base['best_val_mrr']})

    csv_write(OUT / 'per_run_metrics.csv', list(rows[0]), rows)
    csv_write(OUT / 'aggregate_mean_std.csv', list(aggregate_rows[0]), aggregate_rows)
    csv_write(OUT / 'paired_seed_differences.csv', list(comparisons[0]), comparisons)
    curve_rows = [dict(run_id=per_run[(arm, seed)]['run_id'], arm=arm, seed=seed,
                       epoch=int(ep), val_mrr=float(v))
                  for arm in ARMS for seed in SEEDS
                  for ep, v in zip(all_histories[(arm, seed)]['epoch'],
                                   all_histories[(arm, seed)]['mrr'])]
    csv_write(OUT / 'validation_curves.csv', ['run_id', 'arm', 'seed', 'epoch', 'val_mrr'], curve_rows)

    # Publication-friendly, minimalistic plots. No pretrained models or GPU needed.
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    figures = []
    for seed in SEEDS:
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        for arm in ARMS:
            vals = all_histories[(arm, seed)]
            ax.plot(vals['epoch'], vals['mrr'], label=LABELS[arm], color=COLORS[arm], lw=1.7)
        for ep in MARKERS:
            ax.axvline(ep, lw=0.7, ls=':', color='#999999', alpha=0.5)
        ax.set(xlabel='Epoch', ylabel='Validation RAW MRR',
               title=f'Validation trajectories — seed {seed}', xlim=(70, 300))
        ax.grid(axis='y', alpha=0.17)
        ax.legend(frameon=False, ncol=3, loc='upper right', fontsize=8)
        fig.tight_layout()
        name = f'validation_seed_{seed}.png'
        fig.savefig(OUT / name)
        plt.close(fig)
        figures.append(name)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for arm in ARMS:
        y = [statistics.mean(all_histories[(arm, seed)]['mrr'][i] for seed in SEEDS)
             for i in range(231)]
        ax.plot(list(range(70, 301)), y, lw=2, label=LABELS[arm], color=COLORS[arm])
    for ep in MARKERS:
        ax.axvline(ep, lw=0.7, ls=':', color='#999999', alpha=0.5)
    ax.set(xlabel='Epoch', ylabel='Mean Validation RAW MRR (3 seeds)',
           title='Validation trajectories — mean across seeds', xlim=(70, 300))
    ax.grid(axis='y', alpha=0.17)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'validation_mean_3seeds.png')
    plt.close(fig)
    figures.append('validation_mean_3seeds.png')

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, key, ylabel, scale in [(axes[0], 'raw_hits1_pct', 'Test RAW Hits@1 (%)', 1),
                                   (axes[1], 'raw_mrr', 'Test RAW MRR', 1)]:
        values = [agg[a][key]['mean'] * scale for a in ARMS]
        errs = [agg[a][key]['std_sample'] * scale for a in ARMS]
        ax.bar(range(3), values, yerr=errs, capsize=4,
               color=[COLORS[a] for a in ARMS], width=0.62)
        ax.set_xticks(range(3), ['ON', 'OFF', 'Preserve Gold'])
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.17)
    fig.suptitle('Final Test — mean ± sample standard deviation (3 seeds)')
    fig.tight_layout()
    fig.savefig(OUT / 'final_test_raw_comparison.png')
    plt.close(fig)
    figures.append('final_test_raw_comparison.png')

    # Paired RAW Test differences, descriptive only (n=3).
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for label, arm, symbol in [('OFF − ON', 'deterministic_method_off', 'o'),
                               ('Preserve − ON', 'deterministic_preserve_gold_on', 's')]:
        vals = [per_run[(arm, s)]['raw_mrr'] - per_run[('deterministic_method_on', s)]['raw_mrr']
                for s in SEEDS]
        ax.plot(SEEDS, vals, marker=symbol, label=label, color=COLORS[arm], lw=1.4)
    ax.axhline(0, color='#333333', lw=1)
    ax.set(xticks=SEEDS, xlabel='Random seed', ylabel='Paired Δ RAW Test MRR',
           title='Differences against Self-training ON (descriptive)')
    ax.grid(axis='y', alpha=0.17)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'paired_raw_mrr_differences.png')
    plt.close(fig)
    figures.append('paired_raw_mrr_differences.png')

    # Stable best checkpoints at epoch 70 may coincide across arms.
    coincident = {}
    for seed in SEEDS:
        ids = {arm: per_run[(arm, seed)]['best_model_sha256'] for arm in ARMS}
        if len(set(ids.values())) < len(ARMS):
            coincident[str(seed)] = ids

    off_minus_on = [c for c in comparisons if c['comparison'].startswith('Self-training OFF')]
    delta = mean_sd(c['delta_raw_mrr'] for c in off_minus_on)
    verdict = ('OFF has higher mean RAW Test MRR within this fixed Protocol B experiment; '
               'do not treat n=3 as statistical significance or as a paper-faithful benchmark.')
    report = {
        'schema': 'CLAMEA_CELL18_FINAL_REPRODUCTION_ANALYSIS_V1',
        'status': 'PASS', 'cell15': CELL15, 'audit_id': audit_id,
        'source_cell17_status': final['status'], 'test_selection_used': False,
        'retraining_performed': False, 'test_re_evaluated': False,
        'test_reused_only_for_reporting': True,
        'primary_metric': 'raw_no_csls', 'secondary_metric': 'csls10',
        'seeds': SEEDS, 'n_runs': len(rows), 'per_run': rows, 'aggregate': agg,
        'paired_comparisons': comparisons,
        'off_minus_on_raw_mrr': delta,
        'identical_best_checkpoint_sha_by_seed': coincident,
        'figures': figures, 'finding': verdict,
        'limitations': [
            'Method-level reproduction using Protocol B / VGG16 4096-dimensional image features, not exact paper-data replication.',
            'Validation ranks 1,284 candidate entities; Test ranks 8,993. Their MRR values are not directly comparable as identical tasks.',
            'VAE sampling in evaluation is stochastic; Cell17 used one fixed run-seed draw, not an average across inference draws.',
            'Only three training seeds; paired differences are descriptive, not significance-tested.',
            'Preserve Gold is a modified implementation, not a paper-original arm.',
            'All arms show late-epoch Validation deterioration; do not infer a single cause from these observations.',
            'The analysis does not use Test for checkpoint selection or hyperparameter tuning.',
        ],
    }
    report_path = OUT / 'analysis_report.json'
    save_json(report_path, report)

    # Markdown narrative and reusable tables for academic presentation.
    def pm(arm, key, fmt='.6f'):
        z = agg[arm][key]
        return f"{format(z['mean'], fmt)} ± {format(z['std_sample'], fmt)}"
    md = [
        '# CLAMEA — Final Reproduction Analysis', '',
        '**Protocol:** FB15K–DB15K, split 721 (Protocol B); VGG16 4096d image features.',
        '**Evaluation:** Frozen best Validation checkpoints; 8,993 Test pairs; RAW primary, CSLS-10 secondary.',
        '**Status:** 9/9 training and Test runs complete; Cell16.1 audit PASS.', '',
        '## Final Test metrics (mean ± sample SD, 3 seeds)', '',
        '| Arm | RAW Hits@1 (%) | RAW Hits@5 (%) | RAW Hits@10 (%) | RAW MRR | CSLS-10 Hits@1 (%) | CSLS-10 MRR |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        md.append(f'| {LABELS[arm]} | {pm(arm, "raw_hits1_pct", ".3f")} | '
                  f'{pm(arm, "raw_hits5_pct", ".3f")} | {pm(arm, "raw_hits10_pct", ".3f")} | '
                  f'{pm(arm, "raw_mrr")} | {pm(arm, "csls_hits1_pct", ".3f")} | '
                  f'{pm(arm, "csls_mrr")} |')
    md.extend(['', '## Validation (best checkpoint chosen before Test)', '',
               '| Arm | Best RAW MRR | Best Hits@1 (%) | Epoch-300 RAW MRR |',
               '|---|---:|---:|---:|'])
    for arm in ARMS:
        md.append(f'| {LABELS[arm]} | {pm(arm,"best_val_mrr")} | '
                  f'{pm(arm,"best_val_hits1_pct",".3f")} | {pm(arm,"epoch300_val_mrr")} |')
    md.extend(['', '## Paired differences in Test RAW MRR', '',
               '| Comparison | Seed 66 | Seed 190 | Seed 1234 | Mean delta |',
               '|---|---:|---:|---:|---:|'])
    for arm in ['deterministic_method_off', 'deterministic_preserve_gold_on']:
        vs = [per_run[(arm, s)]['raw_mrr'] - per_run[('deterministic_method_on', s)]['raw_mrr']
              for s in SEEDS]
        md.append('| ' + LABELS[arm] + ' − ON | ' + ' | '.join(f'{v:+.6f}' for v in vs) +
                  f' | {statistics.mean(vs):+.6f} |')
    md.extend(['', '## Interpretation', '',
               f'- {verdict}',
               '- OFF outperforms ON in two seeds and ties in one on RAW Test MRR.',
               '- Preserve Gold does not improve consistently across seeds.',
               '- All three configurations have best checkpoints early in training; later Validation performance declines.',
               '- Matching checkpoints across arms (where hashes match) reflect identical selected weights, not independent evidence.',
               '', '## Scope and limitations', ''])
    md.extend(f'- {x}' for x in report['limitations'])
    md.extend(['', '## Figures', ''])
    md.extend(f'- `{name}`' for name in figures)
    md.append('')
    markdown = OUT / 'reproduction_report.md'
    markdown.write_text('\n'.join(md), encoding='utf-8')

    remote_out = f'{D16}/final_reproduction_analysis/{OUT.name}'
    outputs = ([OUT / n for n in ('per_run_metrics.csv', 'aggregate_mean_std.csv',
                                 'paired_seed_differences.csv', 'validation_curves.csv')]
               + [OUT / f for f in figures] + [markdown, report_path])
    # Write last: analysis_report.json is a completed artifact marker.
    if UPLOAD_TO_DRIVE:
        for p in outputs:
            upload(p, f'{remote_out}/{p.name}')
        print('[C18 BACKUP]', remote_out, flush=True)

    print('=' * 74)
    print('[CELL 18] PASS | 9/9 | NO TRAINING | NO TEST RE-EVALUATION')
    for arm in ARMS:
        print(f'  {LABELS[arm]}: RAW H@1={pm(arm,"raw_hits1_pct",".3f")}% '
              f'RAW MRR={pm(arm,"raw_mrr")} CSLS10 MRR={pm(arm,"csls_mrr")}', flush=True)
    print(f'  OFF − ON: paired mean Δ RAW Test MRR={delta["mean"]:+.6f}')
    print('[LOCAL REPORT]', markdown)
    print('[JSON REPORT]', report_path)
    print('[DRIVE]', remote_out)
    return report

cell18_final_analysis()

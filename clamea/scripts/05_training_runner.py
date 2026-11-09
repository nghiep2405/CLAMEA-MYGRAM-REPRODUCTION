import textwrap as _textwrap_runner

RUNNER_CODE = _textwrap_runner.dedent(r"""
#!/usr/bin/env python3

import os
import sys
import json
import time
import random
import hashlib
import traceback

from pathlib import Path

import numpy as np
import torch
import wandb


def sha256(p):
    h = hashlib.sha256()

    with open(p, 'rb') as f:
        for c in iter(
            lambda: f.read(4 * 1024 * 1024),
            b''
        ):
            h.update(c)

    return h.hexdigest()


# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

run_dir = Path(sys.argv[1]).resolve()
source_dir = Path(sys.argv[2]).resolve()

cfg = json.loads(
    (run_dir / 'config.json').read_text()
)

manifest = json.loads(
    (run_dir / 'manifest.json').read_text()
)

# Verify frozen dataset splits.
for name, sha in manifest['split_hashes'].items():
    assert sha256(
        Path(cfg['training_data']) / name
    ) == sha, 'Split hash mismatch: ' + name

# Verify exact patched source files.
for fn, label in [
    ('main.py', 'patched_main_sha256'),
    ('model.py', 'patched_model_sha256'),
    ('utils.py', 'patched_utils_sha256'),
]:
    assert sha256(
        source_dir / fn
    ) == manifest[label], 'Source mismatch: ' + fn

assert torch.cuda.is_available(), 'No GPU available'

sys.path.insert(0, str(source_dir))

# Avoid CLAMEA upstream circular import.
import kgs
import main as official

from kgs import KGs
from evaluation import greedy_alignment


# ============================================================
# 2. RANDOM SEED
# ============================================================

run_id = manifest['run_id']
seed = int(cfg['random_seed'])

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = official.ARGs(cfg)


# ============================================================
# 3. RESUME OR NEW RUN
# ============================================================

resume_file = run_dir / 'resume_state.pt'

if resume_file.exists():

    out = (
        run_dir / 'out_folder.txt'
    ).read_text().strip()

    assert Path(out).is_dir(), \
        'Resume output folder missing'

    os.environ['CLAMEA_RESUME_STATE'] = str(resume_file)

    saved = torch.load(
        resume_file,
        map_location='cpu',
        weights_only=False
    )

    checkpoint_epoch = int(saved['epoch'])

    trace = Path(out) / 'epoch_eval.jsonl'

    if trace.exists():
        retained = []

        for raw in trace.read_text().splitlines():
            try:
                record = json.loads(raw)

                if int(record['epoch']) <= checkpoint_epoch:
                    retained.append(raw)

            except (
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError
            ):
                continue

        trace.write_text(
            ('\n'.join(retained) + '\n')
            if retained else ''
        )

    del saved

    print(
        'RESUME from verified own snapshot:',
        resume_file,
        flush=True
    )

else:

    out = official.generate_out_folder(
        args.output,
        args.training_data,
        args.dataset_division,
        'CLAMEA_paper_method'
    )

    (run_dir / 'out_folder.txt').write_text(out)

    os.environ.pop('CLAMEA_RESUME_STATE', None)


# ============================================================
# 4. TRAINING OPTIONS
# ============================================================

os.environ['CLAMEA_WANDB_ENABLED'] = '1'
os.environ['CLAMEA_RESUME_SNAPSHOTS'] = '1'
os.environ['CLAMEA_RESUME_INTERVAL'] = '30'


# ============================================================
# 5. INITIALIZE W&B
# ============================================================

wandb.init(
    project='MLG-Seminar-CLAMEA',

    id=manifest['wandb_id'],
    name=run_id,
    resume='allow',

    job_type='actual-gpu-training-paper-method-adaptation',

    tags=[
        'CLAMEA',
        'VGG16',
        'Protocol-B',
        'F19-fixed',
        'F7-fixed',
        'live-training',
        'resume-capable'
    ],

    config={
        **cfg,

        'source_commit': manifest['source_commit'],

        'dataset_sha256': manifest['dataset_sha256'],

        'implementation': (
            'method-adaptation-not-exact-table1'
        )
    },

    notes=(
        'Independent actual GPU training; alternative '
        'HUMEA/MMKB VGG16 Protocol B. '
        'Accurate validation; no test lookahead; '
        'periodic training state snapshots.'
    )
)


# ============================================================
# 6. TRAINING RESULT MANIFEST
# ============================================================

result = {
    'run_id': run_id,
    'seed': seed,

    'status': 'RUNNING',

    'start_utc': time.strftime(
        '%FT%TZ', time.gmtime()
    ),

    'resumed': resume_file.exists(),

    'test_monitor_policy': (
        'Observational raw Test Hits@1 each epoch >=70; '
        'not for model selection'
    ),

    'best_checkpoint_selection': 'Validation MRR only'
}


# ============================================================
# 7. TRAIN MODEL
# ============================================================

try:

    kg = KGs(
        args.training_data,
        args.dataset_division,
        args.random_seed,
        ordered=True,
        modality='ria'
    )

    model = official.CLAMEA(kg, args).cuda()

    assert model.fc_i.in_features == 4096

    print(
        'BEGIN TRAINING:',
        run_id,
        flush=True
    )

    official.train(
        model,
        kg,
        args,
        out
    )


    # ========================================================
    # 8. VERIFY OBSERVATIONAL EVALUATION HISTORY
    # ========================================================

    curve_path = Path(out) / 'epoch_eval.jsonl'

    assert curve_path.exists(), (
        'Missing observational epoch evaluation trace'
    )

    curve = [
        json.loads(line)
        for line in curve_path.read_text().splitlines()
        if line.strip()
    ]

    expected = set(
        range(
            args.start_valid,
            args.max_epoch + 1
        )
    )

    observed = [
        int(r['epoch'])
        for r in curve
    ]

    assert len(observed) == len(set(observed)), (
        'Duplicate epoch evaluation rows'
    )

    assert set(observed) == expected, (
        'Incomplete epoch Test Hits@1 curve'
    )

    result['epoch_eval_records'] = len(curve)
    result['epoch_eval_filename'] = 'epoch_eval.jsonl'


    # ========================================================
    # 9. LOAD BEST CHECKPOINT
    # ========================================================

    ckpt = Path(out) / 'model_best.pkl'

    assert ckpt.exists(), (
        'Missing best validation checkpoint'
    )

    result['checkpoint_sha256'] = sha256(ckpt)
    result['checkpoint_bytes'] = ckpt.stat().st_size

    model.load_state_dict(
        torch.load(
            ckpt,
            map_location='cuda',
            weights_only=True
        ),
        strict=True
    )

    model.eval()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(
        'FINAL TEST on best validation-selected checkpoint:',
        run_id,
        flush=True
    )


    # ========================================================
    # 10. FINAL TEST EMBEDDINGS
    # ========================================================

    with torch.no_grad():

        e1 = torch.LongTensor(
            kg.test_entities1
        ).cuda()

        e2 = torch.LongTensor(
            kg.test_entities2
        ).cuda()

        a1 = torch.ByteTensor([
            kg.attr_mask[x]
            for x in kg.test_entities1
        ]).cuda()

        a2 = torch.ByteTensor([
            kg.attr_mask[x]
            for x in kg.test_entities2
        ]).cuda()

        i1 = torch.ByteTensor([
            kg.image_mask[x]
            for x in kg.test_entities1
        ]).cuda()

        i2 = torch.ByteTensor([
            kg.image_mask[x]
            for x in kg.test_entities2
        ]).cuda()

        emb1, emb2, *_ = model.predict(
            e1, e2,
            a1, i1,
            a2, i2
        )

        del e1, e2, a1, a2, i1, i2


    # ========================================================
    # 11. FINAL RAW EVALUATION
    # ========================================================

    _, raw_hits, raw_mr, raw_mrr = greedy_alignment(
        emb1,
        emb2,
        args.top_k,
        args.test_threads_num,
        metric=args.eval_metric,
        normalize=args.eval_norm,
        csls_k=0,
        accurate=True
    )

    # ========================================================
    # 12. FINAL CSLS EVALUATION
    # ========================================================

    _, csls_hits, csls_mr, csls_mrr = greedy_alignment(
        emb1,
        emb2,
        args.top_k,
        args.test_threads_num,
        metric=args.eval_metric,
        normalize=args.eval_norm,
        csls_k=args.csls,
        accurate=True
    )

    result['final_test'] = {
        'raw': {
            'hits1': float(raw_hits[0]),
            'hits5': float(raw_hits[1]),
            'hits10': float(raw_hits[2]),
            'mr': float(raw_mr),
            'mrr': float(raw_mrr)
        },

        'csls': {
            'hits1': float(csls_hits[0]),
            'hits5': float(csls_hits[1]),
            'hits10': float(csls_hits[2]),
            'mr': float(csls_mr),
            'mrr': float(csls_mrr)
        }
    }


    # ========================================================
    # 13. LOG FINAL METRICS TO W&B
    # ========================================================

    for method, d in result['final_test'].items():
        for name, val in d.items():
            wandb.summary[
                'final_test/' + method + '/' + name
            ] = val

    wandb.summary[
        'checkpoint_sha256'
    ] = result['checkpoint_sha256']

    wandb.summary['training_seed'] = seed
    wandb.summary['inference_on_best_checkpoint'] = True

    result['status'] = 'COMPLETED'


except Exception:

    result['status'] = 'ERROR'
    result['traceback'] = traceback.format_exc()

    print(
        result['traceback'],
        flush=True
    )

    raise


finally:

    result['finish_utc'] = time.strftime(
        '%FT%TZ', time.gmtime()
    )

    (
        run_dir / 'training_summary.json'
    ).write_text(
        json.dumps(result, indent=2)
    )

    wandb.finish()
""").lstrip()

runner_path = WORK / 'run_paper_method.py'

runner_path.write_text(
    RUNNER_CODE,
    encoding='utf-8'
)

compile(
    RUNNER_CODE,
    str(runner_path),
    'exec'
)

print(
    'TRAINING RUNNER CREATED but NOT EXECUTED:',
    runner_path
)

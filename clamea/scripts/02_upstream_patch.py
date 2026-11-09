run_checked([
    sys.executable, '-m', 'pip', 'install', '-q',
    'h5py', 'scipy', 'faiss-cpu', 'scikit-learn',
    'pillow', 'matplotlib', 'transformers==5.20.2', 'wandb'
])

run_checked([
    sys.executable, '-c',
    'import h5py,scipy,faiss,sklearn,transformers,wandb; print("deps OK")'
])

repo_dir = WORK / 'official_repo'

if not (repo_dir / '.git').exists():
    run_checked([
        'git', 'clone',
        'https://github.com/Quanquan429/CLAMEA.git',
        str(repo_dir)
    ])

run_checked([
    'git', '-C', str(repo_dir),
    'checkout', '--detach', PINNED_COMMIT
])

assert run_checked([
    'git', '-C', str(repo_dir), 'rev-parse', 'HEAD'
]) == PINNED_COMMIT

original_source = repo_dir / 'CLAMEA'
source_dir = WORK / 'patched_source' / 'CLAMEA'

if source_dir.exists():
    shutil.rmtree(source_dir)

shutil.copytree(original_source, source_dir)

# ============================================================
# 1. PATCH MODEL: VISUAL EMBEDDINGS 4096D
# ============================================================

model_path = source_dir / 'model.py'
model_txt = model_path.read_text(encoding='utf-8')

visual_anchor = 'self.fc_i = nn.Linear(2048, self.args.dim)'

assert model_txt.count(visual_anchor) == 1, \
    'Unexpected upstream visual layer'

model_txt = model_txt.replace(
    visual_anchor,
    'self.fc_i = nn.Linear(getattr(self.args, "visual_dim", 2048), self.args.dim)',
    1
)

model_path.write_text(model_txt, encoding='utf-8')

# ============================================================
# 2. PATCH RANDOM SEED
# ============================================================

utils_path = source_dir / 'utils.py'
utils_txt = utils_path.read_text(encoding='utf-8')

assert 'torch.cuda.manual_seed(42)' in utils_txt
assert 'torch.cuda.manual_seed_all(42)' in utils_txt

utils_txt = utils_txt.replace(
    'torch.cuda.manual_seed(42)',
    'torch.cuda.manual_seed(seed)', 1
)

utils_txt = utils_txt.replace(
    'torch.cuda.manual_seed_all(42)',
    'torch.cuda.manual_seed_all(seed)', 1
)

utils_path.write_text(utils_txt, encoding='utf-8')

# ============================================================
# 3. PATCH MAIN.PY
# ============================================================

main_path = source_dir / 'main.py'
main_txt = main_path.read_text(encoding='utf-8')

# F19: preserve gold training tensors
gold_anchor = '    label_ = torch.eye(len(kgs.train_entities1)).cuda()'

assert main_txt.count(gold_anchor) == 1

main_txt = main_txt.replace(
    gold_anchor,
    gold_anchor + '\n    gold_tensors = '
    '(train_e1, train_e2, at_mask1, at_mask2, '
    'im_mask1, im_mask2, label_)',
    1
)

restore_anchor = '            epoch_sup_celoss += sup_align_loss.item()'

assert main_txt.count(restore_anchor) == 1

main_txt = main_txt.replace(
    restore_anchor,
    restore_anchor
    + '\n            train_e1, train_e2, at_mask1, at_mask2, '
      'im_mask1, im_mask2, label_ = gold_tensors'
    + '\n            assert torch.equal(train_e1, gold_tensors[0])',
    1
)

# F7: accurate validation ranking
v0 = main_txt.index('def valid(')
v1 = main_txt.index('\ndef train(', v0)

valid_fn = main_txt[v0:v1]

assert valid_fn.count('accurate=False') == 1

main_txt = (
    main_txt[:v0]
    + valid_fn.replace('accurate=False', 'accurate=True', 1)
    + main_txt[v1:]
)

# Remove original Test call in training.
# Replace it with explicitly observational Test Hits@1.
train0 = main_txt.index('def train(')

assert main_txt[train0:].count(
    '            test(i, model, kgs, args)'
) == 1

main_txt = (
    main_txt[:train0]
    + main_txt[train0:].replace(
        '            test(i, model, kgs, args)',
        '            # Test monitoring is performed by '
        'monitor_test_hits1() above the best-checkpoint decision',
        1
    )
)

# ============================================================
# 4. ADD OBSERVATIONAL TEST MONITOR
# ============================================================

monitor_fn = """
def monitor_test_hits1(epoch, model, kgs, args):
    '''Observational raw Test Hits@1. NEVER use for selection/tuning.'''
    model.eval()

    # Preserve training RNG streams.
    _rng_torch = torch.get_rng_state()
    _rng_cuda = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else []
    )
    _rng_python = random.getstate()
    _rng_numpy = np.random.get_state()

    try:
        with torch.no_grad():
            e1 = torch.LongTensor(kgs.test_entities1).cuda()
            e2 = torch.LongTensor(kgs.test_entities2).cuda()

            at1 = torch.ByteTensor([
                kgs.attr_mask[x] for x in kgs.test_entities1
            ]).cuda()

            at2 = torch.ByteTensor([
                kgs.attr_mask[x] for x in kgs.test_entities2
            ]).cuda()

            im1 = torch.ByteTensor([
                kgs.image_mask[x] for x in kgs.test_entities1
            ]).cuda()

            im2 = torch.ByteTensor([
                kgs.image_mask[x] for x in kgs.test_entities2
            ]).cuda()

            embeds1, embeds2, *_ = model.predict(
                e1, e2, at1, im1, at2, im2
            )

            _, hits, _, _ = greedy_alignment(
                embeds1,
                embeds2,
                [1],
                args.test_threads_num,
                metric=args.eval_metric,
                normalize=args.eval_norm,
                csls_k=0,
                accurate=True
            )

            value = float(hits[0])

            assert np.isfinite(value), 'Non-finite Test Hits@1'

            print(
                '[TEST_MONITOR] epoch={} raw_hits1={:.6f}'.format(
                    epoch, value
                ),
                flush=True
            )

            return value

    finally:
        random.setstate(_rng_python)
        np.random.set_state(_rng_numpy)
        torch.set_rng_state(_rng_torch)

        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(_rng_cuda)


"""

main_txt = main_txt.replace(
    'def train(model, kgs, args, out_folder):',
    monitor_fn + 'def train(model, kgs, args, out_folder):',
    1
)

# ============================================================
# 5. RETURN VALIDATION HITS@1 AND MRR
# ============================================================

valid_return = (
    "    return hits1_12[0] if args.stop_metric == 'hits1' "
    "else mrr_12"
)

assert main_txt.count(valid_return) == 1

main_txt = main_txt.replace(
    valid_return,
    "    return (hits1_12[0] if args.stop_metric == 'hits1' "
    "else mrr_12), float(hits1_12[0])",
    1
)

# ============================================================
# 6. TRAINING LOSS LOGGING
# ============================================================

loss_anchor = '        loss_list.append(epoch_loss)'

assert main_txt.count(loss_anchor) == 1

loss_log_patch = """
        if os.environ.get('CLAMEA_WANDB_ENABLED') == '1':
            import wandb
            wandb.log({
                'train/total_loss': epoch_loss,
                'train/relation_loss': epoch_rloss,
                'train/alignment_loss': epoch_celoss,
                'train/pseudo_aux_loss': epoch_sup_celoss,
                'train/orth_loss': epoch_orth_loss,
                'train/vae_and_latent_loss': epoch_mmd_loss,
            }, step=i, commit=(i < args.start_valid))
"""

main_txt = main_txt.replace(
    loss_anchor,
    loss_anchor + loss_log_patch,
    1
)

# ============================================================
# 7. VALIDATION + TEST OBSERVATIONAL MONITORING
# ============================================================

val_anchor = '            flag = valid(i, model, kgs, args)'

assert main_txt.count(val_anchor) == 1

val_log_patch = """
            flag, val_hits1 = valid(i, model, kgs, args)

            print(
                '[VALID_MONITOR] epoch={} raw_hits1={:.6f} raw_mrr={:.8f}'.format(
                    i, float(val_hits1), float(flag)
                ),
                flush=True
            )

            test_hits1 = monitor_test_hits1(i, model, kgs, args)

            # Observational metrics only.
            # Best checkpoint selection still uses Validation MRR.
            with open(
                os.path.join(out_folder, 'epoch_eval.jsonl'),
                'a',
                encoding='utf-8'
            ) as _eval_file:
                _eval_file.write(json.dumps({
                    'epoch': int(i),
                    'validation_raw_hits1': float(val_hits1),
                    'validation_raw_mrr': float(flag),
                    'test_raw_hits1_observational': float(test_hits1),
                    'checkpoint_selection_metric': 'validation_raw_mrr',
                }, ensure_ascii=False) + '\\n')
                _eval_file.flush()

            if os.environ.get('CLAMEA_WANDB_ENABLED') == '1':
                import wandb
                wandb.log({
                    'val/raw_hits1': float(val_hits1),
                    'val/accurate_raw_mrr': float(flag),
                    'test_monitor/raw_hits1': float(test_hits1),
                }, step=i, commit=True)
"""

main_txt = main_txt.replace(
    val_anchor,
    val_log_patch,
    1
)

# ============================================================
# 8. RESUME SUPPORT
# ============================================================

resume_begin_anchor = (
    '    new_e1 = []\n'
    '    new_e2 = []\n'
    '    for i in range(1, args.max_epoch + 1):'
)

assert main_txt.count(resume_begin_anchor) == 1

resume_patch = """
    new_e1 = []
    new_e2 = []

    # Optional recovery from an independently saved state.
    _resume_file = os.environ.get('CLAMEA_RESUME_STATE')
    _start_epoch = 1

    if _resume_file:
        _saved = torch.load(
            _resume_file,
            map_location='cpu',
            weights_only=False
        )

        assert _saved['format'] == 'CLAMEA_METHOD_V12_STATE'
        assert _saved['random_seed'] == int(args.random_seed)
        assert _saved['epoch'] < args.max_epoch

        model.load_state_dict(
            _saved['model_state'],
            strict=True
        )

        # Restore validation-best model associated with checkpoint.
        if _saved['best_model_state'] is not None:
            torch.save(
                _saved['best_model_state'],
                os.path.join(out_folder, 'model_best.pkl')
            )
        else:
            _best_path = os.path.join(out_folder, 'model_best.pkl')
            if os.path.isfile(_best_path):
                os.remove(_best_path)

        optimizer.load_state_dict(_saved['optimizer_state'])

        max_mrr = _saved['max_mrr']
        flag1 = _saved['flag1']
        flag2 = _saved['flag2']

        new_e1 = _saved['new_e1']
        new_e2 = _saved['new_e2']

        rest_set_1 = _saved['rest_set_1']
        rest_set_2 = _saved['rest_set_2']

        kgs.relation_triples_list1 = _saved['relation_triples_list1']
        kgs.relation_triples_list2 = _saved['relation_triples_list2']

        kgs.relation_triples_set1 = set(kgs.relation_triples_list1)
        kgs.relation_triples_set2 = set(kgs.relation_triples_list2)

        random.setstate(_saved['python_rng'])
        np.random.set_state(_saved['numpy_rng'])
        torch.set_rng_state(_saved['torch_rng'])

        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(_saved['cuda_rng'])

        _start_epoch = int(_saved['epoch']) + 1

        print(
            'RESUMED TRAINING after epoch',
            _saved['epoch'],
            flush=True
        )

        del _saved

    for i in range(_start_epoch, args.max_epoch + 1):
"""

main_txt = main_txt.replace(
    resume_begin_anchor,
    resume_patch.rstrip('\n'),
    1
)

# ============================================================
# 9. PERIODIC RESUME CHECKPOINT
# ============================================================

finish_anchor = (
    '    print("Training ends. Total time = {:.3f} s."'
    '.format(time.time() - t))'
)

assert main_txt.count(finish_anchor) == 1

snapshot_patch = """
        # Save optimizer, model, pseudo-label and RNG state.
        if os.environ.get('CLAMEA_RESUME_SNAPSHOTS') == '1':

            _interval = max(
                1,
                int(os.environ.get('CLAMEA_RESUME_INTERVAL', '30'))
            )

            if (
                i % _interval == 0
                or i == args.start_valid
                or i == args.max_epoch
            ):
                _best_path = os.path.join(
                    out_folder, 'model_best.pkl'
                )

                _state = dict(
                    format='CLAMEA_METHOD_V12_STATE',
                    epoch=i,
                    random_seed=int(args.random_seed),

                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),

                    best_model_state=(
                        torch.load(
                            _best_path,
                            map_location='cpu',
                            weights_only=True
                        )
                        if os.path.isfile(_best_path)
                        else None
                    ),

                    max_mrr=max_mrr,
                    flag1=flag1,
                    flag2=flag2,

                    new_e1=new_e1,
                    new_e2=new_e2,

                    rest_set_1=rest_set_1,
                    rest_set_2=rest_set_2,

                    relation_triples_list1=kgs.relation_triples_list1,
                    relation_triples_list2=kgs.relation_triples_list2,

                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),

                    cuda_rng=(
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else []
                    )
                )

                _resume_out = os.path.join(
                    out_folder, 'resume_state.pt'
                )
                _temp_out = _resume_out + '.tmp'

                torch.save(_state, _temp_out)
                os.replace(_temp_out, _resume_out)

                print(
                    'SAVED RESUME STATE epoch',
                    i,
                    flush=True
                )

                del _state

"""

main_txt = main_txt.replace(
    finish_anchor,
    snapshot_patch + finish_anchor,
    1
)

# ============================================================
# 10. WRITE AND VERIFY PATCHES
# ============================================================

main_path.write_text(main_txt, encoding='utf-8')

compile(model_txt, str(model_path), 'exec')
compile(main_txt, str(main_path), 'exec')

assert 'gold_tensors = (train_e1, train_e2' in main_txt
assert 'label_ = gold_tensors' in main_txt
assert 'CLAMEA_METHOD_V12_STATE' in main_txt
assert 'optimizer_state=optimizer.state_dict()' in main_txt

assert 'accurate=True' in main_txt[
    main_txt.index('def valid('):main_txt.index('def train(')
]

assert 'monitor_test_hits1(i, model, kgs, args)' in main_txt
assert 'test_raw_hits1_observational' in main_txt
assert 'if flag > max_mrr:' in main_txt

assert main_txt.count(
    'monitor_test_hits1(i, model, kgs, args)'
) == 1

PATCH_INFO = {
    'base_commit': PINNED_COMMIT,

    'patched_main_sha256': file_sha256(main_path),
    'patched_model_sha256': file_sha256(model_path),
    'patched_utils_sha256': file_sha256(utils_path),

    'f19': 'gold tensors restored after each pseudo step',
    'f7': 'accurate validation MRR',

    'test_during_training': True,

    'test_monitor_policy': (
        'observational raw Hits@1 from start_valid; '
        'not used for checkpoint'
    ),

    'loss_policy': (
        'keep 4-branch alignment loss; no duplicate L_mcl'
    ),

    'resume_mode': (
        'periodic optimizer+model+pseudo/RNG snapshot, '
        'approximate replay on GPU'
    )
}

print(
    'SOURCE PATCH PASS:',
    json.dumps(PATCH_INFO, indent=2)
)

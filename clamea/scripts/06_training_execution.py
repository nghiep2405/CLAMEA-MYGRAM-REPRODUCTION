# import hashlib as _hashlib_for_run

# # ============================================================
# # 0. TRAINING CONFIGURATION
# # ============================================================

# APPROVAL = 'GO_AFTER_PREFLIGHT_REVIEW'

# TRAIN_SEED = 66

# # Empty = start a new run.
# RESUME_RUN_ID = ''

# # Offline avoids CoreWeave Forge initialization timeout.
# WANDB_TRAIN_MODE = 'offline'


# # ============================================================
# # 1. PREFLIGHT VALIDATION
# # ============================================================

# mo.stop(
#     APPROVAL != 'GO_AFTER_PREFLIGHT_REVIEW',
#     mo.md(
#         '🔒 Full training LOCKED. '
#         'Submit Preflight report to coordinator '
#         'before editing approval.'
#     )
# )

# assert PRECHECK['status'] == (
#     'PASS_MOLAB_PREFLIGHT_TECHNICAL'
# )

# assert TRAIN_SEED in (66, 190, 1234)

# assert WANDB_TRAIN_MODE in ('offline', 'online')

# assert torch.cuda.is_available()

# assert runner_path.exists()

# assert source_dir.exists()

# assert data_dir.exists()


# # ============================================================
# # 2. PREPARE TRAINING RUN
# # ============================================================

# if RESUME_RUN_ID:

#     # --------------------------------------------------------
#     # RESUME EXISTING RUN
#     # --------------------------------------------------------

#     run_id = RESUME_RUN_ID

#     run_dir = WORK / 'runs' / run_id

#     run_dir.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     remote_run = 'paper_method_runs/' + run_id

#     for fn in (
#         'config.json',
#         'manifest.json',
#         'out_folder.txt',
#         'resume_state.pt'
#     ):
#         assert fs_drive.exists(
#             remote_run + '/' + fn
#         ), 'Resume missing: ' + fn

#     for fn in (
#         'config.json',
#         'manifest.json',
#         'out_folder.txt'
#     ):
#         fs_drive.get_file(
#             remote_run + '/' + fn,
#             str(run_dir / fn)
#         )

#     manifest = json.loads(
#         (run_dir / 'manifest.json').read_text()
#     )

#     assert manifest['run_id'] == run_id
#     assert manifest['seed'] == TRAIN_SEED

#     assert (
#         manifest['dataset_sha256'] == DATA_SHA256
#     )

#     assert (
#         manifest['patched_main_sha256']
#         == PATCH_INFO['patched_main_sha256']
#     )

#     assert (
#         manifest['patched_model_sha256']
#         == PATCH_INFO['patched_model_sha256']
#     )

#     assert (
#         manifest['patched_utils_sha256']
#         == PATCH_INFO['patched_utils_sha256']
#     )

#     fs_drive.get_file(
#         remote_run + '/resume_state.pt',
#         str(run_dir / 'resume_state.pt')
#     )

#     out = Path(
#         (run_dir / 'out_folder.txt').read_text().strip()
#     )

#     out.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     if fs_drive.exists(
#         remote_run + '/model_best.pkl'
#     ):
#         fs_drive.get_file(
#             remote_run + '/model_best.pkl',
#             str(out / 'model_best.pkl')
#         )

#     if fs_drive.exists(
#         remote_run + '/epoch_eval.jsonl'
#     ):
#         fs_drive.get_file(
#             remote_run + '/epoch_eval.jsonl',
#             str(out / 'epoch_eval.jsonl')
#         )

#     train_config = json.loads(
#         (run_dir / 'config.json').read_text()
#     )

#     assert train_config['random_seed'] == TRAIN_SEED

#     print('[CONFIG] Resuming:', run_id)

# else:

#     # --------------------------------------------------------
#     # CREATE NEW RUN
#     # --------------------------------------------------------

#     run_id = (
#         time.strftime(
#             'paper_method_%Y%m%d_%H%M%S'
#         )
#         + '_s'
#         + str(TRAIN_SEED)
#     )

#     run_dir = WORK / 'runs' / run_id

#     run_dir.mkdir(
#         parents=True,
#         exist_ok=False
#     )

#     train_config = dict(base_config)

#     train_config['random_seed'] = TRAIN_SEED

#     train_config['output'] = (
#         str(run_dir / 'results') + '/'
#     )

#     (
#         run_dir / 'config.json'
#     ).write_text(
#         json.dumps(train_config, indent=2)
#     )

#     manifest = {
#         'run_id': run_id,
#         'seed': TRAIN_SEED,

#         'wandb_id': (
#             _hashlib_for_run.sha256(
#                 run_id.encode()
#             ).hexdigest()[:8]
#         ),

#         'source_commit': PINNED_COMMIT,

#         'dataset_sha256': DATA_SHA256,

#         'split_hashes': {
#             k: v[0]
#             for k, v in EXPECTED_SPLITS.items()
#         },

#         'patched_main_sha256': (
#             PATCH_INFO['patched_main_sha256']
#         ),

#         'patched_model_sha256': (
#             PATCH_INFO['patched_model_sha256']
#         ),

#         'patched_utils_sha256': (
#             PATCH_INFO['patched_utils_sha256']
#         ),

#         'protocol': (
#             'alternative FBDB15K Protocol-B; '
#             'transductive unlabeled self-training'
#         ),

#         'encoder': 'VGG16 4096d',

#         'claim': (
#             'paper method adaptation, not exact Table1'
#         )
#     }

#     (
#         run_dir / 'manifest.json'
#     ).write_text(
#         json.dumps(manifest, indent=2)
#     )

#     remote_run = 'paper_method_runs/' + run_id

#     fs_drive.makedirs(
#         remote_run,
#         exist_ok=False
#     )

#     print('[CONFIG] New run:', run_id)


# # ============================================================
# # 3. INITIAL DRIVE BACKUP
# # ============================================================

# for fn in ('config.json', 'manifest.json'):

#     fs_drive.put_file(
#         str(run_dir / fn),
#         remote_run + '/' + fn
#     )

# (
#     run_dir / 'TRAIN_STARTED.lock'
# ).write_text('started')

# fs_drive.put_file(
#     str(run_dir / 'TRAIN_STARTED.lock'),
#     remote_run + '/TRAIN_STARTED.lock'
# )


# # ============================================================
# # 4. PREPARE SUBPROCESS
# # ============================================================

# log_file = run_dir / 'train.log'

# cmd = [
#     sys.executable,
#     '-u',
#     str(runner_path),
#     str(run_dir),
#     str(source_dir)
# ]

# training_env = os.environ.copy()

# # Prevent reuse of Marimo W&B service.
# training_env.pop('WANDB_SERVICE', None)

# training_env['WANDB_MODE'] = WANDB_TRAIN_MODE

# training_env['WANDB_DIR'] = str(run_dir)

# training_env['WANDB_INIT_TIMEOUT'] = '180'

# if WANDB_TRAIN_MODE == 'offline':
#     training_env.pop('WANDB_API_KEY', None)


# print(
#     '[CONFIG] W&B:',
#     WANDB_TRAIN_MODE,
#     '| GPU:',
#     torch.cuda.get_device_name(0),
#     '| seed:',
#     TRAIN_SEED,
#     '| max_epoch:',
#     train_config['max_epoch'],
#     flush=True
# )

# print(
#     '[CONFIG] Training log:',
#     log_file,
#     flush=True
# )

# print(
#     '[CONFIG] Test monitor: raw Hits@1 for every '
#     'epoch >= 70; Validation MRR still selects best',
#     flush=True
# )


# # ============================================================
# # 5. LIVE LOG TRACKER
# # ============================================================

# progress = {
#     'offset': 0,
#     'last_epoch': 0,
#     'completed_epoch': 0,
#     'last_text_at': time.monotonic(),
#     'last_warning_at': 0.0
# }


# def display_new_training_lines():

#     if not log_file.is_file():
#         return

#     with log_file.open(
#         'r',
#         encoding='utf-8',
#         errors='replace'
#     ) as f:

#         if progress['offset'] > log_file.stat().st_size:
#             progress['offset'] = 0

#         f.seek(progress['offset'])

#         while True:

#             position = f.tell()
#             raw = f.readline()

#             if not raw:
#                 break

#             # Wait for complete line.
#             if not raw.endswith('\n'):
#                 f.seek(position)
#                 break

#             progress['offset'] = f.tell()

#             line = raw.strip()

#             if not line:
#                 continue

#             progress['last_text_at'] = time.monotonic()

#             # --------------------------------------------
#             # EPOCH START
#             # --------------------------------------------

#             if (
#                 line.startswith('epoch ')
#                 and line.endswith(':')
#             ):

#                 try:
#                     progress['last_epoch'] = int(
#                         line.split()[1][:-1]
#                     )

#                 except (ValueError, IndexError):
#                     pass

#                 print(
#                     '[EPOCH]',
#                     line,
#                     flush=True
#                 )

#             # --------------------------------------------
#             # TRAINING LOSS
#             # --------------------------------------------

#             elif line.startswith('loss:'):

#                 progress['completed_epoch'] = (
#                     progress['last_epoch']
#                 )

#                 print(
#                     f"[TRAIN {progress['last_epoch']:03d}/"
#                     f"{train_config['max_epoch']}] {line}",
#                     flush=True
#                 )

#             # --------------------------------------------
#             # VALIDATION METRICS
#             # --------------------------------------------

#             elif line.startswith('accurate results:'):

#                 print(
#                     '[METRIC]',
#                     line,
#                     flush=True
#                 )

#             elif line.startswith('[VALID_MONITOR]'):

#                 print(
#                     line,
#                     flush=True
#                 )

#             # --------------------------------------------
#             # TEST METRICS
#             # --------------------------------------------

#             elif line.startswith('[TEST_MONITOR]'):

#                 print(
#                     line,
#                     flush=True
#                 )

#             # --------------------------------------------
#             # BEST CHECKPOINT
#             # --------------------------------------------

#             elif line.startswith('best '):

#                 print(
#                     '[BEST CHECKPOINT]',
#                     line,
#                     flush=True
#                 )

#             # --------------------------------------------
#             # SELF-TRAINING
#             # --------------------------------------------

#             elif line.startswith(
#                 'generate new semi-pairs:'
#             ):

#                 print(
#                     '[SELF-TRAIN]',
#                     line,
#                     flush=True
#                 )

#             # --------------------------------------------
#             # RESUME CHECKPOINT
#             # --------------------------------------------

#             elif line.startswith(
#                 'SAVED RESUME STATE'
#             ):

#                 print(
#                     '[RESUME]',
#                     line,
#                     flush=True
#                 )

#             # --------------------------------------------
#             # TRAINING STAGES
#             # --------------------------------------------

#             elif line.startswith((
#                 'BEGIN TRAINING',
#                 'read relation triples:',
#                 'read image embeddings:',
#                 'read attr embeddings:',
#                 'KG statistics:',
#                 'results output folder:',
#                 'FINAL TEST',
#                 'Training ends.'
#             )):

#                 print(
#                     '[STAGE]',
#                     line,
#                     flush=True
#                 )

#             # --------------------------------------------
#             # ERRORS
#             # --------------------------------------------

#             elif (
#                 line.startswith((
#                     'Traceback',
#                     'wandb: ERROR'
#                 ))
#                 or 'Error:' in line
#                 or 'ERROR' in line
#             ):

#                 print(
#                     '[ERROR]',
#                     line,
#                     flush=True
#                 )


# # ============================================================
# # 6. GOOGLE DRIVE SYNCHRONIZATION
# # ============================================================

# uploaded_signatures = {}

# offline_upload = {
#     'last': time.monotonic()
# }


# def sync_training_artifacts(*, final=False):

#     # --------------------------------------------------------
#     # SYNC TRAINING LOG
#     # --------------------------------------------------------

#     if log_file.is_file():

#         fs_drive.put_file(
#             str(log_file),
#             remote_run + '/train.log'
#         )

#     # --------------------------------------------------------
#     # SYNC EVALUATION HISTORY AND CHECKPOINTS
#     # --------------------------------------------------------

#     output_txt = run_dir / 'out_folder.txt'

#     if output_txt.is_file():

#         out_path = Path(
#             output_txt.read_text().strip()
#         )

#         curve_path = (
#             out_path / 'epoch_eval.jsonl'
#         )

#         if curve_path.is_file():

#             fs_drive.put_file(
#                 str(curve_path),
#                 remote_run + '/epoch_eval.jsonl'
#             )

#         for name in (
#             'model_best.pkl',
#             'resume_state.pt'
#         ):

#             obj = out_path / name

#             if not obj.is_file():
#                 continue

#             # Avoid copying a file still being written.
#             if (
#                 not final
#                 and time.time() - obj.stat().st_mtime <= 45
#             ):
#                 continue

#             signature = (
#                 obj.stat().st_size,
#                 obj.stat().st_mtime_ns
#             )

#             if (
#                 not final
#                 and uploaded_signatures.get(name)
#                 == signature
#             ):
#                 continue

#             snapshot = (
#                 run_dir / ('upload_' + name)
#             )

#             try:

#                 shutil.copy2(
#                     obj,
#                     snapshot
#                 )

#                 if (
#                     snapshot.stat().st_size
#                     != signature[0]
#                 ):
#                     raise RuntimeError(
#                         'Checkpoint changed while copying'
#                     )

#                 fs_drive.put_file(
#                     str(snapshot),
#                     remote_run + '/' + name
#                 )

#                 uploaded_signatures[name] = signature

#                 print(
#                     f'[DRIVE] {name} backed up '
#                     f'({signature[0]} bytes)',
#                     flush=True
#                 )

#             finally:

#                 snapshot.unlink(
#                     missing_ok=True
#                 )

#     # --------------------------------------------------------
#     # BACKUP OFFLINE W&B
#     # --------------------------------------------------------

#     offline_root = run_dir / 'wandb'

#     if (
#         WANDB_TRAIN_MODE == 'offline'
#         and offline_root.is_dir()
#     ):

#         now = time.monotonic()

#         if (
#             final
#             or now - offline_upload['last'] >= 600
#         ):

#             backup_zip = Path(
#                 shutil.make_archive(
#                     str(
#                         run_dir / 'wandb_offline_backup'
#                     ),
#                     'zip',
#                     root_dir=str(run_dir),
#                     base_dir='wandb'
#                 )
#             )

#             fs_drive.put_file(
#                 str(backup_zip),
#                 remote_run + '/wandb_offline_backup.zip'
#             )

#             offline_upload['last'] = now

#             print(
#                 '[DRIVE] Offline W&B backup uploaded',
#                 flush=True
#             )


# # ============================================================
# # 7. START TRAINING
# # ============================================================

# with log_file.open(
#     'a' if RESUME_RUN_ID else 'w',
#     encoding='utf-8',
#     buffering=1
# ) as stream:

#     proc = subprocess.Popen(
#         cmd,
#         cwd=str(source_dir),
#         stdout=stream,
#         stderr=subprocess.STDOUT,
#         env=training_env
#     )

#     last_sync_at = time.monotonic()

#     try:

#         while True:

#             # Read newly written log lines.
#             display_new_training_lines()

#             now = time.monotonic()

#             # --------------------------------------------
#             # SYNC EVERY ~90 SECONDS
#             # --------------------------------------------

#             if now - last_sync_at >= 90:

#                 try:
#                     sync_training_artifacts()

#                 except Exception as exc:

#                     print(
#                         '[DRIVE WARNING]',
#                         repr(exc),
#                         flush=True
#                     )

#                 last_sync_at = time.monotonic()

#             # --------------------------------------------
#             # NO-PROGRESS WARNING
#             # --------------------------------------------

#             if (
#                 proc.poll() is None
#                 and now - progress['last_text_at'] > 300
#                 and now - progress['last_warning_at'] > 300
#             ):

#                 print(
#                     '[WARNING] No new train.log output '
#                     'for >5 min; last completed epoch='
#                     f"{progress['completed_epoch']}. "
#                     'Check GPU and child process.',
#                     flush=True
#                 )

#                 progress['last_warning_at'] = now

#             # --------------------------------------------
#             # CHECK PROCESS TERMINATION
#             # --------------------------------------------

#             if proc.poll() is not None:

#                 display_new_training_lines()
#                 break

#             time.sleep(3)

#     except BaseException:

#         proc.terminate()

#         try:
#             proc.wait(timeout=20)

#         except subprocess.TimeoutExpired:
#             proc.kill()

#         display_new_training_lines()

#         try:
#             sync_training_artifacts(final=True)

#         except Exception as exc:

#             print(
#                 '[DRIVE WARNING DURING INTERRUPT]',
#                 repr(exc),
#                 flush=True
#             )

#         raise

#     returncode = proc.wait()


# # ============================================================
# # 8. PROCESS COMPLETION
# # ============================================================

# print(
#     f"[PROCESS EXIT] code={returncode}; "
#     f"completed_epoch={progress['completed_epoch']}/"
#     f"{train_config['max_epoch']}",
#     flush=True
# )


# # ============================================================
# # 9. UPLOAD FINAL ARTIFACTS
# # ============================================================

# for name in (
#     'config.json',
#     'manifest.json',
#     'out_folder.txt',
#     'training_summary.json'
# ):

#     local_file = run_dir / name

#     if local_file.is_file():

#         fs_drive.put_file(
#             str(local_file),
#             remote_run + '/' + name
#         )


# try:

#     sync_training_artifacts(final=True)

# except Exception as exc:

#     print(
#         '[DRIVE FINAL SYNC WARNING]',
#         repr(exc),
#         flush=True
#     )


# # ============================================================
# # 10. VERIFY EXIT CODE
# # ============================================================

# if returncode != 0:

#     raise RuntimeError(
#         f"Training process exited {returncode}; "
#         f"last completed epoch="
#         f"{progress['completed_epoch']}. "
#         "Inspect train.log "
#         "(printed errors above) and Drive."
#     )


# print(
#     'TRAINING COMPLETE; Drive folder:',
#     remote_run
# )

import hashlib as _hashlib_for_run

# ============================================================
# 0. TRAINING CONFIGURATION
# ============================================================

APPROVAL = "GO_AFTER_PREFLIGHT_REVIEW"

TRAIN_SEED = 66

# Empty = start a NEW run.
RESUME_RUN_ID = ""

# Offline avoids CoreWeave Forge initialization timeout.
WANDB_TRAIN_MODE = "offline"

# ============================================================
# 1. PREFLIGHT VALIDATION
# ============================================================

mo.stop(
    APPROVAL != "GO_AFTER_PREFLIGHT_REVIEW",
    mo.md("🔒 Full training LOCKED.")
)

assert PRECHECK["status"] == "PASS_MOLAB_PREFLIGHT_TECHNICAL"
assert TRAIN_SEED in (66, 190, 1234)
assert WANDB_TRAIN_MODE in ("offline", "online")

assert torch.cuda.is_available()
assert runner_path.exists()
assert source_dir.exists()
assert data_dir.exists()

# ============================================================
# 2. PREPARE TRAINING RUN
# ============================================================

if RESUME_RUN_ID:

    # --------------------------------------------------------
    # RESUME EXISTING RUN
    # --------------------------------------------------------

    run_id = RESUME_RUN_ID
    run_dir = WORK / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    remote_run = "paper_method_runs/" + run_id

    for fn in (
        "config.json",
        "manifest.json",
        "out_folder.txt",
        "resume_state.pt",
    ):
        assert fs_drive.exists(
            remote_run + "/" + fn
        ), "Resume missing: " + fn

    for fn in (
        "config.json",
        "manifest.json",
        "out_folder.txt",
    ):
        fs_drive.get_file(
            remote_run + "/" + fn,
            str(run_dir / fn),
        )

    manifest = json.loads(
        (run_dir / "manifest.json").read_text()
    )

    assert manifest["run_id"] == run_id
    assert manifest["seed"] == TRAIN_SEED
    assert manifest["dataset_sha256"] == DATA_SHA256

    assert (
        manifest["patched_main_sha256"]
        == PATCH_INFO["patched_main_sha256"]
    )
    assert (
        manifest["patched_model_sha256"]
        == PATCH_INFO["patched_model_sha256"]
    )
    assert (
        manifest["patched_utils_sha256"]
        == PATCH_INFO["patched_utils_sha256"]
    )

    fs_drive.get_file(
        remote_run + "/resume_state.pt",
        str(run_dir / "resume_state.pt"),
    )

    out = Path(
        (run_dir / "out_folder.txt").read_text().strip()
    )
    out.mkdir(parents=True, exist_ok=True)

    if fs_drive.exists(remote_run + "/model_best.pkl"):
        fs_drive.get_file(
            remote_run + "/model_best.pkl",
            str(out / "model_best.pkl"),
        )

    if fs_drive.exists(remote_run + "/epoch_eval.jsonl"):
        fs_drive.get_file(
            remote_run + "/epoch_eval.jsonl",
            str(out / "epoch_eval.jsonl"),
        )

    train_config = json.loads(
        (run_dir / "config.json").read_text()
    )

    assert train_config["random_seed"] == TRAIN_SEED
    assert train_config["iterative"] is False

    print("[CONFIG] Resuming:", run_id)

else:

    # --------------------------------------------------------
    # CREATE NEW ABLATION RUN
    # --------------------------------------------------------

    run_id = (
        time.strftime("ablation_no_iter_%Y%m%d_%H%M%S")
        + "_s"
        + str(TRAIN_SEED)
    )

    run_dir = WORK / "runs" / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    train_config = dict(base_config)

    train_config["random_seed"] = TRAIN_SEED

    # ========================================================
    # STAGE 4 — DIAGNOSTIC ABLATION
    # ========================================================

    train_config["iterative"] = False
    train_config["max_epoch"] = 120

    # Keep all other model hyperparameters unchanged.

    train_config["output"] = (
        str(run_dir / "results") + "/"
    )

    (run_dir / "config.json").write_text(
        json.dumps(train_config, indent=2)
    )

    manifest = {
        "run_id": run_id,
        "seed": TRAIN_SEED,

        "wandb_id": _hashlib_for_run.sha256(
            run_id.encode()
        ).hexdigest()[:8],

        "source_commit": PINNED_COMMIT,
        "dataset_sha256": DATA_SHA256,

        "split_hashes": {
            k: v[0]
            for k, v in EXPECTED_SPLITS.items()
        },

        "patched_main_sha256": (
            PATCH_INFO["patched_main_sha256"]
        ),
        "patched_model_sha256": (
            PATCH_INFO["patched_model_sha256"]
        ),
        "patched_utils_sha256": (
            PATCH_INFO["patched_utils_sha256"]
        ),

        "protocol": (
            "alternative FBDB15K Protocol-B; iterative=False"
        ),
        "encoder": "VGG16 4096d",

        "claim": "diagnostic_no_self_training_ablation",

        "control_run": "paper_method_20261023_160901_s66",
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    remote_run = "paper_method_runs/" + run_id

    fs_drive.makedirs(
        remote_run,
        exist_ok=False,
    )

    print("[CONFIG] New ablation run:", run_id)

# ============================================================
# 3. INITIAL DRIVE BACKUP
# ============================================================

for fn in ("config.json", "manifest.json"):

    fs_drive.put_file(
        str(run_dir / fn),
        remote_run + "/" + fn,
    )

(run_dir / "TRAIN_STARTED.lock").write_text("started")

fs_drive.put_file(
    str(run_dir / "TRAIN_STARTED.lock"),
    remote_run + "/TRAIN_STARTED.lock",
)

# ============================================================
# 4. PREPARE SUBPROCESS
# ============================================================

log_file = run_dir / "train.log"

cmd = [
    sys.executable,
    "-u",
    str(runner_path),
    str(run_dir),
    str(source_dir),
]

training_env = os.environ.copy()

training_env.pop("WANDB_SERVICE", None)

training_env["WANDB_MODE"] = WANDB_TRAIN_MODE
training_env["WANDB_DIR"] = str(run_dir)
training_env["WANDB_INIT_TIMEOUT"] = "180"

if WANDB_TRAIN_MODE == "offline":
    training_env.pop("WANDB_API_KEY", None)

print(
    "[CONFIG] W&B:",
    WANDB_TRAIN_MODE,
    "| GPU:",
    torch.cuda.get_device_name(0),
    "| seed:",
    TRAIN_SEED,
    "| max_epoch:",
    train_config["max_epoch"],
    flush=True,
)

print(
    "[CONFIG] Training log:",
    log_file,
    flush=True,
)

print(
    "[CONFIG] ABLATION: iterative=False | "
    "Test raw Hits@1 for epochs >= 70 | "
    "best=Validation MRR",
    flush=True,
)

# ============================================================
# 5. LIVE LOG TRACKER
# ============================================================

progress = {
    "offset": 0,
    "last_epoch": 0,
    "completed_epoch": 0,
    "last_text_at": time.monotonic(),
    "last_warning_at": 0.0,
}


def display_new_training_lines():

    if not log_file.is_file():
        return

    with log_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        if progress["offset"] > log_file.stat().st_size:
            progress["offset"] = 0

        f.seek(progress["offset"])

        while True:

            position = f.tell()
            raw = f.readline()

            if not raw:
                break

            if not raw.endswith("\n"):
                f.seek(position)
                break

            progress["offset"] = f.tell()
            line = raw.strip()

            if not line:
                continue

            progress["last_text_at"] = time.monotonic()

            # --------------------------------------------
            # EPOCH START
            # --------------------------------------------

            if (
                line.startswith("epoch ")
                and line.endswith(":")
            ):

                try:
                    progress["last_epoch"] = int(
                        line.split()[1][:-1]
                    )
                except (ValueError, IndexError):
                    pass

                print(
                    "[EPOCH]",
                    line,
                    flush=True,
                )

            # --------------------------------------------
            # TRAIN LOSS
            # --------------------------------------------

            elif line.startswith("loss:"):

                progress["completed_epoch"] = (
                    progress["last_epoch"]
                )

                print(
                    f"[TRAIN {progress['last_epoch']:03d}/"
                    f"{train_config['max_epoch']}] {line}",
                    flush=True,
                )

            # --------------------------------------------
            # VALIDATION
            # --------------------------------------------

            elif line.startswith("accurate results:"):
                print("[METRIC]", line, flush=True)

            elif line.startswith("[VALID_MONITOR]"):
                print(line, flush=True)

            # --------------------------------------------
            # TEST MONITOR
            # --------------------------------------------

            elif line.startswith("[TEST_MONITOR]"):
                print(line, flush=True)

            # --------------------------------------------
            # CHECKPOINT
            # --------------------------------------------

            elif line.startswith("best "):
                print(
                    "[BEST CHECKPOINT]",
                    line,
                    flush=True,
                )

            elif line.startswith(
                "generate new semi-pairs:"
            ):
                print(
                    "[SELF-TRAIN]",
                    line,
                    flush=True,
                )

            elif line.startswith(
                "SAVED RESUME STATE"
            ):
                print(
                    "[RESUME]",
                    line,
                    flush=True,
                )

            # --------------------------------------------
            # TRAINING STAGES
            # --------------------------------------------

            elif line.startswith((
                "BEGIN TRAINING",
                "read relation triples:",
                "read image embeddings:",
                "read attr embeddings:",
                "KG statistics:",
                "results output folder:",
                "FINAL TEST",
                "Training ends.",
            )):
                print(
                    "[STAGE]",
                    line,
                    flush=True,
                )

            # --------------------------------------------
            # ERROR HANDLING
            # --------------------------------------------

            elif (
                line.startswith(
                    ("Traceback", "wandb: ERROR")
                )
                or "Error:" in line
                or "ERROR" in line
            ):
                print(
                    "[ERROR]",
                    line,
                    flush=True,
                )


# ============================================================
# 6. GOOGLE DRIVE SYNCHRONIZATION
# ============================================================

uploaded_signatures = {}

offline_upload = {
    "last": time.monotonic()
}


def sync_training_artifacts(*, final=False):

    # --------------------------------------------------------
    # TRAIN LOG
    # --------------------------------------------------------

    if log_file.is_file():

        fs_drive.put_file(
            str(log_file),
            remote_run + "/train.log",
        )

    # --------------------------------------------------------
    # EVALUATION HISTORY + CHECKPOINTS
    # --------------------------------------------------------

    output_txt = run_dir / "out_folder.txt"

    if output_txt.is_file():

        out_path = Path(
            output_txt.read_text().strip()
        )

        curve_path = out_path / "epoch_eval.jsonl"

        if curve_path.is_file():

            fs_drive.put_file(
                str(curve_path),
                remote_run + "/epoch_eval.jsonl",
            )

        for name in (
            "model_best.pkl",
            "resume_state.pt",
        ):

            obj = out_path / name

            if not obj.is_file():
                continue

            # Avoid copying an actively written checkpoint.
            if (
                not final
                and time.time() - obj.stat().st_mtime <= 45
            ):
                continue

            signature = (
                obj.stat().st_size,
                obj.stat().st_mtime_ns,
            )

            if (
                not final
                and uploaded_signatures.get(name)
                == signature
            ):
                continue

            snapshot = run_dir / ("upload_" + name)

            try:

                shutil.copy2(obj, snapshot)

                if (
                    snapshot.stat().st_size
                    != signature[0]
                ):
                    raise RuntimeError(
                        "Checkpoint changed while copying"
                    )

                fs_drive.put_file(
                    str(snapshot),
                    remote_run + "/" + name,
                )

                uploaded_signatures[name] = signature

                print(
                    f"[DRIVE] {name} backed up "
                    f"({signature[0]} bytes)",
                    flush=True,
                )

            finally:

                snapshot.unlink(missing_ok=True)

    # --------------------------------------------------------
    # OFFLINE W&B BACKUP
    # --------------------------------------------------------

    offline_root = run_dir / "wandb"

    if (
        WANDB_TRAIN_MODE == "offline"
        and offline_root.is_dir()
    ):

        now = time.monotonic()

        if (
            final
            or now - offline_upload["last"] >= 600
        ):

            backup_zip = Path(
                shutil.make_archive(
                    str(
                        run_dir / "wandb_offline_backup"
                    ),
                    "zip",
                    root_dir=str(run_dir),
                    base_dir="wandb",
                )
            )

            fs_drive.put_file(
                str(backup_zip),
                remote_run + "/wandb_offline_backup.zip",
            )

            offline_upload["last"] = now

            print(
                "[DRIVE] Offline W&B backup uploaded",
                flush=True,
            )


# ============================================================
# 7. START TRAINING
# ============================================================

with log_file.open(
    "a" if RESUME_RUN_ID else "w",
    encoding="utf-8",
    buffering=1,
) as stream:

    proc = subprocess.Popen(
        cmd,
        cwd=str(source_dir),
        stdout=stream,
        stderr=subprocess.STDOUT,
        env=training_env,
    )

    last_sync_at = time.monotonic()

    try:

        while True:

            # Read newly written log lines.
            display_new_training_lines()

            now = time.monotonic()

            # --------------------------------------------
            # SYNC EVERY ~90 SECONDS
            # --------------------------------------------

            if now - last_sync_at >= 90:

                try:
                    sync_training_artifacts()

                except Exception as exc:

                    print(
                        "[DRIVE WARNING]",
                        repr(exc),
                        flush=True,
                    )

                last_sync_at = time.monotonic()

            # --------------------------------------------
            # NO-PROGRESS WARNING
            # --------------------------------------------

            if (
                proc.poll() is None
                and now - progress["last_text_at"] > 300
                and now - progress["last_warning_at"] > 300
            ):

                print(
                    "[WARNING] No new train.log output "
                    "for >5 min; last completed epoch="
                    f"{progress['completed_epoch']}. "
                    "Check GPU and child process.",
                    flush=True,
                )

                progress["last_warning_at"] = now

            # --------------------------------------------
            # CHECK PROCESS TERMINATION
            # --------------------------------------------

            if proc.poll() is not None:
                display_new_training_lines()
                break

            time.sleep(3)

    except BaseException:

        proc.terminate()

        try:
            proc.wait(timeout=20)

        except subprocess.TimeoutExpired:
            proc.kill()

        display_new_training_lines()

        try:
            sync_training_artifacts(final=True)

        except Exception as exc:

            print(
                "[DRIVE WARNING DURING INTERRUPT]",
                repr(exc),
                flush=True,
            )

        raise

    returncode = proc.wait()

# ============================================================
# 8. PROCESS COMPLETION
# ============================================================

print(
    f"[PROCESS EXIT] code={returncode}; "
    f"completed_epoch={progress['completed_epoch']}/"
    f"{train_config['max_epoch']}",
    flush=True,
)

# ============================================================
# 9. UPLOAD FINAL ARTIFACTS
# ============================================================

for name in (
    "config.json",
    "manifest.json",
    "out_folder.txt",
    "training_summary.json",
):

    local_file = run_dir / name

    if local_file.is_file():

        fs_drive.put_file(
            str(local_file),
            remote_run + "/" + name,
        )

try:
    sync_training_artifacts(final=True)

except Exception as exc:

    print(
        "[DRIVE FINAL SYNC WARNING]",
        repr(exc),
        flush=True,
    )

# ============================================================
# 10. VERIFY EXIT CODE
# ============================================================

if returncode != 0:

    raise RuntimeError(
        f"Training process exited {returncode}; "
        f"last completed epoch="
        f"{progress['completed_epoch']}. "
        "Inspect train.log and Drive."
    )

print(
    "TRAINING COMPLETE; Drive folder:",
    remote_run,
    flush=True,
)

# ============================================================
# 11. FINAL METRICS
# ============================================================

summary_path = run_dir / "training_summary.json"

if summary_path.is_file():

    summary = json.loads(
        summary_path.read_text()
    )

    print(
        "[FINAL] status:",
        summary.get("status"),
        flush=True,
    )

    print(
        "[FINAL] raw:",
        summary.get("final_test", {}).get("raw"),
        flush=True,
    )

    print(
        "[FINAL] CSLS:",
        summary.get("final_test", {}).get("csls"),
        flush=True,
    )

preflight_dir = WORK / 'preflight'

preflight_dir.mkdir(exist_ok=True)

PRECHECK = {'status': 'PASS_MOLAB_PREFLIGHT_TECHNICAL', 'full_training_authorized': False, 'notebook_version': '1.2.4', 'paper_claim': 'METHOD_ADAPTATION_ON_ALTERNATIVE_DATA_NOT_TABLE1_REPRODUCTION', 'scientific_method_fidelity': 'BLOCKED_PENDING_LOSS_EQUATION_AUDIT', 'dataset': DATA_CHECK, 'environment': ENV_INFO, 'patch': PATCH_INFO, 'smoke': SMOKE, 'science_notes': ['Visual features are VGG16 4096d instead of paper ResNet 2048d.', 'Validation/test splits follow frozen Protocol B, not verified paper split.', 'Eq(27) four-branch structure maps to four ContrastiveLoss terms; Eq(35) cosine-distance and normalization parity remain UNVERIFIED. No duplicate term added.', 'Observational Test Hits@1 is evaluated from epoch 70 onward, NEVER for checkpoint selection or tuning; Validation MRR selects the best model.', 'Pseudo-label inference on unlabeled valid/test entities is transductive, not use of gold test links.', '3 independent training seeds are a project-level robustness requirement, not demonstrated paper setup.', 'Training state snapshots for restart may not preserve bitwise stochastic worker behavior.']}

preflight_file = preflight_dir / 'preflight_report.json'

preflight_file.write_text(json.dumps(PRECHECK, ensure_ascii=False, indent=2))

fs_drive.makedirs('preflight_paper_method', exist_ok=True)

fs_drive.put_file(str(preflight_file), 'preflight_paper_method/preflight_report.json')

fs_drive.put_file(str(source_dir / 'main.py'), 'preflight_paper_method/main_patched.py')

fs_drive.put_file(str(source_dir / 'model.py'), 'preflight_paper_method/model_patched.py')

fs_drive.put_file(str(source_dir / 'utils.py'), 'preflight_paper_method/utils_patched.py')

assert fs_drive.exists('preflight_paper_method/preflight_report.json')

with fs_drive.open('preflight_paper_method/preflight_report.json', 'rb') as _f:
    assert json.loads(_f.read().decode('utf-8'))['notebook_version'] == '1.2.4'

print('TECHNICAL PREFLIGHT = PASS; PAPER MATH PARITY = NOT VERIFIED; FULL TRAINING = LOCKED')

print('Drive: preflight_paper_method/preflight_report.json')

print(json.dumps({'smoke_loss': SMOKE['finite_loss'], 'grad_params': SMOKE['finite_grad_parameter_tensors'], 'dataset_sha256': DATA_SHA256}, indent=2))

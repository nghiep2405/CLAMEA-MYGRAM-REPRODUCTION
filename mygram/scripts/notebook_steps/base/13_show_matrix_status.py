status_path = DRIVE_ROOT / 'runs/experiments/matrix_status.json'
status = json.loads(status_path.read_text(encoding='utf-8'))
print(json.dumps(status['counts'], indent=2))
print('Safe session stop requires: no active child process, runner.lock absent, and the latest snapshot verified.')

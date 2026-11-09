TEST_LOG = DRIVE_ROOT / 'environment/latest_unittest.log'
test_result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], text=True, capture_output=True)
test_output = test_result.stdout + '\n' + test_result.stderr
TEST_LOG.write_text(test_output, encoding='utf-8')
print(test_output)
if test_result.returncode != 0:
    raise RuntimeError(f'Unit tests failed; complete log: {TEST_LOG}')
subprocess.run([sys.executable, 'scripts/run_experiments.py', '--store-root', str(DRIVE_ROOT / 'runs/experiments'), '--data-root', '/content/mygram_data', '--work-root', '/content/mygram_work/runs', '--device', 'cuda:0', '--dry-run'], check=True)

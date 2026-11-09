ENV_REPORT = DRIVE_ROOT / 'environment/latest_environment.json'
ENV_REPORT_LOCAL = WORKSPACE / 'environment.json'
ENV_REPORT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, 'scripts/check_colab_environment.py', '--require-cuda', '--archive', str(PMF_ARCHIVE), '--drive-root', str(DRIVE_ROOT), '--local-path', '/content', '--report', str(ENV_REPORT_LOCAL)], check=True)
shutil.copy2(ENV_REPORT_LOCAL, ENV_REPORT)

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements/colab.lock'], check=True)
DEPENDENCY_REPORT = DRIVE_ROOT / 'environment/latest_dependency_check.json'
DEPENDENCY_REPORT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, 'scripts/check_mygram_dependencies.py', '--report', str(DEPENDENCY_REPORT)], check=True)

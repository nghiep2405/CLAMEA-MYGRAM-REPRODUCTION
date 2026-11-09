"""Convert pip --report into an exact Windows wheel URL/hash lock (no resolver)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    lines = ['# Windows AMD64 / CPython 3.13 / native torch CPU.',
             '# Exact wheel URLs and SHA-256 from pip resolution; validation in environment.json.',
             '# Install with pip install --require-hashes --no-deps -r this-file.']
    for item in sorted(report['install'], key=lambda x: x['metadata']['name'].lower()):
        download = item['download_info']
        digest = download['archive_info']['hashes']['sha256']
        lines.append(f"{item['metadata']['name']} @ {download['url']} --hash=sha256:{digest}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

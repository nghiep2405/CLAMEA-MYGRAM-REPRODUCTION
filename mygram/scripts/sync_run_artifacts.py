"""Publish or restore one verified MyGram run directory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_store import publish_run, restore_run, validate_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    publish = sub.add_parser('publish')
    publish.add_argument('--local-run', type=Path, required=True)
    publish.add_argument('--store-run', type=Path, required=True)
    restore = sub.add_parser('restore')
    restore.add_argument('--store-run', type=Path, required=True)
    restore.add_argument('--local-run', type=Path, required=True)
    restore.add_argument('--no-previous-fallback', action='store_true')
    validate = sub.add_parser('validate')
    validate.add_argument('--store-run', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'publish':
        result = publish_run(args.local_run, args.store_run)
    elif args.command == 'restore':
        result = restore_run(
            args.store_run, args.local_run,
            allow_previous=not args.no_previous_fallback,
        )
    else:
        result = validate_snapshot(args.store_run)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

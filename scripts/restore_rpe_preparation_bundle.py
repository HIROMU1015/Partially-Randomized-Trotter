#!/usr/bin/env python3
"""Verify or restore a preparation input bundle from its frozen REQUEST."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trottertracks.rpe_preparation_retention import restore, restore_run, restore_resume_archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--request', type=Path)
    target.add_argument('--archived-run', type=Path, help='restore all delivery bundles for a frozen run')
    target.add_argument('--resume-run', type=Path, help='rehydrate entries reclaimed from this run')
    parser.add_argument('--state-directory', type=Path)
    parser.add_argument('--sha256')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--repository', type=Path, help='checkout retaining the fixed input ref and objects')
    source.add_argument('--archive', type=Path, help='standalone preparation input archive')
    parser.add_argument('--restore-into', type=Path, help='publish matching bytes to this artifact cache')
    args = parser.parse_args()
    try:
        if args.resume_run is not None:
            if args.state_directory is None or any((args.archive,args.repository,args.sha256,args.restore_into)):
                parser.error('--resume-run requires --state-directory and uses the central archive registry')
            result = restore_resume_archives(args.state_directory, args.resume_run)
        elif args.archived_run:
            if args.archive is None or args.sha256 is not None:
                parser.error('--archived-run needs --archive and no --sha256')
            result = restore_run(args.archived_run, args.archive, args.restore_into)
        else:
            if args.sha256 is None:
                parser.error('--request needs --sha256')
            result = restore(json.loads(args.request.read_bytes()), args.sha256, args.repository,
                             args.restore_into, archive=args.archive)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f'Input bundle reconstruction failed: {error}\n')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Input bundle reconstruction failed: {error}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

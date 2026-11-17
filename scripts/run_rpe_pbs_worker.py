#!/usr/bin/env python3
"""Run the existing durable worker inside one explicitly allocated PBS node."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.rpe_pbs_worker import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--allocation-dir', type=Path,
                        help='Persistent shared spool for this PBS allocation')
    parser.add_argument('--poll-seconds', type=float, default=1.)
    args = parser.parse_args()
    return run(args.profile, allocation_dir=args.allocation_dir,
               poll_seconds=args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())

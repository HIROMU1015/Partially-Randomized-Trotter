#!/usr/bin/env python3
"""Explicit one-campaign activation; do not run science before chat permission."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', help='full published scientific SOURCE commit')
    p.add_argument('--execute-one-campaign', action='store_true')
    args = p.parse_args(argv)
    if not args.execute_one_campaign:
        print('SCIENCE_NOT_STARTED; separate explicit permission and execution option required')
        return 0
    if not args.source:
        p.error('--source is required for execution')
    from trottertracks.algorithm_codesign.cs_v0_execution import run
    print(run(ROOT, args.source))
    return 0


if __name__ == '__main__':
    sys.exit(main())

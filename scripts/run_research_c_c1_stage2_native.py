#!/usr/bin/env python3
"""Byte preflight by default. Run science only after user permission in chat."""
import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.execute and args.output is None:
        parser.error('Fresh output required; obtain user permission and adopt conditions in chat before executing')
    if not args.execute and args.output:
        parser.error('Preflight accepts no output')
    from trottertracks import c1_stage2_execution as run
    v = run.preflight(ROOT, args.contract, args.input)
    if args.execute:
        result = run.execute(v, args.output)
        print(json.dumps(dict(status=result['status'], mandatory_stop=True)))
    else:
        print(json.dumps(dict(status='STAGE2_BYTE_PREFLIGHT_PASS_SCIENCE_UNAPPROVED',
                              proposal_adopted=False, new_fit=0, scientific_calls=0)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Saved-fit preflight by default; no fit/worker/optimizer dispatch exists."""
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
    for flag in ('contract', 'source', 'input'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.execute and (args.approval is None or args.output is None):
        parser.error('new explicit evaluation-only approval and fresh output are required')
    if not args.execute and (args.approval or args.output):
        parser.error('preflight accepts no approval/output')
    from trottertracks import c1_stage1_eval_only as run
    v = run.preflight(ROOT, args.contract, args.source, args.input)
    if args.execute:
        result = run.execute(v, args.approval, args.output)
        print(json.dumps(dict(status=result['status'], new_fit_calls=0, mandatory_stop=True)))
    else:
        print(json.dumps(dict(status='EVAL_ONLY_TECHNICAL_PREFLIGHT_PASS_NO_DECODE',
                              new_fit_calls=0, execution_authorized=False, grant_issued=False)))


if __name__ == '__main__':
    main()

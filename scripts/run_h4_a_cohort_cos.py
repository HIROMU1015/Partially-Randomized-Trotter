#!/usr/bin/env python3
"""Saved Pauli PR cosine cohort, 254 new compiles, one statistics batch, STOP."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('d1_private')
package.__path__ = [str(ROOT / 'src/trotterlib')]
sys.modules['d1_private'] = package
from d1_private import h4_a_cohort as D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'launch', 'worker', 'task'):
        modes.add_argument('--' + mode, action='store_true')
    parser.add_argument('--kind', choices=['compile'])
    parser.add_argument('--case', choices=D.CASES)
    args = parser.parse_args()
    os.environ.update(D.JOB_ENV)
    os.environ['MPLCONFIGDIR'] = str(D.BASE / 'mplconfig')
    if args.prepare:
        print(json.dumps(D.prepare()))
    elif args.launch:
        print(json.dumps(D.launch()))
    elif args.worker:
        sys.exit(D.worker())
    else:
        if args.kind != 'compile' or args.case is None:
            parser.error('task requires --kind compile --case INDEX')
        sys.exit(D.task_run(args.case))

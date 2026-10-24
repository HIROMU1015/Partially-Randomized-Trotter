#!/usr/bin/env python3
"""D2 one-shot only. No quantum imports before sealed authority."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType('d1_private')
pkg.__path__ = [str(ROOT/'src/trotterlib')]
sys.modules['d1_private'] = pkg
from d1_private import h4_d2_control as D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'launch', 'worker', 'task'):
        modes.add_argument('--'+mode, action='store_true')
    parser.add_argument('--kind', choices=['validate', 'build', 'compile'])
    parser.add_argument('--case', choices=list(D.CASES))
    args = parser.parse_args()
    os.environ['MPLCONFIGDIR'] = str(D.BASE/'mplconfig')
    if args.prepare:
        print(json.dumps(D.prepare()))
    elif args.launch:
        print(json.dumps(D.launch()))
    elif args.worker:
        sys.exit(D.worker())
    else:
        if not args.kind or not args.case:
            parser.error('task requires kind and case')
        sys.exit(D.task_run(args.kind, args.case))

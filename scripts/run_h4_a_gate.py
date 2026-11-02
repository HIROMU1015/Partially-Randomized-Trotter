#!/usr/bin/env python3
"""Exactly one saved order2 body, two fixed axis compiles, then STOP."""
import argparse
import json
import os
from pathlib import Path
import sys
import types
ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('d1_private');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules['d1_private']=pkg
from d1_private import h4_a_gate as D
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','launch','worker','task'):g.add_argument('--'+mode,action='store_true')
    p.add_argument('--kind',choices=['compile']);p.add_argument('--case',choices=D.AXES);args=p.parse_args()
    os.environ['MPLCONFIGDIR']=str(D.BASE/'mplconfig')
    if args.prepare:print(json.dumps(D.prepare()))
    elif args.launch:print(json.dumps(D.launch()))
    elif args.worker:sys.exit(D.worker())
    else:
        if args.kind!='compile' or not args.case:p.error('task requires compile kind and one axis')
        sys.exit(D.compile_task(args.case))

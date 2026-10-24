#!/usr/bin/env python3
"""256 new sine targets using saved controlled body; paired statistics; STOP."""
import argparse
import json
import os
from pathlib import Path
import sys
import types
ROOT=Path(__file__).resolve().parents[1]
p=types.ModuleType('d1_private');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
from d1_private import h4_a_axes as D
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','launch','worker','task'):g.add_argument('--'+mode,action='store_true')
    p.add_argument('--kind',choices=['compile']);p.add_argument('--case',choices=D.CASES);a=p.parse_args()
    os.environ.update(D.JOB_ENV);os.environ['MPLCONFIGDIR']=str(D.BASE/'mplconfig')
    if a.prepare:print(json.dumps(D.prepare()))
    elif a.launch:print(json.dumps(D.launch()))
    elif a.worker:sys.exit(D.worker())
    else:
        if a.kind!='compile' or not a.case:p.error('task requires compile kind and one new sine target')
        sys.exit(D.task_run(a.case))

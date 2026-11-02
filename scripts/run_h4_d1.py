#!/usr/bin/env python3
"""One-shot D1 only; all numerical imports follow a bound grant/claim."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('d1_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['d1_private']=pkg
from d1_private import h4_d1_control as D

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',metavar='APPROVAL_FILE')
    g.add_argument('--launch',action='store_true')
    g.add_argument('--worker',action='store_true')
    g.add_argument('--task',metavar='NAMESPACE')
    p.add_argument('--kind',choices=['build','compile']);p.add_argument('--variant')
    a=p.parse_args()
    os.environ['MPLCONFIGDIR']=str(D.BASE/'mplconfig')
    if a.prepare:print(json.dumps(D.prepare(a.prepare)))
    elif a.launch:print(json.dumps(D.launch()))
    elif a.worker:sys.exit(D.worker())
    else:
        if not a.kind or not a.variant:p.error('task requires kind and variant')
        sys.exit(D.task_run(a.task,a.kind,a.variant))

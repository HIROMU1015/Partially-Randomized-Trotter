#!/usr/bin/env python3
"""Approved finite D1 only. SOURCE/grant/claim checks precede quantum imports."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('d1_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['d1_private']=pkg
from d1_private import h4_d1_source_control as D

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--launch',action='store_true')
    g.add_argument('--worker',action='store_true');g.add_argument('--task',action='store_true')
    p.add_argument('--kind',choices=['validate','build','compile']);p.add_argument('--variant')
    a=p.parse_args();os.environ['MPLCONFIGDIR']=str(D.BASE/'mplconfig')
    if a.prepare:print(json.dumps(D.prepare()))
    elif a.launch:print(json.dumps(D.launch()))
    elif a.worker:sys.exit(D.worker())
    else:
        if not a.kind or not a.variant:p.error('task requires kind and variant')
        sys.exit(D.task_run(a.kind,a.variant))

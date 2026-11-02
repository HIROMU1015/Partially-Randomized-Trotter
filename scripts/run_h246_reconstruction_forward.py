"""Restore missing references and necessary H2 signals under new grants."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('forward',ROOT/'src/trotterlib/h246_reconstruction_forward.py')
forward=importlib.util.module_from_spec(spec)
spec.loader.exec_module(forward)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--system',choices=('H2','H4','H6'))
    a=p.parse_args()
    if a.prepare:r=forward.prepare(a.private)
    else:
        forward.m.require(a.system,'SYSTEM_REQUIRED');r=forward.execute(a.private,a.system)
    print(json.dumps(r))

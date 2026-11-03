#!/usr/bin/env python3
"""Passive by default; one fixed paired r32 run only under chat permission."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',required=True,type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    from trottertracks import c_near_practical_r32 as m
    if not a.execute:
        m.preflight(ROOT,a.contract);print(json.dumps(dict(status='PASSIVE_READY',scientific_calls=0,maximum_logical_units=m.MAX_CALLS)));return
    # One native32 cohort at a time; independent samples/decodes use the existing
    # resource plan and Qiskit mutex. No shared environment or global cache edits.
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    for key,leaf in [('XDG_CACHE_HOME','cache'),('TMPDIR','tmp')]:
        path=ROOT/'.cache/near-r32'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    m.run(ROOT,a.contract,execute_science=True)


if __name__=='__main__':main()

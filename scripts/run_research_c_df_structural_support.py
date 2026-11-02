#!/usr/bin/env python3
"""Proposed structural DF pilot; passive by default. New science is unapproved."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    from trottertracks import c_df_structural_execution as m
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',required=True,type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--fit-python',type=Path);p.add_argument('--worker',choices=['prepare','fit']);p.add_argument('--output',type=Path);a=p.parse_args()
    c=m.preflight(ROOT,a.contract,fit=bool(a.worker))
    if a.worker:
        if a.execute or a.output is None or a.output.resolve()!=ROOT/m.FRESH/a.worker:raise ValueError('WORKER_FRESH_SCOPE')
        m.worker(ROOT,c,a.output,a.worker)
    elif a.execute:
        if not a.fit_python:raise ValueError('FIT_PYTHON_REQUIRED')
        m.run(ROOT,a.contract,a.fit_python)
    else:print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=m.MAX_CALLS)))
if __name__=='__main__':main()

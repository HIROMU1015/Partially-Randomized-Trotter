#!/usr/bin/env python3
"""PR-aware DF Stage-A proposal; passive preflight by default."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true')
    mode.add_argument('--worker',choices=['prepare','baseline','proposed']);p.add_argument('--fit-python',type=Path)
    a=p.parse_args();from trottertracks import c_pr_aware_df_pilot as m
    c=m.preflight(ROOT,a.contract,a.input,fit=bool(a.worker))
    if not (a.execute or a.worker):
        print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=m.MAX_CALLS)));return
    if not a.output:p.error('Fresh output required')
    expected=ROOT/c['fresh_output']
    if a.worker:expected=expected/a.worker
    if a.output.resolve()!=expected.resolve() or a.output.exists():p.error('Contract-fixed fresh output required')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp'),('JAX_COMPILATION_CACHE_DIR','jax')]:
        d=ROOT/'.cache/pr-aware-df'/leaf;d.mkdir(parents=True,exist_ok=True);os.environ[key]=str(d)
    if a.worker=='prepare':m.prepare_worker(ROOT,a.output,c,a.input)
    elif a.worker:m.fit_worker(ROOT,a.output,c,a.worker)
    else:
        if not a.fit_python:p.error('fit-python required')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        m.run(ROOT,a.contract,a.output,a.input,a.fit_python)


if __name__=='__main__':main()

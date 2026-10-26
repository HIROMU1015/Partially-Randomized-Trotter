#!/usr/bin/env python3
"""Task-local B2 technical relaunch. Default is passive preflight only."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    from trottertracks import c_pr_aware_df_stage_b2_compat as m
    if not a.execute:
        m.preflight(ROOT,a.contract,a.input)
        print(json.dumps(dict(status='B2_COMPAT_PASSIVE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=m.base.MAX_CALLS)));return
    if a.output is None:p.error('Fresh run02 output required')
    for key,leaf in [('XDG_CACHE_HOME',''),('TMPDIR','tmp'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba')]:
        path=ROOT/'.cache/pr-aware-df-stage-b2-compat'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    m.run(ROOT,a.contract,a.input,a.output)

if __name__=='__main__':main()

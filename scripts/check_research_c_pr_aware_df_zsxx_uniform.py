#!/usr/bin/env python3
"""Passive SOURCE check or invented-domain evidence;no science execute entry."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--contract',type=Path)
    g.add_argument('--artificial-output',type=Path)
    a=p.parse_args()
    if a.contract:
        from trottertracks.c_pr_aware_df_zsxx_uniform import preflight
        print(json.dumps(preflight(ROOT,a.contract)))
    else:
        from trottertracks.c_pr_aware_df_zsxx_uniform_artificial import evidence
        result=evidence()
        with a.artificial_output.open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(dict(status=result['status'],new_H4_scientific_calls=0)))

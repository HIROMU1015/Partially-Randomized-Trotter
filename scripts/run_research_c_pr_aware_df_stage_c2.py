#!/usr/bin/env python3
"""C2B saved-table qualification proposal. Default passive hashes only."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    from trottertracks import c_pr_aware_df_stage_c2 as m
    if not a.execute:
        m.preflight(ROOT,a.contract)
        print(json.dumps(dict(status='C2B_PASSIVE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=m.MAX_CALLS)));return
    if a.output is None:p.error('Fresh output required')
    m.run(ROOT,a.contract,a.output)

if __name__=='__main__':main()

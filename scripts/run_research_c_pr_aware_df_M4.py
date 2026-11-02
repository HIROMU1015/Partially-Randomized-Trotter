#!/usr/bin/env python3
"""M4 r128 RTE/shots/breakeven only; passive preflight unless --execute."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    from trottertracks import c_pr_aware_df_m4 as m
    if a.execute:m.run(ROOT,a.contract)
    else:
        m.preflight(ROOT,a.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=m.MAX_CALLS)))

if __name__=='__main__':main()

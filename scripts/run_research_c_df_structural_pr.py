#!/usr/bin/env python3
"""Tail-P4 conditional Stage A→native32 program; passive by default."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    from trottertracks import c_df_structural_pr as m
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:m.run(ROOT,a.contract)
    else:
        m.preflight(ROOT,a.contract);print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=m.MAX_CALLS)))
if __name__=='__main__':main()

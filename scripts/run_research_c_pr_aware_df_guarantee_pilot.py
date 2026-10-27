#!/usr/bin/env python3
"""Passive by default;--execute is a separately permitted scientific action."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    from trottertracks.c_pr_aware_df_guarantee_execution import preflight,run
    print(json.dumps(run(ROOT,a.contract) if a.execute else preflight(ROOT,a.contract)))

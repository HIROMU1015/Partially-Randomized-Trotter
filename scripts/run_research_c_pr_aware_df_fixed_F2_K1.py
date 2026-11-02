#!/usr/bin/env python3
"""Fixed F2-K1; passive default. Science needs separate explicit chat permission."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    from trottertracks.c_pr_aware_df_fixed_k1_execution import preflight,run
    print(json.dumps(run(ROOT,a.contract) if a.execute else preflight(ROOT,a.contract)))

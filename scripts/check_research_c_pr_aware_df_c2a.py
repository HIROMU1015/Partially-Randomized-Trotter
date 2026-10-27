#!/usr/bin/env python3
"""C2-A passive identity check only. No science execution option exists."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    args=parser.parse_args()
    from trottertracks.c_pr_aware_df_c2a import preflight
    print(json.dumps(preflight(ROOT,args.contract)))

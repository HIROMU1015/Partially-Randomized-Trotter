#!/usr/bin/env python3
"""Passive preflight by default; --execute needs separate chat permission."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_roundwise_det_pr_saved as saved

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.execute:saved.run(ROOT,ROOT/args.contract)
    else:saved.preflight(ROOT,ROOT/args.contract);print('PASSIVE_PREFLIGHT_PASS; no scientific calls')

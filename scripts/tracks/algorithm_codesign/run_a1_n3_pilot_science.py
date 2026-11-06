#!/usr/bin/env python3
"""Adopted one-campaign pilot; no resume or scientific retry."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import a1_n3_pilot_activation as pilot
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--submit',action='store_true')
    for name in ('dispatch','manager','parent'):group.add_argument('--'+name)
    args=parser.parse_args()
    if args.submit:print(pilot.submit(ROOT),flush=True)
    else:
        name=next(n for n in ('dispatch','manager','parent') if getattr(args,n))
        raise SystemExit(getattr(pilot,name)(getattr(args,name)))

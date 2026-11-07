#!/usr/bin/env python3
"""Deferred 448-shot SOURCE. Scientific submit requires separate user permission."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import scv_signed_activation as activation
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--submit',action='store_true')
    for name in ('dispatch','manager','parent'):group.add_argument('--'+name)
    args=parser.parse_args()
    if args.submit:print(activation.submit(ROOT),flush=True)
    else:
        name=next(n for n in ('dispatch','manager','parent') if getattr(args,n))
        raise SystemExit(getattr(activation,name)(getattr(args,name)))

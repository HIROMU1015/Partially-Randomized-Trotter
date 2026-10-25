#!/usr/bin/env python3
"""Fixed-P0 cost-only rerun with task-local compiler mutex; default preflight."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.execute!=bool(args.output):p.error('--execute requires fresh --output')
    from trottertracks.c1_stage3_cost_compat import preflight,execute
    if not args.execute:
        preflight(ROOT,args.contract,args.input)
        print(json.dumps(dict(status='COST_COMPAT_BYTE_PREFLIGHT_PASS',scientific_calls=0)));return
    r=execute(ROOT,args.contract,args.input,args.output)
    print(json.dumps(dict(status=r['status'],scientific_units=r['scientific_units'],mandatory_stop=True)))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Eight-candidate cost-only entry. Default: byte preflight, no science."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['cost-only'],default='cost-only')
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.execute!=bool(args.output):p.error('--execute requires fresh --output')
    from trottertracks.c1_stage3_cost_only import preflight,execute
    if not args.execute:
        preflight(ROOT,args.contract,args.input)
        print(json.dumps(dict(status='COST_ONLY_BYTE_PREFLIGHT_PASS',scientific_calls=0)));return
    result=execute(ROOT,args.contract,args.input,args.output)
    print(json.dumps(dict(status=result['status'],scientific_units=result['scientific_units'],mandatory_stop=True)))


if __name__=='__main__':main()

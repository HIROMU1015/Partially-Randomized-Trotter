#!/usr/bin/env python3
"""Byte preflight by default. --execute requires separate science permission in chat."""
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
    p.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.execute != bool(args.output):p.error('--execute requires a fresh --output; preflight writes no science')
    from trottertracks.c1_stage3_integrated import preflight,execute
    if not args.execute:
        preflight(ROOT,args.contract,args.input)
        print(json.dumps(dict(status='INTEGRATED_BYTE_PREFLIGHT_PASS',scientific_calls=0,
            core_pilot_adopted=True,scientific_execution_authorized=False)))
        return
    result=execute(dict(root=ROOT,contract_path=args.contract,input=args.input),args.output)
    print(json.dumps(dict(status=result['status'],conclusion=result['conclusion'],
                         scientific_units=result['scientific_units'],mandatory_stop=True)))


if __name__=='__main__':main()

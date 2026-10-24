#!/usr/bin/env python3
"""Feasibility-only Stage3 entry. Default: bytes only; science needs chat permission."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['feasibility-only'],default='feasibility-only')
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    if args.execute != bool(args.output): p.error('--execute requires fresh --output; preflight writes no science')
    from trottertracks.c1_stage3_feasibility import preflight,execute
    if not args.execute:
        preflight(ROOT,args.contract,args.input)
        print(json.dumps(dict(status='FEASIBILITY_BYTE_PREFLIGHT_PASS',mode=args.mode,scientific_calls=0)))
        return
    result = execute(dict(root=ROOT,contract_path=args.contract,input=args.input),args.output)
    print(json.dumps(dict(status=result['status'],mode=args.mode,scientific_units=result['scientific_units'],
                          candidate_status_counts=result['candidate_status_counts'],mandatory_stop=True)))


if __name__=='__main__': main()

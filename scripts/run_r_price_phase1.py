"""Prepare metadata, or explicitly run one separately authorized Phase 1 stage."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import r_price_phase1 as m
from trottertracks.r_price_phase1_runner import execute_stage, read, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','calibrate','validate','rare'])
    parser.add_argument('--plan',required=True);parser.add_argument('--public-output');parser.add_argument('--private-output')
    parser.add_argument('--frozen-commit');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':
        if args.execute:parser.error('Preparation is metadata only')
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        plan=m.make_plan(ROOT,source);write(args.plan,plan)
        print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',normal_wrapper_budget=64,rare_wrapper_budget=16,new_science_calls=0)));return
    if not args.execute:parser.error('Scientific stage needs --execute AND separately authorized chat scope')
    if not args.public_output or not args.private_output:parser.error('Explicit fresh public/private outputs required')
    print(json.dumps(execute_stage(ROOT,read(args.plan),args.public_output,args.private_output,args.stage,
        execute=True,frozen_commit=args.frozen_commit)))


if __name__=='__main__':main()

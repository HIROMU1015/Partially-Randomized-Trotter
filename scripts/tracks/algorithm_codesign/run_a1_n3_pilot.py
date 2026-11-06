#!/usr/bin/env python3
"""Inspect preparation only; scientific entry remains disabled in this SOURCE."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import a1_n3_pilot_campaign as pilot

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--show-plan',action='store_true');group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.show_plan:print(json.dumps(pilot.preparation_plan(),ensure_ascii=False,indent=2))
    else:
        try:pilot.execution_blocked()
        except RuntimeError as exc:parser.exit(2,str(exc)+'\n')

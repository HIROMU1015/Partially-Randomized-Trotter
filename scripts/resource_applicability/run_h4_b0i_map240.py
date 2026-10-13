#!/usr/bin/env python3
"""Future closed 240-wrapper map; drafts refuse before scientific actions."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path);parser.add_argument('--authorization',type=Path)
    parser.add_argument('--review',type=Path);parser.add_argument('--owned-worker',type=int)
    args=parser.parse_args()
    from trottertracks.resource_applicability.h4_b0i_map.runtime import execute,worker_entry
    if args.owned_worker is not None:
        worker_entry(args.owned_worker);return
    if not all((args.plan,args.authorization,args.review)):parser.error('new plan, authorization and review required')
    result=execute(ROOT,args.plan,args.authorization,args.review)
    print(json.dumps(result));raise SystemExit(0 if result['status']=='MAP240_COMPLETE_STOP' else 1)


if __name__=='__main__':main()

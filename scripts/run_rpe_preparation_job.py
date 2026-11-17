#!/usr/bin/env python3
"""Dry-check/execute/collect a single source-bound preparation task."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import rpe_preparation_worker as P


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--task-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['dry-check','execute','collect'],required=True)
    args=parser.parse_args();assignment=P.read(args.assignment);profile=P.read(args.profile)
    if args.mode=='dry-check':
        result=P.dry_check(assignment['request'],profile,assignment['source_manifest']);P.put(args.output/'DRY_CHECK.json',result)
    elif args.mode=='execute':result=P.execute(assignment,profile,args.task_root)
    else:result=P.collect(assignment,profile,args.task_root)
    print(json.dumps({'status':result['status'],'request_id':assignment['job_id'],'mode':args.mode}))
    return 0 if result['status']in ('DRY_CHECK_PASS','PREPARATION_COMPLETE')else 2


if __name__=='__main__':raise SystemExit(main())

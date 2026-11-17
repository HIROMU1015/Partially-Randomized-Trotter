#!/usr/bin/env python3
"""Inspect/install an already selected release; never start scientific work."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.rpe_release_deployment import ensure_components,read,validate_assignment_release


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--layout',type=Path,required=True)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--components',nargs='+')
    parser.add_argument('--assignment',type=Path)
    parser.add_argument('--assignment-stdin',action='store_true')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest,layout=read(args.manifest),read(args.layout)
    result=ensure_components(manifest,layout,install=args.install,components=args.components)
    if args.assignment or args.assignment_stdin:
        if not args.profile:parser.error('--assignment requires --profile')
        try:
            assignment=json.load(sys.stdin)if args.assignment_stdin else read(args.assignment)
            result['assignment']=validate_assignment_release(manifest,layout,assignment,read(args.profile))
        except Exception as error:
            result['status']='HOLD';result['holds'].append(dict(assignment=str(error),error_type=type(error).__name__))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['status']=='PASS'else 2


if __name__=='__main__':raise SystemExit(main())

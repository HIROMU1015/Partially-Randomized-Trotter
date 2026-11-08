#!/usr/bin/env python3
"""DEFERRED real-H4 resource transaction; separate science instruction required."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-source-commit',required=True)
    args=p.parse_args()
    from trottertracks.algorithm_codesign.vacuum_h4_execution import run
    result=run(ROOT,args.output,expected_source_commit=args.expected_source_commit)
    print(result['status'])
    return 0 if result['status']=='RESOURCE_COMPARISON_AND_UNIFORM_BOUNDS_RECORDED' else 1


if __name__=='__main__':raise SystemExit(main())

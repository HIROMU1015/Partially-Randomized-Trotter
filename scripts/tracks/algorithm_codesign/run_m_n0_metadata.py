#!/usr/bin/env python3
"""Future separately authorized N1 metadata pass. No default/source auto-run."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from trottertracks.algorithm_codesign.m_n0_metadata_execution import run_metadata_pass
    result=run_metadata_pass(ROOT,args.output,expected_source_commit=args.source_commit)
    print(result['status'])
    return 0 if result['status'] in ('NONTRIVIAL_GROUP_FOUND','NO_NONTRIVIAL_GROUP_IN_SELECTED_SLOT') else 1


if __name__=='__main__':raise SystemExit(main())

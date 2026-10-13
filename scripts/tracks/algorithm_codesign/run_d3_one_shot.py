#!/usr/bin/env python3
"""D3 one-shot owner/observer. Requires separate source-bound authorization."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_robustness.runner import execute

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    try:sys.exit(execute(ROOT,a.source))
    except Exception as exc:
        print('D3_LAUNCH_OR_RECORD_FAILURE '+type(exc).__name__+': '+str(exc)[:1500],file=sys.stderr)
        sys.exit(2)

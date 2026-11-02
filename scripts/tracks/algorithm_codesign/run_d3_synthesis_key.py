#!/usr/bin/env python3
"""Internal one-key worker. Cannot be used without owned D3 execution job."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_robustness.worker import execute

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    try:sys.exit(execute(ROOT,a.job))
    except Exception as exc:
        print('D3_KEY_LAUNCH_OR_RECORD_FAILURE '+type(exc).__name__+': '+str(exc)[:1500],file=sys.stderr)
        sys.exit(2)

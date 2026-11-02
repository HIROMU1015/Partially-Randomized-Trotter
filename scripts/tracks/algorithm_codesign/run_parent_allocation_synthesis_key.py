#!/usr/bin/env python3
"""Private source-bound one-key worker; no direct registered fixture entry."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    from trottertracks.algorithm_codesign.parent_experiment.worker import execute
    sys.exit(execute(ROOT,a.job))

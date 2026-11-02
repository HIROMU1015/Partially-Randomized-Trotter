#!/usr/bin/env python3
"""Internal ledger child, owned by the exclusive D3 observer."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_robustness.runner import ledger_job

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    try:sys.exit(ledger_job(ROOT,a.job))
    except Exception as exc:
        print('D3_LEDGER_LAUNCH_OR_RECORD_FAILURE '+type(exc).__name__+': '+str(exc)[:1500],file=sys.stderr)
        sys.exit(2)

#!/usr/bin/env python3
"""Source-bound one-shot. Pending authority fails before registered access."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--source');g.add_argument('--ledger-job');a=p.parse_args()
    from trottertracks.algorithm_codesign.parent_experiment.runner import execute,ledger_job
    sys.exit(execute(ROOT,a.source) if a.source else ledger_job(ROOT,a.ledger_job))

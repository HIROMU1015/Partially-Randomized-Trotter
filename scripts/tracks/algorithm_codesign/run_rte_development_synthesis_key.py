"""Internal one-key worker; no execution without new D1 authority and marker."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--job',required=True)
    args=parser.parse_args()
    from trottertracks.algorithm_codesign.rte_development.worker import run_worker
    raise SystemExit(run_worker(ROOT,args.job))

#!/usr/bin/env python3
"""Default passive preflight; real LP execution needs separate chat permission."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    from trottertracks import c_fixed_u_tail_lp_runner as m
    if not a.execute:
        m.preflight(ROOT,a.contract);print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',science_calls=0,proposed_maximum=m.MAX_CALLS)));return
    if a.output is None:p.error('fresh output required')
    m.run(ROOT,a.contract,a.output)


if __name__=='__main__':main()

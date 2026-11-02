#!/usr/bin/env python3
"""Separately unapproved native128 path. Default is passive byte preflight."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',required=True,type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    from trottertracks import c_pr_aware_df_m4_native as n
    if a.execute:n.run(ROOT,a.contract)
    else:
        n.preflight(ROOT,a.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=n.MAX_CALLS,science_permission='SEPARATE_USER_PERMISSION_REQUIRED')))

if __name__=='__main__':main()

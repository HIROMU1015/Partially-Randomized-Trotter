#!/usr/bin/env python3
"""Passive by default; --execute needs separate user scientific permission."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_basis_third as study


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,default=ROOT/'docs/research/research_c_df_basis_third_contract_unapproved_20261029.json')
    p.add_argument('--execute',action='store_true',help='Only after separate explicit scientific permission')
    a=p.parse_args()
    if a.execute:
        result=study.run(ROOT,a.contract)
        print(json.dumps(dict(status=result['status'],mandatory_stop=True)))
    else:
        study.preflight(ROOT,a.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',maximum_units=study.MAX_CALLS,
                              scientific_calls=0,executed=False)))


if __name__=='__main__':main()

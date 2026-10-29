#!/usr/bin/env python3
"""Metadata repair; --execute only within the user-permitted fresh run02."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_basis_third_v2 as study


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,default=ROOT/'docs/research/research_c_df_basis_third_contract_repair_v2_20261029.json')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.execute:
        result=study.run(ROOT,a.contract)
        print(json.dumps(dict(status=result['status'],mandatory_stop=True)))
    else:
        study.preflight(ROOT,a.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',maximum_units=study.MAX_CALLS,scientific_calls=0,executed=False)))


if __name__=='__main__':main()

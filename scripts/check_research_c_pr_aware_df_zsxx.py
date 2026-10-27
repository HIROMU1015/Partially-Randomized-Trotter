#!/usr/bin/env python3
"""Passive identities or fixed artificial2x2 cases;no molecular execution."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path)
    p.add_argument('--artificial-output',type=Path)
    a=p.parse_args()
    if bool(a.contract)==bool(a.artificial_output):p.error('Choose passive contract or fixed artificial output')
    if a.artificial_output:
        from trottertracks.c_pr_aware_df_zsxx_artificial import evidence
        result=evidence()
        with a.artificial_output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(dict(status=result['status'],cases=result['validated_cases'],new_H4_scientific_calls=0)))
    else:
        from trottertracks.c_pr_aware_df_zsxx_contract import preflight
        print(json.dumps(preflight(ROOT,a.contract)))

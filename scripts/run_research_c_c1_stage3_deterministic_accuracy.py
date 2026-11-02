#!/usr/bin/env python3
"""Passive byte preflight; deterministic saved-value pilot needs permission."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.execute != bool(a.output): p.error('--execute requires fresh --output')
    from trottertracks.c1_stage3_deterministic_accuracy import preflight,execute
    if not a.execute:
        preflight(ROOT,a.contract,a.input)
        print(json.dumps(dict(status='DETERMINISTIC_ACCURACY_BYTE_PREFLIGHT_PASS',scientific_calls=0)))
        return
    r=execute(ROOT,a.contract,a.input,a.output)
    print(json.dumps(dict(status=r['status'],scientific_units=r['scientific_units'],mandatory_stop=True)))


if __name__=='__main__':main()

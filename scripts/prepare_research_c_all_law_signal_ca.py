#!/usr/bin/env python3
"""Passive design preflight; current all-law scientific design is not executable."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from trottertracks import c_all_law_signal_ca as m
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,default=ROOT/m.CONTRACT)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.execute:m.run(ROOT,a.contract,execute_science=True)
    else:
        _,status=m.preflight(ROOT,a.contract)
        print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__':main()

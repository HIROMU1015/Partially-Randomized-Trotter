#!/usr/bin/env python3
"""Passive saved-native audit by default; --execute requires separate chat permission."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from trottertracks import c_pr_xdf_statistics as m
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,default=ROOT/m.CONTRACT)
    p.add_argument('--execute',action='store_true')
    args = p.parse_args()
    if args.execute:
        m.run(ROOT,args.contract,execute_science=True)
    else:
        _, audit = m.preflight(ROOT,args.contract)
        print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__': main()

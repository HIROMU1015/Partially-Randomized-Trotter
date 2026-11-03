#!/usr/bin/env python3
"""Passive preflight; --execute is reserved for separately permitted B2 science."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

if __name__ == '__main__':
    from trottertracks import c_reference_bound_contract as m
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    args = p.parse_args()
    result = m.run(ROOT,ROOT/m.CONTRACT,execute_science=True) if args.execute else m.preflight(ROOT,ROOT/m.CONTRACT)[1]
    print(json.dumps(result,ensure_ascii=False))

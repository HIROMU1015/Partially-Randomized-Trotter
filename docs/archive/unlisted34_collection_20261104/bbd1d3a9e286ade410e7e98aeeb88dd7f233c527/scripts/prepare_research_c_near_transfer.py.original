#!/usr/bin/env python3
"""Passive readiness only; this design has no molecular science entry."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    from trottertracks import c_near_transfer as m
    if a.execute:
        m.execute()
    print(json.dumps(m.preflight(ROOT, a.contract)))


if __name__ == '__main__':
    main()

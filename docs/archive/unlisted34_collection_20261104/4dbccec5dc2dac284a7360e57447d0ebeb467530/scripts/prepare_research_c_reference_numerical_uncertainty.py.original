#!/usr/bin/env python3
"""Passive B2 preflight only; no scientific execution option."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    from trottertracks import c_reference_numerical_uncertainty as b2
    print(json.dumps(b2.preflight(ROOT, ROOT/b2.CONTRACT), ensure_ascii=False))

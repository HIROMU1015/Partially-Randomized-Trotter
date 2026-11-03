#!/usr/bin/env python3
"""Metadata/byte/schema preflight only; no scientific execution flag."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from trottertracks import c_b1_b2_reference_bound_integration as m
    r=m.preflight(ROOT,ROOT/m.CONTRACT)
    print(json.dumps({k:v for k,v in r.items() if k!='records'},ensure_ascii=False))

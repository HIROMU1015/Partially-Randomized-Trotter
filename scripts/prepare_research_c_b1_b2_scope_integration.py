#!/usr/bin/env python3
"""Passive by default and in entirety; no scientific execution option."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from trottertracks import c_b1_b2_scope_integration as m
    print(json.dumps(m.preflight(ROOT,ROOT/m.CONTRACT),ensure_ascii=False))

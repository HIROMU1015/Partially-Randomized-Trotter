#!/usr/bin/env python3
"""Passive preparation only; no scientific execution option."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
if __name__ == '__main__':
    from trottertracks import c_iid_grid_expected_bias as m
    print(json.dumps(m.preflight(ROOT, ROOT/m.CONTRACT), ensure_ascii=False))

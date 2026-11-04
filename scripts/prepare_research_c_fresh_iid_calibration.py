#!/usr/bin/env python3
"""Passive preparation only; deliberately no scientific execution CLI."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
if __name__ == '__main__':
    from trottertracks import c_fresh_iid_calibration as adapter
    print(json.dumps(adapter.preflight(ROOT, ROOT/adapter.CONTRACT), ensure_ascii=False))

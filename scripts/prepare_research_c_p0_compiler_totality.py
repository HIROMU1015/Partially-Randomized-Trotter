#!/usr/bin/env python3
"""Passive technical contract check; no scientific execution option."""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise SystemExit('Passive only: no execution or sampling arguments accepted')
    from trottertracks import c_p0_compiler_totality as m
    print(json.dumps(m.preflight(ROOT, ROOT/m.CONTRACT)))

#!/usr/bin/env python3
"""Passive schema/SOURCE preflight by default; science needs separate permission."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    from trottertracks import c_per_word_range_schema as m
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-stage-R', action='store_true')
    args = parser.parse_args()
    if args.execute_stage_R:
        result = m.run(ROOT, ROOT / m.CONTRACT, execute_science=True)
    else:
        result = m.preflight(ROOT, ROOT / m.CONTRACT)
    print(json.dumps(result, ensure_ascii=False))

#!/usr/bin/env python3
"""Passive E2/r16 all-law metadata check; no scientific execution option."""
import argparse
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
from trottertracks.c_entry_e2_all_law_readiness import preflight

if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(preflight(root), ensure_ascii=False, allow_nan=False, indent=2))

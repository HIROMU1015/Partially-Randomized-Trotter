#!/usr/bin/env python3
"""Passive default. Separate permission is required for the dormant local pilot."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from trottertracks import c_compiled_raw_readiness as m
    p=argparse.ArgumentParser();p.add_argument('--execute-local-pilot',action='store_true');a=p.parse_args()
    result=m.run_pilot(ROOT,execute_science=True) if a.execute_local_pilot else m.preflight(ROOT)
    print(json.dumps(result,ensure_ascii=False))

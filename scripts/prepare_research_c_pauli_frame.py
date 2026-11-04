#!/usr/bin/env python3
"""Passive default;scientific switch is only for separately authorized execution."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from trottertracks import c_pauli_frame_readiness as m
    p=argparse.ArgumentParser();p.add_argument('--execute-frame-pilot',action='store_true');args=p.parse_args()
    print(json.dumps(m.run_pilot(ROOT,execute_science=True) if args.execute_frame_pilot else m.preflight(ROOT),ensure_ascii=False))

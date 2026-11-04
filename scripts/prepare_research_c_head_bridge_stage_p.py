#!/usr/bin/env python3
"""Passive default; H and P require their own separately permitted contracts."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
 from trottertracks import c_head_stage_p_execution as m
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('H','P'),required=True)
 p.add_argument('--execute-science',action='store_true');a=p.parse_args()
 f=ROOT/m.CONTRACTS[a.stage]
 r=m.run(ROOT,f,execute_science=True) if a.execute_science else m.preflight(ROOT,f)
 print(json.dumps(r,ensure_ascii=False))

#!/usr/bin/env python3
"""Default passive byte/metadata preflight; Stage R needs separate permission."""
from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from trottertracks import c_per_word_range_execution as m
    p=argparse.ArgumentParser();p.add_argument('--execute-stage-R',action='store_true')
    a=p.parse_args()
    if a.execute_stage_R:result=m.run(ROOT,ROOT/m.method.CONTRACT,execute_science=True)
    else:result=m.preflight(ROOT,ROOT/m.method.CONTRACT)
    print(json.dumps(result,ensure_ascii=False))

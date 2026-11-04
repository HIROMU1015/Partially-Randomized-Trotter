#!/usr/bin/env python3
"""Passive technical entry; scientific fixed-pair pilot needs separate permission."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from trottertracks import c_totality_calibration as m
    from trottertracks import c_batch_pair_pilot as pilot
    p=argparse.ArgumentParser();p.add_argument('--execute-fixed-pair-pilot',action='store_true');a=p.parse_args()
    result=(pilot.run(ROOT,ROOT/pilot.CONTRACT,execute_science=True) if a.execute_fixed_pair_pilot else
            dict(integration=m.preflight(ROOT,ROOT/m.CONTRACT),fixed_pair_pilot=pilot.preflight(ROOT,ROOT/pilot.CONTRACT)))
    print(json.dumps(result,ensure_ascii=False))

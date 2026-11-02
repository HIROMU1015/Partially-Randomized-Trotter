#!/usr/bin/env python3
"""Recover completed paired-fit JSON only; no fit or numerical kernel entry."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--recover-saved",action="store_true")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    from trottertracks import c_common_convergence_export_repair as m
    m.preflight(ROOT,a.plan)
    if not a.recover_saved:
        print(json.dumps(dict(status="REPAIR_PREFLIGHT_PASS",saved_scalar_units=0,new_fits=0)))
        return
    if a.output is None:p.error("Fresh repair output required")
    m.recover(ROOT,a.plan,a.output)


if __name__=="__main__":main()

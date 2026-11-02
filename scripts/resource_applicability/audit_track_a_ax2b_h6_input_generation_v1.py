#!/usr/bin/env python3
"""Audit existing H6 input-run bytes only; never generate molecules or states."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.ax2b_h6_input_generation_audit_v1 import audit_saved
from trottertracks.resource_applicability.ax2b_limits import exclusive_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit_saved(args.saved_run)
    exclusive_json(args.output,report)
    print(report['status'])
    return 0

if __name__=='__main__':
    raise SystemExit(main())

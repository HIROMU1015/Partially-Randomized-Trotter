#!/usr/bin/env python3
"""Audit future saved diagnostic outputs only; never call the diagnostic runner."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.ax2b_h6_df_diagnostic_audit_v1 import audit_saved
from trottertracks.resource_applicability.ax2b_limits import exclusive_json
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--saved-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=audit_saved(a.saved_run);exclusive_json(a.output,r);print(r['status']);return 0
if __name__=='__main__':raise SystemExit(main())

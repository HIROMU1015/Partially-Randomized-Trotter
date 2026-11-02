#!/usr/bin/env python3
"""Saved byte/schema audit only; no scientific libraries imported."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.ax2b_h6_saved_completion_audit_v2 import audit_saved
from trottertracks.resource_applicability.ax2b_limits import exclusive_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saved-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit_saved(a.saved_run);exclusive_json(a.output,result);print(result['status'])

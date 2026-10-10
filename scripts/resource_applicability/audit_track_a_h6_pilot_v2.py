#!/usr/bin/env python3
"""Read saved pilot bytes only; no scientific imports."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.ax2b_h6_pilot_audit_v2 import audit_saved
from trottertracks.resource_applicability.ax2b_limits import exclusive_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();r=audit_saved(a.input);exclusive_json(a.output,r);print(r['status'])


if __name__=='__main__':main()

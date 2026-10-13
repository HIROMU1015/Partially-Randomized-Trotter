#!/usr/bin/env python3
"""Audit saved sealed results. Does not launch science or regenerate inputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    from trottertracks.representation_exploration.qc_mixed_access_checker_v1 import check_raw
    contract=json.loads((ROOT/'artifacts/qc_mixed_access_feasibility/2026-10-13/feasibility_contract.json').read_text())
    print(json.dumps(check_raw(args.directory,contract),indent=2))


if __name__=='__main__': main()

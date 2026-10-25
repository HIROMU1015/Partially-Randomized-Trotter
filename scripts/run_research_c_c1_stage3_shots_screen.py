#!/usr/bin/env python3
"""Byte preflight by default; saved-value shots science needs chat permission."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.execute != bool(args.output): parser.error('--execute requires fresh --output')
    from trottertracks.c1_stage3_shots_screen import preflight,execute
    if not args.execute:
        preflight(ROOT,args.contract)
        print(json.dumps(dict(status='SHOTS_BYTE_PREFLIGHT_PASS',scientific_calls=0)))
        return
    result = execute(ROOT,args.contract,args.output)
    print(json.dumps(dict(status=result['status'],scientific_units=result['scientific_units'],mandatory_stop=True)))


if __name__ == '__main__': main()

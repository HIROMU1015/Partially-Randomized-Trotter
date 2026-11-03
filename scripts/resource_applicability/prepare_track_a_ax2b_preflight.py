#!/usr/bin/env python3
"""Metadata-only AX-2B proposal/preflight writer; no launch option."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.ax2b_preflight import preparation_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    payload=preparation_bundle(ROOT)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(payload['status'])


if __name__=='__main__':main()

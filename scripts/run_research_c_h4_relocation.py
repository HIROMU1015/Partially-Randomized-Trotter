#!/usr/bin/env python3
"""Default is byte-only preflight. --execute needs separate user permission."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_h4_z_relocation as m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', default='docs/research/research_c_h4_relocation_contract_unapproved_20261106.json')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--preflight-output', type=Path)
    args = p.parse_args()
    if args.execute:
        print(m.execute(ROOT, args.contract))
    else:
        result = m.preflight(ROOT, args.contract)
        small = {k:v for k,v in result.items() if k not in ('contract','inventory')}
        if args.preflight_output:
            m.io.save(args.preflight_output, small)
        print(json.dumps(small, indent=2))


if __name__ == '__main__': main()

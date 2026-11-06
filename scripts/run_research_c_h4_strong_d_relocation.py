#!/usr/bin/env python3
"""Default: passive preflight. --execute requires separate scientific permission."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_h4_strong_d_relocation_execution as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', default='docs/research/research_c_h4_strong_d_relocation_contract_unapproved_20261107.json')
    parser.add_argument('--execute', action='store_true', help='only after separate explicit scientific execution permission')
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(runner.execute(ROOT, args.contract)))
    else:
        c = runner.preflight(ROOT, args.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS', scope=c['scope'], arrays_decoded=0,
            scientific_runs=0, execution_permission=False, fresh_output=c['fresh_output'])))


if __name__ == '__main__': main()

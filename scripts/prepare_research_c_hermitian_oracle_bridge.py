#!/usr/bin/env python3
"""Read the NOT_READY proposal only. No array, science or resume entry point."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root/'docs/research/research_c_hermitian_oracle_bridge_contract_unapproved_20261105.json'
    contract = json.loads(path.read_text())
    if contract['status'] != 'NOT_READY' or contract['executable']:
        raise ValueError('PASSIVE_PREPARATION_ONLY')
    print(json.dumps(dict(status='NOT_READY', scientific_calls=0,
        H_target=contract['H_target'], logical_calls=contract['logical_calls'],
        retry=0, resume=0, mandatory_STOP=True), sort_keys=True))


if __name__ == '__main__':
    main()

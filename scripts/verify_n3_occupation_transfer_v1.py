#!/usr/bin/env python3
"""Recheck saved OCC records without generator, numerical solver or new science."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration.n3_occupation_transfer_check_v1 import check_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    args = parser.parse_args()
    out = args.run_directory
    receipt = json.loads((out / 'launch_receipt.json').read_bytes())
    plan = json.loads((ROOT / 'docs/research/n3_occupation_transfer_v1_execution_plan_20261012.json').read_bytes())
    if receipt['source_fingerprint'] != plan['source_fingerprint'] or receipt['source_sha256'] != plan['source_sha256']:
        raise ValueError('source binding')
    for path, digest in plan['source_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest: raise ValueError('frozen source identity')
    data = (ROOT / plan['input_path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != plan['input_sha256']: raise ValueError('input binding')
    pack = json.loads(data)
    raw = (out / 'primary_ledger.json').read_bytes()
    seal = hashlib.sha256(raw).hexdigest()
    if json.loads((out / 'primary_seal.json').read_bytes())['sha256'] != seal: raise ValueError('primary seal')
    result = check_saved(pack, json.loads(raw), json.loads((out / 'secondary_ledger.json').read_bytes()), plan['source_fingerprint'], seal)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()

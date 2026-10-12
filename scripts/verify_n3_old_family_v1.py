#!/usr/bin/env python3
"""Independent exact saved diagnostic audit; no producer or numerical solver."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration.n3_old_family_check_v1 import verify
from trottertracks.representation_exploration.n3_old_family_inputs_v1 import (
    IDS, PREFIX, audit_saved_winner, digest, verify_inputs,
)


def verify_directory(directory):
    plan = json.loads((ROOT / 'docs/research/n3_old_family_v1_execution_plan_20261012.json').read_bytes())
    for path, sha in dict(plan['source_sha256'], **plan['saved_input_sha256']).items():
        if digest(ROOT / path) != sha: raise ValueError('FROZEN_BYTES')
    verify_inputs(ROOT); rows = []
    for name in IDS:
        row = json.loads((ROOT / PREFIX / 'inputs' / (name + '.json')).read_bytes())
        record = json.loads((directory / (name + '.json')).read_bytes())
        audit = verify(row['instance'], record)
        rows.append({'input_id': name, 'diagnostic': audit, 'saved_OCC': audit_saved_winner(ROOT, row)})
    return {'status': 'INDEPENDENT_SAVED_OLD_FAMILY_DIAGNOSTIC_PASS', 'cases': rows,
            'new_eigensolver_calls': 0, 'new_certificate_proposals': 0,
            'mandatory_stop': True, 'next_stage_authorized': False, 'central_theme': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_directory(args.directory), indent=2))


if __name__ == '__main__': main()

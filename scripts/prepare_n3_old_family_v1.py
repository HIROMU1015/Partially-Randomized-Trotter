#!/usr/bin/env python3
"""Prepare or verify lossless fixed S1 inputs; never proposes certificates."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration.n3_old_family_inputs_v1 import PREFIX, digest, select, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        out = ROOT / PREFIX / 'inputs'
        if out.exists(): raise ValueError('NO_INPUT_OVERWRITE')
        rows = select(ROOT); out.mkdir(parents=True)
        inventory = []
        for row in rows:
            name = row['instance']['id']; path = out / (name + '.json')
            path.write_text(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
            if path.stat().st_size > 8192: raise ValueError('INPUT_SHARD_CAP')
            inventory.append({'input_id': name, 'path': path.relative_to(ROOT).as_posix(),
                              'sha256': digest(path), 'bytes': path.stat().st_size,
                              'instance_sha256': row['instance_sha256']})
        (out / 'index.json').write_text(json.dumps({'cases': inventory, 'diagnostic_authorized': False}, indent=2) + '\n')
    print(json.dumps(verify_inputs(ROOT), indent=2))


if __name__ == '__main__': main()

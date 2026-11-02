#!/usr/bin/env python3
"""Fixed update064 downstream proposal; passive unless separately permitted --execute."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True, type=Path)
    p.add_argument('--candidate-root', required=True, type=Path,
        help='read-only worktree fixed at ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6')
    p.add_argument('--execute', action='store_true', help='requires separate user scientific permission')
    args = p.parse_args()
    from trottertracks import c_update064_r32_downstream as m
    if not args.execute:
        m.preflight(ROOT, args.contract, args.candidate_root)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS', scientific_calls=0,
            candidate=m.CANDIDATE, maximum_logical_units=m.MAX_CALLS)))
        return
    for key, leaf in [('XDG_CACHE_HOME', 'cache'), ('TMPDIR', 'tmp')]:
        directory = ROOT/'.cache/update064-r32-downstream'/leaf
        directory.mkdir(parents=True, exist_ok=True); os.environ[key] = str(directory)
    m.run(ROOT, args.contract, args.candidate_root)


if __name__ == '__main__': main()

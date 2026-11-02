#!/usr/bin/env python3
"""Prospective r32 price calibration proposal; passive unless --execute."""
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
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    from trottertracks import c_r32_price_statistics as m
    if not args.execute:
        m.preflight(ROOT, args.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS', scientific_calls=0,
            samples=m.N, maximum_logical_units=m.MAX_CALLS,
            policy='proposal requires user/research C adoption and execution permission')))
        return
    for key, leaf in [('XDG_CACHE_HOME', 'cache'), ('TMPDIR', 'tmp')]:
        directory = ROOT/'.cache/r32-price-statistics'/leaf
        directory.mkdir(parents=True, exist_ok=True); os.environ[key] = str(directory)
    m.run(ROOT, args.contract)


if __name__ == '__main__':
    main()

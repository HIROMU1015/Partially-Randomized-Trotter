#!/usr/bin/env python3
"""Selected fixed-J/U r retuning; passive preflight unless --execute."""
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
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    from trottertracks import c_rte_r_retuning as m
    if not args.execute:
        m.preflight(ROOT, args.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS', scientific_calls=0,
            r_values=m.R_VALUES, maximum_logical_units=m.MAX_CALLS)))
        return
    for key, leaf in [('XDG_CACHE_HOME', 'cache'), ('TMPDIR', 'tmp')]:
        directory = ROOT/'.cache/rte-r-retuning'/leaf
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    m.run(ROOT, args.contract)


if __name__ == '__main__':
    main()

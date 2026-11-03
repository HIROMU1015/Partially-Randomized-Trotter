#!/usr/bin/env python3
"""Byte-only preflight by default. Scientific execution needs separate permission."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    from trottertracks import c_xdf_cdf_common_ideal_lane as m
    if not a.execute:
        m.preflight(ROOT, a.contract)
        print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS', scientific_calls=0,
                              maximum_logical_units=m.MAX_CALLS)))
        return
    for key, leaf in [('XDG_CACHE_HOME', ''), ('MPLCONFIGDIR', 'matplotlib'), ('TMPDIR', 'tmp')]:
        directory = ROOT / '.cache/xdf-cdf-common-ideal' / leaf
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    m.run(ROOT, a.contract, execute_science=True)


if __name__ == '__main__':
    main()

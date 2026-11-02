#!/usr/bin/env python3
"""H4 saved-evidence CA/RPE/serial-resource pilot; passive by default."""
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
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    from trottertracks import c1_h4_ca_resources as m
    m.preflight(ROOT, a.contract)
    if not a.execute:
        print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS', science_calls=0, maximum_calls=m.MAX_CALLS)))
        return
    if not a.output:
        p.error('Fresh output required')
    m.fresh_output(ROOT, a.output)
    for key, leaf in [('XDG_CACHE_HOME', ''), ('MPLCONFIGDIR', 'matplotlib'), ('TMPDIR', 'tmp')]:
        directory = ROOT / '.cache/h4-ca-resources' / leaf
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    m.run(ROOT, a.contract, a.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""B3 byte preflight only by default; science needs separate user permission."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    from trottertracks import c1_h4_b3 as m
    if not a.execute:
        m.preflight(ROOT, a.contract, a.input)
        print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS', scientific_calls=0, maximum_calls=m.MAX_CALLS)))
        return
    if a.output is None: p.error('Fresh B3 output required')
    m.run(ROOT, a.contract, a.input, a.output)


if __name__ == '__main__':
    main()

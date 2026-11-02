#!/usr/bin/env python3
"""Default passive preflight; new molecular LP needs separate chat permission."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from trottertracks import c_fixed_u_tail_lp_numerics as m
    if not args.execute:
        m.preflight(ROOT, args.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS', science_calls=0,
                              proposed_maximum=m.MAX_CALLS)))
        return
    if args.output is None:
        parser.error('fresh output required')
    m.run(ROOT, args.contract, args.output)


if __name__ == '__main__':
    main()

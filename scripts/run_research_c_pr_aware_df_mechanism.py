#!/usr/bin/env python3
"""M1 saved-scalar mechanism analysis. Default is byte-only preflight."""
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
    parser.add_argument('--execute', action='store_true', help='Only after explicit scientific permission')
    a = parser.parse_args()
    from trottertracks import c_pr_aware_df_mechanism as m
    if a.execute:
        m.run(ROOT, a.contract)
    else:
        m.preflight(ROOT, a.contract)
        print(json.dumps(dict(status='PASSIVE_SOURCE_INPUT_IDENTITY_PASS',
                              scientific_calls=0, maximum_calls=m.MAX_CALLS)))


if __name__ == '__main__':
    main()

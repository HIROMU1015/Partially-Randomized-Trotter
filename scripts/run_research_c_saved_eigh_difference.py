#!/usr/bin/env python3
"""Passive by default; --execute-science requires separate user permission."""
import argparse
import json
from pathlib import Path
from trottertracks import c_saved_eigh_difference as m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--execute-science', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = (m.run(root, args.contract, execute_science=True) if args.execute_science
              else m.preflight(root, args.contract))
    print(json.dumps(result, sort_keys=True))

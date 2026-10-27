#!/usr/bin/env python3
"""κ diagnostic: passive by default; --execute needs separate chat permission."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    from trottertracks.c_pr_aware_df_kappa_diagnostic import preflight, run
    print(json.dumps(run(ROOT, args.contract) if args.execute else preflight(ROOT, args.contract)))

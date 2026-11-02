#!/usr/bin/env python3
"""Saved RC2/PR Stage B. Default: passive preflight, no molecular decode."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Separate scientific chat permission required')
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    from trottertracks import c_rc2_pr_stage_b as m
    if not args.execute:
        m.preflight(ROOT,args.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=m.MAX_CALLS)))
        return
    if args.output is None: p.error('fresh output required')
    # Only this task's process/cache/temp; existing venv/shared settings untouched.
    for key,leaf in [('XDG_CACHE_HOME','cache'),('TMPDIR','tmp'),
            ('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba')]:
        d = ROOT/'.cache/rc2-pr-stage-b'/leaf
        d.mkdir(parents=True,exist_ok=True)
        os.environ[key] = str(d)
    m.run(ROOT,args.contract,args.output)


if __name__ == '__main__': main()

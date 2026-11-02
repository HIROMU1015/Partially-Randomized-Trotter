#!/usr/bin/env python3
"""New saved RC2/PR native-only calibration. Default passive, separately unapproved."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    from trottertracks import c_rc2_pr_native as m
    if not a.execute:
        m.preflight(ROOT,a.contract);print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=m.MAX_CALLS)));return
    if a.output is None:p.error('fresh output required')
    for key,leaf in [('XDG_CACHE_HOME','cache'),('TMPDIR','tmp'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba')]:
        d=ROOT/'.cache/rc2-pr-native'/leaf;d.mkdir(parents=True,exist_ok=True);os.environ[key]=str(d)
    m.run(ROOT,a.contract,a.output)


if __name__=='__main__':main()

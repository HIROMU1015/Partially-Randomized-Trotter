#!/usr/bin/env python3
"""Saved CDF7 reference/RPE/price, passive by default;science needs permission."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    from trottertracks import c_cdf7_pr as m
    if not a.execute:
        m.preflight(ROOT,a.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',science_calls=0,maximum_calls=m.MAX_CALLS)))
        return
    for name,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=ROOT/'.cache'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    m.run(ROOT,a.contract)


if __name__=='__main__':main()

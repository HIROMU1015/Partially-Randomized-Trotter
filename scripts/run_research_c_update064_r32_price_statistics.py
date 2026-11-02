#!/usr/bin/env python3
"""Update064 price proposal; passive by default, separate permission for --execute."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='requires separate user scientific permission;not proof of permission')
    a=p.parse_args()
    from trottertracks import c_update064_r32_price_statistics as m
    if not a.execute:
        m.preflight(ROOT,a.contract,a.candidate_root)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scientific_calls=0,samples=m.N,
            maximum_logical_units=m.MAX_CALLS,statistics_status='UNAPPROVED_PROPOSAL')))
        return
    for key,leaf in [('XDG_CACHE_HOME','cache'),('TMPDIR','tmp'),('MPLCONFIGDIR','mpl')]:
        d=ROOT/'.cache/update064-r32-price-statistics'/leaf;d.mkdir(parents=True,exist_ok=True);os.environ[key]=str(d)
    m.run(ROOT,a.contract,a.candidate_root,execution_opt_in=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Saved selected-J/U deterministic control; byte-only preflight by default."""
import argparse,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',required=True,type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    from trottertracks import c_same_ju_strong_d as m
    if not a.execute:
        m.preflight(ROOT,a.contract);print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scientific_calls=0,maximum_calls=m.MAX_CALLS)));return
    for key,leaf in [('XDG_CACHE_HOME','cache'),('TMPDIR','tmp')]:
        q=ROOT/'.cache/same-ju-strong-d'/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
    m.run(ROOT,a.contract)

if __name__=='__main__':main()

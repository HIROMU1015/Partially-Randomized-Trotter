#!/usr/bin/env python3
"""Default: passive independent-price preflight. Science needs separate permission."""
import argparse,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from trottertracks.c_cdf0_entry_independent_price import PROPOSAL,preflight,execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',default=PROPOSAL)
    p.add_argument('--execute',action='store_true',help='Only after separate user scientific execution permission')
    a=p.parse_args()
    if a.execute:execute(root,a.proposal,execute_science=True)
    else:print(json.dumps(preflight(root,a.proposal)[1],ensure_ascii=False,indent=2))

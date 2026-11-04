#!/usr/bin/env python3
"""Passive by default. --execute-science is for a separately permitted run."""
import argparse
import json
from pathlib import Path
from trottertracks import c_direct_m0_bound as m


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path)
    p.add_argument('--execute-science',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    result=m.run(root,args.contract,execute_science=True) if args.execute_science else m.preflight(root,args.contract)
    print(json.dumps(result,sort_keys=True))

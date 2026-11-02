#!/usr/bin/env python3
"""Fixed C1 pilot entry; separate approval required before numerical imports."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.representation_exploration.c1_macrocycle_execution_v1 import authorize,preflight


def main():
    p=argparse.ArgumentParser(); p.add_argument('--authorization'); p.add_argument('--preflight',action='store_true')
    args=p.parse_args(); root=Path(__file__).resolve().parents[1]
    if args.preflight: print(json.dumps(preflight(root),indent=2)); return
    root,out=authorize(root,args.authorization)
    try:
        from trottertracks.representation_exploration.c1_macrocycle_pilot_v1 import run
        print(json.dumps(run(root,out),indent=2))
    except Exception as error:
        (out/'failure.json').write_text(json.dumps({'error':type(error).__name__,
            'message':str(error),'retry':0,'mandatory_stop':True},indent=2)+'\n')
        raise


if __name__=='__main__': main()

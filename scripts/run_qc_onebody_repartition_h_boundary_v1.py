#!/usr/bin/env python3
"""h-boundary preflight or separately authorized one-shot; no default science."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.ground_pf_repartition.qc_onebody_repartition_h_boundary_execution_v1 import preflight,authorize
def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--preflight',action='store_true');m.add_argument('--execute',action='store_true');p.add_argument('--authorization',type=Path)
    a=p.parse_args()
    try:
        if a.preflight:result=preflight(ROOT)
        else:
            context=authorize(ROOT,a.authorization)
            from trottertracks.ground_pf_repartition.qc_onebody_repartition_h_boundary_pilot_v1 import run_registered
            result=run_registered(context)
        print(json.dumps(result,indent=2));return 0
    except Exception as e:
        print(json.dumps({'status':'MANDATORY_STOP','reason':str(e)[:4096],'retry':0,'mandatory_stop':True}),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Identity-only preflight or separately approved one-shot molecular pilot."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.ground_pf_repartition.qc_onebody_repartition_execution_v1 import preflight,authorize,LaunchError,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args()
    context=None
    try:
        if args.preflight:
            print(json.dumps(preflight(ROOT),indent=2));return 0
        context=authorize(ROOT,args.authorization)
        # Heavy/numerical imports are BELOW the explicit authorization gate.
        from trottertracks.ground_pf_repartition.qc_onebody_repartition_pipeline_v1 import run_registered
        print(json.dumps(run_registered(context),indent=2));return 0
    except Exception as error:
        failure={'status':'MANDATORY_STOP','reason':str(error)[:4096],'retry':0,'mandatory_stop':True}
        if context is not None and not (context.output/'stop.json').exists():
            write_json(context.output/'stop.json',failure)
        print(json.dumps(failure),file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())

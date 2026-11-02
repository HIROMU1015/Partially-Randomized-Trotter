#!/usr/bin/env python3
"""Read-only preflight or separately authorized one-shot science execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.representation_exploration.qc_mixed_access_execution_v1 import preflight,authorize,LaunchError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args()
    try:
        if args.preflight: print(json.dumps(preflight(ROOT),indent=2)); return 0
        context=authorize(ROOT,args.authorization)
        # Imports occur after explicit approval, source/environment checks, and
        # consumption of the single attempt. No implicit science default.
        from trottertracks.representation_exploration.qc_mixed_access_pipeline_v1 import run_registered
        print(json.dumps(run_registered(context),indent=2)); return 0
    except (LaunchError,ValueError,RuntimeError) as error:
        print(json.dumps({'status':'MANDATORY_STOP','reason':str(error)}),file=sys.stderr); return 2


if __name__=='__main__': raise SystemExit(main())

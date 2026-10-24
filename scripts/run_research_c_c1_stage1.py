#!/usr/bin/env python3
"""Read-only preflight by default; execution requires a new explicit approval."""
import argparse
import json
from pathlib import Path
import sys
import signal

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    def terminate(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fit-python',type=Path,default=ROOT/'.venv-fit/bin/python')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--worker-fit',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    from trottertracks import c1_stage1_execution as run
    # Refuse execution before reading even the physical packet bytes.
    if (args.execute or args.worker_fit) and (args.approval is None or args.output is None):
        parser.error('new explicit approval and fresh bound output are required')
    v=run.preflight(ROOT,args.contract,args.source,args.input)
    if args.worker_fit:
        run.verify_approval(v,args.approval,args.output,worker=True)
        run.worker_fit(v,args.output)
    elif args.execute:
        result=run.execute(v,args.approval,args.output,args.fit_python.resolve())
        print(json.dumps(dict(status=result['status'],mandatory_stop=True)))
    else:
        if args.approval or args.output:
            parser.error('preflight accepts no approval/output')
        print(json.dumps(dict(status='TECHNICAL_PREFLIGHT_PASS_NO_SCIENCE',
                              execution_authorized=False,grant_issued=False)))


if __name__=='__main__':
    main()

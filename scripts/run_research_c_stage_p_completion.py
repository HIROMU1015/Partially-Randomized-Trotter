#!/usr/bin/env python3
"""Passive default; foreground child, exit/signal capture, no timeout or restart."""
import argparse
import faulthandler
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def supervise(command, launch, *, poll_seconds=30):
    from trottertracks.c_stage_p_completion import save_atomic
    launch=Path(launch);launch.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic();handlers={};process=None
    with (launch/'worker.log').open('xb') as log:
        try:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            def forward(number,frame):
                save_atomic(launch/'forwarded_signal.json',dict(signal=number,elapsed_seconds=time.monotonic()-tick))
                if process.poll() is None:process.send_signal(number)
            for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):handlers[sig]=signal.signal(sig,forward)
            while True:
                try:
                    code=process.wait(timeout=poll_seconds);break
                except subprocess.TimeoutExpired:
                    save_atomic(launch/'heartbeat.json',dict(elapsed_seconds=time.monotonic()-tick,status='RUNNING'))
                    print('Stage P completion worker running;elapsed %.1fs'%(time.monotonic()-tick),flush=True)
            save_atomic(launch/'exit_status.json',dict(returncode=code,
                terminating_signal=-code if code<0 else None,wall_seconds=time.monotonic()-tick,
                retry=0,resume=0,mandatory_stop=True))
            return code
        finally:
            if process is not None and process.poll() is None:
                process.terminate();process.wait()
            for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    faulthandler.enable()
    from trottertracks import c_stage_p_completion as m
    p=argparse.ArgumentParser();p.add_argument('--execute-science',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    contract=ROOT/m.CONTRACT
    if not args.execute_science:
        if args.worker:raise ValueError('WORKER_REQUIRES_EXECUTION_SCOPE')
        print(json.dumps(m.preflight(ROOT,contract)));sys.exit(0)
    if args.worker:
        result=m.run(ROOT,contract,execute_science=True,launched=True)
        print(json.dumps(dict(status=result['status'],mandatory_stop=True)),flush=True)
    else:
        m.preflight(ROOT,contract)
        c=m.io.read(contract)
        code=supervise([sys.executable,'-u','-X','faulthandler',str(Path(__file__).resolve()),
                        '--execute-science','--worker'],ROOT/c['launch_output'])
        sys.exit(code if code>=0 else 128-code)

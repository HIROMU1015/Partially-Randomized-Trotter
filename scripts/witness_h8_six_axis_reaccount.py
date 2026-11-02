"""Save the launcher's native exit separately from its child's receipt."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_h8_six_axis_reaccount as runner
c = runner.c


def witness(config, grant):
    c.authorize(config, grant)
    c.verify(config)
    c.paths(config)
    path = c.home(config['_witness_receipt_path'])
    marker = c.home(config['_witness_lock'])
    c.require(not path.exists(), 'NATIVE_WITNESS_ALREADY_USED')
    c.write(marker, dict(SOURCE=config['SOURCE'], config_sha256=c.digest(config),
                         grant_sha256=c.digest(grant), dispatches_reserved=1))
    requested = [None]
    handlers = {}
    def latch(sig, frame):
        if requested[0] is None: requested[0]='OUTER_SIGNAL_'+str(sig)
    def cooperative_stop():
        # Let the launcher stop and reap its own arithmetic child before the
        # witness reaps the launcher. Do not race nested cleanup with a kill.
        request = c.home(config['_stop_request_path'])
        if requested[0] is not None and not request.exists():
            c.write(request, dict(config_sha256=c.digest(config), grant_sha256=c.digest(grant),
                                  first_stop=requested[0]))
    child = None
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig]=signal.signal(sig,latch)
        child=subprocess.Popen([c.PYTHON,'-I','-S','-P','-B',str(c.ROOT/'scripts/run_h8_six_axis_reaccount.py'),
            '--config',config['_config_path'],'--grant',config['_grant_path']], stdin=subprocess.DEVNULL,
            close_fds=True)
        receipt=runner.wait_owned(child,path,poll=cooperative_stop)
        return 0 if receipt['native_exit_code']==0 and receipt['reaped'] and not receipt['cleanup_errors'] else 1
    finally:
        # SIGINT/TERM are latched while cleanup runs, so an additional signal
        # cannot skip wait/reap or overwrite the first reason.
        if child is not None and child.returncode is None:
            def stop(): raise KeyboardInterrupt(requested[0] or 'OUTER_EXCEPTION')
            runner.wait_owned(child,path,interrupt=stop)
        for sig,handler in handlers.items(): signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--grant',required=True)
    args=parser.parse_args()
    return witness(c.read(args.config),c.read(args.grant))


if __name__=='__main__': sys.exit(main())

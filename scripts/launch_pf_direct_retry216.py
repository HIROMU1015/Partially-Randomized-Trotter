"""One-shot detached launch; source/grant gate precedes every new process."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import pf_direct_compile_contract as c
import pf_direct_retry_contract as r


def launch_detached(command,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    log=(directory/'observer_launcher.log').open('x')
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        cwd=r.ROOT,start_new_session=True,close_fds=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    log.close()
    return process.pid


def main(seal):
    spec=c.read_json(seal/(r.GROUPS[0]+'_spec.json'));grant=c.read_json(seal/(r.GROUPS[0]+'_grant.json'))
    r.authorize(spec,grant);gate=r.gate(spec,grant)
    c.require(not (r.PRIVATE/'run01').exists() and not (r.PRIVATE/'observer01').exists(),'NEW_ATTEMPT_ALREADY_USED')
    directory=r.PRIVATE/'observer01'
    command=[spec['observer_python'],'-B',str(r.ROOT/'scripts/observe_pf_direct_retry216.py'),
        '--sealed-directory',str(seal),'--directory',str(directory)]
    pid=launch_detached(command,directory)
    r.save(directory/'launch_receipt.json',dict(observer_pid=pid,SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),
        launcher_pid=os.getpid(),launcher_session=os.getsid(0),new_session_requested=True,
        command=command,prelaunch_gate=gate,new_attempt_axes=216,automatic_retry=False))
    print(__import__('json').dumps(dict(status='DETACHED_OBSERVER_LAUNCHED',pid=pid,SOURCE=spec['SOURCE'],axes=216)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sealed-directory',type=Path,required=True)
    main(p.parse_args().sealed_directory)

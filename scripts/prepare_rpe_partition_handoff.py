#!/usr/bin/env python3
"""Explicitly prepare or register one fixed independent PBS handoff.

No scientific job is started. This is never called by periodic maintenance.
The existing manager lock prevents racing a running central controller.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.rpe_pbs import Gateway,PbsManager,read

def operation(run,unit,action,*,gateway_factory=Gateway):
    run=Path(run).resolve()
    plan=read(run/'PBS_PARTITION_PLAN.json')
    directory=run/'distributed'/'pbs'
    if not directory.is_dir():raise ValueError('Missing fixed PBS manager directory')
    with (directory/'MANAGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manager=SimpleNamespace(offline_plan=plan,directory=directory)
        if action=='register':
            return PbsManager.register_local_partition_completion(manager,unit)
        if action!='prepare':raise ValueError('Unknown handoff operation')
        settings=dict(read(directory/'SETTINGS.json'))
        distributed=read(run/'ROUND.json')['executor']['distributed']
        dispatch=distributed.get('pbs_dispatch_gateway')
        if dispatch:settings.update(gateway_script=dispatch['path'],gateway_sha256=dispatch['sha256'])
        manager.gateway=gateway_factory(settings)
        try:
            return PbsManager.prepare_unstarted_partition_handoff(manager,unit)
        finally:
            manager.gateway.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--unit',required=True)
    ap.add_argument('--action',choices=('prepare','register'),required=True)
    args=ap.parse_args()
    print(json.dumps(operation(args.run_dir,args.unit,args.action),indent=2))

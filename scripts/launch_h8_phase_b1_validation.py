"""New-grant B1 launcher; no default execution and no automatic B2/C/retry."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import h8_phase_b1_validation as gate
import h8_b0_anchor_runtime as cpu_io
import launch_h8_b0_anchors as strict_probe
import launch_h8_strengthened_phase_a as observer


def core_gate(cpus):
    identities=[];siblings=set()
    for cpu in cpus:
        base=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
        identities.append(((base/'physical_package_id').read_text(),(base/'core_id').read_text()))
        for part in (base/'thread_siblings_list').read_text().strip().split(','):
            ends=list(map(int,part.split('-')));siblings.update(range(ends[0],ends[-1]+1))
    gate.require(len(set(identities))==2,'TWO_DISTINCT_PHYSICAL_CORES_REQUIRED')
    root=Path('/sys/fs/cgroup')
    group=root/next(x.split(':',2)[2].lstrip('/') for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    while group.is_relative_to(root):
        if (group/'cpu.max').exists():
            quota,period=(group/'cpu.max').read_text().split()
            gate.require(quota=='max' or int(quota)>=2*int(period),'CPU_QUOTA_BELOW_TWO')
        if group==root:break
        group=group.parent
    first=cpu_io.cpu_counters();time.sleep(1)
    cpu_io.check_idle(first,cpu_io.cpu_counters(),siblings)


def launch(config_path,grant_path):
    config,grant=map(gate.io.read,(config_path,grant_path))
    cpus=gate.authorize(config,grant,config_sha=gate.io.sha(config_path),affinity=os.sched_getaffinity(0))
    for name in ('output','control','lock'):gate.require(not Path(config[name]).exists(),'UNUSED_B1_PATHS_REQUIRED')
    base=Path(config['control']).parent;fresh_path=base/'B1_fresh_gate.json';attempt=base/'B1_launch_attempt.json'
    gate.require(not fresh_path.exists() and not attempt.exists(),'B1_ATTEMPT_ALREADY_SPENT')
    gate.verify_source(config);gate.normalize(config)  # Receipt inspection, never a fit.
    core_gate(cpus);value=strict_probe.snapshot();limits=gate.model.RESOURCES
    gate.require(observer.resource_reason(value,value,limits['aggregate_RSS_bytes']+limits['admission_reserve_bytes']) is None,
                 'FRESH_MEMORY_PSI_OOM')
    fs=cpu_io.filesystem(base)
    gate.require(fs['bytes']>=2*2**30 and fs['inodes']>=limits['minimum_free_inodes'],'FRESH_FILESYSTEM')
    fresh=dict(SOURCE=config['SOURCE'],config_sha256=gate.io.sha(config_path),grant_sha256=gate.model.digest(grant),
        monotonic_seconds=time.monotonic(),observer_identity=list(observer.owned.identity(os.getpid())),
        allowed_cpus=cpus,resources_verified=True,quota_verified=fs['quota_verified'],filesystem=fs)
    gate.write(fresh_path,fresh);gate.write(attempt,dict(SOURCE=config['SOURCE'],one_shot=True,scope=gate.model.SCOPE))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONSAFEPATH='1',TMPDIR=str(base),PYTHONPATH='')
    for key in gate.io.THREAD_KEYS:env[key]='1'
    command=[config['python'],'-P','-B',str(gate.ROOT/'scripts/run_h8_phase_b1_validation.py'),
             '--execute-approved','--config',str(config_path),'--grant',str(grant_path),'--fresh-gate',str(fresh_path)]
    old=os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0,{cpus[1]})
        observer.supervise(command,env=env,control=config['control'],output=config['output'],
            limits=limits,allowed_cpus=cpus,probe=strict_probe.snapshot)
        gate.verify_source(config);gate.verify_receipts(config['inputs'])
        gate.write(base/'B1_final_gate.json',dict(status='PASS_MANDATORY_STOP',SOURCE=config['SOURCE'],
            input_source_unchanged=True,B2_authorized=False,C_authorized=False,next_stage_authorized=False))
    except BaseException as exc:
        gate.write(base/'B1_final_STOP.json',dict(status='STOP',SOURCE=config['SOURCE'],
            first_stop_reason=type(exc).__name__+':'+str(exc),retry=False,refit=False,
            B2_authorized=False,C_authorized=False,next_stage_authorized=False))
        raise
    finally:os.sched_setaffinity(0,old)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--launch-approved',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--grant',type=Path);a=p.parse_args()
    if not a.launch_approved:
        if a.config or a.grant:p.error('real paths require approved launch mode')
        print(json.dumps(dict(status='PLAN_ONLY_NOT_AUTHORIZED',scope=gate.model.SCOPE)));return
    if not a.config or not a.grant:p.error('config and new explicit grant required')
    launch(a.config,a.grant)


if __name__=='__main__':main()

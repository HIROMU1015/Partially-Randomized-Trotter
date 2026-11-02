"""Plan-only default; numerical fit requires a new source/input-bound B1 grant."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import h8_phase_b1_validation as gate
import launch_h8_strengthened_phase_a as observer


def execute(config_path,grant_path,fresh_path):
    config,grant,fresh=map(gate.io.read,(config_path,grant_path,fresh_path))
    cpus=gate.authorize(config,grant,config_sha=gate.io.sha(config_path),affinity=os.sched_getaffinity(0))
    gate.verify_source(config);rows,binding=gate.normalize(config)
    stamp=fresh.get('monotonic_seconds')
    gate.require(type(stamp) in (int,float) and 0<=time.monotonic()-stamp<=30
        and fresh.get('SOURCE')==config['SOURCE'] and fresh.get('config_sha256')==gate.io.sha(config_path)
        and fresh.get('grant_sha256')==gate.model.digest(grant) and fresh.get('allowed_cpus')==cpus
        and fresh.get('resources_verified') is True and fresh.get('quota_verified') is True,
        'FRESH_B1_GATE_REQUIRED')
    gate.require(list(observer.owned.identity(os.getppid()) or ())==fresh.get('observer_identity'),
                 'LIVE_OWNED_OBSERVER_REQUIRED')
    gate.require(Path(sys.executable).resolve()==Path(config['python']).resolve()
        and all(os.environ.get(key)=='1' for key in gate.io.THREAD_KEYS),'PYTHON_THREAD_ONE_REQUIRED')
    output,lock=[gate.io.private_path(config[k]) for k in ('output','lock')]
    gate.require(not output.exists() and not lock.exists(),'ONE_SHOT_ALREADY_SPENT')
    gate.write(lock,dict(SOURCE=config['SOURCE'],scope=gate.model.SCOPE,grant_sha256=gate.model.digest(grant)))
    output.mkdir(mode=0o700)
    os.sched_setaffinity(0,{cpus[0]});limits=gate.model.RESOURCES
    resource.setrlimit(resource.RLIMIT_AS,(limits['driver_AS_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(limits['CPU_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(limits['output_bytes'],)*2)
    counts={k:0 for k in gate.model.CAPS};started=time.monotonic();events=0
    baseline=observer.snapshot()
    def guard():
        record=observer.process_record(os.getpid())
        gate.require(record is not None and not record['zombie'],'OWN_PROCESS_MISSING')
        gate.require(record['threads']==1 and record['RSS_bytes']<=limits['driver_RSS_bytes']
            and record['AS_bytes']<=limits['driver_AS_bytes'],'FIT_RESOURCE_LIMIT')
        gate.require(time.monotonic()-started<=limits['wall_seconds'] and
                     record['CPU_seconds']<=limits['CPU_seconds'],'FIT_TIME_LIMIT')
        gate.require(observer.resource_reason(observer.snapshot(),baseline,limits['admission_reserve_bytes']) is None,
                     'FIT_MEMORY_PSI_OOM')
        gate.require(list(observer.owned.identity(os.getppid()) or ())==fresh['observer_identity'],
                     'OBSERVER_OWNERSHIP_LOST')
    def save(name,value):
        size=len(json.dumps(value,allow_nan=False).encode())*2+1024
        gate.require(sum(p.stat().st_size for p in output.iterdir())+size<=limits['output_bytes'],
                     'AGGREGATE_OUTPUT_LIMIT')
        gate.write(output/name,value)
    def take(key):
        guard();gate.require(key in counts and counts[key]<gate.model.CAPS[key],'CALL_CAP:'+key)
        counts[key]+=1
        with (output/'calls.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(call=key,used=counts[key],refunded=False))+'\n');stream.flush();os.fsync(stream.fileno())
    def emit(event,value):
        nonlocal events
        events+=1;save(f'event_{events:03d}_{event}.json',value)
    def stopped(*_):raise RuntimeError('FIT_STOP_SIGNAL_OR_WALL_LIMIT')
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    for sig in previous:signal.signal(sig,stopped)
    signal.alarm(limits['wall_seconds'])
    try:
        guard()
        result=gate.model.evaluate(rows,acquisition_source=config['B0_acquisition_SOURCE'],
            input_binding=binding,take=take,emit=emit,guard=guard)
        gate.require(counts==gate.model.CAPS,'INCOMPLETE_FIXED_CALL_COUNTS')
        gate.verify_source(config);gate.verify_receipts(config['inputs']);guard()
        save('frozen_model.json',result.pop('frozen_model'))
        save('terminal.json',dict(result,SOURCE=config['SOURCE'],scope=gate.model.SCOPE,
            actual_calls=counts,wall_seconds=time.monotonic()-started,retry=False,
            B2_authorized=False,C_authorized=False,next_stage_authorized=False))
    except BaseException as exc:
        # Disable further signal exceptions while preserving the first reason.
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        gate.write(output/'terminal.json',dict(status='STOP',SOURCE=config['SOURCE'],
            scope=gate.model.SCOPE,first_stop_reason=type(exc).__name__+':'+str(exc),
            actual_calls=counts,retry=False,refit=False,additional_measurement=False,
            B2_authorized=False,C_authorized=False,next_stage_authorized=False))
        raise
    finally:
        signal.alarm(0)
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-approved',action='store_true')
    for key in ('config','grant','fresh-gate'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if not a.execute_approved:
        if any((a.config,a.grant,a.fresh_gate)):p.error('real inputs require approved mode')
        print(json.dumps(gate.model.plan(),indent=2));return
    if not all((a.config,a.grant,a.fresh_gate)):p.error('all sealed execution inputs required')
    execute(a.config,a.grant,a.fresh_gate)


if __name__=='__main__':main()

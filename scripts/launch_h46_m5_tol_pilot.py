"""Tol-only four-lane pilot; reuses the original scheduling and wait/reap algorithm.

No scientific imports on scope/draft rejection. Failure in a lane stops new calls
only in that lane. A shared-pool crash is recorded as a real own-process failure.
"""
import argparse
import concurrent.futures as futures
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

import h46_m5_tol_contract as c

CONFIGS={};STOPS={}


def memory_available():
    return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))


def observe():
    return dict(available_RAM_bytes=memory_available(),affinity=sorted(os.sched_getaffinity(0)),
        host_PSI_observation_only=Path('/proc/pressure/memory').read_text())


def fresh_resources(tasks,estimates):
    affinity=os.sched_getaffinity(0)
    def cpu():
        return {int(p[0][3:]):list(map(int,p[1:])) for line in Path('/proc/stat').read_text().splitlines()
                if (p:=line.split()) and p[0].startswith('cpu') and p[0][3:].isdigit() and int(p[0][3:]) in affinity}
    before=cpu();time.sleep(1);after=cpu();idle=0.
    for i in affinity:
        delta=[b-a for a,b in zip(before[i],after[i])];idle+=(delta[3]+delta[4])/max(1,sum(delta))
    c.require(all(type(v) is int and v>0 for v in estimates.values()),'POSITIVE_RSS_PLANNING_PROXIES')
    memory=memory_available();workers=min(len(tasks),len(affinity),math.floor(idle),memory//max(estimates.values()))
    c.require(workers>=1,'FRESH_RESOURCES_CANNOT_ADMIT_ONE_WORKER')
    return dict(workers=workers,idle_CPU_equivalents=idle,observation=observe(),
        RSS_proxies_are_bounds=False,historical_worker_cap_applied=False)


def decisions(tasks,completed,failed,pending,capacity):
    """Pure scheduler: representatives before replica1; no cross-lane STOP."""
    done=set(completed);bad=set(failed);busy=set(pending)
    return [t for t in tasks if t['lane'] not in bad and t['id'] not in done|busy
        and (t['sample']==0 or t['group']['id']+'__0' in done)][:max(0,capacity)]


def initialize(configs,stops):
    global CONFIGS,STOPS
    CONFIGS=configs;STOPS=stops


def reference_worker(lane):
    from h46_m5_tol_native import preflight
    return preflight(CONFIGS[lane])


def native_worker(task):
    from h46_m5_tol_native import run_task
    return run_task(CONFIGS[task['lane']],{k:v for k,v in task.items() if k!='lane'},STOPS[task['lane']])


def run_pool(configs,resources,*,initializer=initialize,worker=native_worker,reference=reference_worker):
    ctx=mp.get_context('spawn');stops={lane:ctx.Event() for lane in configs}
    pool=futures.ProcessPoolExecutor(max_workers=resources['workers'],mp_context=ctx,
        initializer=initializer,initargs=(configs,stops))
    groups={lane:cfg['plan']['scope']['tasks'] for lane,cfg in configs.items()}
    # Round-robin lanes even when the fresh resource allocation is small.
    tasks=[dict(**group[i],lane=lane) for i in range(max(map(len,groups.values())))
           for lane,group in groups.items() if i<len(group)]
    pending={};done=set();failed={};reference_results={};receipts=[];handlers={};start=time.monotonic()
    root=Path(next(iter(configs.values()))['run_root']);interrupted=False
    def interrupt(signum,frame):
        for e in stops.values():e.set()
        raise InterruptedError('OWN_RUN_SIGNAL_'+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupt)
    try:
        # These are the only full-H reference solver calls, one per system.
        refs={pool.submit(reference,lane):lane for lane in configs if configs[lane]['plan']['scope']['kind']=='DET'}
        for f in futures.as_completed(refs):
            lane=refs[f];system=configs[lane]['system']
            try:
                r=f.result();c.require(r.get('status')=='PASS','REFERENCE_NOT_PASS');reference_results[system]=r
            except Exception as exc:
                for other in configs:
                    if configs[other]['system']==system:failed[other]='SHARED_REFERENCE_FAILED:'+str(exc);stops[other].set()
        with (root/'observation_PRIVATE.jsonl').open('x') as journal:
            while True:
                for t in decisions(tasks,done,failed,[x['id'] for x in pending.values()],resources['workers']-len(pending)):
                    if memory_available()<configs[t['lane']]['RSS_proxy_bytes']:
                        if not pending:failed[t['lane']]='ACTUAL_RAM_CANNOT_ADMIT_NEXT_TASK';stops[t['lane']].set()
                        break
                    pending[pool.submit(worker,t)]=t
                if not pending:break
                finished,_=futures.wait(pending,timeout=2,return_when=futures.FIRST_COMPLETED)
                for f in finished:
                    t=pending.pop(f)
                    try:
                        r=f.result();c.require(r.get('task')==t['id'] and r.get('lane')==t['lane'],'RESULT_IDENTITY')
                        if r.get('status')!='PASS':raise ValueError(r.get('reason','WORKER_NOT_PASS'))
                        c.require(not stops[t['lane']].is_set(),'RESULT_AFTER_OWN_LANE_STOP')
                        done.add(t['id']);receipts.append(dict(lane=t['lane'],task=t['id'],status='PASS'))
                    except Exception as exc:
                        failed[t['lane']]=str(exc);stops[t['lane']].set()
                journal.write(json.dumps(dict(elapsed=time.monotonic()-start,completed=len(done),
                    failed=failed,pending=len(pending),resources=observe()))+'\n');journal.flush()
    except BaseException as exc:
        interrupted=True
        for lane in configs:failed.setdefault(lane,'OWN_LAUNCHER_FAILURE:'+str(exc));stops[lane].set()
        for f in pending:f.cancel()
        # Own broken/interrupted launcher only. Never respond to host PSI by kill.
        for p in list(pool._processes.values()):
            if p.is_alive():p.terminate()
    finally:
        processes=list(pool._processes.values());pool.shutdown(wait=True,cancel_futures=True)
        for sig,handler in handlers.items():signal.signal(sig,handler)
    witnesses={}
    reaped=all(not p.is_alive() for p in processes)
    exits=[dict(pid=p.pid,exit_code=p.exitcode,reaped=not p.is_alive()) for p in processes]
    for lane,cfg in configs.items():
        expected={t['id'] for t in cfg['plan']['scope']['tasks']}
        entries=len(list(Path(cfg['ledger']).glob('*.json')))
        error=failed.get(lane)
        try:
            c.verify_seals(cfg)
            c.require(expected<=done and entries==cfg['plan']['scope']['compile_axes'],'FIXED_LANE_COVERAGE')
            c.require(reaped and all(p['exit_code']==0 for p in exits),'OWN_NATIVE_EXIT_AND_REAP')
        except Exception as exc:error=error or str(exc)
        witness=dict(status='COMPLETE_MANDATORY_STOP' if error is None else 'STOP',error=error,
            lane=lane,SOURCE=cfg['SOURCE'],completed_tasks=len(expected&done),compile_entries=entries,
            native_workers=exits,all_reaped=reaped,retry=False,phase3_authorized=False,mandatory_STOP=True)
        c.save(Path(cfg['output'])/'completion_witness.json',witness);witnesses[lane]=witness
    return dict(lanes=witnesses,wall_seconds=time.monotonic()-start,interrupted=interrupted,
        all_reaped=reaped,next_phase_started=False,mandatory_STOP=True)


def execute(bundle):
    configs={lane:c.read(path) for lane,path in bundle['configs'].items()}
    c.require(set(configs)==set(c.LANES),'FOUR_PILOT_CONFIGS_REQUIRED')
    for lane,cfg in configs.items():c.require(cfg['lane']==lane,'LANE_KEY');c.authorize(cfg)
    roots={cfg['run_root'] for cfg in configs.values()};c.require(len(roots)==1,'ONE_OWNED_RUN_PARENT')
    root=Path(roots.pop());c.require(not root.exists(),'FRESH_RUN_ROOT_REQUIRED')
    c.require(len({cfg['grant']['grant_id'] for cfg in configs.values()})==4,'FOUR_DISTINCT_GRANTS')
    c.require(len({Path(cfg[k]).resolve() for cfg in configs.values() for k in ('output','ledger','marker')})==12,'LANE_PATH_COLLISION')
    c.require(len({cfg['SOURCE'] for cfg in configs.values()})==1,'ONE_SEALED_SCIENTIFIC_SOURCE')
    c.require(len({c.core.digest(cfg['runtime']) for cfg in configs.values()})==1,'ONE_SHARED_SEALED_RUNTIME')
    c.require(sys.executable==next(iter(configs.values()))['runtime']['python'],'FIXED_PYTHON_EXECUTABLE')
    tasks=[t for cfg in configs.values() for t in cfg['plan']['scope']['tasks']]
    resources=fresh_resources(tasks,{lane:cfg['RSS_proxy_bytes'] for lane,cfg in configs.items()})
    root.mkdir(parents=True,exist_ok=False)
    for cfg in configs.values():
        Path(cfg['output']).mkdir();Path(cfg['ledger']).mkdir()
        c.save(cfg['marker'],dict(state='CONSUMED_NO_REFUND_NO_RETRY',grant=cfg['grant']))
    c.save(root/'fresh_gate.json',resources)
    # Changes only this launcher's newly spawned children; restored on return.
    names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_PARALLEL')
    previous={n:os.environ.get(n) for n in names}
    try:
        for n in names:os.environ[n]='FALSE' if n=='QISKIT_PARALLEL' else '1'
        witness=run_pool(configs,resources);c.save(root/'completion_witness.json',witness)
    finally:
        for n,v in previous.items():
            if v is None:os.environ.pop(n,None)
            else:os.environ[n]=v
    return 0 if all(x['status']=='COMPLETE_MANDATORY_STOP' for x in witness['lanes'].values()) else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status='SCOPE_ONLY',lanes={x:c.scope(x) for x in c.LANES},sampling_started=False)));return 0
    try:
        c.require(a.bundle,'APPROVED_BUNDLE_REQUIRED');return execute(c.read(a.bundle))
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())

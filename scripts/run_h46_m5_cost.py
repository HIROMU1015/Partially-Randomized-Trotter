"""One selected stage per invocation. Default scope only; no draft execution."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import threading
from types import FunctionType
import h46_m5_cost_contract as c
import run_h6_m5_saved_phase as observed

supervise=FunctionType(observed.supervise.__code__,dict(observed.__dict__,c=c,__file__=__file__),'supervise',observed.supervise.__defaults__,observed.supervise.__closure__)
WORKER_CFG=None
WORKER_STOP=None

def initialize(cfg,stop):
    global WORKER_CFG,WORKER_STOP
    WORKER_CFG,WORKER_STOP=cfg,stop

def worker(task):
    from h46_m5_cost_native import run_task
    return run_task(WORKER_CFG,task,WORKER_STOP)

def fresh(cfg):
    from run_h46_m5_signal import fresh_resources
    # Same fresh CPU/cgroup/disk observer, then memory admission for this stage.
    probe=dict(cfg,systems={'H4':{},'H6':{}},resources=dict(system_RSS_proxy_bytes={'H4':1,'H6':1},disk_proxy_bytes=cfg['resources']['disk_proxy_bytes']))
    r=fresh_resources(probe);proxy=cfg['resources']['worker_RSS_proxy_bytes']
    c.require(type(proxy) is int and proxy>0,'RESOURCE_PROXY_REQUIRED_NOT_A_BOUND')
    workers=min(len(cfg['tasks']),math.floor(r['idle_CPU_equivalents']),r['effective_available_RAM_bytes']//proxy)
    c.require(workers>=1,'CANNOT_ADMIT_ONE_COST_WORKER')
    return dict(workers=workers,worker_RSS_proxy_bytes=proxy,proxy_is_upper_bound=False,runtime_caps=False,observation=r)

def execute(cfg,*,target=worker):
    c.authorize(cfg);c.require(sys.executable==cfg['runtime']['python'],'COST_FIXED_PYTHON')
    root=Path(cfg['run_root']);c.require(not root.exists(),'UNUSED_COST_RUN_ROOT');resources=fresh(cfg)
    root.mkdir();Path(cfg['output']).mkdir();Path(cfg['ledger']).mkdir();c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']));c.save(root/'fresh_gate.json',resources)
    ctx=mp.get_context('spawn');stop=ctx.Event();errors={};results=[];signals=[];started=time.monotonic();handlers={};pids={}
    keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_PARALLEL');oldenv={k:os.environ.get(k) for k in keys}
    def interrupt(sig,frame):signals.append(sig);stop.set()
    monitoring_done=threading.Event();monitor=None
    def monitor_resources():
        with (root/'resource_observation.jsonl').open('x') as journal:
            while not monitoring_done.is_set():
                try:row=dict(elapsed=time.monotonic()-started,resources=observed.observe())
                except Exception as exc:row=dict(elapsed=time.monotonic()-started,observation_error=str(exc),kill_worker=False)
                journal.write(json.dumps(row)+'\n');journal.flush();monitoring_done.wait(5)
    try:
        for k in keys:os.environ[k]='FALSE' if k=='QISKIT_PARALLEL' else '1'
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupt)
        monitor=threading.Thread(target=monitor_resources,daemon=True);monitor.start()
        with ProcessPoolExecutor(max_workers=resources['workers'],mp_context=ctx,initializer=initialize,initargs=(cfg,stop)) as pool:
            # Retain own PID/exit evidence before shutdown releases pool handles.
            futures={pool.submit(target,t):t for t in cfg['tasks']}
            processes=list(pool._processes.values());pids={p.pid:p for p in processes}
            with (root/'task_completion.jsonl').open('x') as journal:
                for future in as_completed(futures):
                    task=futures[future]
                    try:
                        entry=future.result();result=c.read(Path(cfg['output'])/task['id']/'result.json')
                        c.require(entry['id']==task['id'] and entry['result_sha256']==c.sha(Path(cfg['output'])/task['id']/'result.json'),'COST_RESULT_SHA')
                        c.require(result['status']=='UNIT_COST_PASS_NOT_ACCOUNTING' and result['task']==task and result['SOURCE']==cfg['SOURCE'] and result['plan_sha256']==cfg['plan_sha256'] and result['stage']==cfg['stage'],'COST_RESULT_IDENTITY')
                        c.require([x['axis'] for x in result['axis_costs']]==['cosine','sine'] and all(x['compiler_settings_hash']==c.design.COMPILER_HASH and x['native_policy']==c.design.NATIVE_POLICY for x in result['axis_costs']),'PAIRED_COST_RECEIPTS')
                        c.require(not result['shot_accounting'] and not result['cost_ranking'] and not result['conditional_sampling'] and not result['cohort_pooling'],'UNIT_COST_SCOPE')
                        results.append(entry)
                    except BaseException as exc:
                        errors.setdefault(task['id'],str(exc));stop.set()
                        for f in futures:f.cancel()
                    journal.write(json.dumps(dict(elapsed=time.monotonic()-started,completed=len(results),errors=errors,resources=observed.observe()))+'\n');journal.flush()
    except BaseException as exc:errors.setdefault('launcher',str(exc));stop.set()
    finally:
        monitoring_done.set()
        if monitor is not None:monitor.join()
        for sig,h in handlers.items():signal.signal(sig,h)
        for k,v in oldenv.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    workers={str(pid):dict(exit=p.exitcode,reaped=not p.is_alive()) for pid,p in pids.items()}
    try:
        c.verify(cfg);c.consumed(cfg)
        c.require(workers and all(x['exit']==0 and x['reaped'] for x in workers.values()),'COST_WORKER_EXIT_OR_REAP')
        c.require(len(results)==len(cfg['tasks']),'INCOMPLETE_COST_STAGE')
        expected={t['id']+'__'+kind+'__'+str(i)+'.json' for t in cfg['tasks'] for kind,n in c.task_limits(t).items() for i in range(n)}
        c.require({p.name for p in Path(cfg['ledger']).glob('*.json')}==expected,'EXACT_COST_CALL_LEDGER')
    except Exception as exc:errors.setdefault('end_gate',str(exc))
    if signals:errors['signals']=signals
    c.save(root/'completion_witness.json',dict(status='STOP' if errors else 'UNIT_COST_STAGE_COMPLETE_MANDATORY_STOP',SOURCE=cfg['SOURCE'],stage=cfg['stage'],plan_sha256=cfg['plan_sha256'],results=results,errors=errors,workers=workers,wall_seconds=time.monotonic()-started,shot_calculation=0,cost_accounting=False,ranking=False,retry=False,next_phase_started=False,mandatory_STOP=True))
    return 1 if errors else 0

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--execute',action='store_true');g.add_argument('--supervise',action='store_true');p.add_argument('--config');a=p.parse_args()
    if not a.execute and not a.supervise:
        print(json.dumps(dict(status='SCOPE_ONLY',stages=c.STAGES,n_E=None,science_started=False,contract='SEPARATE_FROM_PF788',mandatory_STOP=True)));return 0
    try:
        c.require(a.config,'COST_CONFIG_REQUIRED');cfg=c.read(a.config);c.authorize(cfg)
        if a.supervise:return supervise(cfg,a.config)
        c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),'EXTERNAL_COST_SUPERVISOR_REQUIRED')
        return execute(cfg)
    except Exception as exc:print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2

if __name__=='__main__':raise SystemExit(main())

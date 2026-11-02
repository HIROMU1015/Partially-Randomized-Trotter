"""One-shot bounded campaign, adaptive trajectory pool, per-group sample0 gate."""
import argparse
import concurrent.futures as futures
import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

from h46_acquire_native import initialize, run_task, verify, write, sha


def validate(c):
    plan = c['plan']
    if not c.get('authorized') or c['grant']['valid'] is not True:
        raise ValueError('EXECUTION_APPROVAL_REQUIRED')
    if c['grant']['phase'] != 'B_ACQUIRE_MISSING' or c['grant']['SOURCE'] != c['SOURCE']:
        raise ValueError('GRANT_SCOPE_SOURCE')
    if c['grant']['plan_sha256'] != c['plan_sha256']:
        raise ValueError('GRANT_PLAN_BINDING')
    if len(plan['groups']) != 12 or len(set(g['id'] for g in plan['groups'])) != 12:
        raise ValueError('TWELVE_FIXED_GROUPS_REQUIRED')
    expected = {(s, label, q) for s in ('H4','H6') for label,qs in
                (('2nd',(8,)),('4th',(4,8)),('4th(new_2)',(2,4,8))) for q in qs}
    if {(g['system'],g['label'],g['q']) for g in plan['groups']} != expected:
        raise ValueError('SCIENTIFIC_SCOPE')
    if any(g['prefix'] != (3 if g['system']=='H4' else 5) for g in plan['groups']):
        raise ValueError('PREFIX_SCOPE')
    tasks = [g['id']+'/sample_%02d'%i for g in plan['groups'] for i in range(8)]
    if sorted(tasks) != sorted(plan['tasks']) or len(set(plan['tasks'])) != 96:
        raise ValueError('TASK_REGISTRY')
    if plan['sample_axes'] != 192 or plan['extra_scientific_compiles'] != 0:
        raise ValueError('COMPILE_BUDGET')
    return tasks


def memory_available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))


def resource_observation():
    return {'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'MemAvailable':memory_available(),'affinity':sorted(os.sched_getaffinity(0)),
            'host_memory_pressure':Path('/proc/pressure/memory').read_text()}


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    c=json.loads(Path(args.config).read_text());validate(c);verify(c)
    out=Path(c['output']);out.mkdir(exist_ok=False)
    Path(c['call_ledger']).mkdir(exist_ok=False)
    write(out/'consumed_grant_PRIVATE.json',{'state':'CONSUMED_ONE_SHOT_NO_RETRY','grant':c['grant']})
    write(out/'fresh_resources_PRIVATE.json',resource_observation())
    ctx=mp.get_context('spawn');stop=ctx.Event();started=time.monotonic()
    def interrupted(signum, frame):
        stop.set();raise InterruptedError('OWN_RUN_SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    workers=c['resources']['workers'];pool=futures.ProcessPoolExecutor(max_workers=workers,mp_context=ctx,
            initializer=initialize,initargs=(args.config,stop))
    pending={};complete=[];waiting=[(g,0) for g in c['plan']['groups']];worker_processes=[];error=None
    try:
        with (out/'observation_PRIVATE.jsonl').open('x') as journal:
            while waiting or pending:
                if stop.is_set():raise RuntimeError('OWN_CAMPAIGN_STOP')
                # Reuse idle processes and cached immutable preparations. In-flight
                # reservations derive from measured old RSS scaled by q, not a cap.
                while waiting and len(pending)<workers:
                    g,sample=waiting[0]
                    reserved=sum(c['resources']['group_estimated_RSS'][v[0]['id']] for v in pending.values())
                    needed=c['resources']['group_estimated_RSS'][g['id']]
                    if reserved+needed>memory_available() and pending:break
                    if needed>memory_available():raise RuntimeError('ACTUAL_RESOURCES_CANNOT_ADMIT_NEXT_TASK')
                    waiting.pop(0)
                    future=pool.submit(run_task,g,sample);pending[future]=(g,sample)
                done,_=futures.wait(pending,timeout=2,return_when=futures.FIRST_COMPLETED)
                for future in done:
                    group,sample=pending.pop(future);receipt=future.result();complete.append(receipt)
                    if sample==0:
                        waiting.extend((group,i) for i in range(1,8))
                    write(out/(group['id']+'__%02d_COMPLETE.json'%sample),receipt)
                observation={'elapsed':time.monotonic()-started,'complete':len(complete),'running_or_queued':len(pending),
                             'waiting':len(waiting),'resources':resource_observation()}
                journal.write(json.dumps(observation)+'\n');journal.flush()
                if done:print(json.dumps(observation),flush=True)
        verify(c)
        entries=list(Path(c['call_ledger']).glob('*.json'))
        if len(complete)!=96 or len(entries)!=192:raise ValueError('FINAL_FIXED_BUDGET_NOT_COMPLETE')
    except BaseException as exc:
        stop.set();error={'type':type(exc).__name__,'reason':str(exc)}
        for future in pending:future.cancel()
        for process in list(pool._processes.values()):
            if process.is_alive():process.terminate()
    finally:
        worker_processes=list(pool._processes.values())
        pool.shutdown(wait=True,cancel_futures=True)
        completion={'status':'COMPLETE_ACQUISITION_ONLY' if error is None else 'STOP','SOURCE':c['SOURCE'],
                    'error':error,'completed_trajectories':len(complete),'compile_entries':len(list(Path(c['call_ledger']).glob('*.json'))),
                    'pool_workers':[{'pid':p.pid,'exit_code':p.exitcode,'reaped':not p.is_alive()} for p in worker_processes],
                    'all_pool_workers_reaped':all(not p.is_alive() for p in worker_processes),
                    'wall_seconds':time.monotonic()-started,'weighted_accounting_started':False,'ranking_started':False,
                    'C_INDEPENDENT_started':False,'retry':False,'mandatory_STOP':True}
        write(out/'completion_witness_PRIVATE.json',completion)
    return 0 if error is None else 1


if __name__=='__main__':raise SystemExit(main())

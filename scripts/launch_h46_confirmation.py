"""One-shot H6 confirmation. Default is scope only; no native imports on refusal."""
import argparse
import concurrent.futures as futures
import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

from h46_confirmation_contract import GROUPS, PROPOSED_N, authorize, task_id, write


def memory_available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))


def observe():
    return {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'available_RAM_bytes': memory_available(), 'affinity': sorted(os.sched_getaffinity(0)),
            'host_memory_PSI_observation_only': Path('/proc/pressure/memory').read_text()}


def fresh_resources(config):
    affinity = os.sched_getaffinity(0)
    def cpu():
        rows = [line.split() for line in Path('/proc/stat').read_text().splitlines()]
        return {int(p[0][3:]): list(map(int,p[1:])) for p in rows if p[0].startswith('cpu') and p[0][3:].isdigit() and int(p[0][3:]) in affinity}
    before=cpu();time.sleep(1);after=cpu();idle=0.
    for index in affinity:
        delta=[b-a for a,b in zip(before[index],after[index])]
        idle += (delta[3]+delta[4])/max(1,sum(delta))
    estimates=config['resources']['group_estimated_RSS']
    if set(estimates)!=set(PROPOSED_N) or any(type(v) is not int or v<=0 for v in estimates.values()):
        raise ValueError('MEASURED_RSS_PROXIES_REQUIRED')
    memory=memory_available()
    if memory < max(estimates.values()): raise ValueError('FRESH_RAM_CANNOT_ADMIT_TASK')
    if idle < 1: raise ValueError('FRESH_CPU_CANNOT_ADMIT_WORKER')
    workers=min(103,len(affinity),math.floor(idle),memory//max(estimates.values()))
    return {'workers': workers, 'idle_CPU_equivalents':idle, 'observation':observe(),
            'policy':'MEASURED_RSS_ADMISSION_DYNAMIC_CPU_NO_HISTORICAL_CAP',
            'memory_estimates_are_bounds':False}


def run_pool(config, config_path, workers, initializer, worker):
    """Programmatic worker injection is for artificial tests, never a CLI backend."""
    out=Path(config['output']);ctx=mp.get_context('spawn');stop=ctx.Event()
    pool=futures.ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initializer,initargs=(config_path,stop))
    pending={};waiting=[(g,0) for g in GROUPS];complete=[];error=None;started=time.monotonic()
    old_handlers={}
    def interrupted(signum,frame):
        stop.set();raise InterruptedError('OWN_RUN_SIGNAL_'+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT):
        old_handlers[signum]=signal.signal(signum,interrupted)
    try:
        with (out/'observation_PRIVATE.jsonl').open('x') as journal:
            while waiting or pending:
                if stop.is_set(): raise RuntimeError('OWN_CONFIRMATION_STOP')
                while waiting and len(pending)<workers:
                    group,sample=waiting[0]
                    reserved=sum(config['resources']['group_estimated_RSS'][v[0]['id']] for v in pending.values())
                    needed=config['resources']['group_estimated_RSS'][group['id']]
                    available=memory_available()
                    if needed>available and not pending: raise RuntimeError('ACTUAL_RAM_CANNOT_ADMIT_NEXT_TASK')
                    if reserved+needed>available: break
                    waiting.pop(0);pending[pool.submit(worker,group,sample)]=(group,sample)
                if not pending: raise RuntimeError('NO_TASK_CAN_BE_ADMITTED')
                done,_=futures.wait(pending,timeout=2,return_when=futures.FIRST_COMPLETED)
                for future in done:
                    group,sample=pending.pop(future);receipt=future.result()
                    if receipt.get('status')!='PASS' or receipt.get('stage')!='CONFIRMATION' or (receipt.get('group'),receipt.get('sample'))!=(group['id'],sample):
                        raise ValueError('CONFIRMATION_COMPLETION_IDENTITY')
                    complete.append(receipt)
                    write(out/(group['id']+'__%03d_COMPLETE.json'%sample),receipt)
                    if sample==0: waiting.extend((group,i) for i in range(1,PROPOSED_N[group['id']]))
                journal.write(json.dumps({'elapsed':time.monotonic()-started,'completed':len(complete),
                    'pending':len(pending),'waiting':len(waiting),'resources':observe()})+'\n');journal.flush()
        if len(complete)!=103 or len(list(Path(config['call_ledger']).glob('*.json')))!=206:
            raise ValueError('FIXED_CONFIRMATION_BUDGET_INCOMPLETE')
    except BaseException as exc:
        stop.set();error={'type':type(exc).__name__,'reason':str(exc)}
        for future in pending: future.cancel()
        # These processes were created by this pool, never another user's jobs.
        for process in list(pool._processes.values()):
            if process.is_alive(): process.terminate()
    finally:
        processes=list(pool._processes.values())
        pool.shutdown(wait=True,cancel_futures=True)
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
    return {'status':'COMPLETE_NATIVE_CONFIRMATION' if error is None else 'STOP','error':error,
            'completed_trajectories':len(complete),'compile_entries':len(list(Path(config['call_ledger']).glob('*.json'))),
            'workers':[{'pid':p.pid,'exit_code':p.exitcode,'reaped':not p.is_alive()} for p in processes],
            'all_workers_reaped':all(not p.is_alive() for p in processes),
            'wall_seconds':time.monotonic()-started,'retry':False,'mandatory_STOP':True}


def execute(config, config_path):
    # All these gates precede output, one-shot marker, imports and scientific work.
    seed_audit=authorize(config);resources=fresh_resources(config)
    for key in ('output','call_ledger','grant_marker'):
        if Path(config[key]).exists():raise ValueError('ONE_SHOT_PATH_ALREADY_EXISTS:'+key)
    marker=Path(config['grant_marker']);marker.parent.mkdir(parents=True,exist_ok=True)
    write(marker,{'state':'CONSUMED_NO_REFUND_NO_RETRY','grant':config['grant']})
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    Path(config['call_ledger']).mkdir(parents=True,exist_ok=False)
    write(out/'fresh_gate.json',{'resources':resources,'seed_audit':seed_audit})
    # Dedicated child processes use trajectory parallelism, avoiding nested BLAS
    # and transpiler pools. This changes this launcher's child environment only.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    os.environ['QISKIT_PARALLEL']='FALSE'
    from h46_confirmation_native import initialize,run_task
    witness=run_pool(config,config_path,resources['workers'],initialize,run_task)
    if witness['status']=='COMPLETE_NATIVE_CONFIRMATION':
        try:
            authorize(config)
            if any(p['exit_code']!=0 for p in witness['workers']) or not witness['all_workers_reaped']:
                raise ValueError('NATIVE_EXIT_OR_REAP')
            from h46_confirmation_collect import collect
            result=collect(config)
            write(out/'confirmation_comparison.json',result)
            witness['comparison_status']=result['point_order_status']
        except BaseException as exc:
            witness.update(status='STOP',error={'type':type(exc).__name__,'reason':str(exc)})
    write(out/'completion_witness.json',witness)
    return 0 if witness['status']=='COMPLETE_NATIVE_CONFIRMATION' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'status':'SCOPE_ONLY_UNAPPROVED','n_proposed':PROPOSED_N,'trajectories':103,'cost_axes':206,'sampling_started':False}));return 0
    try:
        if not args.config: raise ValueError('APPROVED_CONFIG_REQUIRED')
        return execute(json.loads(Path(args.config).read_text()),str(Path(args.config).resolve()))
    except Exception as exc:
        print(json.dumps({'status':'REFUSED','reason':str(exc),'automatic_retry':False}));return 2


if __name__=='__main__': raise SystemExit(main())

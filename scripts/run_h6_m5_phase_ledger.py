"""One fixed scalar ledger. Reuse observed resource/wait4 code, no replay path."""
import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from types import FunctionType
import h6_m5_phase_ledger_contract as c
import run_h6_m5_saved_phase as observed

# Reuse precisely the reviewed external supervisor with isolated globals.
supervise=FunctionType(observed.supervise.__code__,dict(observed.__dict__,c=c,__file__=__file__),
    'supervise',observed.supervise.__defaults__,observed.supervise.__closure__)


def execute(cfg):
    c.authorize(cfg)
    c.require(sys.executable==cfg['runtime']['python'],'FIXED_PYTHON')
    root=Path(cfg['run_root']);c.require(not root.exists(),'UNUSED_OUTPUT_ROOT')
    resources=observed.fresh_resources(cfg);resources['workers']=min(2,resources['workers'])
    resources['productive_jobs']=2
    root.mkdir(exist_ok=False);Path(cfg['output']).mkdir();Path(cfg['ledger']).mkdir()
    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
    c.save(root/'fresh_gate.json',resources)
    started=time.monotonic();processes=[];pool=None;handlers={};signals=[];error=None;results={}
    def interrupt(signum,frame):signals.append(signum)
    names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_PARALLEL')
    prior={key:os.environ.get(key) for key in names}
    try:
        for key in names:os.environ[key]='FALSE' if key=='QISKIT_PARALLEL' else '1'
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
        import mpmath
        c.require(mpmath.__version__=='1.3.0' and mpmath.libmp.BACKEND=='python','FIXED_MPMATH_BACKEND')
        c.require(str(Path(mpmath.__file__).resolve()) in {r['path'] for r in cfg['sealed_files']},'LOADED_MPMATH_SEALED')
        from h6_m5_phase_ledger_core import extract_worker,evaluate
        pool=futures.ProcessPoolExecutor(max_workers=resources['workers'],mp_context=mp.get_context('spawn'))
        pending={};kinds=iter(('WRAPPER','COMPILED'));exhausted=False
        with (root/'observation.jsonl').open('x') as journal:
            while True:
                if signals:error=error or 'OWN_RUN_SIGNAL'
                while not error and not exhausted and len(pending)<resources['workers']:
                    if observed.available_memory()<cfg['resources']['per_worker_RSS_proxy_bytes']:
                        if not pending:error='ACTUAL_RAM_CANNOT_ADMIT_NEXT_EXTRACTION'
                        break
                    try:kind=next(kinds)
                    except StopIteration:exhausted=True;break
                    c.reserve(cfg,'EXTRACT_'+kind)
                    pending[pool.submit(extract_worker,cfg,kind)]=kind
                    processes=list(pool._processes.values())
                if not pending:break
                finished,_=futures.wait(pending,timeout=1,return_when=futures.FIRST_COMPLETED)
                for future in finished:
                    kind=pending.pop(future)
                    try:
                        result=future.result()
                        c.require(result['kind']==kind and result['SOURCE']==cfg['SOURCE'] and result['plan_sha256']==cfg['plan_sha256'],'EXTRACTION_RESULT_IDENTITY')
                        results[kind]=result
                    except Exception as exc:error=error or str(exc)
                journal.write(json.dumps(dict(elapsed=time.monotonic()-started,completed=list(results),pending=list(pending.values()),error=error,resources=observed.observe()))+'\n');journal.flush()
        pool.shutdown(wait=True,cancel_futures=True);pool=None
        c.require(error is None,error);c.require(not signals,'OWN_RUN_SIGNAL')
        c.require(set(results)=={'WRAPPER','COMPILED'},'EXACT_SAVED_PAIR')
        c.require(all(p.exitcode==0 and not p.is_alive() for p in processes),'WORKER_EXIT_AND_REAP')
        c.verify_seals(cfg);c.consumed(cfg)
        c.reserve(cfg,'SCALAR_LEDGER')
        saved=c.read(cfg['inputs']['saved_report']['path'])
        result=evaluate(results['WRAPPER'],results['COMPILED'],{mode:saved[mode]['common_phase'] for mode in ('ordinary','extended')})
        result.update(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
            extraction_result_sha256={kind:c.sha(Path(cfg['output'])/('extract_'+kind+'.json')) for kind in results})
        c.save(root/'scalar_ledger_result.json',result)
    except BaseException as exc:
        error=error or str(exc)
        c.save(root/'STOP.json',dict(status='STOP',reason=error,retry=False))
    finally:
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        for key,value in prior.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    try:c.verify_seals(cfg);c.consumed(cfg)
    except Exception as exc:error=error or str(exc)
    workers=[dict(pid=p.pid,exit=p.exitcode,reaped=not p.is_alive()) for p in processes]
    if any(not w['reaped'] or w['exit']!=0 for w in workers):error=error or 'NATIVE_WORKER_FAILURE'
    ledger_names={p.stem for p in Path(cfg['ledger']).glob('*.json')}
    if error is None and ledger_names!={'EXTRACT_WRAPPER','EXTRACT_COMPILED','SCALAR_LEDGER'}:error='EXACT_STAGE_LEDGER'
    result_path=root/'scalar_ledger_result.json'
    c.save(root/'completion_witness.json',dict(status='STOP' if error else 'SCALAR_LEDGER_COMPLETE_MANDATORY_STOP',reason=error,
        SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],native_workers=workers,all_reaped=all(w['reaped'] for w in workers),
        stage_entries=sorted(ledger_names),wall_seconds=time.monotonic()-started,signals=signals,
        result_sha256=c.sha(result_path) if result_path.exists() else None,
        new_compile=0,new_statevector_replays=0,new_sampling=0,accepted_new_cost_axes=0,
        old_STOP_remains_STOP=True,retry=False,next_phase_started=False,mandatory_STOP=True))
    return 1 if error else 0


def main():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group();m.add_argument('--supervise',action='store_true');m.add_argument('--execute',action='store_true');p.add_argument('--config');args=p.parse_args()
    if not args.execute and not args.supervise:
        print(json.dumps(dict(status='SCOPE_ONLY',scope=c.scope(),science_started=False)));return 0
    try:
        c.require(args.config,'NEW_CONFIG_REQUIRED');cfg=c.read(args.config)
        if args.supervise:return supervise(cfg,args.config)
        c.authorize(cfg)
        c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),'SUPERVISOR_REQUIRED')
        return execute(cfg)
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())

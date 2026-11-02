"""One authorized point, sequential axes, observed child and explicit exit/reap."""
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import h6_m5_recovery_contract as c
from launch_h46_m5_tol_pilot import fresh_resources, observe


def worker_entry(cfg, stop):
    try:
        from h6_m5_recovery_native import run_task
        result = run_task(cfg, stop)
        c.save(Path(cfg['run_root'])/'worker_return_PRIVATE.json', result)
    except BaseException as exc:
        c.save(Path(cfg['run_root'])/'worker_error_PRIVATE.json',
            dict(status='STOP', reason=str(exc), traceback=traceback.format_exc()))
        raise


def execute(cfg, *, worker=worker_entry):
    c.authorize(cfg)
    root = Path(cfg['run_root'])
    c.require(not root.exists(), 'FRESH_RUN_ROOT_REQUIRED')
    c.require(sys.executable == cfg['runtime']['python'], 'FIXED_PYTHON_EXECUTABLE')
    resources = fresh_resources([c.TASK], {c.LANE:cfg['RSS_proxy_bytes']})
    disk = shutil.disk_usage(root.parent).free
    c.require(disk >= cfg['disk_free_planning_proxy_bytes'], 'FRESH_DISK_CANNOT_ADMIT_EVIDENCE')
    resources.update(disk_free_bytes=disk, disk_proxy_is_bound=False,
        point_count=1, axes_sequential=True, historical_worker_cap_applied=False)
    root.mkdir(exist_ok=False)
    Path(cfg['output']).mkdir();Path(cfg['ledger']).mkdir()
    c.save(cfg['marker'], dict(state='CONSUMED_NO_REFUND_NO_RETRY', grant=cfg['grant']))
    c.save(root/'fresh_gate.json', resources)
    names = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_PARALLEL')
    previous = {k:os.environ.get(k) for k in names}
    ctx=mp.get_context('spawn');stop=ctx.Event();process=ctx.Process(target=worker,args=(cfg,stop))
    handlers={};interrupted=False;started=time.monotonic()
    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted=True;stop.set()
    try:
        for k in names:os.environ[k]='FALSE' if k=='QISKIT_PARALLEL' else '1'
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
        process.start()
        c.save(root/'worker_start_PRIVATE.json',dict(pid=process.pid, launcher_pid=os.getpid()))
        with (root/'observation_PRIVATE.jsonl').open('x') as journal:
            while process.is_alive():
                process.join(timeout=2)
                journal.write(json.dumps(dict(elapsed=time.monotonic()-started,
                    worker_alive=process.is_alive(), resources=observe()))+'\n');journal.flush()
        process.join()
    finally:
        if process.pid is not None:process.join()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    error=None;result=None
    try:
        c.require(not interrupted, 'OWN_RUN_SIGNAL')
        c.require(process.exitcode==0 and not process.is_alive(), 'NATIVE_CHILD_EXIT_AND_REAP')
        c.verify_seals(cfg);c.consumed(cfg)
        result=c.read(root/'worker_return_PRIVATE.json')
        c.require(result.get('status')=='PASS', result.get('reason','FIXED_POINT_NOT_PASS'))
        c.require(result['lane']==c.LANE and result['task']==c.TASK['id']
            and result['SOURCE']==cfg['SOURCE'] and result['plan_sha256']==cfg['plan_sha256'], 'RESULT_IDENTITY')
        c.require(len(result['checks'])==8 and all(math.isfinite(x['error']) and 0<=x['error']<1e-9 for x in result['checks']), 'REPRESENTATIVE_CHECKS')
        c.require(len(list(Path(cfg['ledger']).glob('*.json')))==2, 'FIXED_TWO_AXIS_COVERAGE')
        taskroot=Path(cfg['output'])/c.TASK['id']
        c.require(c.read(taskroot/'result.json')==result, 'PERSISTED_RESULT_IDENTITY')
        c.require(all((taskroot/('cost_'+axis+'.json')).is_file() for axis in ('cosine','sine')), 'ACCEPTED_PAIRED_COST_FILES')
    except Exception as exc:error=str(exc)
    witness=dict(status='COMPLETE_MANDATORY_STOP' if error is None else 'STOP', reason=error,
        SOURCE=cfg['SOURCE'], origin_SOURCE=cfg['origin_SOURCE'], task=c.TASK['id'],
        compile_entries=len(list(Path(cfg['ledger']).glob('*.json'))),
        accepted_cost_axes=len(list(Path(cfg['output']).glob('*/cost_*.json'))),
        worker_pid=process.pid, worker_exit=process.exitcode, worker_reaped=not process.is_alive(),
        wall_seconds=time.monotonic()-started, interrupted=interrupted,
        retry=False, next_phase_started=False, mandatory_STOP=True)
    c.save(root/'completion_witness.json', witness)
    return 0 if error is None else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--config');a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status='SCOPE_ONLY', scope=c.scope(), science_started=False)));return 0
    try:
        c.require(a.config,'APPROVED_CONFIG_REQUIRED');return execute(c.read(a.config))
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED', reason=str(exc),retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())

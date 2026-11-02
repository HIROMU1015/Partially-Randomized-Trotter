"""One-shot saved-phase diagnostic, scope-only by default. No compile path."""
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
import h6_m5_phase_contract as c


def available_memory():
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    row=next((s for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')),None)
    if row:
        p=Path('/sys/fs/cgroup')/row[3:].lstrip('/')
        while p.is_relative_to('/sys/fs/cgroup'):
            if (p/'memory.max').is_file() and (p/'memory.current').is_file():
                maximum=(p/'memory.max').read_text().strip()
                if maximum!='max':available=min(available,max(0,int(maximum)-int((p/'memory.current').read_text())))
            if p==Path('/sys/fs/cgroup'):break
            p=p.parent
    return available


def observe():
    return dict(effective_available_RAM_bytes=available_memory(),
        affinity=sorted(os.sched_getaffinity(0)),host_memory_PSI_observation_only=Path('/proc/pressure/memory').read_text())


def fresh_resources(cfg):
    affinity=os.sched_getaffinity(0)
    def cpu():
        return {int(p[0][3:]):list(map(int,p[1:])) for line in Path('/proc/stat').read_text().splitlines()
            if (p:=line.split()) and p[0].startswith('cpu') and p[0][3:].isdigit() and int(p[0][3:]) in affinity}
    before=cpu();time.sleep(1);after=cpu();idle=0.
    for i in affinity:
        d=[b-a for a,b in zip(before[i],after[i])];idle+=(d[3]+d[4])/max(1,sum(d))
    resources=cfg['resources'];memory=available_memory()
    workers=min(4,len(affinity),math.floor(idle),memory//resources['per_worker_RSS_proxy_bytes'])
    c.require(workers>=1,'FRESH_RESOURCES_CANNOT_ADMIT_ONE_REPLAY')
    disk=shutil.disk_usage(Path(cfg['run_root']).parent).free
    c.require(disk>=resources['disk_proxy_bytes'],'FRESH_DISK_CANNOT_ADMIT_DIAGNOSTICS')
    return dict(workers=workers,idle_CPU_equivalents=idle,observation=observe(),disk_free_bytes=disk,
        proxies_are_runtime_caps=False,historical_worker_cap=False,host_PSI_kill=False)


def child_entry(cfg, job, stop):
    from h6_m5_phase_adapter import worker
    try:
        worker(cfg,job,stop)
    except BaseException as exc:
        c.save(Path(cfg['output'])/(job+'_STOP.json'),dict(job=job,status='STOP',reason=str(exc),traceback=traceback.format_exc()))
        raise


def decisions(done, active, failed, capacity):
    if failed:return []
    return [j for j in c.JOBS if j not in set(done)|set(active)][:max(0,capacity)]


def supervise(cfg,config_path):
    """External wait4 witness, including launcher's native status and RSS.

    This function is stdlib-only until exec. It handles only its own child.
    A failed attempt remains recorded; it never starts a second launcher.
    """
    c.authorize(cfg)
    c.require(c.read(config_path)==cfg,'SUPERVISOR_CONFIG_IDENTITY')
    c.require(not Path(cfg['run_root']).exists(),'UNUSED_RUN_ROOT')
    paths=cfg['supervisor']
    c.require(all(not Path(p).exists() for p in paths.values()),'UNUSED_SUPERVISOR_OUTPUTS')
    c.save(paths['entered'],dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False))
    started=time.monotonic();handlers={};signals=[]
    log=Path(paths['log']).open('xb')
    pid=os.fork()
    if pid==0:
        try:
            os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
            os.execv(cfg['runtime']['python'],[cfg['runtime']['python'],'-B',str(Path(__file__).resolve()),'--execute','--config',str(Path(config_path).resolve())])
        except BaseException:
            os._exit(127)
    log.close()
    def forward(signum,frame):
        signals.append(signum)
        try:os.kill(pid,signum)
        except ProcessLookupError:pass
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,forward)
        while True:
            try:_,status,usage=os.wait4(pid,0);break
            except InterruptedError:continue
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    code=os.waitstatus_to_exitcode(status)
    c.save(paths['exit'],dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pid=pid,
        native_exit=code,wait4_status=status,reaped=True,ru_maxrss_KiB=usage.ru_maxrss,
        ru_utime=usage.ru_utime,ru_stime=usage.ru_stime,wall_seconds=time.monotonic()-started,
        forwarded_signals=signals,retry=False,next_phase_started=False,mandatory_STOP=True))
    return code if code>=0 else 1


def execute(cfg, *, worker=child_entry):
    # No science import, QPY/NPZ restore or directory creation before authorization.
    c.authorize(cfg)
    c.require(sys.executable==cfg['runtime']['python'],'FIXED_PYTHON_EXECUTABLE')
    root=Path(cfg['run_root']);c.require(not root.exists(),'UNUSED_RUN_ROOT')
    resources=fresh_resources(cfg)
    root.mkdir(exist_ok=False);Path(cfg['output']).mkdir();Path(cfg['ledger']).mkdir()
    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
    c.save(root/'fresh_gate.json',resources)
    ctx=mp.get_context('spawn');stop=ctx.Event();processes={};active={};results={};error=None
    start=time.monotonic();handlers={};signalled=[]
    names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_PARALLEL')
    oldenv={k:os.environ.get(k) for k in names}
    def interrupt(signum,frame):
        signalled.append(signum);stop.set()
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
        for key in names:os.environ[key]='FALSE' if key=='QISKIT_PARALLEL' else '1'
        from h6_m5_phase_adapter import saved_gate, circuit_gate, report
        from h6_m5_phase_engine import precision
        import numpy as np
        import qiskit
        c.require(qiskit.__version__=='1.3.0' and np.__version__=='1.26.4','FROZEN_RUNTIME_VERSIONS')
        sealed={row['path'] for row in cfg['sealed_files']} if 'sealed_files' in cfg else set()
        c.require(all(str(Path(module.__file__).resolve()) in sealed for module in (np,qiskit)), 'LOADED_RUNTIME_IDENTITY')
        c.save(root/'precision_gate.json',precision())
        c.save(root/'saved_metrics_gate.json',saved_gate(cfg))
        schemas={kind:circuit_gate(cfg,kind) for kind in ('WRAPPER','COMPILED')}
        c.save(root/'QPY_schema_gate.json',schemas)
        with (root/'resource_observation.jsonl').open('x') as journal:
            while True:
                if signalled:error=error or 'OWN_RUN_SIGNAL'
                for job,p in list(active.items()):
                    if p.is_alive():continue
                    p.join();del active[job]
                    try:
                        c.require(p.exitcode==0,'OWN_WORKER_EXIT:'+job)
                        r=c.read(Path(cfg['output'])/job/'result.json')
                        c.require(r['job']==job and r['SOURCE']==cfg['SOURCE'] and r['plan_sha256']==cfg['plan_sha256'],'WORKER_RESULT_IDENTITY')
                        if job.endswith('ORDINARY'):c.require(r['ordinary_reproduced'] is True,'ORDINARY_REPLAY_DID_NOT_REPRODUCE:'+job)
                        results[job]=r
                    except Exception as exc:error=error or str(exc)
                for job in decisions(results,active,error,resources['workers']-len(active)):
                    if available_memory()<cfg['resources']['per_worker_RSS_proxy_bytes']:
                        if not active:error=error or 'ACTUAL_RAM_CANNOT_ADMIT_NEXT_REPLAY'
                        break
                    c.reserve(cfg,job)
                    p=ctx.Process(target=worker,args=(cfg,job,stop))
                    p.start();processes[job]=p;active[job]=p
                    c.save(root/(job+'_start.json'),dict(job=job,pid=p.pid,launcher_pid=os.getpid()))
                if not active:break
                journal.write(json.dumps(dict(elapsed=time.monotonic()-start,completed=list(results),
                    active=list(active),error=error,resources=observe()))+'\n');journal.flush()
                for p in active.values():p.join(timeout=.5)
        c.require(error is None,error)
        c.require(not signalled,'OWN_RUN_SIGNAL')
        c.verify_seals(cfg);c.consumed(cfg)
        c.require({p.name for p in Path(cfg['ledger']).glob('*.json')}=={j+'.json' for j in c.JOBS},'EXACT_REPLAY_LEDGER')
        result=report(cfg,results)
        c.save(root/'diagnostic_report.json',result)
    except BaseException as exc:
        error=error or str(exc)
        # On unexpected own errors, request cooperative recovery. Never kill for PSI.
        stop.set()
        c.save(root/'STOP.json',dict(reason=error,status='STOP',traceback=traceback.format_exc()))
    finally:
        for p in processes.values():p.join()
        for sig,h in handlers.items():signal.signal(sig,h)
        for key,value in oldenv.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    try:c.verify_seals(cfg);c.consumed(cfg)
    except Exception as exc:error=error or str(exc)
    witnesses={job:dict(pid=p.pid,exit=p.exitcode,reaped=not p.is_alive()) for job,p in processes.items()}
    all_reaped=all(w['reaped'] for w in witnesses.values())
    if not all_reaped or any(w['exit']!=0 for w in witnesses.values()):error=error or 'NATIVE_EXIT_OR_REAP_FAILED'
    c.save(root/'completion_witness.json',dict(status='STOP' if error else 'DIAGNOSTIC_COMPLETE_MANDATORY_STOP',reason=error,
        SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],native_workers=witnesses,all_reaped=all_reaped,
        replay_entries=len(list(Path(cfg['ledger']).glob('*.json'))),result_jobs=list(results),
        wall_seconds=time.monotonic()-start,new_compile=0,new_sampling=0,accepted_new_cost_axes=0,
        signalled=signalled,retry=False,old_STOP_remains_STOP=True,next_phase_started=False,mandatory_STOP=True,
        report_sha256=c.sha(root/'diagnostic_report.json') if (root/'diagnostic_report.json').exists() else None))
    return 1 if error else 0


def main():
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true');modes.add_argument('--supervise',action='store_true');parser.add_argument('--config')
    args=parser.parse_args()
    if not args.execute and not args.supervise:
        print(json.dumps(dict(status='SCOPE_ONLY',scope=c.scope(),science_started=False)));return 0
    try:
        c.require(args.config,'APPROVED_CONFIG_REQUIRED')
        cfg=c.read(args.config)
        if args.supervise:return supervise(cfg,args.config)
        c.authorize(cfg)
        entered=c.read(cfg['supervisor']['entered'])
        c.require(entered==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),'EXTERNAL_SUPERVISOR_REQUIRED')
        return execute(cfg)
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())

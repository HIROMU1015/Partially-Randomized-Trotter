"""Signal-only supervisor/launcher. No valid grant means no scientific import."""
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
from types import FunctionType
import h46_m5_signal_contract as c
import run_h6_m5_saved_phase as observed

supervise=FunctionType(observed.supervise.__code__,dict(observed.__dict__,c=c,__file__=__file__),'supervise',observed.supervise.__defaults__,observed.supervise.__closure__)


def fresh_resources(cfg):
    affinity=os.sched_getaffinity(0)
    def cpu():
        return {int(s[0][3:]):list(map(int,s[1:])) for line in Path('/proc/stat').read_text().splitlines()
            if (s:=line.split()) and s[0].startswith('cpu') and s[0][3:].isdigit() and int(s[0][3:]) in affinity}
    before=cpu();time.sleep(1);after=cpu();idle=0.
    for i in before:
        d=[b-a for a,b in zip(before[i],after[i])];idle+=(d[3]+d[4])/max(1,sum(d))
    memory=observed.available_memory();proxy=max(cfg['resources']['system_RSS_proxy_bytes'].values())
    workers=min(2,len(cfg['systems']),len(affinity),math.floor(idle),memory//proxy)
    c.require(workers>=1,'SIGNAL_CANNOT_ADMIT_ONE_SYSTEM')
    disk=shutil.disk_usage(Path(cfg['run_root']).parent).free;c.require(disk>=cfg['resources']['disk_proxy_bytes'],'SIGNAL_DISK_ADMISSION')
    return dict(system_workers=workers,threads_per_system=max(1,math.floor(idle/workers)),idle_CPU_equivalents=idle,
      effective_available_RAM_bytes=memory,disk_free_bytes=disk,proxy_is_runtime_cap=False,host_PSI_kill=False)


def execute(cfg,*,target=None):
    c.authorize(cfg);c.require(sys.executable==cfg['runtime']['python'],'SIGNAL_FIXED_PYTHON')
    root=Path(cfg['run_root']);c.require(not root.exists(),'UNUSED_SIGNAL_RUN_ROOT');fresh=fresh_resources(cfg)
    root.mkdir();Path(cfg['output']).mkdir();Path(cfg['ledger']).mkdir()
    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']));c.save(root/'fresh_gate.json',fresh)
    context=mp.get_context('spawn');stops={s:context.Event() for s in cfg['systems']};active={};processes={};pending=list(cfg['systems']);signals=[];errors={};start=time.monotonic();handlers={}
    keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_PARALLEL');oldenv={k:os.environ.get(k) for k in keys}
    def interrupt(sig,frame):
        signals.append(sig)
        for e in stops.values():e.set()
    try:
        for k in keys:os.environ[k]='FALSE' if k=='QISKIT_PARALLEL' else '1'
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupt)
        if target is None:
            from h46_m5_signal_adapter import system_worker
            target=system_worker
        with (root/'resource_observation.jsonl').open('x') as journal:
            while pending or active:
                while pending and len(active)<fresh['system_workers'] and not signals:
                    s=pending.pop(0)
                    if observed.available_memory()<cfg['resources']['system_RSS_proxy_bytes'][s]:errors[s]='RAM_CANNOT_ADMIT_NEXT_SYSTEM';continue
                    threads=min(len(c.rows(s)),fresh['threads_per_system'])
                    p=context.Process(target=target,args=(cfg,s,threads,stops[s]));p.start();active[s]=p;processes[s]=p
                    c.save(root/(s+'_start.json'),dict(SOURCE=cfg['SOURCE'],system=s,pid=p.pid,threads=threads,launcher_pid=os.getpid()))
                for s,p in list(active.items()):
                    if p.is_alive():continue
                    p.join();active.pop(s)
                    try:
                        c.require(p.exitcode==0 and not (Path(cfg['output'])/(s+'_STOP.json')).exists(),'SIGNAL_SYSTEM_STOP_OR_NATIVE_EXIT')
                        r=c.read(Path(cfg['output'])/(s+'_complete.json'))
                        c.require(r['status']=='SIGNAL_SYSTEM_COMPLETE_MANDATORY_STOP' and r['SOURCE']==cfg['SOURCE'] and r['plan_sha256']==cfg['plan_sha256'] and r['system']==s,'SIGNAL_SYSTEM_REPORT_IDENTITY')
                        expected={x['id'] for x in c.rows(s)}
                        c.require(len(r['results'])==len(expected) and {x['id'] for x in r['results']}==expected,'SIGNAL_SYSTEM_COVERAGE')
                        c.require(r['new_signal']==len(expected)-len(cfg['systems'][s]['reuse']) and r['reused_signal']==len(cfg['systems'][s]['reuse']),'SIGNAL_NEW_REUSE_COUNTS')
                        for row in r['results']:
                            path=Path(cfg['output'])/row['file'];c.require(path.name==row['id']+'.json' and c.sha(path)==row['sha256'],'SIGNAL_RESULT_SHA')
                            data=c.read(path);candidate=next(x for x in c.rows(s) if x['id']==row['id'])
                            c.require(data['status']=='SIGNAL_DIAGNOSTIC_PASS' and data['candidate']==candidate and data['SOURCE']==cfg['SOURCE'] and data['plan_sha256']==cfg['plan_sha256'],'SIGNAL_CANDIDATE_IDENTITY')
                            c.require(data['new_compile']==data['new_sampling']==0 and not data['cost_accounting'] and not data['ranking'],'SIGNAL_NO_COST_OR_SAMPLING')
                            c.require(len(data['empirical_eligibility']['rows'])==10 and all(x['shots'] is None and x['cost'] is None for x in data['empirical_eligibility']['rows']),'SIGNAL_ELIGIBILITY_ONLY')
                    except Exception as exc:errors[s]=str(exc)
                if signals:
                    for s in pending:errors[s]='OWN_SIGNAL_BEFORE_DISPATCH'
                    pending=[]
                if active:
                    journal.write(json.dumps(dict(elapsed=time.monotonic()-start,active=list(active),errors=errors,resources=observed.observe()))+'\n');journal.flush()
                    for p in active.values():p.join(timeout=.5)
    except BaseException as exc:
        errors['launcher']=str(exc)
        for e in stops.values():e.set()
    finally:
        for p in processes.values():p.join()
        for sig,h in handlers.items():signal.signal(sig,h)
        for k,v in oldenv.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    try:
        c.verify(cfg);c.consumed(cfg)
        if not errors:c.require({p.stem for p in Path(cfg['ledger']).glob('*.json')}=={r['id'] for r in c.rows()},'EXACT_88_SIGNAL_ENTRIES')
    except Exception as exc:errors['end_seal']=str(exc)
    workers={s:dict(pid=p.pid,exit=p.exitcode,reaped=not p.is_alive()) for s,p in processes.items()}
    if signals:errors['signals']=signals
    if set(workers)!=set(cfg['systems']):errors['missing_worker']='INCOMPLETE_SYSTEM_DISPATCH'
    if any(w['exit']!=0 or not w['reaped'] for w in workers.values()):errors['workers']='EXIT_OR_REAP'
    c.save(root/'completion_witness.json',dict(status='STOP' if errors else 'SIGNAL_COMPLETE_MANDATORY_STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
       errors=errors,workers=workers,wall_seconds=time.monotonic()-start,new_compile=0,new_sampling=0,cost_accounting=False,ranking=False,retry=False,next_phase_started=False,mandatory_STOP=True))
    return 1 if errors else 0


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--execute',action='store_true');g.add_argument('--supervise',action='store_true');p.add_argument('--config');a=p.parse_args()
    if not a.execute and not a.supervise:print(json.dumps(dict(status='SCOPE_ONLY',scope=c.scope(),science_started=False)));return 0
    try:
        c.require(a.config,'SIGNAL_CONFIG_REQUIRED');cfg=c.read(a.config);c.authorize(cfg)
        if a.supervise:return supervise(cfg,a.config)
        c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),'EXTERNAL_SIGNAL_SUPERVISOR_REQUIRED')
        return execute(cfg)
    except Exception as exc:print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2


if __name__=='__main__':raise SystemExit(main())

"""Plan-only default; a separately approved cost-only one-shot ends with STOP."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import h8_noncancellation_cost_contract as r
from run_h8_phase_c import OwnedChild, observe, protected_job_cpus, save
c=r.c


def effective_memory(state):
    available=state['memory']['MemAvailable']
    rows=Path('/proc/self/cgroup').read_text().splitlines()
    name=next(line.split(':',2)[2] for line in rows if line.startswith('0::'))
    mount=Path('/sys/fs/cgroup');current=mount/name.lstrip('/')
    while current.is_relative_to(mount):
        maximum,current_bytes=current/'memory.max',current/'memory.current'
        if maximum.exists() and current_bytes.exists():
            limit=maximum.read_text().strip()
            if limit!='max':available=min(available,max(0,int(limit)-int(current_bytes.read_text())))
        if current==mount:break
        current=current.parent
    return available


def fresh(config,grant):
    cpus=r.authorize(config,grant);r.verify(config)
    c.require(not set(cpus)&protected_job_cpus(),'OWN_CPUS_OVERLAP_RUNNING_B2')
    state=observe();available=effective_memory(state)
    # Admission estimates, never process RSS/AS or wall caps. PSI is observational.
    c.require(available>=r.RESOURCE_POLICY['estimates']['peak_RSS_bytes'],'INSUFFICIENT_EFFECTIVE_MEMORY')
    parent=Path(config['output']).parent;fs=os.statvfs(parent)
    c.require(fs.f_bavail*fs.f_frsize>=r.RESOURCE_POLICY['estimates']['output_bytes'] and
              fs.f_favail>=r.RESOURCE_POLICY['estimates']['output_inodes'],'OUTPUT_BLOCKS_INODES')
    mounts=[line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    mount=max((x for x in mounts if parent.is_relative_to(x[1])),key=lambda x:len(x[1]))
    c.require(mount[2] in ('ext4','xfs') and not any(('quota' in x and not x.startswith('no')) or
              x in ('uquota','gquota','pquota') for x in mount[3].split(',')),'QUOTA_UNRESOLVED')
    return dict(SOURCE=config['SOURCE'],config_sha256=c.digest(config),grant_sha256=c.digest(grant),
        allowed_cpus=cpus,monotonic_seconds=time.monotonic(),launcher_identity=r.identity(os.getpid()),
        observation=state,effective_memory_available_bytes=available,
        source_input_verified=True,CPU_selection_verified=True,resources_admission_verified=True,
        quota_verified=True,protected_jobs_unchanged=True,launcher_wait_ready=True)


def driver(config_path,grant_path,fresh_path):
    config,grant,gate=map(r.read,(config_path,grant_path,fresh_path))
    cpus=r.authorize(config,grant);r.fresh_binding(config,grant,gate)
    c.require(r.identity(os.getppid())==gate['launcher_identity'],'OWN_LIVE_LAUNCHER_REQUIRED')
    c.require(Path(sys.executable).resolve()==Path(config['python']).resolve() and
              all(os.environ.get(k)=='1' for k in r.THREAD_KEYS),'PINNED_PYTHON_THREADS')
    r.verify(config,git=False,protection=False)
    output,control,lock=r.path_set(config,supervisor_control_exists=True)
    with lock.open('x') as stream:
        stream.write(c.digest(grant));stream.flush();os.fsync(stream.fileno())
    output.mkdir(mode=0o700);calls={k:0 for k in r.CAPS};interrupted=[]
    def guard():c.require(not interrupted,'OWN_PROCESS_SIGNAL:'+str(interrupted[:1]))
    def take(key):
        guard();c.require(calls[key]<r.CAPS[key],'COST_CALL_CAP:'+key);calls[key]+=1
    def stop(signum,frame):
        interrupted.append(signum);raise RuntimeError('OWN_PROCESS_SIGNAL:'+str(signum))
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for s in handlers:signal.signal(s,stop)
    try:
        data,registry,permit=r.load_saved(config)
        with (output/'noncancellation_cost_rows.jsonl').open('x') as rows:
            def emit(kind,row):
                if kind=='accounting_row':rows.write(json.dumps(row,allow_nan=False)+'\n')
            result=c.evaluate(data,registry,permit,take=take,emit=emit,guard=guard)
        c.require((calls['reaccounting_pass'],calls['saved_row'],calls['comparison'])==(1,1900,20),'EXACT_COST_CALL_COUNTS')
        guard();r.verify(config,git=False,protection=False)
        result.update(SOURCE=config['SOURCE'],plan_sha256=config['plan_sha256'],grant_sha256=c.digest(grant))
        save(output/'noncancellation_cost_comparisons.json',result)
        save(output/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=config['SOURCE'],actual_calls=calls,
             stages_executed=['COST_COMPARISON'],new_shots=0,usual_Phase_C_recalculated=0,
             automatic_retry=False,next_stage_authorized=False))
    except BaseException as error:
        save(output/'terminal.json',dict(status='STOP',SOURCE=config['SOURCE'],first_stop=str(error),
             actual_calls=calls,automatic_retry=False,next_stage_authorized=False));raise
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)


def launch(config_path,grant_path):
    config,grant=map(r.read,(config_path,grant_path))
    cpus=r.authorize(config,grant) # Invalid grants rejected before input reads, locks, children or affinity.
    r.path_set(config);before_protection=r.verify(config)
    parent=Path(config['output']).parent;marker=parent/'NC_COST_LAUNCH_ATTEMPT'
    save(marker,dict(scope=r.SCOPE,SOURCE=config['SOURCE'],automatic_retry=False))
    owned=None;first_stop=None;pending=[];cleanup_errors=[];control=Path(config['control']);handlers={}
    before=None;started=time.monotonic()
    def stop(signum,frame):
        nonlocal first_stop
        pending.append(signum)
        if first_stop is None:first_stop='OWN_LAUNCHER_SIGNAL:'+str(signum)
    def guard():c.require(not pending,first_stop or 'OWN_LAUNCHER_SIGNAL')
    def native_receipt():
        save(control/'native_exit.json',dict(owned.native,wall_seconds=time.monotonic()-started,
             before=before,after=observe(),first_stop=first_stop,cleanup_errors=cleanup_errors,
             all_children_reaped=owned.native is not None,PSI_is_STOP_gate=False,automatic_retry=False))
    try:
        for s in (signal.SIGINT,signal.SIGTERM):handlers[s]=signal.signal(s,stop)
        guard();gate=fresh(config,grant);save(parent/'NC_COST_fresh_gate.json',gate)
        control.mkdir(mode=0o700);save(control/'other_work_observation_before.json',before_protection)
        env=dict(os.environ,TMPDIR=str(control),PYTHONDONTWRITEBYTECODE='1',**{k:'1' for k in r.THREAD_KEYS})
        cmd=[config['python'],'-P','-B',str(Path(__file__).resolve()),'--driver-approved',
             '--config',str(config_path),'--grant',str(grant_path),'--fresh-gate',str(parent/'NC_COST_fresh_gate.json')]
        before=observe();guard()
        with (control/'stdout.log').open('xb') as out,(control/'stderr.log').open('xb') as err:
            child=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,preexec_fn=lambda:os.sched_setaffinity(0,set(cpus)))
            owned=OwnedChild(child)
            while owned.wait() is None:guard();time.sleep(.05)
        guard();c.require(owned.native['native_exit_code']==0,'OWN_DRIVER_NATIVE_EXIT:'+str(owned.native['native_exit_code']))
        save(control/'other_work_observation_after.json',r.verify(config));guard()
        c.require(r.read(Path(config['output'])/'terminal.json')['status']=='COMPLETE_MANDATORY_STOP','DRIVER_NOT_COMPLETE')
        native_receipt()
        save(parent/'NC_COST_final_gate.json',dict(status='PASS_MANDATORY_STOP',SOURCE=config['SOURCE'],next_stage_authorized=False))
    except BaseException as error:
        if first_stop is None:first_stop=str(error) or type(error).__name__
        if owned is not None:
            cleanup_errors.extend(owned.cleanup())
            if owned.native is not None and not (control/'native_exit.json').exists():
                try:native_receipt()
                except BaseException as receipt_error:cleanup_errors.append('NATIVE_RECEIPT:'+str(receipt_error))
        save(parent/'NC_COST_STOP.json',dict(status='STOP',first_stop=first_stop,cleanup_errors=cleanup_errors,
             SOURCE=config['SOURCE'],all_children_reaped=owned is None or owned.native is not None,
             automatic_retry=False,next_stage_authorized=False));raise
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--launch-approved',action='store_true');mode.add_argument('--driver-approved',action='store_true')
    for key in ('config','grant','fresh-gate'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if not a.launch_approved and not a.driver_approved:
        if a.config or a.grant or a.fresh_gate:p.error('separate approved execution mode required')
        print(json.dumps(r.plan(),indent=2));return
    if not a.config or not a.grant:p.error('sealed config and new grant required')
    if a.launch_approved:launch(a.config,a.grant)
    else:
        if not a.fresh_gate:p.error('fresh gate required')
        driver(a.config,a.grant,a.fresh_gate)


if __name__=='__main__':main()

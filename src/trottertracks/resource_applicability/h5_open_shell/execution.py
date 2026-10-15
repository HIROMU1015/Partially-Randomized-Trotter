"""One-shot H5 coordinator. Import-time code is metadata/stdlib only.

Children alone do scientific work; the coordinator observes them independently
of their GIL. Forked cost groups inherit the same immutable preparations.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

from .contract import Calls, call_limits, digest, tasks, verify_grant


def write_json(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+'\n').encode()
    if len(raw)>32*2**20:raise RuntimeError('RECORD_BYTES')
    with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def verify_assets(root,manifest):
    import subprocess
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip()!=manifest['SOURCE']:
        raise ValueError('SOURCE_CHECKOUT')
    for name,expected in manifest['source_hashes'].items():
        p=Path(root)/name
        if file_hash(p)!=expected:raise ValueError('SOURCE_BYTES:'+name)
        raw=subprocess.check_output(['git','show',manifest['SOURCE']+':'+name],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('SOURCE_COMMIT_BYTES:'+name)
    profile=manifest['profile']
    if digest(profile)!=manifest['profile_digest']:raise ValueError('PROFILE_DIGEST')
    for name,expected in profile['files'].items():
        if file_hash(name)!=expected:raise ValueError('ENVIRONMENT_BYTES:'+name)


class Child:
    def __init__(self,pid,identity,pidfd,directory,task):
        self.pid,self.identity,self.pidfd,self.directory,self.task=pid,identity,pidfd,Path(directory),task
        self.reaped=False;self.exit_code=None
    def exited(self):
        return os.waitid(os.P_PID,self.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is not None
    def reap(self):
        if self.reaped:raise ValueError('DOUBLE_REAP')
        _,status=os.waitpid(self.pid,0)
        self.exit_code=os.waitstatus_to_exitcode(status);self.reaped=True
        self.close()
        return self.exit_code
    def close(self):
        if self.pidfd is not None:os.close(self.pidfd);self.pidfd=None
    def terminate(self):
        if self.reaped:return
        if self.pidfd is None:raise ValueError('MISSING_OWN_PIDFD')
        try:signal.pidfd_send_signal(self.pidfd,signal.SIGKILL)
        except OSError as e:
            if e.errno!=errno.ESRCH:raise


def cleanup(children):
    """Signal every pinned identity, then reap all; retain non-ESRCH errors."""
    errors=[]
    for child in children:
        try:child.terminate()
        except BaseException as e:errors.append(dict(pid=child.pid,operation='terminate',error=repr(e)))
    deadline=time.monotonic()+10
    for child in children:
        try:
            while not child.reaped and not child.exited():
                if time.monotonic()>deadline:raise TimeoutError('OWN_WAIT_REAP')
                time.sleep(.02)
            if not child.reaped:child.reap()
        except BaseException as e:errors.append(dict(pid=child.pid,operation='reap',error=repr(e)))
        finally:child.close()
    return errors


def spawn(directory,task,function,cpu,limits):
    from ..h6_server_resources_v1 import process_identity,install_parent_guard
    directory=Path(directory);directory.mkdir(exist_ok=False)
    parent=process_identity(os.getpid());pid=os.fork()
    if pid==0:
        code=1;result=None
        try:
            os.setsid();install_parent_guard(parent)
            for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_DFL)
            fd=os.open(directory/'native.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
            os.sched_setaffinity(0,{cpu})
            resource.setrlimit(resource.RLIMIT_AS,(limits['AS_bytes'],limits['AS_bytes']))
            resource.setrlimit(resource.RLIMIT_FSIZE,(32*2**20,32*2**20))
            result=function(directory,task)
            write_json(directory/'result.json',result)
            write_json(directory/'terminal.json',dict(status='COMPLETE',task=task,identity=process_identity(os.getpid())))
            code=0
        except BaseException as e:
            traceback.print_exc()
            try:write_json(directory/'terminal.json',dict(status='STOP',task=task,error=repr(e)))
            except BaseException:pass
        finally:
            sys.stdout.flush();sys.stderr.flush();os._exit(code)
    pidfd=None
    try:
        identity=process_identity(pid);pidfd=os.pidfd_open(pid)
        return Child(pid,identity,pidfd,directory,task)
    except BaseException:
        # The unreaped direct child pins the PID; never signal an arbitrary PID.
        os.kill(pid,signal.SIGKILL);os.waitpid(pid,0)
        if pidfd is not None:os.close(pidfd)
        raise


def supervise(output,registered,function,cpus,limits,baseline,*,spawn_fn=spawn,observe=None):
    """Fixed registration only; completion or first STOP, never retry/resume."""
    from ..h6_server_resources_v1 import host_memory,process_memory,memory_reason,process_identity
    pending=list(enumerate(registered));active=[];completed=[];retired=[];first=None;errors=[]
    previous=time.monotonic()
    policy=dict(fresh_host_available_bytes=0,fresh_effective_available_bytes=0,
        running_host_available_bytes=limits['headroom_bytes'],running_effective_available_bytes=limits['headroom_bytes'],
        host_full_psi_stop_percent=1.,cgroup_full_psi_stop_percent=0.)
    observe=observe or host_memory
    log=Path(output)/'resource_observations.jsonl'
    try:
        with log.open('xb') as journal:
            while pending or active:
                start=time.monotonic();interval=start-previous;previous=start
                if interval>5:raise RuntimeError('OBSERVATION_STALE')
                memory=observe();reason=memory_reason(memory,baseline,policy)
                if reason:raise RuntimeError(reason)
                workers=[];aggregate=0
                for child in list(active):
                    if child.exited():
                        code=child.reap();active.remove(child);retired.append(dict(pid=child.pid,identity=child.identity,exit_code=code,reaped=True))
                        terminal=json.loads((child.directory/'terminal.json').read_bytes())
                        if code or terminal['status']!='COMPLETE':raise RuntimeError('WORKER_STOP:'+str(child.pid))
                        result=json.loads((child.directory/'result.json').read_bytes())
                        if result['task']!=child.task:raise RuntimeError('WORKER_TASK_BINDING')
                        completed.append(result);continue
                    current=process_identity(child.pid)
                    if current['start_ticks']!=child.identity['start_ticks']:raise RuntimeError('OWN_IDENTITY_CHANGED')
                    try:usage=process_memory(child.pid)
                    except (FileNotFoundError,ValueError):
                        if child.exited():continue
                        raise
                    aggregate+=usage['VmRSS_bytes'];workers.append(dict(identity=child.identity,memory=usage))
                    if usage['VmRSS_bytes']>limits['worker_RSS_bytes'] or usage['VmSize_bytes']>limits['AS_bytes']:
                        raise RuntimeError('WORKER_MEMORY_LIMIT')
                    try:
                        status=Path(f'/proc/{child.pid}/status').read_text().splitlines()
                        threads=int(next(v.split(':')[1] for v in status if v.startswith('Threads:')))
                        if threads>1:raise RuntimeError('INTERNAL_THREAD_LIMIT')
                    except FileNotFoundError:
                        if not child.exited():raise
                if aggregate>limits['aggregate_RSS_bytes']:raise RuntimeError('AGGREGATE_RSS_LIMIT')
                stat=os.statvfs(output)
                if stat.f_bavail*stat.f_frsize<limits['minimum_free_bytes'] or stat.f_favail<1000:raise RuntimeError('FILESYSTEM_HEADROOM')
                size=sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
                if size>limits['output_bytes']:raise RuntimeError('OUTPUT_LIMIT')
                duration=time.monotonic()-start
                if duration>5:raise RuntimeError('OBSERVATION_DURATION')
                record=dict(monotonic=start,interval_seconds=interval,observation_seconds=duration,
                    memory=memory,workers=workers,aggregate_RSS_bytes=aggregate,first_stop=None)
                raw=(json.dumps(record,allow_nan=False)+'\n').encode()
                if journal.tell()+len(raw)>64*2**20:raise RuntimeError('MONITOR_OUTPUT_CAP')
                journal.write(raw);journal.flush()
                occupied={child.task['_cpu'] for child in active}
                for cpu in cpus:
                    if cpu in occupied or not pending:continue
                    index,task=pending.pop(0);task=dict(task,_cpu=cpu)
                    child=spawn_fn(Path(output)/f'group_{index:03d}',task,function,cpu,limits);active.append(child)
                if active:time.sleep(max(.01,1-(time.monotonic()-start)))
    except BaseException as e:first=repr(e)
    finally:
        errors=cleanup(active)
        retired.extend(dict(pid=c.pid,identity=c.identity,exit_code=c.exit_code,reaped=c.reaped) for c in active)
        summary=dict(completed=completed,registered_count=len(registered),pending=len(pending),
            retired=retired,first_stop=first,cleanup_errors=errors,mandatory_stop=first is not None)
        write_json(Path(output)/'summary.json',summary)
    if first or errors:raise RuntimeError('H5_FAIL_CLOSED:'+str(first or errors))
    if len(completed)!=len(registered):raise RuntimeError('MISSING_COMPLETIONS')
    return summary


def input_worker(directory,task):
    from .input import rohf_integrals,weighted_df,array_identity
    from .science import solve_state,preparations,panel_signals,validate_primitives
    import numpy as np
    calls=Calls(call_limits(100))  # Admission cap, never a rank selection.
    def record(key,used):
        if used[key]==1 or used[key]%100==0:
            path=directory/'call_reservations.jsonl'
            if path.exists() and path.stat().st_size>4*2**20:raise RuntimeError('CALL_JOURNAL_CAP')
            with path.open('a') as f:f.write(json.dumps(dict(before_operation=key,used=used))+'\n');f.flush()
    calls.recorder=record
    scratch=directory/'scratch';scratch.mkdir()
    arrays,receipt=rohf_integrals(scratch,calls)
    with (directory/'integrals.npz').open('xb') as f:np.savez(f,**arrays)
    write_json(directory/'integral_receipt.json',receipt)
    ham,projection,raw=weighted_df(float(arrays['constant']),arrays['one_body'],arrays['two_body'],calls)
    with (directory/'raw_df.npz').open('xb') as f:np.savez(f,**raw)
    write_json(directory/'projection_receipt.json',projection)
    actual=call_limits(len(ham.lambdas))
    if any(calls.used[k]>actual[k] for k in calls.used):raise RuntimeError('ACTUAL_RANK_CALL_PLAN')
    calls.caps=actual
    sector,state,full,checks,matrix=solve_state(ham,calls)
    prepared=preparations(ham)
    primitives=validate_primitives(ham,sector,full,state,prepared,calls)
    checks['primitive_max_error']=primitives['max_error']
    records=panel_signals(ham,sector,state,matrix,checks,prepared,calls)
    # No hidden compressed cache or pickle; preserve the raw generation and state.
    snapshot=dict(constant=np.array(ham.constant),one_body=ham.one_body,lambdas=ham.lambdas,
        g_matrices=np.array(ham.g_matrices),sector_basis=sector.basis_indices,state=state,full=full)
    with (directory/'snapshot.npz').open('xb') as f:np.savez(f,**snapshot)
    checks=dict(checks,reference=dict(real=checks['reference'].real,imag=checks['reference'].imag))
    result=dict(task=task,input=receipt,DF=projection,state_checks=checks,primitive_checks=primitives,
        signal_records=records,calls_attempted=calls.used,call_caps=actual,
        snapshot=dict(path=str(directory/'snapshot.npz'),sha256=file_hash(directory/'snapshot.npz'),
            arrays={k:array_identity(v) for k,v in snapshot.items()}))
    return result


def execute(root,manifest,grant):
    """Only explicit fresh H5 authorization reaches this function."""
    from ..h6_server_resources_v1 import host_memory
    output=verify_grant(manifest,grant,manifest['SOURCE']);verify_assets(root,manifest)
    cpus=manifest['worker_cpus'];limits=manifest['limits']
    if not 1<=len(cpus)<=4 or len(set(cpus))!=len(cpus) or not set(cpus)<=os.sched_getaffinity(0):raise ValueError('FRESH_CPUS')
    if manifest['coordinator_cpu'] in cpus or manifest['coordinator_cpu'] not in os.sched_getaffinity(0):raise ValueError('COORDINATOR_CPU')
    if resource.getrlimit(resource.RLIMIT_AS)[1] not in (-1,resource.RLIM_INFINITY) and resource.getrlimit(resource.RLIMIT_AS)[1]<limits['AS_bytes']:
        raise ValueError('PARENT_HARD_AS_INHERITANCE')
    if output.exists() or output.is_symlink():raise ValueError('OUTPUT_OR_ONE_SHOT_USED')
    from ..h6_server_quota_v1 import quota_readonly
    fs=os.statvfs(output.parent)
    if fs.f_bavail*fs.f_frsize<limits['minimum_free_bytes'] or fs.f_favail<1000:raise ValueError('FRESH_FILESYSTEM')
    quota=quota_readonly(output.parent)
    if quota['status']!='KNOWN':raise ValueError('FRESH_QUOTA_UNKNOWN')
    for item in quota['items']:
        if item['status']=='ACTIVE':
            if item['available_bytes'] is not None and item['available_bytes']<limits['output_bytes']:raise ValueError('FRESH_QUOTA_BYTES')
            if item['available_inodes'] is not None and item['available_inodes']<1000:raise ValueError('FRESH_QUOTA_INODES')
    def cpu_ticks():
        return {int(line.split()[0][3:]):[int(v) for v in line.split()[1:9]]
            for line in Path('/proc/stat').read_text().splitlines() if line.split()[0].startswith('cpu') and line.split()[0][3:].isdigit()}
    def physical(cpu):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        return ((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
    assigned=cpus+[manifest['coordinator_cpu']]
    if len({physical(v) for v in assigned})!=len(assigned):raise ValueError('CPU_PHYSICAL_SIBLING')
    before=cpu_ticks();time.sleep(1);after=cpu_ticks()
    for cpu in assigned:
        delta=[b-a for a,b in zip(before[cpu],after[cpu])];total=sum(delta)
        if total<=0 or 1-(delta[3]+delta[4])/total>.1:raise ValueError('FRESH_CPU_BUSY')
    baseline=host_memory()
    if baseline['effective_available_bytes']<limits['aggregate_RSS_bytes']+limits['headroom_bytes']:raise ValueError('FRESH_MEMORY')
    if any(v>=(1. if k=='host' else 1e-300) for k,v in baseline['full_psi_avg10'].items()):raise ValueError('FRESH_PRESSURE')
    output.mkdir(exist_ok=False)  # Exclusive one-shot claim before any science.
    os.sched_setaffinity(0,{manifest['coordinator_cpu']})
    write_json(output/'authorization.json',grant);write_json(output/'manifest.json',manifest)
    terminal=dict(status='STOP',first_stop=None,mandatory_stop=True,SOURCE=manifest['SOURCE'],
        coordinator_identity=None,all_workers_reaped=False)
    try:
        initial=output/'input';initial.mkdir()
        demand=call_limits(100)
        write_json(output/'input_reservation.json',dict(reserved=demand,actual=None,refunded=False))
        summary=supervise(initial,[dict(stage='input',SOURCE=manifest['SOURCE'])],input_worker,cpus[:1],limits,baseline)
        saved=summary['completed'][0];snapshot=saved['snapshot'];rank=saved['DF']['actual_rank']
        if file_hash(snapshot['path'])!=snapshot['sha256']:raise ValueError('SNAPSHOT_BYTES')
        from .science import preparations,cost_group
        from .input import array_identity
        from .sector import sector_for
        from trotterlib.df_hamiltonian import DFHamiltonian
        import numpy as np
        with np.load(snapshot['path'],allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        if {k:array_identity(v) for k,v in data.items()}!=snapshot['arrays']:raise ValueError('SNAPSHOT_ARRAY_IDENTITY')
        ham=DFHamiltonian(float(data['constant']),data['one_body'],data['lambdas'],tuple(data['g_matrices']),{})
        sector=sector_for(5,3,2)
        if not np.array_equal(data['sector_basis'],sector.basis_indices):raise ValueError('SNAPSHOT_SECTOR')
        prepared=preparations(ham)  # Once in idle coordinator; forked read-only reuse.
        binding=digest(dict(snapshot=snapshot['sha256'],projection=saved['DF'],source=manifest['SOURCE']))
        from .contract import registry
        write_json(output/'realized_input_binding.json',dict(SOURCE=manifest['SOURCE'],
            snapshot=snapshot,projection_digest=digest(saved['DF']),actual_rank=rank,
            registry=registry(rank),call_caps=call_limits(rank),input_digest=binding,
            compiler_hash=manifest['compiler_hash'],profile_digest=manifest['profile_digest'],
            carry_digest=manifest['carry_digest'],authorization_digest=digest(grant),output=str(output),
            initial_tasks=tasks(rank,manifest['SOURCE'],binding),technical_input_checks='PASS_ENGINEERING',
            scientific_certificate=False))
        def cost(directory,task):
            caps=dict.fromkeys(call_limits(rank),0);caps.update(compile=2,trajectory=1 if task['cell']['R'] else 0,
                sampling_call=1 if task['cell']['R'] else 0,occurrence=task['cell']['R'] or 0,control_probe=4 if task['validate'] else 0)
            expected=next(r['signals']['corrected'] for r in saved['signal_records'] if r['cell']['id']==task['cell']['id'])
            return cost_group(ham,sector,data['full'],data['state'],prepared,task,Calls(caps),expected_signal=expected)
        accepted=[]
        for stage,registered in [('initial',tasks(rank,manifest['SOURCE'],binding))]:
            # The first six groups finish correctness before extension selection.
            heavy=[t for t in registered if t['heavy']];normal=[t for t in registered if not t['heavy']]
            for label,group,assigned in ((stage+'_heavy',heavy,cpus[:1]),(stage+'_normal',normal,cpus)):
                directory=output/label;directory.mkdir();write_json(directory/'reservations.json',dict(tasks=group,compile=2*len(group),refund=False))
                accepted+=supervise(directory,group,cost,assigned,limits,baseline)['completed']
        eligible=[r['cell']['id'] for r in saved['signal_records'] if r['cell']['R'] and
            any(v['1.0']['cost_acquisition_eligible'] for v in r['precision_rows'].values())]
        group=tasks(rank,manifest['SOURCE'],binding,extension=True,eligible=eligible)
        directory=output/'extension';directory.mkdir();write_json(directory/'reservations.json',dict(tasks=group,compile=2*len(group),refund=False))
        accepted+=supervise(directory,group,cost,cpus,limits,baseline)['completed']
        used=dict(saved['calls_attempted'])
        for result in accepted:
            if len(result['rows'])!=2 or {r['axis'] for r in result['rows']}!={'cosine','sine'}:raise ValueError('PAIRED_METRICS')
            for k,v in result['calls_attempted'].items():used[k]+=v
        if any(used[k]>call_limits(rank)[k] for k in used):raise ValueError('FINAL_CALL_ACCOUNTING')
        write_json(output/'call_totals.json',dict(new_actual=used,old_carry_digest=manifest['carry_digest'],
            reset=False,refund=False,reservations_are_not_actual=True))
        verify_assets(root,manifest)
        terminal.update(status='COMPLETE_MANDATORY_STOP',groups=len(accepted),compiled=2*len(accepted),
            skipped_extension_PR=[r['cell']['id'] for r in saved['signal_records'] if r['cell']['R'] and r['cell']['id'] not in eligible])
    except BaseException as e:terminal['first_stop']=repr(e);raise
    finally:
        summaries=[json.loads(p.read_bytes()) for p in sorted(output.glob('*/summary.json'))]
        terminal['cleanup_errors']=[e for s in summaries for e in s['cleanup_errors']]
        terminal['retired']=[r for s in summaries for r in s['retired']]
        terminal['all_workers_reaped']=bool(summaries) and all(r['reaped'] for r in terminal['retired']) and not terminal['cleanup_errors']
        from ..h6_server_resources_v1 import process_identity
        terminal['coordinator_identity']=process_identity(os.getpid())
        write_json(output/'terminal.json',terminal)

"""Future closed full-map runner; science lowering is unchanged from pilot6."""
import hashlib
import json
import os
import pickle
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from ..h4_geometry.identity import Stop, require
from ..h4_geometry.prelaunch_audit import private_path, streaming_sha, host_readonly
from ..h4_b0i.adapter import evolution
from ..h4_b0i.runtime import prepare_kept_only
from .contract import verify_reuse_match, NAMESPACE, CAPS
from .contract import authorize, validate_plan, fresh_gate, digest, claim_lock, CallLedger, FirstStop, cleanup, bound_fingerprint
WORKER_STATE={'phase':'bootstrap','task_id':None}


def load_grant(repo,*paths):
    values=[]
    for path in paths:
        path=private_path(Path(path).absolute())
        require(path.is_relative_to(repo/'artifacts/resource_applicability/track_a_h4_b0i_map_preparation'),'new local grant files only')
        require(path.stat().st_uid==os.getuid() and path.stat().st_size<=4*2**20,'owned bounded grant JSON')
        values.append(json.loads(path.read_text()))
    require(len(values)==3,'plan/auth/review paths')
    return values


def reuse_gate(repo,plan):
    """Fixed old public proof and unchanged old source closure; no old runtime/cache load."""
    proof=plan['pilot6_reuse_proof']
    old=json.loads((repo/proof['plan_path']).read_text())
    compiled=json.loads((repo/proof['metrics_path']).read_text())
    native=json.loads((repo/proof['native_receipt_path']).read_text())
    terminal=native['terminal']
    require(terminal['status']=='PILOT6_COMPLETE_STOP' and terminal['cleanup_complete'] and
        terminal['records']==6 and terminal['native_worker_exit_code']==native['driver_native_exit_code']==0 and
        terminal['native_observer_exit_code']==-15 and native['first_failure'] is None and
        all(not r['same_run_identity_present'] for r in native['run_residual_audit']),'pilot native completion/cleanup required')
    require(old['compiler_options']==plan['compiler_options'],'reuse compiler options')
    for item in old['source_closure']:
        require(item in plan['source_closure'],'old pilot source closure changed')
    from ..h4_b0i.contract import digest as old_digest
    require(compiled['plan_digest']==old_digest(old),'fixed pilot plan digest')
    prior={v['task_id']:v for v in compiled['results']};old_tasks={v['task_id']:v for v in old['tasks']};accepted={}
    for task in plan['tasks']:
        if task['action']=='VERIFY_REUSE':
            old_id=task['reuse_receipt']['old_task_id'];receipt=prior[old_id]
            require(receipt['plan_digest']==old_digest(old),'pilot receipt plan binding')
            verify_reuse_match(task,old_tasks[old_id],receipt);accepted[task['task_id']]=receipt
    require(len(accepted)==6,'six uniquely mapped complete receipts')
    return accepted


def source_gate(repo,plan):
    require(Path(__file__).absolute().parents[4]==repo,'loaded source root differs')
    for item in plan['source_closure']:
        data=(repo/item['path']).read_bytes()
        require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'source closure drift: '+item['path'])
    for item in plan['bound_public_files']:
        data=(repo/item['path']).read_bytes()
        require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'profile/saved input drift: '+item['path'])


def profile_gate(repo,plan):
    from ..h4_geometry.prelaunch_audit import environment_profile, compiler_profile
    env=json.loads((repo/plan['environment_profile']).read_text())
    comp=json.loads((repo/plan['compiler_profile']).read_text())
    observed=environment_profile(env['dependencies'],[Path(p) for p in env['installed_sources']])
    require(observed==env,'existing private environment drift; never install or repair')
    require(compiler_profile(comp['explicit_options'],comp['inherited_defaults'])==comp,'compiler profile drift')
    require(sys.executable==env['python'],'wrong private Python')
    require(comp['explicit_options']==plan['compiler_options'],'fixed compiler options')
    from ..h4_geometry.gates import THREAD_ENV
    require(all(os.environ.get(k)==v for k,v in THREAD_ENV.items()),'existing process-only thread/Qiskit limits')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        require(os.environ.get(key)=='1','internal thread1 environment: '+key)
    require(sys.flags.safe_path and sys.dont_write_bytecode,'Python -P -B required')


class Journal:
    def __init__(self,root):self.root=root;self.bytes=0
    def write(self,name,value):
        require(Path(name).name==name,'local journal name')
        raw=json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()+b'\n'
        require(len(raw)<=65536 and self.bytes+len(raw)<=CAPS['journal_bytes'],'metadata journal cap')
        path=self.root/name
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:require(os.write(fd,raw)==len(raw),'journal write');os.fsync(fd)
        finally:os.close(fd)
        from ..h4_geometry.resources import fsync_directory
        fsync_directory(self.root);self.bytes+=len(raw)


def owned_bytes(root):
    base=root.stat();require(base.st_uid==os.getuid(),'owned run root')
    total=base.st_blocks*512;inodes=1
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            path=Path(directory)/name;require(not path.is_symlink(),'unexpected run symlink')
            stat=path.stat();require(stat.st_uid==os.getuid(),'foreign run file')
            total+=stat.st_blocks*512;inodes+=1
            require(inodes<=CAPS['run_inode_cap'],'run inode cap')
    return total,inodes


def disk_guard(plan):
    output=Path(plan['output']);control=Path(plan['control'])
    sizes=[owned_bytes(p) for p in (output,control)]
    lock=Path(plan['one_shot_lock']).stat()
    count=sum(v[0] for v in sizes)+lock.st_blocks*512
    require(sum(v[1] for v in sizes)+1<=plan['caps']['run_inode_cap'],'combined output/control/lock inode cap')
    require(count+65536<=plan['caps']['stage_output_bytes'],'run block/output cap and terminal metadata reserve')
    stat=os.statvfs(control)
    require(stat.f_bavail*stat.f_frsize>=64*2**20 and stat.f_favail>=16,'terminal free block/inode reserve')


def host_oom_counter():
    fields=[line.split() for line in Path('/proc/vmstat').read_text().splitlines() if line.startswith('oom_kill ')]
    require(len(fields)==1 and len(fields[0])==2 and fields[0][1].isdigit(),'host OOM counter unavailable')
    return int(fields[0][1])


def poll_owned(monitor,plan):
    monitor.poll();disk_guard(plan)
    require(host_oom_counter()==monitor.host_oom_baseline,'host OOM kill counter changed')


def receive_wait(process,monitor,plan,started):
    deadline=min(started+plan['caps']['stage_wall_seconds'],time.monotonic()+plan['caps']['task_wall_seconds'])
    data=bytearray();length=None
    while True:
        require(time.monotonic()<deadline,'bounded map task/stage wall')
        poll_owned(monitor,plan)
        if select.select([process.stdout],[],[],max(.001,min(1,deadline-time.monotonic())))[0]:
            require(process.poll() is None,'worker exit before complete result')
            piece=os.read(process.stdout.fileno(),(8-len(data)) if length is None else length-len(data))
            require(bool(piece),'owned worker EOF before complete frame');data.extend(piece)
            if length is None and len(data)==8:
                length=int.from_bytes(data,'big');require(0<length<=8192,'bounded response metadata frame');data.clear()
            elif length is not None and len(data)==length:return pickle.loads(data)


def send_wait(process,value,monitor,plan,started):
    payload=pickle.dumps(value,protocol=5);require(0<len(payload)<=plan['caps']['bootstrap_frame_bytes'],'bounded parent command')
    pending=memoryview(len(payload).to_bytes(8,'big')+payload)
    deadline=min(started+plan['caps']['stage_wall_seconds'],time.monotonic()+plan['caps']['task_wall_seconds'])
    while pending:
        require(time.monotonic()<deadline,'bounded command I/O wall');poll_owned(monitor,plan)
        if select.select([],[process.stdin],[],max(.001,min(1,deadline-time.monotonic())))[1]:
            require(process.poll() is None,'worker exit during command')
            try:count=os.write(process.stdin.fileno(),pending)
            except BlockingIOError:continue
            require(count>0,'worker command write');pending=pending[count:]


def accept_result(plan,task,result):
    require(result.get('status')!='ERROR','worker first error: '+str(result.get('error','missing error')))
    require(result.get('status')=='COMPLETE' and result.get('task_id')==task['task_id'] and
            result.get('source_commit')==plan['source_commit'] and result.get('old_cache_reuse') is False,'complete-only task/source/no-old-cache')
    require(result.get('plan_digest')==digest(plan),'complete-only plan binding')
    from ..h4_geometry.identity import hash_id
    hash_id(result['numeric_circuit_fingerprint']);hash_id(result['validation_wrapper_fingerprint'])
    require(result['validation_wrapper_fingerprint']==bound_fingerprint(task,plan['source_commit'],result['numeric_circuit_fingerprint']),'wrapper identity mismatch')
    expected={'rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size'}
    require(set(result['metrics'])==expected and all(type(v) is int and v>=0 for v in result['metrics'].values()),'complete metric schema')
    require(result.get('trajectory_seed') is None,'no trajectory seed')
    expected_reuse=task['action']=='VERIFY_REUSE'
    require(result.get('verified_pilot6_reuse') is expected_reuse and result.get('new_compile_invocations')==int(not expected_reuse),'compile/reuse result accounting')
    if expected_reuse:
        expected=task['reuse_receipt']
        require(result.get('original_pilot_task_id')==expected['old_task_id'] and result['numeric_circuit_fingerprint']==expected['expected_numeric_fingerprint'],'complete mapped pilot numeric identity')
    require(type(result.get('transpile_wall_seconds')) is float and result['transpile_wall_seconds']>=0,'measured transpile wall')
    return result


def retire_observer_after_map(monitor,journal,records):
    """End observation after all compute, while the sole worker is still idle/alive.

    The legacy observer treats an unexpectedly exited worker as STOP. Do not
    create an exit/reap race by exiting the worker before retiring this observer.
    This is an intentional owned SIGTERM with a measured, checked native exit.
    """
    require(len(records)==240,'no observer retirement during incomplete work')
    monitor.poll();monitor.phase('map240_complete_receipts_observer_retirement')
    require(monitor.owner.terminate(),'observer exited before intentional retirement')
    monitor.process.wait(timeout=2)
    require(monitor.process.returncode==-signal.SIGTERM,'unexpected observer terminal exit')
    journal.write('observer-retirement.json',dict(kind='OWNED_INTENTIONAL_TERMINAL_SIGTERM',
        observer_identity=monitor.owner.expected,native_exit_code=monitor.process.returncode,
        complete_task_ids=[row['task_id'] for row in records],worker_still_idle_before_shutdown=True))


def execute(repo,plan_path,auth_path,review_path):
    plan,auth,review=load_grant(repo,plan_path,auth_path,review_path)
    authorize(plan,auth,review)
    source_gate(repo,plan);profile_gate(repo,plan);reuse_gate(repo,plan)
    output=private_path(plan['output']);control=private_path(plan['control']);lock=private_path(plan['one_shot_lock'])
    require(not output.exists() and not control.exists() and not lock.exists(),'unused output/control/one-shot lock required')
    require(output.parent==control.parent==lock.parent and output.parent.is_dir(),'owned existing map parent')
    host_oom_before=host_oom_counter()
    fresh=host_readonly(output.parent,sample_seconds=.5);fresh_gate(plan,auth,fresh)
    require(host_oom_counter()==host_oom_before,'host OOM changed during fresh admission')
    claim_lock(lock,digest(plan));output.mkdir(mode=0o700);control.mkdir(mode=0o700)
    temp=control/'private-temp';temp.mkdir(mode=0o700)
    for name in ('mpl','cache'):(temp/name).mkdir()
    journal=Journal(output);first=FirstStop();children=[];monitor=None;records=[];started=time.monotonic()
    from ..h4_geometry.observer import IndependentObserver,OwnedIdentity,identity,process_sample
    from ..h4_geometry.resources import limit_owned_address_space
    from ..h4_geometry.pressure_policy import PARAMETERS
    old_term=signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(Stop('observer signalled owned driver')))
    old_alrm=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(Stop('map stage wall deadline')))
    old_alarm=signal.alarm(plan['caps']['stage_wall_seconds'])
    try:
        limit_owned_address_space(plan['caps']['driver_AS_RSS'])
        os.sched_setaffinity(0,{auth['role_cpus']['driver']})
        # Guard before importing NumPy/Qiskit or decoding scientific arrays.
        journal.write('started.json',dict(plan_digest=digest(plan),auth_digest=digest(auth),source=plan['source_commit'],
            carry=plan['carry'],grant_namespace=plan['namespace'],roles=auth['role_cpus']))
        monitor=IndependentObserver(sys.executable,control/'observer.jsonl',scope='PRODUCTION',workers=1,
            output_cap=plan['caps']['observer_output_bytes'],runtime_authorization=True,wall_started=started,
            prior_wall=0,allowed_cpus=auth['allowed_cpus'],role_cpus=[auth['role_cpus']['observer']],
            pressure_profile=PARAMETERS,pressure_baseline=fresh['memory'],worker_cap=plan['caps']['worker_AS_RSS'])
        monitor.host_oom_baseline=host_oom_before
        require(host_oom_counter()==host_oom_before,'host OOM changed during observer startup')
        journal.write('host-oom-baseline.json',{'oom_kill':monitor.host_oom_baseline})
        monitor.phase('verify_frozen_bytes')
        for info in {t['input_file']['path']:t['input_file'] for t in plan['tasks']}.values():
            actual=streaming_sha(Path(info['path']))
            require(actual=={'sha256':info['sha256'],'bytes':info['bytes']},'fresh frozen input SHA/bytes')
        env=dict(os.environ,TMPDIR=str(temp),MPLCONFIGDIR=str(temp/'mpl'),XDG_CACHE_HOME=str(temp/'cache'),
                 PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',QISKIT_PARALLEL='false')
        monitor.phase('owned_worker_start')
        process=subprocess.Popen([sys.executable,'-P','-B',str(repo/'scripts/resource_applicability/run_h4_b0i_map240.py'),
            '--owned-worker',str(os.getpid())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,cwd=repo,env=env,close_fds=True,bufsize=0)
        children.append((process,None))
        owner=OwnedIdentity(identity(process_sample(process.pid)),os.getpid());children[-1]=(process,owner);monitor.own_child(process.pid)
        os.set_blocking(process.stdin.fileno(),False);os.set_blocking(process.stdout.fileno(),False)
        send_wait(process,{'plan_path':str(plan_path),'auth_path':str(auth_path),'review_path':str(review_path),'repo':str(repo),'plan_digest':digest(plan)},monitor,plan,started)
        ready=receive_wait(process,monitor,plan,started)
        require(ready.get('status')!='ERROR','worker bootstrap first error: '+str(ready.get('error')))
        require(ready=={'ready':process.pid},'new worker source/auth handshake')
        calls=CallLedger(plan['tasks'],reserve=lambda row:journal.write('reservation-%02d.json'%row['call'],row))
        for task in plan['tasks']:
            monitor.phase(task['action'].lower()+'_'+task['geometry']+'_p'+str(task['prefix'])+'_q'+str(task['template']['q'])+'_'+task['axis'])
            if task['action']=='COMPILE':calls.before_call(task['task_id'])
            send_wait(process,{'task_id':task['task_id']},monitor,plan,started)
            result=accept_result(plan,task,receive_wait(process,monitor,plan,started))
            journal.write('task-%s.json'%task['task_id'],result);records.append(result)
            blocks,_=owned_bytes(control);require(blocks+journal.bytes<=plan['caps']['stage_output_bytes'],'stage cumulative block/metadata cap')
        retire_observer_after_map(monitor,journal,records)
        # No further computation is dispatched after observer retirement.
        # The final tiny shutdown frame cannot fill a drained idle worker pipe.
        from ..h4_geometry.workers import write_frame
        os.set_blocking(process.stdin.fileno(),True)
        write_frame(process.stdin,{'shutdown':True});process.stdin.close()
        process.wait(timeout=2);require(process.returncode==0,'native worker completion exit')
        # Completes are final only after registered workers exit/reap and all FDs close.
    except BaseException as exc:
        if monitor is not None:
            try:monitor.capture_failure(exc)
            except BaseException:pass
            if monitor.first_failure is not None:first.keep(monitor.first_failure['reason'])
        first.keep(exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        clean=cleanup(children,monitor,first)
        try:
            blocks,inodes=owned_bytes(control)
            require(blocks+journal.bytes<=plan['caps']['stage_output_bytes'],'terminal output cap')
        except BaseException as exc:
            blocks=None;clean=False;first.keep('terminal storage audit: '+str(exc))
        terminal=dict(status='MAP240_COMPLETE_STOP' if first.reason is None and clean and len(records)==240 else 'MAP240_FAIL_CLOSED_STOP',
            first_stop_reason=first.reason,cleanup_complete=clean,records=len(records),
            actual_consumed_or_reserved=len(calls.reserved) if 'calls' in locals() else 0,
            verified_pilot6_reuses=sum(v['verified_pilot6_reuse'] for v in records),
            native_worker_exit_code=children[0][0].returncode if children else None,
            native_observer_exit_code=monitor.process.returncode if monitor is not None else None,
            stage_wall_seconds=time.monotonic()-started,stage_allocated_control_bytes=blocks,
            carry=plan['carry'],runtime_next_stage=False,mandatory_stop=True)
        try:journal.write('terminal.json',terminal)
        except BaseException as exc:
            first.keep('terminal publication: '+str(exc));terminal.update(status='PILOT6_FAIL_CLOSED_STOP',
                first_stop_reason=first.reason,terminal_receipt_written=False)
        finally:
            signal.signal(signal.SIGALRM,old_alrm);signal.signal(signal.SIGTERM,old_term)
            if old_alarm:signal.alarm(max(1,old_alarm-int(time.monotonic()-started)))
    return terminal


def worker_main(parent):
    require(os.getppid()==parent,'private new worker parent handshake')
    from ..h4_geometry.workers import read_frame,write_frame,CappedText
    WORKER_STATE['phase']='configuration_auth_source_profile'
    config=read_frame(sys.stdin.buffer);repo=Path(config['repo'])
    plan,auth,review=load_grant(repo,*[Path(config[k]) for k in ('plan_path','auth_path','review_path')])
    require(digest(plan)==config['plan_digest'],'worker bootstrap plan digest')
    authorize(plan,auth,review);source_gate(repo,plan);profile_gate(repo,plan);reuse=reuse_gate(repo,plan)
    from ..h4_geometry.resources import limit_owned_address_space
    limit_owned_address_space(plan['caps']['worker_AS_RSS']);os.sched_setaffinity(0,{auth['role_cpus']['worker']})
    write_frame(sys.stdout.buffer,{'ready':os.getpid()})
    cached=None;cached_key=None;calls=CallLedger(plan['tasks']);seen=set()
    while True:
        WORKER_STATE['phase']='job_decode'
        job=read_frame(sys.stdin.buffer)
        if job=={'shutdown':True}:return
        require(set(job)=={'task_id'} and job['task_id'] not in seen,'closed unique full-map worker command')
        seen.add(job['task_id'])
        task=next(t for t in plan['tasks'] if t['task_id']==job['task_id'])
        if task['action']=='COMPILE':calls.before_call(task['task_id'])
        WORKER_STATE['task_id']=task['task_id']
        from contextlib import redirect_stdout,redirect_stderr
        log=CappedText()
        with redirect_stdout(log),redirect_stderr(log):
            if cached_key!=(task['geometry'],task['prefix']):
                WORKER_STATE['phase']='kept_circuit_preparation'
                cached=prepare_kept_only(task);cached_key=(task['geometry'],task['prefix'])
            # Separate real invocation for each registered axis; never old B0 metric/cache.
            from qiskit import transpile
            WORKER_STATE['phase']='build_fingerprint_compile_metrics'
            result=_compile_task(plan,task,cached,transpile,reuse.get(task['task_id']))
        result['log']=log.getvalue()
        require(len(json.dumps(result))<=8192,'bounded worker metadata frame')
        write_frame(sys.stdout.buffer,result)


def worker_entry(parent):
    try:worker_main(parent)
    except BaseException as exc:
        from ..h4_geometry.workers import exception_text,write_frame
        try:
            write_frame(sys.stdout.buffer,dict(status='ERROR',task_id=WORKER_STATE['task_id'],
                error=exception_text(exc,WORKER_STATE['phase'])))
            if os.getppid()==parent:
                while sys.stdin.buffer.read(65536):pass  # preserve cause before parent STOP/EOF, no further dispatch
        except (OSError,ValueError):pass
        raise SystemExit(1)


def _compile_task(plan,task,b0,compiler,reuse=None):
    """Same scientific builder/compiler; reused metrics require a fresh exact numeric FP."""
    from ..h4_geometry.circuits import wrapper,numerical_fingerprint,metrics
    build_started=time.monotonic()
    circuit=wrapper(evolution(b0,task['template'],task['tail_identity']),task['axis'])
    numeric=numerical_fingerprint(circuit,task['axis'])
    validation=bound_fingerprint(task,plan['source_commit'],numeric)
    build_wall=time.monotonic()-build_started
    measured=0.;old_id=None
    if task['action']=='VERIFY_REUSE':
        require(reuse is not None,'verified complete pilot receipt required')
        require(numeric==task['reuse_receipt']['expected_numeric_fingerprint']==reuse['numeric_circuit_fingerprint'],'fresh reused numeric circuit mismatch; no automatic compile fallback')
        value=dict(reuse['metrics']);old_id=reuse['task_id']
    else:
        require(reuse is None,'no metric cache on a new compile')
        begun=time.monotonic();compiled=compiler(circuit,**plan['compiler_options']);measured=time.monotonic()-begun
        value=metrics(compiled)
    require(numerical_fingerprint(circuit,task['axis'])==numeric,'compiler input mutation')
    status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    return dict(status='COMPLETE',task_id=task['task_id'],source_commit=plan['source_commit'],
        plan_digest=digest(plan),numeric_circuit_fingerprint=numeric,validation_wrapper_fingerprint=validation,
        metrics=value,old_cache_reuse=False,trajectory_seed=None,
        verified_pilot6_reuse=task['action']=='VERIFY_REUSE',new_compile_invocations=int(task['action']=='COMPILE'),
        original_pilot_task_id=old_id,transpile_wall_seconds=measured,build_fingerprint_wall_seconds=build_wall,
        worker_lifetime_peak_RSS_bytes=int(status['VmHWM'].split()[0])*1024,
        worker_peak_RSS_scope='native worker VmHWM through this task; not isolated per-task peak')

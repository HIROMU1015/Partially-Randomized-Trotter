"""One-shot outer owner, isolated ledger and capped synthesis workers."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from .design import integer_io_limit
from .launch import verify_launch,verify_worker,execution_identity,check_identities,read_contract
from ..rte_development.evidence import atomic_json,digest
from .evidence import BatchedEvidenceWriter as EvidenceWriter
from ..rte_development.launch import Guard,consume_marker,git

COMPLETE='PARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW'
INCONCLUSIVE='PARENT_ALLOCATION_TECHNICAL_INCONCLUSIVE'


def runtime(root,c):
    if digest(Path(sys.executable).resolve())!=c['acquisition']['runtime_executable_sha256']:
        raise PermissionError('fixed runtime executable required')
    from ..synthesis_placement.wrapper_launch import verify_runtime
    return verify_runtime(root,dict(tool_identity=c['acquisition']['tool_identity']))


def ledger_job(root,path):
    root=Path(root);path=Path(path);c,job,d=verify_worker(root,path,'ledger')
    atomic_json(path.with_suffix('.started.json'),dict(**job,pid=os.getpid(),role='ledger'))
    guard=Guard(c['caps'],d)
    resource.setrlimit(resource.RLIMIT_CPU,(c['caps']['total_CPU_seconds'],c['caps']['total_CPU_seconds']+1))
    try:
        with guard,integer_io_limit(c['caps']['max_integer_decimal_digits']):
            if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
                raise PermissionError('ledger threads')
            r=runtime(root,c);writer=EvidenceWriter(d,c['caps']['output_bytes'],guard.check,c['caps']['chunk_bytes'],
                write_block_bytes=c['execution']['json_write_block_bytes'])
            writer.write('runtime_receipt.json',dict(**job,runtime=r))
            from .pipeline import run_pipeline
            coverage=run_pipeline(root,c,job,writer,guard)
            check_identities(root,c)
            read_contract(root)
            if digest(root/c['execution']['source_manifest'])!=job['source_manifest_sha256']:
                raise PermissionError('source manifest changed during run')
            if digest(d/'one_shot_consumed.json')!=job['marker_sha256']:raise PermissionError('marker changed during run')
            if git(root,'rev-parse','HEAD')!=job['execution_HEAD']:raise PermissionError('execution HEAD changed')
            if digest(root/c['execution']['authorization_path'])!=job['authorization_sha256']:
                raise PermissionError('authorization changed during run')
            guard.check()
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>c['caps']['parent_RSS_MiB']*1024:
                raise RuntimeError('peak ledger RSS cap')
            writer.write('evidence_manifest.json',dict(**job,files=writer.files,coverage=coverage,status=COMPLETE))
            terminal=dict(**job,status=COMPLETE,**coverage,retries=0,resource=guard.snapshot(),prefix_scientific_use=False);code=0
    except BaseException as exc:
        terminal=dict(**job,status=INCONCLUSIVE,retries=0,reason=type(exc).__name__+': '+str(exc)[:2000],
                      resource=guard.snapshot(),prefix_scientific_use=False);code=1
    atomic_json(d/'inner_terminal.json',dict(terminal,mandatory_STOP=True,next_science_authorized=False),32768)
    return code


def process_tree(pid):
    records={};ticks=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:
            raw=(path/'stat').read_text();f=raw[raw.rfind(')')+2:].split();cmd=(path/'cmdline').read_bytes().split(b'\0')
            records[int(path.name)]=dict(ppid=int(f[1]),RSS=int(f[21])*page,own_CPU=(int(f[11])+int(f[12]))/ticks,
                completed_child_CPU=(int(f[13])+int(f[14]))/ticks,key=any(v.endswith(b'/run_parent_allocation_synthesis_key.py') for v in cmd))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    selected={pid}
    while True:
        new={p for p,v in records.items() if v['ppid'] in selected}
        if new<=selected:break
        selected|=new
    values=[dict(pid=p,**records[p]) for p in selected if p in records]
    return dict(processes=values,RSS=sum(v['RSS'] for v in values),CPU=sum(v['own_CPU']+v['completed_child_CPU'] for v in values))


def resource_reason(snapshot,caps,output_bytes,log_sizes):
    if snapshot['CPU']>caps['total_CPU_seconds']:return 'aggregate CPU cap'
    if snapshot['RSS']>caps['aggregate_RSS_MiB']*1048576:return 'aggregate RSS cap'
    for v in snapshot['processes']:
        limit=caps['worker_RSS_MiB'] if v['key'] else caps['parent_RSS_MiB']
        if v['RSS']>limit*1048576:return 'process RSS cap'
        if v['key'] and v['own_CPU']+v['completed_child_CPU']>caps['per_key_CPU_seconds']:return 'per-key CPU cap'
    if sum(v['key'] for v in snapshot['processes'])>caps['workers']:return 'worker count cap'
    if output_bytes>caps['output_bytes']-32768:return 'output cap'
    if any(n>caps['per_key_log_bytes'] for n in log_sizes):return 'log cap'
    return None


def final_status(code,inner,identity,reason,c):
    if (reason is None and code==0 and inner and inner.get('status')==COMPLETE
        and all(inner.get(k)==v for k,v in identity.items()) and inner.get('retries')==0
        and type(inner.get('configurations')) is int and 1<=inner['configurations']<=c['caps']['configurations']
        and type(inner.get('laws')) is int and 1<=inner['laws']<=c['caps']['laws']
        and type(inner.get('new_synthesis_calls')) is int and 0<=inner['new_synthesis_calls']<=c['caps']['new_synthesis_keys']):return COMPLETE
    return INCONCLUSIVE


def execute(root,source):
    root=Path(root);c,a,head,m=verify_launch(root,source);r=runtime(root,c)
    identity=execution_identity(root,c,a,head);d=root/c['execution']['result_directory']
    marker,mark=consume_marker(d,dict(**identity,runs=1,retries=0,mandatory_STOP=True))
    identity['marker_sha256']=mark['sha256'];job=dict(**identity,parent_pid=os.getpid())
    atomic_json(d/'ledger_job.json',job)
    atomic_json(d/'job_start_receipt.json',dict(**identity,owner_pid=os.getpid(),runtime=r))
    child=None;reason=None;exit_code=None;started=time.monotonic();peak=0;cpu=0;usage=None
    outpath=d/'ledger.stdout.log';errpath=d/'ledger.stderr.log'
    try:
        env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with outpath.open('xb') as out,errpath.open('xb') as err:
            child=subprocess.Popen([c['acquisition']['runtime_executable'],'-B',c['execution']['runner'],'--ledger-job',str(d/'ledger_job.json')],
                cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
            while child.poll() is None:
                snapshot=process_tree(os.getpid());peak=max(peak,snapshot['RSS']);cpu=max(cpu,snapshot['CPU'])
                output=sum(p.stat().st_size for p in d.rglob('*') if p.is_file())
                logs=[p.stat().st_size for p in d.rglob('*.log')]
                reason=resource_reason(snapshot,c['caps'],output,logs)
                if reason:
                    os.killpg(child.pid,signal.SIGTERM);break
                time.sleep(.1)
            try:exit_code=child.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);exit_code=child.wait()
    except BaseException as exc:
        reason=type(exc).__name__+': '+str(exc)[:2000]
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);exit_code=child.wait()
    # Descendants may remain after a ledger crash; terminate its session before STOP.
    if child:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    if usage.ru_maxrss>c['caps']['parent_RSS_MiB']*1024:reason=reason or 'completed child peak RSS cap'
    if usage.ru_utime+usage.ru_stime+time.process_time()>c['caps']['total_CPU_seconds']:reason=reason or 'completed tree CPU cap'
    inner=None
    if (d/'inner_terminal.json').exists():
        try:inner=json.loads((d/'inner_terminal.json').read_text())
        except (OSError,ValueError) as exc:reason=reason or 'invalid inner terminal: '+str(exc)[:500]
    try:
        read_contract(root);check_identities(root,c)
        current=execution_identity(root,c,a,git(root,'rev-parse','HEAD'))
        if any(current.get(k)!=v for k,v in identity.items() if k!='marker_sha256') or digest(marker)!=identity['marker_sha256']:
            raise PermissionError('final source/authorization/manifest/marker identity')
    except BaseException as exc:
        reason=reason or type(exc).__name__+': '+str(exc)[:2000]
    status=final_status(exit_code,inner,identity,reason,c)
    receipt=dict(**identity,exit_code=exit_code,inner_status=inner.get('status') if inner else None,
        observed_peak_aggregate_RSS_bytes=peak,observed_CPU_seconds=cpu,completed_tree_CPU_seconds=usage.ru_utime+usage.ru_stime,
        completed_child_peak_RSS_KiB=usage.ru_maxrss,outer_process_CPU_seconds=time.process_time(),wall_seconds=time.monotonic()-started,
        final_output_bytes=sum(p.stat().st_size for p in d.rglob('*') if p.is_file()),resource_failure=reason,
        effective_status=status,inner_terminal_sha256=digest(d/'inner_terminal.json') if inner else None,
        backend_call_tokens=len(list(d.glob('jobs/*.backend_call_consumed.json'))),
        tokens_are_not_completed_calls=True,retries=0,mandatory_STOP=True)
    atomic_json(d/'outer_process_receipt.json',receipt,32768)
    terminal=dict(**identity,status=status,technical_reason=reason or (inner.get('reason') if inner else 'missing inner terminal'),
        coverage={k:inner.get(k) for k in ('configurations','laws','old_guards','new_synthesis_calls','new_root_matrix_checks')} if inner and status==COMPLETE else None,
        prefix_scientific_use=False,runs=1,retries=0,mandatory_STOP=True,next_science_authorized=False,
        outer_process_receipt_sha256=digest(d/'outer_process_receipt.json'))
    atomic_json(d/'terminal.json',terminal,32768)
    atomic_json(d/'STOP.json',dict(mandatory_STOP=True,retries=0,next_science_authorized=False,terminal_sha256=digest(d/'terminal.json')))
    print(status);return 0 if status==COMPLETE else 1

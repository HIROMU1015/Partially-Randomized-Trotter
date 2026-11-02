"""Exclusive one-shot owner and external process resource observer."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from .launch import (verify_launch, verify_job, execution_identity, check_identities,
                     check_execution_identity, COMPLETE, INCONCLUSIVE, CONTRACT)
from ..rte_development.evidence import atomic_json, digest, EvidenceWriter
from ..rte_development.launch import Guard, consume_marker


def runtime_preflight(root,c):
    if digest(Path(sys.executable).resolve()) != c['acquisition']['runtime_executable_sha256']:
        raise PermissionError('fixed D3 runtime executable required')
    from ..synthesis_placement.wrapper_launch import verify_runtime
    return verify_runtime(Path(root),dict(tool_identity=c['acquisition']['tool_identity']))


def process_tree(pid, proc=Path('/proc')):
    """CPU includes exited descendants via cutime/cstime; RSS sums live tree."""
    records={}
    ticks=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')
    for path in proc.iterdir():
        if not path.name.isdecimal():continue
        try:
            raw=(path/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            cmd=(path/'cmdline').read_bytes().split(b'\0')
            records[int(path.name)]=dict(ppid=int(fields[1]),RSS=int(fields[21])*page,
                  own_CPU=(int(fields[11])+int(fields[12]))/ticks,
                  completed_child_CPU=(int(fields[13])+int(fields[14]))/ticks,
                  key=any(x.endswith(b'/run_d3_synthesis_key.py') for x in cmd))
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
    selected={pid}
    while True:
        new={p for p,v in records.items() if v['ppid'] in selected}
        if new <= selected:break
        selected |= new
    values=[dict(pid=p,**records[p]) for p in selected if p in records]
    return dict(processes=values,RSS=sum(v['RSS'] for v in values),
                CPU=sum(v['own_CPU']+v['completed_child_CPU'] for v in values))


def resource_reason(snapshot,caps,output_bytes,log_sizes,own_CPU=0):
    if snapshot['CPU']+own_CPU > caps['total_CPU_seconds']:return 'total CPU cap'
    if snapshot['RSS'] > caps['aggregate_RSS_MiB']*1048576:return 'aggregate RSS cap'
    if any(v['RSS']>caps['worker_RSS_MiB']*1048576 for v in snapshot['processes']):
        return 'process RSS cap'
    if any(v['key'] and v['own_CPU']+v['completed_child_CPU']>caps['per_key_CPU_seconds'] for v in snapshot['processes']):
        return 'per-key CPU cap'
    if sum(v['key'] for v in snapshot['processes'])>caps['workers']:return 'worker count cap'
    if output_bytes > caps['output_bytes']-32768:return 'output reserve cap'
    if any(v>caps['per_key_log_bytes'] for v in log_sizes):return 'log cap'
    return None


def final_status(exit_code,inner,reason,identity):
    if (reason is None and exit_code==0 and inner
        and inner.get('status')==COMPLETE
        and all(inner.get(k)==v for k,v in identity.items())
        and inner.get('rows')==48 and inner.get('new_synthesis_calls')==36
        and inner.get('registered_pair_evaluations')==0
        and inner.get('retries')==0):
        return COMPLETE
    return INCONCLUSIVE


def ledger_job(root,job_path):
    root=Path(root);c,job,directory=verify_job(root,job_path,'ledger')
    atomic_json(directory/'ledger_started.json',dict(**job,pid=os.getpid(),role='D3-ledger'))
    identity={k:job[k] for k in execution_identity(root,c,{},job['execution_HEAD'],job['source_commit'])}
    identity['marker_sha256']=job['marker_sha256']
    guard=Guard(c['caps'],directory)
    try:
        with guard:
            runtime=runtime_preflight(root,c);check_identities(root,c)
            writer=EvidenceWriter(directory,c['caps']['output_bytes'],guard.check,c['caps']['review_chunk_bytes'])
            writer.write('preflight.json',dict(**identity,runtime=runtime,source_contract_unchanged=True))
            from .pipeline import run
            coverage=run(root,c,identity,writer,guard)
            check_identities(root,c);check_execution_identity(root,c,identity);guard.check()
            writer.write('evidence_manifest.json',dict(**identity,files=list(writer.files),
                         classification=COMPLETE,coverage=coverage))
            inner=dict(status=COMPLETE,**identity,**coverage,retries=0,runs=1,
                       resource=guard.snapshot(),mandatory_STOP=True,next_science_authorized=False,
                       provider_or_signal_matrix_evaluations=0,strict_guard_2x2_key_checks=36,
                       circuit_builds=0,native_compile=0,
                       sampling=0,GPU=0,prefix_scientific_use=False)
            code=0
    except BaseException as exc:
        inner=dict(status=INCONCLUSIVE,**identity,retries=0,runs=1,
                   reason=type(exc).__name__+': '+str(exc)[:1500],resource=guard.snapshot(),
                   mandatory_STOP=True,next_science_authorized=False,prefix_scientific_use=False)
        code=1
    atomic_json(directory/'inner_terminal.json',inner,16384)
    return code


def execute(root,source):
    root=Path(root);c,a,head,manifest=verify_launch(root,source)
    runtime=runtime_preflight(root,c)
    identity=execution_identity(root,c,a,head,source)
    directory=root/c['execution']['result_directory']
    marker,mark=consume_marker(directory,dict(kind='D3_ONE_SHOT_CONSUMED',**identity,
                           runs=1,retries=0,mandatory_STOP=True))
    identity['marker_sha256']=mark['sha256']
    job=dict(**identity,parent_pid=os.getpid())
    atomic_json(directory/'ledger_job.json',job)
    atomic_json(directory/'job_start_receipt.json',dict(**identity,owner_pid=os.getpid(),runtime=runtime))
    start=time.monotonic();child=None;usage=None;reason=None;inner=None;exit_code=None
    peak_aggregate=peak_process=0
    stdout=directory/'process.stdout.log';stderr=directory/'process.stderr.log'
    try:
        with stdout.open('xb') as out,stderr.open('xb') as err:
            env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1',
                     OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            child=subprocess.Popen([c['acquisition']['runtime_executable'],'-B',
                      'scripts/tracks/algorithm_codesign/run_d3_ledger_job.py',
                      '--job',str(directory/'ledger_job.json')],cwd=root,env=env,
                      stdout=out,stderr=err,start_new_session=True)
            while True:
                waited,status,usage=os.wait4(child.pid,os.WNOHANG)
                if waited:
                    exit_code=os.waitstatus_to_exitcode(status);child.returncode=exit_code
                    break
                snap=process_tree(child.pid)
                # The observer itself is included in aggregate RSS/CPU accounting.
                own=process_tree(os.getpid())['processes']
                own=next((v for v in own if v['pid']==os.getpid()),None)
                if own:snap['RSS']+=own['RSS']
                peak_aggregate=max(peak_aggregate,snap['RSS'])
                peak_process=max([peak_process]+[v['RSS'] for v in snap['processes']]+([own['RSS']] if own else []))
                files=[p for p in directory.rglob('*') if p.is_file()]
                n=sum(p.stat().st_size for p in files)
                logs=[p.stat().st_size for p in files if p.suffix=='.log']
                reason=resource_reason(snap,c['caps'],n,logs,time.process_time())
                if own and own['RSS']>c['caps']['parent_RSS_MiB']*1048576:reason='observer RSS cap'
                if reason:
                    os.killpg(child.pid,signal.SIGKILL)
                    _,status,usage=os.wait4(child.pid,0)
                    exit_code=os.waitstatus_to_exitcode(status);child.returncode=exit_code;break
                time.sleep(.1)
        if usage and (usage.ru_utime+usage.ru_stime+time.process_time()>c['caps']['total_CPU_seconds']
                      or usage.ru_maxrss>c['caps']['worker_RSS_MiB']*1024):
            reason=reason or 'final CPU/peak RSS cap'
        # A fast exiting child can evade the live polling interval. Validate
        # final output/log sizes too, before accepting its complete terminal.
        files=[p for p in directory.rglob('*') if p.is_file()]
        final_snapshot=dict(CPU=usage.ru_utime+usage.ru_stime if usage else 0,
                            RSS=0,processes=[])
        final_reason=resource_reason(final_snapshot,c['caps'],
                     sum(p.stat().st_size for p in files),
                     [p.stat().st_size for p in files if p.suffix=='.log'],time.process_time())
        reason=reason or final_reason
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>c['caps']['parent_RSS_MiB']*1024:
            reason=reason or 'final observer peak RSS cap'
        path=directory/'inner_terminal.json'
        inner=json.loads(path.read_text()) if path.exists() else None
        check_identities(root,c);check_execution_identity(root,c,identity)
    except BaseException as exc:
        reason=type(exc).__name__+': '+str(exc)[:1500]
    finally:
        if child and child.returncode is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            _,status,usage=os.wait4(child.pid,0)
            exit_code=os.waitstatus_to_exitcode(status);child.returncode=exit_code
    status=final_status(exit_code,inner,reason,identity)
    started=len(list(directory.glob('jobs/*.started.json')))
    consumed=len(list(directory.glob('jobs/*.backend_call_consumed.json')))
    result_files=list(directory.glob('jobs/*.result.json'))
    receipt=dict(**identity,exit_code=exit_code,signal=-exit_code if exit_code is not None and exit_code<0 else None,
                 wall_seconds=time.monotonic()-start,
                 CPU_seconds_including_children=(usage.ru_utime+usage.ru_stime+time.process_time()) if usage else time.process_time(),
                 child_peak_RSS_KiB=usage.ru_maxrss if usage else None,
                 observed_peak_aggregate_RSS_bytes=peak_aggregate,observed_peak_process_RSS_bytes=peak_process,
                 resource_failure=reason,inner_status=inner.get('status') if inner else None,
                 inner_terminal_sha256=digest(directory/'inner_terminal.json') if (directory/'inner_terminal.json').exists() else None,
                 status_consistent=(status==COMPLETE),effective_status=status,
                 retries=0,runs=1,mandatory_STOP=True,next_science_authorized=False,
                 key_jobs_started=started,backend_call_tokens_consumed=consumed,
                 key_result_files_saved=len(result_files),
                 token_count_is_not_completed_synthesis_count=True,
                 external_logs=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in (stdout,stderr) if p.exists()])
    atomic_json(directory/'outer_process_receipt.json',receipt,16384)
    terminal=dict(**identity,status=status,technical_reason=reason or (inner.get('reason') if inner else 'inner terminal missing'),
                  rows=48 if status==COMPLETE else None,new_synthesis_calls=36 if status==COMPLETE else None,
                  registered_pair_evaluations=0,runs=1,retries=0,
                  prefix_scientific_use=False,mandatory_STOP=True,next_science_authorized=False,
                  outer_process_receipt_sha256=digest(directory/'outer_process_receipt.json'))
    atomic_json(directory/'terminal.json',terminal,16384)
    atomic_json(directory/'STOP.json',dict(mandatory_STOP=True,retries=0,next_science_authorized=False,
                terminal_sha256=digest(directory/'terminal.json')),4096)
    print(status)
    return 0 if status==COMPLETE else 1

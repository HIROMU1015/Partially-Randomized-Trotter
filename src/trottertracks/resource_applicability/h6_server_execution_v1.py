"""Exactly one registered H6 group, then STOP. Numerical port remains frozen."""
from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from .ax2a_preparation import digest
from .ax2b_limits import output_size
from .ax2b_supplement_records_v1 import AtomicWriter
from .h6_matched_execution_v2 import install_limits,worker_environment,process_tree_rss,Progress
from .h6_matched_execution_v3 import ResourceTraceV3
from .h6_matched_reuse_v3 import read,COMPILER_HASH
from .h6_server_contract_v1 import (RUNNER,flags,validate_manifest,validate_grant,verify_bindings,
    phase_scope,source_paths,environment,file_hash,safe_path,verify_science)
from .h6_server_resources_v1 import (THREAD_KEYS,process_identity,process_memory,host_memory,
    memory_reason,stop_owned_group,install_parent_guard)


def supervise_one(command,output,m,*,progress=None):
    """Independent coordinator watches the science child, including native/GIL work.

    waitid(WNOWAIT) pins the leader until its group is retired and reaped.
    First failure survives later cleanup errors. No retry, no second task.
    """
    output=Path(output);r=m['assigned_resources'];policy=m['observation_policy']
    caps=m['plan']['caps_proposed'];baseline=host_memory();process=None;identity=None
    started=time.monotonic();last_sample=started;last_emit=0.;first=None;code=None
    peak_rss=peak_as=aggregate_peak=0;journal_bytes=0;cleanup_error=None
    cache=output.parent/'.runtime_cache';cache.mkdir(exist_ok=True)
    temporary=cache/'tmp';temporary.mkdir(exist_ok=True)
    env=worker_environment(1,cache)
    env.update(TMPDIR=str(temporary),TEMP=str(temporary),TMP=str(temporary),XDG_CACHE_HOME=str(cache/'xdg'))
    journal_path=output.parent/(output.name+'_resource_observations.jsonl')
    with (output/'worker.log').open('xb') as log, journal_path.open('xb') as journal:
        try:
            process=subprocess.Popen(command,env=env,cwd=m['repository_root'],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=process_identity(process.pid)
            while True:
                now=time.monotonic()
                # Do not reap before group retirement: the numeric PGID cannot be reused.
                exit_info=os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                if exit_info is not None:break
                try:
                    before=time.monotonic();current=process_identity(process.pid)
                    if current['start_ticks']!=identity['start_ticks']:raise ValueError('WORKER_IDENTITY_CHANGED')
                    mem=process_memory(process.pid);host=host_memory();rss=process_tree_rss(os.getpid())
                    after=time.monotonic();duration=after-before;staleness=before-last_sample
                    last_sample=after
                    reason=memory_reason(host,baseline,policy)
                    if duration>policy['maximum_observation_seconds'] or staleness>policy['maximum_observation_seconds']:
                        reason=reason or 'OBSERVATION_STALE'
                    if rss>r['aggregate_rss_bytes']:reason=reason or 'AGGREGATE_RSS_CAP'
                    if mem['VmSize_bytes']>r['cost_address_space_bytes']:reason=reason or 'WORKER_AS_CAP'
                    if (output.parent/'STOP_REQUEST').exists():reason=reason or 'USER_STOP_REQUEST'
                    if output_size(output.parent)>caps['output_bytes']-65536:reason=reason or 'AGGREGATE_OUTPUT_CAP'
                    if (output/'worker.log').stat().st_size>caps['log_bytes']:reason=reason or 'WORKER_LOG_CAP'
                    peak_rss=max(peak_rss,mem['VmHWM_bytes'],mem['VmRSS_bytes'])
                    peak_as=max(peak_as,mem.get('VmPeak_bytes',mem['VmSize_bytes']));aggregate_peak=max(aggregate_peak,rss)
                    first=first or reason
                    if now-last_emit>=policy['journal_emit_seconds'] or first:
                        diagnostics=sorted(output.glob('diagnostic_*.json'))
                        phase=read(diagnostics[-1]).get('context',{}) if diagnostics else {'stage':'startup'}
                        row=dict(monotonic=after,wall_seconds=after-started,worker=mem,aggregate_rss_bytes=rss,driver_phase=phase,
                            memory=host,interval_seconds=policy['interval_seconds'],observation_seconds=duration,
                            staleness_seconds=staleness,first_stop_reason=first)
                        payload=(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode()
                        if journal_bytes+len(payload)>policy['journal_bytes']:first=first or 'MONITOR_JOURNAL_CAP'
                        else:journal.write(payload);journal.flush();journal_bytes+=len(payload)
                        last_emit=now
                except FileNotFoundError:
                    if os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is None:
                        first=first or 'OBSERVATION_MISSING'
                    else:break
                except Exception as exc:first=first or 'OBSERVATION_ERROR:'+type(exc).__name__+':'+str(exc)[:256]
                if first:break
                time.sleep(policy['interval_seconds'])
        except BaseException as exc:
            first=first or 'SUPERVISOR_ERROR:'+type(exc).__name__+':'+str(exc)[:256]
        finally:
            if process is not None:
                try:code=stop_owned_group(process,identity)
                except Exception as exc:
                    cleanup_error=type(exc).__name__+':'+str(exc)[:256]
                    first=first or 'CLEANUP_ERROR:'+cleanup_error
            if progress:progress.update(point='phase_a_cleanup',worker_pid=process.pid if process else None,
                worker_exit_code=code,first_stop_reason=first,cleanup_error=cleanup_error)
    return dict(worker_exit_code=code,first_stop_reason=first,cleanup_error=cleanup_error,
        worker_peak_rss_bytes=peak_rss,worker_peak_vmsize_bytes=peak_as,
        live_aggregate_peak_rss_bytes=aggregate_peak,wall_seconds=time.monotonic()-started,
        journal_bytes=journal_bytes,identity=identity,process_reaped=process is not None and process.returncode is not None,
        log_and_journal_closed=True,retry=False)


def verify_worker(root,directory,run_root):
    root=Path(root);out=Path(run_root).resolve();directory=Path(directory).resolve()
    if directory.parent!=out or directory.name!='cost_exploration_0447' or Path(run_root).is_symlink():
        raise ValueError('PHASE_A_WORKER_PATH')
    m=read(out/'frozen_manifest.json');g=read(out/'authorization.json')
    validate_manifest(m);validate_grant(m,g,out)
    if read(out/'launch_binding.json')!=dict(manifest_digest=digest(m),authorization_digest=digest(g),
        authorization_source_sha256=file_hash(out/'authorization_source.json')) or read(out/'authorization_source.json')!=g:
        raise ValueError('PHASE_A_WORKER_GRANT_BINDING')
    parent=read(out/'execution_claim.json')['coordinator_identity']
    install_parent_guard(parent)
    b=read(directory/'task_binding.json');r=m['assigned_resources'];task=m['phase_scope']['task']
    if b!=dict(manifest_digest=digest(m),task=task,cpus=r['cost_cpu_sets'][0],threads=1,
        address_space_bytes=r['cost_address_space_bytes'],output_bytes=32*2**20):raise ValueError('PHASE_A_WORKER_BINDING')
    if set(m['source_hashes'])!=set(source_paths(root)):raise ValueError('PHASE_A_WORKER_CLOSURE')
    for p,sha in m['source_hashes'].items():
        if file_hash(safe_path(root,p))!=sha:raise ValueError('PHASE_A_WORKER_SOURCE:'+p)
    for field in ('snapshot','snapshot_receipt','df_receipt'):
        if file_hash(safe_path(root,m['input_identity'][field+'_path']))!=m['input_identity'][field+'_sha256']:
            raise ValueError('PHASE_A_WORKER_INPUT')
    if environment()!=m['environment']:raise ValueError('PHASE_A_WORKER_ENVIRONMENT')
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS) or os.environ.get('QISKIT_PARALLEL')!='FALSE':
        raise ValueError('PHASE_A_WORKER_THREADS')
    limits=install_limits(b['cpus'],b['address_space_bytes'],b['output_bytes'])
    writer=AtomicWriter(directory,byte_cap=b['output_bytes'])
    writer.write('worker_claim.json',dict(task_digest=digest(task),source_commit=m['source_commit'],
        resource_limits=limits,actual_cpus=sorted(os.sched_getaffinity(0)),threads=1,retry=False,resume=False))
    return m,task,writer


def execute_worker(root,directory,run_root):
    m,task,writer=verify_worker(root,directory,run_root)
    started=time.monotonic();trace=ResourceTraceV3(writer.write,lambda:time.monotonic()-started)
    reserve=bytearray(2**20);reason=None
    try:
        # Tighten invocation scope, preserving the registered scientific plan itself.
        effective=deepcopy(m)
        effective['plan']['caps_proposed'].update(m['phase_scope']['task_caps'])
        writer.write('effective_task_caps.json',m['phase_scope']['task_caps'])
        with trace.bind(),trace.stage('cost_task',cell_id=task['cell_id'],phase=task['phase'],replica=task['replica']):
            from .h6_matched_cost_port_v3 import cost_task
            cost_task(Path(root),effective,task,writer,Progress(writer),trace)
        for p,sha in m['source_hashes'].items():
            if file_hash(safe_path(root,p))!=sha:raise ValueError('PHASE_A_SOURCE_AFTER:'+p)
        status='TASK_COMPLETE'
    except Exception as exc:
        del reserve;reason=type(exc).__name__+':'+str(exc)[:512];status='TASK_STOP'
        try:trace.failure(exc)
        except Exception:pass
    writer.write('worker_terminal.json',dict(status=status,reason=reason,task=task,
        source_commit=m['source_commit'],memory=process_memory(os.getpid()),**flags()),terminal=True)
    return 0 if status=='TASK_COMPLETE' else 1


def admit_new_group(directory,task,source):
    """Same frozen result checks, with the truthful new 32 GiB limit receipt."""
    directory=Path(directory);term=read(directory/'worker_terminal.json');claim=read(directory/'worker_claim.json')
    binding=read(directory/'task_binding.json');result=read(directory/'cost_result.json');trajectory=read(directory/'trajectory.json')
    if (term.get('status')!='TASK_COMPLETE' or term.get('reason') is not None or
        any(digest(x.get('task'))!=digest(task) for x in (term,binding,result,trajectory)) or
        term.get('source_commit')!=source or claim.get('source_commit')!=source or
        claim.get('task_digest')!=digest(task) or claim.get('resource_limits',{}).get('rlimit_as')!=[32*2**30]*2):
        raise ValueError('PHASE_A_RESULT_IDENTITY')
    if trajectory.get('events') is not None or trajectory.get('observed_orders')!=[] or trajectory.get('finite_distribution') is not None:
        raise ValueError('PHASE_A_DETERMINISTIC_EVENTS')
    if trajectory.get('event_digest')!=digest(None):raise ValueError('PHASE_A_EVENT_DIGEST')
    keys=[]
    for row in result['rows']:
        key=(row['control'],row['axis']);keys.append(key)
        if (any(type(row.get(k)) is not type(task[k]) or row[k]!=task[k] for k in ('cell_id','phase','replica','seed'))
            or row['event_digest']!=digest(None) or row['compiler_hash']!=COMPILER_HASH
            or len(row.get('fingerprint',''))!=64 or set(row['metrics'])!={'RZ','CX','RZ_depth','CX_depth','depth','size'}
            or any(type(v) is not int or v<0 for v in row['metrics'].values())
            or read(directory/f'wrapper_{key[0]}_{key[1]}.json')!=row):raise ValueError('PHASE_A_WRAPPER_IDENTITY')
    if set(keys)!={('symmetric_directional','cosine'),('symmetric_directional','sine')} or len(keys)!=2:
        raise ValueError('PHASE_A_PAIRED_AXES')
    if task['validate']:
        v=read(directory/'validation_symmetric_directional.json')['max_error']
        if not isinstance(v,(int,float)) or not 0<=v<=1e-9:raise ValueError('PHASE_A_VALIDATION')
    if result['calls_attempted']['compile']!=2:raise ValueError('PHASE_A_COMPILE_COUNT')
    return result


def orchestrate(root,output,m,writer,fresh):
    root=Path(root);output=Path(output);r=m['assigned_resources'];status='H6_SERVER_PHASE_A_STOP';reason=None
    limits=install_limits(r['coordinator_cpu_set'],r['coordinator_address_space_bytes'],
        m['plan']['caps_proposed']['output_bytes'],hard_address_bytes=r['coordinator_hard_address_space_bytes'])
    writer.write('coordinator_limits.json',limits);report=None
    try:
        writer.write('execution_claim.json',dict(source_commit=m['source_commit'],
            coordinator_identity=process_identity(os.getpid()),phase_scope=m['phase_scope'],fresh_resources=fresh,
            phase_b_authorized=False,confirmation_authorized=False,retry=False))
        task=m['phase_scope']['task'];directory=output/'cost_exploration_0447';directory.mkdir(exist_ok=False)
        AtomicWriter(directory,byte_cap=32*2**20).write('task_binding.json',dict(manifest_digest=digest(m),
            task=task,cpus=r['cost_cpu_sets'][0],threads=1,address_space_bytes=r['cost_address_space_bytes'],output_bytes=32*2**20))
        command=[sys.executable,'-I','-P','-B',str(root/RUNNER),'--worker',str(directory),'--run-root',str(output)]
        report=supervise_one(command,directory,m,progress=Progress(writer))
        writer.write('phase_a_resource_summary.json',report)
        if report['first_stop_reason'] or report['worker_exit_code']!=0:
            raise RuntimeError(report['first_stop_reason'] or 'WORKER_EXIT:'+str(report['worker_exit_code']))
        result=admit_new_group(directory,task,m['source_commit'])
        final_memory=read(directory/'worker_terminal.json').get('memory',{})
        report['worker_peak_rss_bytes']=max(report['worker_peak_rss_bytes'],final_memory.get('VmHWM_bytes',0))
        report['worker_peak_vmsize_bytes']=max(report['worker_peak_vmsize_bytes'],final_memory.get('VmPeak_bytes',0))
        writer.write('phase_a_resource_final_summary.json',report)
        verify_bindings(root,m)
        begin={};transpile=[]
        for path in sorted(directory.glob('diagnostic_*.json')):
            d=read(path)
            if d.get('context',{}).get('stage')=='transpile':
                axis=d['context'].get('axis')
                if d['kind']=='begin':begin[axis]=d['elapsed_seconds']
                elif d['kind']=='end' and axis in begin:
                    transpile.append(dict(axis=axis,seconds=d['elapsed_seconds']-begin[axis],
                        input_instructions=d['context'].get('instructions')))
        writer.write('phase_a_result_receipt.json',dict(index=447,task_digest=digest(task),wrappers=len(result['rows']),
            transpile_timings=transpile,wrapper_metrics=[dict(axis=x['axis'],metrics=x['metrics']) for x in result['rows']],
            cost_result_sha256=file_hash(directory/'cost_result.json'),reused_groups=449,completed_groups_now=450,
            remaining_groups=142,selection_not_performed=True,independent_confirmation_not_started=True))
        status='H6_SERVER_PHASE_A_COMPLETE_MANDATORY_STOP'
    except Exception as exc:reason=type(exc).__name__+':'+str(exc)[:512]
    inventory={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*')
        if p.is_file() and '.runtime_cache' not in p.parts}
    writer.write('execution_inventory.json',dict(source_commit=m['source_commit'],file_hashes=inventory,
        output_bytes=output_size(output),science_source=m['science_source'],environment=m['environment'],
        phase_a_only=True,resource_summary=report),terminal=True)
    writer.write('execution_terminal.json',dict(status=status,reason=reason,source_commit=m['source_commit'],
        phase_b_authorized=False,confirmation_authorized=False,retry=False,resume=False,**flags()),terminal=True)
    return 0 if reason is None else 1

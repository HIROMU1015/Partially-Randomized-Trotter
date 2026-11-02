"""Independent32 only; frozen monitor/cost port, no selection after launch."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .ax2a_preparation import digest
from .ax2b_limits import output_size
from .ax2b_supplement_records_v1 import AtomicWriter
from .h6_matched_execution_v2 import install_limits, worker_environment, process_tree_rss, Progress, resource_maps
from .h6_matched_execution_v3 import ResourceTraceV3
from .h6_matched_accounting_v1 import order_diagnostic
from .h6_matched_reuse_v3 import read, file_hash
from .h6_parallel_execution_v1 import admit_parallel_group
from .h6_confirmation_contract_v1 import RUNNER, validate_grant, validate_plan, worker_scope, sources, slots
from .h6_server_resources_v1 import (THREAD_KEYS, host_memory, memory_reason, process_memory,
    process_identity, stop_owned_group, install_parent_guard)
from .h6_server_contract_v1 import safe_path, environment

EXIT_PENDING_SECONDS = 1.
PF_EXITING = 0x00000004

def owned_exit_flag(child, identity):
    """Check kernel exit state AND stable owned unreaped group identity."""
    raw=Path(f'/proc/{child.pid}/stat').read_text()
    fields=raw[raw.rfind(')')+2:].split()
    if (child.returncode is not None or identity is None
        or identity['pid']!=child.pid or int(fields[19])!=identity['start_ticks']
        or int(fields[2])!=child.pid or int(fields[3])!=child.pid):
        raise ValueError('EXIT_PENDING_OWNERSHIP_MISMATCH')
    return bool(int(fields[6])&PF_EXITING)



def retain_exiting_memory(v, r, now):
    """One nonblocking exit deadline, never reset by another missing sample.

    The caller permits this ONLY after kernel PF_EXITING and identity checks.
    Keep observed memory or the approved slot reservation, never invent zero.
    No slot is retired/admitted until WEXITED; live missing data still STOPs.
    """
    if 'exit_pending_since' not in v:v['exit_pending_since']=now
    age=now-v['exit_pending_since']
    if age>EXIT_PENDING_SECONDS:raise ValueError('EXIT_PENDING_MEMORY_TIMEOUT:'+str(v['process'].pid))
    if 'last_memory' in v:
        memory=dict(v['last_memory']);kind='exit_pending_retained'
    else:
        memory=dict(VmRSS_bytes=r['worker_rss_stop_bytes'],VmSize_bytes=r['cost_address_space_bytes'],
            VmHWM_bytes=r['worker_rss_stop_bytes'],VmPeak_bytes=r['cost_address_space_bytes'])
        kind='exit_pending_reserved'
    return memory,dict(kind=kind,kernel_exiting=True,exit_pending_seconds=age,
        maximum_exit_pending_seconds=EXIT_PENDING_SECONDS,live_sample=False)



def supervise(root,out,m,*,command=None,admission=None):
    """Only owned unreaped leaders are killed; later cleanups survive an error.

    Fixed task/seed order, exclusive slots/dirs, first STOP preserved, no retry.
    Result ordering is original index order, independent of completion order.
    """
    root=Path(root);out=Path(out);r=m['proposed_resources'];policy=m['observation_policy']
    caps=m['base_manifest']['plan']['caps_proposed'];pending=list(m['registrations'])
    active={};completed=[];dispatch=[];retired=[];errors=[];first=None;peak=0;transitions=[]
    begun=time.monotonic();last_sample=begun;last_emit=0.;written=0
    baseline=host_memory();max_slots=slots(r)
    if max_slots<1:raise ValueError('PARALLEL_NO_MEMORY_SLOT')
    journal_path=out/'resource_observations.jsonl'
    observation_subject=None
    def retire(v):
        nonlocal first
        try:
            code=stop_owned_group(v['process'],v['identity'])
            retired.append(dict(index=v['row']['index'],pid=v['process'].pid,exit_code=code,reaped=True))
            return code
        except Exception as exc:
            message=type(exc).__name__+':'+str(exc)[:256];errors.append(message)
            first=first or 'CLEANUP_ERROR:'+message
            return None
    try:
        with journal_path.open('xb') as journal:
            while pending or active:
                if (out/'STOP_REQUEST').exists():first=first or 'USER_STOP_REQUEST'
                if first:break
                while pending and len(active)<max_slots:
                    row=pending.pop(0);slot=next(i for i in range(max_slots) if i not in active)
                    directory=out/f'cost_confirmation_{row["index"]:04d}';directory.mkdir(exist_ok=False)
                    cache=directory/'.runtime_cache';cache.mkdir();temp=cache/'tmp';temp.mkdir()
                    env=worker_environment(1,cache);env.update(TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),XDG_CACHE_HOME=str(cache/'xdg'))
                    AtomicWriter(directory,byte_cap=32*2**20).write('task_binding.json',dict(
                        manifest_digest=digest(m),index=row['index'],task=row['task'],slot=slot,
                        cpus=r['cost_cpu_sets'][slot],threads=1,address_space_bytes=r['cost_address_space_bytes']))
                    if output_size(out)+max_slots*32*2**20+65536>caps['output_bytes']:
                        first='PARALLEL_OUTPUT_RESERVATION';break
                    argv=command(row,directory,slot) if command else [sys.executable,'-I','-P','-B',str(root/RUNNER),
                        '--worker',str(directory),'--run-root',str(out)]
                    with (directory/'worker.log').open('xb') as log:
                        child=subprocess.Popen(argv,env=env,cwd=root,stdin=subprocess.DEVNULL,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    # Keep the Popen object even if identity acquisition fails.
                    v=dict(process=child,identity=None,row=row,path=directory,slot=slot);active[slot]=v
                    v['identity']=process_identity(child.pid)
                    dispatch.append(dict(index=row['index'],pid=child.pid,slot=slot,task_digest=digest(row['task'])))
                if first:break
                before=time.monotonic();samples=[]
                for slot,v in list(active.items()):
                    child=v['process']
                    observation_subject=dict(role='worker',index=v['row']['index'],pid=child.pid,slot=slot)
                    if os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is not None:
                        code=retire(v);del active[slot]
                        if code!=0:first=first or 'WORKER_EXIT:'+str(code)
                        else:
                            if admission:admission(v['path'],v['row']['task'])
                            completed.append(v['row']['index'])
                        if 'exit_pending_since' in v:transitions.append(dict(index=v['row']['index'],
                            pid=child.pid,state='exit_confirmed',exit_code=code))
                        continue
                    current=process_identity(child.pid)
                    if current['start_ticks']!=v['identity']['start_ticks']:raise ValueError('PARALLEL_WORKER_IDENTITY')
                    observation=dict(kind='live',live_sample=True)
                    try:
                        mem=process_memory(child.pid);v['last_memory']=dict(mem)
                        v.pop('exit_pending_since',None)
                    except (FileNotFoundError,ValueError) as exc:
                        if isinstance(exc,ValueError) and str(exc)!='PROCESS_MEMORY_MISSING':raise
                        if os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is not None:continue
                        if not owned_exit_flag(child,v['identity']):raise
                        initial='exit_pending_since' not in v
                        mem,observation=retain_exiting_memory(v,r,time.monotonic())
                        if initial:transitions.append(dict(index=v['row']['index'],pid=child.pid,
                            state='kernel_exit_pending',retained_memory_kind=observation['kind']))
                    if mem['VmSize_bytes']>r['cost_address_space_bytes']:first=first or 'WORKER_AS_CAP'
                    if mem['VmRSS_bytes']>r['worker_rss_stop_bytes']:first=first or 'WORKER_RSS_CAP'
                    latest=sorted(v['path'].glob('diagnostic_*.json'))
                    phase=read(latest[-1]).get('context') if latest else {'stage':'startup'}
                    samples.append(dict(index=v['row']['index'],slot=slot,pid=child.pid,memory=mem,
                        phase=phase,memory_observation=observation))
                observation_subject=dict(role='host',pid=os.getpid())
                host=host_memory();rss=process_tree_rss(os.getpid());peak=max(peak,rss)
                observation_subject=dict(role='coordinator',pid=os.getpid())
                coordinator=process_memory(os.getpid())
                # Aggregate includes retained/reserved dying-worker memory;
                # kernel process_tree_rss may omit a task whose mm is gone.
                rss=max(rss,coordinator['VmRSS_bytes']+sum(x['memory']['VmRSS_bytes'] for x in samples))
                peak=max(peak,rss);after=time.monotonic()
                first=first or memory_reason(host,baseline,policy)
                if after-before>policy['maximum_observation_seconds'] or before-last_sample>policy['maximum_observation_seconds']:
                    first=first or 'OBSERVATION_STALE'
                staleness=before-last_sample;last_sample=after
                if rss>r['aggregate_rss_bytes']:first=first or 'AGGREGATE_RSS_CAP'
                if coordinator['VmRSS_bytes']>r['coordinator_rss_reserve_bytes']:first=first or 'COORDINATOR_RSS_CAP'
                if output_size(out)>caps['output_bytes']-65536:first=first or 'AGGREGATE_OUTPUT_CAP'
                if sum(p.stat().st_size for p in out.glob('cost_*/worker.log'))>caps['log_bytes']:first=first or 'AGGREGATE_LOG_CAP'
                if after-last_emit>=policy['journal_emit_seconds'] or first:
                    row=dict(wall_seconds=after-begun,workers=samples,coordinator=coordinator,aggregate_rss_bytes=rss,
                        memory=host,observation_seconds=after-before,staleness_seconds=staleness,
                        first_stop_reason=first,active_slots=len(active),pending=len(pending))
                    payload=(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode()
                    if written+len(payload)>policy['journal_bytes']:first=first or 'MONITOR_JOURNAL_CAP'
                    else:journal.write(payload);journal.flush();written+=len(payload)
                    last_emit=after
                if first:break
                if active:time.sleep(policy['interval_seconds'])
    except BaseException as exc:
        first=first or 'SUPERVISOR_ERROR:'+type(exc).__name__+':'+str(exc)[:256]
        try:
            AtomicWriter(out,byte_cap=caps['output_bytes']).write('supervisor_failure.json',dict(
                first_stop_reason=first,error_type=type(exc).__name__,message=str(exc)[:256],
                observation_subject=observation_subject,observation_transitions=transitions[-16:]),terminal=True)
        except Exception:pass
    finally:
        for v in active.values():retire(v)
    return dict(completed_indices=sorted(completed),dispatched=dispatch,retired=retired,pending=len(pending),
        first_stop_reason=first,cleanup_errors=errors,all_dispatched_reaped=len(retired)==len(dispatch) and not errors,
        aggregate_peak_rss_bytes=peak,wall_seconds=time.monotonic()-begun,journal_bytes=written,
        ownership_cleanup_verified=len(retired)==len(dispatch) and not errors,
        log_and_journal_closed=True,retry=False,exit_observation_transitions=transitions)



def worker(root,directory,run_root):
    root=Path(root);out=Path(run_root).resolve();directory=Path(directory).resolve()
    if directory.parent!=out or any(p.is_symlink() for p in (Path(run_root),Path(directory))):raise ValueError('PARALLEL_WORKER_PATH')
    m=read(out/'frozen_manifest.json');g=read(out/'authorization.json');validate_grant(m,g)
    worker_scope(root,out,m)
    if (read(out/'authorization_source.json')!=g or read(out/'launch_binding.json')!=dict(
        manifest_digest=digest(m),authorization_digest=digest(g),authorization_source_sha256=file_hash(out/'authorization_source.json'))):
        raise ValueError('PARALLEL_WORKER_GRANT')
    install_parent_guard(read(out/'execution_claim.json')['coordinator_identity'])
    b=read(directory/'task_binding.json');r=m['proposed_resources'];slot=b['slot']
    if type(slot) is not int or not 0<=slot<slots(r):raise ValueError('PARALLEL_SLOT')
    expected=dict(manifest_digest=digest(m),index=b['index'],task=b['task'],slot=slot,
        cpus=r['cost_cpu_sets'][slot],threads=1,address_space_bytes=r['cost_address_space_bytes'])
    if (b!=expected or {'index':b['index'],'task':b['task']} not in m['registrations']
        or directory.name!=f'cost_confirmation_{b["index"]:04d}'):
        raise ValueError('PARALLEL_WORKER_TASK')
    if set(m['source_hashes'])!=set(sources(root)):raise ValueError('PARALLEL_WORKER_CLOSURE')
    for p,sha in m['source_hashes'].items():
        if file_hash(safe_path(root,p))!=sha:raise ValueError('PARALLEL_WORKER_SOURCE')
    base=m['base_manifest']
    for p,sha in base['input_identity']['parent_raw_hashes'].items():
        if file_hash(safe_path(root,p))!=sha:raise ValueError('PARALLEL_WORKER_INPUT')
    if environment()!=base['environment']:raise ValueError('PARALLEL_WORKER_ENVIRONMENT')
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS) or os.environ.get('QISKIT_PARALLEL')!='FALSE':
        raise ValueError('PARALLEL_WORKER_THREADS')
    limits=install_limits(b['cpus'],r['cost_address_space_bytes'],32*2**20)
    writer=AtomicWriter(directory,byte_cap=32*2**20)
    writer.write('worker_claim.json',dict(source_commit=m['source_commit'],task_digest=digest(b['task']),
        resource_limits=limits,actual_cpus=sorted(os.sched_getaffinity(0)),threads=1,retry=False,resume=False))
    started=time.monotonic();trace=ResourceTraceV3(writer.write,lambda:time.monotonic()-started)
    reserve=bytearray(2**20);reason=None;status='TASK_COMPLETE'
    try:
        with trace.bind(),trace.stage('cost_task',cell_id=b['task']['cell_id'],phase='confirmation',replica=b['task']['replica']):
            from .h6_roundoff_cost_port_v1 import cost_task
            from .h6_roundoff_preparation_v1 import read_preparations,use_preparations
            from .h6_parallel_contract_v1 import roundoff_binding
            roundoff_binding(root,m)
            ref=m['roundoff_preparation_binding']
            preparations,archive=read_preparations(ref['archive_path'],ref['archive_sha256'])
            if (archive['binding']['input_identity']!=base['input_identity'] or
                archive['binding']['profile_sha256']!=base['profile']['sha256'] or
                archive['binding']['compiler']!=base['plan']['compiler_proposed']):raise ValueError('ROUNDING_WORKER_BINDING')
            effective=deepcopy(base)
            effective['plan']['caps_proposed']['compile']=4 if b['task']['ordinary'] else 2
            with use_preparations(preparations):
                cost_task(root,effective,b['task'],writer,Progress(writer),trace)
        for p,sha in m['source_hashes'].items():
            if file_hash(safe_path(root,p))!=sha:raise ValueError('PARALLEL_WORKER_SOURCE_AFTER')
    except Exception as exc:
        del reserve;reason=type(exc).__name__+':'+str(exc)[:512];status='TASK_STOP'
        try:trace.failure(exc)
        except Exception:pass
    writer.write('worker_terminal.json',dict(status=status,reason=reason,task=b['task'],source_commit=m['source_commit'],
        memory=process_memory(os.getpid()),mandatory_stop=True,next_stage_authorized=False),terminal=True)
    return 0 if reason is None else 1




def admit_confirmation_group(directory, task, *, source):
    if (task['phase'] != 'confirmation' or task['cell']['method'] != 'B2'
            or task['validate'] is not False or task['ordinary'] is not False):
        raise ValueError('CONFIRMATION_ADMISSION_SCOPE')
    result = admit_parallel_group(directory, task, source=source)
    expected = dict(compile=2, control_probe=0, trajectory=1,
        occurrence=task['cell']['q'], primitive=0, reference_matvec=0)
    if result['calls_attempted'] != expected:
        raise ValueError('CONFIRMATION_TASK_CALLS')
    return result


def complete_confirmation(m, groups):
    registrations = m['registrations']
    if set(groups) != {x['index'] for x in registrations}:
        raise ValueError('CONFIRMATION_INCOMPLETE_GROUPS')
    rows = []
    counts = {}
    for registration in registrations:
        i, task = registration['index'], registration['task']
        group = groups[i]
        if digest(group['task']) != digest(task):
            raise ValueError('CONFIRMATION_TASK_CHANGED')
        keys = [(r['control'], r['axis']) for r in group['rows']]
        if len(keys) != 2 or set(keys) != {('symmetric_directional','cosine'),('symmetric_directional','sine')}:
            raise ValueError('CONFIRMATION_PAIRED_AXES')
        counts.setdefault(task['cell_id'], set()).add(task['replica'])
        rows.extend(group['rows'])
    if (set(counts) != set(m['exploration_binding']['selection']['selected_ids'])
            or any(v != set(range(32)) for v in counts.values())
            or len(rows) != 64*len(counts)):
        raise ValueError('CONFIRMATION_COMPLETE32')
    return rows


def orchestrate(root, out, m, writer, fresh):
    from .h6_confirmation_contract_v1 import private_path
    r = m['proposed_resources']; base = m['base_manifest']; report = None; reason = None
    limits = install_limits(r['coordinator_cpu_set'], r['coordinator_address_space_bytes'],
        base['plan']['caps_proposed']['output_bytes'], hard_address_bytes=r['coordinator_hard_address_space_bytes'])
    writer.write('execution_claim.json', dict(coordinator_identity=process_identity(os.getpid()), resource_limits=limits, fresh=fresh))
    try:
        report = supervise(root, out, m, admission=lambda p,t: admit_confirmation_group(p,t,source=m['source_commit']))
        writer.write('confirmation_resource_summary.json', report)
        if report['first_stop_reason']:
            raise RuntimeError(report['first_stop_reason'])
        if report['completed_indices'] != [x['index'] for x in m['registrations']]:
            raise ValueError('CONFIRMATION_INCOMPLETE')
        groups = {x['index']: admit_confirmation_group(Path(out)/f'cost_confirmation_{x["index"]:04d}',
            x['task'], source=m['source_commit']) for x in m['registrations']}
        rows = complete_confirmation(m, groups)
        parent = read(Path(root)/'artifacts/resource_applicability/track_a_h6_exploration_summary_v1/2026-10-12/integrated_exploration_v1.json')
        deterministic = []
        original = base['reuse_registry']['original_tasks']
        for row in parent['groups']:
            if original[row['index']]['cell']['method'] != 'B2':
                deterministic += read(private_path(row['cost_result_path']))['rows']
        signals = read(Path(root)/base['reuse_registry']['original_raw_path']/'signal/signal_summary.json')['signals']
        writer.write('confirmation_resource_map.json', resource_maps(signals, deterministic+rows, base['plan'], 'confirmation'))
        calls = dict(m['cumulative_accounting']['known_calls'])
        for group in groups.values():
            for k,v in group['calls_attempted'].items():
                calls[k] += v
        accounting = m['cumulative_accounting']
        if any(v+accounting['reservations'].get(k,0)>accounting['proposed_caps'][k] for k,v in calls.items()):
            raise ValueError('CONFIRMATION_CUMULATIVE_CALL_CAP')
        by_cell = {}
        for registration in m['registrations']:
            tr = read(Path(out)/f'cost_confirmation_{registration["index"]:04d}'/'trajectory.json')
            by_cell.setdefault(tr['task']['cell_id'], []).append(tr)
        order_rows = [dict(cell_id=cid, phase='confirmation', **order_diagnostic(trs[0]['finite_distribution'],
            [k for tr in trs for k in tr['observed_orders']], R=trs[0]['task']['cell']['R'], trajectories=32))
            for cid,trs in sorted(by_cell.items())]
        writer.write('confirmation_order_diagnostics.json', dict(rows=order_rows, population_cost_tail_bound=None,
            exploration_samples_pooled=False, additional_sampling_authorized=False))
        writer.write('confirmation_coverage.json', dict(exploration_groups=592, confirmation_groups=len(groups),
            confirmation_wrappers=len(rows), selected_ids=m['exploration_binding']['selection']['selected_ids'],
            unconfirmed_B2_ids=sorted(s['cell']['id'] for s in signals if s['cell']['method']=='B2'
                and s['cell']['id'] not in by_cell), full_grid_population_best_method_certified=False,
            exploration_B2_in_confirmation_mean=False, deterministic_baseline_reused=True))
        writer.write('aggregate_calls.json', dict(known_calls=calls, historical_unknown_actual_totals=None,
            reservations=accounting['reservations'], reservations_are_not_actual_totals=True,
            failed_calls_refunded=False, budget_reset=False))
        # The lock has intentionally been consumed. Recheck all other bindings.
        validate_plan(root,m)
        writer.write('confirmation_complete_receipt.json', dict(group_count=len(groups), wrapper_count=len(rows),
            confirmation_rows_digest=digest(rows), selection_digest=digest(m['exploration_binding']['selection']),
            population_winner='UNDETERMINED_POPULATION_MEAN', formal_accuracy_certified=False,
            source_commit=m['source_commit'], next_stage_authorized=False))
    except Exception as exc:
        reason = type(exc).__name__+':'+str(exc)[:512]
    writer.write('execution_terminal.json', dict(status='H6_CONFIRMATION_COMPLETE_MANDATORY_STOP' if reason is None else 'H6_CONFIRMATION_STOP',
        reason=reason, source_commit=m['source_commit'], resource_summary=report,
        mandatory_stop=True, H8_authorized=False, next_stage_authorized=False, retry=False, resume=False,
        population_winner='UNDETERMINED_POPULATION_MEAN'), terminal=True)
    return 0 if reason is None else 1

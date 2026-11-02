"""New cost-only evidence supplement. One-shot; no scientific wall deadline."""
from __future__ import annotations
import os
import resource
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from .ax2a_preparation import digest
from .ax2b_limits import output_size
from .ax2b_supplement_records_v1 import AtomicWriter
from .h6_matched_execution_v2 import (bounded_json, install_limits, install_coordinator_limits,
    worker_environment, process_tree_rss, Progress, resource_maps)
from .h6_matched_contract_v3 import (RUNNER, file_hash, safe_path, source_paths, validate_manifest,
    validate_grant, verify_sources, verify_parent, environment)
from .h6_matched_reuse_v3 import (read, admit_group, complete_exploration, verify_evidence, OLD_RAW)
from .h6_matched_accounting_v1 import tasks, confirmation_ids, order_diagnostic
from .ax2b_diagnostics_v4 import ResourceTrace


class ResourceTraceV3(ResourceTrace):
    def event(self,kind,**fields):
        super().event(kind,rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)),**fields)


def supervise_batch(root, output, manifest, registrations, *, signal_stage=False, progress=None):
    """Launch each registered task once, retire its entire process group.

    A partial batch stops; completed artifacts remain immutable. No resume or
    automatic retry. Memory/output/cancellation guards are independent of time.
    """
    output=Path(output);r=manifest['assigned_resources'];caps=manifest['plan']['caps_proposed']
    if caps['phase_wall_seconds'] is not None or caps['total_wall_seconds'] is not None:
        raise ValueError('MATCHED_NO_WALL_CAP_POLICY')
    active={};done=[];pending=list(registrations);peak=0;failure=None
    concurrency=1 if signal_stage else r['cost_workers']
    cache=output/'.runtime_cache';cache.mkdir(exist_ok=True)
    try:
        if (output/'STOP_REQUEST').exists():
            raise RuntimeError('USER_STOP_REQUEST')
        while pending or active:
            while pending and len(active)<concurrency and failure is None:
                index,task=pending.pop(0)
                directory=output/('signal' if signal_stage else f'cost_{task["phase"]}_{index:04d}')
                directory.mkdir(exist_ok=False)
                slot=next(i for i in range(concurrency) if i not in [v['slot'] for v in active.values()])
                cpus=r['assigned_cpus'] if signal_stage else r['cost_cpu_sets'][slot]
                threads=r['signal_threads'] if signal_stage else r['cost_threads']
                local_cap=64*2**20 if signal_stage else 32*2**20
                # Reserve enough for simultaneously bounded private writers and terminal records.
                if output_size(output)+concurrency*local_cap+65536>caps['output_bytes']:
                    raise RuntimeError('MATCHED_OUTPUT_RESERVATION_CAP')
                binding=dict(manifest_digest=digest(manifest),task=task,cpus=cpus,
                             address_space_bytes=r['signal_address_space_bytes'] if signal_stage else r['cost_address_space_bytes'],
                             output_bytes=local_cap,threads=threads)
                writer=AtomicWriter(directory,byte_cap=local_cap)
                writer.write('task_binding.json',binding)
                logfile=(directory/'worker.log').open('xb')
                command=[sys.executable,str(Path(root)/RUNNER),'--worker',str(directory),'--run-root',str(output)]
                child=subprocess.Popen(command,cwd=root,env=worker_environment(threads,cache),
                    stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
                logfile.close();active[child.pid]=dict(process=child,path=directory,slot=slot,task=task)
                if progress:progress.update(point='task_dispatched',path=directory.name,task=task)
            if (output/'STOP_REQUEST').exists():
                failure='USER_STOP_REQUEST'
            rss=process_tree_rss(os.getpid());peak=max(peak,rss)
            if rss>r['aggregate_rss_bytes']:failure='AGGREGATE_RSS_CAP'
            if output_size(output)>caps['output_bytes']-65536:failure='AGGREGATE_OUTPUT_CAP'
            logs=sum(p.stat().st_size for p in output.rglob('worker.log'))
            if logs>caps['log_bytes']:failure='AGGREGATE_LOG_CAP'
            for pid,v in list(active.items()):
                code=v['process'].poll()
                if code is None:
                    continue
                # Kill descendants even after the leader exited.
                try:os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError:pass
                active.pop(pid)
                terminal_path=v['path']/'worker_terminal.json'
                terminal=bounded_json(terminal_path) if terminal_path.exists() else None
                if code!=0 or not terminal or terminal.get('status')!='TASK_COMPLETE':
                    failure=(terminal or {}).get('reason') or 'WORKER_EXIT:'+str(code)
                else:
                    done.append(v['path'])
                    if progress:progress.update(point='task_completed',path=v['path'].name,completed=len(done))
            if failure:
                raise RuntimeError(failure)
            if active:
                time.sleep(.1)
    finally:
        for pid,v in active.items():
            try:os.killpg(pid,signal.SIGKILL)
            except ProcessLookupError:pass
            v['process'].wait()
        if progress:
            progress.update(point='batch_finally',live_aggregate_peak_rss_bytes=peak,completed=len(done),
                killed=list(active),pending_count=len(pending),failure=failure)
    return done,peak


def verify_worker(root,directory,run_root):
    root=Path(root);directory=Path(directory).resolve();run_root=Path(run_root).resolve()
    if directory.parent!=run_root or run_root.is_symlink():raise ValueError('SUPPLEMENT_WORKER_PATH')
    m=read(run_root/'frozen_manifest.json');g=read(run_root/'authorization.json')
    validate_manifest(m);validate_grant(m,g,run_root)
    if (read(run_root/'authorization_source.json')!=g or read(run_root/'launch_binding.json')!={
        'manifest_digest':digest(m),'authorization_digest':digest(g),
        'authorization_source_sha256':file_hash(run_root/'authorization_source.json')}):
        raise ValueError('SUPPLEMENT_WORKER_GRANT')
    b=read(directory/'task_binding.json');task=b['task'];r=m['assigned_resources']
    if b['manifest_digest']!=digest(m) or task.get('phase') not in ('exploration','confirmation'):
        raise ValueError('SUPPLEMENT_COST_ONLY')
    registration=read(run_root/(task['phase']+'_tasks.json'))['tasks']
    try:index=registration.index(task)
    except ValueError:raise ValueError('SUPPLEMENT_UNREGISTERED_TASK')
    if directory.name!=f'cost_{task["phase"]}_{index:04d}':raise ValueError('SUPPLEMENT_ORIGINAL_INDEX')
    if task['phase']=='exploration' and index not in {x['index'] for x in m['reuse_registry']['missing']}:
        raise ValueError('SUPPLEMENT_RECOMPUTATION_FORBIDDEN')
    if set(m['source_hashes'])!=set(source_paths(root)):raise ValueError('SUPPLEMENT_WORKER_CLOSURE')
    for p,sha in m['source_hashes'].items():
        if file_hash(safe_path(root,p))!=sha:raise ValueError('SUPPLEMENT_WORKER_SOURCE:'+p)
    for field in ('snapshot','snapshot_receipt','df_receipt'):
        if file_hash(safe_path(root,m['input_identity'][field+'_path']))!=m['input_identity'][field+'_sha256']:
            raise ValueError('SUPPLEMENT_WORKER_INPUT')
    if environment()!=m['environment']:raise ValueError('SUPPLEMENT_WORKER_ENVIRONMENT')
    if (b['cpus'] not in r['cost_cpu_sets'] or b['threads']!=r['cost_threads'] or
        b['address_space_bytes']!=r['cost_address_space_bytes'] or b['output_bytes']!=32*2**20):
        raise ValueError('SUPPLEMENT_WORKER_RESOURCES')
    for key in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):
        if os.environ.get(key)!='1':raise ValueError('SUPPLEMENT_WORKER_THREADS')
    if os.environ.get('QISKIT_PARALLEL')!='FALSE':raise ValueError('SUPPLEMENT_QISKIT_PARALLEL')
    limits=install_limits(b['cpus'],b['address_space_bytes'],b['output_bytes'])
    writer=AtomicWriter(directory,byte_cap=b['output_bytes'])
    writer.write('worker_claim.json',dict(task_digest=digest(task),source_commit=m['source_commit'],
        resource_limits=limits,actual_cpus=sorted(os.sched_getaffinity(0)),threads=1,retry=False,resume=False))
    return m,task,writer


def execute_worker(root,directory,run_root):
    m,task,writer=verify_worker(root,directory,run_root)
    started=time.monotonic();trace=ResourceTraceV3(writer.write,lambda:time.monotonic()-started)
    reserve=bytearray(2**20)
    try:
        with trace.bind(),trace.stage('cost_task',cell_id=task['cell_id'],phase=task['phase'],replica=task['replica']):
            from .h6_matched_cost_port_v3 import cost_task
            cost_task(Path(root),m,task,writer,Progress(writer),trace)
        for p,sha in m['source_hashes'].items():
            if file_hash(safe_path(root,p))!=sha:raise ValueError('SUPPLEMENT_SOURCE_AFTER')
        for field in ('snapshot','snapshot_receipt','df_receipt'):
            if file_hash(safe_path(root,m['input_identity'][field+'_path']))!=m['input_identity'][field+'_sha256']:
                raise ValueError('SUPPLEMENT_INPUT_AFTER')
        status,reason='TASK_COMPLETE',None
    except Exception as exc:
        del reserve
        reason=type(exc).__name__+':'+str(exc)[:512]
        try:trace.failure(exc)
        except Exception:pass  # Terminal still required if diagnostics cannot be allocated.
        status='TASK_STOP'
    writer.write('worker_terminal.json',dict(status=status,reason=reason,task=task,
        source_commit=m['source_commit'],mandatory_stop=True,next_stage_authorized=False,
        H6_status='H6_NOT_AUTHORIZED',contract_status='DRAFT_NOT_AUTHORIZATION'),terminal=True)
    return 0 if status=='TASK_COMPLETE' else 1


def combined_call_counts(root,registry,new_paths):
    counts=dict(registry['original_signal_calls'])
    old=[Path(root)/x['directory'] for x in registry['accepted']]
    for path in old+list(new_paths):
        for k,v in read(path/'cost_result.json')['calls_attempted'].items():counts[k]+=v
    return counts


def orchestrate(root,output,m,writer):
    root=Path(root);output=Path(output);p=m['plan'];registry=m['reuse_registry']
    progress=Progress(writer);peak=0;new_paths=[];selected=None
    limits=install_coordinator_limits(m['assigned_resources'],p['caps_proposed']['output_bytes'])
    writer.write('coordinator_limits.json',limits)
    try:
        writer.write('execution_claim.json',dict(source_commit=m['source_commit'],reuse_registry_digest=digest(registry),retry=False,resume=False))
        writer.write('reuse_binding.json',dict(original_source=registry['source_commit'],original_result=registry['result_commit'],
            signals_reused=92,groups_reused=449,wrappers_reused=902,old_status_unchanged=True))
        signals=read(root/OLD_RAW/'signal/signal_summary.json')['signals']
        registered=registry['original_tasks'];groups={}
        for x in registry['accepted']:groups[x['index']]=admit_group(root/x['directory'],registered[x['index']])
        writer.write('exploration_tasks.json',dict(tasks=registered,missing=registry['missing'],accepted_indices=sorted(groups)))
        first=registry['missing_first_index']
        missing={x['index']:x['task'] for x in registry['missing']}
        if missing[first]['cell_id']!='B1_p19_q8_4th':raise ValueError('SUPPLEMENT_FIRST_TASK')
        # This is its one registered acquisition, not an extra trial/retry.
        for registrations in ([(first,missing[first])],[(i,t) for i,t in sorted(missing.items()) if i!=first]):
            paths,rss=supervise_batch(root,output,m,registrations,progress=progress)
            peak=max(peak,rss);new_paths+=paths
            for path in paths:
                i=int(path.name.rsplit('_',1)[1])
                if i in groups:raise ValueError('SUPPLEMENT_DUPLICATE_GROUP')
                groups[i]=admit_group(path,registered[i],source=m['source_commit'])
        rows=complete_exploration(registered,groups)
        if len(rows)!=1188:raise ValueError('SUPPLEMENT_EXPLORATION_WRAPPERS')
        writer.write('integrated_exploration.json',dict(groups=[dict(index=i,task_digest=digest(t),
            cost_result_path=str((root/next(x['directory'] for x in registry['accepted'] if x['index']==i)/'cost_result.json').relative_to(root)) if i not in missing else
                str((output/f'cost_exploration_{i:04d}'/'cost_result.json').relative_to(root)),
            cost_result_sha256=file_hash(root/next(x['directory'] for x in registry['accepted'] if x['index']==i)/'cost_result.json') if i not in missing else
                file_hash(output/f'cost_exploration_{i:04d}'/'cost_result.json'),
            source_commit=registry['source_commit'] if i not in missing else m['source_commit']) for i,t in enumerate(registered)],
            group_count=592,wrapper_count=1188,complete_before_selection=True))
        writer.write('exploration_resource_map.json',resource_maps(signals,rows,p,'exploration'))
        selected=confirmation_ids(signals,rows,p);confirm=tasks(signals,p,'confirmation',selected)
        if len(confirm)>256:raise ValueError('SUPPLEMENT_CONFIRMATION_CAP')
        writer.write('confirmation_tasks.json',dict(tasks=confirm,selected=selected,
            exploration_binding_sha256=file_hash(output/'integrated_exploration.json'),
            exploration_rows_digest=digest(rows),rule='original top2 per epsilon, union<=8, independent32, no pooling'))
        paths,rss=supervise_batch(root,output,m,list(enumerate(confirm)),progress=progress)
        peak=max(peak,rss);new_paths+=paths
        confirmation=[r for i,path in enumerate(paths) for r in admit_group(path,confirm[int(path.name.rsplit('_',1)[1])],source=m['source_commit'])['rows']]
        if len(paths)!=len(confirm):raise ValueError('SUPPLEMENT_CONFIRMATION_COVERAGE')
        deterministic=[r for r in rows if next(s for s in signals if s['cell']['id']==r['cell_id'])['cell']['method']!='B2']
        writer.write('confirmation_resource_map.json',resource_maps(signals,deterministic+confirmation,p,'confirmation'))
        calls=combined_call_counts(root,registry,new_paths)
        if any(v>p['caps_proposed'][k] for k,v in calls.items()):raise ValueError('SUPPLEMENT_COMPLETED_CALL_CAP')
        new_wrapper_count=sum(len(read(path/'cost_result.json')['rows']) for path in new_paths)
        if new_wrapper_count>798 or 902+new_wrapper_count>1700:raise ValueError('SUPPLEMENT_WRAPPER_CAP')
        writer.write('aggregate_calls.json',dict(completed_calls=calls,old_aborted_attempts=None,
            old_aborted_attempts_not_inferred=True,new_aborted_attempts_not_inferred=True,
            original_compile_cap=1712,counts_include_reused_records=True))
        order_rows=[]
        exploratory_paths=[root/x['directory'] for x in registry['accepted']]+[x for x in new_paths if x.name.startswith('cost_exploration_')]
        for phase,paths in (('exploration',exploratory_paths),('confirmation',[x for x in new_paths if x.name.startswith('cost_confirmation_')])):
            by_cell={}
            for path in paths:
                tr=read(path/'trajectory.json');by_cell.setdefault(tr['task']['cell_id'],[]).append(tr)
            for cid,trs in sorted(by_cell.items()):
                d=trs[0]['finite_distribution']
                if d is not None:
                    order_rows.append(dict(cell_id=cid,phase=phase,**order_diagnostic(d,
                        [k for t in trs for k in t['observed_orders']],R=trs[0]['task']['cell']['R'],trajectories=len(trs))))
        writer.write('cost_order_diagnostics.json',dict(rows=order_rows,component_sequence_and_compiled_mean_tail_certified=False))
        writer.write('cost_coverage.json',dict(exploration_groups=592,confirmation_groups=len(confirm),
            reused_wrapper_count=902,new_wrapper_count=new_wrapper_count,wrapper_count=902+new_wrapper_count,
            full_grid_best_method_certified=False,rare_event_population_mean_certified=False,
            legacy_pilot_included_in_fresh_mean=False,old_pilot_status_unchanged=True))
        verify_sources(root,m['source_commit'],m['source_hashes']);verify_evidence(root,registry)
        if verify_parent(root)!=m['input_identity'] or environment()!=m['environment']:raise ValueError('SUPPLEMENT_IDENTITY_AFTER')
        status,reason='H6_MATCHED_SUPPLEMENT_COMPLETE_MANDATORY_STOP',None
    except Exception as exc:
        status,reason='H6_MATCHED_SUPPLEMENT_STOP',type(exc).__name__+':'+str(exc)[:512]
    inventory={str(f.relative_to(output)):file_hash(f) for f in output.rglob('*') if f.is_file() and '.runtime_cache' not in f.parts}
    writer.write('execution_inventory.json',dict(source_commit=m['source_commit'],file_hashes=inventory,
        reuse_registry_digest=digest(registry),source_input_after_verified=status=='H6_MATCHED_SUPPLEMENT_COMPLETE_MANDATORY_STOP',
        live_aggregate_peak_rss_successful_batches_bytes=peak,failed_batch_peaks_in_progress_receipts=True,
        output_bytes=output_size(output),measurement_shots_sampled=False,phase_wall_seconds=None,total_wall_seconds=None),terminal=True)
    writer.write('execution_terminal.json',dict(status=status,reason=reason,source_commit=m['source_commit'],
        selected=selected,mandatory_stop=True,next_stage_authorized=False,H6_status='H6_NOT_AUTHORIZED',
        contract_status='DRAFT_NOT_AUTHORIZATION',retry=False,resume=False,new_input_generated=False,
        formal_winner_certified=False,ground_state_certified=False),terminal=True)
    return 0 if status=='H6_MATCHED_SUPPLEMENT_COMPLETE_MANDATORY_STOP' else 1

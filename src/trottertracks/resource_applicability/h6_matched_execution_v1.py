"""Stdlib disposable-process execution. No phase or total wall deadline."""
from __future__ import annotations
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from .ax2a_preparation import digest
from .ax2b_limits import output_size
from .ax2b_supplement_records_v1 import AtomicWriter
from .h6_matched_contract_v1 import file_hash, safe_path, verify_sources, verify_parent, environment, RUNNER
from .h6_matched_accounting_v1 import tasks, confirmation_ids, resource_row, order_diagnostic


def bounded_json(path, cap=16*2**20):
    p = Path(path)
    if p.stat().st_size>cap:
        raise ValueError('MATCHED_JSON_CAP')
    return json.loads(p.read_text())


def install_limits(cpus, address_bytes, output_bytes):
    os.sched_setaffinity(0,cpus)
    resource.setrlimit(resource.RLIMIT_AS,(address_bytes,address_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE,(output_bytes,output_bytes))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def worker_environment(threads, cache):
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                'NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):
        env[key]='1'
    env.update(NUMBA_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads),
               QISKIT_PARALLEL='FALSE',PYTHONDONTWRITEBYTECODE='1',
               NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'mpl'))
    return env


class Progress:
    def __init__(self, writer, cap=5000):
        self.writer=writer;self.count=0;self.cap=cap;self.start=time.monotonic()
    def update(self, **values):
        if self.count>=self.cap:
            raise RuntimeError('MATCHED_PROGRESS_CAP')
        self.writer.write('progress_%04d.json'%self.count,dict(sequence=self.count,
            elapsed_seconds=time.monotonic()-self.start,**values))
        self.count+=1
    def phase(self, name):
        self.update(phase=name)


def process_tree_rss(pid):
    """Live Linux RSS including descendants, separate from AS high-water."""
    total=0;todo=[pid];seen=set()
    while todo:
        p=todo.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            for line in Path(f'/proc/{p}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total+=int(line.split()[1])*1024
            todo += [int(x) for x in Path(f'/proc/{p}/task/{p}/children').read_text().split()]
        except (FileNotFoundError,ProcessLookupError):
            pass
    return total


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
    return done,peak


def verify_worker(root, directory, run_root):
    """Gate precedes numerical imports, claims are exclusive and task-bound."""
    root=Path(root);directory=Path(directory).resolve();run_root=Path(run_root).resolve()
    if directory.parent!=run_root or run_root.is_symlink():
        raise ValueError('MATCHED_WORKER_PATH')
    m=bounded_json(run_root/'frozen_manifest.json');g=bounded_json(run_root/'authorization.json')
    binding=bounded_json(run_root/'launch_binding.json');task_binding=bounded_json(directory/'task_binding.json')
    if (g.get('schema')!='h6_matched_authorization_v1' or g.get('approved_by_user') is not True
            or g.get('kind')!=m['kind'] or g.get('manifest_digest')!=digest(m)
            or g.get('source_commit')!=m['source_commit'] or g.get('one_shot') is not True
            or g.get('retry') is not False or g.get('resume') is not False
            or g.get('exclusive_output')!=str(run_root)
            or binding!={'manifest_digest':digest(m),'authorization_digest':digest(g),
                         'authorization_source_sha256':file_hash(run_root/'authorization_source.json')}
            or bounded_json(run_root/'authorization_source.json')!=g
            or task_binding['manifest_digest']!=digest(m)):
        raise ValueError('MATCHED_WORKER_GRANT_BINDING')
    from .h6_matched_contract_v1 import plan, preparation, source_paths
    expected=preparation()
    for k in ('schema','kind','status','science_authorized','launch_allowed','H6_status','contract_status','mandatory_stop','next_stage_authorized'):
        if type(m.get(k)) is not type(expected[k]) or m[k]!=expected[k]:
            raise ValueError('MATCHED_WORKER_FLAGS')
    if digest(m['plan'])!=digest(plan()) or m['execution_plan_sealed'] is not True:
        raise ValueError('MATCHED_WORKER_PLAN')
    if set(m['source_hashes'])!=set(source_paths(root)):
        raise ValueError('MATCHED_WORKER_CLOSURE')
    for p,sha in m['source_hashes'].items():
        if file_hash(safe_path(root,p))!=sha:
            raise ValueError('MATCHED_WORKER_SOURCE_CHANGED:'+p)
    if file_hash(safe_path(root,m['input_identity']['snapshot_path']))!=m['input_identity']['snapshot_sha256']:
        raise ValueError('MATCHED_WORKER_INPUT')
    for field in ('snapshot_receipt','df_receipt'):
        if file_hash(safe_path(root,m['input_identity'][field+'_path']))!=m['input_identity'][field+'_sha256']:
            raise ValueError('MATCHED_WORKER_RECEIPT')
    if environment()!=m['environment']:
        raise ValueError('MATCHED_WORKER_ENVIRONMENT')
    task=task_binding['task'];r=m['assigned_resources']
    signal_stage=task.get('phase')=='signal'
    if signal_stage:
        if task!={'phase':'signal'} or directory.name!='signal':raise ValueError('SIGNAL_TASK_BINDING')
        valid_cpu=[r['assigned_cpus']];threads=r['signal_threads'];cap=64*2**20;address=r['signal_address_space_bytes']
    else:
        registration=bounded_json(run_root/(task['phase']+'_tasks.json'))['tasks']
        if task not in registration:raise ValueError('UNREGISTERED_COST_TASK')
        valid_cpu=r['cost_cpu_sets'];threads=r['cost_threads'];cap=32*2**20;address=r['cost_address_space_bytes']
    if (task_binding['cpus'] not in valid_cpu or task_binding['threads']!=threads or
            task_binding['output_bytes']!=cap or task_binding['address_space_bytes']!=address):
        raise ValueError('WORKER_RESOURCE_BINDING')
    for k in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS'):
        if os.environ.get(k)!=str(threads):raise ValueError('WORKER_THREAD_BINDING')
    for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):
        if os.environ.get(k)!='1':raise ValueError('WORKER_THREAD_BINDING')
    if os.environ.get('QISKIT_PARALLEL')!='FALSE':raise ValueError('WORKER_QISKIT_BINDING')
    install_limits(task_binding['cpus'],address,cap)
    writer=AtomicWriter(directory,byte_cap=cap)
    writer.write('worker_claim.json',dict(task_digest=digest(task),source_commit=m['source_commit'],
        actual_cpus=sorted(os.sched_getaffinity(0)),threads=threads,retry=False,resume=False))
    return m,task,writer


def execute_worker(root,directory,run_root):
    m,task,writer=verify_worker(root,directory,run_root)
    progress=Progress(writer,cap=5000)
    try:
        from .h6_matched_port_v1 import MatchedPort, cost_task
        if task['phase']=='signal':
            port=MatchedPort(root,m,writer,progress);port.setup();port.correctness()
            if port.completed!=len(m['plan']['cells']):raise ValueError('SIGNAL_COVERAGE')
        else:
            cost_task(Path(root),m,task,writer,progress)
        for p,sha in m['source_hashes'].items():
            if file_hash(safe_path(root,p))!=sha:raise ValueError('MATCHED_TASK_SOURCE_AFTER')
        for field in ('snapshot','snapshot_receipt','df_receipt'):
            if file_hash(safe_path(root,m['input_identity'][field+'_path']))!=m['input_identity'][field+'_sha256']:
                raise ValueError('MATCHED_TASK_INPUT_AFTER')
        status,reason='TASK_COMPLETE',None
    except Exception as exc:
        status,reason='TASK_STOP',type(exc).__name__+':'+str(exc)[:512]
    writer.write('worker_terminal.json',dict(status=status,reason=reason,task=task,
        source_commit=m['source_commit'],mandatory_stop=True,next_stage_authorized=False,
        H6_status='H6_NOT_AUTHORIZED',contract_status='DRAFT_NOT_AUTHORIZATION'),terminal=True)
    return 0 if status=='TASK_COMPLETE' else 1


def resource_maps(signals, rows, plan, phase):
    result=[]
    for epsilon in plan['epsilons']:
        for s in signals:
            cr=[c for c in rows if c['cell_id']==s['cell']['id']]
            for metric in ('RZ','CX','depth','size'):
                for factor in ('1.0','10.0'):
                    row=resource_row(s,cr,epsilon,metric=metric,factor=factor)
                    if row is not None:
                        row.update(phase=phase,allowance_factor=factor);result.append(row)
    primary=[r for r in result if r['metric']=='RZ' and r['allowance_factor']=='1.0']
    crossings=[]
    for a in primary:
        if a['method']!='B2':continue
        for b in primary:
            if b['method']=='B2' or a['epsilon']!=b['epsilon']:continue
            dn=a['N_total']-b['N_total']
            p=(b['G_point']-a['G_point'])/dn if dn else None
            crossings.append(dict(B2=a['cell_id'],baseline=b['cell_id'],epsilon=a['epsilon'],
                                  crossing_P=p if p is not None and p>=0 else None,
                                  linear_difference_intercept=a['G_point']-b['G_point'],linear_difference_slope=dn))
    comparisons=[]
    for epsilon in plan['epsilons']:
        best={}
        for method in ('B0','B1','B2'):
            candidates=[r for r in primary if r['epsilon']==epsilon and r['method']==method]
            best[method]=min(candidates,key=lambda r:(r['G_point'],r['cell_id'])) if candidates else None
        b2=best['B2']
        for method in ('B0','B1'):
            baseline=best[method]
            if b2 and baseline:
                se=b2['G_standard_error']
                comparisons.append(dict(epsilon=epsilon,B2=b2['cell_id'],baseline=baseline['cell_id'],
                    ratio_point=b2['G_point']/baseline['G_point'] if baseline['G_point'] else None,
                    descriptive_two_SE_separated=(abs(b2['G_point']-baseline['G_point'])>2*se if se is not None else None),
                    winner='UNDETERMINED_POPULATION_MEAN',formal_winner_certified=False))
    return dict(rows=result,common_preparation_crossings=crossings,comparisons=comparisons,formal_winner_certified=False,
        mean_scope='conditional point estimates; SD/SE descriptive, no tail or simultaneous guarantee')


def orchestrate(root,output,manifest,writer):
    r=manifest['assigned_resources'];p=manifest['plan'];progress=Progress(writer)
    install_limits(r['assigned_cpus'],r['coordinator_address_space_bytes'],p['caps_proposed']['output_bytes'])
    completed=[];peak=0
    try:
        writer.write('execution_claim.json',dict(source_commit=manifest['source_commit'],retry=False,resume=False))
        _,peak=supervise_batch(root,output,manifest,[(0,{'phase':'signal'})],signal_stage=True,progress=progress)
        signals=bounded_json(Path(output)/'signal/signal_summary.json')['signals']
        explore=tasks(signals,p,'exploration')
        by_id={s['cell']['id']:s for s in signals}
        for t in explore:
            if t['cell']['method']!='B2':t['expected_signal']=by_id[t['cell_id']]['signals']['corrected']
        if len(explore)>596:raise ValueError('EXPLORATION_TASK_CAP')
        writer.write('exploration_tasks.json',dict(tasks=explore,coverage=[dict(cell_id=s['cell']['id'],
            status='COST_REGISTERED' if any(t['cell_id']==s['cell']['id'] for t in explore) else
                   'NO_RESOLVED_ELIGIBLE_PRECISION; COST_NOT_ACQUIRED') for s in signals]))
        paths,rss=supervise_batch(root,output,manifest,list(enumerate(explore)),progress=progress)
        peak=max(peak,rss);completed+=paths
        rows=[c for path in paths for c in bounded_json(path/'cost_result.json')['rows']]
        writer.write('exploration_resource_map.json',resource_maps(signals,rows,p,'exploration'))
        selected=confirmation_ids(signals,rows,p)
        confirm=tasks(signals,p,'confirmation',selected)
        if len(confirm)>256:raise ValueError('CONFIRMATION_TASK_CAP')
        writer.write('confirmation_tasks.json',dict(tasks=confirm,selected=selected,
            rule='top 2 exploratory B2 point scores per epsilon, union, frozen before fresh draws'))
        paths,rss=supervise_batch(root,output,manifest,list(enumerate(confirm)),progress=progress)
        peak=max(peak,rss);completed+=paths
        confirmation=[c for path in paths for c in bounded_json(path/'cost_result.json')['rows']]
        deterministic=[c for c in rows if by_id[c['cell_id']]['cell']['method']!='B2']
        writer.write('confirmation_resource_map.json',resource_maps(signals,deterministic+confirmation,p,'confirmation'))
        writer.write('cost_coverage.json',dict(exploration_groups=len(explore),confirmation_groups=len(confirm),
            wrapper_count=len(rows)+len(confirmation),unconfirmed_B2=[s['cell']['id'] for s in signals
                if s['cell']['method']=='B2' and s['cell']['id'] not in selected],
            full_grid_best_method_certified=False,rare_event_population_mean_certified=False,
            legacy_pilot_included_in_fresh_mean=False,old_pilot_status_unchanged=True))
        calls={k:0 for k in ('primitive','control_probe','compile','trajectory','occurrence','reference_matvec')}
        for k,v in bounded_json(Path(output)/'signal/signal_summary.json')['calls_attempted'].items():calls[k]+=v
        for path in completed:
            for k,v in bounded_json(path/'cost_result.json')['calls_attempted'].items():calls[k]+=v
        if any(v>p['caps_proposed'][k] for k,v in calls.items()):raise ValueError('AGGREGATE_CALL_CAP')
        writer.write('aggregate_calls.json',dict(completed_calls=calls,
            attempts_in_aborted_task_not_inferred=True,maximum_groups=p['maximum_cost_groups']))
        order_rows=[]
        for phase,paths in (('exploration',[p for p in completed if p.name.startswith('cost_exploration_')]),
                            ('confirmation',[p for p in completed if p.name.startswith('cost_confirmation_')])):
            for cell_id in sorted({bounded_json(path/'trajectory.json')['task']['cell_id'] for path in paths}):
                trajectories=[bounded_json(path/'trajectory.json') for path in paths
                              if bounded_json(path/'trajectory.json')['task']['cell_id']==cell_id]
                d=trajectories[0]['finite_distribution']
                if d is not None:
                    observed=[k for t in trajectories for k in t['observed_orders']]
                    order_rows.append(dict(cell_id=cell_id,phase=phase,**order_diagnostic(d,observed,
                        R=by_id[cell_id]['cell']['R'],trajectories=len(trajectories))))
        writer.write('cost_order_diagnostics.json',dict(rows=order_rows,
            component_sequence_and_compiled_mean_tail_certified=False))
        verify_sources(root,manifest['source_commit'],manifest['source_hashes'])
        if verify_parent(root)!=manifest['input_identity'] or environment()!=manifest['environment']:
            raise ValueError('MATCHED_IDENTITY_AFTER')
        status,reason='H6_MATCHED_RESOURCE_COMPLETE_MANDATORY_STOP',None
    except Exception as exc:
        status,reason='H6_MATCHED_RESOURCE_STOP',type(exc).__name__+':'+str(exc)[:512]
    inventory={str(f.relative_to(output)):file_hash(f) for f in Path(output).rglob('*')
               if f.is_file() and '.runtime_cache' not in f.parts}
    writer.write('execution_inventory.json',dict(source_commit=manifest['source_commit'],file_hashes=inventory,
        source_input_after_verified=status=='H6_MATCHED_RESOURCE_COMPLETE_MANDATORY_STOP',
        output_bytes=output_size(output),live_aggregate_peak_rss_bytes=peak,
        measurement_shots_sampled=False,phase_wall_seconds=None,total_wall_seconds=None,
        H6_status='H6_NOT_AUTHORIZED',contract_status='DRAFT_NOT_AUTHORIZATION'),terminal=True)
    writer.write('execution_terminal.json',dict(status=status,reason=reason,source_commit=manifest['source_commit'],
        mandatory_stop=True,next_stage_authorized=False,H6_status='H6_NOT_AUTHORIZED',
        contract_status='DRAFT_NOT_AUTHORIZATION',new_input_generated=False,
        formal_winner_certified=False,ground_state_certified=False,retry=False,resume=False),terminal=True)
    return 0 if status=='H6_MATCHED_RESOURCE_COMPLETE_MANDATORY_STOP' else 1

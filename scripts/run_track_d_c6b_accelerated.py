#!/usr/bin/env python3
"""Authorized SOURCE handoff: finish only never-started C6b tasks, no sampling.

The old controller must have drained all submitted workers. Its original output
and STOP stay immutable. Native compiler/circuit/precision/model stay unchanged;
new workers explicitly select the separately tested state-action implementation.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import run_track_d_c6b as frozen
ROOT, OLD = frozen.ROOT, frozen.ROOT/frozen.OUTPUT
OUTPUT = 'artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion'
TECH = 'artifacts/track_d_c6b_q8/2026-11-07/accelerated_technical'
OLD_SOURCE = '23725fc6f6d6d0c2bec93e31e32074711804e250'
INHERITANCE = TECH+'/inherited_evidence.json'
NEW = ('scripts/run_track_d_c6b_accelerated.py', 'tests/check_track_d_c6b_accelerated.py',
       'src/trottertracks/track_d_state_action_acceleration.py',
       'tests/check_track_d_state_action_acceleration.py',
       'docs/research/track_d_c6b_q8_20261107/ACCELERATION.md',
       'artifacts/track_d_c6b_q8/2026-11-07/acceleration_technical/synthetic_checks.json',
       'docs/research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md',
       INHERITANCE, TECH+'/synthetic_tests.json')
write = frozen.write


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inheritance_plan(old):
    """No QPY decode or numerical work. Reject possibly-started incomplete work."""
    old = Path(old); result=read(old/'result.json'); stop=read(old/'STOP.json')
    failure=read(old/'failure.json')
    if (result['source_commit'] != OLD_SOURCE or not stop['mandatory_stop'] or
            result['status'] != 'STOP_FAILURE' or failure['error_type'] != 'KeyboardInterrupt'):
        raise ValueError('ONLY_AUTHORIZED_CONTROLLER_DRAIN_NOT_SCIENTIFIC_FAILURE')
    ids=[f'q8_sample{i:02d}' for i in range(32)]
    if result['planned_ids'] != ids:
        raise ValueError('ORIGINAL_FIXED_IDS')
    raw=[]; wrappers=[]; members=[]
    for sid in ids:
        directory=old/'local'/sid
        if not directory.exists():
            continue
        check=directory/'precompile_checks.json'
        if not check.exists() or read(check)['status']!='PASS':
            raise ValueError('RAW_ALREADY_STARTED_BUT_NOT_COMPLETE:'+sid)
        if list(directory.glob('*failure.json')) or (directory/'partial_checks.json').exists():
            raise ValueError('NO_INHERITANCE_AFTER_SCIENTIFIC_FAILURE:'+sid)
        raw.append(sid)
        for method in frozen.c2.ARMS:
            for axis in frozen.c2.AXES:
                name=method+'_'+axis; marker=directory/(name+'_compile.json')
                compiled=directory/'circuits'/(name+'_compiled.qpy')
                post=directory/(name+'_check.json'); observable=directory/(name+'_observable.npz')
                if marker.exists():
                    m=read(marker)
                    if m['status']!='PASS' or m['compile_calls']!=1 or not compiled.exists():
                        raise ValueError('STARTED_COMPILE_INCOMPLETE_NO_RETRY:'+sid+'/'+name)
                    if not post.exists() or read(post)['status']!='PASS' or not observable.exists():
                        raise ValueError('POSSIBLY_STARTED_POSTCHECK_INCOMPLETE_NO_RETRY:'+sid+'/'+name)
                    wrappers.append([sid,method,axis])
                elif compiled.exists() or post.exists() or observable.exists():
                    raise ValueError('UNBOUND_PARTIAL_WRAPPER:'+sid+'/'+name)
        for p in sorted(directory.glob('*.json')):
            members.append(p)
        members.extend(sorted((directory/'circuits').glob('*.qpy')))
        members.extend(sorted(directory.glob('*.npz')))
    events=read(old/'event_bindings.json')
    if [r['sample_id'] for r in events]!=ids:
        raise ValueError('ALL32_EVENTS_ALREADY_SAVED_REQUIRED')
    if len(wrappers)!=result['compile_calls_started'] or len(wrappers)>128:
        raise ValueError('ORIGINAL_COMPILE_ACCOUNTING')
    for row in events:
        p=old/'local/events'/(row['sample_id']+'.json')
        if sha(p)!=row['record_sha256']:
            raise ValueError('SEALED_EVENT_RECORD_CHANGED')
        members.append(p)
    members += [old/'local/car_context.npz']
    members += [old/n for n in ['SOURCE.json','STOP.json','result.json','failure.json',
                                'seed_plan.json','frozen_model.json','event_bindings.json']]
    return dict(original_source=OLD_SOURCE,reason='USER_AUTHORIZED_SOURCE_HANDOFF_AFTER_CONTROLLER_DRAIN',
        original_scope=dict(q=8,paths=32,occurrences_already_sampled=256,total_compile_limit=128),
        old_controller_recorded_postchecks=result.get('completed_postchecks'),
        inherited_local_postcheck_evidence=len(wrappers),
        controller_summary_may_omit_tasks_that_finished_during_drain=True,
        raw_reused=raw,raw_remaining=[sid for sid in ids if sid not in raw],wrappers_reused=wrappers,
        compile_remaining=128-len(wrappers),sampling_remaining=0,
        members=[dict(path=str(p.relative_to(old)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(members)])


def verify_inheritance(old, packet):
    for row in packet['members']:
        p=Path(old)/row['path']
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise ValueError('INHERITED_OLD_EVIDENCE_CHANGED:'+row['path'])


def install_state_actions():
    """Explicit selection in a NEW own worker, never in the drained old process."""
    from trottertracks import c1_h6_state_action as primitive
    from trottertracks.track_d_state_action_acceleration import BatchedStateActions
    primitive.StateActions = BatchedStateActions


def raw_worker(row, origin, local, threads, stop):
    frozen.c2.old().bootstrap(origin)
    install_state_actions()
    return frozen.raw_worker(row,origin,local,threads,stop)


def compiled_worker(sid, task, stop):
    origin,directory,method,axis,threads=task
    frozen.c2.old().bootstrap(origin)
    install_state_actions()
    name=method+'_'+axis
    try:
        if read(Path(directory)/'precompile_checks.json')['status']!='PASS':
            raise ValueError('PATH_RAW_PASS_REQUIRED_BEFORE_NEW_COMPILE')
        if (Path(directory)/(name+'_compile.json')).exists() or (Path(directory)/'circuits'/(name+'_compiled.qpy')).exists():
            raise ValueError('COMPILE_ALREADY_STARTED_NO_DUPLICATE')
        return frozen.previous.compiled_worker(sid,task,stop)
    except BaseException:
        stop.set()
        raise


def preflight(origin,saved):
    source=frozen.preflight(origin,saved);p=frozen.c2.prior();head=source['execution_source']
    for name in NEW:
        if (ROOT/name).read_bytes()!=p.git('show',head+':'+name):
            raise ValueError('UNCOMMITTED_ACCELERATED_SOURCE:'+name)
    inheritance=read(ROOT/INHERITANCE)
    verify_inheritance(OLD,inheritance)
    original=read(OLD/'SOURCE.json')
    if {**source,'execution_source':OLD_SOURCE} != original:
        raise ValueError('ORIGINAL_SOURCE_INPUT_LAW_PROFILE_CHANGED')
    if read(ROOT/TECH/'synthetic_tests.json')['status']!='PASS':
        raise ValueError('ACCELERATED_ARTIFICIAL_CHECKS_REQUIRED')
    source.update(accelerated_source={n:sha(ROOT/n) for n in NEW},old_execution_source=OLD_SOURCE,
        state_action='explicit worker-local StateActions -> BatchedStateActions; official AerJob/native callback; no source file mutation',
        handoff='completed old tasks reused byte-bound; never-started tasks only; no resampling or recompile',
        original_raw_reused=len(inheritance['raw_reused']),original_wrappers_reused=len(inheritance['wrappers_reused']))
    return source,inheritance


def initial_tasks(events, inherited, estimates):
    reused={tuple(x) for x in inherited['wrappers_reused']}; tasks=[]
    for sid in inherited['raw_reused']:
        for method in frozen.c2.ARMS:
            for axis in frozen.c2.AXES:
                if (sid,method,axis) not in reused:
                    tasks.append(dict(kind='compiled',sample_id=sid,method=method,axis=axis,
                        memory_bytes=estimates[('reference' if method=='reference' else 'direct')+'_task_memory_estimate_bytes']))
    for row in events:
        if row['sample_id'] in inherited['raw_remaining']:
            tasks.append(dict(kind='raw',sample_id=row['sample_id'],row=row,method=None,
                              memory_bytes=estimates['raw_task_memory_estimate_bytes']))
    return tasks


def pipeline(events, inherited, origin, local, resources, estimates, raw, compiled, stop, out):
    budget=resources['numerical_threads'];pending=deque(initial_tasks(events,inherited,estimates));active={}
    workers=min(budget,len(pending)+4*len(inherited['raw_remaining']))
    workers=max(1,workers)
    write(local/'resources.json',dict(resources,stage_estimates=estimates,worker_processes=workers,
        policy='observed free resources, separate completed-stage memory estimates, CPU remainder, native thread budget enforced inside Aer executor'))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or active:
            used_cpu=sum(r['threads'] for r in active.values())
            used_mem=sum(r['task']['memory_bytes'] for r in active.values())
            free_mem=min(frozen.memory_available(),max(0,resources['memory']['MemAvailable']-used_mem))
            if not stop.is_set():
                for task,threads in frozen.task_allocation(budget-used_cpu,free_mem,list(pending),workers,len(active)):
                    pending.remove(task)
                    if task['kind']=='raw':
                        future=pool.submit(raw_worker,task['row'],origin,local,threads,stop)
                    else:
                        args=(origin,local/task['sample_id'],task['method'],task['axis'],threads)
                        future=pool.submit(compiled_worker,task['sample_id'],args,stop)
                    active[future]=dict(task=task,threads=threads)
                    write(local/'dispatch_progress.json',dict(active=len(active),pending=len(pending),
                        cpu_slots_used=sum(r['threads'] for r in active.values()),latest_task={k:v for k,v in task.items() if k!='row'}))
            if stop.is_set():pending.clear()
            if not active:
                if pending:raise RuntimeError('CURRENT_MEMORY_INSUFFICIENT_FOR_NEXT_TASK')
                break
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                task=active.pop(future)['task'];value=future.result()
                if task['kind']=='raw':
                    sid,row=value;raw[sid]=row
                    print(sid,'raw',row['status'],flush=True)
                    if row['status']=='PASS' and not stop.is_set():
                        for method in frozen.c2.ARMS:
                            for axis in frozen.c2.AXES:
                                pending.appendleft(dict(kind='compiled',sample_id=sid,method=method,axis=axis,
                                    memory_bytes=estimates[('reference' if method=='reference' else 'direct')+'_task_memory_estimate_bytes']))
                    else:stop.set()
                else:
                    sid,name,row=value;compiled.setdefault(sid,{})[name]=row
                    print(sid,name,row['status'],flush=True)
                    if row['status']!='PASS':stop.set()
                write(out/'progress.json',dict(status='ACCELERATED_REMAINING_TASKS',
                    raw_pass=sum(r['status']=='PASS' for r in raw.values()),
                    postchecks_pass=sum(r['status']=='PASS' for v in compiled.values() for r in v.values()),
                    inherited_raw=len(inherited['raw_reused']),inherited_postchecks=len(inherited['wrappers_reused']),
                    active=len(active),pending=len(pending),stop_requested=stop.is_set()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin',required=True,type=Path);parser.add_argument('--saved',required=True,type=Path)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise PermissionError('EXPLICIT_AUTHORIZED_SOURCE_HANDOFF_ONLY')
    out=ROOT/OUTPUT
    if out.exists():raise ValueError('FRESH_COMPLETION_OUTPUT_NO_RETRY_RESUME')
    out.mkdir(parents=True);local=out/'local';local.mkdir()
    os.environ['MPLCONFIGDIR']=str(local/'matplotlib-cache'/'parent')
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:os.environ[key]='1'
    os.environ['QISKIT_PARALLEL']='false'
    source=dict(execution_source=frozen.c2.prior().git('rev-parse','HEAD').decode().strip())
    raw={};compiled={};inherited=None;success=preserved=False;diagnostic=None
    try:
        frozen.c2.old().bootstrap(args.origin)
        from trottertracks import track_d_c6b_holdout as pilot
        from trottertracks.track_d_state_action_acceleration import stage_memory_estimates
        source,inherited=preflight(args.origin,args.saved);write(out/'SOURCE.json',source)
        write(out/'inherited_evidence.json',inherited)
        for row in inherited['members']:
            if not row['path'].startswith('local/'):continue
            dest=out/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise ValueError('NO_INHERITANCE_OVERWRITE')
            shutil.copyfile(OLD/row['path'],dest)
            if sha(dest)!=row['sha256']:raise ValueError('COPIED_EVIDENCE_BYTES')
        for name in ['seed_plan.json','frozen_model.json','event_bindings.json']:
            shutil.copyfile(OLD/name,out/name)
        events=read(out/'event_bindings.json');model=read(out/'frozen_model.json')
        for sid in inherited['raw_reused']:raw[sid]=read(local/sid/'precompile_checks.json')
        for sid,method,axis in inherited['wrappers_reused']:
            compiled.setdefault(sid,{})[method+'_'+axis]=read(local/sid/(method+'_'+axis+'_check.json'))
        old_prices=read(ROOT/frozen.c5.C3_OUTPUT/'result.json')['compiled']
        estimates=stage_memory_estimates(raw=list(raw.values()),
            reference=[r for k,r in old_prices['8'].items() if k.startswith('reference')],
            direct=[r for k,r in old_prices['4'].items() if k.startswith('symmetric')])
        resources=frozen.c2.prior().cpu_observation()
        resources['memory']['MemAvailable']=min(resources['memory']['MemAvailable'],frozen.memory_available())
        quota=Path('/sys/fs/cgroup/cpu.max')
        if quota.exists() and quota.read_text().split()[0]!='max':
            q,period=map(int,quota.read_text().split());resources['numerical_threads']=min(resources['numerical_threads'],max(1,q//period))
        with multiprocessing.Manager() as manager:
            stop=manager.Event()
            pipeline(events,inherited,args.origin,local,resources,estimates,raw,compiled,stop,out)
        if len(raw)!=32 or any(r['status']!='PASS' for r in raw.values()):raise ValueError('ALL32_RAW_REQUIRED')
        if sum(map(len,compiled.values()))!=128 or any(r['status']!='PASS' for v in compiled.values() for r in v.values()):
            raise ValueError('ALL128_POSTCHECKS_REQUIRED')
        if sum(read(p).get('compile_calls',0) for p in local.glob('q8_sample*/*_compile.json'))!=128:
            raise ValueError('TOTAL_ORIGINAL_PLUS_COMPLETION_COMPILE_BUDGET')
        prices=[];old_wrappers={tuple(x) for x in inherited['wrappers_reused']}
        for row in events:
            sid=row['sample_id'];directory=local/sid
            frozen.previous.paired_postcheck(directory,8)
            p=local/'events'/(sid+'.json')
            if sha(p)!=row['record_sha256']:raise ValueError('EVENTS_CHANGED')
            original_request=pilot.request_from_record(read(p))
            request=replace(original_request,execution_source=source['execution_source'])
            for method in frozen.c2.ARMS:
                for axis in frozen.c2.AXES:
                    name=method+'_'+axis;reused=(sid,method,axis) in old_wrappers
                    price=pilot.price_record(request,method=method,axis=axis,postcheck=compiled[sid][name],
                        raw_status=raw[sid]['status'],paired_status='PASS',execution_source=source['execution_source'],
                        compiler=read(directory/(name+'_compile.json'))['compiler'])
                    price['provenance']=dict(sampling_source=OLD_SOURCE,original_event_request_fingerprint=original_request.fingerprint,
                        raw_source=OLD_SOURCE if sid in inherited['raw_reused'] else source['execution_source'],
                        compile_and_postcheck_source=OLD_SOURCE if reused else source['execution_source'],
                        price_evaluator_source=source['execution_source'],compiled_bytes_reused=reused)
                    prices.append(price)
        if source!=preflight(args.origin,args.saved)[0]:raise ValueError('SOURCE_INPUT_INHERITANCE_CHANGED')
        preserved=True;write(out/'prices.json',prices)
        summary=pilot.paired_statistics(prices);write(out/'paired_statistics.json',summary)
        diagnostic=pilot.diagnose(summary,model);write(out/'exploratory_diagnostics.json',diagnostic)
        success=True
    except BaseException as error:
        message=str(error)
        for p in [str(ROOT),str(args.origin),str(args.saved)]:message=message.replace(p,'<task-path>')
        write(out/'failure.json',dict(status='STOP',error_type=type(error).__name__,error=message))
    for directory in sorted(local.glob('q8_sample*')):
        dest=out/'paths'/directory.name;dest.mkdir(parents=True,exist_ok=True)
        for p in directory.glob('*.json'):
            if p.name not in ['conditions.json','fusion.json']:write(dest/p.name,frozen.c5.public_view(read(p)))
    reused_count=len(inherited['wrappers_reused']) if inherited else 0
    calls=sum(read(p).get('compile_calls',0) for p in local.glob('q8_sample*/*_compile.json'))
    planned=[f'q8_sample{i:02d}' for i in range(32)]
    completed=[sid for sid in planned if raw.get(sid,{}).get('status')=='PASS' and len(compiled.get(sid,{}))==4
        and all(r['status']=='PASS' for r in compiled[sid].values()) and (local/sid/'paired_observables.json').exists()]
    result=dict(status='TECHNICAL_PASS_MANDATORY_STOP' if success else 'STOP_FAILURE',source_commit=source['execution_source'],
        original_source_commit=OLD_SOURCE,authorized_SOURCE_handoff=True,original_STOP_preserved=True,
        planned_ids=planned,completed_ids=completed,missing_or_failed_ids=[sid for sid in planned if sid not in completed],
        raw_PASS=sum(r['status']=='PASS' for r in raw.values()),raw_reused=len(inherited['raw_reused']) if inherited else 0,
        completed_postchecks=sum(r['status']=='PASS' for v in compiled.values() for r in v.values()),
        compile_calls_across_original_and_completion=calls,new_compile_calls_started=max(0,calls-reused_count),
        reused_compile_and_postcheck_pairs=reused_count,total_compile_limit=128,new_sampling_calls=0,
        original_canonical_occurrences=256,new_ground_solves=0,new_DF_refits=0,N_evaluations=0,quantum_Born_shots=0,
        source_input_and_original_evidence_preserved=preserved,completed_scientific_tasks_rerun=0,
        exploratory_status=diagnostic['status'] if diagnostic else 'INCOMPLETE',statistical_qualification='STATISTICALLY_UNQUALIFIED',
        confirmatory_PASS=False,long_q_qualified=False,whole_law_guaranteed=False,RPE_total_cost_evaluated=False,
        model_refits=0,calibration_added=0,q_above8=False,retry=0,mandatory_stop=True)
    write(out/'result.json',result);write(out/'STOP.json',dict(status=result['status'],mandatory_stop=True,next_stage_authorized=False))
    write(out/'progress.json',dict(status=result['status']));print(result['status'],flush=True)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Authorized C7-3 source handoff: reuse drained raw, split P0/post stages."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
from dataclasses import replace,asdict
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shutil

import run_track_d_c7_3_q32 as frozen
ROOT=frozen.ROOT
OLD_SOURCE='35bfd0423d08114b0a24107501ce545cdfb28f32'
OUTPUT='artifacts/track_d_c7_3_q32/2026-11-07/run01_accelerated_completion'
TECH='artifacts/track_d_c7_3_q32/2026-11-07/accelerated_technical'
INHERITANCE=TECH+'/inherited_evidence.json'
NEW=('src/trottertracks/track_d_c7_3_acceleration.py','scripts/run_track_d_c7_3_accelerated.py',
     'tests/check_track_d_c7_3_acceleration.py','tests/check_track_d_c7_3_handoff.py',
     'docs/research/track_d_c7_3_q32_20261107/ACCELERATED_COMPLETION_PLAN.md',
     TECH+'/synthetic_tests.json',INHERITANCE)
write=frozen.write
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def inheritance_plan(old):
    """Reused C6b stopped-task policy, narrowed to complete C7-3 raw8."""
    old=Path(old);r=read(old/'result.json');stop=read(old/'STOP.json');failure=read(old/'failure.json')
    ids=[f'q32_sample{i:02d}' for i in range(8)]
    if r['source_commit']!=OLD_SOURCE or r['status']!='STOP_FAILURE' or not stop['mandatory_stop'] or failure['error_type']!='KeyboardInterrupt':
        raise ValueError('ONLY_AUTHORIZED_CONTROLLER_DRAIN')
    if r['planned_ids']!=ids:raise ValueError('FIXED_EIGHT_IDS')
    members=[];wrappers=[]
    for sid in ids:
        d=old/'local'/sid
        if not (d/'precompile_checks.json').exists() or read(d/'precompile_checks.json')['status']!='PASS':
            raise ValueError('DRAINED_RAW_PASS_REQUIRED:'+sid)
        if list(d.glob('*failure.json')) or (d/'partial_checks.json').exists():raise ValueError('SCIENTIFIC_FAILURE_NOT_RESUMABLE:'+sid)
        if not (d/'structural_features_raw.json').exists():raise ValueError('ORIGINAL_RAW_FEATURE_REQUIRED')
        for leaf in ['conditions.json','fusion.json','references.npz']:
            if not (d/leaf).exists():raise ValueError('ORIGINAL_RAW_EVIDENCE_REQUIRED:'+leaf)
        for method in frozen.c2.ARMS:
            for axis in frozen.c2.AXES:
                name=method+'_'+axis;marker=d/(name+'_compile.json');compiled=d/'circuits'/(name+'_compiled.qpy')
                post=d/(name+'_check.json');observable=d/(name+'_observable.npz')
                if marker.exists():
                    m=read(marker)
                    if m['status']!='PASS' or m['compile_calls']!=1 or not compiled.exists():raise ValueError('STARTED_COMPILE_INCOMPLETE')
                    if not post.exists() or read(post)['status']!='PASS' or not observable.exists():raise ValueError('STARTED_POST_INCOMPLETE')
                    if not (d/(name+'_features.json')).exists():raise ValueError('STARTED_FEATURE_INCOMPLETE')
                    wrappers.append([sid,method,axis])
                elif compiled.exists() or post.exists() or observable.exists():raise ValueError('UNBOUND_PARTIAL_WRAPPER')
        members+=list(d.glob('*.json'))+list((d/'circuits').glob('*.qpy'))+list(d.glob('*.npz'))
    events=read(old/'event_bindings.json')
    if [row['sample_id'] for row in events]!=ids or r['canonical_occurrence_sampling_calls_started']!=256:raise ValueError('ALL256_ALREADY_SAMPLED')
    for row in events:
        p=old/'local/events'/(row['sample_id']+'.json')
        if sha(p)!=row['record_sha256']:raise ValueError('SEALED_EVENT_CHANGED')
        members.append(p)
    if len(wrappers)!=r['compile_calls_started'] or len(wrappers)>32:raise ValueError('COMPILE_ACCOUNTING')
    members += [old/n for n in ['SOURCE.json','STOP.json','result.json','failure.json','seed_plan.json','frozen_model.json','event_bindings.json']]
    return dict(original_source=OLD_SOURCE,original_controller_status=r['status'],
        original_source_input_preserved_field=r['source_input_preserved'],reason='USER_AUTHORIZED_SOURCE_HANDOFF_AFTER_CONTROLLER_DRAIN',
        raw_reused=ids,raw_remaining=[],wrappers_reused=wrappers,compile_remaining=32-len(wrappers),sampling_remaining=0,
        scope=dict(q=32,cut=16,paths=8,total_occurrences=256,total_events=8192,total_compile_limit=32),
        members=[dict(path=str(p.relative_to(old)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(members))])


def verify_inheritance(old,packet):
    for row in packet['members']:
        p=Path(old)/row['path']
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('OLD_EVIDENCE_CHANGED:'+row['path'])


def preflight_inputs(origin,saved,old):
    source=frozen.preflight(origin,saved);head=source['execution_source']
    original=read(Path(old)/'SOURCE.json')
    if {**source,'execution_source':OLD_SOURCE}!=original:raise ValueError('ORIGINAL_SOURCE_INPUT_MODEL_P0_CHANGED')
    for name in NEW:
        if (ROOT/name).read_bytes()!=frozen.c2.prior().git('show',head+':'+name):raise ValueError('UNCOMMITTED_HANDOFF_SOURCE:'+name)
    # Protect the actual old execution files, not unrelated worktrees/HEADs.
    old_root=Path(old).parents[3]
    for name,want in original['C7_3_source'].items():
        if sha(old_root/name)!=want:raise ValueError('ACTUAL_OLD_EXECUTION_SOURCE_CHANGED:'+name)
    t=read(ROOT/TECH/'synthetic_tests.json')
    if t['status']!='PASS' or t['scientific_runs']!=0:raise ValueError('FAST_SYNTHETIC_CHECKS_REQUIRED')
    source.update(accelerated_source={n:sha(ROOT/n) for n in NEW},old_execution_source=OLD_SOURCE,
        handoff='all8 old raw/checks reused; only never-started compile/post; no sampling or scientific retry',
        validation_runtime='official Aer double unitary <=9 only; existing17-qubit statevector/batched action with exact local-gate cache',
        compiler_unchanged=True,full17_unitary=False,raw_reused=8,raw_remaining=0)
    return source


def preflight(origin,saved,old):
    source=preflight_inputs(origin,saved,old);inherited=read(ROOT/INHERITANCE)
    if inherited!=inheritance_plan(old):raise ValueError('FIXED_DRAINED_INHERITANCE_CHANGED')
    verify_inheritance(old,inherited)
    source['drained_inheritance_sha256']=hashlib.sha256(json.dumps(inherited,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return source,inherited


def stage_worker(sid,task,stage,stop):
    origin,directory,method,axis,threads=task;directory=Path(directory);name=method+'_'+axis
    os.environ['MPLCONFIGDIR']=str(directory/'matplotlib-cache'/(stage+'_'+name))
    try:
        if stop.is_set():raise RuntimeError('STOP_BEFORE_'+stage.upper())
        if read(directory/'precompile_checks.json')['status']!='PASS':raise ValueError('PATH_RAW_PASS_REQUIRED')
        runner=frozen.c2.old();runner.bootstrap(origin)
        from threadpoolctl import threadpool_limits
        if stage=='compile':
            if (directory/(name+'_compile.json')).exists() or (directory/'circuits'/(name+'_compiled.qpy')).exists():raise ValueError('COMPILE_ALREADY_STARTED')
            with threadpool_limits(limits=1):runner.compile_one(task)
            row=dict(status='PASS',peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write(directory/(name+'_compile_stage.json'),row)
        elif stage=='post':
            if read(directory/(name+'_compile.json'))['status']!='PASS' or (directory/(name+'_check.json')).exists():raise ValueError('NEW_POST_REQUIRES_COMPLETED_COMPILE')
            from trottertracks import track_d_c7_3_acceleration as fast
            engine=fast.install(threads)
            name,row=runner.verify_one(task)
            from trottertracks import track_d_c7_3_q32 as pilot
            circuit=pilot.api.c3.base.load_qpy(directory/'circuits'/(name+'_compiled.qpy'))
            write(directory/(name+'_features.json'),pilot.wrapper_features(circuit,method))
            write(directory/(name+'_acceleration.json'),dict(cache=fast.cache_statistics(),
                small_operator_backend='official Aer double unitary, at most9 qubits',
                reference_backend='existing official Aer double statevector all256 input columns',
                compiler_changed=False,tolerance_changed=False,phase_fit=False))
            write(directory/(name+'_native_metadata_private.json'),engine.last_metadata)
        else:raise ValueError('STAGE')
        return sid,name,stage,row
    except BaseException as error:
        stop.set();message=str(error)
        for p in [str(ROOT),str(origin),str(directory)]:message=message.replace(p,'<task-path>')
        row=dict(status='FAIL',error_type=type(error).__name__,error=message)
        write(directory/(name+'_'+stage+'_failure.json'),row)
        return sid,name,stage,row


def allocation(cpu,memory,pending,workers,active):
    # Reuse greedy memory/CPU allocator. Compile is a one-thread P0 task;
    # both post arms now have native parallelism. Preserve real task metadata.
    proxies=[dict(t,kind='raw' if t['stage']=='post' else 'compiled',
                  method='symmetric_directional',original_index=i) for i,t in enumerate(pending)]
    return [(pending[p['original_index']],threads) for p,threads in
            frozen.task_allocation(cpu,memory,proxies,workers,active)]


def pipeline(events,inherited,origin,local,resources,stop,out,compiled):
    reused={tuple(x) for x in inherited['wrappers_reused']};pending=deque()
    for e in events:
        for method in frozen.c2.ARMS:
            for axis in frozen.c2.AXES:
                if (e['sample_id'],method,axis) not in reused:
                    pending.append(dict(stage='compile',sample_id=e['sample_id'],method=method,axis=axis,
                        memory_bytes=resources['direct_task_memory_estimate_bytes']))
    budget=resources['numerical_threads'];workers=max(1,min(budget,2*len(pending)));active={};launches=0
    write(local/'resources.json',dict(resources,max_worker_processes=workers,
        policy='separate unchanged P0 compile and native post stages; observed CPU/memory admission; no low worker cap'))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or active:
            used_cpu=sum(r['threads'] for r in active.values());used_mem=sum(r['task']['memory_bytes'] for r in active.values())
            free_mem=min(frozen.memory_available(),max(0,resources['memory']['MemAvailable']-used_mem))
            if not stop.is_set():
                for task,threads in allocation(budget-used_cpu,free_mem,list(pending),workers,len(active)):
                    pending.remove(task);args=(origin,local/task['sample_id'],task['method'],task['axis'],threads)
                    f=pool.submit(stage_worker,task['sample_id'],args,task['stage'],stop)
                    active[f]=dict(task=task,threads=threads);launches+=1
                    write(local/'dispatch_progress.json',dict(launches=launches,active=len(active),pending=len(pending),
                        cpu_slots_used=sum(r['threads'] for r in active.values()),latest_task=task,threads=threads))
            if stop.is_set():pending.clear()
            if not active:
                if pending:raise RuntimeError('CURRENT_MEMORY_INSUFFICIENT')
                break
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                task=active.pop(future)['task'];sid,name,stage,row=future.result()
                print(sid,name,stage,row['status'],flush=True)
                if row['status']!='PASS':stop.set()
                elif stage=='compile' and not stop.is_set():
                    pending.appendleft(dict(task,stage='post',memory_bytes=resources[
                        ('reference' if task['method']=='reference' else 'direct')+'_task_memory_estimate_bytes']))
                elif stage=='post':compiled.setdefault(sid,{})[name]=row
                write(out/'progress.json',dict(status='ACCELERATED_COMPILE_POST_PIPELINE',raw_reused=8,
                    postchecks_pass=sum(r['status']=='PASS' for v in compiled.values() for r in v.values()),
                    active=len(active),pending=len(pending),stop_requested=stop.is_set()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['origin','saved','old-output','previous-q16']:p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:raise PermissionError('AUTHORIZED_C7_3_SOURCE_HANDOFF_ONLY')
    out=ROOT/OUTPUT
    if out.exists():raise ValueError('FRESH_COMPLETION_ONLY_NO_RETRY')
    out.mkdir(parents=True);local=out/'local';local.mkdir();(local/'.gitignore').write_text('*\n')
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:os.environ[key]='1'
    os.environ['QISKIT_PARALLEL']='false';os.environ['MPLCONFIGDIR']=str(local/'matplotlib-cache'/'parent')
    success=preserved=False;diagnostic=None;compiled={};inherited=None
    source=dict(execution_source=frozen.c2.prior().git('rev-parse','HEAD').decode().strip())
    try:
        frozen.c2.old().bootstrap(args.origin)
        from trottertracks import track_d_c7_3_q32 as pilot
        source,inherited=preflight(args.origin,args.saved,args.old_output);write(out/'SOURCE.json',source)
        write(out/'inherited_evidence.json',inherited)
        for row in inherited['members']:
            if not row['path'].startswith('local/'):continue
            dest=out/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise ValueError('NO_INHERITANCE_OVERWRITE')
            shutil.copyfile(args.old_output/row['path'],dest)
            if sha(dest)!=row['sha256']:raise ValueError('COPIED_BYTES')
        for leaf in ['seed_plan.json','frozen_model.json','event_bindings.json']:shutil.copyfile(args.old_output/leaf,out/leaf)
        events=read(out/'event_bindings.json');model=read(out/'frozen_model.json')
        for sid,method,axis in inherited['wrappers_reused']:
            compiled.setdefault(sid,{})[method+'_'+axis]=read(local/sid/(method+'_'+axis+'_check.json'))
        resources=frozen.c2.prior().cpu_observation();resources['memory']['MemAvailable']=min(resources['memory']['MemAvailable'],frozen.memory_available())
        quota=Path('/sys/fs/cgroup/cpu.max')
        if quota.exists() and quota.read_text().split()[0]!='max':
            q,period=map(int,quota.read_text().split());resources['numerical_threads']=min(resources['numerical_threads'],max(1,q//period))
        resources.update(frozen.memory_estimates(args.previous_q16))
        with multiprocessing.Manager() as manager:
            stop=manager.Event();pipeline(events,inherited,args.origin,local,resources,stop,out,compiled)
        if sum(map(len,compiled.values()))!=32 or any(r['status']!='PASS' for v in compiled.values() for r in v.values()):raise ValueError('ALL32_POSTCHECKS_REQUIRED')
        if sum(read(p)['compile_calls'] for p in local.glob('q32_sample*/*_compile.json'))!=32:raise ValueError('TOTAL32_P0_ACCOUNTING')
        prices=[];reused={tuple(x) for x in inherited['wrappers_reused']}
        for e in events:
            sid=e['sample_id'];directory=local/sid;frozen.previous.paired_postcheck(directory,32)
            event=local/'events'/(sid+'.json')
            if sha(event)!=e['record_sha256']:raise ValueError('INHERITED_EVENT_CHANGED')
            raw_request=pilot.request_from_record(read(event))
            if read(directory/'conditions.json')['request_fingerprint']!=raw_request.fingerprint:raise ValueError('RAW_REQUEST_BINDING')
            for method in frozen.c2.ARMS:
                for axis in frozen.c2.AXES:
                    execution=OLD_SOURCE if (sid,method,axis) in reused else source['execution_source']
                    request=replace(raw_request,execution_source=execution)
                    row=pilot.price_record(request,method=method,axis=axis,postcheck=compiled[sid][method+'_'+axis],
                        raw_status='PASS',paired_status='PASS',execution_source=execution,
                        compiler=read(directory/(method+'_'+axis+'_compile.json'))['compiler'])
                    row.update(sampling_source=OLD_SOURCE,raw_source=OLD_SOURCE,raw_request_fingerprint=raw_request.fingerprint,
                               compile_post_source=execution,raw_request=asdict(raw_request))
                    prices.append(row)
        if source!=preflight(args.origin,args.saved,args.old_output)[0]:raise ValueError('SOURCE_INPUT_CHANGED')
        preserved=True;write(out/'prices.json',prices)
        summary=pilot.paired_statistics(prices);write(out/'paired_statistics.json',summary)
        diagnostic=pilot.diagnose(summary,model);write(out/'exploratory_diagnostics.json',diagnostic)
        write(out/'cx_candidate_residuals.json',pilot.cx_candidate_residuals(summary));success=True
    except BaseException as error:
        message=str(error)
        for x in [str(ROOT),str(args.origin),str(args.saved),str(args.old_output),str(args.previous_q16)]:message=message.replace(x,'<task-path>')
        write(out/'failure.json',dict(status='STOP',error_type=type(error).__name__,error=message))
    for directory in sorted(local.glob('q32_sample*')):
        dest=out/'paths'/directory.name;dest.mkdir(parents=True,exist_ok=True)
        for path in directory.glob('*.json'):
            if path.name in ['conditions.json','fusion.json'] or path.name.endswith('_private.json'):continue
            write(dest/path.name,frozen.c5.public_view(read(path)))
    ids=[f'q32_sample{i:02d}' for i in range(8)]
    complete=[sid for sid in ids if len(compiled.get(sid,{}))==4 and (local/sid/'paired_observables.json').exists()]
    calls=sum(read(p).get('compile_calls',0) for p in local.glob('q32_sample*/*_compile.json'))
    result=dict(status='TECHNICAL_PASS_MANDATORY_STOP' if success else 'STOP_FAILURE',source_commit=source['execution_source'],
        original_source=OLD_SOURCE,planned_ids=ids,completed_ids=complete,missing_or_failed_ids=[sid for sid in ids if sid not in complete],
        raw_pass=8 if inherited else 0,raw_reused=8 if inherited else 0,raw_executed_new=0,
        completed_postchecks=sum(r['status']=='PASS' for v in compiled.values() for r in v.values()),
        compile_calls_across_original_and_completion=calls,compile_calls_started_new=calls-(len(inherited['wrappers_reused']) if inherited else 0),
        canonical_occurrence_sampling_calls_started=0,canonical_occurrence_sampling_calls_across_original_and_completion=256 if inherited else 0,
        source_input_preserved=preserved,old_STOP_preserved=True,model_refits=0,ground_solves=0,DF_refits=0,N_evaluations=0,
        quantum_Born_shots=0,calibration_added=0,q_above32=False,holdout_generated=False,planned_whole_paths=8,
        exploratory_status=diagnostic['status'] if diagnostic else 'INCOMPLETE',statistical_qualification='STATISTICALLY_UNQUALIFIED',
        confirmatory_PASS=False,long_q_qualified=False,whole_law_guaranteed=False,RPE_total_cost_evaluated=False,
        authorized_source_handoff=True,retry=0,resume=0,mandatory_stop=True)
    write(out/'result.json',result);write(out/'STOP.json',dict(status=result['status'],mandatory_stop=True,next_stage_authorized=False,retry=0,resume=0))
    write(out/'progress.json',dict(status=result['status']));print(result['status'],flush=True)
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())

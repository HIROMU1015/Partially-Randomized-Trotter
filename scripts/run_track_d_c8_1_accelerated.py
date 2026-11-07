#!/usr/bin/env python3
"""Authorized drained C8 task handoff; compile reserves exactly one CPU."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import replace,asdict
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

import run_track_d_c8_1_q64 as frozen
ROOT=frozen.ROOT
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import track_d_c8_1_handoff as handoff

OUTPUT='artifacts/track_d_c8_1_q64/2026-11-08/run01_accelerated_completion'
TECH='artifacts/track_d_c8_1_q64/2026-11-08/accelerated_technical'
INHERITANCE=TECH+'/fixed_prior_evidence.json'
NEW=('src/trottertracks/track_d_c8_1_handoff.py','scripts/run_track_d_c8_1_accelerated.py',
    'tests/check_track_d_c8_1_handoff.py','docs/research/track_d_c8_1_q64_20261108/ACCELERATED_COMPLETION_PLAN.md',
    TECH+'/synthetic_tests.json',INHERITANCE)
write=frozen.write


def preflight(origin,saved,old):
    source=frozen.preflight(origin,saved);head=source['execution_source']
    if {**source,'execution_source':handoff.OLD_SOURCE}!=handoff.read(Path(old)/'SOURCE.json'):
        raise ValueError('OLD_SCIENTIFIC_INPUT_OR_RUNTIME_CHANGED')
    old_root=Path(old).parents[3]
    for path,want in handoff.read(Path(old)/'SOURCE.json')['C8_source'].items():
        if handoff.sha(old_root/path)!=want:raise ValueError('ACTUAL_OLD_SOURCE_CHANGED:'+path)
    for path in NEW:
        if (ROOT/path).read_bytes()!=frozen.c2.prior().git('show',head+':'+path):raise ValueError('UNFIXED_HANDOFF_SOURCE:'+path)
    if handoff.read(ROOT/TECH/'synthetic_tests.json')['status']!='PASS':raise ValueError('SYNTHETIC_TESTS_REQUIRED')
    packet=handoff.read(ROOT/INHERITANCE)
    if packet!=handoff.prior_plan(old):raise ValueError('FIXED_COMPLETED_INPUTS_CHANGED')
    handoff.verify(old,packet)
    source.update(original_source=handoff.OLD_SOURCE,
        accelerated_source={p:handoff.sha(ROOT/p) for p in NEW},
        source_handoff='user authorized, old workers drained, completed tasks reused byte-for-byte',
        scheduling='compile one CPU; raw/post share remaining native CPU; sample00 gate preserved',
        inherited_raw=1,new_raw_limit=7,inherited_compile=4,new_compile_limit=28,new_sampling=0,
        compiler_changed=False,precision_changed=False)
    return source


def copy_inherited(old,out,packet):
    for row in packet['members']:
        if not row['path'].startswith('local/'):continue
        dest=out/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise ValueError('INHERITED_DEST_ALREADY_EXISTS')
        shutil.copyfile(Path(old)/row['path'],dest)
        if handoff.sha(dest)!=row['sha256']:raise ValueError('INHERITED_COPY_MISMATCH')
    for name in ['seed_plan.json','frozen_models.json','event_bindings.json']:
        shutil.copyfile(Path(old)/name,out/name)


def pipeline(events,origin,local,resources,stop,out,raw,compiled,compile_done):
    pending=deque();active={};released=False;budget=resources['numerical_threads']
    first=events[0]['sample_id']
    def next_wrapper(sid,method,axis):
        name=method+'_'+axis
        if name in compiled.get(sid,{}):return
        stage='post' if (sid,method,axis) in compile_done else 'compile'
        pending.append(dict(stage=stage,sample_id=sid,method=method,axis=axis,
            memory_bytes=resources['post_memory_estimate'] if stage=='post' else resources['compile_memory_estimate']))
    for method in frozen.c2.ARMS:
        for axis in frozen.c2.AXES:next_wrapper(first,method,axis)
    def release_remaining():
        nonlocal released
        if released:return
        if len(compiled.get(first,{}))!=4 or not all(v['status']=='PASS' for v in compiled[first].values()):return
        paired=local/first/'paired_observables.json'
        if not paired.exists():frozen.previous.paired_postcheck(local/first,64)
        write(out/'numerical_preflight.json',dict(status='PASS',sample_id=first,raw_reused=True,
            postchecks=4,paired_XY_chi='PASS',included_in_original32_compile_limit=True))
        released=True
        for e in events[1:]:
            pending.append(dict(stage='raw',sample_id=e['sample_id'],row=e,method=None,axis=None,
                memory_bytes=resources['raw_memory_estimate']))
        print('sample00 full numerical gate PASS; remaining7 released',flush=True)
    release_remaining()
    with ProcessPoolExecutor(max_workers=budget,mp_context=multiprocessing.get_context('spawn')) as pool:
        launches=0
        while pending or active:
            used_cpu=sum(x['threads'] for x in active.values());used_mem=sum(x['task']['memory_bytes'] for x in active.values())
            free_mem=min(frozen.old.frozen.memory_available(),max(0,resources['memory']['MemAvailable']-used_mem))
            if not stop.is_set():
                for task,threads in handoff.allocation(frozen.old.frozen.task_allocation,budget-used_cpu,free_mem,list(pending),budget,len(active)):
                    pending.remove(task)
                    if task['stage']=='raw':future=pool.submit(frozen.raw_worker,task['row'],origin,local,threads,stop)
                    else:
                        args=(origin,local/task['sample_id'],task['method'],task['axis'],threads)
                        future=pool.submit(frozen.old.stage_worker,task['sample_id'],args,task['stage'],stop)
                    active[future]=dict(task=task,threads=threads);launches+=1
                    write(local/'dispatch_progress.json',dict(launches=launches,active=len(active),pending=len(pending),
                        cpu_slots_used=sum(x['threads'] for x in active.values()),threads=threads,
                        latest_task={k:v for k,v in task.items() if k!='row'}))
            if stop.is_set():pending.clear()
            if not active:
                if pending:raise RuntimeError('ACTUAL_MEMORY_ADMISSION_UNAVAILABLE')
                break
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                task=active.pop(future)['task'];sid=task['sample_id'];stage=task['stage']
                if stage=='raw':sid,row=future.result();raw[sid]=row
                else:sid,name,stage,row=future.result()
                print(sid,stage,row['status'],flush=True)
                if row['status']!='PASS':stop.set()
                elif stage=='raw' and not stop.is_set():
                    resources['raw_memory_estimate']=max(resources['raw_memory_estimate'],2*row['peak_RSS_KiB']*1024)
                    for method in frozen.c2.ARMS:
                        for axis in frozen.c2.AXES:next_wrapper(sid,method,axis)
                elif stage=='compile' and not stop.is_set():
                    compile_done.add((sid,task['method'],task['axis']))
                    pending.appendleft(dict(task,stage='post',memory_bytes=resources['post_memory_estimate']))
                elif stage=='post':
                    compiled.setdefault(sid,{})[name]=row
                    if len(compiled[sid])==4 and not stop.is_set():
                        if sid==first:release_remaining()
                        else:frozen.previous.paired_postcheck(local/sid,64)
                write(out/'progress.json',dict(status='ACCELERATED_Q64_PREFLIGHT' if not released else 'ACCELERATED_Q64_PIPELINE',
                    raw_reused=1,raw_pass=sum(v['status']=='PASS' for v in raw.values()),
                    postchecks_pass=sum(len(v) for v in compiled.values()),sample00_gate_pass=released,
                    active=len(active),pending=len(pending),stop_requested=stop.is_set()))
    if not released:raise ValueError('SAMPLE00_NUMERICAL_PREFLIGHT_NOT_PASSED')


def main():
    parser=argparse.ArgumentParser()
    for name in ['origin','saved','old-output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise PermissionError('AUTHORIZED_TASK_HANDOFF_ONLY')
    out=ROOT/OUTPUT
    if out.exists():raise ValueError('FRESH_COMPLETION_NO_RETRY')
    out.mkdir(parents=True);local=out/'local';local.mkdir();(local/'.gitignore').write_text('*\n')
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:os.environ[name]='1'
    os.environ['QISKIT_PARALLEL']='false';os.environ['MPLCONFIGDIR']=str(local/'matplotlib-cache'/'parent')
    source={};packet=None;raw={};compiled={};success=preserved=False
    try:
        frozen.c2.old().bootstrap(args.origin)
        from trottertracks import track_d_c8_1_q64 as pilot
        source=preflight(args.origin,args.saved,args.old_output);write(out/'SOURCE.json',source)
        write(out/'progress.json',dict(status='WAITING_FOR_ORIGINAL_ACTIVE_TASK_DRAIN',sampling_new=0,science_tasks_started_new=0))
        print('fixed SOURCE ready; waiting for original STOP, no new science dispatched',flush=True)
        while not (Path(args.old_output)/'STOP.json').exists():
            # The original controller is already draining. Never signal workers
            # or dispatch dependent tasks before its immutable STOP exists.
            time.sleep(10)
        if source!=preflight(args.origin,args.saved,args.old_output):raise ValueError('FIXED_INPUT_CHANGED_DURING_DRAIN')
        packet=handoff.inheritance_plan(args.old_output)
        handoff.verify(args.old_output,packet)
        write(out/'inherited_evidence.json',packet);copy_inherited(args.old_output,out,packet)
        events=handoff.read(out/'event_bindings.json');models=handoff.read(out/'frozen_models.json')
        for sid in packet['raw_reused']:raw[sid]=handoff.read(local/sid/'precompile_checks.json')
        for sid,m,a in packet['postchecks_reused']:compiled.setdefault(sid,{})[m+'_'+a]=handoff.read(local/sid/(m+'_'+a+'_check.json'))
        resources=handoff.read(Path(args.old_output)/'local/resources.json')
        resources['memory']['MemAvailable']=frozen.old.frozen.memory_available()
        resources['numerical_threads']=len(os.sched_getaffinity(0));quota=Path('/sys/fs/cgroup/cpu.max')
        if quota.exists() and quota.read_text().split()[0]!='max':
            q,period=map(int,quota.read_text().split());resources['numerical_threads']=min(resources['numerical_threads'],max(1,q//period))
        resources['compile_memory_estimate']=resources['raw_memory_estimate']
        resources['raw_memory_estimate']=max(resources['raw_memory_estimate'],2*raw[events[0]['sample_id']]['peak_RSS_KiB']*1024)
        resources['policy']='correct stage CPU reservation; original initial compile/post memory estimate, raw completed-peak admission; no low worker cap'
        write(local/'resources.json',resources)
        with multiprocessing.Manager() as manager:
            stop=manager.Event();pipeline(events,args.origin,local,resources,stop,out,raw,compiled,{tuple(x) for x in packet['compiles_reused']})
        if len(raw)!=8 or any(v['status']!='PASS' for v in raw.values()) or sum(map(len,compiled.values()))!=32:raise ValueError('ALL8_RAW_ALL32_POST_REQUIRED')
        calls=sum(handoff.read(p).get('compile_calls',0) for p in local.glob('q64_sample*/*_compile.json'))
        if calls!=32:raise ValueError('ORIGINAL_PLUS_NEW_MAX32_COMPILE_ACCOUNTING')
        prices=[]
        for e in events:
            sid=e['sample_id'];directory=local/sid;event=local/'events'/(sid+'.json')
            if handoff.sha(event)!=e['record_sha256']:raise ValueError('FIXED_EVENTS_CHANGED')
            request=pilot.request_from_record(handoff.read(event))
            if handoff.read(directory/'conditions.json')['request_fingerprint']!=request.fingerprint:raise ValueError('RAW_REQUEST_BINDING')
            for method in frozen.c2.ARMS:
                for axis in frozen.c2.AXES:
                    execution=handoff.OLD_SOURCE if [sid,method,axis] in packet['postchecks_reused'] else source['execution_source']
                    row=pilot.price_record(replace(request,execution_source=execution),method=method,axis=axis,
                        postcheck=compiled[sid][method+'_'+axis],raw_status='PASS',paired_status='PASS',execution_source=execution,
                        compiler=handoff.read(directory/(method+'_'+axis+'_compile.json'))['compiler'])
                    row.update(sampling_source=handoff.OLD_SOURCE,raw_source=handoff.OLD_SOURCE if sid in packet['raw_reused'] else source['execution_source'],
                        compile_source=handoff.OLD_SOURCE if [sid,method,axis] in packet['compiles_reused'] else source['execution_source'],
                        postcheck_source=execution,raw_request=asdict(request),raw_request_fingerprint=request.fingerprint)
                    prices.append(row)
        write(out/'prices.json',prices);summary=pilot.paired_statistics(prices);write(out/'paired_statistics.json',summary)
        write(out/'cx_candidate_residuals.json',pilot.cx_candidate_residuals(summary));write(out/'model_diagnostics.json',pilot.diagnose(summary,models))
        success=True
    except BaseException as error:
        text=str(error)
        for path in [str(ROOT),str(args.origin),str(args.saved),str(args.old_output)]:text=text.replace(path,'<task-path>')
        write(out/'failure.json',dict(status='STOP',error_type=type(error).__name__,error=text))
    try:
        preserved=bool(source) and source==preflight(args.origin,args.saved,args.old_output)
        if packet:handoff.verify(args.old_output,packet)
    except BaseException:preserved=False
    frozen.collect(out,local)
    calls=sum(handoff.read(p).get('compile_calls',0) for p in local.glob('q64_sample*/*_compile.json'))
    result=dict(status='TECHNICAL_PASS_MANDATORY_STOP' if success and preserved else 'STOP_FAILURE',
        source_commit=source.get('execution_source'),original_source=handoff.OLD_SOURCE,authorized_source_handoff=True,
        raw_pass=sum(v['status']=='PASS' for v in raw.values()),raw_reused=1 if packet else 0,
        raw_executed_new=max(0,len(raw)-(1 if packet else 0)),postchecks_pass=sum(map(len,compiled.values())),
        compile_calls_across_original_and_completion=calls,compile_calls_new=calls-(4 if packet else 0),
        sampling_new=0,total_occurrences_original=512,total_events_original=16384,source_input_preserved=preserved,
        old_STOP_preserved=True,q=64,cut=32,model_refits=0,N_evaluations=0,RPE_total_cost_updated=False,q_above64=False,
        actual_success_guaranteed=False,whole_law_guaranteed=False,long_q_guaranteed=False,retry=0,resume=0,mandatory_stop=True)
    write(out/'result.json',result);write(out/'STOP.json',dict(status=result['status'],mandatory_stop=True,next_stage_authorized=False,next_actor='user/GPT'))
    write(out/'progress.json',dict(status=result['status']));print(result['status'],flush=True)
    return 0 if success and preserved else 1


if __name__=='__main__':raise SystemExit(main())

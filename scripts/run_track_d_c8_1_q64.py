#!/usr/bin/env python3
"""One fresh q64 campaign; sample00 numerical gate precedes seven paths."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time

import run_track_d_c7_3_accelerated as old
ROOT=old.ROOT
previous=old.frozen.previous
c2,c3=old.frozen.c2,old.frozen.c3
OUTPUT='artifacts/track_d_c8_1_q64/2026-11-08/run01'
TECH='artifacts/track_d_c8_1_q64/2026-11-08/technical'
NEW=('src/trottertracks/track_d_c8_1_q64.py','scripts/run_track_d_c8_1_q64.py',
    'tests/check_track_d_c8_1.py','docs/research/track_d_c8_1_q64_20261108/PLAN.md',
    TECH+'/seed_plan.json',TECH+'/frozen_models.json',TECH+'/synthetic_tests.json')
REUSED=('src/trottertracks/track_d_c7_3_q32.py','src/trottertracks/track_d_c7_1_q16.py',
    'src/trottertracks/track_d_c7_3_acceleration.py','src/trottertracks/track_d_state_action_acceleration.py',
    'src/trottertracks/track_d_c3_whole_path.py','src/trottertracks/track_d_whole_path_price.py',
    'src/trottertracks/track_d_h4_reference.py','src/trottertracks/track_d_h4_paired_reference.py',
    'scripts/run_track_d_c7_3_accelerated.py','scripts/run_track_d_c6b.py',
    'tests/track_d_h4_reference_checks.py')
write=previous.write
_CONTEXT=_CAR=None


def preflight(origin,saved):
    source=previous.preflight(origin,saved);p=c2.prior();head=source['execution_source']
    for name in (*NEW,*REUSED):
        if (ROOT/name).read_bytes()!=p.git('show',head+':'+name):raise ValueError('UNFIXED_C8_SOURCE:'+name)
    from trottertracks import track_d_c8_1_q64 as pilot
    if json.loads((ROOT/TECH/'seed_plan.json').read_text())!=pilot.seed_plan():raise ValueError('FIXED_SEEDS_CHANGED')
    if json.loads((ROOT/TECH/'frozen_models.json').read_text())!=pilot.model_packet(ROOT):raise ValueError('FIXED_MODELS_CHANGED')
    if json.loads((ROOT/TECH/'synthetic_tests.json').read_text())['status']!='PASS':raise ValueError('SYNTHETIC_TESTS_REQUIRED')
    source.update(C8_source={n:p.sha((ROOT/n).read_bytes()) for n in (*NEW,*REUSED)},
        policy='one q64/cut32/fresh8 campaign; sample00 raw and4 P0/post gate before remaining7',
        validation_runtime='C7-3 exact gate cache; official Aer double unitary <=9 and batched17 statevector all256 columns',
        precision=dict(component_abs=1e-10,observable_frobenius=2e-10,rtol=0,global_phase_fit=False),
        model_refit=False,new_N=False,full_RPE_total_update=False)
    return source


def raw_worker(row,origin,local,threads,stop):
    global _CONTEXT,_CAR
    directory=Path(local)/row['sample_id'];directory.mkdir()
    os.environ['MPLCONFIGDIR']=str(directory/'matplotlib-cache'/'raw')
    try:
        if stop.is_set():raise RuntimeError('STOP_BEFORE_RAW')
        c2.old().bootstrap(origin)
        from trottertracks import track_d_c8_1_q64 as pilot
        from trottertracks import track_d_c7_3_acceleration as fast
        from threadpoolctl import threadpool_limits
        import numpy as np
        engine=fast.install(threads)
        with threadpool_limits(limits=1):
            if _CONTEXT is None:
                _CONTEXT=pilot.api.c3.paired.prepare_common(origin,c3.load_contract())
                with np.load(Path(local)/'car_context.npz',allow_pickle=False) as z:
                    _CAR=dict(left=z['left'],right=z['right'],atoms={k[5:]:z[k] for k in z.files if k.startswith('atom:')})
            path=Path(local)/'events'/(row['sample_id']+'.json')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['record_sha256']:raise ValueError('FIXED_EVENTS_CHANGED')
            record=json.loads(path.read_text());bundle,request=pilot.prepare_fresh(_CONTEXT,record,_CAR)
            for method in pilot.api.METHODS:
                for axis in pilot.api.AXES:pilot.api.wrapper_for(bundle,request,method=method,axis=axis)
            checks=c2.load('c8_raw_checks',ROOT/'tests/track_d_h4_reference_checks.py')
            result=pilot.api.c3.save_and_check(bundle,directory,checks,threads)
            if result['status']=='PASS':
                write(directory/'structural_features_raw.json',pilot.raw_features(bundle,record))
        result.update(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,state_action_threads=threads)
        write(directory/'precompile_checks.json',result)
        write(directory/'raw_native_metadata_private.json',engine.last_metadata)
        return row['sample_id'],result
    except BaseException as error:
        stop.set();message=str(error)
        for p in [str(ROOT),str(origin),str(local)]:message=message.replace(p,'<task-path>')
        result=dict(status='STOP_FAILURE',error_type=type(error).__name__,error=message)
        write(directory/'failure.json',result);return row['sample_id'],result


def pipeline(events,origin,local,resources,stop,out,raw,compiled):
    """Reuse worker stages and allocator, adding only the sample00 gate."""
    pending=deque();active={};released=False;budget=resources['numerical_threads']
    def raw_task(e):return dict(stage='raw',sample_id=e['sample_id'],row=e,method=None,axis=None,
        memory_bytes=resources['raw_memory_estimate'])
    pending.append(raw_task(events[0]));launches=0
    with ProcessPoolExecutor(max_workers=budget,mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or active:
            cpu_used=sum(x['threads'] for x in active.values());mem_used=sum(x['task']['memory_bytes'] for x in active.values())
            free_mem=min(old.frozen.memory_available(),max(0,resources['memory']['MemAvailable']-mem_used))
            lookup=list(pending)
            proxies=[dict(t,kind='raw' if t['stage'] in ('raw','post') else 'compiled',
                method=t['method'] or 'symmetric_directional',index=i) for i,t in enumerate(lookup)]
            if not stop.is_set():
                for proxy,threads in old.frozen.task_allocation(budget-cpu_used,free_mem,proxies,budget,len(active)):
                    task=lookup[proxy['index']]
                    pending.remove(task)
                    if task['stage']=='raw':f=pool.submit(raw_worker,task['row'],origin,local,threads,stop)
                    else:
                        args=(origin,local/task['sample_id'],task['method'],task['axis'],threads)
                        f=pool.submit(old.stage_worker,task['sample_id'],args,task['stage'],stop)
                    active[f]=dict(task=task,threads=threads);launches+=1
                    write(local/'dispatch_progress.json',dict(launches=launches,active=len(active),pending=len(pending),latest_task={k:v for k,v in task.items() if k!='row'},threads=threads,cpu_slots_used=sum(x['threads'] for x in active.values())))
            if stop.is_set():pending.clear()
            if not active:
                if pending:raise RuntimeError('ACTUAL_MEMORY_ADMISSION_UNAVAILABLE')
                break
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for f in done:
                task=active.pop(f)['task'];stage=task['stage'];sid=task['sample_id']
                if stage=='raw':sid,row=f.result();raw[sid]=row
                else:sid,name,stage,row=f.result()
                print(sid,stage,row['status'],flush=True)
                if row['status']!='PASS':stop.set()
                elif stage=='raw' and not stop.is_set():
                    resources['raw_memory_estimate']=max(resources['raw_memory_estimate'],2*row['peak_RSS_KiB']*1024)
                    for method in c2.ARMS:
                        for axis in c2.AXES:pending.append(dict(stage='compile',sample_id=sid,method=method,axis=axis,memory_bytes=resources['raw_memory_estimate']))
                elif stage=='compile' and not stop.is_set():
                    pending.appendleft(dict(task,stage='post',memory_bytes=resources['post_memory_estimate']))
                elif stage=='post':
                    compiled.setdefault(sid,{})[name]=row
                    if len(compiled[sid])==4 and all(x['status']=='PASS' for x in compiled[sid].values()):
                        previous.paired_postcheck(local/sid,64)
                        if sid==events[0]['sample_id'] and not stop.is_set():
                            write(out/'numerical_preflight.json',dict(status='PASS',sample_id=sid,raw='PASS',postchecks=4,paired_XY_chi='PASS',included_in_compile_limit=True))
                            released=True
                            pending.extend(raw_task(e) for e in events[1:])
                            print('sample00 numerical preflight PASS; remaining7 released',flush=True)
                write(out/'progress.json',dict(status='Q64_NUMERICAL_PREFLIGHT' if not released else 'Q64_REMAINING_PIPELINE',
                    raw_pass=sum(r['status']=='PASS' for r in raw.values()),postchecks_pass=sum(len(v) for v in compiled.values()),
                    active=len(active),pending=len(pending),sample00_gate_pass=released,stop_requested=stop.is_set()))
    if not released:raise ValueError('SAMPLE00_NUMERICAL_PREFLIGHT_NOT_PASSED')


def collect(out,local):
    """Preserve scalar checks and hash local payloads; never upload raw arrays."""
    members=[]
    for p in sorted(local.rglob('*')):
        if not p.is_file():continue
        if p.suffix in ('.qpy','.npz') or p.parent.name=='events':
            members.append(dict(path=str(p.relative_to(local)),bytes=p.stat().st_size,sha256=old.sha(p)))
    write(out/'payload_index.json',dict(members=members,raw_payload_uploaded=False))
    for directory in sorted(local.glob('q64_sample*')):
        if not directory.is_dir():continue
        dest=out/'paths'/directory.name;dest.mkdir(parents=True,exist_ok=True)
        for p in sorted(directory.glob('*.json')):
            if p.name in ('conditions.json','fusion.json') or p.name.endswith('_private.json'):continue
            write(dest/p.name,old.frozen.c5.public_view(json.loads(p.read_text())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--origin',required=True,type=Path);p.add_argument('--saved',required=True,type=Path)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:raise PermissionError('EXPLICIT_ONE_CAMPAIGN_ONLY')
    out=ROOT/OUTPUT
    if out.exists():raise ValueError('FRESH_OUTPUT_NO_RETRY_RESUME')
    out.mkdir(parents=True);local=out/'local';local.mkdir();(local/'.gitignore').write_text('*\n');(local/'events').mkdir()
    os.environ['MPLCONFIGDIR']=str(local/'matplotlib-cache'/'parent')
    os.environ['QISKIT_PARALLEL']='false';os.environ['RAYON_NUM_THREADS']='1'
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    source={};raw={};compiled={};events=[];sampling_started=0;success=preserved=False
    try:
        c2.old().bootstrap(args.origin)
        from trottertracks import track_d_c8_1_q64 as pilot
        from threadpoolctl import threadpool_limits
        import numpy as np
        source=preflight(args.origin,args.saved);write(out/'SOURCE.json',source)
        plan=json.loads((ROOT/TECH/'seed_plan.json').read_text());models=json.loads((ROOT/TECH/'frozen_models.json').read_text())
        write(out/'seed_plan.json',plan);write(out/'frozen_models.json',models)
        budget=len(os.sched_getaffinity(0));quota=Path('/sys/fs/cgroup/cpu.max')
        if quota.exists() and quota.read_text().split()[0]!='max':
            q,period=map(int,quota.read_text().split());budget=min(budget,max(1,q//period))
        memory=old.frozen.memory_available()
        # Initial admission estimate = 16 full all-column state arrays. Update
        # raw memory with the actual completed sample00 peak before seven paths.
        state_bytes=(1<<17)*256*16
        resources=dict(numerical_threads=budget,memory=dict(MemAvailable=memory),
            raw_memory_estimate=16*state_bytes,post_memory_estimate=16*state_bytes,
            affinity_cpus=budget,load=os.getloadavg(),policy='observed affinity/quota/MemAvailable, task memory admission; no artificial CPU/RAM cap')
        write(local/'resources.json',resources)
        write(out/'progress.json',dict(status='SAMPLING',planned_paths=8,occurrences=512))
        with threadpool_limits(limits=1):
            context=pilot.sampling_context(args.origin,c3.load_contract())
            def before_occurrence(sid,occurrence,seed):
                nonlocal sampling_started
                sampling_started+=1;write(local/'sampling_progress.json',dict(calls_started=sampling_started,sample_id=sid,occurrence=occurrence,seed=seed))
            for row in plan['generation_order']:
                record=pilot.sample_record(context,row['sample'],source['execution_source'],before_occurrence=before_occurrence)
                path=local/'events'/(row['sample_id']+'.json');write(path,record)
                events.append(dict(**row,record_sha256=old.sha(path),ordered_events_sha256=pilot.digest(record['ordered_events'])))
            write(out/'event_bindings.json',events)
            if sampling_started!=512:raise ValueError('FIXED_512_OCCURRENCES')
            print('All8 events sealed:512 occurrences/16384 events; sample00 preflight starts',flush=True)
            with np.load(args.origin/c3.load_contract()['input'],allow_pickle=False) as z:packet={k:z[k] for k in ('h','g','constant')}
            car=pilot.calibration.car_context(context,packet)
            np.savez(local/'car_context.npz',left=car['left'],right=car['right'],**{'atom:'+k:v for k,v in car['atoms'].items()})
            del context,car
        with multiprocessing.Manager() as manager:
            stop=manager.Event();pipeline(events,args.origin,local,resources,stop,out,raw,compiled)
        if len(raw)!=8 or sum(len(v) for v in compiled.values())!=32 or any(r['status']!='PASS' for r in raw.values()):raise ValueError('INCOMPLETE_FIXED_COHORT')
        prices=[]
        for row in events:
            sid=row['sample_id'];directory=local/sid;path=local/'events'/(sid+'.json')
            if old.sha(path)!=row['record_sha256']:raise ValueError('FIXED_EVENTS_CHANGED')
            request=pilot.request_from_record(json.loads(path.read_text()))
            for method in c2.ARMS:
                for axis in c2.AXES:
                    prices.append(pilot.price_record(request,method=method,axis=axis,postcheck=compiled[sid][method+'_'+axis],
                        raw_status=raw[sid]['status'],paired_status='PASS',execution_source=source['execution_source'],
                        compiler=json.loads((directory/(method+'_'+axis+'_compile.json')).read_text())['compiler']))
        write(out/'prices.json',prices);summary=pilot.paired_statistics(prices);write(out/'paired_statistics.json',summary)
        write(out/'cx_candidate_residuals.json',pilot.cx_candidate_residuals(summary));write(out/'model_diagnostics.json',pilot.diagnose(summary,models))
        success=True
    except BaseException as e:
        message=str(e)
        for path in (str(ROOT),str(args.origin),str(args.saved)):message=message.replace(path,'<task-path>')
        write(out/'failure.json',dict(status='STOP',error_type=type(e).__name__,error=message))
    try:
        preserved=bool(source) and source==preflight(args.origin,args.saved)
    except BaseException:preserved=False
    success=success and preserved
    collect(out,local)
    compile_calls=sum(json.loads(p.read_text()).get('compile_calls',0) for p in local.glob('q64_sample*/*_compile.json'))
    result=dict(status='TECHNICAL_PASS_MANDATORY_STOP' if success else 'STOP_FAILURE',source_commit=source.get('execution_source'),
        q=64,cut=32,planned_whole_paths=8,raw_pass=sum(r['status']=='PASS' for r in raw.values()),
        postchecks_pass=sum(len(v) for v in compiled.values()),compile_calls_started=compile_calls,
        canonical_occurrences_started=sampling_started,events_sealed=len(events),source_input_preserved=preserved,
        model_refits=0,N_evaluations=0,RPE_total_cost_updated=False,q_above64=False,ground_solves=0,DF_refits=0,
        actual_success_guaranteed=False,whole_law_guaranteed=False,long_q_guaranteed=False,
        retry=0,resume=0,mandatory_stop=True)
    write(out/'result.json',result);write(out/'STOP.json',dict(status=result['status'],mandatory_stop=True,next_stage_authorized=False,next_actor='user/GPT'))
    write(out/'progress.json',dict(status=result['status']));print(result['status'],flush=True)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())

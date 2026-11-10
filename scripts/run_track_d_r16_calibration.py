#!/usr/bin/env python3
"""Fresh fixed 12-path r16 price pilot; reuse one path, <=44 new P0, STOP."""
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import asdict
import argparse
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import run_track_d_r16_q4 as old
from run_track_d_b1_c1 import cpu_observation
ROOT=old.ROOT
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import track_d_r16_calibration as pilot
OUTPUT=ROOT/pilot.FAMILY
PLAN=ROOT/'docs/research/track_d_r16_price_calibration_20261110/sampling_plan.json'
NEW=('src/trottertracks/track_d_r16_calibration.py',
     'scripts/run_track_d_r16_calibration.py','tests/check_track_d_r16_calibration.py',
     'docs/research/track_d_r16_price_calibration_20261110/CALIBRATION_PLAN.md',str(PLAN.relative_to(ROOT)))
REUSED=(*old.NEW_SOURCE,'src/trottertracks/track_d_c6a_pilot.py')
_CONTEXT=_CAR=None


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):return pilot.fixed.digest(Path(path).read_bytes())
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def preflight():
    source=git('rev-parse','HEAD').decode().strip()
    files={n:sha(ROOT/n) for n in dict.fromkeys((*NEW,*REUSED))}
    for n,want in files.items():
        if pilot.fixed.digest(git('show',source+':'+n))!=want:raise ValueError('UNCOMMITTED_SOURCE:'+n)
    for n in REUSED:
        if (ROOT/n).read_bytes()!=git('show',pilot.BASE+':'+n):raise ValueError('FIXED_R16_SOURCE_CHANGED:'+n)
    if read(PLAN)!=pilot.sampling_plan():raise ValueError('COMMITTED_PLAN_BINDING')
    saved=ROOT/pilot.OLD
    binding={}
    for n in ('SOURCE_S1.json','SOURCE_S2.json','S1_result.json','r16_law.json','seed_plan.json',
        'event_binding.json','compiled_results.json','conditions_summary.json','STOP.json','payload_index.json',
        'reference_X_compile.json','reference_Y_compile.json','symmetric_directional_X_compile.json','symmetric_directional_Y_compile.json',
        'raw_checks.json','paired_observables.json'):
        p=saved/n
        if p.read_bytes()!=git('show',pilot.BASE+':'+str(p.relative_to(ROOT))):raise ValueError('SAVED_R16_CHANGED:'+n)
        binding[n]=sha(p)
    s1=read(saved/'S1_result.json');condition=read(saved/'conditions_summary.json')
    if (not s1['S2_allowed'] or not read(saved/'STOP.json')['mandatory_stop']
            or condition['q']!=4 or condition['r']!=16 or condition['cut_after_complete_step']!=2
            or condition['C1_extra_fusion'] or condition['reference_gadgets']!=1
            or read(saved/'raw_checks.json')['status']!='PASS'):
        raise ValueError('R16_SAVED_QUALIFICATION')
    for row in read(saved/'compiled_results.json').values():
        if row['status']!='PASS':raise ValueError('SAVED_POSTCHECK')
    for method in pilot.METHODS:
        for axis in pilot.AXES:
            x=read(saved/(method+'_'+axis+'_compile.json'))
            if x['status']!='PASS' or x['compile_calls']!=1 or x['compiler']!=dict(
                qiskit='1.3.0',optimization_level=1,seed_transpiler=17,basis_gates=['rz','sx','x','cx'],
                approximation_degree=1.,qubits_initially_zero=False):raise ValueError('SAVED_P0')
    payload=saved/'local/S2_attempt02/q4'
    for n,x in read(saved/'payload_index.json')['files'].items():
        p=payload/n
        if p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise ValueError('SAVED_PRIVATE_PAYLOAD:'+n)
    if importlib.metadata.version('qiskit')!='1.3.0':raise ValueError('QISKIT_P0_VERSION')
    return dict(execution_source=source,files=files,saved_r16_commit=pilot.BASE,saved_r16_hashes=binding,
        law_sha256=read(saved/'r16_law.json')['sha256'],
        packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','qiskit','qiskit-aer','symengine')})


def raw_worker(row,local,threads):
    global _CONTEXT,_CAR
    import numpy as np
    from threadpoolctl import threadpool_limits
    from trottertracks import track_d_h4_paired_reference as paired
    from trottertracks import track_d_c3_whole_path as c3
    directory=Path(local)/row['sample_id'];directory.mkdir()
    os.environ['MPLCONFIGDIR']=str(directory/'matplotlib-cache'/'raw')
    tick=time.monotonic()
    try:
        old.install_state_actions()
        with threadpool_limits(limits=1):
            if _CONTEXT is None:
                arm,tail,config,_,_,_,law=pilot.load_arm(ROOT)
                _CONTEXT=paired.context_for_arm(arm,tail,config)
                with np.load(Path(local)/'car_context.npz',allow_pickle=False) as z:
                    _CAR=dict(left=z['left'],right=z['right'],atoms={k[5:]:z[k] for k in z.files if k.startswith('atom:')})
            path=Path(local)/'events'/(row['sample_id']+'.json')
            if sha(path)!=row['record_sha256']:raise ValueError('EVENT_RECORD_CHANGED')
            record=read(path)
            bundle=pilot.build_path(_CONTEXT,record,_CAR)
            result=c3.save_and_check(bundle,directory,old.checks_module(),threads)
        result.update(wall_seconds=time.monotonic()-tick,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(directory/'precompile_checks.json',result)
        return row['sample_id'],result
    except BaseException as error:
        r=dict(status='STOP_FAILURE',error_type=type(error).__name__,error=str(error).replace(str(ROOT),'<task>'))
        save(directory/'failure.json',r);return row['sample_id'],r


def compiled_worker(task,local):
    try:
        key,row=old.compiler_worker(task['method'],task['axis'],Path(local)/task['sample_id'],task['threads'])
        return task['sample_id'],key,row
    except BaseException as error:
        return task['sample_id'],task['method']+'_'+task['axis'],dict(status='STOP_FAILURE',
            error_type=type(error).__name__,error=str(error).replace(str(ROOT),'<task>'))


def available_memory():
    values={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()
            if l.startswith('MemAvailable:')}
    available=values['MemAvailable']
    limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and current.exists() and limit.read_text().strip()!='max':
        available=min(available,max(0,int(limit.read_text())-int(current.read_text())))
    return available


def compile_pipeline(rows,local,budget,peak,compiled):
    """Same compile+post function; CPU-weighted dispatch, no healthy-task kill."""
    pending=[]
    total_steps=2*sum(r['q'] for r in rows)
    for row in sorted(rows,key=lambda x:(-x['q'],x['sample'])):
        for method in pilot.METHODS:
            for axis in pilot.AXES:
                threads=max(1,budget*row['q']//total_steps) if method=='reference' else 1
                pending.append(dict(sample_id=row['sample_id'],method=method,axis=axis,
                    threads=threads,memory=peak*max(1,row['q']/4)))
    active={};failed=False
    max_workers=min(len(pending),budget,max(1,int(available_memory()//peak)))
    with ProcessPoolExecutor(max_workers=max_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or active:
            used=sum(t['threads'] for t in active.values())
            if not failed:
                for task in pending[:]:
                    if len(active)>=max_workers:break
                    if used+task['threads']>budget or task['memory']>available_memory():continue
                    future=pool.submit(compiled_worker,task,local);active[future]=task
                    used+=task['threads'];pending.remove(task)
            else:pending.clear()
            if not active:raise RuntimeError('NO_RESOURCE_SLOT_FOR_PENDING_TASK')
            done,_=wait(active,timeout=20,return_when=FIRST_COMPLETED)
            for f in done:
                active.pop(f)
                sid,key,result=f.result();compiled.setdefault(sid,{})[key]=result
                if result['status']!='PASS':failed=True
                print(sid,key,result['status'],flush=True)
                save(OUTPUT/'progress.json',dict(status='POSTCHECKS',completed=sum(map(len,compiled.values())),
                    planned_new=44,failed=failed,in_flight=len(active),pending=len(pending)))
    if failed:raise ValueError('COMPILE_POSTCHECK_FAILURE')


def prices_for(row,results,source,reused=False):
    return [dict(q=row['q'],sample=row['sample'],sample_id=row['sample_id'],method=m,axis=a,
        metrics={g:results[m+'_'+a]['metrics'][g] for g in pilot.METRICS},
        source=source,reused=reused,cut=row['cut_after_complete_step'],postcheck='PASS')
        for m in pilot.METHODS for a in pilot.AXES]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true',required=True)
    parser.parse_args()
    if OUTPUT.exists():raise ValueError('FRESH_CAMPAIGN_NO_SCIENTIFIC_RETRY_OR_RESUME')
    source=preflight();OUTPUT.mkdir(parents=True);local=OUTPUT/'local';local.mkdir();(local/'events').mkdir()
    os.environ['MPLCONFIGDIR']=str(local/'matplotlib-cache'/'parent')
    os.environ['QISKIT_PARALLEL']='false';os.environ['RAYON_NUM_THREADS']='1'
    save(OUTPUT/'SOURCE.json',source);plan=read(PLAN);save(OUTPUT/'sampling_plan.json',plan)
    raw={};compiled={};bindings=[];calls=0;success=preserved=False;tick=time.monotonic()
    saved_row=next(r for r in plan['rows'] if r['reused'])
    prices=prices_for(saved_row,read(ROOT/pilot.OLD/'compiled_results.json'),pilot.BASE,reused=True)
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        from trottertracks import track_d_h4_paired_reference as paired
        from trottertracks.track_d_c6a_pilot import car_context
        resources=cpu_observation();budget=resources['numerical_threads']
        old_peaks=[r['peak_RSS_KiB']*1024 for r in read(ROOT/pilot.OLD/'compiled_results.json').values()]
        peak=max(old_peaks)
        new=[r for r in plan['rows'] if not r['reused']]
        workers=min(len(new),budget,max(1,available_memory()//(2*peak)))
        resources.update(raw_workers=workers,compile_tasks=44,memory_estimate_from_saved_q4_bytes=peak,
            policy='observed idle CPU / actual memory; weighted by q; fixed task count')
        save(local/'resources.json',resources)
        with threadpool_limits(limits=1):
            arm,tail,config,dist,packet,inputs,law=pilot.load_arm(ROOT)
            save(OUTPUT/'input_identity.json',inputs)
            def before(sid,i,seed):
                nonlocal calls
                calls+=1;save(local/'sampling_progress.json',dict(calls_started=calls,sample_id=sid,occurrence=i,seed=seed))
            for row in new:
                record=pilot.sample_path(row,arm,tail,config,dist,source['execution_source'],law,before_occurrence=before)
                path=local/'events'/(row['sample_id']+'.json');save(path,record)
                bindings.append(dict(**row,record_sha256=sha(path),bytes=path.stat().st_size))
            save(OUTPUT/'event_bindings.json',bindings)
            print('52 occurrences / 832 events SAVED; raw correctness STARTED',flush=True)
            context=paired.context_for_arm(arm,tail,config)
            car=car_context(context,packet)
            np.savez(local/'car_context.npz',left=car['left'],right=car['right'],**{'atom:'+k:v for k,v in car['atoms'].items()})
            del car,context,arm
        save(OUTPUT/'progress.json',dict(status='RAW_CHECKS',planned_new=11,sampling_calls=calls))
        total_q=sum(r['q'] for r in new)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(raw_worker,row,local,max(1,budget*row['q']//total_q)) for row in bindings]
            for f in as_completed(futures):
                sid,result=f.result();raw[sid]=result
                save(OUTPUT/'progress.json',dict(status='RAW_CHECKS',completed=len(raw),planned_new=11,
                    raw_pass=sum(r['status']=='PASS' for r in raw.values())))
                print(sid,'raw',result['status'],flush=True)
        if len(raw)!=11 or any(r['status']!='PASS' for r in raw.values()):raise ValueError('RAW_FAILURE_NO_NEW_COMPILE')
        compile_pipeline(new,local,budget,max(peak,max(r['peak_RSS_KiB']*1024 for r in raw.values())),compiled)
        if sum(map(len,compiled.values()))!=44:raise ValueError('FIXED44_REQUIRED')
        with threadpool_limits(limits=1):
            for row in new:
                directory=local/row['sample_id']
                paired_result=old.paired_postcheck(directory)
                save(directory/'paired_observables.json',paired_result)
                prices.extend(prices_for(row,compiled[row['sample_id']],source['execution_source']))
        for row in bindings:
            if sha(local/'events'/(row['sample_id']+'.json'))!=row['record_sha256']:raise ValueError('EVENT_CHANGED')
        if source!=preflight():raise ValueError('POST_SOURCE_OR_INPUT_CHANGED')
        preserved=True
        save(OUTPUT/'prices.json',prices)
        save(OUTPUT/'paired_statistics.json',pilot.paired_statistics(prices))
        success=True
    except BaseException as error:
        save(OUTPUT/'failure.json',dict(type=type(error).__name__,error=str(error).replace(str(ROOT),'<task>')))
    # Original payload stays local. Publish only results and hash indexes.
    for directory in sorted(local.glob('r16_cal_q*_sample*')):
        dest=OUTPUT/'paths'/directory.name;dest.mkdir(parents=True,exist_ok=True)
        for path in directory.glob('*.json'):
            if path.name in ('conditions.json','fusion.json'):continue
            save(dest/path.name,read(path))
        condition=directory/'conditions.json'
        if condition.exists():save(dest/'conditions_summary.json',{k:v for k,v in read(condition).items() if k!='per_step_events'})
    index={str(p.relative_to(local)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in local.rglob('*')
           if p.is_file() and 'matplotlib-cache' not in p.parts}
    save(OUTPUT/'payload_index.json',dict(local_only=True,files=index))
    compiles=sum(read(p).get('compile_calls',0) for p in local.glob('r16_cal_q*_sample*/*_compile.json'))
    status='FINITE_CAMPAIGN_COMPLETE_MANDATORY_STOP' if success else 'CAMPAIGN_INCOMPLETE_MANDATORY_STOP'
    save(OUTPUT/'result.json',dict(status=status,execution_source=source['execution_source'],
        planned_paths=12,saved_path_reused=1,new_sampling_calls=calls,planned_new_occurrences=52,
        new_raw_pass=sum(r.get('status')=='PASS' for r in raw.values()),new_compiles=compiles,
        new_postchecks_pass=sum(r.get('status')=='PASS' for d in compiled.values() for r in d.values()),
        saved_compiles_reused=4,source_input_preserved=preserved,wall_seconds=time.monotonic()-tick,
        new_PF_RTE_error_evaluations=0,new_N_evaluations=0,new_ground_solves=0,new_DF_fits=0,
        total_RPE_resources_evaluated=False,new_price_model_registered=False,scientific_retry=0,resume=0))
    save(OUTPUT/'STOP.json',dict(mandatory_stop=True,reason=status,next_actor='user/GPT',
        additional_samples_authorized=False,scientific_retry=0,resume=0))
    save(OUTPUT/'progress.json',dict(status=status));print(status,flush=True)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())

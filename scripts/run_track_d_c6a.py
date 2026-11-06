#!/usr/bin/env python3
"""C6a one-shot q2/q4 x16, canonical sampling, <=128 P0 wrappers, STOP."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time

import run_track_d_c5_q2 as c5
c3, c2, ROOT = c5.c3, c5.c2, c5.ROOT
OUTPUT = 'artifacts/track_d_c6a_paired_price/2026-11-06/run01'
NEW = ('src/trottertracks/track_d_c6a_pilot.py', 'scripts/run_track_d_c6a.py',
    'src/trottertracks/track_d_c3_whole_path.py', 'src/trottertracks/track_d_whole_path_price.py',
    'tests/check_track_d_c6a.py', 'docs/research/track_d_c6a_paired_price_20261106/PLAN.md')
_CONTEXT = _CAR = None


def write(path,value):
    c2.write(Path(path),value)


def preflight(origin,saved):
    source=c5.preflight(origin,saved);p=c2.prior()
    for name in NEW:
        if (ROOT/name).read_bytes()!=p.git('show',source['execution_source']+':'+name):
            raise ValueError('UNCOMMITTED_C6A_SOURCE:'+name)
    for leaf in ['SOURCE.json','STOP.json','result.json','prices.json','payload_archive.json','circuit_payload.tar.gz']:
        name=c5.OUTPUT+'/'+leaf
        if (ROOT/name).read_bytes()!=p.git('show','9a6920b8355643701ab7436a4fd3b34862e9e3bd:'+name):
            raise ValueError('C5_PRESERVATION:'+leaf)
    source.update(C6a_source={n:p.sha((ROOT/n).read_bytes()) for n in NEW},
        C5_base='9a6920b8355643701ab7436a4fd3b34862e9e3bd',
        B4_2_reference='07d18bfe1d5145d6f8de026c0d994d5cb1c783c8',
        policy='C6a q2/q4 each16 canonical fresh shared paths; fixed half-step cuts; max128 full P0 wrappers')
    return source


def raw_worker(row,origin,local,threads,stop):
    global _CONTEXT,_CAR
    directory=Path(local)/row['sample_id'];directory.mkdir()
    os.environ['MPLCONFIGDIR']=str(directory/'matplotlib-cache'/'raw')
    try:
        if stop.is_set():raise RuntimeError('STOP_BEFORE_RAW')
        c2.old().bootstrap(origin)
        from threadpoolctl import threadpool_limits
        import numpy as np
        from trottertracks import track_d_c6a_pilot as pilot
        with threadpool_limits(limits=1):
            if _CONTEXT is None:
                _CONTEXT=pilot.api.c3.paired.prepare_common(origin,c3.load_contract())
                with np.load(Path(local)/'car_context.npz',allow_pickle=False) as z:
                    _CAR=dict(left=z['left'],right=z['right'],atoms={k[5:]:z[k] for k in z.files if k.startswith('atom:')})
            event_path=Path(local)/'events'/(row['sample_id']+'.json')
            if hashlib.sha256(event_path.read_bytes()).hexdigest()!=row['record_sha256']:
                raise ValueError('RECORDED_EVENTS_CHANGED')
            record=json.loads(event_path.read_text())
            bundle,request=pilot.prepare_fresh(_CONTEXT,record,_CAR)
            for method in pilot.api.METHODS:
                for axis in pilot.api.AXES:pilot.api.wrapper_for(bundle,request,method=method,axis=axis)
            checks=c2.load('c6a_raw_checks',ROOT/'tests/track_d_h4_reference_checks.py')
            result=pilot.api.c3.save_and_check(bundle,directory,checks,threads)
        result.update(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(directory/'precompile_checks.json',result)
        return row['sample_id'],result
    except BaseException as error:
        stop.set();result=dict(status='STOP_FAILURE',error_type=type(error).__name__,error=str(error))
        write(directory/'failure.json',result);return row['sample_id'],result


def compiled_worker(sample_id,task,stop):
    name,result=c2.worker(task,stop)
    return sample_id,name,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin',required=True,type=Path);parser.add_argument('--saved',required=True,type=Path)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise PermissionError('C6A_EXPLICIT_EXECUTION_ONLY')
    out=ROOT/OUTPUT
    if out.exists():raise ValueError('FRESH_OUTPUT_NO_RETRY_RESUME')
    out.mkdir(parents=True);local=out/'local';local.mkdir();(local/'events').mkdir()
    os.environ['MPLCONFIGDIR']=str(local/'matplotlib-cache'/'parent')
    os.environ['QISKIT_PARALLEL']='false';os.environ['RAYON_NUM_THREADS']='1'
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    source={'execution_source':c2.prior().git('rev-parse','HEAD').decode().strip()}
    raw={};compiled={};events=[];prices=[];success=preserved=False
    plan=None;sampling_started=0
    try:
        source=preflight(args.origin,args.saved);write(out/'SOURCE.json',source)
        c2.old().bootstrap(args.origin)
        import numpy as np
        from threadpoolctl import threadpool_limits
        from trottertracks import track_d_c6a_pilot as pilot
        plan=pilot.seed_plan();write(out/'seed_plan.json',plan)
        resources=c2.prior().cpu_observation();budget=resources['numerical_threads']
        # Empirical q4 Reference peak, rather than a historical worker cap.
        old=json.loads((ROOT/'artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/result.json').read_text())
        q4=old['compiled'].get('4',{})
        peaks=[r['peak_RSS_KiB']*1024 for r in q4.values() if 'peak_RSS_KiB' in r]
        if not peaks:raise ValueError('Q4_MEMORY_EVIDENCE_REQUIRED')
        slots=max(1,resources['memory']['MemAvailable']//max(peaks))
        raw_workers=min(32,budget,slots)
        workers=min(128,budget,slots)
        raw_threads=max(1,budget//raw_workers)
        reference_threads=max(1,budget//workers)
        resources.update(raw_workers=raw_workers,raw_threads=raw_threads,compile_workers=workers,reference_threads=reference_threads,
            empirical_q4_peak_RSS_bytes=max(peaks),policy='observed idle CPU/RAM; tasks and measured memory determine concurrency')
        write(local/'resources.json',resources)
        print('PREPARATION / canonical events STARTED',flush=True)
        with threadpool_limits(limits=1):
            context=pilot.api.c3.paired.prepare_common(args.origin,c3.load_contract())
            def before_occurrence(sample_id,occurrence,seed):
                nonlocal sampling_started
                sampling_started+=1
                write(local/'sampling_progress.json',dict(calls_started=sampling_started,
                    sample_id=sample_id,occurrence=occurrence,seed=seed))
            for row in plan['generation_order']:
                record=pilot.sample_record(context,row['q'],row['sample'],source['execution_source'],
                                           before_occurrence=before_occurrence)
                path=local/'events'/(row['sample_id']+'.json');write(path,record)
                events.append(dict(**row,record_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            write(out/'event_bindings.json',events)
            # All event rows now persisted and fixed before any path circuit.
            with np.load(args.origin/c3.load_contract()['input'],allow_pickle=False) as z:
                packet={k:z[k] for k in ['h','g','constant']}
            car=pilot.car_context(context,packet)
            np.savez(local/'car_context.npz',left=car['left'],right=car['right'],
                     **{'atom:'+k:v for k,v in car['atoms'].items()})
            del car,context
        print('96 occurrences SAVED / 32 raw checks STARTED',flush=True)
        write(out/'progress.json',dict(status='RAW_STARTED',planned_paths=32,event_records=32))
        with multiprocessing.Manager() as manager:
            stop=manager.Event()
            with ProcessPoolExecutor(max_workers=raw_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(raw_worker,row,args.origin,local,raw_threads,stop) for row in events]
                for f in as_completed(futures):
                    key,result=f.result();raw[key]=result
                    write(out/'progress.json',dict(status='RAW_CHECKS',raw={k:r['status'] for k,r in raw.items()}))
                    print(key,'raw',result['status'],flush=True)
                    if result['status']!='PASS':stop.set()
            if len(raw)!=32 or any(r['status']!='PASS' for r in raw.values()):
                raise ValueError('RAW_FAILURE_NO_COMPILE')
            tasks=[(row['sample_id'],(args.origin,local/row['sample_id'],m,a,
                reference_threads if m=='reference' else 1)) for row in events for m in c2.ARMS for a in c2.AXES]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(compiled_worker,sid,task,stop) for sid,task in tasks]
                for f in as_completed(futures):
                    sid,name,result=f.result();compiled.setdefault(sid,{})[name]=result
                    write(out/'progress.json',dict(status='POSTCHECKS',completed=sum(map(len,compiled.values())),planned=128))
                    print(sid,name,result['status'],flush=True)
                    if result['status']!='PASS':stop.set()
            if sum(map(len,compiled.values()))!=128 or any(r['status']!='PASS' for rows in compiled.values() for r in rows.values()):
                raise ValueError('COMPILE_POSTCHECK_FAILURE')
        for row in events:
            directory=local/row['sample_id']
            # Reuse the C3 paired/chi checker at its existing q directory API.
            paired_postcheck(directory,row['q'])
            path=local/'events'/(row['sample_id']+'.json')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['record_sha256']:raise ValueError('EVENTS_CHANGED')
            request=pilot.request_from_record(json.loads(path.read_text()))
            for method in c2.ARMS:
                for axis in c2.AXES:
                    prices.append(pilot.price_record(request,method=method,axis=axis,
                        postcheck=compiled[row['sample_id']][method+'_'+axis],raw_status=raw[row['sample_id']]['status'],
                        paired_status='PASS',execution_source=source['execution_source'],
                        compiler=json.loads((directory/(method+'_'+axis+'_compile.json')).read_text())['compiler']))
        if source!=preflight(args.origin,args.saved):raise ValueError('SOURCE_INPUT_CHANGED')
        preserved=True;write(out/'prices.json',prices)
        summary=pilot.paired_statistics(prices);write(out/'paired_statistics.json',summary)
        write(out/'exploratory_affine.json',pilot.exploratory_affine(summary))
        success=True
    except BaseException as error:
        message=str(error)
        for path in [str(ROOT),str(args.origin),str(args.saved)]:message=message.replace(path,'<task-path>')
        write(out/'failure.json',dict(status='STOP',error_type=type(error).__name__,error=message))
    # Originals stay local; scientific views omit operational observations.
    for directory in sorted(local.glob('q*_sample*')):
        dest=out/'paths'/directory.name;dest.mkdir(parents=True,exist_ok=True)
        for path in sorted(directory.glob('*.json')):
            write(dest/path.name,c5.public_view(json.loads(path.read_text())))
    planned=[r['sample_id'] for r in plan['generation_order']] if plan else [f'q{q}_sample{i:02d}' for q in [2,4] for i in range(16)]
    complete=[sid for sid in planned if (local/sid/'paired_observables.json').exists() and raw.get(sid,{}).get('status')=='PASS' and len(compiled.get(sid,{}))==4
              and all(r.get('status')=='PASS' for r in compiled[sid].values())]
    calls=sum(json.loads(p.read_text()).get('compile_calls',0) for p in local.glob('q*_sample*/*_compile.json'))
    result=dict(status='PILOT_COMPLETE_MANDATORY_STOP' if success else 'STOP_FAILURE',source_commit=source['execution_source'],
        planned_paths=32,raw_pass=sum(r.get('status')=='PASS' for r in raw.values()),planned_ids=planned,
        completed_ids=complete,missing_or_failed_ids=[sid for sid in planned if sid not in complete],
        completed_postchecks=sum(r.get('status')=='PASS' for rows in compiled.values() for r in rows.values()),
        compile_calls_started=calls,canonical_occurrence_sampling_calls_started=sampling_started,
        source_input_preserved=preserved,paired_summary_available=success,holdout_generated=False,
        long_q_qualified=False,whole_law_guaranteed=False,RPE_total_cost_evaluated=False,
        ground_solves=0,DF_refits=0,N_evaluations=0,quantum_shots=0,retry=0,resume=0,mandatory_stop=True)
    write(out/'result.json',result);write(out/'STOP.json',dict(status=result['status'],mandatory_stop=True,
        next_stage_authorized=False,retry=0,resume=0));write(out/'progress.json',dict(status=result['status']))
    print(result['status'],flush=True)
    return 0 if success else 1


def paired_postcheck(directory,q):
    """Adapt only directory shape; C3 checker itself stays unchanged."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix='c6a-paired-',dir=directory.parent) as temp:
        (Path(temp)/f'q{q}').symlink_to(directory.resolve(),target_is_directory=True)
        c3.paired_postcheck(Path(temp),q)


if __name__=='__main__':raise SystemExit(main())

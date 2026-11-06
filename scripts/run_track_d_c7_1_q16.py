#!/usr/bin/env python3
"""One C7-1 q16/n16 run: frozen model, new seeds, raw gate, <=64 P0, STOP."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time

import run_track_d_c6a as previous
import run_track_d_c6b as scheduling
import run_track_d_c6b_accelerated as acceleration
c5, c3, c2, ROOT = previous.c5, previous.c3, previous.c2, previous.ROOT
OUTPUT = 'artifacts/track_d_c7_1_q16/2026-11-07/run01'
TECH = 'artifacts/track_d_c7_1_q16/2026-11-07/technical'
NEW = ('src/trottertracks/track_d_c7_1_q16.py', 'scripts/run_track_d_c7_1_q16.py',
       'tests/check_track_d_c7_1_q16.py', 'docs/research/track_d_c7_1_q16_20261107/PLAN.md',
       'scripts/run_track_d_c6b.py', 'scripts/run_track_d_c6b_accelerated.py',
       'src/trottertracks/track_d_state_action_acceleration.py',
       'src/trottertracks/track_d_c6b_holdout.py',
       TECH+'/frozen_model.json', TECH+'/seed_plan.json', TECH+'/synthetic_tests.json')
_CONTEXT = _CAR = None
write = previous.write


def preflight(origin, saved):
    # Reuse the established scientific input/source/P0 preservation checks.
    source = previous.preflight(origin, saved); p = c2.prior(); head = source['execution_source']
    for name in NEW:
        if (ROOT/name).read_bytes() != p.git('show', head+':'+name):
            raise ValueError('UNCOMMITTED_C7_1_SOURCE:'+name)
    from trottertracks import track_d_c7_1_q16 as pilot
    bindings={}
    for commit,folder,leaves in [
        (pilot.CALIBRATION,pilot.CALDIR,['SOURCE.json','STOP.json','result.json','seed_plan.json',
            'paired_statistics.json','exploratory_affine.json','C6b_planning_proposal.json']),
        ('fed5f458be2f625de4e1b8c98b8cc5da6e551b14',
         'artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion',
         ['SOURCE.json','STOP.json','result.json']),
        ('fed5f458be2f625de4e1b8c98b8cc5da6e551b14',
         'artifacts/track_d_c6b_q8/2026-11-07/technical',['seed_plan.json'])]:
        for leaf in leaves:
            name=folder+'/'+leaf;data=(ROOT/name).read_bytes()
            if data!=p.git('show',commit+':'+name):raise ValueError('HISTORICAL_EVIDENCE_CHANGED:'+name)
            bindings[name]=dict(commit=commit,sha256=p.sha(data))
    if json.loads((ROOT/TECH/'seed_plan.json').read_text())!=pilot.seed_plan():
        raise ValueError('C7_1_PRE_FIXED_SEEDS')
    if json.loads((ROOT/TECH/'frozen_model.json').read_text())!=pilot.frozen_model(ROOT):
        raise ValueError('C7_1_PRE_FIXED_MODEL_WIDTH_COVARIANCE')
    tests=json.loads((ROOT/TECH/'synthetic_tests.json').read_text())
    if tests['status']!='PASS' or tests['scientific_runs']!=0:raise ValueError('ARTIFICIAL_REGRESSION_REQUIRED')
    source['dependency_preflight_policy']=source.pop('policy')
    source.update(C7_1_source={n:p.sha((ROOT/n).read_bytes()) for n in NEW},
        C7_0_base=pilot.BASE,C6a_result=pilot.CALIBRATION,historical_bindings=bindings,
        state_action='worker-local BatchedStateActions, unchanged official Aer double/native callback',
        policy='q16_16_fresh_cut8_single_gadget_path_raw_gate_max64_P0_exploratory_only')
    return source


def raw_worker(row, origin, local, threads, stop):
    global _CONTEXT, _CAR
    directory = Path(local)/row['sample_id']; directory.mkdir()
    os.environ['MPLCONFIGDIR'] = str(directory/'matplotlib-cache'/'raw')
    try:
        if stop.is_set():
            raise RuntimeError('STOP_BEFORE_RAW')
        c2.old().bootstrap(origin)
        acceleration.install_state_actions()
        from threadpoolctl import threadpool_limits
        import numpy as np
        from trottertracks import track_d_c7_1_q16 as pilot
        with threadpool_limits(limits=1):
            if _CONTEXT is None:
                _CONTEXT = pilot.api.c3.paired.prepare_common(origin, c3.load_contract())
                with np.load(Path(local)/'car_context.npz', allow_pickle=False) as z:
                    _CAR = dict(left=z['left'], right=z['right'],
                                atoms={k[5:]:z[k] for k in z.files if k.startswith('atom:')})
            event_path = Path(local)/'events'/(row['sample_id']+'.json')
            if hashlib.sha256(event_path.read_bytes()).hexdigest() != row['record_sha256']:
                raise ValueError('RECORDED_EVENTS_CHANGED')
            record = json.loads(event_path.read_text())
            bundle, request = pilot.prepare_fresh(_CONTEXT, record, _CAR)
            for method in pilot.api.METHODS:
                for axis in pilot.api.AXES:
                    pilot.api.wrapper_for(bundle, request, method=method, axis=axis)
            checks = c2.load('c7_1_raw_checks', ROOT/'tests/track_d_h4_reference_checks.py')
            result = pilot.api.c3.save_and_check(bundle, directory, checks, threads)
        result.update(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      state_action_threads=threads)
        write(directory/'precompile_checks.json', result)
        return row['sample_id'], result
    except BaseException as error:
        stop.set()
        message = str(error)
        for path in [str(ROOT), str(origin), str(local)]:
            message = message.replace(path, '<task-path>')
        result = dict(status='STOP_FAILURE', error_type=type(error).__name__, error=message)
        write(directory/'failure.json', result)
        return row['sample_id'], result


memory_available = scheduling.memory_available
task_allocation = scheduling.task_allocation
compiled_worker = acceleration.compiled_worker


def memory_estimates(previous_q8):
    """Initial q16 admission estimate from completed q8 peaks; not an upper bound."""
    rows={k:[] for k in ['raw','reference','direct']}
    for path in (Path(previous_q8)/'local').glob('q8_sample*/*.json'):
        if path.name!='precompile_checks.json' and not path.name.endswith('_check.json'):continue
        report=json.loads(path.read_text())
        if report.get('status')!='PASS' or 'peak_RSS_KiB' not in report:continue
        kind='raw' if path.name=='precompile_checks.json' else 'reference' if path.name.startswith('reference') else 'direct'
        rows[kind].append(report)
    from trottertracks.track_d_state_action_acceleration import stage_memory_estimates
    q8=stage_memory_estimates(**rows)
    # Circuit length doubles; array dimensions do not. Scaling the entire peak
    # is an initial conservative estimate, not a resource cap or q16 guarantee.
    return {key:2*value for key,value in q8.items()}


def pipeline(events, origin, local, resources, stop, out, raw, compiled):
    budget = resources['numerical_threads']
    pending = deque(dict(kind='raw', row=row, sample_id=row['sample_id'], method=None,
                         memory_bytes=resources['raw_task_memory_estimate_bytes']) for row in events)
    active = {}; launches = 0
    workers = min(80, budget)  # Per-task memory admission below, not a heavy-task cap on cheap tasks.
    resources.update(max_worker_processes=workers,
        scheduling='path raw PASS -> four independent wrapper tasks; CPU remainder distributed at dispatch; future tasks reuse released slots; no running task restart')
    write(Path(local)/'resources.json', resources)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or active:
            cpu_used = sum(row['threads'] for row in active.values())
            memory_used = sum(row['task']['memory_bytes'] for row in active.values())
            memory_free = min(max(0, resources['memory']['MemAvailable']-memory_used), memory_available())
            if not stop.is_set():
                for task, threads in task_allocation(budget-cpu_used, memory_free, list(pending), workers, len(active)):
                    pending.remove(task)
                    if task['kind']=='raw':
                        future = pool.submit(raw_worker, task['row'], origin, local, threads, stop)
                    else:
                        args = (origin, Path(local)/task['sample_id'], task['method'], task['axis'], threads)
                        future = pool.submit(compiled_worker, task['sample_id'], args, stop)
                    active[future] = dict(task=task, threads=threads)
                    launches += 1
                    write(Path(local)/'dispatch_progress.json', dict(launches=launches,
                        latest_task={k:v for k,v in task.items() if k!='row'}, threads=threads,
                        active=len(active), pending=len(pending), cpu_slots_used=sum(r['threads'] for r in active.values())))
            if stop.is_set():
                pending.clear()
            if not active:
                if pending:
                    # Genuine current memory shortage: keep evidence, no replacement run.
                    raise RuntimeError('CURRENT_MEMORY_INSUFFICIENT_FOR_NEXT_TASK')
                break
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)['task']
                result = future.result()
                if task['kind']=='raw':
                    sid, row = result; raw[sid] = row
                    print(sid, 'raw', row['status'], flush=True)
                    if row['status']=='PASS' and not stop.is_set():
                        # Compile queues only after this path's persisted raw PASS.
                        for method in c2.ARMS:
                            for axis in c2.AXES:
                                pending.appendleft(dict(kind='compiled', sample_id=sid, method=method, axis=axis,
                                    memory_bytes=resources['reference_task_memory_estimate_bytes'] if method=='reference'
                                    else resources['direct_task_memory_estimate_bytes']))
                    else:
                        stop.set()
                else:
                    sid, name, row = result
                    compiled.setdefault(sid, {})[name] = row
                    print(sid, name, row['status'], flush=True)
                    if row['status']!='PASS':
                        stop.set()
                write(out/'progress.json', dict(status='RAW_COMPILE_PIPELINE',
                    raw={k:r['status'] for k,r in raw.items()},
                    postchecks={k:{n:r['status'] for n,r in v.items()} for k,v in compiled.items()},
                    raw_pass=sum(r['status']=='PASS' for r in raw.values()),
                    completed_postchecks=sum(r['status']=='PASS' for v in compiled.values() for r in v.values()),
                    planned_paths=16, planned_postchecks=64, stop_requested=stop.is_set()))
    return raw, compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--saved', type=Path, required=True)
    parser.add_argument('--previous-q8', type=Path, required=True)
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute:
        raise PermissionError('C7_1_EXPLICIT_EXECUTION_ONLY')
    out = ROOT/OUTPUT
    if out.exists():
        raise ValueError('FRESH_OUTPUT_NO_RETRY_RESUME')
    out.mkdir(parents=True); local=out/'local'; local.mkdir(); (local/'events').mkdir()
    os.environ['MPLCONFIGDIR'] = str(local/'matplotlib-cache'/'parent')
    os.environ['QISKIT_PARALLEL'] = 'false'; os.environ['RAYON_NUM_THREADS'] = '1'
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '1'
    source = dict(execution_source=c2.prior().git('rev-parse', 'HEAD').decode().strip())
    raw = {}; compiled = {}; events = []; prices = []; sampling_started = 0
    success = preserved = False; plan = None; diagnostic = None
    try:
        c2.old().bootstrap(args.origin)
        from trottertracks import track_d_c7_1_q16 as pilot
        from threadpoolctl import threadpool_limits
        import numpy as np
        source = preflight(args.origin, args.saved); write(out/'SOURCE.json', source)
        plan = json.loads((ROOT/TECH/'seed_plan.json').read_text()); write(out/'seed_plan.json', plan)
        model = json.loads((ROOT/TECH/'frozen_model.json').read_text()); write(out/'frozen_model.json', model)
        resources = c2.prior().cpu_observation()
        resources['memory']['MemAvailable'] = min(resources['memory']['MemAvailable'], memory_available())
        quota = Path('/sys/fs/cgroup/cpu.max')
        if quota.exists() and quota.read_text().split()[0] != 'max':
            q, period = map(int, quota.read_text().split())
            resources['numerical_threads'] = min(resources['numerical_threads'], max(1, q//period))
        resources.update(memory_estimates(args.previous_q8))
        resources['memory_evidence']='completed q8 stage peaks scaled by length ratio2 for initial admission only; not a q16 bound'
        print('canonical q16 events STARTED / fixed model and seeds SAVED', flush=True)
        write(out/'progress.json', dict(status='SAMPLING', calls_started=0))
        with threadpool_limits(limits=1):
            context = pilot.api.c3.paired.prepare_common(args.origin, c3.load_contract())
            def before_occurrence(sid, occurrence, seed):
                nonlocal sampling_started
                sampling_started += 1
                write(local/'sampling_progress.json', dict(calls_started=sampling_started, sample_id=sid,
                                                          occurrence=occurrence, seed=seed))
            for row in plan['generation_order']:
                record = pilot.sample_record(context, row['sample'], source['execution_source'], before_occurrence=before_occurrence)
                path = local/'events'/(row['sample_id']+'.json'); write(path, record)
                events.append(dict(**row, record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   ordered_events_sha256=pilot.digest(record['ordered_events'])))
            write(out/'event_bindings.json', events)
            # Every planned event record fixed before CAR/circuit/price evaluation.
            with np.load(args.origin/c3.load_contract()['input'], allow_pickle=False) as z:
                packet = {k:z[k] for k in ['h','g','constant']}
            car = pilot.calibration.car_context(context, packet)
            np.savez(local/'car_context.npz', left=car['left'], right=car['right'],
                     **{'atom:'+k:v for k,v in car['atoms'].items()})
            del car, context
        print('256 occurrences SAVED / path-gated raw+compile STARTED', flush=True)
        with multiprocessing.Manager() as manager:
            stop = manager.Event()
            pipeline(events, args.origin, local, resources, stop, out, raw, compiled)
        if len(raw)!=16 or any(r['status']!='PASS' for r in raw.values()):
            raise ValueError('RAW_FAILURE_OR_MISSING_PLANNED_PATH')
        if sum(map(len, compiled.values()))!=64 or any(r['status']!='PASS' for v in compiled.values() for r in v.values()):
            raise ValueError('COMPILE_POSTCHECK_FAILURE_OR_MISSING')
        if sampling_started != 256 or sum(json.loads(p.read_text()).get('compile_calls',0)
                for p in local.glob('q16_sample*/*_compile.json')) != 64:
            raise ValueError('FIXED_SAMPLING_OR_COMPILE_ACCOUNTING')
        for row in events:
            sid = row['sample_id']; directory = local/sid
            previous.paired_postcheck(directory, 16)
            path = local/'events'/(sid+'.json')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['record_sha256']:
                raise ValueError('EVENTS_CHANGED')
            request = pilot.request_from_record(json.loads(path.read_text()))
            for method in c2.ARMS:
                for axis in c2.AXES:
                    prices.append(pilot.price_record(request, method=method, axis=axis,
                        postcheck=compiled[sid][method+'_'+axis], raw_status=raw[sid]['status'], paired_status='PASS',
                        execution_source=source['execution_source'],
                        compiler=json.loads((directory/(method+'_'+axis+'_compile.json')).read_text())['compiler']))
        if source != preflight(args.origin, args.saved):
            raise ValueError('SOURCE_INPUT_CHANGED')
        preserved = True; write(out/'prices.json', prices)
        summary = pilot.paired_statistics(prices); write(out/'paired_statistics.json', summary)
        diagnostic = pilot.diagnose(summary, model); write(out/'exploratory_diagnostics.json', diagnostic)
        write(out/'cx_candidate_residuals.json', pilot.cx_candidate_residuals(summary))
        success = True
    except BaseException as error:
        message = str(error)
        for path in [str(ROOT), str(args.origin), str(args.saved), str(args.previous_q8)]:
            message = message.replace(path, '<task-path>')
        write(out/'failure.json', dict(status='STOP', error_type=type(error).__name__, error=message))
    # Public science scalar views only; conditions contain event payload, stay local.
    for directory in sorted(local.glob('q16_sample*')):
        dest = out/'paths'/directory.name; dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.glob('*.json')):
            if path.name in ('conditions.json', 'fusion.json'):
                continue
            write(dest/path.name, c5.public_view(json.loads(path.read_text())))
    planned = [f'q16_sample{i:02d}' for i in range(16)]
    complete = [sid for sid in planned if (local/sid/'paired_observables.json').exists()
        and raw.get(sid, {}).get('status')=='PASS' and len(compiled.get(sid, {}))==4
        and all(r.get('status')=='PASS' for r in compiled[sid].values())]
    calls = sum(json.loads(p.read_text()).get('compile_calls',0) for p in local.glob('q16_sample*/*_compile.json'))
    result = dict(status='TECHNICAL_PASS_MANDATORY_STOP' if success else 'STOP_FAILURE', source_commit=source['execution_source'],
        planned_ids=planned, completed_ids=complete, missing_or_failed_ids=[sid for sid in planned if sid not in complete],
        raw_pass=sum(r['status']=='PASS' for r in raw.values()),
        completed_postchecks=sum(r['status']=='PASS' for v in compiled.values() for r in v.values()),
        compile_calls_started=calls, canonical_occurrence_sampling_calls_started=sampling_started,
        source_input_preserved=preserved, holdout_generated=bool(events), planned_whole_paths=16,
        exploratory_status=diagnostic['status'] if diagnostic else 'INCOMPLETE',
        statistical_qualification='STATISTICALLY_UNQUALIFIED', confirmatory_PASS=False,
        long_q_qualified=False, whole_law_guaranteed=False, RPE_total_cost_evaluated=False,
        model_refits=0, calibration_added=0, q_above16=False, ground_solves=0, DF_refits=0,
        N_evaluations=0, quantum_Born_shots=0, retry=0, resume=0, mandatory_stop=True)
    write(out/'result.json', result)
    write(out/'STOP.json', dict(status=result['status'], mandatory_stop=True, next_stage_authorized=False, retry=0, resume=0))
    write(out/'progress.json', dict(status=result['status']))
    print(result['status'], flush=True)
    return 0 if success else 1


if __name__=='__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""One C6b q8/n32 run: persisted events, path-wise raw gate, <=128 P0, STOP."""
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
c5, c3, c2, ROOT = previous.c5, previous.c3, previous.c2, previous.ROOT
OUTPUT = 'artifacts/track_d_c6b_q8/2026-11-07/run01'
TECH = 'artifacts/track_d_c6b_q8/2026-11-07/technical'
NEW = ('src/trottertracks/track_d_c6b_holdout.py', 'scripts/run_track_d_c6b.py',
       'tests/check_track_d_c6b.py', 'docs/research/track_d_c6b_q8_20261107/PLAN.md',
       TECH+'/frozen_model.json', TECH+'/seed_plan.json', TECH+'/synthetic_tests.json')
_CONTEXT = _CAR = None
write = previous.write


def preflight(origin, saved):
    source = previous.preflight(origin, saved)
    p = c2.prior(); head = source['execution_source']
    for name in NEW:
        if (ROOT/name).read_bytes() != p.git('show', head+':'+name):
            raise ValueError('UNCOMMITTED_C6B_SOURCE:'+name)
    from trottertracks import track_d_c6b_holdout as pilot
    bindings = {}
    for leaf in ['SOURCE.json', 'STOP.json', 'result.json', 'seed_plan.json', 'event_bindings.json',
                 'paired_statistics.json', 'exploratory_affine.json', 'C6b_planning_proposal.json']:
        name = pilot.CALDIR+'/'+leaf
        data = (ROOT/name).read_bytes()
        if data != p.git('show', pilot.CALIBRATION+':'+name):
            raise ValueError('FROZEN_C6A_BINDING:'+leaf)
        bindings[name] = p.sha(data)
    old = json.loads((ROOT/pilot.CALDIR/'result.json').read_text())
    old_seeds = json.loads((ROOT/pilot.CALDIR/'seed_plan.json').read_text())
    if (old['status'] != 'PILOT_COMPLETE_MANDATORY_STOP' or old['holdout_generated'] is not False or
            old_seeds['holdout_generated'] is not False or old_seeds['holdout_domain_reserved'] != pilot.DOMAIN):
        raise ValueError('Q8_HOLDOUT_UNUSED_IDENTITY')
    tests = json.loads((ROOT/TECH/'synthetic_tests.json').read_text())
    if tests['status'] != 'PASS' or tests['scientific_runs'] != 0:
        raise ValueError('SYNTHETIC_REGRESSION_REQUIRED')
    if json.loads((ROOT/TECH/'seed_plan.json').read_text()) != pilot.seed_plan():
        raise ValueError('PRE_FIXED_SEED_PLAN')
    if json.loads((ROOT/TECH/'frozen_model.json').read_text()) != pilot.frozen_model(ROOT):
        raise ValueError('PRE_FIXED_MODEL_FAMILY_WIDTH')
    source.update(C6b_source={n:p.sha((ROOT/n).read_bytes()) for n in NEW},
        C6b_base=pilot.BASE, C6a_result=pilot.CALIBRATION, C6a_frozen_files=bindings,
        C6b_policy='q8_32_fresh_cut4_single_gadget_raw_gate_each_path_max128_P0_exploratory_only',
        C6b_supersedes_old_scope_only_for_new_run=True)
    return source


def raw_worker(row, origin, local, threads, stop):
    global _CONTEXT, _CAR
    directory = Path(local)/row['sample_id']; directory.mkdir()
    os.environ['MPLCONFIGDIR'] = str(directory/'matplotlib-cache'/'raw')
    try:
        if stop.is_set():
            raise RuntimeError('STOP_BEFORE_RAW')
        c2.old().bootstrap(origin)
        from threadpoolctl import threadpool_limits
        import numpy as np
        from trottertracks import track_d_c6b_holdout as pilot
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
            checks = c2.load('c6b_raw_checks', ROOT/'tests/track_d_h4_reference_checks.py')
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


def memory_available():
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    # Respect current namespace cgroup when it exposes a finite limit.
    limit = Path('/sys/fs/cgroup/memory.max'); used = Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and used.exists() and limit.read_text().strip() != 'max':
        available = min(available, max(0, int(limit.read_text())-int(used.read_text())))
    return available


def task_allocation(cpu_free, memory_free, pending, maximum_workers, active_count):
    """No science: distribute observed CPU slots including the remainder."""
    if not pending or cpu_free < 1 or active_count >= maximum_workers:
        return []
    chosen = []; cost = 0
    for task in pending:
        if len(chosen) >= min(cpu_free, maximum_workers-active_count):
            break
        if cost+task['memory_bytes'] <= memory_free:
            chosen.append(task); cost += task['memory_bytes']
    if not chosen:
        return []
    allocations = [1]*len(chosen)
    heavy = [i for i, task in enumerate(chosen) if task['kind']=='raw' or task['method']=='reference']
    if heavy:
        quotient, remainder = divmod(cpu_free-len(chosen), len(heavy))
        for j, i in enumerate(heavy):
            allocations[i] += quotient+(j < remainder)
    return list(zip(chosen, allocations))


def pipeline(events, origin, local, resources, stop, out, raw, compiled):
    budget = resources['numerical_threads']
    pending = deque(dict(kind='raw', row=row, sample_id=row['sample_id'], method=None,
                         memory_bytes=resources['heavy_task_memory_estimate_bytes']) for row in events)
    active = {}; launches = 0
    workers = min(128, budget)  # Per-task memory admission below, not a heavy-task cap on cheap tasks.
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
                        future = pool.submit(previous.compiled_worker, task['sample_id'], args, stop)
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
                                    memory_bytes=resources['heavy_task_memory_estimate_bytes'] if method=='reference'
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
                    planned_paths=32, planned_postchecks=128, stop_requested=stop.is_set()))
    return raw, compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--saved', type=Path, required=True)
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute:
        raise PermissionError('C6B_EXPLICIT_EXECUTION_ONLY')
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
        from trottertracks import track_d_c6b_holdout as pilot
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
        old = json.loads((ROOT/c5.C3_OUTPUT/'result.json').read_text())['compiled']['8']
        old4 = json.loads((ROOT/c5.C3_OUTPUT/'result.json').read_text())['compiled']['4']
        resources.update(heavy_task_memory_estimate_bytes=max(r['peak_RSS_KiB']*1024 for k,r in old.items() if k.startswith('reference')),
            direct_task_memory_estimate_bytes=max(r['peak_RSS_KiB']*1024 for k,r in old4.items() if k.startswith('symmetric')),
            memory_evidence='saved C3 q8 Reference peak and q4 direct peak; q8 direct ru_maxrss inherited an earlier Reference high-water mark')
        print('canonical q8 events STARTED / fixed model and seeds SAVED', flush=True)
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
        if len(raw)!=32 or any(r['status']!='PASS' for r in raw.values()):
            raise ValueError('RAW_FAILURE_OR_MISSING_PLANNED_PATH')
        if sum(map(len, compiled.values()))!=128 or any(r['status']!='PASS' for v in compiled.values() for r in v.values()):
            raise ValueError('COMPILE_POSTCHECK_FAILURE_OR_MISSING')
        for row in events:
            sid = row['sample_id']; directory = local/sid
            previous.paired_postcheck(directory, 8)
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
        success = True
    except BaseException as error:
        message = str(error)
        for path in [str(ROOT), str(args.origin), str(args.saved)]:
            message = message.replace(path, '<task-path>')
        write(out/'failure.json', dict(status='STOP', error_type=type(error).__name__, error=message))
    # Public science scalar views only; conditions contain event payload, stay local.
    for directory in sorted(local.glob('q8_sample*')):
        dest = out/'paths'/directory.name; dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.glob('*.json')):
            if path.name in ('conditions.json', 'fusion.json'):
                continue
            write(dest/path.name, c5.public_view(json.loads(path.read_text())))
    planned = [f'q8_sample{i:02d}' for i in range(32)]
    complete = [sid for sid in planned if (local/sid/'paired_observables.json').exists()
        and raw.get(sid, {}).get('status')=='PASS' and len(compiled.get(sid, {}))==4
        and all(r.get('status')=='PASS' for r in compiled[sid].values())]
    calls = sum(json.loads(p.read_text()).get('compile_calls',0) for p in local.glob('q8_sample*/*_compile.json'))
    result = dict(status='TECHNICAL_PASS_MANDATORY_STOP' if success else 'STOP_FAILURE', source_commit=source['execution_source'],
        planned_ids=planned, completed_ids=complete, missing_or_failed_ids=[sid for sid in planned if sid not in complete],
        raw_pass=sum(r['status']=='PASS' for r in raw.values()),
        completed_postchecks=sum(r['status']=='PASS' for v in compiled.values() for r in v.values()),
        compile_calls_started=calls, canonical_occurrence_sampling_calls_started=sampling_started,
        source_input_preserved=preserved, holdout_generated=bool(events), planned_whole_paths=32,
        exploratory_status=diagnostic['status'] if diagnostic else 'INCOMPLETE',
        statistical_qualification='STATISTICALLY_UNQUALIFIED', confirmatory_PASS=False,
        long_q_qualified=False, whole_law_guaranteed=False, RPE_total_cost_evaluated=False,
        model_refits=0, calibration_added=0, q_above8=False, ground_solves=0, DF_refits=0,
        N_evaluations=0, quantum_Born_shots=0, retry=0, resume=0, mandatory_stop=True)
    write(out/'result.json', result)
    write(out/'STOP.json', dict(status=result['status'], mandatory_stop=True, next_stage_authorized=False, retry=0, resume=0))
    write(out/'progress.json', dict(status=result['status']))
    print(result['status'], flush=True)
    return 0 if success else 1


if __name__=='__main__':
    raise SystemExit(main())

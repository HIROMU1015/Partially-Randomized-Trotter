#!/usr/bin/env python3
"""S2 continuation after saved S1 PASS: one r16/q4 path, <=4 P0, STOP."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from run_track_d_r16_validation import ROOT, OUTPUT, save
sys.path.insert(0, str(ROOT/'src'))

METHODS = ('symmetric_directional', 'reference')
AXES = ('X', 'Y')
NEW_SOURCE = ('scripts/run_track_d_r16_q4.py',
    'src/trottertracks/track_d_r16_validation.py',
    'src/trottertracks/track_d_h4_paired_reference.py',
    'src/trottertracks/track_d_c3_whole_path.py',
    'src/trottertracks/track_d_h4_reference.py',
    'src/trottertracks/c1_h4_strong_baseline.py',
    'src/trottertracks/c1_h4_native_fair.py',
    'src/trottertracks/c1_h6_compiler_precision.py',
    'src/trottertracks/c1_h6_native_precision.py',
    'src/trottertracks/c1_h6_state_action.py',
    'src/trottertracks/c_pr_aware_df_stage_b.py',
    'src/trottertracks/c_pr_aware_df_stage_b2.py',
    'src/trottertracks/c1_stage3_native_bridge.py',
    'src/trottertracks/c1_stage3_connection.py',
    'src/trottertracks/pr_dictionary_validation/gates.py',
    'src/trottertracks/pr_dictionary_validation/circuits.py',
    'src/trottertracks/pr_dictionary_validation/fixture.py',
    'src/trotterlib/rte.py',
    'src/trottertracks/track_d_state_action_acceleration.py',
    'tests/track_d_h4_reference_checks.py')


def read(path):
    return json.loads(path.read_text())


def checks_module():
    name = 'r16_existing_h4_checks'
    spec = importlib.util.spec_from_file_location(name, ROOT/'tests/track_d_h4_reference_checks.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_state_actions():
    from trottertracks import c1_h6_state_action as primitive
    from trottertracks.track_d_state_action_acceleration import BatchedStateActions
    primitive.StateActions = BatchedStateActions


def compiler_worker(method, axis, directory, threads):
    """Unchanged phase-preserving compiler and existing all-input checker."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    from trottertracks import track_d_h4_reference as b
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    install_state_actions()
    directory = Path(directory)
    name = method+'_'+axis
    os.environ['MPLCONFIGDIR'] = str(directory/'matplotlib-cache'/name)
    tick = time.monotonic()
    try:
        if read(directory/'precompile_checks.json')['status'] != 'PASS':
            raise ValueError('RAW_PASS_REQUIRED')
        marker = directory/(name+'_compile.json')
        if marker.exists():
            raise ValueError('NO_RECOMPILE')
        raw = b.load_qpy(directory/'circuits'/(name+'_raw.qpy'))
        profile = CompilerProfile(); profile.validate()
        save(marker, dict(status='STARTED', compile_calls=1, compiler=asdict(profile)))
        with threadpool_limits(limits=1):
            compiled = compile_checked(raw, profile)
        b.save_qpy(directory/'circuits'/(name+'_compiled.qpy'), compiled)
        save(marker, dict(status='PASS', compile_calls=1, compiler=asdict(profile)))
        with np.load(directory/'references.npz', allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files}
        with threadpool_limits(limits=threads):
            obs, result = checks_module().check_compiled(compiled, method, axis, arrays, b, threads)
        np.savez_compressed(directory/(name+'_observable.npz'), observable=obs)
        result.update(method=method, axis=axis, metrics=b.artificial.metrics(compiled),
            wall_seconds=time.monotonic()-tick,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(directory/(name+'_check.json'), result)
        return name, result
    except BaseException as error:
        save(directory/(name+'_failure.json'), dict(status='STOP_FAILURE',
            error_type=type(error).__name__, error=str(error).replace(str(ROOT),'<task>')))
        raise


def prepare_path(source, output):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from trottertracks import track_d_r16_validation as v
    from trottertracks import c_pr_aware_df_stage_b as original
    from trottertracks import track_d_h4_paired_reference as paired
    from trottertracks import track_d_c6a_pilot as old
    from trottertracks import track_d_c3_whole_path as c3
    from trottertracks import track_d_h4_reference as b
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    records = {}
    p = v.fixed(ROOT, v.INPUT, records, v.PACKET, arrays=True)
    d = v.fixed(ROOT, v.CDF, records, v.ENDPOINT, arrays=True)
    arm = original.prepare_arm(**{k:p[k] for k in ('h','g','constant')},
        **{k:d[k] for k in ('J','U')}, role='proposed',
        source_identity=v.ENDPOINT, input_identity=v.PACKET, ld=2)
    tail, config, dist = original.configure(arm, v.H, 16, 0)
    saved = read(OUTPUT/'r16_law.json')['law']
    if (asdict(config) != saved['config'] or tail.lambda_r != saved['lambda_R']
            or arm.plan.extracted_identity != saved['extracted_identity']):
        raise ValueError('S1_S2_LAW_MISMATCH')
    plan = read(OUTPUT/'seed_plan.json')
    if plan != v.seed_plan():
        raise ValueError('PREFIXED_SEEDS_CHANGED')
    steps = []
    record = dict(**plan, source=source, input=v.PACKET, endpoint=v.ENDPOINT,
                  law_sha256=read(OUTPUT/'r16_law.json')['sha256'], ordered_events=[])
    for i, seed in enumerate(plan['seeds']):
        save(output/'sampling_progress.json', dict(calls_started=i+1, occurrence=i, seed=seed))
        events = original.connection.sample_occurrence(tail, dist, 16, seed)
        if len(events) != 16 or any(e.taylor_order or e.product_component_ids or e.phase != 1 for e in events):
            raise ValueError('R16_K0_EVENTS')
        record['ordered_events'].append([e.to_dict() for e in events])
        steps.append(dict(sample=f'{v.SAMPLE_ID}_step{i:02d}', events=events))
        save(output/'events_partial.json', record)
    # Save/fix all 64 events before any circuit or CAR construction.
    save(output/'events.json', record)
    event_hash = v.digest((output/'events.json').read_bytes())
    save(OUTPUT/'event_binding.json', dict(sample_id=v.SAMPLE_ID, source=source,
        bytes=(output/'events.json').stat().st_size, sha256=event_hash,
        q=4, r=16, events=64, seeds=plan['seeds'], cut_after_complete_step=2))
    context = paired.context_for_arm(arm, tail, config)
    # Only the existing CAR factor/atom reference helper is used, not its r32
    # sampler, FreshRequest, fixed cohort or statistics routines.
    car = old.car_context(context, p)
    for step in steps:
        events = step['events']
        step['reference'] = b2.trajectory_reference(arm, car, v.H, events)
        step['raw'] = original.build_kernel(arm, v.H, events, config, builder=context.builder)
    bundle = c3.prepare_steps(context, steps, 2,
        ordered_samples=tuple(s['sample'] for s in steps), cohort=v.DOMAIN,
        fixture_selection='pre-fixed r16/q4 sample00; 4 disjoint occurrence seeds, no replacement')
    bundle.identity.update(r=16, sample_id=v.SAMPLE_ID, events_sha256=event_hash,
        execution_source=source, law_sha256=record['law_sha256'], C1_extra_fusion=False)
    return bundle, event_hash, records


def paired_postcheck(output):
    import numpy as np
    from trottertracks import track_d_h4_reference as b
    from trottertracks.c1_stage3_molecular import component_check
    obs = {}
    for method in METHODS:
        for axis in AXES:
            with np.load(output/(method+'_'+axis+'_observable.npz'),allow_pickle=False) as z:
                obs[method,axis] = z['observable']
    rows = {}
    for axis in AXES:
        r = component_check(obs['reference',axis], obs['symmetric_directional',axis],
                            b.ATOL, label='compiled_paired_'+axis)
        if r['frobenius'] > b.SIGNAL_ATOL:
            raise ValueError('PAIRED_SIGNAL_ALLOWANCE')
        rows[axis] = r
    chi = read(output/'conditions.json')['phase_chi']
    for method in METHODS:
        value = obs[method,'X'][0,0]+1j*obs[method,'Y'][0,0]
        rows[method+'_vacuum_chi'] = component_check(np.array([[value]]),
            np.array([[np.exp(1j*chi)]]), b.ATOL, label='compiled_vacuum_chi_'+method)
    return rows


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-s2', action='store_true', required=True)
    parser.parse_args()
    from trottertracks import track_d_r16_validation as v
    s1 = read(OUTPUT/'S1_result.json')
    if not s1['S2_allowed'] or s1['status'] != 'S1_CONDITIONAL_IDEAL_PASS' or (OUTPUT/'STOP.json').exists():
        raise ValueError('S1_ALL13_PASS_WITHOUT_STOP_REQUIRED')
    output = OUTPUT/'local'/'q4'
    output.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(output/'matplotlib-cache'/'parent')
    os.environ['QISKIT_PARALLEL']='false'
    os.environ['RAYON_NUM_THREADS']='1'
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    hashes={n:v.digest((ROOT/n).read_bytes()) for n in NEW_SOURCE}
    for name in hashes:
        if v.digest(subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)) != hashes[name]:
            raise ValueError('UNCOMMITTED_S2_SOURCE:'+name)
    inherited={p:v.digest((OUTPUT/p).read_bytes()) for p in
        ('S1_result.json','SOURCE_S1.json','input_identity.json','r16_law.json','seed_plan.json')}
    save(OUTPUT/'SOURCE_S2.json',dict(execution_source=source,files=hashes,inherited_S1=inherited,
        S1_execution_source=s1['binding']['source'], packages={n:importlib.metadata.version(n)
            for n in ('qiskit','qiskit-aer','numpy','scipy','gmpy2')}))
    from threadpoolctl import threadpool_limits
    install_state_actions()
    from run_track_d_b1_c1 import cpu_observation
    resources=cpu_observation()
    budget=resources['numerical_threads']
    workers=min(4,budget)
    threads=max(1,(budget-2)//2)
    resources.update(worker_processes=workers,reference_threads=threads,
        raw_reference_threads=budget,task_count=4,
        policy='four fixed tasks; observed idle affinity CPUs; existing batched checker')
    save(output/'resources.json',resources)
    tick=time.monotonic(); results={};success=False
    try:
        with threadpool_limits(limits=1):
            bundle,event_hash,inputs=prepare_path(source,output)
            save(OUTPUT/'progress.json',dict(status='RAW_CHECKS',compile_calls=0))
            from trottertracks.track_d_c3_whole_path import save_and_check
            raw=save_and_check(bundle,output,checks_module(),budget)
        save(output/'precompile_checks.json',raw)
        save(OUTPUT/'raw_checks.json',raw)
        print('raw PASS; four P0 tasks STARTED',flush=True)
        del bundle
        save(OUTPUT/'progress.json',dict(status='COMPILE_POSTCHECK',completed=0,planned=4))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(compiler_worker,m,a,output,threads if m=='reference' else 1)
                     for m in METHODS for a in AXES]
            for f in as_completed(futures):
                key,row=f.result();results[key]=row
                save(OUTPUT/'progress.json',dict(status='POSTCHECKS',completed=len(results),planned=4))
                print(key,row['status'],flush=True)
        if len(results)!=4 or any(r['status']!='PASS' for r in results.values()):
            raise ValueError('ALL4_REQUIRED')
        with threadpool_limits(limits=1):
            paired=paired_postcheck(output)
        save(OUTPUT/'paired_observables.json',paired)
        if v.digest((output/'events.json').read_bytes())!=event_hash:
            raise ValueError('EVENTS_CHANGED')
        for name,want in inherited.items():
            if v.digest((OUTPUT/name).read_bytes())!=want:raise ValueError('S1_CHANGED:'+name)
        if hashes!={n:v.digest((ROOT/n).read_bytes()) for n in hashes}:
            raise ValueError('S2_SOURCE_CHANGED')
        for name,row in inputs.items():
            if v.digest(subprocess.check_output(['git','show',row['commit']+':'+name],cwd=ROOT))!=row['sha256']:
                raise ValueError('S2_INPUT_CHANGED')
        save(OUTPUT/'compiled_results.json',results)
        paired_prices=[]
        for axis in AXES:
            s=results['symmetric_directional_'+axis]['metrics']
            r=results['reference_'+axis]['metrics']
            paired_prices.append(dict(axis=axis,strong=s,reference=r,
                delta_reference_minus_strong={k:r[k]-s[k] for k in s}))
        save(OUTPUT/'resource_comparison.json',dict(rows=paired_prices,
            sample_units=1, XY_independent_samples=False, population_or_total_cost=False))
        success=True
    except BaseException as error:
        save(OUTPUT/'S2_failure.json',dict(type=type(error).__name__,
            error=str(error).replace(str(ROOT),'<task>')))
        raise
    finally:
        for path in output.glob('*_check.json'):
            save(OUTPUT/path.name,read(path))
        for path in output.glob('*_compile.json'):
            save(OUTPUT/path.name,read(path))
        files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=v.digest(p.read_bytes()))
            for p in output.rglob('*') if p.is_file() and 'matplotlib-cache' not in p.parts}
        save(OUTPUT/'payload_index.json',dict(local_only=True,files=files))
        status='S1_S2_CONDITIONAL_PASS_MANDATORY_STOP' if success else 'S2_FAILURE_MANDATORY_STOP'
        save(OUTPUT/'result.json',dict(status=status,source_S2=source,
            source_input_preserved=success,completed_postchecks=len(results),
            wall_seconds=time.monotonic()-tick,quantum_shots_drawn=0,ground_solves=0,
            DF_refits=0,price_fits=0,total_RPE_price_evaluated=False))
        save(OUTPUT/'STOP.json',dict(mandatory_stop=True,reason=status,
            next_actor='user/GPT',retry=0,resume=0))
        print(status,flush=True)


if __name__=='__main__':
    main()

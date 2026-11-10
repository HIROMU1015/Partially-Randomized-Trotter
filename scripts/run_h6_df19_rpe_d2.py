"""Execute only the published eight-candidate H6 D2 scientific scope."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

# Parallelize independent dense exponentials instead of nested BLAS teams.
# This affects this process only. Worker count follows available CPU/RAM.
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import h6_df19_rpe_d2 as d2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blob(commit,path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def worker_resources(ctx, tasks):
    affinity = len(os.sched_getaffinity(0)); load = os.getloadavg()[0]
    idle = max(1,affinity-math.ceil(load))
    mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0])*1024
    matrix_bytes = ctx.full.nbytes
    cached = tasks*matrix_bytes
    memory_workers = max(1,(available-cached)//(20*matrix_bytes))
    workers = max(1,min(tasks,idle,memory_workers))
    return dict(affinity_cpus=affinity,load_average=load,available_memory_bytes=available,
        half_step_cache_bytes=cached,half_step_tasks=tasks,half_step_workers=workers,
        BLAS_threads_per_process=1,choice='parallel independent half-step exponentials; memory/idle observation, no fixed worker cap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args = parser.parse_args(); out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError('Fresh D2 output directory required')
    plan_bytes = (ROOT/d2.PLAN_PATH).read_bytes()
    if plan_bytes != blob(d2.PLAN_COMMIT,d2.PLAN_PATH):
        raise ValueError('Published D2 scientific plan differs')
    plan = json.loads(plan_bytes); d2.validate_plan(plan)
    original = blob(plan['baseline_commit'],plan['baseline_result_path'])
    if ((ROOT/plan['baseline_result_path']).read_bytes() != original
            or hashlib.sha256(original).hexdigest() != plan['baseline_result_sha256']):
        raise ValueError('D1 immutable result differs')
    baseline = json.loads(original)
    if plan['configuration_reuse'] != baseline['configuration']:
        raise ValueError('D1 configuration must be copied without recalculation')
    tail = plan['saved_tail_context']
    index = json.loads(blob(tail['source_index_commit'],tail['source_index_path']))
    info = index['saved_inputs']['H6']; paths = info['inputs']
    for value in paths.values():
        if sha(value['path']) != value['sha256']:
            raise ValueError('Original saved input identity differs')
    context_path = Path(index['T20_reference_files']['H6']['path'])
    if sha(context_path) != tail['sha256']:
        raise ValueError('Saved tail context identity differs')
    context_record = json.loads(context_path.read_bytes())
    if (context_record['binding'] != tail['binding'] or
            {p:context_record['prefixes'][p] for p in ('5','7')} != tail['prefixes']):
        raise ValueError('Saved tail context fields differ')
    fixed = plan['saved_reference_T04']
    fixed_rows = json.loads(blob(fixed['commit'],fixed['path']))
    old_ref = next(x for x in fixed_rows if x['T']==.4 and x['cell']['id']=='B1_p19_q2_2nd')
    if old_ref['reference'] != fixed['reference'] or old_ref['allowance'] != fixed['preserved_empirical_allowance']:
        raise ValueError('Preserved reference/allowance differs')
    files = ['src/trottertracks/h6_df19_rpe_d2.py','scripts/run_h6_df19_rpe_d2.py',
        'scripts/h46_new2_core.py','src/trotterlib/h46_m5_plan.py','src/trotterlib/rte.py',
        'src/trottertracks/h4_df10_rpe_d1.py','src/trottertracks/h4_df10_rpe_connection.py',
        'src/trottertracks/h6_df19_rpe_connection.py','src/trottertracks/c1_stage3_accuracy.py',
        'src/trotterlib/rpe_resource_accounting.py',
        'src/trottertracks/resource_applicability/ax2b_h6_saved_completion_loader_v1.py',
        'src/trottertracks/resource_applicability/ax2b_independent_reference.py']
    sources = {p:sha(ROOT/p) for p in files}
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if any((ROOT/p).read_bytes() != blob(source,p) for p in files):
        raise ValueError('Scientific source files must match the recorded commit')
    out.mkdir(parents=True)
    started = time.monotonic()
    scope = dict(state_restored=False,sector_action_representation_assembled=False,
        new_reference_times=[],new_reference_state_actions=0,saved_reference_times=[.4],
        finite_mean_candidates=0,internal_PR_exact_tail_controls=0,scalar_candidate_evaluations=0,
        D1_point_re_evaluations=0,new_prior_evaluations=0,new_horizon_evaluations=0,
        H_DF_state_regenerations=0,eigensolves=0,trajectory_samples=0,
        circuit_builds=0,circuit_compiles=0,price_fits=0,resource_sums=0,RPE_success_simulations=0)
    result = dict(status='D2_RUNNING',scientific_source=source,plan_commit=d2.PLAN_COMMIT,
        plan_path=d2.PLAN_PATH,plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        D1_result_commit=plan['baseline_commit'],D1_result_sha256=plan['baseline_result_sha256'],
        input_bindings=dict(scientific=plan['scientific_binding'],
            files={k:v['sha256'] for k,v in paths.items()},tail_context_sha256=tail['sha256']),
        configuration=deepcopy(plan['configuration_reuse']),scope=scope,source_files=sources,
        references={},rows=[],errors=[],full_RPE_qualified=False,total_resources=None,mandatory_stop=True)
    try:
        from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
        ham,sector,full,state,metadata,_ = load_h6_snapshot(Path(paths['H6_snapshot.npz']['path']),
            json.loads(Path(paths['H6_snapshot_receipt.json']['path']).read_bytes()),
            json.loads(Path(paths['H6_df_receipt.json']['path']).read_bytes()))
        if metadata['hamiltonian_hash'] != d2.BINDING['hamiltonian'] or metadata['state_hash'] != d2.BINDING['state']:
            raise ValueError('Restored saved H/state binding differs')
        scope['state_restored'] = True
        ctx = d2.make_context(ham,sector,state,tail['prefixes'])
        scope['sector_action_representation_assembled'] = True
        result['sector_dimension'] = len(state)
        print(json.dumps(dict(stage='saved_sector_action_ready',dimension=len(state))),flush=True)
        refs = {.4:dict(reference=deepcopy(fixed['reference']),kind='SAVED_T04_FULL_H_REFERENCE',
            signal_difference=None,preserved_allowance=deepcopy(fixed['preserved_empirical_allowance']),certified=False)}
        for T in plan['planned_scientific_scope']['new_reference_times']:
            scope['new_reference_times'].append(T); scope['new_reference_state_actions'] += 2
            refs[T] = d2.reference_pair(ctx,T)
            result['references'][str(T)] = refs[T]
            print(json.dumps(dict(stage='reference_ready',T=T,pair_difference=refs[T]['signal_difference'])),flush=True)
        result['references']['0.4'] = refs[.4]
        keys = d2.half_step_keys(plan['rows'])
        resources = worker_resources(ctx,len(keys)); result['resources'] = resources
        print(json.dumps(dict(stage='half_step_cache_start',**resources)),flush=True)
        with ThreadPoolExecutor(max_workers=resources['half_step_workers']) as pool:
            jobs = {pool.submit(d2.half_step,ctx,key):key for key in keys}
            for future in as_completed(jobs):
                key,value = future.result(); ctx.half_steps[key] = value
        print(json.dumps(dict(stage='half_step_cache_ready',matrices=len(ctx.half_steps))),flush=True)
        save(out/'progress.json',result)
        # Share the immutable exponential cache and batch all PR columns.
        def candidate_job(row):
            c = row['candidate']; rec = d2.evolve(ctx,c,refs[c['T']]['reference'])
            return d2.analyze(rec,row,plan,state_energy=metadata['solver_energy'],
                state_residual=metadata['solver_residual'],reference_evidence=refs[c['T']])
        with ThreadPoolExecutor(max_workers=min(len(plan['rows']),resources['half_step_workers'])) as pool:
            jobs = {pool.submit(candidate_job,row):row for row in plan['rows']}
            for future in as_completed(jobs):
                planned = jobs[future]; c = planned['candidate']
                scope['finite_mean_candidates'] += 1
                scope['internal_PR_exact_tail_controls'] += int(c['method']=='PR')
                try:
                    row = future.result(); scope['scalar_candidate_evaluations'] += 1
                except Exception as exc:
                    row = dict(candidate_id=c['id'],candidate=c,m=planned['m'],stage='D2',
                        status='NUMERICAL_GATE_FAILURE',reason=str(exc),N_X=None,N_Y=None,
                        compiled_price=None,G=None,full_rpe_energy_success=False)
                    result['errors'].append(dict(candidate_id=c['id'],reason=str(exc)))
                result['rows'].append(row)
                save(out/(c['id']+'.json'),row); save(out/'progress.json',result)
                print(json.dumps(dict(stage='candidate_finished',id=c['id'],
                    status=row.get('ideal_numerical_pilot',row).get('status'),
                    N_X=row.get('ideal_numerical_pilot',row).get('N_X'))),flush=True)
        result['rows'].sort(key=lambda r:d2.IDS.index(r['candidate_id']))
        result['status'] = 'D2_FINITE_MENU_EVALUATED' if not result['errors'] else 'D2_FINITE_MENU_WITH_UNQUALIFIED_CANDIDATES'
    except Exception as exc:
        result['status'] = 'D2_STOP_TECHNICAL_OR_REFERENCE_GATE'
        result['errors'].append(dict(reason=str(exc)))
    result['wall_seconds'] = time.monotonic()-started
    result['source_preserved'] = all(sha(ROOT/p)==h for p,h in sources.items())
    result['inputs_preserved'] = all(sha(v['path'])==v['sha256'] for v in paths.values()) and sha(context_path)==tail['sha256']
    if not result['source_preserved'] or not result['inputs_preserved']:
        result['status'] = 'D2_STOP_SOURCE_OR_INPUT_CHANGED'
    result['D1_preserved'] = (ROOT/plan['baseline_result_path']).read_bytes()==original
    result['plan_preserved'] = (ROOT/d2.PLAN_PATH).read_bytes()==plan_bytes
    if not result['D1_preserved'] or not result['plan_preserved']:
        result['status'] = 'D2_STOP_D1_OR_PLAN_CHANGED'
    save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],rows=len(result['rows']),wall_seconds=result['wall_seconds'],mandatory_stop=True)),flush=True)
    return 0 if result['status']=='D2_FINITE_MENU_EVALUATED' else 1


if __name__=='__main__':
    sys.exit(main())

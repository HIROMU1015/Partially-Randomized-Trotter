"""H6 RPE m6 only: fixed two candidates, no prices or automatic continuation.

Invoke with real inputs only after explicit scientific execution permission.
The published D2 runner and all numerical kernels remain unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_h6_df19_rpe_d2 as io
from trottertracks import h6_df19_rpe_d2 as backend

BASELINE='95d01b6ab1d820921fcd2521324e41e82389713a'
PLAN_PATH='docs/research/h6_df19_rpe_m6_preparation_20261110/PLAN.json'
IDS=('201a8e365ab6f4af6ebc','ef41b65942d56977d6cd')


def validate_m6(plan):
    if plan['baseline_commit'] != BASELINE or plan['scientific_binding'] != backend.BINDING:
        raise ValueError('Fixed m6 baseline and H6 input binding required')
    if tuple(r['candidate']['id'] for r in plan['rows']) != IDS:
        raise ValueError('Exactly the published PF/PR m6 candidates required')
    expected_prior=[-9.695399206821989,-3.1355322139814668]
    config=plan['configuration_reuse']
    expected=dict(base_time=.1,beta_total=.4,beta_stat=.2,alpha_total_per_method=.05,
                  alpha_axis=.05/26,energy_precision_ha=.00159360010199040)
    if (any(config[k]!=v for k,v in expected.items())
            or config['horizon']['maximum_round_index_M']!=12
            or config['horizon']['round_count_M_plus_one']!=13
            or config['energy_prior_ha']!=expected_prior or plan['prior_reuse']!=expected_prior):
        raise ValueError('Reuse the fixed D1/D2 policy without recomputation')
    for i,row in enumerate(plan['rows']):
        c=row['candidate'];method='FULL' if i==0 else 'PR'
        required=dict(system='H6',method=method,formula='4th(m5_best)',T=6.4,q=4,delta=1.6,
            prefix=19 if i==0 else 7,R_outer=None if i==0 else 16,K=None if i==0 else 2)
        if (row['m']!=6 or row['q_RPE']!=64 or row['q_PF']!=4
                or any(c[k]!=v for k,v in required.items())):
            raise ValueError('m6 time/q/partition differs; no fallback menu')


def analyze_m6(record,row,plan,*,state_energy,state_residual,reference_evidence):
    validate_m6(plan)
    if row not in plan['rows']:
        raise ValueError('Row outside the fixed m6 scope')
    record=dict(record,origin='NEW_D2_M6_FINITE_MEAN')
    result=backend.analyze(record,row,plan,state_energy=state_energy,
        state_residual=state_residual,reference_evidence=reference_evidence)
    result.update(stage='D2_M6',price_coverage='NOT_EVALUATED_SEPARATE_PRICE_TASK')
    for lane in ('ideal_numerical_pilot','actual_implementation'):
        result[lane]['stage']='D2_M6'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();out=Path(args.output_dir)
    if out.exists():raise FileExistsError('Fresh m6 output directory required')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    plan_bytes=(ROOT/PLAN_PATH).read_bytes()
    if plan_bytes!=io.blob(source,PLAN_PATH):raise ValueError('Plan differs from the execution SOURCE')
    plan=json.loads(plan_bytes);validate_m6(plan)
    originals={p:io.blob(BASELINE,p) for p in (plan['baseline_result_path'],plan['baseline_coverage_path'])}
    for path,data in originals.items():
        if (ROOT/path).read_bytes()!=data:raise ValueError('Preserve the completed D1/D2 results')
    baseline=json.loads(originals[plan['baseline_result_path']])
    if baseline['configuration']!=plan['configuration_reuse']:raise ValueError('D2 policy differs')
    if any(io.sha(ROOT/p)!=h for p,h in baseline['source_files'].items()):
        raise ValueError('Reuse the completed D2 numerical SOURCE without modification')
    tail=plan['saved_tail_context']
    index=json.loads(io.blob(tail['source_index_commit'],tail['source_index_path']))
    paths=index['saved_inputs']['H6']['inputs']
    context_path=Path(index['T20_reference_files']['H6']['path'])
    if any(io.sha(v['path'])!=v['sha256'] for v in paths.values()) or io.sha(context_path)!=tail['sha256']:
        raise ValueError('Original input/context bytes differ')
    context=json.loads(context_path.read_bytes())
    if context['binding']!=tail['binding'] or context['prefixes']['7']!=tail['prefixes']['7']:
        raise ValueError('Saved H6 tail binding differs')
    files=list(baseline['source_files'])+['scripts/run_h6_df19_rpe_m6.py',PLAN_PATH]
    hashes={p:io.sha(ROOT/p) for p in files}
    if any((ROOT/p).read_bytes()!=io.blob(source,p) for p in files):raise ValueError('Uncommitted scientific SOURCE')
    scope=dict(new_reference_times=[],reference_state_actions=0,finite_mean_candidates=0,
        internal_PR_exact_tail_controls=0,scalar_candidate_evaluations=0,D1_D2_point_re_evaluations=0,
        prior_evaluations=0,horizon_evaluations=0,H_DF_state_regenerations=0,eigensolves=0,
        trajectory_samples=0,circuit_builds=0,circuit_compiles=0,price_lookups=0,price_fits=0,
        price_predictions=0,resource_sums=0,RPE_success_simulations=0)
    result=dict(status='M6_RUNNING',scientific_source=source,baseline_commit=BASELINE,
        configuration=deepcopy(plan['configuration_reuse']),source_files=hashes,scope=scope,
        scientific_binding=plan['scientific_binding'],reference=None,rows=[],errors=[],
        input_bindings=dict(files={k:v['sha256'] for k,v in paths.items()},tail_context_sha256=tail['sha256']),
        previous_rounds=json.loads(originals[plan['baseline_coverage_path']])['rows'],
        full_RPE_qualified=False,total_resources=None,mandatory_stop=True)
    out.mkdir(parents=True);started=time.monotonic()
    try:
        from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
        ham,sector,full,state,metadata,_=load_h6_snapshot(Path(paths['H6_snapshot.npz']['path']),
            json.loads(Path(paths['H6_snapshot_receipt.json']['path']).read_bytes()),
            json.loads(Path(paths['H6_df_receipt.json']['path']).read_bytes()))
        if metadata['hamiltonian_hash']!=backend.BINDING['hamiltonian'] or metadata['state_hash']!=backend.BINDING['state']:
            raise ValueError('Restored H/state differs')
        ctx=backend.make_context(ham,sector,state,{'7':tail['prefixes']['7']})
        scope['new_reference_times']=[6.4];scope['reference_state_actions']=2
        reference=backend.reference_pair(ctx,6.4);result['reference']=reference
        print(json.dumps(dict(stage='m6_reference_ready',pair_difference=reference['signal_difference'])),flush=True)
        keys=backend.half_step_keys(plan['rows']);resources=io.worker_resources(ctx,len(keys))
        result['resources']=resources
        with ThreadPoolExecutor(max_workers=resources['half_step_workers']) as pool:
            for key,value in pool.map(lambda key:backend.half_step(ctx,key),keys):ctx.half_steps[key]=value
        def job(row):
            rec=backend.evolve(ctx,row['candidate'],reference['reference'])
            return analyze_m6(rec,row,plan,state_energy=metadata['solver_energy'],
                state_residual=metadata['solver_residual'],reference_evidence=reference)
        with ThreadPoolExecutor(max_workers=min(2,resources['half_step_workers'])) as pool:
            jobs={pool.submit(job,row):row for row in plan['rows']}
            for future in as_completed(jobs):
                planned=jobs[future];c=planned['candidate'];scope['finite_mean_candidates']+=1
                scope['internal_PR_exact_tail_controls']+=int(c['method']=='PR')
                try:
                    row=future.result();scope['scalar_candidate_evaluations']+=1
                except Exception as exc:
                    row=dict(candidate_id=c['id'],candidate=c,m=6,status='NUMERICAL_GATE_FAILURE',
                        reason=str(exc),N_X=None,N_Y=None,C=None,G=None,full_rpe_energy_success=False)
                    result['errors'].append(dict(candidate_id=c['id'],reason=str(exc)))
                result['rows'].append(row);io.save(out/(c['id']+'.json'),row)
                io.save(out/'progress.json',result)
        result['rows'].sort(key=lambda row:IDS.index(row['candidate_id']))
        result['status']='M6_TWO_CANDIDATES_EVALUATED' if not result['errors'] else 'M6_WITH_UNQUALIFIED_CANDIDATES'
    except Exception as exc:
        result['status']='M6_STOP_TECHNICAL_OR_REFERENCE_GATE';result['errors'].append(dict(reason=str(exc)))
    result['wall_seconds']=time.monotonic()-started
    result['source_preserved']=all(io.sha(ROOT/p)==h for p,h in hashes.items())
    result['input_preserved']=all(io.sha(v['path'])==v['sha256'] for v in paths.values()) and io.sha(context_path)==tail['sha256']
    result['previous_results_preserved']=all((ROOT/p).read_bytes()==data for p,data in originals.items())
    if not all(result[k] for k in ('source_preserved','input_preserved','previous_results_preserved')):
        result['status']='M6_STOP_SOURCE_OR_INPUT_CHANGED'
    io.save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],rows=len(result['rows']),mandatory_stop=True)),flush=True)
    return 0 if result['status']=='M6_TWO_CANDIDATES_EVALUATED' else 1


if __name__=='__main__':sys.exit(main())

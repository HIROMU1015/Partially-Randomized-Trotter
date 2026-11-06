"""Prepared C0 price continuation. Never stops or edits the previous process.

Run only after permission for a planned source switch and after the old worker
has exited. Reuse completed price rows and all saved signal/baseline checks.
No new state solve, fit, signal validation, condition search or price proxy.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_o1_m1_c0_pilot as serial
from trottertracks.o1_m1_parallel_price import ParallelMatchedNoPrepCostProvider

C0=dict(delta=.05,r=2,K=2,q_values=[1,2,4,8,16,32,64],epsilon_E=.1,alpha_total=.05,beta_RPE=.30,
    phase_budgets=dict(PF=.06,RTE=.03,stat=.15,model=.03,implementation=.01,reserve=.02),
    alpha_axis=.05/14,energy_interval=[-10.,0.],state_field='state',state_preparation_included=False,
    identity_policy='faithful_identity_in_tail',baseline_substeps=4,
    baseline_basis_policy='exact_spin_lift_spatial_eigh_v1',relative_fusion=True,
    compiler=dict(version='1.3.0',basis=['rz','sx','x','cx'],optimization=1,seed=7,backend=None),
    classical_MC_samples=32,classical_MC_seed='2026110600+100*arm_index+m',price_relative_SE_limit=.1)
ARMS=('P','DF','strong_DF_PF')
STRONG_ARM='strong_DF_PF_4_substeps_spin_fused'


def require_stopped(pid):
    if type(pid) is not int or pid<1: raise ValueError('Explicit previous worker PID required')
    try: os.kill(pid,0)
    except ProcessLookupError: return
    raise RuntimeError('Previous worker still exists; do not duplicate scientific work')


def validate_prior(prior):
    """Data-only completeness/context checks, no input decode or calculations."""
    if prior.get('status')!='RUNNING' or prior.get('mandatory_stop'):
        raise ValueError('Only a deliberately interrupted unfinished run may be continued')
    if prior.get('configuration')!=C0 or prior['provenance']['base']!=serial.BASE:
        raise ValueError('Previous scientific conditions differ from fixed C0')
    if len(prior.get('signal_rounds',[]))!=7 or len(prior.get('baseline_rounds',[]))!=7:
        raise ValueError('All fixed signal/baseline checks must already be complete')
    for m,row in enumerate(prior['signal_rounds']):
        if row['round_index']!=m or row['q']!=1<<m or any(not row['arms'][a]['feasible'] for a in ('P','DF')):
            raise ValueError('Missing or unqualified PR signal round')
    if any(not row['account']['feasible'] for row in prior['baseline_rounds']):
        raise ValueError('Unqualified baseline round')
    rows={}
    for row in prior.get('priced_rounds',[]):
        key=(row['round_index'],row['arm'])
        if key in rows or key[0] not in range(7) or key[1] not in ARMS:
            raise ValueError('Duplicate or unknown completed point')
        if not row['account']['feasible'] or row['account']['round_total_cost'] is None:
            raise ValueError('Unqualified completed point')
        if row['arm']!='strong_DF_PF' and max(row['CX_relative_sample_SE'].values())>.1:
            raise ValueError('Previous MC price did not qualify')
        rows[key]=row
    return rows,[(m,arm) for m in range(7) for arm in ARMS if (m,arm) not in rows]


def saved_bounds(prior,signal_payload,m,name,preparation,config,bridge,accounting):
    spec=accounting.RPERoundSpecification(m,.05)
    if name=='strong_DF_PF':
        old=prior['baseline_rounds'][m]['account']
        bounds={key:old[key] for key in ('reference_radius_lower_bound','pf_phase_bound',
            'model_phase_bound','implementation_phase_bound','radius_provenance')}
        bounds['corrected_signal_error_bound']=0.
    else:
        old=prior['signal_rounds'][m]['arms'][name]
        if signal_payload['status']!='PASS': raise ValueError('Prior signal validator did not pass')
        bounds=bridge.observed_signal_bounds(preparation,config,spec,signal_payload['records'][0],
            **prior['signal_rounds'][m]['numerical_allowances'])
    arm=STRONG_ARM if name=='strong_DF_PF' else preparation.preparation_hash
    if old['context_hash']!=preparation.context_hash or old['arm_hash']!=arm:
        raise ValueError('Physical/preparation cohort differs from original run')
    allocation=accounting.RPEErrorAllocation(.06,.03,.15,.05/14,.05/14)
    bare=bridge.account_radius_round(context_hash=preparation.context_hash,arm_hash=arm,
        specification=spec,allocation=allocation,beta_rpe=.30,cost_metric='cx_count',config=config,**bounds)
    if not bare['feasible'] or bare['shots']!=old['shots']:
        raise ValueError('Continuation does not reproduce the saved C0 shot conditions')
    return bounds


def run(input_path,prior_path,output_dir,*,previous_pid,max_workers=None):
    require_stopped(previous_pid)
    original_bytes=prior_path.read_bytes();prior=json.loads(original_bytes)
    completed,pending=validate_prior(prior)
    if output_dir.exists(): raise FileExistsError('Fresh continuation output directory required')
    if hashlib.sha256(input_path.read_bytes()).hexdigest()!=serial.INPUT_SHA:
        raise ValueError('Saved scientific input differs')
    adapter,bridge,signals,accounting,reference=serial.modules()
    from trotterlib.rte import CompilerSettings
    from trottertracks import c1_h4_strong_baseline as strong
    import subprocess
    fixed=('src/trotterlib/o1_m1_resource_bridge.py','src/trotterlib/o1_m1_matched_split.py',
           'src/trotterlib/rte_compiled_cost.py','src/trotterlib/rpe_hadamard_interrogation.py',
           'src/trottertracks/c1_h4_strong_baseline.py')
    for path in fixed:
        if (ROOT/path).read_bytes()!=subprocess.check_output(['git','show',serial.BASE+':'+path],cwd=ROOT):
            raise ValueError('Fixed source changed: '+path)
    split=adapter.prepare_matched_a(adapter.load_h4_a_snapshot(input_path))
    preparations={arm:bridge.prepare_matched_resources(split,dictionary=arm) for arm in ('P','DF')}
    laws={arm:bridge.configure_law(p,delta_time=.05,r=2,K=2) for arm,p in preparations.items()}
    provider=ParallelMatchedNoPrepCostProvider(CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,7,'1.3.0'),
                                             max_workers=max_workers)
    if provider.compiler.qiskit_version!='1.3.0': raise ValueError('Fixed compiler version required')
    import importlib.metadata
    if importlib.metadata.version('qiskit')!='1.3.0': raise ValueError('Actual Qiskit differs')
    # Validate all carried price contexts without recalculating their prices.
    for (m,name),row in completed.items():
        p=preparations['DF' if name=='strong_DF_PF' else name]
        md=dict(row['cost']['metadata'])
        arm=STRONG_ARM if name=='strong_DF_PF' else p.preparation_hash
        law=bridge._law_hash(None if name=='strong_DF_PF' else laws[name][0])
        if (md['context_hash']!=p.context_hash or md['arm_hash']!=arm or md['law_hash']!=law
                or md['q']!=1<<m or md['delta_time']!=.05 or md['backend_hash']!='no_backend'
                or md['state_preparation_included'] is not False
                or row['account']['compiler_hash']!=bridge.compiler_settings_hash(provider.compiler)
                or row['cost']['circuit_cost_scope']!=bridge.RPE_HADAMARD_INTERROGATION_SCOPE):
            raise ValueError('Completed price belongs to another scientific context')
    result=deepcopy(prior)
    sources=fixed+('scripts/run_o1_m1_c0_parallel_resume.py','scripts/run_o1_m1_c0_pilot.py',
                   'src/trottertracks/o1_m1_parallel_price.py')
    result['parallel_continuation']=dict(prior_result=str(prior_path),previous_pid=previous_pid,
        prior_result_sha256=hashlib.sha256(original_bytes).hexdigest(),reused_points=len(completed),
        source_files_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        new_execution_records=[],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    result['status']='RUNNING_PARALLEL_CONTINUATION'
    output_dir.mkdir(parents=True)
    serial.dump(output_dir/'result.json',result)
    for m,name in pending:
        print(f'parallel price m={m} q={1<<m} arm={name}',flush=True)
        p=preparations['DF' if name=='strong_DF_PF' else name]
        config=None if name=='strong_DF_PF' else laws[name][0]
        payload=None if name=='strong_DF_PF' else json.loads((prior_path.parent/f'signal_m{m}.json').read_text())
        bounds=saved_bounds(prior,payload,m,name,p,config,bridge,accounting)
        arm=STRONG_ARM if name=='strong_DF_PF' else p.preparation_hash
        if name=='strong_DF_PF':
            circuit=bridge.build_strong_df_baseline(split,delta_time=.05,q=1<<m,substeps=4,basis_policy=strong.SPIN)
            cost=provider.deterministic(circuit,context_hash=p.context_hash,arm_hash=arm,q=1<<m,delta_time=.05)
        else:
            cost=provider.monte_carlo(p,config,laws[name][1],q=1<<m,sample_count=32,
                seed=2026110600+100*('P','DF').index(name)+m)
        account=bridge.account_radius_round(context_hash=p.context_hash,arm_hash=arm,
            specification=accounting.RPERoundSpecification(m,.05),
            allocation=accounting.RPEErrorAllocation(.06,.03,.15,.05/14,.05/14),beta_rpe=.30,
            cost_metric='cx_count',config=config,cost=cost,**bounds)
        row=dict(arm=name,round_index=m,cost=asdict(cost),account=account)
        if name!='strong_DF_PF':
            row['CX_relative_sample_SE']={axis:getattr(cost,axis+'_standard_error').cx_count/
                getattr(cost,axis+'_expected_cost').cx_count for axis in ('cosine','sine')}
        result['priced_rounds'].append(row)
        result['parallel_continuation']['new_execution_records'].append(dict(m=m,arm=name,**provider.last_execution))
        if not account['feasible'] or (name!='strong_DF_PF' and max(row['CX_relative_sample_SE'].values())>.1):
            result.update(status='C0_NOT_QUALIFIED',mandatory_stop=True,stop_reason='Fixed C0 price qualification failed')
            serial.dump(output_dir/'result.json',result);return result
        serial.dump(output_dir/'result.json',result)
    result['total_cost']={name:bridge.aggregate_registered_rounds(
        [r['account'] for r in sorted(result['priced_rounds'],key=lambda r:r['round_index']) if r['arm']==name],
        required_max_round=6,total_alpha_budget=.05) for name in ARMS}
    result.update(status='C0_CONDITIONAL_PILOT_COMPLETE',mandatory_stop=True,
                  total_cost_comparison='conditional mean estimates only; no price CI or definitive advantage')
    serial.dump(output_dir/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--prior-result',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--previous-pid',type=int,required=True)
    p.add_argument('--max-workers',type=int)
    a=p.parse_args()
    result=run(a.input.resolve(),a.prior_result.resolve(),a.output_dir.resolve(),
               previous_pid=a.previous_pid,max_workers=a.max_workers)
    return 0 if result['status']=='C0_CONDITIONAL_PILOT_COMPLETE' else 1


if __name__=='__main__': raise SystemExit(main())

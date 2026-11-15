"""One approved Full PF/S2 (16h/7) m12=1 Q3 from the saved 2-shot point (a006b8e15).

Reuses the fixed m12=2 caller (byte-identical) for SOURCE checks, the PF worker,
supervisor, cache acceptance and result checks; only the case and namespace
are new. No search, signal, compile, price fit or other allocation.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_saved_full_pf_two_caller',ROOT/'scripts/run_full_pf_s2_m12_two_fixed.py')
Q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=Q;spec.loader.exec_module(Q)
C,F,R,T,M=Q.C,Q.F,Q.R,Q.T,Q.M
BASE='a006b8e15f86ee42ed66f89fbd1def721d027e87'  # saved 2-shot input, caller and API
OLD_INPUT=Q.INPUT
OLD_DIRECTORY=Q.DIRECTORY
TWO_COST=559457386
VECTOR=[59,55,51,47,43,39,35,31,27,23,19,13,1]
EXPECTED_COST=531533828
NEW_LABEL='FULL_PF_S2_M12_ONE_FIXED'
NEW_ALLOCATION='FULL_PF_S2_16H_OVER7_M12_ONE_01'
DIRECTORY='docs/research/full_pf_s2_m12_one_results_20261116'
INPUT=DIRECTORY+'/INPUTS.json'
CALLER='scripts/run_full_pf_s2_m12_one_fixed.py'
MEETS=C.PASS
EXCEEDS='MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'


def check_source():
    Q.check_source();path='scripts/run_full_pf_s2_m12_two_fixed.py'
    if (ROOT/path).read_bytes()!=R.blob(BASE,path):raise ValueError('Reused 2-shot caller SOURCE differs')


def build_inputs():
    check_source();prior=F.document(BASE,OLD_INPUT);case=deepcopy(prior['cases'][0]);two=deepcopy(case)
    if [v['N_X'] for v in two['allocation']['shots']]!=Q.VECTOR or two['conditional_cost']!=TWO_COST:
        raise ValueError('Saved 2-shot basis differs')
    case.update(label=NEW_LABEL,edit=dict(m=12,old_N_X=2,old_N_Y=2,new_N_X=1,new_N_Y=1),
        parent_allocation=deepcopy(two['allocation']),
        allocation=dict(allocation_id=NEW_ALLOCATION,shot_policy='USER_SUPPLIED',
            shots=[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(VECTOR)]))
    for term in case['price_terms']:
        term['N']=VECTOR[term['m']];term['N_times_C']=term['N']*term['C']
    case['conditional_cost']=sum(t['N_times_C'] for t in case['price_terms'])
    case['conditional_cost_m0_m11']=sum(t['N_times_C'] for t in case['price_terms'] if t['m']<12)
    case['conditional_cost_m12']=sum(t['N_times_C'] for t in case['price_terms'] if t['m']==12)
    case['reference_cost_RZ']=TWO_COST;case['reduction_RZ']=TWO_COST-case['conditional_cost']
    if (case['conditional_cost']!=EXPECTED_COST or case['allocation']['shots'][:12]!=two['allocation']['shots'][:12]
            or case['conditional_cost_m0_m11']!=two['conditional_cost_m0_m11']
            or case['reduction_RZ']!=sum(t['C'] for t in case['price_terms'] if t['m']==12)):
        raise ValueError('Adopted 1-shot model cost or edit differs')
    case['finite_scope_preflight']=F.count_bounds(case['allocation']['shots'])
    b=case['finite_scope_preflight'];o=prior['options']
    if (b['maximum_round_outcomes']>o['maximum_round_outcomes'] or b['verification_pairs_upper']>o['maximum_verification_pairs']
            or b['transition_pairs_upper']>o['total_state_refinement']['maximum_transition_pairs']):
        raise ValueError('Count-only scope exceeded')
    F.metadata(case)
    inputs=deepcopy(prior)
    ref=F.ref(BASE,OLD_DIRECTORY+'/run01/'+two['label']+'/API_RETURN.json')
    inputs['saved_cache'].append(dict(case=two,API_source=ref))
    inputs['comparisons'].append(dict(m12=2,cost_RZ=TWO_COST,API_source=ref))
    inputs.update(cases=[case],previous_achieved_cost_RZ=TWO_COST,basis_case_source=F.ref(BASE,OLD_INPUT,['cases',0]),
        decision_rule=dict(meets='UPDATE_ACHIEVED_COST_TO_ONE_SHOT',
            exceeds='KEEP_TWO_SHOT_AND_RECORD_TWO_SHOT_CHEAPEST_FOR_OTHER_ROUNDS_FIXED_M12_X_EQ_Y_AT_LEAST_ONE',
            inconclusive_or_unknown='HOLD_ONE_SHOT_AND_KEEP_CONFIRMED_ACHIEVED_COST'),
        duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md')
    if C.cached_record(two,inputs)['status']!=MEETS:raise ValueError('Saved 2-shot qualification differs')
    if C.cached_record(case,inputs) is not None:raise ValueError('Fixed m12=1 vector already saved')
    return inputs


def prepare():
    x=build_inputs();(ROOT/DIRECTORY).mkdir(exist_ok=True);R.write_new(ROOT/INPUT,x)
    c=x['cases'][0];print(c['label'],c['conditional_cost'],c['reduction_RZ'],c['finite_scope_preflight'],flush=True)


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('Caller/INPUTS not fixed')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs():raise ValueError('Fixed one-shot input differs')
    case=inputs['cases'][0];run=ROOT/DIRECTORY/'run01';run.mkdir()
    source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(run/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=1))
    R.write_new(run/'SELECTION.json',dict(source=source,case=case,reason='SEPARATELY_APPROVED_FULL_PF_S2_FIXED_M12_ONE_ONLY'))
    outcome=C.cached_record(case,inputs)
    if outcome is None:
        (run/case['label']).mkdir()
        with F.installed_workers(run):supervised=M.supervise([case],inputs,source,run)
        exits=[c for p in run.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines() for c in json.loads(line)['children']]
        gate=bool(exits) and all(c['exitcode']==0 for c in exits)
        R.write_new(run/'SUPERVISION.json',dict(**supervised,process_exit_gate=dict(recorded_children=len(exits),all_exit_codes_zero=gate)))
        try:
            outcome=C.validate_result(case,inputs,run/case['label'],source)
            if not gate or supervised['errors'] or supervised['monitor_errors']:
                outcome.update(status='UNKNOWN',reason='SUPERVISOR_OR_PROCESS_GATE')
        except Exception as exc:
            stages=run/case['label']/'STAGES.jsonl'
            calls=sum(json.loads(line)['stage']=='SCIENTIFIC_CALL_START' for line in stages.read_text().splitlines()) if stages.exists() else 0
            outcome=dict(status='UNKNOWN',reason=type(exc).__name__+': '+str(exc),scientific_evaluation_calls=calls,
                MSE_lower=None,MSE_upper=None,result_source=source)
    R.write_new(run/'OUTCOME.json',outcome)
    print('STOP:',outcome['status'],'new calls:',outcome['scientific_evaluation_calls'],flush=True)


def decision(status,case,inputs):
    if status==MEETS:
        return dict(one_shot='MODEL_FEASIBLE',achieved_conditional_cost_RZ=case['conditional_cost'],
            achieved_allocation_id=case['allocation']['allocation_id'],achieved_cost_updated=True,cheapest_in_scope=None)
    two=inputs['saved_cache'][-1]['case']
    kept=dict(achieved_conditional_cost_RZ=inputs['previous_achieved_cost_RZ'],
        achieved_allocation_id=two['allocation']['allocation_id'],achieved_cost_updated=False)
    if status==EXCEEDS:
        return dict(kept,one_shot='MODEL_INFEASIBLE',cheapest_in_scope=dict(
            scope='OTHER_ROUNDS_FIXED_AT_[59,55,51,47,43,39,35,31,27,23,19,13]_M12_N_X_EQ_N_Y_AT_LEAST_ONE',
            cheapest_model_feasible_m12=2,conditional_cost_RZ=inputs['previous_achieved_cost_RZ'],
            allocation_id=two['allocation']['allocation_id'],
            basis='m12=1 is the only cheaper point in this scope and exceeds the threshold; m12=2 meets it'))
    return dict(kept,one_shot='HELD_'+status,cheapest_in_scope=None)


def summarize():
    p=ROOT/DIRECTORY;inputs=json.loads((p/'INPUTS.json').read_text());case=inputs['cases'][0]
    outcome=json.loads((p/'run01/OUTCOME.json').read_text());source=json.loads((p/'run01/EXECUTION.json').read_text())['source']
    tau=C.exact(inputs['adopted_evaluation']['MSE_upper']);rows=[]
    for reference in inputs['comparisons']:
        record=T._selected(ROOT,reference['API_source'])[0];risk=record['diagnostic']['deterministic_risk']
        status=F.H.assess(risk,inputs['adopted_evaluation'],record['source'])['status']
        assert status==MEETS
        rows.append(dict(m12=reference['m12'],conditional_cost_RZ=reference['cost_RZ'],allocation=record['allocation'],
            MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],status=status,API_source=reference['API_source'],new_calls=0))
    rows.append(dict(m12=1,conditional_cost_RZ=case['conditional_cost'],allocation=case['allocation'],
        MSE_lower=outcome['MSE_lower'],MSE_upper=outcome['MSE_upper'],status=outcome['status'],
        result_source=outcome['result_source'],new_calls=outcome['scientific_evaluation_calls']))
    for r in rows:
        r['upper_over_tau']=None if r['MSE_upper'] is None else F.exact(C.exact(r['MSE_upper'])/tau)
        r['lower_over_tau']=None if r['MSE_lower'] is None else F.exact(C.exact(r['MSE_lower'])/tau)
    d=decision(outcome['status'],case,inputs)
    summary=dict(source=source,rows=rows,new_scientific_calls=outcome['scientific_evaluation_calls'],outcome=outcome,
        reduction_RZ=case['reduction_RZ'],tau=inputs['adopted_evaluation']['MSE_upper'],**d,
        process_return=json.loads((p/'PROCESS_RETURN.json').read_text()),sufficient_N=None,global_RMSE_bound=None,
        G=None,G_qualified=None,winner=None,mandatory_stop=True)
    R.write_new(p/'EXACT_COMPARISON.json',summary);R.write_new(p/'SUMMARY.json',F.D.display(summary))
    R.write_new(p/'C_RESULT_JOIN.json',dict(source=source,method='FULL_PF',formula='S2',requested_width='16h/7',
        configuration_id=case['configuration_id'],total_view_id=case['total_view_id'],binding_id=case['binding_id'],
        signal_packet=case['signal_packet'],evaluation_target=case['evaluation_target'],allocation=case['allocation'],
        parent_allocation=case['parent_allocation'],price_terms=case['price_terms'],price_policy=inputs['price_policy'],
        conditional_cost_RZ=case['conditional_cost'],reference_cost_RZ=case['reference_cost_RZ'],reduction_RZ=case['reduction_RZ'],
        model_status=outcome['status'],MSE_lower=outcome['MSE_lower'],MSE_upper=outcome['MSE_upper'],
        tau=inputs['adopted_evaluation']['MSE_upper'],result_source=outcome['result_source'],
        reference_results=[dict(m12=r['m12'],conditional_cost_RZ=r['conditional_cost_RZ'],status=r['status'],
            MSE_lower=r['MSE_lower'],MSE_upper=r['MSE_upper'],API_source=r['API_source']) for r in rows if r['m12']!=1],
        **d,sufficient_N=None,G_qualified=None))
    print(json.dumps(F.D.display(dict(decision=d,rows=[dict(m12=r['m12'],cost=r['conditional_cost_RZ'],status=r['status'],
        MSE_lower=r['MSE_lower'],MSE_upper=r['MSE_upper'],upper_over_tau=r['upper_over_tau']) for r in rows])),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute','summarize'))
    globals()[parser.parse_args().mode]()

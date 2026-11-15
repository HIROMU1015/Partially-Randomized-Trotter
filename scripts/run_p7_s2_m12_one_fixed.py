"""One approved m12=1 Q3; reuse the fixed m12=2 execution/persistence path."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_saved_two_shot_caller',ROOT/'scripts/run_p7_s2_m12_two_fixed.py')
Q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=Q;spec.loader.exec_module(Q)
C,F,R,T=Q.C,Q.F,Q.R,Q.T
BASE='93ab9dbbe610d66eba39ea9299b070a13a867a3e'
OLD_INPUT=Q.INPUT
OLD_DIRECTORY=Q.DIRECTORY
DIRECTORY='docs/research/p7_s2_m12_one_results_20261116'
INPUT=DIRECTORY+'/INPUTS.json'
CALLER='scripts/run_p7_s2_m12_one_fixed.py'
VECTOR=[59,55,51,47,43,39,35,31,27,23,19,13,1]
SAVED_CHECK=Q.check_source


def check_source():
    SAVED_CHECK();path='scripts/run_p7_s2_m12_two_fixed.py'
    if (ROOT/path).read_bytes()!=R.blob(BASE,path):raise ValueError('Reused one-case caller SOURCE differs')


def build_inputs():
    check_source();prior=F.document(BASE,OLD_INPUT);case=deepcopy(prior['cases'][0]);two=deepcopy(case)
    case.update(label='P7_S2_M12_ONE_FIXED',family=[12,3],
        edit=dict(m=12,old_N_X=2,old_N_Y=2,new_N_X=1,new_N_Y=1),parent_allocation=deepcopy(two['allocation']))
    case['allocation']=dict(allocation_id='P7_S2_16H_OVER7_M12_ONE_01',shot_policy='USER_SUPPLIED',
        shots=[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(VECTOR)])
    for term in case['price_terms']:
        term['N']=VECTOR[term['m']];term['N_times_C']=term['N']*term['C']
    case['conditional_cost']=sum(term['N_times_C'] for term in case['price_terms'])
    if case['conditional_cost']!=376203472:raise ValueError('Adopted model cost mismatch')
    case['reference_cost_RZ']=395962678;case['reduction_RZ']=395962678-case['conditional_cost']
    case['finite_scope_preflight']=F.count_bounds(case['allocation']['shots'])
    b=case['finite_scope_preflight'];o=prior['options']
    if (b['maximum_round_outcomes']>o['maximum_round_outcomes'] or b['verification_pairs_upper']>o['maximum_verification_pairs']
            or b['transition_pairs_upper']>o['total_state_refinement']['maximum_transition_pairs']):raise ValueError('Count-only scope exceeded')
    F.metadata(case)
    inputs=deepcopy(prior)
    inputs.update(cases=[case],previous_achieved_cost_RZ=395962678,basis_case_source=F.ref(BASE,OLD_INPUT,['cases',0]),
        duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md')
    inputs.pop('fixed_case_source',None)
    ref=F.ref(BASE,OLD_DIRECTORY+'/run01/'+two['label']+'/API_RETURN.json')
    inputs['saved_cache'].append(dict(case=two,API_source=ref))
    inputs['comparisons'].append(dict(m12=2,cost_RZ=395962678,API_source=ref))
    if C.cached_record(two,inputs)['status']!=C.PASS:raise ValueError('Saved 2-shot qualification differs')
    return inputs


def prepare():
    inputs=build_inputs();(ROOT/DIRECTORY).mkdir(exist_ok=True);R.write_new(ROOT/INPUT,inputs)
    c=inputs['cases'][0];print(c['label'],c['conditional_cost'],c['finite_scope_preflight'],flush=True)


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('Caller/INPUTS not fixed')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs():raise ValueError('Fixed one-shot input differs')
    case=inputs['cases'][0];run=ROOT/DIRECTORY/'run01';run.mkdir()
    source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(run/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=1))
    R.write_new(run/'SELECTION.json',dict(source=source,case=case,reason='SEPARATELY_APPROVED_FIXED_M12_ONE_ONLY'))
    outcome=C.cached_record(case,inputs)
    if outcome is None:
        (run/case['label']).mkdir()
        with F.installed_workers(run):supervised=Q.M.supervise([case],inputs,source,run)
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


def summarize():
    p=ROOT/DIRECTORY;inputs=json.loads((p/'INPUTS.json').read_text());case=inputs['cases'][0]
    outcome=json.loads((p/'run01/OUTCOME.json').read_text());source=json.loads((p/'run01/EXECUTION.json').read_text())['source']
    rows=[]
    for reference in inputs['comparisons']:
        record=T._selected(ROOT,reference['API_source'])[0];risk=record['diagnostic']['deterministic_risk']
        status=F.H.assess(risk,inputs['adopted_evaluation'],record['source'])['status']
        assert status==C.PASS
        rows.append(dict(m12=reference['m12'],conditional_cost_RZ=reference['cost_RZ'],allocation=record['allocation'],
            MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],status=status,API_source=reference['API_source'],new_calls=0))
    rows.append(dict(m12=1,conditional_cost_RZ=case['conditional_cost'],allocation=case['allocation'],
        MSE_lower=outcome['MSE_lower'],MSE_upper=outcome['MSE_upper'],status=outcome['status'],
        result_source=outcome['result_source'],new_calls=outcome['scientific_evaluation_calls']))
    achieved=case['conditional_cost'] if outcome['status']==C.PASS else inputs['previous_achieved_cost_RZ']
    summary=dict(source=source,rows=rows,new_scientific_calls=outcome['scientific_evaluation_calls'],outcome=outcome,
        achieved_conditional_cost_RZ=achieved,achieved_cost_updated=outcome['status']==C.PASS,
        process_return=json.loads((p/'PROCESS_RETURN.json').read_text()),sufficient_N=None,global_RMSE_bound=None,
        G=None,G_qualified=None,winner=None,mandatory_stop=True)
    R.write_new(p/'EXACT_COMPARISON.json',summary);R.write_new(p/'SUMMARY.json',F.D.display(summary))
    R.write_new(p/'C_RESULT_JOIN.json',dict(source=source,configuration_id=case['configuration_id'],total_view_id=case['total_view_id'],
        binding_id=case['binding_id'],signal_packet=case['signal_packet'],allocation=case['allocation'],price_terms=case['price_terms'],
        model_status=outcome['status'],conditional_cost_RZ=case['conditional_cost'],achieved_conditional_cost_RZ=achieved,
        result_source=outcome['result_source'],sufficient_N=None,G_qualified=None))
    print(json.dumps(dict(achieved_conditional_cost_RZ=achieved,rows=[dict(m12=r['m12'],cost=r['conditional_cost_RZ'],status=r['status'],
        MSE_lower=F.D.display(r['MSE_lower']),MSE_upper=F.D.display(r['MSE_upper'])) for r in rows]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute','summarize'))
    globals()[parser.parse_args().mode]()

"""One separately approved m12=2 Q3; preserve the previous search's early stop."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_saved_local_search',ROOT/'scripts/run_p7_s2_local_shot_search.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
F,R,T,M=C.F,C.R,C.T,C.M
FIXED='e7d24c2192d4640137bdaf993a4e954a49f7932c'
THREE='0b33b4afdd285a99f610e59d8f5b77960a59a0d0'
DIRECTORY='docs/research/p7_s2_m12_two_results_20261116'
INPUT=DIRECTORY+'/INPUTS.json'
CALLER='scripts/run_p7_s2_m12_two_fixed.py'
VECTOR=[59,55,51,47,43,39,35,31,27,23,19,13,2]


def check_source():
    C.check_source();path='scripts/run_p7_s2_local_shot_search.py'
    if (ROOT/path).read_bytes()!=R.blob(FIXED,path):raise ValueError('Original local caller differs')


def build_inputs():
    check_source();prior=F.document(FIXED,C.INPUT)
    case=deepcopy(prior['single_cases'][1]);three=deepcopy(prior['single_cases'][0])
    if ([r['N_X'] for r in case['allocation']['shots']]!=VECTOR or
            [r['N_Y'] for r in case['allocation']['shots']]!=VECTOR or case['edit']!=dict(r=12,d=2,k=None,s=0)):
        raise ValueError('Only the fixed m12=2 candidate is permitted')
    F.metadata(case)
    bounds=F.count_bounds(case['allocation']['shots']);options=prior['options']
    if (bounds!=case['finite_scope_preflight'] or bounds['maximum_round_outcomes']>options['maximum_round_outcomes']
            or bounds['verification_pairs_upper']>options['maximum_verification_pairs']
            or bounds['transition_pairs_upper']>options['total_state_refinement']['maximum_transition_pairs']):
        raise ValueError('Fixed count-only scope differs')
    cache=deepcopy(prior['saved_cache'])
    cache.append(dict(case=three,API_source=F.ref(THREE,C.DIRECTORY+'/run01/step01/'+three['label']+'/API_RETURN.json')))
    inputs={k:deepcopy(prior[k]) for k in ('kernel_source','C_source','acceleration','statistical_contract',
        'options','adopted_evaluation','price_policy')}
    inputs.update(cases=[case],saved_cache=cache,concurrent_configurations=1,maximum_scientific_calls=1,
        fixed_case_source=F.ref(FIXED,C.INPUT,['single_cases',1]),previous_achieved_cost_RZ=415721884,
        comparisons=[dict(m12=4,cost_RZ=435481090,API_source=prior['saved_cache'][1]['API_source']),
            dict(m12=3,cost_RZ=415721884,API_source=cache[-1]['API_source'])],
        previous_search_preserved=True,duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md',
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)
    if C.cached_record(three,inputs)['status']!=C.PASS:raise ValueError('Saved m12=3 reference differs')
    return inputs


def prepare():
    x=build_inputs();(ROOT/DIRECTORY).mkdir(exist_ok=True);R.write_new(ROOT/INPUT,x)
    c=x['cases'][0];print(c['label'],c['conditional_cost'],c['finite_scope_preflight'],flush=True)


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('Caller/INPUTS not fixed')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs():raise ValueError('Fixed one-case input differs')
    case=inputs['cases'][0];run=ROOT/DIRECTORY/'run01';run.mkdir()
    source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(run/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=1))
    R.write_new(run/'SELECTION.json',dict(source=source,case=case,reason='SEPARATELY_APPROVED_FIXED_M12_TWO_ONLY'))
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
    rows.append(dict(m12=2,conditional_cost_RZ=case['conditional_cost'],allocation=case['allocation'],
        MSE_lower=outcome['MSE_lower'],MSE_upper=outcome['MSE_upper'],status=outcome['status'],
        result_source=outcome['result_source'],new_calls=outcome['scientific_evaluation_calls']))
    achieved=case['conditional_cost'] if outcome['status']==C.PASS else inputs['previous_achieved_cost_RZ']
    summary=dict(source=source,rows=rows,new_scientific_calls=outcome['scientific_evaluation_calls'],outcome=outcome,
        achieved_conditional_cost_RZ=achieved,achieved_cost_updated=outcome['status']==C.PASS,
        process_return=json.loads((p/'PROCESS_RETURN.json').read_text()),sufficient_N=None,global_RMSE_bound=None,
        G=None,G_qualified=None,winner=None,mandatory_stop=True)
    R.write_new(p/'EXACT_COMPARISON.json',summary);R.write_new(p/'SUMMARY.json',F.D.display(summary))
    R.write_new(p/'C_RESULT_JOIN.json',dict(source=source,configuration_id=case['configuration_id'],
        total_view_id=case['total_view_id'],binding_id=case['binding_id'],signal_packet=case['signal_packet'],
        allocation=case['allocation'],price_terms=case['price_terms'],model_status=outcome['status'],
        conditional_cost_RZ=case['conditional_cost'],achieved_conditional_cost_RZ=achieved,
        result_source=outcome['result_source'],sufficient_N=None,G_qualified=None))
    print(json.dumps(F.D.display(dict(rows=rows,achieved_conditional_cost_RZ=achieved)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute','summarize'))
    globals()[parser.parse_args().mode]()

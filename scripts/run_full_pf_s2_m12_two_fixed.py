"""One approved Full PF/S2 (16h/7) m12=2 Q3 from the saved 4-shot B_PRICE point.

Reuses B's saved PF worker, supervisor, cache acceptance and result checks
(byte-identical SOURCE); only the fixed case, its prices and the output
namespace are new. No search, signal, compile, price fit or other allocation.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_saved_local_search_full_pf',ROOT/'scripts/run_p7_s2_local_shot_search.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
F,R,T,M=C.F,C.R,C.T,C.M
BASE=C.BASE  # b4296b110: saved PF6 cost-budget inputs and the 4-shot API
OLD=C.OLD
LOCAL_FIXED='e7d24c2192d4640137bdaf993a4e954a49f7932c'  # reused local caller SOURCE
P7_TWO='93ab9dbbe610d66eba39ea9299b070a13a867a3e'  # p7 m12=2 result (context only)
P7_ONE='cbc471c913072de45c25a0ed22d51dff8a9b7034'  # p7 m12=1 result (context only)
LABEL='FULL_PF_S2_16h_over_7'
CONFIGURATION='11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d'
SEED_ALLOCATION='FULL_PF_S2_16h_over_7_B_PRICE_COST_BUDGET_3O4_01'
SEED_VECTOR=[59,55,51,47,43,39,35,31,27,23,19,13,4]
SEED_COST=615304502
VECTOR=[59,55,51,47,43,39,35,31,27,23,19,13,2]
NEW_LABEL='FULL_PF_S2_M12_TWO_FIXED'
NEW_ALLOCATION='FULL_PF_S2_16H_OVER7_M12_TWO_01'
DIRECTORY='docs/research/full_pf_s2_m12_two_results_20261116'
INPUT=DIRECTORY+'/INPUTS.json'
CALLER='scripts/run_full_pf_s2_m12_two_fixed.py'


def check_source():
    C.check_source();path='scripts/run_p7_s2_local_shot_search.py'
    if (ROOT/path).read_bytes()!=R.blob(LOCAL_FIXED,path):raise ValueError('Reused local caller SOURCE differs')


def seed_case(prior):
    index,seed=next((i,c) for i,c in enumerate(prior['cases']) if c['label']==LABEL+'__B_PRICE')
    if (seed['configuration_id']!=CONFIGURATION or seed['allocation']['allocation_id']!=SEED_ALLOCATION
            or [v['N_X'] for v in seed['allocation']['shots']]!=SEED_VECTOR
            or [v['N_Y'] for v in seed['allocation']['shots']]!=SEED_VECTOR or seed['conditional_cost']!=SEED_COST):
        raise ValueError('Fixed Full PF 4-shot seed differs')
    return index,seed


def price_rows(seed):
    """Full PF's own 26 saved axis prices, joined to physical/canonical IDs as in B's local caller."""
    binding,view=F.metadata(seed)
    allrows=F.document(F.B2,F.B2_DIR+'AXIS_TERMS.json')['rows']
    rows=[dict(p,price_source=F.ref(F.B2,F.B2_DIR+'AXIS_TERMS.json',['rows',i])) for i,p in enumerate(allrows) if p['label']==LABEL]
    rows.sort(key=lambda p:(p['m'],p['axis']))
    if [(p['m'],p['axis']) for p in rows]!=[(m,a) for m in range(13) for a in ('X','Y')]:raise ValueError('26 prices required')
    terms={(t['m'],t['axis']):t for t in seed['price_terms']}
    for p in rows:
        m=p['m'];physical=view['signal_packet']['means'][m]['provider_binding']['physical_context']['physical_candidate']
        if (p['original_configuration_id']!=seed['configuration_id'] or p['total_configuration_id']!=seed['total_view_id']
                or p['physical_id']!=physical['id'] or p['original_candidate_id']!=binding['original_configuration']['rounds'][m]['candidate']['id']
                or p['total_candidate_id']!=binding['total_configuration']['rounds'][m]['candidate']['id']):
            raise ValueError('Price/physical/canonical join differs')
        t=terms[(m,p['axis'])]
        if F.integer(p['C'])!=t['C'] or p['price_class']!=t['price_class'] or p['price_source']!=t['price_source']:
            raise ValueError('Seed price row differs')
    classes=[p['price_class'] for p in rows]
    if (classes.count('DIRECT_COMPILED_DETERMINISTIC')!=16
            or classes.count('CONDITIONAL_FIXED_COMPILER_CLOSURE_DERIVATION')!=10):
        raise ValueError('16 direct / 10 conditional closure required')
    if sum(SEED_VECTOR[p['m']]*p['C'] for p in rows)!=SEED_COST:raise ValueError('Seed cost mismatch')
    return rows


def make_case(seed,rows):
    case={name:deepcopy(seed[name]) for name in ('configuration_id','total_view_id','binding_id',
        'configuration','signal_packet','evaluation_target')}
    shots=[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(VECTOR)]
    case.update(label=NEW_LABEL,edit=dict(m=12,old_N_X=4,old_N_Y=4,new_N_X=2,new_N_Y=2),
        allocation=dict(allocation_id=NEW_ALLOCATION,shot_policy='USER_SUPPLIED',shots=shots),
        parent_allocation=deepcopy(seed['allocation']))
    case['price_terms']=[dict(m=p['m'],axis=p['axis'],N=VECTOR[p['m']],C=p['C'],N_times_C=VECTOR[p['m']]*p['C'],
        physical_id=p['physical_id'],original_candidate_id=p['original_candidate_id'],total_candidate_id=p['total_candidate_id'],
        price_class=p['price_class'],price_source=p['price_source']) for p in rows]
    case['conditional_cost']=sum(t['N_times_C'] for t in case['price_terms'])
    case['conditional_cost_m0_m11']=sum(t['N_times_C'] for t in case['price_terms'] if t['m']<12)
    case['conditional_cost_m12']=sum(t['N_times_C'] for t in case['price_terms'] if t['m']==12)
    case['reference_cost_RZ']=SEED_COST;case['reduction_RZ']=SEED_COST-case['conditional_cost']
    case['finite_scope_preflight']=F.count_bounds(shots)
    return case


def build_inputs():
    check_source();prior=F.document(BASE,OLD+'INPUTS.json')
    index,seed=seed_case(prior);rows=price_rows(seed);case=make_case(seed,rows)
    if case['allocation']['shots'][:12]!=seed['allocation']['shots'][:12]:raise ValueError('Only m12 may change')
    if case['reduction_RZ']!=sum(2*p['C'] for p in rows if p['m']==12):
        raise ValueError('Reduction must be two shots on each m12 axis')
    F.metadata(case)
    b=case['finite_scope_preflight'];o=prior['options']
    if (b['maximum_round_outcomes']>o['maximum_round_outcomes'] or b['verification_pairs_upper']>o['maximum_verification_pairs']
            or b['transition_pairs_upper']>o['total_state_refinement']['maximum_transition_pairs']):
        raise ValueError('Count-only scope exceeded')
    cache=[]
    for rule in ('A_SHAPE','B_PRICE'):
        old=next(c for c in prior['cases'] if c['label']==LABEL+'__'+rule)
        cache.append(dict(case=old,API_source=F.ref(BASE,OLD+'run01/'+old['label']+'/API_RETURN.json')))
    inputs={k:deepcopy(prior[k]) for k in ('kernel_source','C_source','acceleration','statistical_contract',
        'options','adopted_evaluation','price_policy')}
    inputs.update(cases=[case],saved_cache=cache,concurrent_configurations=1,maximum_scientific_calls=1,
        seed_case_source=F.ref(BASE,OLD+'INPUTS.json',['cases',index]),
        price_rows_source=F.ref(F.B2,F.B2_DIR+'AXIS_TERMS.json'),previous_achieved_cost_RZ=SEED_COST,
        comparisons=[dict(m12=4,cost_RZ=SEED_COST,API_source=cache[1]['API_source'])],
        p7_context=dict(m12_two_result=F.ref(P7_TWO,'docs/research/p7_s2_m12_two_results_20261116/EXACT_COMPARISON.json'),
            m12_one_result=F.ref(P7_ONE,'docs/research/p7_s2_m12_one_results_20261116/EXACT_COMPARISON.json'),
            use='CONTEXT_ONLY_DIFFERENT_PHYSICAL_CANDIDATE_NOT_REUSED'),
        duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md',signal_recomputed=False,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)
    if C.cached_record(seed,inputs)['status']!=C.PASS:raise ValueError('Saved 4-shot qualification differs')
    if C.cached_record(case,inputs) is not None:raise ValueError('Fixed m12=2 vector already saved')
    return inputs


def prepare():
    x=build_inputs();(ROOT/DIRECTORY).mkdir(exist_ok=True);R.write_new(ROOT/INPUT,x)
    c=x['cases'][0];print(c['label'],c['conditional_cost'],c['reduction_RZ'],c['finite_scope_preflight'],flush=True)


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('Caller/INPUTS not fixed')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs():raise ValueError('Fixed one-case input differs')
    case=inputs['cases'][0];run=ROOT/DIRECTORY/'run01';run.mkdir()
    source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(run/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=1))
    R.write_new(run/'SELECTION.json',dict(source=source,case=case,reason='SEPARATELY_APPROVED_FULL_PF_S2_FIXED_M12_TWO_ONLY'))
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
    tau=C.exact(inputs['adopted_evaluation']['MSE_upper'])
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
    for r in rows:
        r['upper_over_tau']=None if r['MSE_upper'] is None else F.exact(C.exact(r['MSE_upper'])/tau)
    passed=outcome['status']==C.PASS
    achieved=case['conditional_cost'] if passed else inputs['previous_achieved_cost_RZ']
    summary=dict(source=source,rows=rows,new_scientific_calls=outcome['scientific_evaluation_calls'],outcome=outcome,
        reduction_RZ=case['reduction_RZ'],tau=inputs['adopted_evaluation']['MSE_upper'],
        achieved_conditional_cost_RZ=achieved,achieved_cost_updated=passed,
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
        reference_result_source=inputs['comparisons'][0]['API_source'],
        achieved_conditional_cost_RZ=achieved,achieved_cost_updated=passed,
        achieved_allocation_id=case['allocation']['allocation_id'] if passed else SEED_ALLOCATION,
        sufficient_N=None,G_qualified=None))
    print(json.dumps(F.D.display(dict(achieved_conditional_cost_RZ=achieved,rows=[dict(m12=r['m12'],cost=r['conditional_cost_RZ'],
        status=r['status'],MSE_lower=r['MSE_lower'],MSE_upper=r['MSE_upper'],upper_over_tau=r['upper_over_tau']) for r in rows])),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute','summarize'))
    globals()[parser.parse_args().mode]()

"""Adopted finite, sequential Q3 search; existing PF worker owns all numerics."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_local_saved_pf',ROOT/'scripts/run_pf6_cost_budget_q3_fixed.py')
F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)
R,T,M=F.R,F.T,F.M
exact=T.B.a.exact_scalar
PLAN='9992c594d4c72d1e8b65b42029d2b231641c0d17'
BASE='b4296b110b129d226dbef2743c14f725a27c3bf6'
OLD='docs/research/pf6_cost_budget_q3_results_20261115/'
DIRECTORY='docs/research/p7_s2_local_shot_search_results_20261116'
INPUT=DIRECTORY+'/INPUTS.json'
CALLER='scripts/run_p7_s2_local_shot_search.py'
LABEL='B0I_P7_S2_16h_over_7'
PASS=F.PASS


def key(case):
    return tuple((r['m'],r['N_X'],r['N_Y']) for r in case['allocation']['shots'])


def check_source():
    F.check_source()
    path='scripts/run_pf6_cost_budget_q3_fixed.py'
    if (ROOT/path).read_bytes()!=R.blob(BASE,path):raise ValueError('Reused PF worker SOURCE differs')


def make_case(seed,rows,caps,r,d,k=None,s=0):
    case={name:deepcopy(seed[name]) for name in ('configuration_id','total_view_id','binding_id',
        'configuration','signal_packet','evaluation_target')}
    vector=deepcopy(seed['allocation']['shots'])
    for axis in ('X','Y'):
        vector[r]['N_'+axis]-=d
        if k is not None:vector[k]['N_'+axis]+=s
    if any(type(v['N_'+a]) is not int or not 1<=v['N_'+a]<=caps[v['m']] for v in vector for a in ('X','Y')):
        return None
    suffix=f'R{r}_D{d}'+('' if k is None else f'_K{k}_S{s}')
    case.update(label='P7_S2_LOCAL_'+suffix,family=[r,d],edit=dict(r=r,d=d,k=k,s=s),
        phase='SINGLE' if k is None else 'TRANSFER',
        allocation=dict(allocation_id='P7_S2_16H_OVER7_LOCAL_'+suffix+'_01',shot_policy='USER_SUPPLIED',shots=vector),
        parent_allocation=deepcopy(seed['allocation']))
    case['price_terms']=[dict(m=p['m'],axis=p['axis'],N=vector[p['m']]['N_'+p['axis']],C=p['C'],
        N_times_C=vector[p['m']]['N_'+p['axis']]*p['C'],physical_id=p['physical_id'],
        original_candidate_id=p['original_candidate_id'],total_candidate_id=p['total_candidate_id'],
        price_class=p['price_class'],price_source=p['price_source']) for p in rows]
    case['conditional_cost']=sum(t['N_times_C'] for t in case['price_terms'])
    if case['conditional_cost']>=seed['conditional_cost']:return None
    case['reduction_RZ']=seed['conditional_cost']-case['conditional_cost']
    case['finite_scope_preflight']=F.count_bounds(vector)
    return case


def build_inputs():
    check_source();prior=F.document(BASE,OLD+'INPUTS.json')
    seed=next(c for c in prior['cases'] if c['label']==LABEL+'__B_PRICE')
    if [v['N_X'] for v in seed['allocation']['shots']]!=[59,55,51,47,43,39,35,31,27,23,19,13,4] or seed['conditional_cost']!=435481090:
        raise ValueError('Adopted seed differs')
    if any(v['N_X']!=v['N_Y'] for v in seed['allocation']['shots']):raise ValueError('Fixed seed axes differ')
    binding,view=F.metadata(seed)
    allrows=F.document(F.B2,F.B2_DIR+'AXIS_TERMS.json')['rows']
    rows=[dict(p,price_source=F.ref(F.B2,F.B2_DIR+'AXIS_TERMS.json',['rows',i])) for i,p in enumerate(allrows) if p['label']==LABEL]
    rows.sort(key=lambda p:(p['m'],p['axis']))
    if [(p['m'],p['axis']) for p in rows]!=[(m,a) for m in range(13) for a in ('X','Y')]:raise ValueError('26 prices required')
    for p in rows:
        m=p['m'];physical=view['signal_packet']['means'][m]['provider_binding']['physical_context']['physical_candidate']
        if (p['original_configuration_id']!=seed['configuration_id'] or p['total_configuration_id']!=seed['total_view_id']
                or p['physical_id']!=physical['id'] or p['original_candidate_id']!=binding['original_configuration']['rounds'][m]['candidate']['id']
                or p['total_candidate_id']!=binding['total_configuration']['rounds'][m]['candidate']['id']):
            raise ValueError('Price/physical/canonical join differs')
        F.integer(p['C'])
    if [p['price_class'] for p in rows].count('DIRECT_COMPILED_DETERMINISTIC')!=16 or [p['price_class'] for p in rows].count('CONDITIONAL_FIXED_COMPILER_CLOSURE_DERIVATION')!=10:
        raise ValueError('16 direct / 10 conditional closure required')
    if sum(v['N_'+p['axis']]*p['C'] for p in rows for v in [seed['allocation']['shots'][p['m']]])!=435481090:
        raise ValueError('Seed cost mismatch')
    single=[make_case(seed,rows,F.N0,r,d) for r,d in ((12,1),(12,2),(11,1),(11,2))]
    transfers=[c for r in (12,11) for d in (1,2) for k in range(13) if k!=r for s in (1,2)
        if (c:=make_case(seed,rows,F.N0,r,d,k,s)) is not None]
    if any(c is None for c in single):raise ValueError('Fixed single edit absent')
    options=deepcopy(prior['options'])
    for c in single+transfers:
        b=c['finite_scope_preflight']
        if (b['maximum_round_outcomes']>options['maximum_round_outcomes'] or b['verification_pairs_upper']>options['maximum_verification_pairs']
                or b['transition_pairs_upper']>options['total_state_refinement']['maximum_transition_pairs']):
            raise ValueError('Concrete count scope exceeded')
    cache=[]
    for rule in ('A_SHAPE','B_PRICE'):
        oldcase=next(c for c in prior['cases'] if c['label']==LABEL+'__'+rule)
        cache.append(dict(case=oldcase,API_source=F.ref(BASE,OLD+'run01/'+oldcase['label']+'/API_RETURN.json')))
    return dict(fixed_plan=F.ref(PLAN,'docs/research/p7_s2_local_shot_search_plan_20261116/REPORT.md'),
        seed=seed,caps_N0=F.N0,single_cases=single,transfer_cases=transfers,saved_cache=cache,
        maximum_scientific_calls=8,maximum_transfer_calls=4,concurrent_configurations=1,
        kernel_source=prior['kernel_source'],C_source=prior['C_source'],acceleration=prior['acceleration'],
        statistical_contract=prior['statistical_contract'],options=options,adopted_evaluation=prior['adopted_evaluation'],
        price_policy=prior['price_policy'],duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md',
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)


def search(inputs,visit):
    """Pure finite selection. visit is synchronous; no speculative evaluations."""
    history=[];seen=set();new_calls=0;transfer_calls=0;stop=None
    queue=iter(inputs['single_cases'])
    while stop is None:
        candidate=next(queue,None)
        priority=None
        if candidate is None:
            if transfer_calls>=inputs['maximum_transfer_calls']:stop='TRANSFER_CALL_LIMIT';break
            pool=[]
            for c in inputs['transfer_cases']:
                if key(c) in seen:continue
                family=[r for r in history if r['family']==c['family'] and r.get('MSE_upper') is not None]
                if not family:continue
                score=min(exact(r['MSE_upper']) for r in family)
                e=c['edit'];rank=(score,c['conditional_cost'],e['d']+e['s'],e['r'],e['d'],e['k'],e['s'])
                pool.append((rank,c))
            if not pool:stop='FINITE_CANDIDATES_EXHAUSTED';break
            rank,candidate=min(pool,key=lambda x:x[0]);priority=F.exact(rank[0])
        if key(candidate) in seen:continue
        if new_calls>=inputs['maximum_scientific_calls']:stop='TOTAL_CALL_LIMIT';break
        parents=[dict(label=h['label'],result_source=h.get('result_source')) for h in history if h['family']==candidate['family']]
        outcome=visit(candidate,dict(sequence=len(history)+1,family_priority=priority,parent_results=parents,
            reason='FIXED_SINGLE_ORDER' if candidate['phase']=='SINGLE' else 'FAMILY_MIN_VALID_UPPER_THEN_COST_MOVE_LEXICOGRAPHIC',
            previous_new_calls=new_calls,remaining_new_calls=inputs['maximum_scientific_calls']-new_calls))
        calls=outcome['scientific_evaluation_calls']
        if calls not in (0,1):raise ValueError('One call per visit required')
        new_calls+=calls
        if candidate['phase']=='TRANSFER':transfer_calls+=calls
        row=dict(label=candidate['label'],family=candidate['family'],phase=candidate['phase'],edit=candidate['edit'],
            allocation=candidate['allocation'],conditional_cost=candidate['conditional_cost'],reduction_RZ=candidate['reduction_RZ'],**outcome)
        history.append(row);seen.add(key(candidate))
        if row['status']=='UNKNOWN':stop='UNKNOWN_GATE_OR_PERSISTENCE_STOP'
        elif row['status']==PASS and candidate['conditional_cost']<inputs['seed']['conditional_cost']:
            stop='FIRST_STRICTLY_CHEAPER_MODEL_FEASIBLE_POINT'
    return dict(stop_reason=stop,improvement_found=stop=='FIRST_STRICTLY_CHEAPER_MODEL_FEASIBLE_POINT',
        new_scientific_calls=new_calls,transfer_scientific_calls=transfer_calls,history=history,
        unevaluated_single=[c['label'] for c in inputs['single_cases'] if key(c) not in seen],
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)


def cached_record(case,inputs):
    for item in inputs['saved_cache']:
        if key(item['case'])!=key(case):continue
        record=T._selected(ROOT,item['API_source'])[0]
        old=item['case']
        if (old['configuration_id']!=case['configuration_id'] or record['configuration_id']!=case['total_view_id'] or record['canonical_binding']!=case['binding_id']
                or old['configuration']!=case['configuration'] or old['signal_packet']!=case['signal_packet']
                or record['evaluation_target']!=case['evaluation_target']
                or record['allocation']['shots']!=case['allocation']['shots']):raise ValueError('Cached SOURCE/contract differs')
        risk=record['diagnostic']['deterministic_risk']
        def without_injected_source(value):
            if isinstance(value,dict):
                return {k:without_injected_source(v) for k,v in value.items() if not (k=='source' and v==record['source'])}
            if isinstance(value,list):return [without_injected_source(v) for v in value]
            return value
        if (without_injected_source(record['statistical_contract'])!=inputs['statistical_contract']
                or without_injected_source(record['deterministic_input']['options'])!=inputs['options']
                or record['deterministic_input']['origin_packet_source']!=case['signal_packet']):
            raise ValueError('Cached loss/numerical/probability contract differs')
        if record['deterministic_input']['options']['probability_rule']!=inputs['options']['probability_rule'] or risk['estimator']!=inputs['adopted_evaluation']['estimator']:
            raise ValueError('Cached law/decoder differs')
        assessment=F.H.assess(risk,inputs['adopted_evaluation'],record['source'])
        return dict(status=assessment['status'],scientific_evaluation_calls=0,MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],
            result_source=item['API_source'],input_disposition='REUSED_SAVED_API')
    return None


def validate_result(case,inputs,folder,source):
    result=json.loads((folder/'RESULT.json').read_text());usage=json.loads((folder/'RESOURCE_FINAL.json').read_text())
    receipt=json.loads((folder/'PERSISTED.json').read_text())
    assert result['source']==usage['source']==source
    assert result['allocation']==case['allocation']
    assert all(result[k] is None for k in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'))
    out=dict(status=result['status'],scientific_evaluation_calls=result['scientific_evaluation_calls'],
        MSE_lower=None,MSE_upper=None,result_source=dict(**source,result_path=str((folder/'RESULT.json').relative_to(ROOT))),
        resources=usage,operational_status=receipt['operational_status'],input_disposition='NEW_EVALUATION')
    if result['status']=='UNKNOWN':return dict(out,reason=result.get('reason'))
    assert json.loads((folder/'API_RETURN.json').read_text())==[result['record']]
    risk=result['record']['diagnostic']['deterministic_risk'];dp=risk['bounds']['total_state_dp']
    coarse=json.loads((folder/'PARTIAL_COARSE.json').read_text())
    assert F.H.assess(risk,inputs['adopted_evaluation'],source)['status']==result['status']
    assert coarse['decoder_certificate']['status']=='VERIFIED_FINITE_BINARY64_SUPPORT'
    assert coarse['decoder_certificate']['checked_pairs']<=inputs['options']['maximum_verification_pairs']
    assert sum(exact(v['exact']) for v in coarse['first_unsafe'])+exact(coarse['all_good']['exact'])==1
    assert all(exact(v['good']['exact'])+exact(v['unsafe']['exact'])==1 for v in coarse['rounds'])
    assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
    assert 0<=exact(risk['MSE_lower'])<=exact(risk['MSE_upper'])
    assert exact(risk['MSE_upper'])==min(exact(v) for v in risk['bounds']['upper_candidates'].values())
    assert 0<=exact(dp['residual_probability'])<=1
    assert dp['transition_pairs']<=inputs['options']['total_state_refinement']['maximum_transition_pairs']
    assert all(v['retained_states']<=8192 for v in dp['rounds'][:-1])
    moments=risk['moments']
    assert exact(moments['bias_squared'][0])+exact(moments['variance'][0])<=exact(risk['MSE_upper'])
    assert exact(moments['bias_squared'][1])+exact(moments['variance'][1])>=exact(risk['MSE_lower'])
    return dict(out,MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],moments=moments,
        residual_mass=dp['residual_probability'],residual_loss_upper=dp['residual_loss_contribution_upper'],
        any_zero=dp['any_zero'],per_round=dp['rounds'],closure=coarse['decoder_certificate'],
        transition_pairs=dp['transition_pairs'],numerical_allowance=dp['transition_allowance'])


def prepare():
    inputs=build_inputs();(ROOT/DIRECTORY).mkdir(exist_ok=True);R.write_new(ROOT/INPUT,inputs)
    print('Fixed initial queue and conditional transfer pool:',len(inputs['single_cases']),len(inputs['transfer_cases']))
    for c in inputs['single_cases']:print(c['label'],c['conditional_cost'],c['reduction_RZ'],c['finite_scope_preflight'])


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('SOURCE/INPUTS not fixed')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs():raise ValueError('Adopted inputs differ')
    assert cached_record(inputs['seed'],inputs)['status']==PASS
    run=ROOT/DIRECTORY/'run01';run.mkdir()
    source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(run/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=8))
    def visit(case,selection):
        step=run/f"step{selection['sequence']:02d}";step.mkdir()
        R.write_new(step/'SELECTION.json',dict(selection=selection,case=case,source=source,
            seed_source=F.ref(BASE,OLD+'run01/'+inputs['seed']['label']+'/API_RETURN.json')))
        hit=cached_record(case,inputs)
        if hit is not None:
            R.write_new(step/'REUSE.json',hit);return hit
        (step/case['label']).mkdir()
        with F.installed_workers(step):supervised=M.supervise([case],inputs,source,step)
        exits=[c for p in step.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines() for c in json.loads(line)['children']]
        gate=bool(exits) and all(c['exitcode']==0 for c in exits)
        R.write_new(step/'SUPERVISION.json',dict(**supervised,process_exit_gate=dict(recorded_children=len(exits),all_exit_codes_zero=gate)))
        try:
            row=validate_result(case,inputs,step/case['label'],source)
            if not gate or supervised['errors'] or supervised['monitor_errors']:row.update(status='UNKNOWN',reason='SUPERVISOR_OR_PROCESS_GATE')
        except Exception as exc:
            stages=step/case['label']/'STAGES.jsonl'
            calls=sum(json.loads(line)['stage']=='SCIENTIFIC_CALL_START' for line in stages.read_text().splitlines()) if stages.exists() else 0
            row=dict(status='UNKNOWN',reason=type(exc).__name__+': '+str(exc),scientific_evaluation_calls=
                calls,MSE_lower=None,MSE_upper=None,
                result_source=source,step_path=str(step.relative_to(ROOT)))
        R.write_new(step/'OUTCOME.json',row)
        print(case['label'],row['status'],flush=True)
        return row
    result=search(inputs,visit)
    R.write_new(run/'SEARCH_EXACT.json',dict(result,source=source,finished_UTC=R.utc()))
    R.write_new(ROOT/DIRECTORY/'SUMMARY.json',F.D.display(dict(result,source=source)))
    print('STOP:',result['stop_reason'],'new calls:',result['new_scientific_calls'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()

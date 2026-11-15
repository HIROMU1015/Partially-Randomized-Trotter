"""PR-DF LD7 own-price allocations under the p7/Full PF achieved-cost budgets.

At most four unique vectors (two budgets x A_SHAPE/B_PRICE, caps V0), fixed in
exact-cost order before any Q3; sequential one-call Q3 on D2's unchanged LD7
route (same worker/supervisor/runtime as the saved LD7 V0/A/B calls), stop at
the first feasible point or a gate/persistence failure. No retry, signal,
sampling, compile or price refit.
"""
import argparse
import ast
from copy import deepcopy
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import run_pr_df_ld_split_q3_resume as W

ROOT=W.ROOT;U=W.U
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
D2_HEAD='460d77d8f56ef0bde41dbddaa27852dd8b05513e'
VBASE='0defa286c53a1b9acf513be453907bcad83102b3'
VDIR='docs/research/d_pr_df_ld7_v0_results_20261116'
PDIR='docs/research/d_pr_df_ld7_price_results_20261116'
PRICE=dict(commit='4326422a4633ed46968d6e348fe5ffaa24fbb468',path=PDIR+'/AXIS_PRICES.json')
INDEX=dict(commit='fd010b111e95411702be51adc73e56b0bba6c093',path=PDIR+'/C_HANDOFF.json')
SAVED_AB_INPUTS=dict(commit='37b20a07d8ee3535c925d7abdcd15dea914b9418',path=PDIR+'/Q3_INPUTS.json')
GENERATOR=dict(commit='e094b2d60c5c4468711dea244fa2f91fb587ac12',path='scripts/run_pauli_ld2_reallocation_q3_fixed.py',function='generate')
VALIDATOR=dict(commit='cbc471c913072de45c25a0ed22d51dff8a9b7034',path='scripts/run_p7_s2_local_shot_search.py',function='validate_result')
V0=[64,60,56,52,47,43,38,34,30,25,21,17,12]
BUDGETS=[('P7_S2',395962678),('FULL_PF_S2',559457386)]
RULES=('A_SHAPE','B_PRICE')
DIRECTORY='docs/research/pr_df_ld7_pf_budget_q3_results_20261116'
INPUT=DIRECTORY+'/INPUTS.json'
CALLER='scripts/run_pr_df_ld7_pf_budget_q3_fixed.py'
PASS='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
FAIL='MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'
read=W.read


def function(source,scope):
    tree=ast.parse(W.blob(source['commit'],source['path']))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==source['function']]
    if len(nodes)!=1:raise ValueError('Pinned helper unavailable')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),source['commit']+':'+source['path'],'exec'),scope)
    return scope[source['function']]


def rational(v):
    if isinstance(v,dict) and set(v)=={'numerator','denominator'}:return Fraction(v['numerator'],v['denominator'])
    if type(v) is int or isinstance(v,str):return Fraction(v)
    if isinstance(v,Fraction):return v
    raise ValueError('Exact value required')


def enc(v):
    v=Fraction(v);return dict(numerator=v.numerator,denominator=v.denominator)


generate=function(GENERATOR,dict(Fraction=Fraction))
validate_result=function(VALIDATOR,dict(ROOT=ROOT,json=json,exact=U.T.Q.a.exact_scalar,F=NS(H=U.H)))


def key(shots):
    return tuple((r['m'],r['N_X'],r['N_Y']) for r in shots)


def check_source():
    path='scripts/run_pr_df_ld_split_q3_resume.py'
    if (ROOT/path).read_bytes()!=W.blob(D2_HEAD,path):raise ValueError('Reused LD7 route SOURCE differs')


def prices(config):
    table=read(PRICE['commit'],PRICE['path']);rows=table['rows']
    if [(r['m'],r['axis']) for r in rows]!=[(m,a) for m in range(13) for a in ('X','Y')]:raise ValueError('26 ordered axes required')
    for i,r in enumerate(rows):
        c=config['rounds'][r['m']]['physical_candidate']
        if (r['L_D']!=7 or r['configuration_id']!=config['id'] or r['physical_id']!=c['id'] or r['q']!=c['q']
                or r['R']!=1 or c['R_outer']!=1 or r['law']!=config['rounds'][r['m']]['law_binding'] or rational(r['C_ref'])<=0):
            raise ValueError('Price/configuration/physical join differs at row '+str(i))
    return rows


def saved_catalog(rows,caps):
    """Saved LD7 vectors (V0/A/B) from D2's C index; costs re-derived with the same prices."""
    index=read(INDEX['commit'],INDEX['path']);p=[rational(r['C_ref']) for r in rows];out=[]
    if index['axis_prices']!=PRICE:raise ValueError('Index price SOURCE differs')
    for r in index['rows']:
        vector=[s['N_'+a] for s in r['shots'] for a in ('X','Y')]
        if sum(n*c for n,c in zip(vector,p))!=rational(r['cost']):raise ValueError('Saved cost/price mismatch: '+r['label'])
        if any(not 1<=n<=cap for n,cap in zip(vector,caps)):raise ValueError('Saved vector outside V0 caps')
        inputs=dict(commit=VBASE,path=VDIR+'/Q3_INPUTS.json') if r['rule']=='V0' else SAVED_AB_INPUTS
        out.append(dict(label=r['label'],rule=r['rule'],shots=r['shots'],cost=r['cost'],status=r['status'],
            Q3_source=r['Q3_source'],inputs_source=inputs,index_source=INDEX))
    return out


def build_inputs(runtime=None):
    check_source();prior=read(VBASE,VDIR+'/Q3_INPUTS.json')['inputs'];baseline=deepcopy(prior['cases'][0])
    config=read(VBASE,VDIR+'/generation/CONFIGURATION.json')
    if config['id']!=baseline['original_configuration_id'] or config['spec']['L_D']!=7:raise ValueError('LD7 configuration differs')
    U.metadata(baseline)
    rows=prices(config);caps=[n for n in V0 for _ in ('X','Y')];p=[rational(r['C_ref']) for r in rows]
    catalog=saved_catalog(rows,caps);index=read(INDEX['commit'],INDEX['path'])
    achieved=min((r for r in index['rows'] if r['status']==PASS),key=lambda r:rational(r['cost']))
    proposals=[];cases=[];seen={}
    for tag,budget in BUDGETS:
        for rule in RULES:
            label='LD7_'+tag+'__'+rule;g=generate(caps,p,budget,rule)
            if g is None:
                proposals.append(dict(label=label,budget_name=tag,budget=budget,rule=rule,status='NO_BUDGET_FEASIBLE_BREAKPOINT'));continue
            vector=g['shots'];cost=sum(n*c for n,c in zip(vector,p))
            if cost!=g['conditional_cost'] or cost>budget or any(type(n) is not int or not 1<=n<=cap for n,cap in zip(vector,caps)):
                raise ValueError('Allocation contract')
            shots=[dict(m=m,N_X=vector[2*m],N_Y=vector[2*m+1]) for m in range(13)]
            terms=[dict(m=r['m'],axis=r['axis'],N=n,C_ref=enc(c),N_times_C_ref=enc(n*c),physical_id=r['physical_id'],
                q=r['q'],application=r['application'],price_source=dict(**PRICE,selector=['rows',i]))
                for i,(r,n,c) in enumerate(zip(rows,vector,p))]
            early=sum(n*c for r,n,c in zip(rows,vector,p) if r['m']<7);late=cost-early
            case=deepcopy(baseline)
            case.update(label=label,L_D=7,rule=rule,budget_name=tag,budget=budget,
                allocation=dict(allocation_id='PR_DF_LD7_'+tag+'_'+rule+'_V0_CAP_01',shot_policy='USER_SUPPLIED',shots=shots),
                reference_cost=enc(cost),unused_budget=enc(budget-cost),cost_m0_m6=enc(early),cost_m7_m12=enc(late),
                m7_m12_fraction=enc(late/cost),
                generation={k:(enc(v) if isinstance(v,Fraction) else v) for k,v in g.items() if k!='shots'},price_terms=terms)
            b=W.Q.bounds(case);o=prior['options'];case['finite_scope_preflight']=b
            if (b['maximum_round_outcomes']>o['maximum_round_outcomes'] or b['verification_pairs_upper']>o['maximum_verification_pairs']
                    or b['transition_pairs_upper']>o['total_state_refinement']['maximum_transition_pairs']):
                raise ValueError('Count-only scope exceeded before science: '+label)
            k=key(shots);saved=next((s['label'] for s in catalog if key(s['shots'])==k),None)
            if k in seen:
                seen[k]['aliases'].append(label)
                proposals.append(dict(label=label,budget_name=tag,budget=budget,rule=rule,duplicate_of=seen[k]['label'],
                    allocation=case['allocation'],reference_cost=case['reference_cost'],unused_budget=case['unused_budget']))
                continue
            case['aliases']=[label];case['saved_identical_vector']=saved;seen[k]=case;cases.append(case)
            proposals.append(dict(label=label,budget_name=tag,budget=budget,rule=rule,allocation=case['allocation'],
                reference_cost=case['reference_cost'],unused_budget=case['unused_budget'],saved_identical_vector=saved))
    cases.sort(key=lambda c:(rational(c['reference_cost']),c['label']))
    for order,c in enumerate(cases,1):c['cost_order']=order
    return dict(source_route=dict(commit=D2_HEAD,path='scripts/run_pr_df_ld_split_q3_resume.py'),
        signal_and_contract=dict(commit=VBASE,path=VDIR+'/Q3_INPUTS.json',selector=['inputs']),
        configuration=dict(commit=VBASE,path=VDIR+'/generation/CONFIGURATION.json'),price_source=PRICE,index_source=INDEX,
        generator_source=GENERATOR,validator_source=VALIDATOR,budgets=[dict(name=k,cost_RZ=v) for k,v in BUDGETS],caps_V0=V0,
        proposals=proposals,cases=cases,evaluation_order=[c['label'] for c in cases],saved_catalog=catalog,
        previous_LD7_achieved=dict(label=achieved['label'],cost=achieved['cost'],shots=achieved['shots'],Q3_source=achieved['Q3_source']),
        maximum_scientific_calls=4,concurrent_configurations=1,
        **{k:deepcopy(prior[k]) for k in ('statistical_contract','options','adopted_evaluation','acceleration')},
        runtime=runtime if runtime is not None else W.runtime_check(),
        selection='EXACT_COST_ASCENDING_THEN_LABEL; STOP_AT_FIRST_PASS_OR_UNKNOWN; EXCEEDS/INCONCLUSIVE CONTINUE; NO_RETRY',
        price_quality=dict(kind='LD7_OWN_EXPLORATORY_MODEL_REFERENCE_NOT_LAW_EXPECTATION',m7_m12='LONG_ROUND_EXTRAPOLATION_BEYOND_Q64',CI=None),
        duplicate_check='SCOPE_AND_PREFLIGHT.md',sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)


def cached_record(case,inputs):
    for item in inputs['saved_catalog']:
        if key(case['allocation']['shots'])!=key(item['shots']):continue
        saved=read(item['inputs_source']['commit'],item['inputs_source']['path'])['inputs']
        if any(saved[k]!=inputs[k] for k in ('statistical_contract','options','adopted_evaluation')):
            raise ValueError('Saved risk/loss/law/numerical contract differs')
        old=next(c for c in saved['cases'] if key(c['allocation']['shots'])==key(item['shots']))
        for k in ('configuration','signal_packet','binding_id','original_configuration_id','total_configuration_id','evaluation_target'):
            if case[k]!=old[k]:raise ValueError('Saved SOURCE/context differs: '+k)
        result=read(item['Q3_source']['commit'],item['Q3_source']['path']);record=result['record']
        if record['allocation']['shots']!=case['allocation']['shots'] or record['evaluation_target']!=case['evaluation_target']:
            raise ValueError('Saved canonical result/vector mismatch')
        risk=record['diagnostic']['deterministic_risk'];status=U.H.assess(risk,inputs['adopted_evaluation'],item['Q3_source'])['status']
        return dict(status=status,MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],scientific_evaluation_calls=0,
            input_disposition='REUSED_SAVED_LD7_RESULT',reused_label=item['label'],Q3_source=item['Q3_source'])
    return None


def search(inputs,visit):
    history=[];new_calls=0;stop='FINITE_MENU_EXHAUSTED';best=None
    for sequence,case in enumerate(inputs['cases'],1):
        if new_calls>=inputs['maximum_scientific_calls']:stop='SCIENTIFIC_CALL_LIMIT';break
        outcome=visit(case,sequence)
        if outcome['scientific_evaluation_calls'] not in (0,1):raise ValueError('One call per unique vector')
        new_calls+=outcome['scientific_evaluation_calls'];history.append(dict(label=case['label'],**outcome))
        if outcome['status']=='UNKNOWN':stop='MAJOR_GATE_OR_PERSISTENCE_FAILURE';break
        if outcome['status']==PASS:best=case['label'];stop='FIRST_FEASIBLE_IN_COST_ORDER';break
    held=[r['label'] for r in history if r['status'] not in (PASS,FAIL)]
    cheaper=[] if best is None else [r['label'] for r in history[:-1] if r['status']!=FAIL]
    return dict(stop_reason=stop,history=history,new_scientific_calls=new_calls,best_confirmed=best,
        cheaper_unresolved=cheaper,held=held,minimal_within_generated_set=best is not None and not cheaper,
        unevaluated=[c['label'] for c in inputs['cases'] if c['label'] not in {h['label'] for h in history}],
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)


def prepare():
    inputs=build_inputs();p=ROOT/DIRECTORY;p.mkdir(exist_ok=True);W.write(p/'INPUTS.json',inputs)
    for c in inputs['cases']:
        print(c['cost_order'],c['label'],[r['N_X'] for r in c['allocation']['shots']],float(rational(c['reference_cost'])),
            float(rational(c['unused_budget'])),c['finite_scope_preflight']['transition_pairs_upper'],c['saved_identical_vector'],flush=True)


def execute():
    check_source();sha=W.head()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=W.blob(sha,path):raise ValueError('Unfixed caller/INPUTS')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs():raise ValueError('Fixed menu/price/contract/runtime differs')
    run=ROOT/DIRECTORY/'run01';run.mkdir()
    source=dict(commit=sha,path=CALLER,inputs=dict(commit=sha,path=INPUT),runtime=inputs['runtime'])
    U.R.write_new(run/'EXECUTION.json',dict(source=source,started_UTC=U.R.utc(),maximum_scientific_calls=4,
        evaluation_order=inputs['evaluation_order']))
    def visit(case,sequence):
        step=run/f'{sequence:02}';step.mkdir()
        U.R.write_new(step/'SELECTION.json',dict(sequence=sequence,label=case['label'],reason='FIXED_EXACT_COST_ASCENDING',
            reference_cost=case['reference_cost'],source=source))
        hit=cached_record(case,inputs)
        if hit is not None:
            U.R.write_new(step/'OUTCOME.json',hit);print(case['label'],hit['status'],'(reused)',flush=True);return hit
        folder=step/case['label'];folder.mkdir()
        supervision=W.supervise([case],inputs,source,step)
        gate=bool(supervision['process_exitcodes']) and all(p['exitcode']==0 for p in supervision['process_exitcodes'])
        U.R.write_new(step/'SUPERVISION.json',dict(**supervision,process_exit_gate=dict(all_exit_codes_zero=gate)))
        try:
            outcome=validate_result(case,inputs,folder,source)
            if not gate or supervision['errors'] or supervision['monitor_errors'] or len(supervision['receipts'])!=1:
                outcome.update(status='UNKNOWN',reason='SUPERVISOR_OR_PROCESS_GATE')
        except Exception as exc:
            stages=folder/'STAGES.jsonl'
            calls=sum(json.loads(line)['stage']=='SCIENTIFIC_CALL_START' for line in stages.read_text().splitlines()) if stages.exists() else 0
            outcome=dict(status='UNKNOWN',reason=type(exc).__name__+': '+str(exc),scientific_evaluation_calls=calls,MSE_lower=None,MSE_upper=None)
        U.R.write_new(step/'OUTCOME.json',outcome);print(case['label'],outcome['status'],flush=True)
        return outcome
    result=search(inputs,visit);U.R.write_new(run/'SEARCH_HISTORY.json',dict(result,source=source))
    U.R.write_new(run/'STOP.json',dict(source=source,finished_UTC=U.R.utc(),stop_reason=result['stop_reason'],
        new_scientific_calls=result['new_scientific_calls'],mandatory_stop=True,automatic_retry=False))
    print('STOP:',result['stop_reason'],'calls:',result['new_scientific_calls'],flush=True)
    return 0 if result['stop_reason']!='MAJOR_GATE_OR_PERSISTENCE_FAILURE' else 2


def show(x,digits=6):
    x=Fraction(x);sign='-' if x<0 else '';x=abs(x);s=10**digits;q=(2*x*s).__floor__();q=(q+1)//2
    return '%s%d.%0*d'%(sign,q//s,digits,q%s)


def sci(x):
    return None if x is None else '%.16e'%float(rational(x))


def summarize():
    p=ROOT/DIRECTORY;inputs=json.loads((p/'INPUTS.json').read_text())
    history=json.loads((p/'run01/SEARCH_HISTORY.json').read_text());tau=rational(inputs['adopted_evaluation']['MSE_upper'])
    pf={name:value for name,value in BUDGETS};done={h['label']:h for h in history['history']};rows=[]
    for c in inputs['cases']:
        h=done.get(c['label']);T=rational(c['reference_cost'])
        row=dict(cost_order=c['cost_order'],label=c['label'],aliases=c['aliases'],budget_name=c['budget_name'],budget=c['budget'],rule=c['rule'],
            N_X=[r['N_X'] for r in c['allocation']['shots']],N_Y=[r['N_Y'] for r in c['allocation']['shots']],
            allocation_id=c['allocation']['allocation_id'],conditional_cost=c['reference_cost'],conditional_cost_display=show(T),
            unused_budget=c['unused_budget'],unused_budget_display=show(rational(c['unused_budget'])),
            cost_m0_m6=c['cost_m0_m6'],cost_m7_m12=c['cost_m7_m12'],m7_m12_fraction=c['m7_m12_fraction'],
            m7_m12_percent_display=show(100*rational(c['m7_m12_fraction']),4),
            status='NOT_EVALUATED_EARLY_STOP' if h is None else h['status'],
            MSE_lower=None if h is None else h.get('MSE_lower'),MSE_upper=None if h is None else h.get('MSE_upper'),
            MSE_lower_display=None if h is None else sci(h.get('MSE_lower')),MSE_upper_display=None if h is None else sci(h.get('MSE_upper')),
            upper_over_tau_display=None if h is None or h.get('MSE_upper') is None else '%.6f'%float(rational(h['MSE_upper'])/tau),
            lower_over_tau_display=None if h is None or h.get('MSE_lower') is None else '%.6f'%float(rational(h['MSE_lower'])/tau),
            scientific_evaluation_calls=0 if h is None else h['scientific_evaluation_calls'],
            result_source=None if h is None else (h.get('execution_source') or h.get('result_source') or h.get('Q3_source')),
            API_path=None if h is None else f"{DIRECTORY}/run01/{c['cost_order']:02}/{c['label']}/API_RETURN.json" if h['scientific_evaluation_calls'] else None,
            comparisons=[dict(benchmark=name,T_PF=value,T_minus_T_PF=enc(T-value),T_minus_T_PF_display=show(T-value),
                ratio_display='%.6f'%float(T/value)) for name,value in pf.items()])
        rows.append(row)
    best=history['best_confirmed'];previous=inputs['previous_LD7_achieved']
    if best is not None:
        b=next(r for r in rows if r['label']==best);achieved=dict(label=best,cost=b['conditional_cost'],source='THIS_RUN',
            updated=rational(b['conditional_cost'])<rational(previous['cost']))
    else:
        achieved=dict(label=previous['label'],cost=previous['cost'],source='SAVED_LD7_INDEX',updated=False,Q3_source=previous['Q3_source'])
    achieved['cost_display']=show(rational(achieved['cost']))
    achieved['comparisons']=[dict(benchmark=n,T_PF=v,T_minus_T_PF=enc(rational(achieved['cost'])-v),
        T_minus_T_PF_display=show(rational(achieved['cost'])-v),ratio_display='%.6f'%float(rational(achieved['cost'])/v)) for n,v in pf.items()]
    summary=dict(kind='PR_DF_LD7_PF_BUDGET_FINITE_MENU',source=json.loads((p/'run01/EXECUTION.json').read_text())['source'],
        stop_reason=history['stop_reason'],new_scientific_calls=history['new_scientific_calls'],best_confirmed=best,
        minimal_within_generated_set=history['minimal_within_generated_set'],cheaper_unresolved=history['cheaper_unresolved'],
        held=history['held'],unevaluated=history['unevaluated'],rows=rows,LD7_confirmed_cheapest=achieved,
        PF_benchmarks=pf,tau=inputs['adopted_evaluation']['MSE_upper'],price_quality=inputs['price_quality'],
        process_return=json.loads((p/'PROCESS_RETURN.json').read_text()),
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    U.R.write_new(p/'SUMMARY.json',summary)
    with (p/'SUMMARY.csv').open('x',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['cost_order','label','budget_name','rule','N_X','N_Y','conditional_cost','unused_budget','m7_m12_percent',
            'status','MSE_lower','MSE_upper','upper_over_tau','T_minus_p7','T_minus_FullPF'])
        for r in rows:
            c={x['benchmark']:x for x in r['comparisons']}
            w.writerow([r['cost_order'],r['label'],r['budget_name'],r['rule'],' '.join(map(str,r['N_X'])),' '.join(map(str,r['N_Y'])),
                r['conditional_cost_display'],r['unused_budget_display'],r['m7_m12_percent_display'],r['status'],r['MSE_lower_display'],
                r['MSE_upper_display'],r['upper_over_tau_display'],c['P7_S2']['T_minus_T_PF_display'],c['FULL_PF_S2']['T_minus_T_PF_display']])
    cases={c['label']:c for c in inputs['cases']}
    U.R.write_new(p/'C_RESULT_JOIN.json',dict(schema='PR_DF_LD7_PF_BUDGET_MODEL_COST_Q3_JOIN_FOR_C_V1',
        source=summary['source'],configuration=inputs['configuration'],configuration_id=next(iter(cases.values()))['original_configuration_id'],
        total_configuration_id=next(iter(cases.values()))['total_configuration_id'],binding_id=next(iter(cases.values()))['binding_id'],
        signal_packet=next(iter(cases.values()))['signal_packet'],evaluation_target=next(iter(cases.values()))['evaluation_target'],
        price_source=inputs['price_source'],index_source=inputs['index_source'],price_quality=inputs['price_quality'],
        tau=inputs['adopted_evaluation']['MSE_upper'],PF_benchmarks=pf,
        rows=[dict(label=r['label'],aliases=r['aliases'],cost_order=r['cost_order'],budget_name=r['budget_name'],budget=r['budget'],rule=r['rule'],
            allocation=cases[r['label']]['allocation'],price_terms=cases[r['label']]['price_terms'],conditional_cost=r['conditional_cost'],
            unused_budget=r['unused_budget'],cost_m0_m6=r['cost_m0_m6'],cost_m7_m12=r['cost_m7_m12'],m7_m12_fraction=r['m7_m12_fraction'],
            model_status=r['status'],MSE_lower=r['MSE_lower'],MSE_upper=r['MSE_upper'],result_source=r['result_source'],API_path=r['API_path'],
            comparisons=r['comparisons']) for r in rows],
        stop_reason=summary['stop_reason'],best_confirmed=best,cheaper_unresolved=summary['cheaper_unresolved'],held=summary['held'],
        LD7_confirmed_cheapest=achieved,sufficient_N=None,G_qualified=None))
    print(json.dumps(dict(stop=summary['stop_reason'],best=best,achieved=achieved['cost_display'],
        rows=[(r['label'],r['conditional_cost_display'],r['status'],r['MSE_upper_display']) for r in rows]),indent=1))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute','summarize'))
    mode=parser.parse_args().mode
    if mode=='execute':raise SystemExit(execute())
    globals()[mode]()

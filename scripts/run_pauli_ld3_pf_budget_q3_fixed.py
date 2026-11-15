"""Four adopted LD3 budget proposals, increasing-cost sequential Q3, no retry."""
import argparse
import ast
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_ld3_budget_fixed_route', ROOT/'scripts/run_pauli_ld3_vp_q3_fixed.py')
L = importlib.util.module_from_spec(spec); sys.modules[spec.name] = L; spec.loader.exec_module(L)
T, R, M = L.T, L.R, L.M
BASE = '2de33a4f4daeef7ca899dc98689974bb0eaadaf4'
GENERATOR = dict(commit='e094b2d60c5c4468711dea244fa2f91fb587ac12',
    path='scripts/run_pauli_ld2_reallocation_q3_fixed.py', function='generate')
PRICE = dict(commit='f25a05f8a92aab43cfbe30fc80b974054156aac1',
    path='docs/research/a_pauli_ld3_cal_price_20261116/HANDOFF_BC.json')
DIRECTORY = 'docs/research/pauli_ld3_pf_budget_q3_results_20261116'
INPUT = DIRECTORY+'/INPUTS.json'
CALLER = 'scripts/run_pauli_ld3_pf_budget_q3_fixed.py'
BUDGETS = [('P7',395962678),('FULL_PF',559457386)]
RULES = ('A_SHAPE','B_PRICE')
PASS = 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
FAIL = 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'


def function(source, scope):
    tree = ast.parse(R.blob(source['commit'],source['path']))
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == source['function']]
    if len(nodes) != 1: raise ValueError('Pinned helper unavailable')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),source['commit']+':'+source['path'],'exec'),scope)
    return scope[source['function']]


generate = function(GENERATOR, dict(Fraction=Fraction))
validate_result = function(dict(commit=BASE,path='scripts/run_p7_s2_local_shot_search.py',function='validate_result'),
    dict(ROOT=ROOT,json=json,exact=T.Q.a.exact_scalar,F=NS(H=L.H)))


def document(source):
    value = json.loads(R.blob(source['commit'],source['path']))
    for key in source.get('selector',[]): value = value[key]
    return value


def exact(q):
    return T.Q.a.exact_value(Fraction(q))


def rational(value):
    if type(value) is int or isinstance(value,str): return Fraction(value)
    if isinstance(value,dict) and set(value) == {'numerator','denominator'}:
        return Fraction(value['numerator'],value['denominator'])
    raise ValueError('Exact integer/rational price required')


def key(case):
    return tuple((r['m'],r['N_X'],r['N_Y']) for r in case['allocation']['shots'])


def check_source():
    L.check_source()
    for path in ['scripts/run_pauli_ld3_vp_q3_fixed.py'] + ['src/trottertracks/'+n+'.py' for n in L.ADAPTED]:
        if (ROOT/path).read_bytes() != R.blob(BASE,path): raise ValueError('Fixed LD3 route differs: '+path)


def joined_prices(table,binding,view):
    if (table['kind'] != 'MODEL_REFERENCE_PRICE_NOT_E_C' or table['qualified_for_candidate'] is not False
            or any(table[k] is not None for k in ('E_C','CI','G','winner'))):
        raise ValueError('Reference price must remain unqualified, not E[C]')
    old,new = binding['original_configuration'],binding['total_configuration']
    if table['configuration_source'] != binding['original_source'] or table['total_view']['id'] != new['id']:
        raise ValueError('Price configuration SOURCE differs')
    rows = sorted(enumerate(table['rows']),key=lambda v:(v[1]['m'],v[1]['axis']))
    if [(r['m'],r['axis']) for _,r in rows] != [(m,a) for m in range(13) for a in ('X','Y')]:
        raise ValueError('Exactly 26 distinct price axes required')
    for index,row in rows:
        m = row['m']; origin,total = old['rounds'][m],new['rounds'][m]
        if (row['original_configuration_id'] != old['id'] or row['total_configuration_id'] != new['id']
                or row['original_candidate_id'] != origin['candidate']['id']
                or row['total_candidate_id'] != total['candidate']['id'] or row['binding_id'] != binding['id']
                or (row['q'],row['R'],row['events']) != (origin['q_PF'],origin['R_outer'],origin['structure']['event_count'])
                or row['value_source'] != 'MODEL_APPLICATION_FROM_Q8R660_DECOMPOSITION'
                or row['N'] != L.VECTOR[m] or rational(row['C_ref_exact']) <= 0):
            raise ValueError('Price physical/schedule/model context differs at '+str((m,row['axis'])))
        joined = document(row['b_index_source']); physical = view['signal_packet']['means'][m]['canonical_origin']['provider_binding']['physical_context']
        if (joined['original_physical_context'] != physical or joined['original_law_binding'] != origin['law_binding']
                or joined['dictionary_identity'] != view['signal_packet']['dictionary_identity']
                or joined['signal_source'] != view['signal_packet']['means'][m]['source']
                or joined['axis'] != row['axis'] or joined['m'] != m):
            raise ValueError('Price signal/law/dictionary SOURCE join differs')
        quote = document(row['price_source'])
        if ((quote['m'],quote['axis'],quote['physical_id']) != (m,row['axis'],row['original_candidate_id'])
                or quote['C_ref_exact'] != row['C_ref_exact']):
            raise ValueError('Original price record differs')
    return rows


def build_inputs():
    check_source()
    prior = document(dict(commit=BASE,path=L.INPUT)); seed = deepcopy(prior['cases'][0])
    binding,view = L.metadata(seed); table = document(PRICE); rows = joined_prices(table,binding,view)
    caps = [L.VECTOR[r['m']] for _,r in rows]; prices = [rational(r['C_ref_exact']) for _,r in rows]
    seed_cost = sum(n*c for n,c in zip(caps,prices))
    if seed_cost != rational(table['T_LD3']): raise ValueError('Saved VP model cost differs')
    proposals=[]; cases=[]; seen={}
    for tag,budget in BUDGETS:
        for rule in RULES:
            g = generate(caps,prices,budget,rule)
            if g is None:
                proposals.append(dict(budget_name=tag,budget=budget,rule=rule,status='NO_BUDGET_FEASIBLE_BREAKPOINT')); continue
            case = deepcopy(seed); label = tag+'__'+rule
            case.update(label=label,rule=rule,budget_name=tag,budget=budget,
                allocation=dict(allocation_id='PAULI_LD3_'+label+'_VP_CAP_01',shot_policy='USER_SUPPLIED',
                    shots=[dict(m=m,N_X=g['shots'][2*m],N_Y=g['shots'][2*m+1]) for m in range(13)]),
                conditional_cost=exact(g['conditional_cost']),unused_budget=exact(g['unused_budget']),
                generation={k:(exact(v) if isinstance(v,Fraction) else v) for k,v in g.items() if k != 'shots'},
                price_terms=[dict(m=r['m'],axis=r['axis'],N=n,C_ref=exact(c),N_times_C_ref=exact(n*c),
                    original_candidate_id=r['original_candidate_id'],total_candidate_id=r['total_candidate_id'],
                    price_class=r['value_source'],price_source=dict(**PRICE,selector=['rows',index]),
                    original_price_source=deepcopy(r['price_source']),b_index_source=deepcopy(r['b_index_source']))
                    for (index,r),n,c in zip(rows,g['shots'],prices)])
            if not all(type(n) is int and 1<=n<=cap for n,cap in zip(g['shots'],caps)):
                raise ValueError('Generated integer vector exceeds cap')
            if sum(rational(t['N_times_C_ref']) for t in case['price_terms']) != g['conditional_cost']:
                raise ValueError('Exact axis cost mismatch')
            case['finite_scope_preflight'] = L.H.finite_bounds([r['N_X'] for r in case['allocation']['shots']])
            if any(r['N_X'] != r['N_Y'] for r in case['allocation']['shots']):
                # The fixed X/Y prices are equal. Do not silently use the equal-axis bound otherwise.
                raise ValueError('Expected equal X/Y prices/vector for this fixed table')
            b,o = case['finite_scope_preflight'],prior['options']
            if (b['maximum_round_outcomes']>o['maximum_round_outcomes'] or b['verification_pairs_upper']>o['maximum_verification_pairs']
                    or b['transition_pairs_upper']>o['total_state_refinement']['maximum_transition_pairs']):
                raise ValueError('Count-only scope exceeded before science')
            vector = key(case)
            if vector in seen:
                canonical=seen[vector];canonical['aliases'].append(label)
                proposals.append(dict(label=label,duplicate_of=canonical['label'],budget=budget,rule=rule,
                    allocation=case['allocation'],conditional_cost=case['conditional_cost'],unused_budget=case['unused_budget']))
            else:
                case['aliases']=[label];seen[vector]=case;cases.append(case)
                proposals.append(dict(label=label,budget=budget,rule=rule,allocation=case['allocation'],
                    conditional_cost=case['conditional_cost'],unused_budget=case['unused_budget']))
    cases.sort(key=lambda c:(rational(c['conditional_cost']),c['label']))
    return dict(B_source=BASE,price_source=PRICE,generator_source=GENERATOR,cases=cases,proposals=proposals,
        evaluation_order=[c['label'] for c in cases],maximum_scientific_calls=4,concurrent_configurations=1,
        budgets=[dict(name=k,cost_RZ=v) for k,v in BUDGETS],caps_VP=L.VECTOR,seed_case=seed,
        seed_cost=exact(seed_cost),saved_cache=[dict(case=seed,input_source=dict(commit=BASE,path=L.INPUT),
            API_source=document(dict(commit=BASE,path=L.DIRECTORY+'/RESULT_SOURCE.json')))],
        statistical_contract=prior['statistical_contract'],options=prior['options'],acceleration=prior['acceleration'],
        adopted_evaluation=prior['adopted_evaluation'],kernel_source=prior['kernel_source'],C_source=prior['C_source'],
        price_quality=dict(kind='MODEL_REFERENCE_PRICE_NOT_E_C',model_axes=26,direct_axes=0,
            qualified_for_candidate=False,E_C=None,CI=None),duplicate_check='SCOPE_AND_PREFLIGHT.md',
        selection='EXACT_COST_ASCENDING_THEN_LABEL; STOP_AT_FIRST_PASS_NO_SPECULATIVE_CALLS',
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)


def cached_record(case,inputs):
    for item in inputs['saved_cache']:
        if key(case) != key(item['case']): continue
        old = document(item['input_source']); seed = item['case']; record = document(item['API_source'])[0]
        for k in ('configuration','signal_packet','binding_id','original_configuration_id','total_configuration_id','evaluation_target'):
            if case[k] != seed[k]: raise ValueError('Saved result SOURCE/context differs: '+k)
        if any(old[k] != inputs[k] for k in ('statistical_contract','options','adopted_evaluation')):
            raise ValueError('Saved risk/loss/law/numerical contract differs')
        if (record['configuration_id'] != case['total_configuration_id'] or record['canonical_binding'] != case['binding_id']
                or record['evaluation_target'] != case['evaluation_target'] or record['allocation']['shots'] != case['allocation']['shots']):
            raise ValueError('Saved canonical result/vector mismatch')
        risk = record['diagnostic']['deterministic_risk']; assessment = L.H.assess(risk,inputs['adopted_evaluation'],item['API_source'])
        return dict(status=assessment['status'],MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],
            scientific_evaluation_calls=0,input_disposition='REUSED',API_source=item['API_source'])
    return None


def search(inputs,visit):
    history=[];new_calls=0;stop='FINITE_MENU_EXHAUSTED';best=None
    for sequence,case in enumerate(inputs['cases'],1):
        if new_calls >= inputs['maximum_scientific_calls']: stop='SCIENTIFIC_CALL_LIMIT';break
        outcome = visit(case,sequence)
        if outcome['scientific_evaluation_calls'] not in (0,1): raise ValueError('One call per unique vector')
        new_calls += outcome['scientific_evaluation_calls']; history.append(dict(label=case['label'],**outcome))
        if outcome['status'] == 'UNKNOWN': stop='MAJOR_GATE_OR_PERSISTENCE_FAILURE';break
        if outcome['status'] == PASS:
            best=case['label'];stop='FIRST_FEASIBLE_IN_COST_ORDER';break
    before = [] if best is None else [r['label'] for r in history[:-1] if r['status'] != FAIL]
    return dict(stop_reason=stop,history=history,new_scientific_calls=new_calls,best_confirmed=best,
        cheaper_unresolved=before,minimal_within_generated_set=best is not None and not before,
        unevaluated=[c['label'] for c in inputs['cases'] if c['label'] not in {h['label'] for h in history}],
        achieved_conditional_cost=inputs['seed_cost'] if best is None else next(c['conditional_cost'] for c in inputs['cases'] if c['label']==best),
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)


def prepare():
    inputs=build_inputs();p=ROOT/DIRECTORY;p.mkdir(exist_ok=True);R.write_new(p/'INPUTS.json',inputs)
    for c in inputs['cases']: print(c['label'],[r['N_X'] for r in c['allocation']['shots']],rational(c['conditional_cost']),flush=True)


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path): raise ValueError('Unfixed caller/INPUTS')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs(): raise ValueError('Fixed menu/price/contract differs')
    run=ROOT/DIRECTORY/'run01';run.mkdir();source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(run/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=4))
    def visit(case,sequence):
        hit=cached_record(case,inputs)
        if hit is not None:return hit
        step=run/f'{sequence:02}';step.mkdir();folder=step/case['label'];folder.mkdir()
        R.write_new(step/'SELECTION.json',dict(sequence=sequence,case=case,reason='FIXED_EXACT_COST_ASCENDING'))
        with L.installed(step): supervision=M.supervise([case],inputs,source,step)
        exits=[c for p in step.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines() for c in json.loads(line)['children']]
        gate=bool(exits) and all(c['exitcode']==0 for c in exits)
        R.write_new(step/'SUPERVISION.json',dict(**supervision,process_exit_gate=dict(recorded_children=len(exits),all_exit_codes_zero=gate)))
        try:
            outcome=validate_result(case,inputs,folder,source)
            outcome['execution_source']=outcome.pop('result_source')
            outcome['API_path']=str((folder/'API_RETURN.json').relative_to(ROOT)) if (folder/'API_RETURN.json').exists() else None
            if not gate or supervision['errors'] or supervision['monitor_errors']:
                outcome.update(status='UNKNOWN',reason='SUPERVISOR_OR_PROCESS_GATE')
        except Exception as exc:
            stages=folder/'STAGES.jsonl'
            calls=sum(json.loads(line)['stage']=='SCIENTIFIC_CALL_START' for line in stages.read_text().splitlines()) if stages.exists() else 0
            outcome=dict(status='UNKNOWN',reason=type(exc).__name__+': '+str(exc),scientific_evaluation_calls=calls,MSE_lower=None,MSE_upper=None)
        R.write_new(step/'OUTCOME.json',outcome)
        print(case['label'],outcome['status'],flush=True)
        return outcome
    result=search(inputs,visit);R.write_new(run/'SEARCH_HISTORY.json',result)
    R.write_new(run/'STOP.json',dict(source=source,finished_UTC=R.utc(),stop_reason=result['stop_reason'],
        new_scientific_calls=result['new_scientific_calls'],mandatory_stop=True,automatic_retry=False))
    print('STOP:',result['stop_reason'],'calls:',result['new_scientific_calls'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()

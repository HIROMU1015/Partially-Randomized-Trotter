"""LD7 PF-budget menu caller: generation, reuse, search, validator, entry, summary.

Artificial or saved-file checks only; no real H4 Q3 is started.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
import run_pr_df_ld7_pf_budget_q3_fixed as X
import freeze_pr_df_ld_split as F

EXPECTED={'LD7_P7_S2__A_SHAPE':[15,14,13,12,11,10,9,8,7,6,5,4,2],
          'LD7_P7_S2__B_PRICE':[64,60,56,52,47,43,38,34,26,10,4,1,1],
          'LD7_FULL_PF_S2__A_SHAPE':[22,21,19,18,16,15,13,11,10,8,7,5,4],
          'LD7_FULL_PF_S2__B_PRICE':[64,60,56,52,47,43,38,34,30,19,7,3,1]}


@pytest.fixture(scope='module')
def inputs():
    with patch.object(X.U.T,'evaluate',side_effect=AssertionError('No H4 risk')),\
            patch.object(X.U.P,'worker',side_effect=AssertionError('No worker')):
        return X.build_inputs()


def test_four_fixed_vectors_costs_order_and_caps(inputs):
    cases=inputs['cases'];assert [c['label'] for c in cases]==list(EXPECTED)
    rows=X.read(X.PRICE['commit'],X.PRICE['path'])['rows'];p=[X.rational(r['C_ref']) for r in rows]
    costs=[]
    for c in cases:
        nx=[r['N_X'] for r in c['allocation']['shots']];ny=[r['N_Y'] for r in c['allocation']['shots']]
        assert nx==ny==EXPECTED[c['label']]
        assert all(1<=n<=cap for n,cap in zip(nx,X.V0))
        vector=[n for n in nx for _ in 'XY'];T=sum(n*q for n,q in zip(vector,p));costs.append(T)
        assert X.rational(c['reference_cost'])==T<=c['budget']
        assert X.rational(c['unused_budget'])==c['budget']-T
        assert X.rational(c['cost_m0_m6'])+X.rational(c['cost_m7_m12'])==T
        assert sum(X.rational(t['N_times_C_ref']) for t in c['price_terms'])==T
        assert c['saved_identical_vector'] is None and X.cached_record(c,inputs) is None
    assert costs==sorted(costs) and [c['cost_order'] for c in cases]==[1,2,3,4]
    prior=X.read(X.VBASE,X.VDIR+'/Q3_INPUTS.json')['inputs']
    for k in ('statistical_contract','options','adopted_evaluation','acceleration'):assert inputs[k]==prior[k]
    for c in cases:
        for k in ('configuration','signal_packet','binding_id','original_configuration_id','total_configuration_id','evaluation_target'):
            assert c[k]==prior['cases'][0][k]


def test_generator_reproduces_saved_ld7_menu_and_matches_d2_copy():
    rows=X.read(X.PRICE['commit'],X.PRICE['path'])['rows'];p=[X.rational(r['C_ref']) for r in rows]
    caps=[n for n in X.V0 for _ in 'XY'];budget=Fraction(3,4)*sum(n*q for n,q in zip(caps,p))
    saved=X.read('ca7c016f5b6bc084de82e047ffc30befea73cb4c',X.PDIR+'/ALLOCATION_INPUTS.json')['menu']
    for rule in ('A_SHAPE','B_PRICE'):
        ours=X.generate(caps,p,budget,rule);theirs=F.generate(caps,p,budget,rule)
        assert ours['shots']==theirs['shots'] and ours['conditional_cost']==theirs['conditional_cost']
        old=next(m for m in saved if m['rule']==rule)
        assert [s['N_'+a] for s in old['shots'] for a in 'XY']==ours['shots']
        assert X.rational(old['cost'])==ours['conditional_cost']
    for tag,b in X.BUDGETS:
        for rule in X.RULES:
            assert X.generate(caps,p,b,rule)['shots']==F.generate(caps,p,b,rule)['shots']


def test_identical_saved_vector_is_reused_and_foreign_context_rejected(inputs):
    case=deepcopy(inputs['cases'][0]);a=next(s for s in inputs['saved_catalog'] if s['label']=='LD7_A_SHAPE')
    case['allocation']['shots']=a['shots'];hit=X.cached_record(case,inputs)
    assert hit['status']==X.PASS and hit['scientific_evaluation_calls']==0 and hit['reused_label']=='LD7_A_SHAPE'
    bad=deepcopy(case);bad['binding_id']='FOREIGN'
    with pytest.raises(ValueError):X.cached_record(bad,inputs)
    law=deepcopy(inputs);law['options']=dict(law['options'],probability_rule='FOREIGN')
    with pytest.raises(ValueError):X.cached_record(case,law)


def toy(statuses):
    cases=[dict(label=f'C{i}') for i in range(len(statuses))]
    def visit(case,sequence):
        return dict(status=statuses[sequence-1],scientific_evaluation_calls=1)
    return dict(cases=cases,maximum_scientific_calls=4),visit


@pytest.mark.parametrize('statuses,stop,best,cheaper,held,evaluated',[
    ((X.FAIL,X.PASS,X.FAIL,X.FAIL),'FIRST_FEASIBLE_IN_COST_ORDER','C1',[],[],2),
    (('INCONCLUSIVE',X.PASS,X.FAIL,X.FAIL),'FIRST_FEASIBLE_IN_COST_ORDER','C1',['C0'],['C0'],2),
    ((X.FAIL,'UNKNOWN',X.PASS,X.PASS),'MAJOR_GATE_OR_PERSISTENCE_FAILURE',None,[],['C1'],2),
    ((X.FAIL,X.FAIL,X.FAIL,X.FAIL),'FINITE_MENU_EXHAUSTED',None,[],[],4)])
def test_sequential_search_rules(statuses,stop,best,cheaper,held,evaluated):
    inputs,visit=toy(statuses);r=X.search(inputs,visit)
    assert r['stop_reason']==stop and r['best_confirmed']==best and r['cheaper_unresolved']==cheaper
    assert r['held']==held and len(r['history'])==evaluated and r['new_scientific_calls']==evaluated
    assert r['minimal_within_generated_set']==(best is not None and not cheaper)


def test_validator_accepts_saved_ld7_route_outputs():
    folder=ROOT/X.PDIR/'q3/LD7_A_SHAPE';result=json.loads((folder/'RESULT.json').read_text())
    saved=X.read(X.SAVED_AB_INPUTS['commit'],X.SAVED_AB_INPUTS['path'])['inputs']
    case=next(c for c in saved['cases'] if c['label']=='LD7_A_SHAPE')
    out=X.validate_result(case,saved,folder,result['source'])
    assert out['status']==X.PASS and out['closure']['status']=='VERIFIED_FINITE_BINARY64_SUPPORT'
    assert out['MSE_upper']==result['record']['diagnostic']['deterministic_risk']['MSE_upper']


def test_actual_execute_entry_is_sequential_one_case_and_stops(tmp_path):
    cases=[dict(label=f'TOY{i}',reference_cost=dict(numerator=i+1,denominator=1)) for i in range(4)]
    inputs=dict(cases=cases,maximum_scientific_calls=4,evaluation_order=[c['label'] for c in cases],saved_catalog=[],runtime=dict(toy=True))
    order=iter([X.FAIL,X.PASS,X.FAIL,X.FAIL]);calls=[]
    def supervise(cs,x,source,step):
        assert len(cs)==1 and source['path']==X.CALLER
        calls.append(cs[0]['label']);return dict(receipts=[{}],errors=[],monitor_errors=[],process_exitcodes=[dict(exitcode=0)])
    def validated(case,x,folder,source):
        return dict(status=next(order),scientific_evaluation_calls=1,MSE_lower=0,MSE_upper=0,result_source='SYNTHETIC_ONLY')
    (tmp_path/X.DIRECTORY).mkdir(parents=True);(tmp_path/X.INPUT).write_text(json.dumps(inputs))
    (tmp_path/X.CALLER).parent.mkdir(parents=True);(tmp_path/X.CALLER).write_text('TOY')
    with patch.object(X,'ROOT',tmp_path),patch.object(X,'check_source'),patch.object(X,'build_inputs',return_value=inputs),\
            patch.object(X.W,'head',return_value='TOY_SOURCE'),patch.object(X.W,'blob',side_effect=lambda sha,path:(tmp_path/path).read_bytes()),\
            patch.object(X.W,'supervise',side_effect=supervise),patch.object(X,'validate_result',side_effect=validated):
        assert X.execute()==0
    assert calls==['TOY0','TOY1']
    run=tmp_path/X.DIRECTORY/'run01';stop=json.loads((run/'STOP.json').read_text())
    assert stop['stop_reason']=='FIRST_FEASIBLE_IN_COST_ORDER' and stop['new_scientific_calls']==2
    assert (run/'01/OUTCOME.json').exists() and (run/'02/OUTCOME.json').exists() and not (run/'03').exists()


def test_summary_keeps_saved_ld7_minimum_without_pass_and_compares_pass(inputs,tmp_path):
    for history,best in (([dict(label=c['label'],status=X.FAIL,scientific_evaluation_calls=1,MSE_lower=dict(numerator=1,denominator=10**5),
            MSE_upper=dict(numerator=1,denominator=10**5)) for c in inputs['cases']],None),
            ([dict(label=inputs['cases'][0]['label'],status=X.FAIL,scientific_evaluation_calls=1,MSE_lower=None,MSE_upper=None),
              dict(label=inputs['cases'][1]['label'],status=X.PASS,scientific_evaluation_calls=1,MSE_lower=dict(numerator=1,denominator=10**7),
              MSE_upper=dict(numerator=1,denominator=10**7))],inputs['cases'][1]['label'])):
        d=tmp_path/('s'+str(best));p=d/X.DIRECTORY;(p/'run01').mkdir(parents=True)
        (p/'INPUTS.json').write_text(json.dumps(inputs));(p/'PROCESS_RETURN.json').write_text(json.dumps(dict(returncode=0)))
        (p/'run01/EXECUTION.json').write_text(json.dumps(dict(source=dict(commit='TOY'))))
        outcomes={h['label']:{k:v for k,v in h.items() if k!='label'} for h in history}
        r=X.search(dict(cases=inputs['cases'],maximum_scientific_calls=4),lambda c,s:outcomes[c['label']])
        (p/'run01/SEARCH_HISTORY.json').write_text(json.dumps(r))
        with patch.object(X,'ROOT',d),patch('builtins.print'):X.summarize()
        join=json.loads((p/'C_RESULT_JOIN.json').read_text());summary=json.loads((p/'SUMMARY.json').read_text())
        assert len(join['rows'])==4 and join['best_confirmed']==best
        if best is None:
            assert X.rational(join['LD7_confirmed_cheapest']['cost'])==Fraction(155652932221,126)
            assert join['LD7_confirmed_cheapest']['updated'] is False
        else:
            row=next(x for x in summary['rows'] if x['label']==best)
            assert X.rational(join['LD7_confirmed_cheapest']['cost'])==X.rational(row['conditional_cost'])
            assert [c['benchmark'] for c in row['comparisons']]==['P7_S2','FULL_PF_S2']
            assert summary['rows'][2]['status']=='NOT_EVALUATED_EARLY_STOP'

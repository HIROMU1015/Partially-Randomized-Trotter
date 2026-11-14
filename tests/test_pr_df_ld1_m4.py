"""m4-only technical checks: literal seeds, synthetic sampler, fake compiler, exact errors."""
import json
from fractions import Fraction
from pathlib import Path
import pytest
from compiled_price_artificial import fixture
from test_pr_df_ld1_m3 import toy
import run_pr_df_ld1_m4 as H


def predictions():
    return {axis:dict(predicted_RZ=H.encode(Fraction(9,2)),source=dict(commit='FROZEN',path='MODEL',selector=[axis])) for axis in ('X','Y')}


def rows():
    return [dict(axis=a,index=i,C=(i+1 if a=='X' else i+2),valid=True) for a in ('X','Y') for i in range(4)]


def test_preopen_copies_original_literal_seed_table_and_two_frozen_predictions():
    plan=H.make_inputs();prior=H.read_ref(H.PRIOR);frozen=H.read_ref(H.FROZEN)
    assert plan['cells'][0]['m']==4 and plan['cells'][0]['role']=='HOLDOUT'
    assert plan['cells'][0]['seeds']==prior['future_holdout']['cell']['seeds']
    assert plan['menus']['H4']['candidates']==[prior['future_holdout']['row']]
    assert plan['comparison_rule']==prior['model_plan']['holdout_rule']
    for axis in ('X','Y'):
        assert plan['predictions'][axis]['predicted_RZ']==frozen['models'][axis]['m4_prediction']['predicted_RZ']
        assert plan['predictions'][axis]['source']['selector']==['models',axis,'m4_prediction']
    assert plan['compile_budget']==8 and plan['public']!=prior['public'] and plan['private']!=prior['private']


def test_actual_original_sampler_consumes_literal_saved_seeds(monkeypatch):
    science,ctx,c,cell,plan,_=fixture(q=2,R=3)
    def require(value,reason):
        if not value:raise ValueError(reason)
    def forbidden(*a,**k):raise AssertionError('Seed regeneration / molecular input forbidden')
    ctx.h['require']=require;ctx.h['trial_seed']=forbidden
    science.functions(H.ROOT,H.M.P.CARRIER,'scripts/h46_pr_pilot_repair.py',['canonical_tau'],ctx.h)
    science.functions(H.ROOT,H.M.P.CARRIER,'scripts/h46_long_time_native.py',['sample_requests'],ctx.h,('h46_m5_contract','h46_pr_pilot_repair'))
    cell.update(path_indices=[0,1,2,3],seeds=[[11,12],[21,22],[31,32],[41,42]])
    plan['cells']=[cell]
    import numpy as np
    monkeypatch.setattr(np,'load',forbidden)
    calls=H.install_fixed_seed_sampler(ctx,plan)
    requests,seeds,laws=ctx.h['sample_requests'](c,ctx.prep,cell['cohort'],2)
    assert seeds==[31,32] and [r.seed for r in requests]==[31,32]
    assert calls==[(2,0,0,31),(2,1,0,32)]
    assert len(laws)==2 and all(len(r.rte_occurrence.events)==3 for r in requests)
    with pytest.raises(ValueError,match='identity/order'):
        ctx.h['sample_requests'](c,ctx.prep,'WRONG_COHORT',2)
    assert len(calls)==2


def test_m4_uses_actual_sc_worker_and_separate_m4_record(toy):
    state,draws,compiles=toy;state['plan']['cells'][0].update(m=4,role='HOLDOUT')
    result=H.M.run_path(0,state)
    assert len(draws)==1 and len(compiles)==2 and result['m']==4 and result['role']=='HOLDOUT'
    d=Path(result['private_directory'])
    assert (d/'M4_PATH.json').is_file() and not (d/'M3_PATH.json').exists()
    assert json.loads((d/'M4_PATH.json').read_text())['C']==[101,102]
    assert result['event_digest']==json.loads((d/'STRUCTURAL.json').read_text())['event_digest']
    assert result['full_wrapper_statevector_validation'] is False


def test_exact_comparison_uses_only_axis_json_P_and_four_C():
    ps=predictions();ps['X'].update(ideal_structural={'extra':999999},residual_prediction={'extra':888888})
    v=H.compare(ps,list(reversed(rows())));x,y=v['units']
    assert x['O']==H.encode(Fraction(5,2)) and y['O']==H.encode(Fraction(7,2))
    assert x['signed_error']==H.encode(Fraction(2)) and y['signed_error']==H.encode(Fraction(1))
    assert x['signed_relative_error']==H.encode(Fraction(4,5))
    assert v['campaign']=='DESCRIPTIVE_COMPLETE' and v['model_verdict']==H.VERDICT and v['tolerance'] is None
    ps['Y']['predicted_RZ']=H.encode(Fraction(1,3));v=H.compare(ps,rows());y=v['units'][1]
    assert y['signed_error']==H.encode(Fraction(-19,6))
    assert y['absolute_error']==H.encode(Fraction(19,6)) and y['signed_relative_error']==H.encode(Fraction(-19,21))


def test_zero_observed_has_only_relative_errors_null():
    rs=rows()
    for r in rs:
        if r['axis']=='X':r['C']=0
    v=H.compare(predictions(),rs);x=v['units'][0]
    assert v['campaign']=='DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED' and x['valid']
    assert x['O']==H.encode(Fraction(0)) and x['signed_error']==H.encode(Fraction(9,2))
    assert x['signed_relative_error'] is None and x['absolute_relative_error'] is None
    assert x['reasons']==['ZERO_OBSERVED_PRICE']


@pytest.mark.parametrize('change',['missing','invalid','duplicate'])
def test_missing_or_invalid_axis_never_changes_n_or_pools(change):
    rs=rows()
    if change=='missing':rs.pop(0)
    elif change=='invalid':rs[0].update(valid=False,reason='GATE_FAILED')
    else:rs[0]['index']=1
    v=H.compare(predictions(),rs);x,y=v['units']
    assert v['campaign']=='INCOMPLETE_OR_INVALID' and not x['valid'] and y['valid']
    assert x['n_paths']==4 and x['O'] is None and x['signed_error'] is None
    assert x['absolute_error'] is None and x['absolute_relative_error'] is None
    assert y['O']==H.encode(Fraction(7,2)) and v['model_verdict']==H.VERDICT

"""Artificial scalar data only; no real H4 values read or reanalyzed."""
import copy
from fractions import Fraction as F
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_h4_p3_stage4_accounting as runner
K=runner.modules()


def toy_stage2():
    return dict(signals={f'nom_{arm}':dict(real=.2,imag=.4,u_num=.01,u_bind=.02,u_ref_transfer=.03)
                         for arm in ('P','DF')},
        reference=dict(physical_interval={a:dict(lower='1/10',upper='3/10') for a in ('real','imag')},
                       transfer_uncertainty=.04),
        normalization={arm:dict(B=1.,B_uncertainty=.01,decimal50=dict(B='1.005')) for arm in ('P','DF')})


def toy_prices(n=4):
    prices={};summary=dict(arms={},old_pilot_pooled=False,cross_arm_paired=False)
    for arm,offset in [('P',0),('DF',10)]:
        summary['arms'][arm]=dict(n=n,axes={a:{} for a in K.AXES},cosine_sine_sample_covariance_exact={})
        means,cols={},{}
        for axis in K.AXES:
            col=[i+1+offset if axis=='cosine' else n-i+offset for i in range(n)]
            cols[axis]=col;mean=F(sum(col),n);means[axis]=mean
            var=sum((F(v)-mean)**2 for v in col)/(n-1)
            for metric in K.METRICS:
                summary['arms'][arm]['axes'][axis][metric]=dict(mean_exact=str(mean),unbiased_sample_variance=float(var))
            for i,value in enumerate(col):prices[f'{arm}_{i:02d}/F/{axis}']=dict.fromkeys(K.METRICS,value)
        cv=sum((F(x)-means['cosine'])*(F(y)-means['sine']) for x,y in zip(cols['cosine'],cols['sine']))/(n-1)
        summary['arms'][arm]['cosine_sine_sample_covariance_exact']=dict.fromkeys(K.METRICS,str(cv))
    return prices,summary


def test_direct_bias_counts_reference_and_each_uncertainty_once():
    saved=toy_stage2();bias=K.bias_inputs(saved,'P')['axes']['cosine']
    z=F(.2);distance=max(abs(z-F(1,10)),abs(z-F(3,10)))
    assert F(bias['bias_upper_exact'])==distance+F(.01)+F(.02)+F(.03)+F(.04)
    mutated=copy.deepcopy(saved);mutated['signals']['split']=dict(u_num=1e8)
    mutated['reference']['display']=dict(axis_resolution=dict(real=1000,imag=1000))
    assert K.bias_inputs(mutated,'P')==K.bias_inputs(saved,'P')


def test_missing_and_negative_numerical_evidence():
    saved=toy_stage2();saved['signals']['nom_P']['u_num']=None
    assert K.bias_inputs(saved,'P')['axes'] is None
    saved['signals']['nom_P']['u_num']=-1
    with pytest.raises(ValueError,match='NEGATIVE'):K.bias_inputs(saved,'P')


def test_normalization_uses_engineering_max_and_keeps_exact_anchor():
    value=K.normalization_input(dict(B=1.,B_uncertainty=.1,decimal50=dict(B='1.2')))
    assert F(value['B_engineering_upper_exact'])==F(6,5)
    assert F(value['B_scalar_upper_exact'])>=F(6,5)
    assert value['nominal_signal_rescaled'] is False and value['normalization_certificate'] is False
    assert K.normalization_input(dict(B=1.,B_uncertainty=0.,decimal50=dict(B='1')))['log_B_upper_operand']==0
    assert K.normalization_input(dict(B=None))['status'].startswith('MISSING')


def test_outward_binary64_and_complex_precision_allocation():
    for value in [F(1,3),F(7,10),F(1,10**300)]:
        assert F(K.outward(value,upper=True))>=value
        assert F(K.outward(value,upper=False))<=value
    for epsilon in K.EPSILONS: assert 2*F(K.epsilon_axis(epsilon))**2<=F(epsilon)**2
    with pytest.raises(ValueError):K.outward(float('nan'),upper=True)


def test_negative_axis_covariance_and_direct_weighted_sample_variance():
    prices,summary=toy_prices();moments=K.price_moments(prices,summary,n=4)
    cost=K.weighted_cost(prices,'P','RZ',moments['P']['RZ'],dict(cosine=3,sine=3))
    assert F(cost['G_exact'])==15 and cost['price_standard_error']==0
    cost=K.weighted_cost(prices,'P','RZ',moments['P']['RZ'],dict(cosine=1,sine=3))
    assert F(cost['G_exact'])==10
    assert F(cost['variance_of_mean_exact'])==F(5,3)
    bad=copy.deepcopy(summary);bad['cross_arm_paired']=True
    with pytest.raises(ValueError,match='INDEPENDENT'):K.price_moments(prices,bad,n=4)
    bad=copy.deepcopy(summary);bad['arms']['P']['cosine_sine_sample_covariance_exact']['RZ']='0'
    with pytest.raises(ValueError,match='COVARIANCE'):K.price_moments(prices,bad,n=4)


def test_shots_sufficient_ceil_legacy_reuse_and_domain_null():
    prices,summary=toy_prices();moments=K.price_moments(prices,summary,n=4)['P']
    saved=toy_stage2();bias=K.bias_inputs(saved,'P');norm=K.normalization_input(saved['normalization']['P'])
    row=K.account_row('P','1',bias,norm,moments,prices)
    assert row['status']=='ELIGIBLE_CONDITIONAL' and row['legacy']['evidence_kind']=='EMPIRICAL'
    for a,value in row['axes'].items():
        assert value['shots']>=value['legacy_shots'] and value['sufficient_scalar_inequality_verified']
        assert value['shots']>=F(value['scalar_bound_upper_exact'])
    assert not row['numerical_certificate']
    small=K.account_row('P','0.01',bias,norm,moments,prices)
    assert small['N_total'] is None and small['costs']['RZ'] is None
    tiny_bias=copy.deepcopy(bias)
    for value in tiny_bias['axes'].values():value['bias_upper_operand']=0.
    limit=K.account_row('P','0.000000000001',tiny_bias,norm,moments,prices)
    assert limit['N_total'] is None and all(r['status']=='UNRESOLVED_SAFE_INTEGER_LIMIT' for r in limit['axes'].values())


def test_ten_rows_comparisons_and_missing_field_preserves_nulls():
    prices,summary=toy_prices(32);saved=toy_stage2()
    # Set a zero artificial nominal bias for the fixed precision list.
    for arm in ('P','DF'):
        saved['signals']['nom_'+arm]=dict(real=0.,imag=0.,u_num=0.,u_bind=0.,u_ref_transfer=0.)
    saved['reference']=dict(physical_interval={f:dict(lower='0',upper='0') for f in ('real','imag')},transfer_uncertainty=0.)
    result=K.evaluate(saved,prices,summary)
    assert len(result['rows'])==10 and all(r['status']=='ELIGIBLE_CONDITIONAL' for r in result['rows'])
    for comparison in result['comparisons']:
        value=comparison['metrics']['RZ']
        assert value['P_minus_DF']<0 and value['P_over_DF']<1 and not value['cross_arm_paired']
    assert result['population_tail_bound'] is None and result['compiled_rho_measured'] is False
    saved['signals']['nom_P']['u_bind']=None
    result=K.evaluate(saved,prices,summary)
    assert all(r['N_total'] is None for r in result['rows'] if r['arm']=='P')
    assert all(r['metrics']['RZ'] is None for r in result['comparisons'])

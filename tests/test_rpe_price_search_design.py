"""Artificial checks of price-search mathematics and existing B interfaces.

No molecular records, feature evaluation, circuits, providers, fits or science.
The interval expressions below test the design; they are not a price solver.
"""
from copy import deepcopy
from fractions import Fraction as F

import pytest
from check_pr_rpe_rmse_controller import spec, data
from trottertracks import pr_rpe_rmse_controller as B
from trottertracks import bv2_mean_price as C


def test_nonnegative_prices_allow_partial_lower_bound_without_missing_upper():
    known_lower = sum(n*lo for n,lo in [(3,F(5)), (4,F(7))])
    incumbent_upper = F(40)
    assert known_lower > incumbent_upper
    # For ANY missing nonnegative mean, the losing schedule stays more costly.
    for missing in (F(0),F(1),F(10**100)):
        assert known_lower + 11*missing > incumbent_upper
    # S is not that lower bound: a hypothetical S=100, C=1 is permitted.
    assert F(100) > incumbent_upper and F(1) < incumbent_upper


def test_interval_propagation_and_difference_hold_without_independence():
    bounds = [(F(2),F(5)),(F(3),F(7))]
    nx, ny = [4,1], [1,3]
    coefficients = [x-y for x,y in zip(nx,ny)]
    lower = sum(v*(lo if v>=0 else hi) for v,(lo,hi) in zip(coefficients,bounds))
    upper = sum(v*(hi if v>=0 else lo) for v,(lo,hi) in zip(coefficients,bounds))
    # Include perfectly dependent and antidependent admissible prices.
    for t in [F(0),F(1,3),F(1)]:
        for u in [t,1-t]:
            means = [2+3*t,3+4*u]
            delta = sum(v*mu for v,mu in zip(coefficients,means))
            assert lower <= delta <= upper


def test_only_exact_shared_quantity_can_cancel_an_unknown_price():
    assert 7-7 == 0
    # Different law/lowering is a different random variable, even with same N.
    assert 7*F(1)-7*F(10) != 0
    assert 7*F(10)-7*F(1) != 0


def test_regret_certificate_is_weaker_than_exact_dominance():
    bounds={'incumbent':(10,12),'challenger':(11,20)}
    tau=1
    assert bounds['incumbent'][1]-min(lo for lo,hi in bounds.values()) == 2
    assert not bounds['incumbent'][1] < bounds['challenger'][0]
    assert not bounds['incumbent'][1]-min(lo for lo,hi in bounds.values()) <= tau
    tau=2
    assert bounds['incumbent'][1]-min(lo for lo,hi in bounds.values()) <= tau
    # A valid tolerance certificate still allows the challenger to be cheaper.
    assert F(11) < F(12)


def test_N_weighted_precision_can_reverse_unweighted_priority():
    # Earlier round is more influential here; m alone is not the priority.
    n=[100,1];width=[F(1),F(10)]
    assert width[0] < width[1]
    assert n[0]*width[0] > n[1]*width[1]
    # Equal-price common cells cancel in differences only after weighting.
    assert 100-1 != 0


def test_unknown_N_requires_valid_set_instead_of_nominal_weight():
    price=(F(2),F(4));n_box=(3,9)
    domain=(n_box[0]*price[0],n_box[1]*price[1])
    for n in range(3,10):
        for c in price:
            assert domain[0] <= n*c <= domain[1]
    assert 1*price[1] < domain[0]  # arbitrary nominal N is not a bound


def test_huge_shot_counts_and_one_gate_gap_use_existing_exact_arithmetic():
    n=2**1200+3
    lo=B.a.exact_scalar({'numerator':1,'denominator':3})
    upper=n*lo
    lower=upper+1
    assert B.a.exact_scalar(B.a.exact_value(lower))-upper == 1
    assert B.a.exact_scalar(B.a.exact_value(n)) == n


def test_existing_V2_rejects_other_split_and_foreign_lowering_records():
    original=B.generate(spec(L_D=1));packet,prices,cohorts=data(original)
    s=spec(L_D=2);changed=B.generate(s)
    out=B.assess(changed,packet,prices,cohorts=cohorts)
    assert not out['eligible'] and out['cost'] is None
    bad=deepcopy(prices);bad[0]['identity']['profile']['lowering_id']='FOREIGN'
    out=B.assess(original,packet,bad,cohorts=cohorts)
    assert 'C_IDENTITY_LAW_OR_SOURCE_MISMATCH' in out['reasons']
    bad=deepcopy(prices);bad[0]['quantity']='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT'
    out=B.assess(original,packet,bad,cohorts=cohorts)
    assert 'C_STRUCTURAL_OR_LOCAL_QUANTITY_IS_NOT_MEAN_COMPILED_PRICE' in out['reasons']


def test_existing_assess_does_not_need_point_cost_to_expose_missing_cells():
    c=B.generate(spec());packet,prices,cohorts=data(c)
    out=B.assess(c,packet,prices[:1],cohorts=cohorts)
    assert not out['eligible'] and out['cost'] is None
    assert 'MISSING_OR_AMBIGUOUS_SELECTED_C' in out['reasons']
    assert any(cell['C']['qualification']!='AVAILABLE' for cell in out['coverage'])


def test_existing_lookup_does_not_relabel_structural_quantity_as_compiled_mean():
    c=B.generate(spec());_,prices,_=data(c)
    prices[0]['quantity']='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT'
    integration=type('Integration',(),{'B':B})
    with pytest.raises(ValueError,match='STRUCTURAL_OR_LOCAL'):
        C.V2MeanPriceLookup(c,prices,integration=integration)


def test_finite_multiple_looks_can_be_budgeted_without_zeroing_selection_error():
    alpha=F(1,20)
    budgets=[alpha/F(t*(t+1)) for t in range(1,9)]
    assert sum(budgets)==alpha*(1-F(1,9)) < alpha
    # Eight nominal 95% looks do not have a 5% family failure bound.
    assert sum([alpha]*8)>alpha


def test_paired_axes_need_covariance_and_rich_control_variate_need_not_help():
    nx,ny=3,4;vx,vy=F(4),F(9);cov=F(6)
    joint=nx*nx*vx+ny*ny*vy+2*nx*ny*cov
    assert joint==(nx*2+ny*3)**2
    assert joint>nx*nx*vx+ny*ny*vy
    # Same-unit E[S] is known but C-S may have MORE variance than C.
    c=[F(10),F(12)];s=[F(1),F(20)]
    assert abs((c[1]-s[1])-(c[0]-s[0]))>abs(c[1]-c[0])

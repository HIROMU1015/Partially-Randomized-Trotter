"""Invented scalar laws/budgets only; molecular saved values are never read."""
from copy import deepcopy
from fractions import Fraction
import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from test_o1_m1_matched_split import PACKAGE

screen = importlib.import_module(PACKAGE + '.o1_m1_cost_feasibility')
bridge = importlib.import_module(PACKAGE + '.o1_m1_resource_bridge')
rte = importlib.import_module(PACKAGE + '.rte')
accounting = importlib.import_module(PACKAGE + '.rpe_resource_accounting')


def forbidden(*args, **kwargs):
    raise AssertionError('No molecular I/O, build, samples or prices in preparation')


@pytest.fixture(autouse=True)
def no_saved_values_or_circuits(monkeypatch):
    import qiskit
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(qiskit, 'QuantumCircuit', forbidden)
    for name in ('prepare_matched_resources', 'build_controlled_trajectory',
                 'MatchedNoPrepCostProvider', 'transpile_and_measure_cost'):
        monkeypatch.setattr(bridge, name, forbidden)
    monkeypatch.setattr(rte, 'iter_sample_rte_events', forbidden)
    monkeypatch.setattr(rte, 'sample_rte_events', forbidden)


def invented(*, r=4, radius=.8):
    tail = SimpleNamespace(tail_id='invented-tail', tail_hash='invented-hash',
        lambda_r=.2, components=(rte.RTEComponent('invented', 1., .2, 1),))
    config, _ = rte.make_rte_config(tail, evolution_time=.05, rte_steps=r,
        finite_taylor_order=2, truncation_tolerance=1.)
    specification = accounting.RPERoundSpecification(6, .05)
    attenuation = rte.finite_rte_attenuation(config, tail_evolutions=64)
    bounds = dict(reference_radius_lower_bound=radius, corrected_signal_error_bound=.001,
        pf_phase_bound=.001, model_phase_bound=.001, implementation_phase_bound=.001,
        radius_provenance='invented scalar lower bound, not molecular evidence')
    arm = dict(feasible=True, reasons=[], bounds=bounds, attenuation=attenuation,
        normalization_product=1/attenuation,
        rho_observed_lower_bound=attenuation*(radius-.001))
    saved = dict(q=64, delta=.05, r=r, K=2, arms={key:deepcopy(arm) for key in ('P','DF')})
    return dict(saved_row=saved, dictionary='P', config=config, specification=specification,
        context_hash='invented-physical-context', arm_hash='invented-PR-arm')


def baseline(account, *, total=1000.):
    context = {key:account[key] for key in ('context_hash','backend_hash','cost_metric',
               'cost_scope','delta_time','beta_rpe')}
    context.update(arm_hash='strong_DF_PF_4_substeps_spin_fused', compiler_hash='invented-compiler')
    return dict(scope='registered_rounds_only',round_count=7,maximum_round=6,
        alpha_bound=.05,total_cost=total,context=context,guarantee_status='not_certified',
        price_uncertainty_qualified=False)


def ceiling(account, benchmark=None):
    return screen.optimistic_unit_price_ceiling(account, baseline(account) if benchmark is None else benchmark,
        comparison_compiler_hash='invented-compiler')


@pytest.mark.parametrize('r,dictionary',[(4,'P'),(4,'DF'),(8,'P'),(8,'DF')])
def test_actual_radius_accounting_preserves_fourteen_axis_budget_and_has_no_price(r,dictionary):
    options=invented(r=r);options['dictionary']=dictionary
    result=screen.account_saved_c0_signal(**options)
    rho=options['saved_row']['arms'][dictionary]['rho_observed_lower_bound']
    expected=math.ceil(4*math.log(2/(.05/14))/(rho*rho*math.sin(.15)**2))
    assert result['shots']==dict(cosine=expected,sine=expected)
    assert result['alpha_axis']==.05/14
    assert result['alpha_bound']==2*(.05/14)
    assert result['round_total_cost'] is None and result['compiler_hash'] is None
    assert result['full_rpe_improvement'] is None and result['guarantee_status']=='not_certified'


def test_smaller_unattenuated_radius_increases_shots_without_unit_radius_assumption():
    high=screen.account_saved_c0_signal(**invented(radius=.8))
    low=screen.account_saved_c0_signal(**invented(radius=.5))
    assert low['shots']['cosine']>high['shots']['cosine']
    assert low['reference_radius_lower_bound']==.5


def test_pr_config_none_is_rejected_instead_of_unit_attenuation():
    options=invented();options['config']=None
    with pytest.raises(ValueError,match='finite-RTE'):
        screen.account_saved_c0_signal(**options)


@pytest.mark.parametrize('field,value',[('q',32),('r',2),('K',0),('delta',.1)])
def test_a_saved_q64_record_cannot_fill_another_round_or_law(field,value):
    options=invented();options['saved_row'][field]=value
    with pytest.raises(ValueError,match='no transfer'):
        screen.account_saved_c0_signal(**options)


@pytest.mark.parametrize('field',['attenuation','normalization_product','rho_observed_lower_bound'])
def test_saved_normalization_and_radius_must_match_the_supplied_law(field):
    options=invented();options['saved_row']['arms']['P'][field]*=1.01
    with pytest.raises(ValueError,match='differs'):
        screen.account_saved_c0_signal(**options)


@pytest.mark.parametrize('field,value',[('model_phase_bound',.031),('implementation_phase_bound',.011)])
def test_individual_model_and_implementation_budgets_are_not_silently_reassigned(field,value):
    options=invented();options['saved_row']['arms']['P']['bounds'][field]=value
    with pytest.raises(ValueError,match='exceeded'):
        screen.account_saved_c0_signal(**options)


def test_unqualified_saved_signal_is_rejected_before_accounting():
    options=invented();options['saved_row']['arms']['P']['feasible']=False
    with pytest.raises(ValueError,match='qualified saved'):
        screen.account_saved_c0_signal(**options)


def test_ceiling_is_shot_weighted_and_strict_not_a_full_cost_advantage():
    account=screen.account_saved_c0_signal(**invented())
    account['shots']=dict(cosine=3,sine=7)
    result=ceiling(account,baseline(account,total=101.))
    threshold=Fraction(**result['strict_unit_price_ceiling'])
    assert threshold==Fraction(101,10)
    # These invented axis prices pass this round's necessary condition, yet
    # other rounds can still make the complete candidate more expensive.
    this_round=3*1+7*2
    assert Fraction(this_round,10)<threshold
    assert this_round+1000>101
    assert result['full_candidate_total_cost'] is None
    assert result['full_rpe_improvement'] is None
    assert result['round_price_lower_bound'] is None
    assert result['status']=='NECESSARY_UNIT_PRICE_CONDITION_ONLY'


def test_exact_fraction_ceiling_does_not_underflow_with_large_shot_counts():
    account=screen.account_saved_c0_signal(**invented())
    account['shots']=dict(cosine=10**400,sine=10**400)
    result=ceiling(account)
    assert Fraction(**result['strict_unit_price_ceiling'])==Fraction(1000,2*10**400)


@pytest.mark.parametrize('field,value',[('context_hash','different-state'),('cost_scope','prep_included'),
    ('delta_time',.1),('beta_rpe',.4),('backend_hash','different-backend'),('cost_metric','rz_count'),
    ('compiler_hash','different-compiler')])
def test_baseline_context_or_full_no_prep_compiler_drift_is_rejected(field,value):
    account=screen.account_saved_c0_signal(**invented());benchmark=baseline(account)
    benchmark['context'][field]=value
    with pytest.raises(ValueError):ceiling(account,benchmark)


def test_partial_baseline_and_reallocated_three_round_alpha_are_rejected():
    account=screen.account_saved_c0_signal(**invented());benchmark=baseline(account)
    benchmark['round_count']=3
    with pytest.raises(ValueError,match='complete seven-round'):ceiling(account,benchmark)
    account['alpha_axis']=.05/6
    with pytest.raises(ValueError,match='seven-round allocation'):ceiling(account)


def test_pr_aggregate_is_not_accepted_as_the_strong_baseline_budget():
    account=screen.account_saved_c0_signal(**invented());benchmark=baseline(account)
    benchmark['context']['arm_hash']=account['arm_hash']
    with pytest.raises(ValueError,match='strong DF-PF baseline'):ceiling(account,benchmark)


def test_a_price_is_never_implied_by_an_old_priced_record():
    account=screen.account_saved_c0_signal(**invented());account['round_total_cost']=10.
    with pytest.raises(ValueError,match='unpriced'):ceiling(account)


@pytest.mark.parametrize('total',[0.,-1.,float('nan'),float('inf')])
def test_unknown_or_invalid_baseline_budget_is_not_zero_or_a_threshold(total):
    account=screen.account_saved_c0_signal(**invented())
    with pytest.raises(ValueError):ceiling(account,baseline(account,total=total))

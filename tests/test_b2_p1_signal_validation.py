"""Artificial matrices only. No molecular data, saved state, shots or prices."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.linalg import expm

from trotterlib import b2_p1_signal_validation as p
from trotterlib import b2_rpe_connection as c
from trotterlib import o1_m1_resource_bridge as accounting
from trotterlib import pr2_matched_accuracy_m1_execution as legacy
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.rte import CompilerSettings, finite_rte_operator_moments
from trotterlib.rpe_resource_accounting import RPERoundSpecification


@pytest.fixture(autouse=True)
def no_scientific_IO_or_shots(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved inputs and shot/price/solve calls forbidden in artificial tests')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(accounting,'_hoeffding_shots',forbidden)
    monkeypatch.setattr(legacy,'_axis_record',forbidden)
    import trotterlib.df_hamiltonian as h
    monkeypatch.setattr(h,'solve_df_ground_state',forbidden)


def case():
    ham=DFHamiltonian(.17,np.array([[.2,.1j],[-.1j,-.3]]),
        np.array([.4,-.2,.3,.15,-.11]),tuple(np.array(g,complex) for g in (
            [[1,.1],[.1,.4]],[[.3,.2j],[-.2j,.7]],[[.1,.13],[.13,.9]],
            [[.8,.2j],[-.2j,.1]],[[.2,.21],[.21,.6]])),{'name':'ARTIFICIAL_P1'})
    prep=c.prepare_b2(ham,L_D=3)
    from test_o1_m1_matched_split import annihilators,lifted_one
    a=annihilators(2)
    one=lifted_one(ham.one_body,a)
    fragments=tuple(w*lifted_one(g,a)@lifted_one(g,a) for w,g in zip(ham.lambdas,ham.g_matrices))
    tail=sum(fragments[3:],np.zeros_like(one))-prep.extracted_identity_coefficient*np.eye(4)
    det=legacy._preparation_eigensystems(prep,one,fragments,{})
    eig=legacy._eigendecomposition(tail/prep.exact_rte_lambda_r)
    config,dist,kernel,steps=p.legacy_mean_steps(prep,det,eig)
    ctx=c.RPEConnectionContext('toy_phys',prep.hamiltonian_hash,'toy_original_state','toy_sector',
        'toy_order','toy_SOURCE',CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0'),
        'no_backend','rz_count',None,.01,.95,'toy prior; no guarantee')
    round_=c.B2RPERound(ctx,prep,RPERoundSpecification(0,p.DELTA),3,p.R,p.K,p.DELTA,1,config,dist)
    return prep,one,fragments,tail,config,kernel,steps,round_


def test_extracted_mean_only_kernel_matches_independent_polynomial_and_raw():
    prep,one,fragments,tail,config,kernel,steps,_=case()
    halves=[expm(-.5j*p.DELTA*b) for b in (one,*fragments[:3])]
    moments=finite_rte_operator_moments(tail/prep.exact_rte_lambda_r,config)
    from trotterlib.pauli_signal_adapter import paired_polynomial_action
    independent=paired_polynomial_action(tail,np.eye(4),p.DELTA,p.R,p.K)
    np.testing.assert_allclose(moments.corrected_operator,independent,rtol=0,atol=3e-14)
    for key,op in (('corrected',independent),('raw',moments.attenuated_event_mean_operator),
                   ('ideal_tail',expm(-1j*p.DELTA*tail))):
        np.testing.assert_allclose(steps[key],p.compose_step(halves,op,kernel['phase']),rtol=0,atol=3e-14)


def record(round_, *, pf_phase=0., model_phase=0., finite_error=1e-7, impl=1e-8):
    original=np.exp(.1j)
    df=original*np.exp(1j*model_phase)
    pf=.9*df*np.exp(1j*pf_phase)
    finite=pf*(1-finite_error)
    A=c.finite_rte_attenuation(round_.config,tail_evolutions=round_.q)
    vals=[c.SignalValue(z,round_.context.original_state,
        round_.context.original_hamiltonian if i==0 else round_.context.df_hamiltonian)
        for i,z in enumerate((original,df,pf,finite,A*finite))]
    return c.B2SignalRecord(round_.binding,*vals,1/A,impl)


def test_signal_only_qualification_never_calls_shots_and_keeps_model_none():
    *_,round_=case()
    result=p.qualify_signals(round_,record(round_))
    assert result['feasible'] and result['guarantee_status']=='not_certified'
    assert round_.context.model_error_bound is None
    assert result['shot_accounting_called'] is False


@pytest.mark.parametrize('kwargs,reason',[
    ({'pf_phase':.061},'PF_phase_budget'),
    ({'model_phase':.021},'model_phase_cap'),
    ({'finite_error':.05},'RTE_phase_budget'),
    ({'impl':.021},'implementation_phase_cap')])
def test_component_caps_fail_without_budget_reallocation(kwargs,reason):
    *_,round_=case()
    result=p.qualify_signals(round_,record(round_,**kwargs))
    assert not result['feasible'] and reason in result['reasons']


def test_original_state_substitution_and_double_normalization_rejected():
    *_,round_=case()
    good=record(round_)
    bad=replace(good,df=replace(good.df,state_identity='DF_ground_state'))
    with pytest.raises(ValueError,match='state mismatch'):
        p.qualify_signals(round_,bad)
    with pytest.raises(ValueError,match='normalization'):
        p.qualify_signals(round_,replace(good,normalization_product=good.normalization_product*2))


def test_private_hash_failure_before_array_loading(tmp_path):
    for name in ('state.npz','provenance.json','H4_sto-3g_singlet_df_d100.hdf5'):
        (tmp_path/name).write_bytes(b'ARTIFICIAL_WRONG_FILE')
    with pytest.raises(ValueError,match='STATE_FILE_SHA256'):
        p.load_private_inputs(tmp_path)


def test_selected_native_probe_has_prefix_and_signed_event_relative_phase():
    prep,one,fragments,_,_,_,_,round_=case()
    psi=np.array([.3,.4j,-.2,.5j]);psi/=np.linalg.norm(psi)
    report,_,_=p.controlled_checks(dict(state=psi,fragments=fragments),prep,round_,[one,*fragments[:3]])
    assert report['status']=='PASS'
    assert report['outer_step_cases']==2
    assert any(e['taylor_order']==2 and e['phase_real']==-1 for e in report['representative_events'])


def test_fixed_scope_configuration():
    assert p.DELTA==.02 and (p.R,p.K,p.LD)==(4,2,3)
    assert math.isclose(sum(p.CAPS.values()),.4)
    assert p.ALLOCATION.alpha_cosine==p.ALLOCATION.alpha_sine==.05/(2*18)

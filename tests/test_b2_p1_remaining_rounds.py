"""Only artificial matrices and mocked round records; no molecular runs."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from trotterlib import b2_p1_remaining_rounds as m
from trotterlib.df_rte_tail import exact_df_diagonal_coefficients
from trotterlib.rte import make_rte_config


@pytest.fixture(autouse=True)
def forbid_private_circuits_and_shots(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('No circuit construction, private inputs, sampling or shots')
    from qiskit import QuantumCircuit
    monkeypatch.setattr(QuantumCircuit,'__init__',forbidden)
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(m.p.connection,'prepare_b2',forbidden)
    monkeypatch.setattr(m.p,'controlled_checks',forbidden)
    monkeypatch.setattr(m.p.connection.accounting,'_hoeffding_shots',forbidden)


def test_fixed_ascending_scope_and_first_failure_stop():
    seen=[];saved=[]
    def evaluate(index):
        seen.append(index)
        return dict(m=index,independent_numerical_match=True,
                    qualification=dict(feasible=index!=4,reasons=[] if index!=4 else ['PF_phase_budget']))
    rows,status=m.evaluate_in_order(evaluate,lambda rows:saved.append([r['m'] for r in rows]))
    assert seen==[1,2,3,4] and saved[-1]==seen
    assert status=='STOP_P1_INELIGIBLE' and rows[-1]['status']==status
    assert 0 not in seen and 17 not in seen


def test_fixed_16_scope_and_numeric_failure_no_qualification_required():
    seen=[]
    def evaluate(index):
        seen.append(index)
        return dict(m=index,independent_numerical_match=False)
    rows,status=m.evaluate_in_order(evaluate,lambda rows:None)
    assert seen==[1] and status=='STOP_NUMERICAL_MATCH'
    rows,status=m.evaluate_in_order(lambda i:dict(m=i,independent_numerical_match=True,
        qualification=dict(feasible=True)),lambda rows:None)
    assert [r['m'] for r in rows]==list(range(1,17))
    assert status=='REMAINING_16_ROUND_CONDITIONAL_PASS'


def toy_inputs():
    n=2
    one=np.array([[.4,.13j],[-.13j,-.2]])
    gs=tuple(np.array(g,complex) for g in (
        [[1,.1],[.1,.4]],[[.3,.2j],[-.2j,.7]],[[.1,.13],[.13,.9]],
        [[.8,.2j],[-.2j,.1]],[[.2,.21],[.21,.6]]))
    weights=np.array([.4,-.2,.3,.15,-.11]);constant=.17
    components=[];identities=[]
    for i in range(3,5):
        _,eta=m.diag_hermitian(gs[i],assume_hermitian=True)
        for support,c in exact_df_diagonal_coefficients(eta,weights[i]):
            if not support:
                identities.append(c)
            elif c:
                components.append(dict(fragment=i,support=list(support),coefficient=c))
    lam=sum(abs(c['coefficient']) for c in components)
    identity=sum(identities)
    projection=dict(components=components,lambda_R=lam,extracted_tail_identity=identity,projection_hash='toy_projection')
    screen=dict(input=dict(sha256=m.p.DF_SHA,df_hamiltonian_hash='toy_DF'),coefficient_projection=projection)
    prior=dict(input_identity=dict(df_hamiltonian='toy_DF'),normalization_law=dict(lambda_R=lam,tail_identity=identity))
    ref=m.p.car_reference();a=ref.annihilators(n)
    lifted=ref.one_body(one,a).toarray()
    fragments=tuple(w*(ref.one_body(g,a)@ref.one_body(g,a)).toarray() for w,g in zip(weights,gs))
    ham=SimpleNamespace(one_body=one,lambdas=weights,g_matrices=gs,constant=constant,n_qubits=n)
    return dict(ham=ham,ref=ref,indices=np.arange(4),one_body=lifted,fragments=fragments),screen,prior


def test_no_circuit_mean_projection_and_independent_gaussian_semantics():
    inputs,screen,prior=toy_inputs()
    prep=m.mean_preparation(inputs['ham'],screen,prior)
    tail=sum(inputs['fragments'][3:],np.zeros((4,4),complex))-prep.extracted_identity_coefficient*np.eye(4)
    det=[inputs['one_body'],*inputs['fragments'][:3]]
    deterministic=[(str(i),m.p.legacy._eigendecomposition(b)) for i,b in enumerate(det)]
    eig=m.p.legacy._eigendecomposition(tail/prep.exact_rte_lambda_r)
    config,dist,kernel,steps=m.p.legacy_mean_steps(prep,deterministic,eig)
    native=m.ideal_program_mean(inputs,screen,config,kernel['phase'])
    np.testing.assert_allclose(native,steps['corrected'],rtol=0,atol=3e-14)
    moments=m.finite_rte_operator_moments(tail/config.lambda_r,config)
    raw=m.p.compose_step([expm(-.5j*m.p.DELTA*b) for b in det],moments.attenuated_event_mean_operator,kernel['phase'])
    np.testing.assert_allclose(raw,steps['raw'],rtol=0,atol=3e-14)
    assert not hasattr(prep,'deterministic_blocks') and not hasattr(prep,'native')


def test_extracted_round_evaluator_uses_existing_caps_and_never_shots():
    inputs,screen,prior=toy_inputs()
    prep=m.mean_preparation(inputs['ham'],screen,prior)
    config,_=make_rte_config(prep.rte_preparation.symbolic_tail,evolution_time=m.p.DELTA,
        rte_steps=4,finite_taylor_order=2,truncation_tolerance=1.)
    ctx=m.p.connection.RPEConnectionContext('toy_phys','toy_DF','toy_original_state','toy_sector',
        'toy_order','toy_SOURCE',m.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0'),
        'no_backend','rz_count',None,.01,.95,'toy prior')
    state=np.array([1.,0,0,0],complex);identity=np.eye(4,dtype=complex)
    inputs.update(state=state,sector_state=state,energy=.1,physical=.1*identity,df=.1*identity)
    step=np.exp(-.1j*m.p.DELTA)*identity
    B=config.distribution_normalization**4
    steps=dict(ideal_tail=step,corrected=step,raw=step/B)
    round_=m.MeanRound(ctx,m.RPERoundSpecification(1,m.p.DELTA),config,'toy_projection','toy_preparation')
    row=m.p.evaluate_signal_round(inputs,round_,kernel=dict(occurrence_log_normalization=np.log(B)),
        steps=steps,independent=steps,native_step=step)
    assert row['m']==1 and row['Q']==2 and row['independent_numerical_match']
    assert row['qualification']['feasible'] and not row['qualification']['shot_accounting_called']
    assert not hasattr(round_,'native') and not hasattr(round_,'sampled_request')

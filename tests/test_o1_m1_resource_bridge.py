"""Invented two-mode matrices and mock prices only; no saved science inputs."""
from dataclasses import replace
import importlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Pauli, Statevector
from scipy.linalg import expm

from test_o1_m1_matched_split import PACKAGE, adapter, toy_inputs, annihilators, lifted_one

bridge=importlib.import_module(PACKAGE+'.o1_m1_resource_bridge')
rte=importlib.import_module(PACKAGE+'.rte')
accounting=importlib.import_module(PACKAGE+'.rpe_resource_accounting')
tail_module=importlib.import_module(PACKAGE+'.df_rte_tail')
validation=importlib.import_module(PACKAGE+'.o1_m1_circuit_signal_validation')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


@pytest.fixture(autouse=True)
def no_saved_input_or_sampling(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('No saved input, scientific sampling or numerical replay permitted')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(rte,'sample_rte_events',forbidden)
    monkeypatch.setattr(rte,'iter_sample_rte_events',forbidden)


def prepared(dictionary='P',r=1,K=0):
    p=bridge.prepare_matched_resources(adapter.prepare_matched_a(toy_inputs()),dictionary=dictionary)
    c,d=bridge.configure_law(p,delta_time=.13,r=r,K=K)
    return p,c,d


def compiler():
    return rte.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,7,'1.3.0')


def fake_compile(circuit,settings,**kwargs):
    # Inspect the actual full wrapper; prices distinguish axes and trajectories.
    assert circuit.num_clbits==1 and circuit.data[-1].operation.name=='measure'
    assert circuit.data[0].operation.name=='h'
    assert kwargs['actual_circuit_fingerprint']
    assert all(item.operation.name not in ('initialize','state_preparation') for item in circuit.data)
    n=len(circuit.data)+(.5 if circuit.count_ops().get('sdg',0) else 0)
    return rte.CircuitCost(n,n/2,3,2,n+3,n+4,settings,5,'exact_compiled_rpe_hadamard_interrogation_expectation')


def priced(p,c,d):
    events=rte.enumerate_rte_events(p.tail.components,d)
    provider=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=fake_compile)
    return provider.exact(p,c,d,q=1,trajectories=((e,) for e in events))


def round_args(p,c,*,m=0):
    return dict(context_hash=p.context_hash,arm_hash=p.preparation_hash,
        specification=accounting.RPERoundSpecification(m,c.evolution_time),
        allocation=accounting.RPEErrorAllocation(.05,.05,.1,.01,.02),beta_rpe=.3,
        reference_radius_lower_bound=.62,corrected_signal_error_bound=.001,
        pf_phase_bound=.03,model_phase_bound=.01,implementation_phase_bound=.001,
        radius_provenance='invented analytical lower bound',cost_metric='rz_count',config=c)


def test_common_context_original_state_and_content_sensitive_dictionary_hash():
    p,c,d=prepared(); f,_,_=prepared('DF')
    assert p.context_hash==f.context_hash and p.preparation_hash!=f.preparation_hash
    np.testing.assert_array_equal(p.split.inputs.state,toy_inputs().state)
    assert np.vdot(p.split.inputs.state,p.split.inputs.state).real>1
    inputs=toy_inputs()
    changed=replace(inputs,state=inputs.state+np.array([0,1e-10,0,0]))
    other=bridge.prepare_matched_resources(adapter.prepare_matched_a(changed),dictionary='P')
    assert p.context_hash!=other.context_hash and p.tail.tail_hash!=other.tail.tail_hash
    weights=inputs.df_lambdas.copy(); weights[0]+=1e-12
    changed=replace(inputs,df_lambdas=weights)
    other=bridge.prepare_matched_resources(adapter.prepare_matched_a(changed),dictionary='P')
    assert p.tail.tail_hash!=other.tail.tail_hash


@pytest.mark.parametrize('dictionary',['P','DF'])
def test_controlled_trajectory_common_T_signed_identity_and_scalar(dictionary):
    p,c,d=prepared(dictionary,r=2,K=2)
    events=validation.representative_events(p.tail.components,d)
    chosen=(events[-1],events[0],events[-2],events[1])
    actual=Operator(bridge.build_controlled_trajectory(p,c,d,q=2,events=chosen)).data
    if dictionary=='P':
        operators={x.component_id:x.coefficient_sign*Pauli(p.labels[x.component_id]).to_matrix()
                   for x in p.tail.components}
    else:
        extraction=adapter.extract_matched_df_tail(p.split,identity_policy='faithful_identity_in_tail')
        dense=tail_module.extraction_to_normalized_rte_tail(extraction)
        operators=dict(zip((x.component_id for x in dense.components),dense.operators))
    h=lifted_one(toy_inputs().df_one_body,annihilators(2))
    half=expm(-1j*c.evolution_time*h/2)
    expected=np.eye(4,dtype=complex)
    for j in range(2):
        expected=half@expected
        for event in chosen[2*j:2*j+2]: expected=rte.event_unitary(event,operators)@expected
        expected=np.exp(-1j*c.evolution_time*toy_inputs().nuclear_repulsion)*half@expected
    np.testing.assert_allclose(actual,validation.controlled_reference(expected),rtol=0,atol=3e-14)
    state=toy_inputs().state/np.linalg.norm(toy_inputs().state)
    signal=np.vdot(state,expected@state)
    for axis,target in [('cosine',signal.real),('sine',signal.imag)]:
        circuit=bridge.wrap_controlled_no_prep(bridge.build_controlled_trajectory(p,c,d,q=2,events=chosen),
            ancilla_qubit=2,axis=axis,include_measurement=False)
        value=Statevector(np.r_[state,np.zeros(4)]).evolve(circuit).expectation_value(Pauli('ZII'))
        assert value.real==pytest.approx(target,abs=3e-14)


@pytest.mark.parametrize('dictionary',['P','DF'])
def test_complete_exact_provider_prices_full_axis_wrappers(dictionary):
    p,c,d=prepared(dictionary)
    cost=priced(p,c,d)
    assert cost.circuit_cost_scope==bridge.RPE_HADAMARD_INTERROGATION_SCOPE
    assert cost.classical_sample_count is None
    assert cost.sine_expected_cost.rz_count>cost.cosine_expected_cost.rz_count
    assert dict(cost.metadata)['state_preparation_included'] is False
    events=rte.enumerate_rte_events(p.tail.components,d)
    expected=math.fsum(e.event_probability*fake_compile(bridge.wrap_controlled_no_prep(
        bridge.build_controlled_trajectory(p,c,d,q=1,events=(e,)),ancilla_qubit=2,axis='cosine'),
        compiler(),actual_circuit_fingerprint='toy').rz_count for e in events)
    assert cost.cosine_expected_cost.rz_count==pytest.approx(expected,abs=1e-13)


def test_incomplete_duplicate_or_forged_events_rejected_before_build(monkeypatch):
    p,c,d=prepared()
    events=rte.enumerate_rte_events(p.tail.components,d)
    def forbidden(*a,**kw): raise AssertionError('Must reject before circuit build')
    monkeypatch.setattr(bridge,'build_controlled_trajectory',forbidden)
    provider=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=fake_compile)
    with pytest.raises(ValueError,match='complete'): provider.exact(p,c,d,q=1,trajectories=((events[0],),))
    with pytest.raises(ValueError,match='duplicate'): provider.exact(p,c,d,q=1,trajectories=((events[0],),)*len(events))
    forged=replace(events[0],rotation_angle=events[0].rotation_angle+.1)
    with pytest.raises(ValueError,match='phase'): provider.exact(p,c,d,q=1,trajectories=((forged,),))
    with pytest.raises(ValueError,match='canonical backend'):
        bridge.MatchedNoPrepCostProvider(compiler(),backend=object())


def test_actual_radius_normalization_single_qr_exponent_and_no_MC_shot_multiplier():
    p,c,d=prepared(r=1); cost=priced(p,c,d)
    record=bridge.account_radius_round(**round_args(p,c),cost=cost)
    assert record['guarantee_status']=='not_certified' and record['feasible']
    A=c.distribution_normalization**(-c.rte_steps)
    assert record['attenuation']==pytest.approx(A)
    assert record['rho_observed_lower_bound']==pytest.approx(A*(.62-.001))
    assert record['shots']['cosine']==bridge._hoeffding_shots(attenuation=A,reference_radius=.62,
        signal_error_bound=.001,beta_stat_budget=.1,alpha_axis=.01)
    expected=record['shots']['cosine']*cost.cosine_expected_cost.rz_count+record['shots']['sine']*cost.sine_expected_cost.rz_count
    assert record['round_total_cost']==pytest.approx(expected)
    mc_metadata=replace(cost,classical_sample_count=204)
    assert bridge.account_radius_round(**round_args(p,c),cost=mc_metadata)['round_total_cost']==record['round_total_cost']
    p,c,d=prepared(r=3)
    record=bridge.account_radius_round(**round_args(p,c,m=2))
    assert record['attenuation']==pytest.approx(c.distribution_normalization**(-12))
    assert record['round_total_cost'] is None


def test_default_compiler_on_invented_two_qubit_controlled_phase():
    # Actual Qiskit compile primitive, solely an invented one-system-qubit gate.
    from qiskit import QuantumCircuit
    circuit=QuantumCircuit(2); circuit.cp(.17,1,0); circuit.p(-.11,1)
    provider=bridge.MatchedNoPrepCostProvider(compiler())
    cost=provider.deterministic(circuit,context_hash='toy_phase',arm_hash='toy_DF_PF',q=1,delta_time=.1)
    assert cost.cosine_expected_cost.compiler==compiler()
    assert cost.cosine_expected_cost.rz_count>0 and cost.sine_expected_cost.circuit_size>0


def test_wrong_law_deterministic_scope_backend_and_phase_domain_fail_closed():
    p,c,d=prepared(); cost=priced(p,c,d); args=round_args(p,c)
    other,_=bridge.configure_law(p,delta_time=c.evolution_time,r=2,K=0)
    for overrides in ({'config':other},{'config':None},{'backend_hash':'other_backend'},
                      {'cost':replace(cost,circuit_cost_scope='compiled_time_evolution_subcircuit')}):
        with pytest.raises(ValueError,match='full no-prep'):
            bridge.account_radius_round(**(args|{'cost':cost}|overrides))
    for beta in (0,math.pi/2+1e-6):
        with pytest.raises(ValueError,match='pi/2'): bridge.account_radius_round(**(args|{'beta_rpe':beta}))
    invalid=replace(args['allocation'],beta_stat_budget=math.pi)
    with pytest.raises(ValueError,match='pi/2'): bridge.account_radius_round(**(args|{'allocation':invalid}))
    record=bridge.account_radius_round(**(args|{'corrected_signal_error_bound':.7}),cost=cost)
    assert not record['feasible'] and record['shots'] is None and record['round_total_cost'] is None


def test_registered_round_aggregation_refuses_unknown_incomplete_or_mixed_costs():
    p,c,d=prepared(); cost=priced(p,c,d)
    r0=bridge.account_radius_round(**round_args(p,c),cost=cost)
    aggregate=bridge.aggregate_registered_rounds((r0,),required_max_round=0,total_alpha_budget=.1)
    assert aggregate['total_cost']==r0['round_total_cost'] and aggregate['guarantee_status']=='not_certified'
    with pytest.raises(ValueError,match='Every registered'): bridge.aggregate_registered_rounds((r0,),required_max_round=1,total_alpha_budget=.1)
    with pytest.raises(ValueError,match='unpriced'): bridge.aggregate_registered_rounds((r0|{'round_total_cost':None},),required_max_round=0,total_alpha_budget=.1)
    with pytest.raises(ValueError,match='Mixed'): bridge.aggregate_registered_rounds((r0,r0|{'round_index':1,'backend_hash':'different'}),required_max_round=1,total_alpha_budget=.1)
    with pytest.raises(ValueError,match='failure budget'): bridge.aggregate_registered_rounds((r0,),required_max_round=0,total_alpha_budget=.001)


def test_strong_deterministic_DF_baseline_circuit_and_same_full_no_prep_pricing():
    p,c,d=prepared('DF')
    circuit=bridge.build_strong_df_baseline(p.split,delta_time=.13,q=2,substeps=2,
        basis_policy='legacy_spin_orbital_eigh')
    a=annihilators(2); inputs=toy_inputs()
    generators=[lifted_one(inputs.df_one_body,a)]
    generators += [w*lifted_one(g,a)@lifted_one(g,a) for w,g in zip(inputs.df_lambdas,inputs.df_g_matrices)]
    expected=np.eye(4,dtype=complex)
    schedule=[(0,.13/4),(1,.13/4),(2,.13/2),(1,.13/4),(0,.13/4)]
    for _ in range(4):
        for i,t in schedule: expected=expm(-1j*t*generators[i])@expected
    expected*=np.exp(-1j*.26*inputs.nuclear_repulsion)
    np.testing.assert_allclose(Operator(circuit).data,validation.controlled_reference(expected),rtol=0,atol=3e-14)
    assert circuit.metadata['relative_fusion'] and circuit.metadata['scalar_once']
    provider=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=fake_compile)
    cost=provider.deterministic(circuit,context_hash=p.context_hash,arm_hash='strong_toy',q=2,delta_time=.13)
    args=round_args(p,c,m=1)|{'arm_hash':'strong_toy','config':None,'corrected_signal_error_bound':0.}
    record=bridge.account_radius_round(**args,cost=cost)
    assert record['feasible'] and record['attenuation']==record['normalization_product']==1
    assert record['round_total_cost']>0

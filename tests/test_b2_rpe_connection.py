"""Artificial two-mode Hamiltonians/rounds and mock prices; no saved science I/O."""
from dataclasses import replace
import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Pauli, Statevector
from scipy.linalg import expm

from trotterlib import b2_rpe_connection as m
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder
from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash
from trotterlib.rpe_resource_accounting import RPEErrorAllocation, RPERoundSpecification
from trotterlib.rte import CompilerSettings, CircuitCost, enumerate_rte_events, make_rte_config
from trottertracks import c1_h4_strong_baseline as strong
from test_o1_m1_matched_split import annihilators, lifted_one


@pytest.fixture(autouse=True)
def no_scientific_inputs(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('No saved numerical input or scientific state solve')
    monkeypatch.setattr(np, 'load', forbidden)
    import trotterlib.df_hamiltonian as h
    for name in ('solve_df_ground_state', 'build_df_h_d_from_molecule'):
        monkeypatch.setattr(h, name, forbidden)


def toy():
    return DFHamiltonian(constant=.17, one_body=np.array([[.4,.13j],[-.13j,-.2]]),
        lambdas=np.array([.6,-.35,.22]), g_matrices=(
        np.array([[1.,.19],[.19,.35]],complex),
        np.array([[.2,-.23j],[.23j,1.2]],complex),
        np.array([[.7,.31],[.31,-.4]],complex)), metadata={'name':'ARTIFICIAL_B2'})


def compiler():
    return CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,7,'1.3.0')


def context(h):
    return m.RPEConnectionContext('toy_original_H', df_hamiltonian_hash(h),
        'toy_original_H_state', 'toy_N1', 'qubit0_LSB; interleaved_spin', 'toy_SOURCE',
        compiler(), 'no_backend', 'rz_count', .001, .01, .9,
        'toy energy interval; supplied nearest-branch conditions, unqualified')


def case(index=0, r=2, K=2, L_D=2):
    h=toy(); p=m.prepare_b2(h,L_D=L_D); ctx=context(h)
    spec=RPERoundSpecification(index,.13)
    c,d=make_rte_config(p.rte_preparation.symbolic_tail,evolution_time=spec.delta_time,
                       rte_steps=r,finite_taylor_order=K,truncation_tolerance=1.)
    return h,m.B2RPERound(ctx,p,spec,L_D,r,K,.13,spec.q_m,c,d)


def fake_compile(circuit, settings, **kwargs):
    assert circuit.num_clbits==1 and circuit.data[-1].operation.name=='measure'
    assert circuit.data[0].operation.name=='h'
    assert all(x.operation.name not in ('initialize','state_preparation') for x in circuit.data)
    assert kwargs['actual_circuit_fingerprint']
    n=len(circuit.data)+(.5 if circuit.count_ops().get('sdg',0) else 0)
    return CircuitCost(n,n/2,3,2,n+3,n+4,settings,5,
                       'exact_compiled_rpe_hadamard_interrogation_expectation')


def provider():
    return m.accounting.MatchedNoPrepCostProvider(compiler(),compile_function=fake_compile)


def signals(round):
    # An invented attenuated signal, deliberately distinct from its PF reference.
    vals=[.6*np.exp(-.13j), .6*np.exp(-.131j), .59*np.exp(-.132j),
          .5899*np.exp(-.1321j)]
    A=m.finite_rte_attenuation(round.config,tail_evolutions=round.q)
    vals.append(A*vals[-1])
    sv=[m.SignalValue(z,round.context.original_state,
        round.context.original_hamiltonian if i==0 else round.context.df_hamiltonian)
        for i,z in enumerate(vals)]
    return m.B2SignalRecord(round.binding,*sv,1/A,1e-13)


def account(round, record=None, cost=None):
    return m.account_signal_round(round, signals(round) if record is None else record,
        allocation=RPEErrorAllocation(.02,.02,.1,.01,.015), beta_rpe=.3,cost=cost,
        radius_allowance=1e-12,complex_allowance=1e-12,phase_allowance=1e-12,
        normalization_tolerance=1e-12)


@pytest.mark.parametrize('event_case',['sampled','canonical_order2'])
@pytest.mark.parametrize('index',[0,1])
@pytest.mark.parametrize('L_D',[1,2])
def test_nonzero_prefix_replay_controlled_phase_and_both_wrappers(index,L_D,event_case):
    h,round=case(index=index,L_D=L_D)
    if event_case=='sampled':
        request=round.sampled_request(seed=41)
    else:
        events=enumerate_rte_events(round.tail.components,round.distribution,max_events=10000)
        ordered=[e for e in events if e.taylor_order==2 and
                 e.product_component_ids[0]!=e.product_component_ids[1]]
        assert ordered and ordered[0].phase==-1
        selected=(ordered[0],events[0],ordered[-1],events[1])[:round.q*round.r]
        request=round.explicit_request(selected)
    evolution=round.native(request,max_instructions=100000)
    legacy=QiskitDFPartialS2RepeatedCircuitBuilder().build(request)
    np.testing.assert_allclose(Operator(evolution.circuit).data,Operator(legacy.circuit).data,rtol=0,atol=4e-14)
    # Independent ordered product with unchanged scalar and event operators.
    p=round.preparation; a=annihilators(2)
    blocks=[lifted_one(h.one_body,a)]+[h.lambdas[i]*lifted_one(h.g_matrices[i],a)@lifted_one(h.g_matrices[i],a)
                                      for i in range(L_D)]
    from trotterlib.rte import event_unitary
    from trotterlib.df_rte_tail import extraction_to_normalized_rte_tail
    tail=extraction_to_normalized_rte_tail(p.tail_extraction)
    expected=np.eye(4,dtype=complex)
    for occurrence in request.rte_occurrences:
        for block in blocks: expected=expm(-.5j*round.delta*block)@expected
        for event in occurrence.events:
            expected=event_unitary(event,{c.component_id:o for c,o in zip(tail.components,tail.operators)})@expected
        for block in reversed(blocks): expected=expm(-.5j*round.delta*block)@expected
        expected*=np.exp(-1j*round.delta*(h.constant+p.extracted_identity_coefficient))
    actual=Operator(evolution.circuit).data
    np.testing.assert_allclose(actual[:4,:4],np.eye(4),rtol=0,atol=4e-14)
    np.testing.assert_allclose(actual[4:,4:],expected,rtol=0,atol=4e-14)
    psi=np.array([.2,.3j,-.4,.5j]); psi/=np.linalg.norm(psi)
    z=np.vdot(psi,expected@psi)
    for axis,target in [('cosine',z.real),('sine',z.imag)]:
        wrapper=m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=100000)
        measured=Statevector(np.r_[psi,np.zeros(4)]).evolve(wrapper).expectation_value(Pauli('ZII')).real
        assert measured==pytest.approx(target,abs=3e-14)
    # Replaying the same event tuple keeps exact chronological event order.
    events=tuple(e for x in request.rte_occurrences for e in x.events)
    replay=round.explicit_request(events)
    assert tuple(e for x in replay.rte_occurrences for e in x.events)==events
    np.testing.assert_allclose(Operator(round.native(replay,max_instructions=100000).circuit).data,actual,rtol=0,atol=1e-14)


def test_hamiltonian_reconstruction_scalar_and_identity_once():
    h,round=case(); p=round.preparation; a=annihilators(2)
    from trotterlib.df_rte_tail import dense_extracted_df_tail
    original=h.constant*np.eye(4)+lifted_one(h.one_body,a)
    for w,g in zip(h.lambdas,h.g_matrices): original+=w*lifted_one(g,a)@lifted_one(g,a)
    prefix=lifted_one(h.one_body,a)
    for i in range(p.ld): prefix+=h.lambdas[i]*lifted_one(h.g_matrices[i],a)@lifted_one(h.g_matrices[i],a)
    tail=dense_extracted_df_tail(p.tail_extraction)
    assert p.extracted_identity_coefficient!=0
    np.testing.assert_allclose(prefix+tail+p.constant_coefficient*np.eye(4),original,rtol=0,atol=3e-14)


@pytest.mark.parametrize('field,value',[
    ('L_D',1),('q',2),('r',3),('K',0),('delta',.2)])
def test_round_config_mismatch_refused(field,value):
    _,round=case()
    with pytest.raises(ValueError): replace(round,**{field:value})


def test_forged_law_and_event_refused_before_build():
    _,round=case()
    with pytest.raises(ValueError): replace(round,distribution=replace(round.distribution,exact_finite_distribution=2.))
    events=tuple(e for x in round.sampled_request(seed=19).rte_occurrences for e in x.events)
    with pytest.raises(ValueError,match='phase'):
        round.explicit_request((replace(events[0],rotation_angle=events[0].rotation_angle+.1),*events[1:]))


def test_raw_corrected_radius_accounting_and_paired_shot_prices():
    _,round=case(index=1); cost=round.price(provider(),sample_count=4,seed=19,max_instructions=100000)
    record=signals(round); result=account(round,record,cost)
    A=round.config.distribution_normalization**(-round.q*round.r)
    assert result['attenuation']==pytest.approx(A)
    assert result['raw_signal_radius']==pytest.approx(A*result['corrected_signal_radius'])
    assert result['rho_observed_lower_bound']==pytest.approx(A*(result['reference_radius_lower_bound']-abs(record.finite_corrected.value-record.ideal_tail_pf.value)-1e-12))
    assert result['shots']['cosine']!=result['shots']['sine']
    expected=result['shots']['cosine']*cost.cosine_expected_cost.rz_count+result['shots']['sine']*cost.sine_expected_cost.rz_count
    assert result['round_total_cost']==pytest.approx(expected)
    assert account(round,record,replace(cost,classical_sample_count=999))['round_total_cost']==result['round_total_cost']
    assert result['guarantee_status']=='not_certified'
    assert result['phase_budgets']==dict(PF=.02,RTE=.02,statistical=.1)


@pytest.mark.parametrize('field',['original','df','ideal_tail_pf','finite_corrected','raw_measured'])
def test_every_signal_refuses_df_state_substitution(field):
    _,round=case(); record=signals(round)
    wrong=replace(record,**{field:replace(getattr(record,field),state_identity='DF_ground_state')})
    with pytest.raises(ValueError,match='state'): account(round,wrong)


@pytest.mark.parametrize('defect',['raw_twice','corrected_twice','normalization','binding','wrong_H','state_role'])
def test_signal_normalization_and_identity_rejection(defect):
    _,round=case(); record=signals(round)
    if defect=='raw_twice': record=replace(record,raw_measured=replace(record.raw_measured,value=record.raw_measured.value/record.normalization_product))
    if defect=='corrected_twice': record=replace(record,finite_corrected=replace(record.finite_corrected,value=record.finite_corrected.value*record.normalization_product))
    if defect=='normalization': record=replace(record,normalization_product=record.normalization_product**2)
    if defect=='binding': record=replace(record,round_binding='old_fixed_time_B2')
    if defect=='wrong_H': record=replace(record,original=replace(record.original,hamiltonian_identity=round.context.df_hamiltonian))
    if defect=='state_role': record=replace(record,df=replace(record.df,state_role='DF_ground_state'))
    with pytest.raises(ValueError): account(round,record)


@pytest.mark.parametrize('field',['source','original_state','original_hamiltonian','sector','spin_orbital_order'])
def test_source_and_physical_context_bound_to_signals(field):
    _,round=case(); changed=replace(round,context=replace(round.context,**{field:'different'}))
    with pytest.raises(ValueError,match='binding'): account(changed,signals(round))


@pytest.mark.parametrize('field',['model_error_bound','energy_precision','success_probability','alias_branch_conditions'])
def test_missing_scientific_condition_stays_unevaluated(field):
    _,round=case(); changed=replace(round,context=replace(round.context,**{field:None}))
    with pytest.raises(ValueError,match='unevaluated'): account(changed)


def test_cost_context_law_compiler_scope_mismatch_refused():
    _,round=case(); cost=round.price(provider(),sample_count=2,seed=17,max_instructions=100000)
    for field in ('context_hash','arm_hash','q','delta_time','law_hash','backend_hash','state_preparation_included'):
        wrong=replace(cost,metadata=tuple(dict(cost.metadata,**{field:'wrong'}).items()))
        with pytest.raises(ValueError,match='full no-prep'): account(round,cost=wrong)
    with pytest.raises(ValueError,match='full no-prep'):
        account(round,cost=replace(cost,circuit_cost_scope='compiled_time_evolution_subcircuit'))
    bad=replace(cost.sine_expected_cost,compiler=replace(compiler(),transpiler_seed=9))
    with pytest.raises(ValueError,match='compiler'): account(round,cost=replace(cost,sine_expected_cost=bad))
    bad_provider=m.accounting.MatchedNoPrepCostProvider(replace(compiler(),transpiler_seed=9),compile_function=fake_compile)
    with pytest.raises(ValueError,match='compiler'): round.price(bad_provider,sample_count=2,seed=17,max_instructions=100000)


def test_paired_axes_share_events_fresh_trajectories_and_seed_replay(monkeypatch):
    _,round=case(); observed=[]; pairs=[]
    original=m.partial_native_from_step_requests
    def capture(requests,**kwargs):
        observed.append(tuple(e for req in requests for e in req.rte_occurrence.events))
        return original(requests,**kwargs)
    monkeypatch.setattr(m,'partial_native_from_step_requests',capture)
    def compile_capture(circuit,settings,**kwargs):
        pairs.append(circuit)
        return fake_compile(circuit,settings,**kwargs)
    p=m.accounting.MatchedNoPrepCostProvider(compiler(),compile_function=compile_capture)
    first=round.price(p,sample_count=6,seed=21,max_instructions=100000)
    assert len(observed)==6 and len(pairs)==12
    assert len(set(observed))>1
    for X,Y in zip(pairs[::2],pairs[1::2]):
        assert X.data[:-2]==Y.data[:-3]  # same H + full trajectory; Y alone has Sdg
    saved=tuple(observed)
    second=round.price(p,sample_count=6,seed=21,max_instructions=100000)
    assert tuple(observed[6:])==saved
    assert first==second and first.classical_sample_count==6
    assert dict(first.metadata)['paired_axes'] is True


def test_all_rounds_required_and_existing_sum_formula():
    records=[]
    for i in range(2):
        _,round=case(index=i)
        price=round.price(provider(),sample_count=2,seed=10+i,max_instructions=100000)
        records.append(account(round,cost=price))
    kwargs=dict(required_max_round=1,total_alpha_budget=.06,context=round.context)
    total=m.aggregate_b2_rounds(records,**kwargs)
    assert total['total_cost']==pytest.approx(sum(r['round_total_cost'] for r in records))
    for bad in (records[:1],records+records[:1],[records[0],records[1]|{'round_total_cost':None}]):
        with pytest.raises(ValueError): m.aggregate_b2_rounds(bad,**kwargs)
    with pytest.raises(ValueError): m.aggregate_b2_rounds(records,**(kwargs|{'context':replace(round.context,source='other')}))
    with pytest.raises(ValueError): m.aggregate_b2_rounds(records,**(kwargs|{'total_alpha_budget':.11}))


@pytest.mark.parametrize('formula',['Full_S2','Yoshida4'])
@pytest.mark.parametrize('fusion',[False,True])
def test_strong_baseline_same_input_controlled_scalar_wrapper_and_schedule(formula,fusion):
    h,round=case(index=1)
    baseline=m.build_strong_baseline(h,context=round.context,specification=round.specification,
        substeps=2,formula=formula,basis_policy=strong.LEGACY,relative_fusion=fusion)
    a=annihilators(2); generators=[lifted_one(h.one_body,a)]
    generators += [w*lifted_one(g,a)@lifted_one(g,a) for w,g in zip(h.lambdas,h.g_matrices)]
    if formula=='Full_S2': weights=(1.,)
    else:
        w=1/(2-2**(1/3)); weights=(w,1-2*w,w)
    expected=np.eye(4,dtype=complex)
    for _ in range(round.q*2):
        for w in weights:
            for i,t in [(0,.25),(1,.25),(2,.25),(3,.5),(2,.25),(1,.25),(0,.25)]:
                expected=expm(-1j*round.delta*w*t*generators[i])@expected
    expected*=np.exp(-1j*round.q*round.delta*h.constant)
    matrix=Operator(baseline.circuit).data
    np.testing.assert_allclose(matrix[:4,:4],np.eye(4),rtol=0,atol=8e-14)
    np.testing.assert_allclose(matrix[4:,4:],expected,rtol=0,atol=8e-14)
    cost=m.price_strong_baseline(baseline,provider())
    assert cost.circuit_cost_scope==m.RPE_HADAMARD_INTERROGATION_SCOPE
    assert dict(cost.metadata)['context_hash']==round.context_hash
    with pytest.raises(ValueError,match='input'):
        m.build_strong_baseline(replace(h,constant=h.constant+.1),context=round.context,
            specification=round.specification,substeps=2,formula=formula,basis_policy=strong.LEGACY,relative_fusion=fusion)


def test_spin_policy_is_opt_in_and_rejects_ineligible_structure():
    h,round=case()
    with pytest.raises(ValueError,match='SPIN_LIFT'):
        m.build_strong_baseline(h,context=round.context,specification=round.specification,
            substeps=1,formula='Full_S2',basis_policy=strong.SPIN,relative_fusion=True)


def test_declared_ld3_r4_k2_round_on_invented_factors_only():
    h=toy()
    h=replace(h,lambdas=np.r_[h.lambdas,-.12],g_matrices=(*h.g_matrices,
        np.array([[.8,.14j],[-.14j,-.1]],complex)))
    p=m.prepare_b2(h,L_D=3); spec=RPERoundSpecification(0,.07)
    c,d=make_rte_config(p.rte_preparation.symbolic_tail,evolution_time=.07,
                       rte_steps=4,finite_taylor_order=2,truncation_tolerance=1.)
    round=m.B2RPERound(context(h),p,spec,3,4,2,.07,1,c,d)
    request=round.sampled_request(seed=7)
    assert p.deterministic_fragment_indices==(0,1,2)
    assert len(request.rte_occurrences[0].events)==4
    np.testing.assert_allclose(Operator(round.native(request,max_instructions=100000).circuit).data,
        Operator(QiskitDFPartialS2RepeatedCircuitBuilder().build(request).circuit).data,rtol=0,atol=4e-14)


def test_same_accounting_for_strong_baseline_without_rte_attenuation():
    h,round=case()
    baseline=m.build_strong_baseline(h,context=round.context,specification=round.specification,
        substeps=1,formula='Full_S2',basis_policy=strong.LEGACY,relative_fusion=True)
    values=[.6*np.exp(-.13j),.6*np.exp(-.131j),.59*np.exp(-.132j)]
    sv=[m.SignalValue(z,round.context.original_state,
        round.context.original_hamiltonian if i==0 else round.context.df_hamiltonian)
        for i,z in enumerate(values)]
    record=m.B2SignalRecord(baseline.binding,*sv,sv[-1],sv[-1],1.,1e-13)
    cost=m.price_strong_baseline(baseline,provider())
    result=account(baseline,record,cost)
    assert result['attenuation']==result['normalization_product']==1
    assert result['finite_rte_phase_bound']==pytest.approx(1e-12/(.59-1e-12))
    assert result['context_hash']==round.context_hash
    assert result['round_total_cost']>0


def test_actual_whole_circuit_compile_on_synthetic_b2_only():
    _,round=case(r=1,K=0,L_D=2)
    p=m.accounting.MatchedNoPrepCostProvider(compiler())
    cost=round.price(p,sample_count=2,seed=3,max_instructions=100000)
    assert cost.cosine_expected_cost.compiler==compiler()
    assert cost.sine_expected_cost.compiler==compiler()
    assert cost.cosine_expected_cost.rz_count>0
    assert cost.classical_sample_count==2

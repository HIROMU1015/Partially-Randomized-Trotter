"""Native circuit correctness on fixed toys only; no compile/MC/molecular I/O."""
from dataclasses import replace
from pathlib import Path
import builtins
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator, Statevector

import trottertracks.resource_applicability.ax2b_native_df_v4 as native
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec, DFDeterministicFragmentSpec


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    original = builtins.open
    path_open = Path.open
    def check(path):
        if isinstance(path,(str,Path)) and (str(path).endswith('.npz') or '/artifacts/' in str(path) or '/.runtime/' in str(path)):
            raise AssertionError('Scientific file access forbidden: '+str(path))
    def guarded(path,*args,**kwargs):
        check(path)
        return original(path,*args,**kwargs)
    def guarded_path(path,*args,**kwargs):
        check(path)
        return path_open(path,*args,**kwargs)
    def forbidden(*args,**kwargs):
        raise AssertionError('Compilation/sampling/scientific loading forbidden in synthetic tests')
    monkeypatch.setattr(builtins,'open',guarded)
    monkeypatch.setattr(Path,'open',guarded_path)
    monkeypatch.setattr(np,'load',forbidden)
    import qiskit
    import trotterlib.rte_compiled_cost as cost
    import trotterlib.rte as rte
    import trotterlib.df_rte_circuit as event
    monkeypatch.setattr(qiskit,'transpile',forbidden)
    monkeypatch.setattr(cost,'transpile',forbidden)
    monkeypatch.setattr(event.DFRTEEventPreparation,'sample_occurrence_request',forbidden)
    monkeypatch.setattr(rte,'sample_rte_events',forbidden)


def basis(theta,beta):
    circuit=QuantumCircuit(2)
    circuit.p(.17,0)
    circuit.append(XXPlusYYGate(theta,beta),[0,1])
    circuit.p(-.23,1)
    return tuple((item.operation,tuple(circuit.find_bit(q).index for q in item.qubits)) for item in circuit.data)


def blocks():
    one=DFDeterministicOneBodySpec('toy_one','one_body',None,None,'toy_U0','toy_hash0',(),
        (.3,-.17),2,0,basis(.61,.2))
    frag=DFDeterministicFragmentSpec('toy_df','df_fragment',0,0,'toy_U1','toy_hash1',(),
        (.7,-.4),-.37,2,1,basis(-.43,-.37))
    other=replace(frag,block_id='toy_df2',original_fragment_index=1,rank=1,
                  diagonal_eta=(.13,.8),lam=.23,order_index=2,runtime_basis_operations=basis(.29,.51))
    return (one,frag,other)


def independent_term(block,time):
    # Fock occupations in Qiskit bit order; no shared diagonal primitive code.
    occupations=np.array([[(i>>j)&1 for j in range(2)] for i in range(4)])
    if isinstance(block,DFDeterministicOneBodySpec):
        energy=occupations@np.array(block.diagonal_eigenvalues)
    else:
        energy=block.lam*(occupations@np.array(block.diagonal_eta))**2
    circuit=QuantumCircuit(2)
    for gate,indices in block.runtime_basis_operations:
        circuit.append(gate,list(indices))
    U=Operator(circuit).data
    return U@np.diag(np.exp(-1j*time*energy))@U.conj().T


def ordinary(U):
    zero=np.zeros_like(U)
    return np.block([[np.eye(len(U)),zero],[zero,U]])


def independent_s2(bs,time):
    result=np.eye(4,dtype=complex)
    for block in tuple(bs)+tuple(reversed(bs)):
        result=independent_term(block,time/2)@result
    return result


@pytest.mark.parametrize('mode',['UNCONTROLLED','ORDINARY','DIRECTIONAL'])
@pytest.mark.parametrize('index',[0,1])
@pytest.mark.parametrize('time',[.8,-.8])
def test_native_block_matches_independent_fock_energies_and_all_phases(mode,index,time):
    block=blocks()[index]
    circuit=QuantumCircuit(3)
    native._append_block(circuit,block,time,mode,2)
    U=independent_term(block,time)
    if mode=='ORDINARY':expected=ordinary(U)
    elif mode=='DIRECTIONAL':
        zero=np.zeros_like(U)
        expected=np.block([[independent_term(block,-time),zero],[zero,U]])
    else:expected=np.kron(np.eye(2),U)
    np.testing.assert_allclose(Operator(circuit).data,expected,atol=3e-13)
    assert len(circuit.data)<=native.block_instruction_bound(block,mode)


@pytest.mark.parametrize('policy',['ordinary','symmetric_directional'])
@pytest.mark.parametrize('formula',['2nd','4th'])
@pytest.mark.parametrize('q',[1,3])
@pytest.mark.parametrize('T',[.8,-.8])
def test_native_global_pf_matches_independent_noncommuting_composition(policy,formula,q,T):
    bs=blocks()
    result=native.build_deterministic_native(bs,num_system_qubits=2,T=T,q=q,formula=formula,
        scalar=.31,control_policy=policy,max_instructions=10000)
    w=1/(2-2**(1/3));middle=-2**(1/3)*w
    weights=(1,) if formula=='2nd' else (w,middle,w)
    U=np.eye(4,dtype=complex)
    for _ in range(q):
        for weight in weights:U=independent_s2(bs,T*weight/q)@U
    U*=np.exp(-1j*.31*T)
    np.testing.assert_allclose(Operator(result.circuit).data,ordinary(U),atol=5e-13)
    assert len(result.circuit.data)<=result.instruction_upper_bound


@pytest.mark.parametrize('policy',['ordinary','symmetric_directional'])
@pytest.mark.parametrize('q',[1,3])
@pytest.mark.parametrize('T',[.8,-.8])
def test_partial_keeps_explicit_tail_event_order_and_relative_phase(policy,q,T):
    bs=blocks()[:2];tails=[];expected=np.eye(4,dtype=complex)
    for i in range(q):
        tail=QuantumCircuit(3)
        tail.crz(.2+.07*i,2,0)
        tail.p(.19-.04*i,2)
        tails.append(tail)
        central=Operator(tail).data[4:,4:]
        forward=np.eye(4,dtype=complex);reverse=forward.copy()
        for b in bs:forward=independent_term(b,T/(2*q))@forward
        for b in reversed(bs):reverse=independent_term(b,T/(2*q))@reverse
        expected=reverse@central@forward@expected
    expected*=np.exp(-1j*.37*T)
    result=native.build_partial_native(bs,num_system_qubits=2,T=T,q=q,scalar=.37,
        controlled_tails=tails,control_policy=policy,max_instructions=10000)
    np.testing.assert_allclose(Operator(result.circuit).data,ordinary(expected),atol=3e-13)
    assert [len(t.data) for t in tails]==[2]*q


@pytest.mark.parametrize('policy',['ordinary','symmetric_directional'])
def test_empty_tail_b0_and_scalar_only_do_not_change_control_zero(policy):
    bs=blocks()[:2]
    b0=native.build_partial_native(bs,num_system_qubits=2,T=.8,q=3,scalar=.37,
        controlled_tails=None,control_policy=policy,max_instructions=10000)
    expected=np.exp(-.8j*.37)*np.linalg.matrix_power(independent_s2(bs,.8/3),3)
    np.testing.assert_allclose(Operator(b0.circuit).data,ordinary(expected),atol=3e-13)
    for builder,kwargs in [(native.build_deterministic_native,{'formula':'4th'}),
                           (native.build_partial_native,{'controlled_tails':None})]:
        result=builder((),num_system_qubits=2,T=.8,q=10**12,scalar=.37,
            control_policy=policy,max_instructions=10,**kwargs)
        np.testing.assert_allclose(Operator(result.circuit).data,ordinary(np.exp(-.8j*.37)*np.eye(4)),atol=3e-13)


def test_full_vector_callback_preserves_nonunit_norm_and_phase():
    vector=np.array([1,1j,.3,-.4j],dtype=complex)
    for block in blocks():
        action=native.make_native_block_action(block,max_instructions=100)
        actual=action(vector,-.8)
        np.testing.assert_allclose(actual,independent_term(block,-.8)@vector,atol=3e-13)
        np.testing.assert_allclose(np.linalg.norm(actual),np.linalg.norm(vector),atol=3e-13)


@pytest.mark.parametrize('K',[2,6])
@pytest.mark.parametrize('T',[.8,-.8])
def test_native_callback_connects_to_corrected_and_raw_finite_mean(K,T):
    from trotterlib.rte import finite_taylor_operator,finite_rte_distribution
    from trottertracks.resource_applicability.ax2a_state_action import partial_s2_signal,ActionBudget
    bs=blocks()[:2]
    actions=[native.make_native_block_action(b,max_instructions=100) for b in bs]
    psi=np.array([1,1j,.3,-.4j],dtype=complex);psi/=np.linalg.norm(psi)
    tail=np.diag([.1,-.3,.5,.8]);lam=.9;q=3;r=2
    result=partial_s2_signal(psi,actions,lambda v:tail@v,lambda_r=lam,T=T,q=q,r=r,K=K,
        phase_energy=.37,budget=ActionBudget(1000,1000))
    forward=np.eye(4,dtype=complex);reverse=forward.copy()
    for b in bs:forward=independent_term(b,T/(2*q))@forward
    for b in reversed(bs):reverse=independent_term(b,T/(2*q))@reverse
    P=finite_taylor_operator(tail,lam*T/(q*r),K)
    step=np.exp(-1j*.37*T/q)*reverse@np.linalg.matrix_power(P,r)@forward
    expected=np.vdot(psi,np.linalg.matrix_power(step,q)@psi)
    b=finite_rte_distribution(lam*T/(q*r),K).exact_finite_distribution
    np.testing.assert_allclose(result.corrected,expected,atol=5e-13)
    np.testing.assert_allclose(result.raw,expected/b**(q*r),atol=5e-13)


@pytest.mark.parametrize('axis',['cosine','sine'])
def test_legacy_wrapper_axis_and_measurement_scope(axis):
    evolution=native.build_deterministic_native(blocks(),num_system_qubits=2,T=.8,q=1,
        formula='4th',scalar=.37,control_policy='symmetric_directional',max_instructions=10000)
    wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=10000)
    psi=np.array([1,1j,.3,-.4j],dtype=complex);psi/=np.linalg.norm(psi)
    U=Operator(evolution.circuit).data[4:,4:]
    z=np.vdot(psi,U@psi)
    output=Statevector(np.concatenate((psi,np.zeros(4)))).evolve(wrapper).data
    signal=np.sum(abs(output[:4])**2)-np.sum(abs(output[4:])**2)
    np.testing.assert_allclose(signal,z.real if axis=='cosine' else z.imag,atol=4e-13)
    measured=native.build_native_hadamard_wrapper(evolution,axis=axis,max_instructions=10000)
    assert measured.num_clbits==1 and measured.data[-1].operation.name=='measure'
    assert measured.find_bit(measured.data[-1].qubits[0]).index==2
    assert measured.cregs[0].name=='rpe_measure'


def test_bound_is_checked_before_schedule_or_circuit_allocation(monkeypatch):
    bs=blocks()
    monkeypatch.setattr(native,'QuantumCircuit',lambda *a,**k:pytest.fail('Circuit allocated before preflight'))
    with pytest.raises(RuntimeError,match='PRE_BUILD'):
        native.build_deterministic_native(bs,num_system_qubits=2,T=.8,q=10**12,
            formula='4th',scalar=0,control_policy='symmetric_directional',max_instructions=100)


def test_malformed_register_phase_count_and_mutation_are_rejected():
    bs=blocks()
    with pytest.raises(ValueError,match='diagonal'):
        native.build_deterministic_native((replace(bs[0],diagonal_eigenvalues=(.2,)),),
            num_system_qubits=2,T=.8,q=1,formula='2nd',scalar=0,control_policy='ordinary',max_instructions=100)
    with pytest.raises(ValueError):native.make_native_block_action(replace(bs[1],lam=float('nan')),max_instructions=100)
    with pytest.raises((ValueError,TypeError)):
        native.build_deterministic_native(bs,num_system_qubits=2,T=.8,q=True,
            formula='2nd',scalar=0,control_policy='ordinary',max_instructions=1000)
    result=native.build_deterministic_native(bs,num_system_qubits=2,T=.8,q=1,
        formula='2nd',scalar=0,control_policy='ordinary',max_instructions=1000)
    result.circuit.x(0)
    with pytest.raises(ValueError,match='mutated'):
        native.build_native_hadamard_wrapper(result,axis='cosine',max_instructions=1000)


@pytest.mark.parametrize('ld',[0,1,2])
@pytest.mark.parametrize('policy',['ordinary','symmetric_directional'])
def test_existing_df_step_request_replay_matches_legacy_without_sampling(ld,policy):
    # Literal two-orbital fixture, never chemistry/integral/state generation.
    from trotterlib.df_hamiltonian import DFHamiltonian
    from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
    from trotterlib.df_partial_s2 import prepare_df_partial_s2, DFPartialS2StepRequest, QiskitDFPartialS2CircuitBuilder
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.rte import make_rte_config,enumerate_rte_events
    ham=DFHamiltonian(constant=.13,one_body=np.array([[.2,.03j],[-.03j,-.1]]),
        lambdas=np.array([.2,-.3]),g_matrices=(np.diag([1.,.4]),np.diag([.2,2.])),metadata={'synthetic':True})
    prep=prepare_df_partial_s2(ham,split_df_hamiltonian_by_ld(ham,ld))
    config=distribution=occurrence=None
    if not prep.is_deterministic_only:
        config,distribution=make_rte_config(prep.rte_preparation.symbolic_tail,evolution_time=-.17,
            rte_steps=2,truncation_tolerance=1.,finite_taylor_order=2)
        enumerated=enumerate_rte_events(prep.rte_preparation.symbolic_tail.components,distribution,max_events=10000)
        chosen=(next(e for e in enumerated if e.taylor_order==0),next(e for e in enumerated if e.taylor_order==2))
        occurrence=DFRTEEventSequenceCircuitRequest(events=chosen,
            component_specs=prep.rte_preparation.component_specs,controlled=True,ancilla_qubit=2,
            tail_id=config.tail_id,tail_hash=config.tail_hash,occurrence_rte_steps=2)
    request=DFPartialS2StepRequest(prep,-.17,config,distribution,occurrence,controlled=True,
        ancilla_qubit=2,seed=None if occurrence is None else 0)
    legacy=QiskitDFPartialS2CircuitBuilder().build_step(request).circuit
    native_result=native.partial_native_from_step_requests((request,request),
        control_policy=policy,max_instructions=10000)
    np.testing.assert_allclose(Operator(native_result.circuit).data,
        Operator(legacy).data@Operator(legacy).data,atol=3e-13)

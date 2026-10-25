"""Artificial local gates/state actions, phase and leakage; no saved H6 data."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator,Statevector
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h6_state_action as a


@pytest.mark.parametrize('phase',[0.,.37,-2.1])
def test_Aer_actions_match_independent_existing_Qiskit_with_nested_phase(phase):
    from trotterlib.df_trotter.circuit import simulate_statevector
    nested=QuantumCircuit(4);nested.global_phase=phase;nested.h(0);nested.cx(0,3)
    nested.crz(.29,2,1);nested.rzz(-.4,0,2)
    qc=QuantumCircuit(4);qc.global_phase=-.13;qc.append(nested.to_gate(),range(4));qc.sdg(2)
    rng=np.random.default_rng(97);states=rng.normal(size=(3,16))+1j*rng.normal(size=(3,16));states/=np.linalg.norm(states,axis=1)[:,None]
    # Statevector's nested-definition phase handling can modify its input view;
    # give the independent legacy helper copies, retaining the same input batch.
    pristine=states.copy()
    expected=np.asarray([simulate_statevector(qc,s.copy()) for s in states])
    actual=a.StateActions(1).apply(qc,states)
    np.testing.assert_array_equal(states,pristine)
    np.testing.assert_allclose(actual,expected,atol=3e-14,rtol=0)
    np.testing.assert_allclose(a.StateActions(1).apply(qc.inverse(),actual),states,atol=5e-14,rtol=0)


def test_only_local_matrices_constructed_even_for_large_composite():
    import qiskit.quantum_info as qi
    qc=QuantumCircuit(13);qc.global_phase=.19
    block=QuantumCircuit(13);block.h(0);block.cx(0,12);block.crz(.3,12,6);block.global_phase=.23
    qc.append(block.to_gate(),range(13))
    seen=[]
    def local(op):
        seen.append(op.num_qubits);assert op.num_qubits<=3
        return Operator(op)
    with patch.object(qi,'Operator',side_effect=local):lowered=a.simulation_circuit(qc)
    assert seen and max(seen)<=3 and lowered.num_qubits==13
    psi=np.zeros(8192,complex);psi[0]=1
    np.testing.assert_allclose(a.StateActions(1).apply(qc,psi[None,:])[0],Statevector(psi).evolve(qc).data,atol=1e-14,rtol=0)


@pytest.mark.parametrize('name',['reset','initialize','measure'])
def test_nonunitary_or_preparation_is_rejected(name):
    qc=QuantumCircuit(2,1 if name=='measure' else 0)
    if name=='reset':qc.reset(0)
    elif name=='initialize':qc.initialize([1.,0.],0)
    else:qc.measure(1,0)
    with pytest.raises(ValueError):a.simulation_circuit(qc)


def test_control_phase_and_out_of_sector_leakage_are_not_phase_fitted_or_projected():
    psi=np.zeros((1,8),complex);psi[0,0]=psi[0,4]=1/np.sqrt(2)
    phase=psi.copy();phase[0,4]*=np.exp(.01j)
    with pytest.raises(ValueError):a.state_gate(phase,psi,label='controlled relative phase')
    leakage=psi.copy();leakage[0,1]=.001;leakage/=np.linalg.norm(leakage)
    with pytest.raises(ValueError):a.state_gate(leakage,psi,label='leakage retained')


def test_terminal_ancilla_readout_policy_and_XY_sign():
    from trottertracks.c1_stage3_connection import readout_segments
    theta=.43;psi=np.array([[1.,np.exp(1j*theta)]])/np.sqrt(2)
    engine=a.StateActions(1);values={}
    # one idle system qubit; ancilla last. The helper expects modes>=1.
    initial=np.array([[1.,0.,np.exp(1j*theta),0.]])/np.sqrt(2)
    for axis in ('X','Y'):
        _,tail=readout_segments(1,axis)
        values[axis]=a.ancilla_z(engine.apply(a.measured_tail(tail),initial))[0]
    assert values['X']==pytest.approx(np.cos(theta),abs=1e-14)
    assert values['Y']==pytest.approx(np.sin(theta),abs=1e-14)
    qc=QuantumCircuit(2,1);qc.measure(0,0)
    with pytest.raises(ValueError,match='ANCILLA'):a.measured_tail(qc)


@pytest.mark.parametrize('angle',[.37,np.pi,np.pi-1e-12])
@pytest.mark.parametrize('improper',[False,True])
def test_number_preserving_native_gaussian_action_matches_small_full_operator(angle,improper):
    from trottertracks.pr_dictionary_validation.circuits import native_gaussian
    U=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],complex)
    if improper:U[:,0]*=-1
    qc=native_gaussian(U)
    rng=np.random.default_rng(73);states=rng.normal(size=(2,4))+1j*rng.normal(size=(2,4));states/=np.linalg.norm(states,axis=1)[:,None]
    expected=np.asarray([Operator(qc).data@x for x in states])
    np.testing.assert_allclose(a.StateActions(1).apply(qc,states),expected,atol=1e-14,rtol=0)

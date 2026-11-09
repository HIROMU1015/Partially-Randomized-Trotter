"""Meaningful exact-action checks on artificial inputs only."""
import sys
from pathlib import Path
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import XXPlusYYGate,RZZGate
from qiskit.quantum_info import Statevector
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h8_full_h_pure_rte as study
import h8_full_h_parallel_aer as fast


@pytest.mark.parametrize('sign',[1,-1])
def test_exact_local_matrices_controls_phase_and_adjoint(sign):
    study.install()  # metadata/source imports only; no molecular loader call
    rng=np.random.default_rng(700);v=rng.normal(size=32)+1j*rng.normal(size=32);v/=np.linalg.norm(v)
    qc=QuantumCircuit(5);qc.global_phase=.317
    qc.h(4);qc.append(XXPlusYYGate(sign*.27,.13),[3,0])
    qc.append(RZZGate(sign*.41).control(ctrl_state=0),[4,2,0])
    qc.append(RZZGate(-sign*.41).control(),[4,2,0]);qc.sdg(4)
    qc.append(XXPlusYYGate(-sign*.17,-.29),[1,2]);qc.h(4)
    baseline=Statevector(v.copy()).evolve(qc).data
    original=qc.copy();source=v.copy();fast.configure(4)
    actual=fast.simulate_statevector(qc,v)
    assert np.linalg.norm(actual-baseline)<1e-13
    assert np.linalg.norm(fast.simulate_statevector(qc.inverse(),actual)-v)<1e-13
    assert qc==original and np.array_equal(v,source)


def test_mutated_gate_definition_is_not_reused():
    study.install();fast.configure(4)
    first=Gate('custom_action',2,[]);definition=QuantumCircuit(2)
    definition.append(XXPlusYYGate(.31,.12),[0,1]);first.definition=definition
    qc=QuantumCircuit(2);qc.append(first,[0,1]);v=np.array([1,2j,3,4j],complex);v/=np.linalg.norm(v)
    initial=fast.simulate_statevector(qc,v)
    altered=QuantumCircuit(2);altered.x(0);qc.data[0].operation.definition=altered
    expected=Statevector(v).evolve(qc).data
    actual=fast.simulate_statevector(qc,v)
    assert np.linalg.norm(actual-expected)<1e-13
    assert np.linalg.norm(actual-initial)>.1

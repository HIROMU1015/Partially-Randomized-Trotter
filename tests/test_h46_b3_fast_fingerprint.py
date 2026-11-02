"""Exact canonical hashes, including changed parameters and custom definitions."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_b3 as b
b.setup()
import h46_b3_fast_fingerprint as fast
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate,UnitaryGate,XXPlusYYGate


@pytest.mark.parametrize('seed',range(4))
def test_standard_and_controlled_gates_emit_identical_bytes(seed):
    rng=np.random.default_rng(seed);qc=QuantumCircuit(5,1)
    for i in range(70):
        q=i%5;r=(i+1)%5
        qc.rz(float(rng.choice([0.,-0.,.1,-.3,np.pi])),q)
        qc.ry(.6,q);qc.rx(-.2,r);qc.cx(q,r);qc.h(q);qc.sx(r)
        qc.append(XXPlusYYGate(.3,-.5),[q,r]);qc.rxx(.13,q,r);qc.ryy(-.7,q,r);qc.rzz(.8,q,r)
        if i%9==0:qc.crz(.4,q,r)
    qc.global_phase=-.17;qc.measure(4,0)
    assert fast.FingerprintSession().fingerprint(qc)==fast.legacy.canonical_qiskit_circuit_fingerprint(qc)


def test_explicit_mutable_definition_is_never_replaced_by_plain_gate_cache():
    qc=QuantumCircuit(1);qc.rz(.3,0)
    changed=RZGate(.3);definition=QuantumCircuit(1);definition.rx(.9,0);changed.definition=definition
    qc.append(changed,[0]);qc.rz(.3,0)
    assert fast.FingerprintSession().fingerprint(qc)==fast.legacy.canonical_qiskit_circuit_fingerprint(qc)


def test_fresh_calls_observe_mutations_to_parameters_definitions_and_phase():
    session=fast.FingerprintSession();qc=QuantumCircuit(2)
    gate=RZGate(.2);qc.append(gate,[0]);qc.cx(0,1)
    old=session.fingerprint(qc)
    qc.global_phase=.12
    assert session.fingerprint(qc)!=old
    assert session.fingerprint(qc)==fast.legacy.canonical_qiskit_circuit_fingerprint(qc)
    custom=QuantumCircuit(1);custom.h(0);g=custom.to_gate();qc.append(g,[1])
    old=session.fingerprint(qc)
    qc.data[-1].operation.definition.x(0)
    assert session.fingerprint(qc)!=old
    assert session.fingerprint(qc)==fast.legacy.canonical_qiskit_circuit_fingerprint(qc)
    qc.data[0]=qc.data[0].replace(operation=RZGate(-.2))
    assert session.fingerprint(qc)==fast.legacy.canonical_qiskit_circuit_fingerprint(qc)


def test_custom_arrays_and_classical_conditions_use_recursive_writer():
    qc=QuantumCircuit(2,1);qc.append(UnitaryGate(np.eye(2,dtype=complex)),[0])
    qc.x(1).c_if(qc.clbits[0],True);qc.barrier();qc.measure(0,0)
    assert fast.FingerprintSession().fingerprint(qc)==fast.legacy.canonical_qiskit_circuit_fingerprint(qc)


def test_nonfinite_parameters_are_rejected():
    qc=QuantumCircuit(1);qc.rz(float('nan'),0)
    with pytest.raises(ValueError,match='finite'):fast.FingerprintSession().fingerprint(qc)

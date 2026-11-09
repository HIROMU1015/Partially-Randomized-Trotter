"""Exercise the actual native custom gate families in the action adapter."""
import sys
from pathlib import Path
import numpy as np
import pytest
import threading
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from h46_full_h_t20_aer import simulate_statevector


@pytest.mark.parametrize('phase',[0.,.17,-.19])
def test_controlled_rzz_gaussian_and_global_phase(phase):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate,RZZGate
    from qiskit.quantum_info import Statevector
    qc=QuantumCircuit(4);qc.global_phase=phase
    qc.append(XXPlusYYGate(.21,.37),(0,1))
    qc.append(RZZGate(-.41).control(),(3,0,2))
    qc.p(.17,3);qc.append(XXPlusYYGate(.21,.37).inverse(),(0,1))
    v=np.arange(1,17,dtype=complex)+1j*np.arange(16);v/=np.linalg.norm(v)
    original=v.copy()
    expected=np.asarray(Statevector(v.copy()).evolve(qc).data).copy()
    actual=simulate_statevector(qc,v)
    assert threading.active_count()==1
    assert np.array_equal(v,original)
    assert float(qc.global_phase)==phase % (2*np.pi)
    assert np.linalg.norm(actual-expected)<2e-13

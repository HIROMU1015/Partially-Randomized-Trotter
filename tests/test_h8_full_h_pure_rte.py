"""Focused local-action checks; no molecular input or scientific run."""
import sys
from pathlib import Path
import numpy as np
import pytest
from qiskit.quantum_info import Statevector,Operator

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from h8_full_h_pure_rte import tensor_action
from h8_full_h_signal import polynomial
import math


@pytest.mark.parametrize('qubits',[(0,),(3,1),(4,1,0),(3,2,1,0)])
def test_vectorized_little_endian_action(qubits):
    rng=np.random.default_rng(828)
    state=rng.normal(size=32)+1j*rng.normal(size=32)
    state/=np.linalg.norm(state);original=state.copy()
    n=2**len(qubits);matrix=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    matrix=np.linalg.qr(matrix)[0]
    expected=Statevector(state.copy()).evolve(Operator(matrix),qargs=list(qubits)).data
    actual=tensor_action(state,matrix,qubits)
    assert np.linalg.norm(actual-expected)<1e-14
    assert np.array_equal(state,original)


@pytest.mark.parametrize('K',[0,2,4])
def test_scalar_finite_paired_numerator(K):
    energy=-3.2;dt=.03;b=1.2
    expected=sum((-1j*dt*energy)**j/math.factorial(j) for j in range(K+2))
    for horner in (False,True):
        assert abs(polynomial(energy,dt,K,b,horner=horner)-expected)<1e-15
        assert abs(polynomial(energy,dt,K,b,raw=True,horner=horner)-expected/b)<1e-15

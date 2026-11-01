"""Hand-specified Givens regressions, no molecular input."""
from pathlib import Path
import sys
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_h4_p3_stage3_stable_basis as runner
K=runner.modules()
from stage1.stable_gaussian_basis import stable_givens_ops,exact_zero_givens
from stage1.df_trotter.ops import U_to_qiskit_ops_jw_givens as original_ops


def matrix(ops,n):
    circuit=QuantumCircuit(n)
    for gate,qubits in ops:circuit.append(gate,qubits)
    m=Operator(circuit).data
    idx=[1<<i for i in range(n)]
    return m[np.ix_(idx,idx)]/m[0,0]


@pytest.mark.parametrize('angle',[2e-13,3e-9,2e-8,.17,-.31])
def test_small_and_ordinary_rotations_keep_nonzero_sine(angle):
    n=8;c,s=np.cos(angle),np.sin(angle)
    u=np.eye(n,dtype=complex);u[:2,:2]=[[c,-s],[s,c]]
    actual=matrix(stable_givens_ops(u),n)
    np.testing.assert_allclose(actual,u,atol=2e-15,rtol=0)
    if abs(angle)<1e-8:
        assert np.max(np.abs(matrix(original_ops(u),n)-u))>abs(angle)/2


def test_givens_preserves_tiny_complex_entry_and_cancellation():
    a=1.+0j;b=2e-13*(1+1j)
    g=exact_zero_givens(a,b)
    assert g[0,1]!=0
    np.testing.assert_allclose(g@np.array([a,b]),[np.hypot(abs(a),abs(b)),0.],atol=1e-27,rtol=1e-15)


def test_eight_mode_complex_reflection_and_mixed_rotations():
    u=np.eye(8,dtype=complex)
    for i,theta,phase in [(0,.31,.2),(2,-.24,-.4),(5,4e-12,.7),(1,.1,-.6)]:
        block=np.array([[np.cos(theta),-np.exp(1j*phase)*np.sin(theta)],
                        [np.exp(-1j*phase)*np.sin(theta),np.cos(theta)]])
        u[[i,i+1],:]=block@u[[i,i+1],:]
    u[7,:]*=-1
    basis=K.basis_from_saved(u)
    assert all(x['status']=='PASS' for x in basis.checks.values())
    assert 'atan2' in basis.metadata['native_family']


def test_original_shared_lowering_function_is_not_modified():
    assert original_ops.__module__=='stage1.df_trotter.ops'
    assert K.U_to_qiskit_ops_jw_givens is stable_givens_ops
    assert runner.original.BASE=='be3de04e6fb501fecc4a07f38ceea7c24ba10472'

"""Artificial precision regressions; no saved molecular array is decoded."""
import itertools
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
from qiskit.quantum_info import Operator
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h6_native_precision as p
from trottertracks.pr_dictionary_validation.circuits import native_gaussian


@pytest.mark.parametrize('theta',[1e-12,1e-10,1e-9,1e-8,1e-7,np.pi-1e-12,np.pi])
@pytest.mark.parametrize('sign',[1.,-1.])
def test_tiny_rotations_and_pi_retained_without_relaxing_gate(theta,sign):
    u=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],complex)
    u[:,0]*=sign
    qc,report=p.gaussian(u);mat=Operator(qc).data
    np.testing.assert_allclose(mat[np.ix_([1,2],[1,2])]/mat[0,0],u,atol=8e-16,rtol=0)
    assert report['single_particle_max_entry']<8e-16
    assert report['allowance']==1e-10


def test_legacy_small_rotation_loss_is_reproduced_and_repaired():
    t=1e-9;u=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    old=Operator(native_gaussian(u)).data
    new,report=p.gaussian(u)
    assert np.max(abs(old[np.ix_([1,2],[1,2])]/old[0,0]-u))>9e-10
    assert report['single_particle_max_entry']<1e-20
    assert new.count_ops().get('xx_plus_yy',0)==1


@pytest.mark.parametrize('n',[2,6])
@pytest.mark.parametrize('complex_input',[False,True])
def test_full_artificial_fock_action_is_exterior_power(n,complex_input):
    rng=np.random.default_rng(29+n);x=rng.normal(size=(n,n))
    if complex_input:x=x+1j*rng.normal(size=(n,n))
    u=np.linalg.qr(x)[0]
    qc,report=p.gaussian(u);actual=Operator(qc).data
    expected=np.zeros_like(actual);D=1<<n
    for column in range(D):
        source=[i for i in range(n) if column>>i&1]
        for target in itertools.combinations(range(n),len(source)):
            row=sum(1<<i for i in target)
            expected[row,column]=np.linalg.det(u[np.ix_(target,source)]) if source else 1.
    np.testing.assert_allclose(actual/actual[0,0],expected,atol=4e-15,rtol=0)
    assert report['full_operator_norm_bound'] is None


def test_twelve_modes_never_build_large_operator():
    rng=np.random.default_rng(128);u=np.linalg.qr(rng.normal(size=(12,12)))[0]
    import qiskit.quantum_info as qi
    original=qi.Operator;seen=[]
    def small(gate):
        seen.append(gate.num_qubits);assert gate.num_qubits<=2
        return original(gate)
    with patch.object(qi,'Operator',side_effect=small):qc,report=p.gaussian(u)
    assert seen and qc.num_qubits==12 and report['single_particle_max_entry']<2e-15


@pytest.mark.parametrize('value',[np.ones((2,3)),np.diag([1.,.99]),np.array([[np.nan]])])
def test_invalid_input_fails_without_fallback(value):
    with pytest.raises(ValueError):p.gaussian(value)


def test_transfer_mismatch_fails_explicitly():
    from qiskit.circuit.library import UnitaryGate
    with patch('qiskit.circuit.library.XXPlusYYGate',return_value=UnitaryGate(np.eye(4))):
        with pytest.raises(ValueError,match='GAUSSIAN_TRANSFER_MISMATCH'):
            p.gaussian([[.8,-.6],[.6,.8]])

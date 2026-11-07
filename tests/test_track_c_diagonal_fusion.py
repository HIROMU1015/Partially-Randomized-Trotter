import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.quantum_info import Operator
from trottertracks.track_c_diagonal_fusion import emit


def test_signed_tiny_and_repeated_controlled_polynomials():
    rotations=[((0,),-.2),((0,2),1e-12),((0,2),-.031),((1,2),.07),((0,),.2),((3,),-3.3)]
    products=[(0,1),(1,2),(0,1)];old=QuantumCircuit(9)
    for s,a in rotations:
        if len(s)==1:old.crz(a,8,s[0])
        else:old.append(RZZGate(a).control(),[8,*s])
    for s in products:
        for q in s:old.cz(8,q)
    old.p(np.pi,8);target=Operator(old).data
    for mode in ('term_parity','merge','network'):
        q=QuantumCircuit(9);emit(q,rotations,products,mode=mode);q.p(np.pi,8)
        assert np.linalg.norm(target-Operator(q).data,'fro')<5e-13
        assert np.linalg.norm(Operator(q).data[:256,:256]-np.eye(256))<5e-13


def test_network_retains_negative_phase_without_numeric_modulo():
    q=QuantumCircuit(9);emit(q,[((0,),-1e-9),((0,1),-.4)],mode='network')
    states=np.arange(512);z0=1-2*(states&1);z1=1-2*((states>>1)&1)
    expected=np.where(states<256,1.,np.exp(-.5j*(-1e-9*z0-.4*z0*z1)))
    assert np.linalg.norm(Operator(q).data-np.diag(expected),'fro')<5e-13


def test_controlled_two_pi_rotation_retains_relative_minus_sign():
    for mode in ('term_parity','merge','network'):
        q=QuantumCircuit(9);emit(q,[((0,),2*np.pi)],mode=mode)
        target=np.diag(np.r_[np.ones(256),-np.ones(256)])
        assert np.linalg.norm(Operator(q).data-target,'fro')<5e-13

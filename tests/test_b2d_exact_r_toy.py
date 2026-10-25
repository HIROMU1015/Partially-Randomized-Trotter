"""Numerical tests ONLY on a different complex toy, never the fixed fixture.

No transpilation, pricing, real candidate input loader or saved result analysis.
"""
from fractions import Fraction as F
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_s2 import QiskitDFPartialS2CircuitBuilder
from trottertracks.b2d_adapter.adapter import DContext
from trottertracks.b2d_evaluator import exact_r as e


@pytest.fixture
def toy(monkeypatch):
    # Deliberately different from _nontrivial_basis_case, including complex g.
    g = np.array([[.31, .09+.14j], [.09-.14j, -.47]], dtype=complex)
    h = DFHamiltonian(constant=-.29, one_body=np.zeros((2,2),complex),
        lambdas=np.array([.13,.43,-.37]),
        g_matrices=(np.diag([.2,.6]),np.diag([-.1,.3]),g), metadata={'name':'INDEPENDENT_TOY_ONLY'})
    n = object.__new__(e.Native)
    n.h = h; n.prep = NS(randomized_block_indices=(2,), extracted_identity_coefficient=.081)
    n.tail_block = n.prepare_tail()
    n.context = DContext('TOY_PREPARATION','TOY_TAIL',('toy-a','toy-b','toy-c'),2)
    n.block_builder = QiskitDFPartialS2CircuitBuilder()
    n.QC = QuantumCircuit; n.cache = {}; n.compile_cache = {}
    n.np = np; n.eye = np.eye(4,dtype=complex); n.Operator = Operator; n.scalar = .19
    n.call = lambda kind,key,fn: fn()
    # Independent CAR with no Gaussian eigenvectors, primitives or sector shortcut.
    ann = []
    for p in range(2):
        a = np.zeros((4,4),complex)
        for col in range(4):
            if col & (1<<p): a[col^(1<<p),col] = (-1)**((col&((1<<p)-1)).bit_count())
        ann.append(a)
    q = sum((g[i,j]*ann[i].conj().T@ann[j] for i in range(2) for j in range(2)), np.zeros((4,4),complex))
    r = -.37*(q@q)-.081*np.eye(4)
    z = np.diag([1.,-1.]); x = np.array([[0,1],[1,0]])
    n.matrices = {'D_0':.17*np.kron(z,np.eye(2)), 'D_1':.23*np.kron(np.eye(2),x),
                  'D_2':.11*np.kron(x,z), 'R':r}
    n.exp = lambda label,t: expm(-1j*float(t)*n.matrices[label])
    def block(label,t,controlled):
        u = n.exp(label,t)
        qc = QuantumCircuit(3 if controlled else 2)
        qc.append(UnitaryGate(n.controlled(u) if controlled else u), list(range(qc.num_qubits)))
        return qc
    n.block = block
    # A numerical toy test must never drift into the native compiler.
    monkeypatch.setattr('qiskit.transpile', lambda *a,**k: pytest.fail('no native transpile in toy tests'))
    monkeypatch.setattr('trotterlib.rte.make_rte_config', lambda *a,**k: pytest.fail('Exact-R has no RTE config'))
    monkeypatch.setattr(e.Native, 'rte', lambda *a,**k: pytest.fail('Exact-R has no RTE event circuit'))
    n.compile = lambda *a,**k: pytest.fail('no compiler in toy tests')
    return n


@pytest.mark.parametrize('time',[F(137,1000),F(-137,1000)])
@pytest.mark.parametrize('controlled',[False,True])
def test_complex_basis_squared_primitive_signed_centered_control(toy,time,controlled):
    actual = Operator(toy.tail(time,controlled)).data
    reference = toy.exp('R',time)
    expected = toy.controlled(reference) if controlled else reference
    assert np.linalg.norm(actual-expected,ord=2) < 1e-12
    if controlled:
        assert np.linalg.norm(actual[:4,:4]-np.eye(4),ord=2) < 1e-12
        assert np.linalg.norm(actual[:4,4:],ord=2) < 1e-12
    # Omitting or flipping the identity correction would be detectable here.
    bad = np.exp(-2j*float(time)*.081)*reference
    assert np.linalg.norm(bad-reference,ord=2) > 1e-5


def test_basis_forward_inverse_uses_actual_complex_adjoint(toy):
    forward = QuantumCircuit(2); inverse = QuantumCircuit(2)
    toy.block_builder._append_basis(forward,toy.tail_block,inverse=False)
    toy.block_builder._append_basis(inverse,toy.tail_block,inverse=True)
    a,b = Operator(forward).data, Operator(inverse).data
    assert np.linalg.norm(b-a.conj().T,ord=2) < 1e-12
    assert np.linalg.norm(a@b-np.eye(4),ord=2) < 1e-12
    assert np.linalg.norm(a.imag) > 1e-3


@pytest.mark.parametrize('index',range(4))
def test_toy_full_order_scalar_once_and_prepared_XY(toy,index):
    row = e.cases()[index]  # Fixed schedule only; H/R/D inputs above are all toy.
    ref = toy.exact_reference(row)
    for controlled in (False,True):
        actual = Operator(toy.render_exact(row,controlled)).data
        expected = toy.controlled(ref) if controlled else ref
        assert np.linalg.norm(actual-expected,ord=2) < 1e-12
    psi = np.array([0.,1.,0.,0.],complex)
    signal = np.vdot(psi,ref@psi)
    for axis,expected in (('cosine',signal.real),('sine',signal.imag)):
        wrapped = toy.wrapper(toy.render_exact(row,True),axis)
        unitary = Operator(wrapped.remove_final_measurements(inplace=False)).data
        independent = toy.wrapper_reference(ref,axis)
        assert np.linalg.norm(unitary-independent,ord=2) < 1e-12
        vec = unitary[:,0]
        actual = np.sum(abs(vec[:4])**2)-np.sum(abs(vec[4:])**2)
        assert abs(actual-expected) < 1e-12
    wrong_double_scalar = np.exp(-1j*float(row['case']['total_time'])*toy.scalar)*ref
    assert np.linalg.norm(wrong_double_scalar-ref,ord=2) > 1e-3

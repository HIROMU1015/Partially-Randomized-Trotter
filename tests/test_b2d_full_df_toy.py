"""Independent toy checks, with AX-2B toy helpers reused unchanged.

Helpers:31cdff47f3282d2898c153a1af22f90e500be4c6,
tests/tracks/resource_applicability/test_ax2b_native_df_v5.py.
No real fixture, saved results, transpilation, metrics, sampling or molecular IO.
"""
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec, DFDeterministicFragmentSpec
from trottertracks.b2d_evaluator import full_df as f
from trottertracks.b2d_evaluator import _full_df_native as provider


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


@pytest.fixture
def toy(monkeypatch):
    def forbidden(*a,**k): pytest.fail('No fixture IO, sampling, compile or pricing in toy tests')
    monkeypatch.setattr('qiskit.transpile',forbidden)
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr('trotterlib.rte.make_rte_config',forbidden)
    monkeypatch.setattr('trotterlib.rte_compiled_cost.transpile',forbidden)
    n = object.__new__(f.Native)
    bs = blocks()
    n.full_blocks = bs+(replace(bs[2],block_id='toy_df3',original_fragment_index=2,rank=2,
        diagonal_eta=(.32,-.51),lam=-.29,order_index=3,runtime_basis_operations=basis(.39,-.26)),)
    n.QC,n.Operator,n.provider,n.np = QuantumCircuit,Operator,provider,np
    n.eye = np.eye(4,dtype=complex);n.state = np.array([0,1,0,0],complex)
    n.full_scalar = .127;n.cache = {};n.compile_cache = {}
    n.call = lambda kind,key,fn:fn()
    n.matrices = {}
    occupations = np.array([[(i>>j)&1 for j in range(2)] for i in range(4)])
    for key,b in zip(f.LABELS,n.full_blocks):
        energies = (occupations@np.asarray(b.diagonal_eigenvalues) if b.block_kind=='one_body'
                    else b.lam*(occupations@np.asarray(b.diagonal_eta))**2)
        circuit = QuantumCircuit(2)
        for gate,indices in b.runtime_basis_operations:circuit.append(gate,list(indices))
        u = Operator(circuit).data
        n.matrices[key] = u@np.diag(energies)@u.conj().T
    n.matrices['physical'] = n.full_scalar*n.eye+sum((n.matrices[k] for k in f.LABELS),np.zeros((4,4),complex))
    n.h = NS(constant=.127,lambdas=np.array([-.37,.23,-.29]))
    n.exp = lambda label,t:expm(-1j*float(t)*n.matrices[label])
    return n


@pytest.mark.parametrize('index',range(4))
def test_existing_full_builders_match_independent_Fock_product_and_XY(toy,index):
    row = f.cases()[index]  # Only schedule metadata is fixed; every block above is an independent toy.
    system = toy.system(row);controlled = toy.controlled_system(row)
    ref = toy.full_reference(row)
    assert np.linalg.norm(Operator(system).data-ref,ord=2) < 1e-12
    assert np.linalg.norm(Operator(controlled).data-ordinary(ref),ord=2) < 1e-12
    assert np.linalg.norm(Operator(controlled).data[:4,:4]-np.eye(4),ord=2) < 1e-12
    # Independently compose the old toy oracle, without either stage generator.
    w1 = 1/(2-2**(1/3));w0 = -2**(1/3)/(2-2**(1/3))
    pieces = (1.,) if row['formula']=='2nd' else (w1,w0,w1)
    expected = np.eye(4,dtype=complex)
    for _ in range(2):
        for w in pieces:expected = independent_s2(toy.full_blocks,float(row['total_time'])*w/2)@expected
    expected *= np.exp(-1j*float(row['total_time'])*.127)
    assert np.linalg.norm(ref-expected,ord=2) < 1e-12
    z = np.vdot(toy.state,ref@toy.state)
    for axis,part in (('cosine',z.real),('sine',z.imag)):
        qc = toy.wrapper(controlled,axis).remove_final_measurements(inplace=False)
        u = Operator(qc).data;wr = toy.wrapper_reference(ref,axis)
        assert np.linalg.norm(u-wr,ord=2) < 1e-12
        v = u[:,0]
        assert abs(np.sum(abs(v[:4])**2)-np.sum(abs(v[4:])**2)-part) < 1e-12
    result = toy.signals(row,ref)
    assert result['normalization']==1 and result['u_num'] is None


@pytest.mark.parametrize('time',[.137,-.137])
@pytest.mark.parametrize('index',range(4))
def test_all_four_blocks_signed_control_and_actual_inverse(toy,time,index):
    b = toy.full_blocks[index]
    for controlled in (False,True):
        actual = Operator(toy.block_probe(index,time,controlled)).data
        expected = independent_term(b,time)
        assert np.linalg.norm(actual-(ordinary(expected) if controlled else expected),ord=2) < 1e-12
    circuit = QuantumCircuit(2)
    from trotterlib.df_partial_s2 import QiskitDFPartialS2CircuitBuilder as Builder
    Builder._append_basis(circuit,b,inverse=True)
    Builder._append_basis(circuit,b,inverse=False)
    assert np.linalg.norm(Operator(circuit).data-np.eye(4),ord=2) < 1e-12


def test_full_definition_rejects_split_scalar_and_missing_fragment(toy):
    assert toy.definition_gate()['physical_identity_error'] < 1e-12
    toy.full_scalar += .081  # A split-extracted identity must not be added again.
    with pytest.raises(f.Stage1Failure):toy.definition_gate()
    toy.full_scalar -= .081;toy.full_blocks = toy.full_blocks[:-1]
    with pytest.raises(f.Stage1Failure):toy.definition_gate()


def test_instruction_cap_and_ordinary_control_are_explicit(toy):
    row = f.cases()[2]
    with pytest.raises(RuntimeError,match='PRE_BUILD_INSTRUCTION_BUDGET'):
        provider.build_deterministic_native(toy.full_blocks,num_system_qubits=2,
            T=float(row['total_time']),q=2,formula='4th',scalar=.127,
            control_policy='ordinary',max_instructions=1)
    with pytest.raises(f.Stage1Failure):toy.controlled_system({**row,'control_policy':'symmetric_directional'})

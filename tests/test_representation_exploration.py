from __future__ import annotations

import math
import numpy as np
import pytest
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from trottertracks.representation_exploration.mechanisms import (
    I, X, Z, CircuitAudit, conditional_mixed, controlled, lie_dimension,
    reflected_dictionary, rotate_factors, rotation,
    second_quantize_one_body, vacuum_reflection,
)
from trotterlib.df_rte_tail import dense_df_block_hamiltonian
from trotterlib.df_trotter.model import DFModel
from trotterlib.df_trotter.ops import build_df_blocks_givens
from trotterlib.rte import (InvolutoryTailTerm, enumerate_rte_events,
    exact_enumerated_event_mean_operator, finite_rte_distribution,
    finite_taylor_operator, normalize_involutory_tail)


@pytest.mark.parametrize("theta",[0.,.17,-.4,math.pi/4])
def test_noncommuting_square_identity_and_fock_space(theta):
    factors=[I+X,I+Z]
    assert np.linalg.norm(factors[0]@factors[1]-factors[1]@factors[0])>1
    mixed=rotate_factors(factors,rotation(theta))
    original=sum(second_quantize_one_body(f)@second_quantize_one_body(f) for f in factors)
    actual=sum(second_quantize_one_body(f)@second_quantize_one_body(f) for f in mixed)
    np.testing.assert_allclose(actual,original,atol=1e-12,rtol=0)


@pytest.mark.parametrize("o",[np.array([[1.,.1],[0.,1.]]),np.array([[1j,0],[0,1]])])
def test_reject_nonorthogonal_or_complex_factor_mixing(o):
    with pytest.raises(ValueError):
        rotate_factors([X,Z],o)


def test_independent_jw_reference_matches_existing_df_builder():
    gs=[np.array([[.9,.2],[.2,-.3]]),np.diag([1.,-.4,.7])]
    for g in gs:
        n=len(g)
        model=DFModel(np.ones(1),[g],np.zeros((n,n)),0.,n)
        block=build_df_blocks_givens(model)[0]
        reference=second_quantize_one_body(g)
        np.testing.assert_allclose(dense_df_block_hamiltonian(block),reference@reference,atol=1e-12,rtol=0)


@pytest.mark.parametrize("k",[0,2,4])
@pytest.mark.parametrize("t",[.15,-.15])
def test_generator_symmetry_mean_does_not_equal_whole_word_twirl(k,t):
    p=np.diag([1.,0.]).astype(complex)
    s=2*p-I
    terms=[InvolutoryTailTerm("x",.4,X),InvolutoryTailTerm("z",.7,Z)]
    tail=normalize_involutory_tail("reflected",reflected_dictionary(terms,s))
    hbar=.7*Z
    dist=finite_rte_distribution(t*tail.lambda_r,k)
    events=enumerate_rte_events(tail.components,dist,max_events=2000)
    ops={c.component_id:o for c,o in zip(tail.components,tail.operators)}
    corrected=dist.exact_finite_distribution*exact_enumerated_event_mean_operator(events,ops)
    expected=finite_taylor_operator(hbar/tail.lambda_r,t*tail.lambda_r,k)
    np.testing.assert_allclose(corrected,expected,atol=2e-12,rtol=0)
    assert np.linalg.norm((I-p)@corrected@p)<2e-12
    if k:
        wrong=finite_taylor_operator((.4*X+.7*Z)/tail.lambda_r,t*tail.lambda_r,k)
        assert np.linalg.norm((wrong+s@wrong@s)/2-expected)>1e-4


@pytest.mark.parametrize("n",[1,2,3])
def test_vacuum_reflection_global_phase_is_faithful(n):
    expected=-np.eye(2**n,dtype=complex)
    expected[0,0]=1
    np.testing.assert_allclose(Operator(vacuum_reflection(n)).data,expected,atol=1e-12,rtol=0)
    actual=controlled(Operator(vacuum_reflection(n)).data)
    np.testing.assert_allclose(actual[0::2,0::2],np.eye(2**n),atol=1e-12,rtol=0)
    np.testing.assert_allclose(actual[1::2,1::2],expected,atol=1e-12,rtol=0)


@pytest.mark.parametrize("target",[0,1])
@pytest.mark.parametrize("t",[.2,-.2])
def test_conditional_su2_mixed_primitive(target,t):
    a=np.kron(I,Z)+.7*np.kron(Z,I)+.3*np.kron(Z,Z)
    b=[np.kron(I,X),np.kron(X,I)][target]
    actual=Operator(conditional_mixed(t,.1,target)).data
    np.testing.assert_allclose(actual,expm(-1j*t*(a+.1*b)),atol=1e-12,rtol=0)


def test_known_lie_algebras_and_generic_two_qubit_failure_control():
    assert lie_dimension([Z,X])==3
    assert lie_dimension([np.kron(I,Z),np.kron(Z,I)])==2
    a=np.kron(I,Z)+.7*np.kron(Z,I)+.3*np.kron(Z,Z)
    assert lie_dimension([a,np.kron(I,X),np.kron(X,I)])==15


def test_small_toy_compiler_phase_certificate_remains_control_sensitive():
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(2)
    qc.h(0);qc.crx(.36,0,1);qc.h(0)
    reference=Operator(qc).data
    record=CircuitAudit().compile(qc,reference)
    assert record["certified_scalar_residual"]<1e-12
    assert record["compiled_operator_residual"]<1e-12
    assert record["control_sensitive_repaired_residual"]<1e-12


def test_compile_certificate_does_not_hide_wrong_controlled_branch():
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(2)
    qc.crz(.3,0,1)
    wrong=controlled(expm(-.4j*Z))
    with pytest.raises(AssertionError):
        CircuitAudit().compile(qc,wrong)

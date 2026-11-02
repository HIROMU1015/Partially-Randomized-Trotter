"""Adversarial mathematics/phase/workspace checks on synthetic inputs only."""
import itertools
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from trottertracks.representation_exploration.algorithm_design import (
    Audit, active_certificate, charge_circuit, charges, density_bound, embed, n1_input,
    fock_frame, gaussian_circuit, increment, n2_input, ordered_frame, sector_indices,
    sector_norm, spectral_candidates, square_bound, subspace_frame, second_quantize_one_body,
)


@pytest.mark.parametrize('nu',range(5))
def test_sector_norm_signed_and_square_bound(nu):
    eta=np.array([-.8,.2,1.1,-.37]);et=np.array([-.7,.15,1.05,-.37])
    sums=[sum(eta[list(c)]) for c in itertools.combinations(range(4),nu)]
    assert sector_norm(eta,nu)==pytest.approx(max(map(abs,sums)))
    actual=max(abs(sum(eta[list(c)])**2-sum(et[list(c)])**2)
               for c in itertools.combinations(range(4),nu))
    assert .7*actual<=square_bound(eta,et,nu,-.7)+1e-14


def test_complex_degenerate_gauge_and_absolute_gaussian():
    rng=np.random.default_rng(76)
    u=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
    groups=[[0,1],[2,3]];eta=np.array([.7,.7,-.2,-.2])
    frame=subspace_frame(u,groups);reordered,et=ordered_frame(frame,eta)
    assert np.linalg.norm(u@np.diag(eta)@u.conj().T-reordered@np.diag(et)@reordered.conj().T)<1e-12
    assert np.linalg.norm(Operator(gaussian_circuit(reordered)).data-fock_frame(reordered))<1e-12


def test_near_degeneracy_is_model_change_and_not_exact_gauge():
    g=np.diag([-.4,-.397,1,1.002]);nu=2
    candidates=spectral_candidates(g,nu,-.7,.02)
    assert any(c['bound']>0 and c['family']=='cluster' for c in candidates)
    assert all(c['bound']<=.02 for c in candidates)
    original=second_quantize_one_body(g)
    for c in candidates:
        f=second_quantize_one_body(c['g']);idx=sector_indices(4,nu)
        actual=.7*np.linalg.norm((original@original-f@f)[np.ix_(idx,idx)],2)
        assert actual<=c['bound']+1e-12


def test_compiler_scalar_phase_is_audited_and_absolute_reference_preserved():
    gs,weights,_,_=n1_input('planted_local_gauge')
    frame=spectral_candidates(gs[0],2,weights[0],.02)[0]['frame']
    audit=Audit();cost=audit.compile(gaussian_circuit(frame),fock_frame(frame),4,'phase-regression',controlled=False)
    assert cost['built_residual']<1e-12 and cost['native_residual']<1e-12
    assert audit.records[0]['compiler_phase_audit']['scalar_residual']<1e-12


def test_charge_native_compile_preserves_arbitrary_system_inputs():
    con=charges(n2_input('signed_overlap'))
    qc,workspace,_=charge_circuit(con['S'],con['K'],con['widths'],.7)
    bits=np.array([[s>>i&1 for i in range(4)] for s in range(16)])
    q=bits@np.asarray(con['S']);values=np.einsum('bi,ij,bj->b',q,np.asarray(con['K']),q)
    phase=np.diag(np.exp(-.7j*values));reference=np.block([[np.eye(16),np.zeros((16,16))],[np.zeros((16,16)),phase]])
    audit=Audit();cost=audit.compile(qc,reference,4,'signed-native-regression',workspace)
    assert cost['native_residual']<1e-11
    assert audit.records[0]['qubits_initially_zero'] is False


@pytest.mark.parametrize('sign',[-1,1])
def test_modular_increment_all_register_values(sign):
    qc=QuantumCircuit(4);increment(qc,0,[1,2,3],sign)
    u=Operator(qc).data
    for value in range(8):
        for control in (0,1):
            source=control+2*value;target=control+2*((value+sign*control)%8)
            assert u[target,source]==pytest.approx(1)


@pytest.mark.parametrize('signed',[True,False])
def test_charge_control_signed_phase_and_uncompute_all_states(signed):
    s=np.array([[1],[1],[-1 if signed else 1]])
    k=np.array([[-.3]]);duration=-.7;widths=[3]
    qc,workspace,_=charge_circuit(s,k,widths,duration,signed=signed)
    for bits in itertools.product((0,1),repeat=3):
        value=int(np.asarray(bits)@s[:,0]);system=sum(b<<i for i,b in enumerate(bits))
        for anc in (0,1):
            source=system+(anc<<(3+workspace));state=Statevector.from_int(source,1<<qc.num_qubits).evolve(qc).data
            expected=np.exp(-1j*duration*float(k[0,0])*value**2*anc)
            assert state[source]==pytest.approx(expected,abs=2e-12)
            assert np.linalg.norm(state-np.eye(1,1<<qc.num_qubits,source).ravel()*expected)<2e-12


def test_J_only_search_recovers_signed_representation_and_error_bound():
    j=n2_input('signed_overlap');con=charges(j)
    assert con['status']=='COMPRESSED_CANDIDATE_FOUND'
    assert con['examined']==820
    approx=np.asarray(con['S'])@np.asarray(con['K'])@np.asarray(con['S']).T
    assert density_bound(j-approx)<=.0021
    assert max(np.count_nonzero(con['S'],axis=1))<=2
    assert charges(n2_input('sparse_no_collective'))['status']=='NO_ELIGIBLE_COMPRESSED_CANDIDATE'


def test_N3_truth_free_positive_and_unresolved_gap():
    hops=[(1,2,.2),(0,3,.1),(2,4,.08),(3,5,.04),(4,5,.05)];dens=[(0,1,.3)]
    cert=active_certificate([-.2,-.1,.1,.25,6,8],hops,dens,[0,1,2,3])
    assert cert['status']=='COEFFICIENT_BOUND_AVAILABLE' and 0<cert['delta']<.004
    assert cert['delta']>=sum(row['beta']**2/(row['gap_lower']+cert['delta']) for row in cert['classes'])-1e-17
    cert=active_certificate([-.2,-.1,.1,.25,.4,.6],hops,dens,[0,1,2,3])
    assert cert['status']=='UNRESOLVED_LOWER_BOUND' and cert['delta'] is None


def test_invalid_particles_and_unsupported_constructor_input():
    with pytest.raises(ValueError):sector_norm([1,2],3)
    with pytest.raises(ValueError):charges(np.eye(5))
    with pytest.raises(ValueError):spectral_candidates(np.array([[0,1],[0,0]]),1,1,.1)

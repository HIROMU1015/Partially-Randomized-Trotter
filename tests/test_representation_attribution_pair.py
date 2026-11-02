"""Independent identities and phase/accounting checks, no science batch."""
import itertools
import math
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from trottertracks.representation_exploration import attribution_pair as a
from trottertracks.representation_exploration import construction_comparison as c
from trotterlib.rte import iter_rte_events, finite_rte_distribution, event_unitary


def annihilators(n):
    out = []
    for j in range(n):
        op = np.zeros((2**n, 2**n), complex)
        for k in range(2**n):
            if (k >> j) & 1: op[k ^ (1 << j), k] = (-1)**((k & ((1 << j)-1)).bit_count())
        out.append(op)
    return out


@pytest.mark.parametrize('complex_input', [False, True])
def test_entrywise_complete_core_equals_independent_occupation_diagonal(complex_input):
    rng = np.random.default_rng(81)
    factors = []
    for _ in range(2):
        z = rng.normal(size=(3, 3))
        if complex_input: z = z+1j*rng.normal(size=(3, 3))
        factors.append((z+z.conj().T)/2)
    aa = annihilators(3)
    lift = lambda g: sum(g[i, j]*aa[i].conj().T@aa[j] for i, j in itertools.product(range(3), repeat=2))
    h = sum(lift(g)@lift(g) for g in factors)
    con = a.complete_core(factors)
    assert c.norm(con.blocks[0].dense()-np.diag(np.diag(h))) < 2e-12
    assert c.norm(con.dense()-h) < 2e-12
    assert c.norm(a.whole_pauli(factors).dense()-h) < 2e-12


@pytest.mark.parametrize('complex_input', [False, True])
def test_qr_pair_matches_independent_normal_ordered_quartic(complex_input):
    x = np.array([.4, -.7, .2], complex); y = np.array([.3, .1, -.8], complex)
    if complex_input: x += 1j*np.array([.2, .15, -.1]); y += 1j*np.array([-.3, .7, .2])
    delta, v = a.pair_frame(x, y); q = a.PairInvolution('q', 1., c.gaussian_circuit(v)).dense()
    aa = annihilators(3); cx = sum(x[i]*aa[i] for i in range(3)); cy = sum(y[i]*aa[i] for i in range(3))
    target = cx.conj().T@cy.conj().T@cy@cx
    assert c.norm(delta*(np.eye(8)-q)/2-target) < 1e-12
    assert c.norm(c.pdense(a.quartic_terms(x, y), 3)-target) < 1e-12
    assert c.norm(q@q-np.eye(8)) < 1e-12


def test_exact_dependent_skipped_but_near_collinear_retained():
    x = np.array([1., 0., 0.])
    assert a.pair_frame(x, 2*x) == (0., None)
    delta, v = a.pair_frame(x, np.array([1., 1e-9, 0.]))
    assert 0 < delta < 1e-17 and v is not None
    assert a.pair_frame(x, np.zeros(3)) == (0., None)


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('angle', [None, .17, -.23])
def test_pair_control_product_rotation_absolute_phase(sign, angle):
    _, v = a.pair_frame(np.array([.3, .2j, .7]), np.array([-.4j, .6, .1]))
    prim = a.PairInvolution('q', sign*.2, c.gaussian_circuit(v))
    qc = QuantumCircuit(4); prim.apply(qc, 3, angle=angle)
    target = sign*prim.dense() if angle is None else expm(-1j*angle*sign*prim.dense())
    ref = np.eye(16, dtype=complex); ref[8:, 8:] = target
    assert c.norm(Operator(qc).data-ref) < 1e-12
    receipt = c.NativeAudit(cap=1).compile(qc, ref, 'synthetic/Q')
    assert receipt['compiled_reconstructed_residual'] < c.ATOL


def test_conditional_factory_is_existing_public_law_not_new_sampler():
    con = a.whole_pauli([np.array([[.3, .04], [.04, -.2]])]); tail = con.tail()
    # Two-mode dictionary is small enough to enumerate every existing event.
    dist = finite_rte_distribution(tail.lambda_r*.12, 2)
    expected = list(iter_rte_events(tail.components, dist))
    actual = [a._make_event(indices, tail.components, dist, k)
              for k, order in enumerate(dist.orders)
              for indices in itertools.product(range(len(tail.components)), repeat=order+1)]
    assert [e.to_dict() for e in actual] == [e.to_dict() for e in expected]
    ops = {x.component_id: op for x, op in zip(tail.components, tail.operators)}
    mean = sum(e.event_probability*event_unitary(e, ops) for e in actual)*dist.exact_finite_distribution
    h = tail.dense_hamiltonian; z = -.12j*h
    assert c.norm(mean-(np.eye(4)+z+z@z/2+z@z@z/6)) < 1e-12


def test_weighted_exact_and_paired_mc_cost_statistics():
    def draw(weight, x, y): return {'weight': weight, 'costs': {'X': {'rz': x}, 'Y': {'rz': y}}}
    strata = [{'probability': .9, 'exact': True, 'draws': [draw(.25, 2, 3), draw(.75, 4, 5)]},
              {'probability': .1, 'exact': False, 'draws': [draw(.5, 10, 10), draw(.5, 20, 30)]}]
    stats = a.mean_se(strata, 'XY', 'rz')
    assert stats['mean'] == pytest.approx(.9*8+.1*35)
    assert stats['se'] == pytest.approx(1.5)


def test_shared_uniform_inverse_cdf_and_exact_small_stratum():
    factors = [np.array([[.3, .04, 0], [.04, -.2, .03], [0, .03, .1]])]
    con = a.whole_pauli(factors); tail = con.tail(); uniforms = np.random.default_rng(4).random((96, 3))
    design = a.conditional_design(tail, finite_rte_distribution(.2, 2), uniforms)
    assert design[0]['exact'] and not design[1]['exact']
    probs = np.array([x.probability for x in tail.components]); probs /= probs.sum()
    indices = np.searchsorted(np.cumsum(probs), uniforms, side='right')
    assert [x['indices_rotation_first'] for x in design[1]['draws']] == indices.tolist()
    assert sum(x['weight'] for x in design[0]['draws']) == pytest.approx(1.)


def test_same_isometry_all_physical_dictionaries_reconstruct():
    reflected, direct, meta = c.enlarged_connection()
    u = np.array(meta['orbital_isometry']['real']); edges = meta['diagonal_density_edges']
    for con in [a.coefficient_direct(u, edges), a.physical_pairs(u, edges, involution=False),
                a.physical_pairs(u, edges, involution=True)]:
        assert c.norm(con.dense()-direct.dense()) < 1e-12
    q = a.physical_pairs(u, edges, involution=True)
    assert q.tail().lambda_r == pytest.approx(.5196286029667954)
    assert q.identity == pytest.approx(.46962860296679537)

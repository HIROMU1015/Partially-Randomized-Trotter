"""Off-domain algebra only; no saved arrays, RNG, circuits, prices or shots.

Reuse the production RTE event/reference functions. Additional sums below are
independent 2x2 witnesses of the survey's statements, not a new sampling law.
"""
import math

import numpy as np
import pytest

from trotterlib import rte


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


def toy():
    tail = rte.normalize_involutory_tail(
        "invented-signed-XZ", (
            rte.InvolutoryTailTerm("x", .7, X),
            rte.InvolutoryTailTerm("z", -.3, Z),
        ))
    return tail, {c.component_id: p for c, p in zip(tail.components, tail.operators)}


@pytest.mark.parametrize("tau", [.37, -.37])
@pytest.mark.parametrize("cutoff", [0, 2])
def test_actual_paired_law_mean_order_and_normalizer(tau, cutoff):
    tail, operators = toy()
    distribution = rte.finite_rte_distribution(tau, cutoff)
    events = rte.enumerate_rte_events(tail.components, distribution)
    mean = rte.exact_enumerated_event_mean_operator(events, operators)
    target = rte.finite_taylor_operator(tail.normalized_hamiltonian, tau, cutoff)
    B = distribution.exact_finite_distribution
    np.testing.assert_allclose(B * mean, target, rtol=0, atol=1e-14)
    assert math.fsum(e.event_probability for e in events) == pytest.approx(1.)
    assert all(e.phase == (-1) ** (e.taylor_order // 2) for e in events)
    if cutoff == 0:
        assert len(events) == 2 and B == pytest.approx(math.hypot(1, tau))
        np.testing.assert_allclose(target, I - 1j * tau * tail.normalized_hamiltonian)
        # B^-1 is the declared normalization attenuation, not |raw signal|.
        state = np.array([1., 0.], dtype=complex)
        assert abs(np.vdot(state, mean @ state)) > 1 / B


def test_probability_change_needs_weights_and_keeps_measurement_noise():
    tail, operators = toy()
    events = rte.enumerate_rte_events(tail.components, rte.finite_rte_distribution(.37, 0))
    V = [rte.event_unitary(e, operators) for e in events]
    p = np.array([e.event_probability for e in events])
    q = np.array([.2, .8])
    old = sum(pi * v for pi, v in zip(p, V))
    assert np.linalg.norm(sum(qi * v for qi, v in zip(q, V)) - old) > .1
    w = p / q
    np.testing.assert_allclose(sum(qi * wi * v for qi, wi, v in zip(q, w, V)), old)
    second = float(q @ w**2)
    assert second > 1 and max(w) > 1
    state = np.array([1., 0.], dtype=complex)
    axis_means = np.array([np.vdot(state, v @ state).real for v in V])
    mean = float(p @ axis_means)
    # Y is a real measured +/-1 outcome: E[(wY)^2]=E[w^2], not
    # E[(w E[Y|path])^2]. This difference persists with exact path means.
    assert second - mean**2 > float(q @ (w * axis_means)**2) - mean**2


def test_affine_dictionary_dependence_allows_unweighted_probability_change():
    theta = .37
    V = [math.cos(theta) * I - 1j * math.sin(theta) * p for p in (X, X, Z)]
    p, q = [.2, .3, .5], [.1, .4, .5]
    old = sum(pi * v for pi, v in zip(p, V))
    new = sum(qi * v for qi, v in zip(q, V))
    np.testing.assert_allclose(new, old, rtol=0, atol=1e-15)
    state = np.array([1., 0.], dtype=complex)
    # Equal normalized operator mean -> equal per-axis Bernoulli marginal.
    a, b = np.vdot(state, old @ state), np.vdot(state, new @ state)
    assert (a.real, a.imag) == pytest.approx((b.real, b.imag))
    assert (1 - a.real**2, 1 - a.imag**2) == pytest.approx((1 - b.real**2, 1 - b.imag**2))


def test_parent_split_is_euler_identity_and_common_mass_minimum():
    # Existing Track B parent identity, evaluated only as an algebra witness.
    a, b = 1., [.21, .16]
    common = [a * bi / sum(b) for bi in b]
    target = a * I - 1j * (b[0] * X + b[1] * -Z)
    minimum = math.hypot(a, sum(b))
    for shares in (common, [a, 0.], [.4, .6]):
        mass = [math.hypot(zi, bi) for zi, bi in zip(shares, b)]
        angles = [math.atan2(bi, zi) for zi, bi in zip(shares, b)]
        total = sum(ci * (math.cos(phi) * I - 1j * math.sin(phi) * Q)
                    for ci, phi, Q in zip(mass, angles, (X, -Z)))
        np.testing.assert_allclose(total, target, rtol=0, atol=5e-16)
        assert sum(mass) >= minimum - 5e-16
    assert sum(math.hypot(zi, bi) for zi, bi in zip(common, b)) == pytest.approx(minimum)


def test_correlated_marginals_change_ordered_mean_and_path_is_restores_it():
    theta, eta = .37, .6
    V = [math.cos(theta) * I - 1j * math.sin(theta) * Q for Q in (X, Z)]
    R = (X + Z) / 2
    iid = sum(.25 * V[j] @ V[i] for i in range(2) for j in range(2))
    q = {(i, j): .5 * ((1 - eta) * .5 + eta * (i == j))
         for i in range(2) for j in range(2)}
    assert all(sum(q[i, j] for j in range(2)) == pytest.approx(.5) for i in range(2))
    assert all(sum(q[i, j] for i in range(2)) == pytest.approx(.5) for j in range(2))
    correlated = sum(q[i, j] * V[j] @ V[i] for i, j in q)
    np.testing.assert_allclose(correlated - iid, -eta * math.sin(theta)**2 * (I - R @ R), atol=1e-15)
    assert np.linalg.norm(correlated - iid) > .01
    weighted = sum(q[i, j] * (.25 / q[i, j]) * V[j] @ V[i] for i, j in q)
    np.testing.assert_allclose(weighted, iid, atol=1e-15)
    assert sum(.25**2 / qi for qi in q.values()) == pytest.approx(1 / (1 - eta**2))
    assert max(.25 / qi for qi in q.values()) == pytest.approx(1 / (1 - eta))


def test_system_channel_equality_does_not_preserve_controlled_phase():
    phase = np.exp(-.37j)
    rho = np.array([[.5, .5], [.5, .5]], dtype=complex)
    V = phase * I
    np.testing.assert_allclose(V @ rho @ V.conj().T, rho)
    controlled = np.block([[I, np.zeros_like(I)], [np.zeros_like(I), V]])
    probe = np.array([1., 0., 1., 0.], dtype=complex) / math.sqrt(2)
    out = controlled @ probe
    # X+iY on the leading control reads the phase that system Ad_V discards.
    coherence = 2 * np.vdot(out[:2], out[2:])
    assert coherence == pytest.approx(phase)
    assert abs(coherence - 1) > .1

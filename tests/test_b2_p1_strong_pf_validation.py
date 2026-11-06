"""Artificial inputs only; forbid circuits, private loads and shot calls."""
from fractions import Fraction

import numpy as np
import pytest

from trotterlib import b2_p1_strong_pf_validation as m
from trotterlib.product_formula import _get_w_list


@pytest.fixture(autouse=True)
def no_private_circuits_or_shots(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No private inputs, circuits, simulation runners or shots')
    from qiskit import QuantumCircuit
    monkeypatch.setattr(QuantumCircuit, '__init__', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(m.p, 'load_private_inputs', forbidden)
    monkeypatch.setattr(m.p, 'controlled_checks', forbidden)
    monkeypatch.setattr(m.p.connection.accounting, '_hoeffding_shots', forbidden)
    monkeypatch.setattr(m.strong.StrongBuilder, 'sequence', forbidden)


@pytest.mark.parametrize('formula,s', m.ARMS)
def test_schedule_extraction_preserves_signed_stages_and_repetitions(formula, s):
    weights = _get_w_list('4th')
    pieces = (1.,) if formula == 'Full_S2' else (weights[1], weights[0], weights[1])
    expected = tuple(item for w in pieces for item in m.strong.native.ordered_schedule(
        3, Fraction(.02)/s*Fraction(w)))*(4*s)
    actual = m.strong_baseline_schedule(3, substeps=s, formula=formula, delta_time=.02, q=4)
    assert actual == expected
    assert any(t < 0 for _, t in actual) == (formula == 'Yoshida4')
    with pytest.raises(ValueError):
        m.strong_baseline_schedule(3, substeps=s, formula='new', delta_time=.02)


@pytest.mark.parametrize('formula,s', m.ARMS)
def test_relative_native_diagonal_vs_existing_car_reference(formula, s):
    one = np.kron(np.array([[.4, .13j], [-.13j, -.2]]), np.eye(2))
    gs = [np.kron(np.array([[1., .17j], [-.17j, -.3]]), np.eye(2)),
          np.kron(np.array([[.3, .21], [.21, .6]]), np.eye(2))]
    weights = [.7, -.35]; scalar = .371
    model, reports = m.strong.prepare_xdf_basis(one, weights, gs, scalar,
        source_identity='artificial_SOURCE', input_identity='artificial_DF', basis_policy=m.strong.SPIN)
    assert len(reports) == 3
    ref = m.p.car_reference(); a = ref.annihilators(4)
    generators = [ref.one_body(one, a).toarray()]
    generators += [w*(ref.one_body(g, a)@ref.one_body(g, a)).toarray() for w, g in zip(weights, gs)]
    schedule = m.strong_baseline_schedule(3, substeps=s, formula=formula, delta_time=.02)
    step, leak = m.native_mean_step(model, schedule, ref, np.arange(16))
    other = np.exp(-1j*scalar*.02)*m._ordered_exponentials(generators, schedule)
    np.testing.assert_allclose(step, other, rtol=0, atol=2e-13)
    assert leak == 0
    # Scalar is physically relative under ordinary control, never phase fitted.
    assert abs(step[0, 0]-np.exp(-1j*scalar*.02)) < 2e-13
    assert np.max(abs(step-other*np.exp(.01j))) > .005


def test_p1_caps_no_reallocation_and_zero_rte():
    z = .9*np.exp(.03j)
    q = m.qualify(0, z, z, z, z)
    assert q['feasible'] and q['finite_RTE_complex_disk'] == 0
    assert q['raw_radius_lower'] == abs(z)-m.p.U
    assert q['RTE_budget_reserved_unreallocated'] == .04
    assert q['phase_total_with_reserved_statistical_and_unused_RTE'] == pytest.approx(.36000002)
    fail = m.qualify(0, z, z, z*np.exp(.061j), z*np.exp(.061j))
    assert not fail['feasible'] and 'PF_phase_budget' in fail['reasons']
    fail = m.qualify(0, z, z, z, z+2e-9)
    assert not fail['feasible'] and 'independent_numerical_match' in fail['reasons']
    assert not m.qualify(0, 1., 1., 1e-10, 1e-10)['feasible']


def test_first_failure_stops_arm_other_arm_independent():
    calls = []
    def evaluate(m_):
        calls.append(m_)
        return dict(m=m_, qualification=dict(feasible=m_ != 3))
    rows, status = m.evaluate_arm('Full_S2', 1, evaluate, lambda r: None)
    assert calls == [0, 1, 2, 3] and status == 'STOP_ARM_INELIGIBLE'
    other, status = m.evaluate_arm('Yoshida4', 8,
        lambda i: dict(m=i, qualification=dict(feasible=True)), lambda r: None)
    assert [r['m'] for r in other] == list(range(18))
    assert status == 'P1_ALL18_CONDITIONAL_SIGNAL_PASS'

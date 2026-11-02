"""Synthetic-only AX-2A checks; no saved H4 snapshot or molecular runner."""

from pathlib import Path
import builtins
import math

import numpy as np
import pytest
from scipy.linalg import expm

from trotterlib.rte import finite_taylor_operator, finite_rte_distribution
from trottertracks.resource_applicability.ax2a_state_action import (
    ActionBudget, partial_s2_signal, deterministic_pf_state,
    finite_taylor_action, project_primitive_checked, df_tail_operator,
    eigenphase_reference,
)
from trottertracks.resource_applicability.ax2a_control_plan import (
    deterministic_control_plan, partial_s2_control_plan,
)
from trottertracks.resource_applicability.ax2a_preparation import (
    axis_headroom, prefix_candidates, pilot_draft, build_preparation,
)

A = np.array([[.3, .2+.1j], [.2-.1j, -.4]], dtype=complex)
D = np.array([[.1, -.13j], [.13j, .2]], dtype=complex)
H = np.array([[.5, .11], [.11, -.2]], dtype=complex)
PSI = np.array([1, 1j], dtype=complex) / math.sqrt(2)


@pytest.fixture(autouse=True)
def forbid_science_file_access(monkeypatch):
    original = builtins.open
    path_open = Path.open

    def check(path):
        if isinstance(path, (str, Path)):
            name = str(path)
            if name.endswith('.npz') or '/.runtime/' in name or '/artifacts/' in name:
                raise AssertionError('Real scientific evidence access forbidden in synthetic test: '+name)

    def guarded(path, *args, **kwargs):
        check(path)
        return original(path, *args, **kwargs)

    def guarded_path(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', guarded)
    monkeypatch.setattr(Path, 'open', guarded_path)


def action(matrix):
    return lambda v, t: expm(-1j*t*matrix) @ v


def budget():
    return ActionBudget(10000, 10000)


@pytest.mark.parametrize('K', [0, 2, 4, 6])
@pytest.mark.parametrize('q,r', [(1, 1), (2, 3), (4, 2)])
@pytest.mark.parametrize('T', [.8, -.8])
def test_matrix_free_mean_matches_existing_dense_oracle(K, q, r, T):
    lam, scalar = .9, .37
    result = partial_s2_signal(PSI, [action(A), action(D)], lambda v: H@v,
        lambda_r=lam, T=T, q=q, r=r, K=K, phase_energy=scalar, budget=budget())
    delta, tau = T/q, lam*T/(q*r)
    P = finite_taylor_operator(H, tau, K)
    b = finite_rte_distribution(tau, K).exact_finite_distribution
    forward = expm(-1j*delta*D/2) @ expm(-1j*delta*A/2)
    reverse = expm(-1j*delta*A/2) @ expm(-1j*delta*D/2)
    step = np.exp(-1j*scalar*delta) * reverse @ np.linalg.matrix_power(P, r) @ forward
    target = np.vdot(PSI, np.linalg.matrix_power(step, q) @ PSI)
    np.testing.assert_allclose(result.corrected, target, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(result.raw, target/(b**(q*r)), atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(result.normalization*result.raw, result.corrected, atol=2e-13)


def test_polynomial_degree_and_no_renormalization():
    Hlarge = 2*np.eye(2)
    out = finite_taylor_action(lambda v: Hlarge@v, PSI, 1, 0, budget=budget())
    assert np.linalg.norm(out) > 2
    np.testing.assert_allclose(out, (1-2j)*PSI)


@pytest.mark.parametrize('bad', [True, -1, 1.5])
def test_invalid_counts_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        partial_s2_signal(PSI, [], None, lambda_r=0, T=.8, q=bad, r=1, K=2,
                          phase_energy=0, budget=budget())


def test_empty_tail_zero_time_and_scalar_phase():
    result = partial_s2_signal(PSI, [], None, lambda_r=0, T=.8, q=4, r=3, K=6,
                              phase_energy=.37, budget=budget())
    np.testing.assert_allclose(result.corrected, np.exp(-1j*.8*.37))
    assert result.normalization == 1
    np.testing.assert_allclose(result.raw, result.corrected)
    b = budget()
    np.testing.assert_array_equal(finite_taylor_action(lambda v: v/0, PSI, 0, 6, budget=b), PSI)
    assert b.tail_matvecs == 0


def test_operation_caps_and_dimension_failure():
    with pytest.raises(RuntimeError, match='TAIL_MATVEC_BUDGET'):
        finite_taylor_action(lambda v: H@v, PSI, .2, 4, budget=ActionBudget(2, 0))
    with pytest.raises(ValueError, match='dimension'):
        finite_taylor_action(lambda v: np.ones(3), PSI, .2, 2, budget=budget())
    with pytest.raises(RuntimeError, match='DETERMINISTIC_ACTION_BUDGET'):
        deterministic_pf_state(PSI, [action(A)], T=.8, q=1, formula='4th', scalar=0,
                               budget=ActionBudget(0, 0))


def test_nonfinite_and_odd_order_rejected():
    with pytest.raises(ValueError):
        finite_taylor_action(lambda v: v, PSI, float('inf'), 2, budget=budget())
    with pytest.raises(ValueError, match='even'):
        finite_taylor_action(lambda v: v, PSI, .2, 3, budget=budget())
    with pytest.raises(ValueError, match='finite'):
        finite_taylor_action(lambda v: np.array([np.nan, 0]), PSI, .2, 2, budget=budget())
    with pytest.raises(ValueError, match='normalized'):
        deterministic_pf_state(2*PSI, [], T=.8, q=1, formula='2nd', scalar=0, budget=budget())


@pytest.mark.parametrize('q', [1, 2, 4])
def test_fourth_order_has_signed_composition_and_improves_global_error(q):
    exact = expm(-1j*.3*(A+D)) @ PSI
    second = deterministic_pf_state(PSI, [action(A), action(D)], T=.3, q=q,
                                   formula='2nd', scalar=0, budget=budget())
    fourth = deterministic_pf_state(PSI, [action(A), action(D)], T=.3, q=q,
                                   formula='4th', scalar=0, budget=budget())
    assert np.linalg.norm(fourth-exact) < .02*np.linalg.norm(second-exact)
    assert any(s.time < 0 for s in deterministic_control_plan(2, .3, q, '4th'))


def test_one_term_fourth_is_exact_and_preserves_scalar():
    actual = deterministic_pf_state(PSI, [action(A)], T=-.8, q=3, formula='4th',
                                    scalar=.37, budget=budget())
    np.testing.assert_allclose(actual, np.exp(1j*.8*.37)*expm(1j*.8*A)@PSI, atol=2e-13)


@pytest.mark.parametrize('formula', ['2nd', '4th'])
@pytest.mark.parametrize('T', [.8, -.8])
def test_control_plan_equals_ordinary_control_on_noncommuting_terms(formula, T):
    actions = [action(A), action(D)]
    plan = deterministic_control_plan(2, T, 3, formula)
    branches = []
    for branch in (0, 1):
        value = PSI.copy()
        for stage in plan:
            value = actions[stage.term](value, stage.branch_time(branch))
        value *= np.exp(-1j*.37*T*branch)
        branches.append(value)
    expected = deterministic_pf_state(PSI, actions, T=T, q=3, formula=formula,
                                     scalar=.37, budget=budget())
    np.testing.assert_allclose(branches[0], PSI, atol=2e-13)
    np.testing.assert_allclose(branches[1], expected, atol=2e-13)
    # Check every basis column and the scalar relative phase in both branches.
    for column in np.eye(2, dtype=complex).T:
        for branch in (0, 1):
            value = column.copy()
            for stage in plan:
                value = actions[stage.term](value, stage.branch_time(branch))
            value *= np.exp(-1j*.37*T*branch)
            expected_column = column if branch == 0 else deterministic_pf_state(
                column, actions, T=T, q=3, formula=formula, scalar=.37, budget=budget())
            np.testing.assert_allclose(value, expected_column, atol=2e-13)


@pytest.mark.parametrize('control', [True, 0.0, 2, -1])
def test_control_index_is_an_actual_bit(control):
    stage = deterministic_control_plan(1, .8, 1, '2nd')[0]
    with pytest.raises((ValueError, TypeError)):
        stage.branch_time(control)


def test_log_normalization_underflow_is_declared_not_returned_as_zero_signal():
    # Scalar toy kernel preserves the corrected state while normalization
    # exceeds binary64 range. No real Hamiltonian or trajectory is involved.
    result = partial_s2_signal(PSI, [], lambda v: np.zeros_like(v),
        lambda_r=100, T=1, q=1, r=200, K=0, phase_energy=.37, budget=budget())
    assert result.raw_status == 'AVAILABLE'
    assert result.normalization > 1
    np.testing.assert_allclose(result.corrected, np.exp(-.37j))
    result = partial_s2_signal(PSI, [], lambda v: np.zeros_like(v),
        lambda_r=10000, T=1, q=1, r=200, K=0, phase_energy=.37, budget=budget())
    assert result.raw is None and result.normalization is None
    assert result.raw_status == 'RAW_NORMALIZATION_UNDERFLOW'
    np.testing.assert_allclose(result.corrected, np.exp(-.37j))


def test_partial_backbone_control_keeps_central_unitary_and_event_phase():
    actions = [action(A), action(D), lambda v,t: np.exp(-.19j)*expm(-1j*t*H)@v]
    plan = partial_s2_control_plan(2, .8, has_tail=True)
    for branch in (0, 1):
        value = PSI.copy()
        for stage in plan:
            if stage.mode == 'ORDINARY' and branch == 0:
                continue  # includes skipping the central event's scalar phase
            value = actions[stage.term](value, stage.branch_time(branch))
        expected = PSI if branch == 0 else (
            expm(-.4j*A)@expm(-.4j*D)@actions[2](expm(-.4j*D)@expm(-.4j*A)@PSI, .8))
        np.testing.assert_allclose(value, expected, atol=2e-13)


def test_each_primitive_sector_is_checked_even_when_net_action_preserves_it():
    swap = np.array([[0,1],[1,0]],dtype=complex)
    # Two swaps return to the selected sector; either primitive leaves it.
    with pytest.raises(ValueError, match='PRIMITIVE_LEAVES_SECTOR'):
        project_primitive_checked(lambda v: swap@v, np.ones(1), [0], 2)
    np.testing.assert_allclose(project_primitive_checked(lambda v: swap@swap@v,
                               np.ones(1), [0], 2), [1])
    with pytest.raises(ValueError, match='duplicate'):
        project_primitive_checked(lambda v: v, np.ones(2), [0,0], 2)


def test_sector_df_tail_removes_one_body_constant_and_identity_once():
    from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
    g = np.array([[.2, .3j],[-.3j,-.1]],dtype=complex)
    ham = DFHamiltonian(constant=5, one_body=7*np.eye(2), lambdas=np.array([.8]),
                        g_matrices=(g,), metadata={'synthetic':True})
    sector = PhysicalSector.number_sector(n_qubits=2,n_electrons=1)
    with pytest.raises(ValueError, match='preservation'):
        df_tail_operator(ham, sector, [0], extracted_identity=.17)
    op, counter = df_tail_operator(ham, sector, [0], extracted_identity=.17,
                                  primitive_sector_certified=True)
    # One-particle second quantization of g in sorted JW basis is reversed.
    gp = g[::-1,::-1]
    np.testing.assert_allclose(op@PSI, (.8*gp@gp-.17*np.eye(2))@PSI, atol=2e-13)
    assert counter['count'] == 1


def test_eigenphase_allowance_is_not_ground_state_claim():
    rec = eigenphase_reference(PSI, lambda v: H@v, .8)
    exact = np.vdot(PSI, expm(-.8j*H)@PSI)
    assert abs(exact-rec['signal']) <= rec['signal_allowance']+1e-13
    assert rec['ground_state_certified'] is False


@pytest.mark.parametrize('T', [.8, -.8])
def test_b0_signed_complex_decomposition_retains_discard_and_pf_terms(T):
    scalar = .37
    exact_full = np.vdot(PSI, np.exp(-1j*scalar*T)*expm(-1j*T*(A+D+H))@PSI)
    exact_truncated = np.vdot(PSI, np.exp(-1j*scalar*T)*expm(-1j*T*(A+D))@PSI)
    b0 = partial_s2_signal(PSI, [action(A),action(D)], None, lambda_r=0,
        T=T, q=2, r=1, K=2, phase_energy=scalar, budget=budget()).corrected
    global_pf = np.vdot(PSI, deterministic_pf_state(PSI,[action(A),action(D)],
        T=T,q=2,formula='2nd',scalar=scalar,budget=budget()))
    np.testing.assert_allclose(b0, global_pf, atol=2e-13)
    discard, pf = exact_truncated-exact_full, b0-exact_truncated
    np.testing.assert_allclose(discard+pf, b0-exact_full, atol=2e-13)
    assert abs(discard) > 1e-3 and abs(pf) > 1e-6


@pytest.mark.parametrize('bias,u,eligible', [(.03,.01,True),(.08,.01,False),(.05,.01,None),(.05,0,False),(None,.01,None)])
def test_headroom_is_three_state(bias,u,eligible):
    assert axis_headroom(.05,bias,u)['eligible'] is eligible


def test_prefix_generation_and_pilot_have_no_model_or_oracle_input():
    assert prefix_candidates(1)==(0,1)
    assert prefix_candidates(12)==tuple(sorted(set(prefix_candidates(12))))
    p=pilot_draft()
    assert len(p['H6_tasks'])==6 and p['H8_tasks']==[]
    assert p['science_authorized'] is False and p['ax2b_authorized'] is False
    assert p['assigned_resources'] is None
    assert p['mandatory_stop'] is True


def test_preparation_hashes_only_explicit_metadata_and_preserves_frozen_model(tmp_path,monkeypatch):
    # Synthetic root; Path reads supplied toy bytes, never real evidence.
    review=tmp_path/'review.md'
    review.write_text('synthetic review')
    # Supply synthetic bytes through a fake read method, even for artifact literals.
    seen=[]
    def read(path):
        seen.append(str(path))
        return b'synthetic bytes'
    monkeypatch.setattr(Path,'read_bytes',read)
    p=build_preparation(tmp_path,review)
    assert p['frozen_H4_model']['refit'] is False
    assert p['new_scientific_calculation_count']==0
    assert not any(name.endswith('.npz') for name in seen)
    assert p['science_authorized'] is False

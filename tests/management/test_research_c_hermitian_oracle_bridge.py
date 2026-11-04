"""Invented matrices/states only. Real artifacts are inspected as metadata bytes."""
from fractions import Fraction as F
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from trottertracks import c_hermitian_oracle_bridge as m
from trottertracks import c_direct_round_signal as direct
from trottertracks import c_pr_aware_df_zsxx_enclosure as z

ROOT = Path(__file__).resolve().parents[2]


def problem(a, uplo='L', **kw):
    return m.eigh_problem(a, uplo=uplo, dtype=kw.pop('dtype', '<c16'),
        source_sha256=kw.pop('source_sha256', 'a'*64),
        input_sha256=kw.pop('input_sha256', 'b'*64), **kw)


def candidate(a):
    return m.candidate_problem(a, kind='candidate_exact_coefficients',
        source_sha256='c'*64, input_sha256='d'*64)


@pytest.mark.parametrize('delta', [F(0), F(1, 2**50), F(100)])
def test_saved_vs_eigh_exact_small_and_large_asymmetry(delta):
    a = [[0, delta], [0, 2]]
    p = problem(a)
    assert p.values == m.matrix([[0, 0], [0, 2]])
    b = m.difference_bound(a, p.values)
    assert F(b['frobenius_squared_exact']) == delta**2
    assert F(b['operator_norm_upper_exact']) >= abs(delta)
    assert b['identical'] == (delta == 0)
    assert p.saved_matrix_id == m.matrix_id(m.matrix(a))
    assert 'NOT_research_target' in p.kind


@pytest.mark.parametrize('uplo', ['L', 'U'])
@pytest.mark.parametrize('a', [
    [[0, 3], [1, 2]],
    [[1+9j, 7-3j], [1+2j, 4-8j]],
    [[0, 1j], [-1j, 2]],
])
def test_triangle_complex_diagonal_and_numpy_semantics(a, uplo):
    p = problem(a, uplo)
    b = np.array([[complex(float(r), float(i)) for r, i in row] for row in p.values])
    vals, vecs = np.linalg.eigh(np.array(a, dtype=np.complex128), UPLO=uplo)
    assert np.max(np.abs(b@vecs-vecs*vals)) < 1e-13
    assert np.array_equal(b, b.conj().T)
    assert all(p.values[i][i][1] == 0 for i in range(len(a)))


def test_lower_and_upper_are_distinct_and_neither_is_an_average():
    a = [[0, 9], [1, 2]]
    lo, up = problem(a), problem(a, 'U')
    assert lo.values[0][1][0] == 1 and up.values[1][0][0] == 9
    assert lo.id != up.id
    assert lo.values[0][1][0] != 5
    with pytest.raises(ValueError, match='NO_SYMMETRIZATION'):
        candidate(a)


@pytest.mark.parametrize('kwargs,match', [
    ({'uplo':'average'}, 'UPLO'), ({'uplo':''}, 'UPLO'),
    ({'dtype':'<f4'}, 'BINARY64'), ({'source_sha256':'wrong'}, 'SHA256'),
    ({'input_sha256':'wrong'}, 'SHA256')])
def test_no_implicit_policy_or_invalid_binding(kwargs, match):
    with pytest.raises(ValueError, match=match):
        problem([[0, 0], [0, 2]], **kwargs)


def test_real_dtype_rejects_complex_and_nonfinite():
    with pytest.raises(ValueError, match='REAL_DTYPE'):
        problem([[0, 1j], [-1j, 2]], dtype='<f8')
    with pytest.raises(ValueError, match='FINITE'):
        problem([[float('nan'), 0], [0, 2]])


@pytest.mark.parametrize('field', ['source', 'input', 'target'])
def test_wrong_source_input_or_target_rejected(field):
    p = problem([[0, 0], [0, 2]])
    kw = dict(expected_id=p.id, source_sha256='a'*64, input_sha256='b'*64)
    kw[{'source':'source_sha256', 'input':'input_sha256', 'target':'expected_id'}[field]] = 'f'*64
    with pytest.raises(ValueError, match='IDENTITY_MISMATCH'):
        m.validate(p, **kw)


def test_real_oracle_reuse_residual_normalization_and_distance():
    p = problem([[0, 0], [0, 2]])
    r = m.qualify(p, [F(999, 1000), F(1, 1000)], 1, expected_id=p.id)
    eta = F(r['vector2_distance_upper_exact'])
    assert eta**2 >= F(2, 10**6) and eta < F(1, 100)
    assert F(r['normalized_residual_squared_exact']) > 0
    assert F(r['ground_gap_lower_exact']) > 0
    assert r['inertia_representation'] == 'original real matrix'
    assert not r['input_normalized_or_projected']
    assert not r['research_target_adopted']


@pytest.mark.parametrize('phase', [1, -1, 1j, -1j])
def test_complex_hermitian_realification_and_phase(phase):
    # Exact eigenvalues 0,2; unit ground (3/5,4i/5).
    p = problem([[F(32,25), (0,F(24,25))], [(0,F(-24,25)), F(18,25)]])
    v = [m.pair(phase)]
    # Multiply by exact phases without binary64 3/5 or 4/5 conversions.
    a, b = v[0]
    r = m.qualify(p, [(F(3,5)*a,F(3,5)*b),(-F(4,5)*b,F(4,5)*a)],
                  1, expected_id=p.id)
    assert r['vector2_distance_upper_exact'] == '0'
    assert r['normalized_residual_squared_exact'] == '0'
    assert r['inertia']['negative'] == r['inertia']['positive'] == 2
    assert r['ground_gap_lower_exact'] == '1'


def test_complex_inexact_vector_distance_contains_analytic_answer():
    p = problem([[F(32,25), (0,F(24,25))], [(0,F(-24,25)), F(18,25)]])
    r = m.qualify(p, [(F(3,5),0),(0,F(79,100))], 1, expected_id=p.id)
    assert F(r['vector2_distance_upper_exact']) >= F(1,100)
    assert F(r['normalized_residual_squared_exact']) > 0
    assert F(r['normalization_distance_upper_exact']) > 0


def test_degenerate_gap_and_zero_pivot_stop():
    p = problem([[0,0,0],[0,0,0],[0,0,2]])
    with pytest.raises(ValueError, match='UNIQUE_GROUND'):
        m.qualify(p, [1,0,0], 1, expected_id=p.id)
    p = problem([[0,1j],[-1j,0]])
    with pytest.raises(ValueError, match='ZERO_LDL'):
        m.qualify(p, [1,0], 0, expected_id=p.id)


def test_near_degenerate_exact_gap_does_not_use_numerical_tolerance():
    eps = F(1,2**100)
    p = problem([[0,0],[0,eps]])
    r = m.qualify(p, [1,0], eps/2, expected_id=p.id)
    assert F(r['ground_gap_lower_exact']) == eps/2
    target = candidate([[eps,0],[0,eps]])
    with pytest.raises(ValueError, match='PERTURBATION_GAP'):
        m.transport_ground(p,target,[1,0],eps/2,
            expected_reference_id=p.id,expected_target_id=target.id)


def test_perturbation_bridge_contains_known_target_ground_distance():
    p = problem([[0,0],[0,2]])
    # Rotate diag(0,2) with exact rational cos=399/401, sin=40/401.
    c,s = F(399,401),F(40,401)
    target = candidate([[2*s*s,-2*s*c],[-2*s*c,2*c*c]])
    r = m.transport_ground(p,target,[1,0],1,
        expected_reference_id=p.id,expected_target_id=target.id)
    eta = F(r['original_centre_to_target_distance_upper_exact'])
    assert eta*eta >= 2-2*c
    assert F(r['target_ground_gap_lower_exact']) > 0
    assert not r['research_target_adopted']
    with pytest.raises(ValueError, match='IDENTITY_MISMATCH'):
        m.transport_ground(p,target,[1,0],1,
            expected_reference_id=p.id,expected_target_id=p.id)


def test_zero_matrix_bridge_adds_no_duplicate_state_error():
    p = problem([[0,0],[0,2]])
    r = m.transport_ground(p,p,[F(999,1000),F(1,1000)],1,
        expected_reference_id=p.id,expected_target_id=p.id)
    assert r['ground_transport_distance_upper_exact'] == '0'
    assert r['original_centre_to_target_distance_upper_exact'] == r['reference_certificate']['vector2_distance_upper_exact']


def test_signal_state_error_unitary_and_nonunitary_reference():
    eta = F(1,100)
    u = m.state_signal_bounds(eta,map_norm_upper=1)
    b = m.state_signal_bounds(eta,map_norm_upper=3)
    assert F(u['X_upper_exact']) == 2*eta
    assert F(b['X_upper_exact']) == 18*eta
    assert F(b['complex_upper_exact'])**2 >= 2*(18*eta)**2
    assert not b['covers_native_synthesis_error']
    # Saved centre is unnormalized: reuse the existing direct radius formula.
    from flint import ctx, acb_mat
    with z._BALL_LOCK, ctx.workprec(192):
        out = direct.observable_z(acb_mat([[3],[0]]),3*eta)
    lo,hi = map(F,out['signal'])
    assert lo <= (3*(1-eta))**2 <= hi
    assert lo <= (3*(1+eta))**2 <= hi
    assert hi > 9  # no probability clipping of a nonunitary reference


@pytest.mark.parametrize('flag', ['--science-permitted','--resume','--retry'])
def test_passive_cli_has_no_scientific_or_retry_entry(flag):
    r = subprocess.run([sys.executable,'-B','scripts/prepare_research_c_hermitian_oracle_bridge.py',flag],
        cwd=ROOT,capture_output=True,text=True)
    assert r.returncode == 2 and 'unrecognized arguments' in r.stderr


def test_passive_preparation_stop_and_old_failure_are_preserved():
    r = subprocess.run([sys.executable,'-B','scripts/prepare_research_c_hermitian_oracle_bridge.py'],
        cwd=ROOT,capture_output=True,text=True,check=True)
    row = json.loads(r.stdout)
    assert row['status'] == 'NOT_READY' and row['scientific_calls'] == 0
    assert row['H_target'] is None and row['logical_calls']['total'] is None
    assert row['retry'] == row['resume'] == 0 and row['mandatory_STOP']
    old = json.loads((ROOT/'artifacts/research_c_direct_m0_oracle_binding/2026-11-05/science-run01/STOP.json').read_text())
    assert old['scientific_units'] == 3 and old['mandatory_stop']
    assert old['failed']['error'] == 'EXACT_SYMMETRY_REQUIRED_NO_HERMITIANIZATION'

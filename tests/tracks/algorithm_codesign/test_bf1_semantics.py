"""BF-1 synthetic tests only. No molecular inputs or circuit constructors."""
from fractions import Fraction
import math
import numpy as np
import mpmath as mp
import pytest
import sys
from types import SimpleNamespace
from scipy.linalg import expm

from trottertracks.algorithm_codesign.domain import Domain, ExactTime, Point, fixed_references
from trottertracks.algorithm_codesign.adapter import (Factor, allocate, dense_operator, fuse,
    polynomial, stage_list, stationary_template, tail_statistics)
from trottertracks.algorithm_codesign.numerics import Spectrum, phase_values
from trottertracks.algorithm_codesign.pilot import Evaluator, Task, parameter_score, search, classify
from trottertracks.algorithm_codesign.formal_check import order_residuals
from trottertracks.algorithm_codesign.shared_cpu import module, forbidden_gpu


@pytest.fixture(scope='module')
def domain():
    return Domain()


def test_all_feasible_components_and_frozen_starts(domain):
    manifest = domain.manifest()
    assert (manifest['chart_count'], manifest['connected_component_count']) == (4, 3)
    assert manifest['stratum_counts'] == [4, 3, 7]
    assert len(manifest['initial_points']) == 16
    for record in manifest['initial_points']:
        p = Point.from_record(record)
        with mp.workdps(80):
            values = [w.value(p.basis) for w in p.weights]
            assert abs(mp.fsum(values)-1) < mp.mpf('1e-65')
            assert abs(mp.fsum(w**3 for w in values)) < mp.mpf('1e-65')
        assert record['eta1'] <= 1e-12 and record['eta3'] <= 1e-12
    assert set(p['component'] for p in manifest['initial_points']) == {0, 1, 2}


def test_exact_fusion_does_not_clip_small_times():
    tiny = ExactTime(Fraction(1, 10**40))
    assert fuse([Factor('R', tiny)]) == (Factor('R', tiny),)
    t = ExactTime(Fraction(1), Fraction(2))
    assert fuse([Factor('D0', t), Factor('R', ExactTime()), Factor('D0', -t)]) == ()
    # An exact inverse stage pair exposes and cancels inner factors recursively.
    p = Point('inverse', None, None, 'test', '0', (ExactTime(Fraction(1)), ExactTime(Fraction(-1))))
    assert stage_list(p, 4) == ()


def test_signed_four_generator_native_list_and_relative_phase(domain):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1., -1.]).astype(complex)
    generators = dict(D0=.13*x, D1=.07*z, D2=.05*(x+z), D3=.03*z, R=.11*x-.09*z)
    point = domain.yoshida()  # Includes a negative central coefficient.
    factors = stage_list(point, 4, simplify=False)
    raw = np.eye(2, dtype=complex)
    for f in factors:
        with mp.workdps(80):
            t = float(f.time.value(point.basis))*.8
        raw = expm(-1j*t*generators[f.generator])@raw
    fused = dense_operator(point, generators, .8, 1, scalar=.37)
    np.testing.assert_allclose(fused, np.exp(-1j*.37*.8)*raw, atol=2e-14)
    controlled = np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), fused]])
    psi = np.array([1., 1j])/math.sqrt(2)
    control_plus = np.concatenate([psi, psi])/math.sqrt(2)
    evolved = controlled@control_plus
    signal = np.vdot(psi, fused@psi)
    re = np.vdot(evolved, np.kron(x, np.eye(2))@evolved).real
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    im = np.vdot(evolved, np.kron(y, np.eye(2))@evolved).real
    assert abs(re-signal.real) < 1e-14 and abs(im-signal.imag) < 1e-14
    assert abs(signal-np.vdot(psi, raw@psi)) > .1


@pytest.mark.parametrize('K', [2, 4])
def test_signed_finite_rte_mean_and_normalization(K):
    r = np.array([[.2, .3-.1j], [.3+.1j, -.2]])
    positive = polynomial(r, .4, K)
    negative = polynomial(r, -.4, K)
    np.testing.assert_allclose(negative, positive.conj().T, atol=2e-15)
    a = tail_statistics([.4], [3], .7, 2, K)
    b = tail_statistics([-.4], [3], .7, 2, K)
    assert a == b
    assert K+1 in (3, 5)


def test_adapter_matches_shared_finite_RTE_and_symbolic_DF_coefficients():
    rte = module('rte')
    z = np.diag([1., -1.]).astype(complex)
    x = np.array([[0., 1.], [1., 0.]], dtype=complex)
    R = .2*z-.3*x
    tail = SimpleNamespace(tail_id='synthetic', tail_hash='synthetic', lambda_r=.5, components=(1,))
    for K in (2, 4):
        for t in (-.4, .4):
            config, distribution = rte.make_rte_config(tail, evolution_time=t, rte_steps=3,
                truncation_tolerance=.01, finite_taylor_order=K)
            moments = rte.finite_rte_operator_moments(R/.5, config)
            ours = np.linalg.matrix_power(polynomial(R, t/3, K), 3)
            np.testing.assert_allclose(ours, moments.corrected_operator, atol=2e-15)
            stats = tail_statistics([t], [3], .5, 1, K)
            assert abs(stats['log_b']-math.log(moments.normalization_product)) < 2e-15
            np.testing.assert_allclose(moments.attenuated_event_mean_operator,
                                      ours/math.exp(stats['log_b']), atol=2e-15)
    coefficients = module('df_rte_tail').exact_df_diagonal_coefficients([.3, -.4], .7)
    reconstructed = np.zeros((4, 4), dtype=complex)
    for support, coefficient in coefficients:
        diagonal = [(-1)**sum((i >> bit)&1 for bit in support) for i in range(4)]
        reconstructed += coefficient*np.diag(diagonal)
    occupations = [.7*(.3*(i&1)-.4*((i>>1)&1))**2 for i in range(4)]
    np.testing.assert_allclose(reconstructed, np.diag(occupations), atol=1e-15)
    assert 'trotterlib.df_gpu_statevector' not in sys.modules
    assert sys.modules['_bf1_shared_cpu.df_gpu_statevector'].simulate_statevector_gpu is forbidden_gpu
    with pytest.raises(RuntimeError, match='GPU'):
        forbidden_gpu()
    # Future loader dependencies can import through the CPU boundary without
    # invoking any loader, molecule builder, GPU query, or circuit constructor.
    module('pr2_new_series_validation')
    module('pr2_matched_accuracy_m1_execution')
    assert 'trotterlib.df_gpu_statevector' not in sys.modules


def test_published_baseline_transcription_and_order_residuals(domain):
    refs = fixed_references(domain)
    assert len(refs[-1].weights) == 21
    for point, degree in ((domain.yoshida(), 4), (domain.suzuki(), 4), (refs[-1], 8)):
        errors = order_residuals(point, degree)
        assert max(float(v) for v in errors.values()) < 1e-26


def test_F_finite_insertion_is_not_S_simplification():
    p = Point('two', None, None, 'synthetic rational stages', '0',
              (ExactTime(Fraction(1, 3)), ExactTime(Fraction(2, 3))))
    r = np.diag([.4, -.3]).astype(complex)
    f = dense_operator(p, {'R': r}, 1., 1, allocation=(1,), K=2, construction='F')
    s = dense_operator(p, {'R': r}, 1., 1, allocation=(1, 1), K=2, construction='S')
    np.testing.assert_allclose(f, polynomial(r, 1., 2))
    np.testing.assert_allclose(s, polynomial(r, 2/3, 2)@polynomial(r, 1/3, 2))
    assert np.linalg.norm(f-s) > 1e-4
    with pytest.raises(ValueError, match='crosses'):
        stationary_template(p, 0, 2)


def test_rounding_stationarity_and_tail_occurrences(domain):
    for p in domain.initial_points()[1]+fixed_references(domain):
        one, full = stationary_template(p, 4, 8)
        tails = [f.time for f in one if f.generator == 'R']
        for budget in (5, 10, 20, 40, 80):
            r = allocate(p, tails, budget)
            if budget < len(tails):
                assert r is None
            else:
                assert sum(r) == budget and min(r) >= 1
        assert len([f for f in full if f.generator == 'R']) == 8*len(tails)
    p = Point('equal', None, None, 'tie test', '0', (ExactTime(Fraction(1, 2)),)*2)
    assert allocate(p, list(p.weights), 5) == (3, 2)


def test_spectral_guard_covers_independent_dense_actions():
    matrix = np.array([[.7, .21-.12j], [.21+.12j, -.4]])
    state = np.array([1., .3j]); state /= np.linalg.norm(state)
    spectrum = Spectrum(matrix)
    for t in (-2., -.3, 0., .8):
        value, error, _ = spectrum.action(state, t)
        assert np.linalg.norm(value-expm(-1j*t*matrix)@state) <= error
        for K in (2, 4):
            value, error, _ = spectrum.action(state, t, r=3, K=K)
            reference = np.linalg.matrix_power(polynomial(matrix, t/3, K), 3)@state
            assert np.linalg.norm(value-reference) <= error
    theta = np.array([-100., -1., 0., .17, 70.])
    value, error = phase_values(theta)
    assert np.max(np.abs(value-np.exp(-1j*theta))) <= error


def test_future_evaluator_agrees_with_independent_synthetic_F_operator(domain):
    x = np.array([[0., 1.], [1., 0.]], dtype=complex)
    z = np.diag([1., -1.]).astype(complex)
    generators = dict(D0=.03*x, D1=.02*z, D2=.01*(x+z), D3=.015*z, R=.02*x-.03*z)
    state = np.array([1., 1j])/math.sqrt(2)
    task = Task(generators, .13, .05, state)
    evaluator = Evaluator(task)
    point = domain.suzuki()
    row = evaluator.finite_cell(point, 2, 10, 2)
    independent = dense_operator(point, generators, .8, 2, allocation=row['allocation'], K=2, scalar=.13)
    actual = complex(*row['signal'])
    expected = np.vdot(state, independent@state)
    assert abs(actual-expected) <= row['u_signal']
    target_matrix = sum(generators.values(), .13*np.eye(2, dtype=complex))
    assert abs(evaluator.target-np.vdot(state, expm(-.8j*target_matrix)@state)) <= evaluator.target_error
    # L is computed from ideal signal and leading statistics only.
    evaluator.finite_cell = lambda *args: (_ for _ in ()).throw(AssertionError('finite objective leaked into L'))
    evaluator.objective(point, 'L')
    evaluator.objective(point, 'O')


def test_all_arms_have_same_budget_and_no_finite_objective_leak(domain):
    class Diagnostic:
        def __init__(self):
            self.calls = []
        def objective(self, p, arm):
            self.calls.append((arm, p.identity))
            return math.inf
    fixture = Diagnostic()
    initial = domain.initial_points()[1]
    sets = []
    for arm in ('O', 'L', 'F'):
        points, records = search(domain, initial, fixture, arm)
        assert len(points) == 32 and len({p.identity for p in points}) == 32
        sets.append([p.identity for p in points])
    assert sets[0] == sets[1] == sets[2]
    assert len(fixture.calls) == 96


def test_C_only_has_one_primary_route_and_always_stops():
    def row(identity, value, q=1):
        return dict(coefficient=identity, value=value, lower=value*(1-1e-7), q=q,
                    R_bud=5, K=2, shots=[50, 50], deterministic_actions=value/200,
                    random_actions=value/200)
    for ratio, expected in ((.94, 'BF-C'), (.96, 'BF-B'), (1., 'BF-A')):
        decision = classify([row('F', 100*ratio)], [row('L', 100)], [row('L', 100)])
        assert decision['outcome'] == expected and decision['mandatory_stop']
        assert decision['automatic_next_stage'] is None and not decision['BF2_authorized']
    assert classify([row('F', 96, q=2)], [row('L', 100)], [row('L', 100)])['outcome'] != 'BF-C'
    assert classify([row('F', 94, q=8)], [row('L', 100)], [row('L', 100)])['outcome'] == 'INCONCLUSIVE'
    assert not parameter_score([.1, .1], 0, 0, 1, 1, .01)['feasible']

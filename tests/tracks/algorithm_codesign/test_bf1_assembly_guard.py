"""Independent synthetic perturbation checks; no molecular input or search."""
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from scipy.linalg import expm

from trottertracks.algorithm_codesign.adapter import dense_operator, polynomial
from trottertracks.algorithm_codesign.domain import Point
from trottertracks.algorithm_codesign.numerics import Spectrum
from trottertracks.algorithm_codesign.pilot import Evaluator, Task

ROOT = Path(__file__).absolute().parents[3]


def frozen_suzuki():
    domain = json.loads((ROOT / 'artifacts/track_b_bf1_preparation/2026-10-05/v1/domain_manifest.json').read_text())
    return Point.from_record(next(p for p in domain['initial_points'] if p['label'] == 'Suzuki5'))


def scalar_witness(K=2, q=1, *, uniform=False):
    point = frozen_suzuki()
    supplied = 5.+1e-6
    distance = math.nextafter(abs(supplied-5.), math.inf)
    matrices = {f'D{i}': np.zeros((1, 1), dtype=complex) for i in range(4)}
    matrices['R'] = np.array([[supplied]], dtype=complex)
    budgets = dict.fromkeys(matrices, 0.)
    budgets['R'] = distance
    task = (Task(matrices, 0., supplied, np.ones(1, dtype=complex), distance) if uniform else
            Task(matrices, 0., supplied, np.ones(1, dtype=complex), distance, budgets, 0.))
    evaluator = Evaluator(task)
    row = evaluator.finite_cell(point, q, 5, K)
    with mp.workdps(80):
        times = [mp.mpf('.8')*w.value(point.basis)/q for w in point.weights]
        one = mp.fprod(mp.fsum((-1j*t*5)**n/mp.factorial(n) for n in range(K+2)) for t in times)
        difference = one**q-mp.exp(-mp.mpf('.8')*5j)
        bias = [float(abs(mp.re(difference))), float(abs(mp.im(difference)))]
    return evaluator, row, bias


def test_original_nonzero_assembly_counterexample_is_covered_without_changing_signal():
    old = json.loads((ROOT / 'artifacts/track_b_bf1_preexecution_review/2026-10-05/296ec7e/assembly_guard_witness.json').read_text())
    evaluator, row, reference_bias = scalar_witness()
    errors = np.abs(np.array(row['bias'])-reference_bias)
    # Preserve the old counterexample and its old bias. Only the bound changes.
    assert errors[0] > old['reported_u_signal']
    np.testing.assert_array_equal(row['bias'], old['reported_bias'])
    assert max(errors) <= row['u_signal']
    assert evaluator.generator_assembly_bounds['D0'] == 0.
    assert evaluator.generator_assembly_bounds['R'] > 0.
    assert evaluator.target_assembly_bound >= evaluator.generator_assembly_bounds['R']


@pytest.mark.parametrize('K,q', [(2, 2), (4, 1), (4, 2)])
def test_polynomial_order_and_product_length_keep_assembly_guard(K, q):
    _, row, reference_bias = scalar_witness(K, q)
    assert max(np.abs(np.array(row['bias'])-reference_bias)) <= row['u_signal']


def test_uniform_budget_is_a_bound_for_each_generator_not_their_sum():
    evaluator, row, reference_bias = scalar_witness(uniform=True)
    assert all(b > 0. for b in evaluator.generator_assembly_bounds.values())
    assert evaluator.target_assembly_bound >= (sum(evaluator.generator_assembly_bounds.values())
                                                +evaluator.scalar_assembly_bound)
    assert max(np.abs(np.array(row['bias'])-reference_bias)) <= row['u_signal']


@pytest.mark.parametrize('finite', [False, True])
def test_noncommuting_hermitian_perturbation_and_signed_time(finite):
    reference = np.array([[.7, .21-.12j], [.21+.12j, -.4]])
    perturbation = 2e-6*np.array([[1., 1j], [-1j, -.3]])
    supplied = reference+perturbation
    budget = float(np.linalg.norm(perturbation, 'fro'))*(1+1e-12)
    spectrum = Spectrum(supplied)
    state = np.array([1., .3j]); state /= np.linalg.norm(state)
    time = -.8
    r = 3 if finite else None
    actual, error, norm = spectrum.action(state, time, r=r, K=2, assembly_error=budget)
    operator = (np.linalg.matrix_power(polynomial(reference, time/3, 2), 3) if finite else
                expm(-1j*time*reference))
    assert np.linalg.norm(actual-operator@state) <= error
    assert np.linalg.norm(operator, ord=2) <= norm+1e-14


def test_scalar_relative_phase_and_generator_errors_both_enter_signal_and_target():
    x = np.array([[0., 1.], [1., 0.]], dtype=complex)
    z = np.diag([1., -1.]).astype(complex)
    reference = dict(D0=.13*x, D1=.07*z, D2=.05*(x+z), D3=.03*z, R=.11*x-.09*z)
    perturbation = dict(D0=1e-6*z, D1=2e-6*x, D2=-1e-6*z, D3=1e-6*x, R=2e-6*(x+z))
    supplied = {k: reference[k]+perturbation[k] for k in reference}
    budgets = {k: float(np.linalg.norm(v, 'fro'))*(1+1e-12) for k, v in perturbation.items()}
    scalar, scalar_delta = .37, 1e-6
    state = np.array([1., 1j])/math.sqrt(2)
    evaluator = Evaluator(Task(supplied, scalar+scalar_delta, .4, state,
                               generator_assembly_bounds=budgets, scalar_assembly_bound=scalar_delta*(1+1e-9)))
    point = frozen_suzuki()
    row = evaluator.finite_cell(point, 2, 10, 2)
    operator = dense_operator(point, reference, .8, 2, allocation=row['allocation'], K=2, scalar=scalar)
    signal = np.vdot(state, operator@state)
    target_matrix = sum(reference.values(), scalar*np.eye(2, dtype=complex))
    target = np.vdot(state, expm(-.8j*target_matrix)@state)
    assert abs(evaluator.target-target) <= evaluator.target_error
    reference_bias = [abs(signal.real-target.real), abs(signal.imag-target.imag)]
    assert max(np.abs(np.array(row['bias'])-reference_bias)) <= row['u_signal']


@pytest.mark.parametrize('bound', [-1., math.inf, math.nan])
def test_invalid_assembly_budget_is_rejected(bound):
    with pytest.raises(ValueError, match='Assembly budgets'):
        Task({'R': np.zeros((1, 1))}, 0., 0., np.ones(1), bound).assembly_errors()


def test_missing_generator_budget_is_rejected():
    with pytest.raises(ValueError, match='each generator'):
        Task({'D0': np.zeros((1, 1)), 'R': np.zeros((1, 1))}, 0., 0., np.ones(1),
             generator_assembly_bounds={'R': 0.}, scalar_assembly_bound=0.).assembly_errors()


def test_guard_checks_the_eigenvalues_used_by_identity_extraction():
    matrix = np.array([[.7, .21-.12j], [.21+.12j, -.4]])
    values, vectors = np.linalg.eigh(matrix)
    values = values+np.array([1e-6, -2e-6])
    supplied = Spectrum(matrix, eigensystem=(values, vectors))
    reconstruction = vectors@(values[:, None]*vectors.conj().T)
    assert np.linalg.norm(matrix-reconstruction, ord=2) <= supplied.matrix_error
    state = np.array([1., 1j])/math.sqrt(2)
    actual, error, _ = supplied.action(state, -.8)
    assert np.linalg.norm(actual-expm(.8j*matrix)@state) <= error

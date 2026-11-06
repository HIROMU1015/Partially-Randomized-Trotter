"""Invented arrays only. Solver success is distinct from H6 qualification."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_h6_factored_newton as newton
from trottertracks import c_h6_fit_conditioning as fit
from trottertracks.c_near_transfer import System, objective_specs
from trottertracks.c1_h6_initialization import reconstruct


def chart(arm):
    rng = np.random.default_rng(391)
    U = np.stack([np.linalg.qr(x)[0] for x in rng.normal(size=(3, 2, 2))])
    J = rng.normal(size=U.shape)*.05
    J = (J+J.transpose(0, 2, 1))/2
    target = J.copy()
    target[2] += np.array([[.003, .004], [.004, -.002]])
    g = reconstruct(target, U)
    spec = objective_specs(g, J, U, System(2, 3, 2, 1., 'a'*64),
        strength=.1, epsilon=1e-8)[arm]
    return fit.Chart(fit.backend(spec).Coefficients(g, U, spec), J, fit.inner.DEFAULT)


@pytest.mark.parametrize('arm', ['PR', 'ENTRY'])
def test_factor_reproduces_unchanged_analytic_Hessian_and_gradient(arm):
    c = chart(arm)
    z = np.zeros(c.dimension)
    D, gradient, rhs = newton.factor(c, z)
    np.testing.assert_allclose(D.T@D, c.hess(z), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(gradient, c.independent(z)[2], atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(-D.T@rhs, gradient, atol=1e-8, rtol=1e-7)
    direction, metadata = newton.direction(D, rhs)
    assert gradient@direction < 0
    assert metadata['directions_dropped'] == 0
    assert not metadata['objective_regularization_added']


@pytest.mark.parametrize('arm', ['PR', 'ENTRY'])
def test_nonzero_tail_solver_monotonic_and_original_physical_threshold(arm):
    c = chart(arm)
    result = newton.solve(c)
    assert result.nit <= 1000
    assert all(row['actual'] >= 0 for row in result.iteration_trace)
    value, gx, pullback = c.independent(result.x)
    assert np.max(abs(gx)) < 1e-5
    assert value < c.vg(np.zeros(c.dimension))[0]
    np.testing.assert_allclose(result.jac, pullback, atol=1e-8, rtol=1e-7)


def test_weak_direction_retained_without_normal_equation_loss():
    D = np.array([[1., 1.], [0., 1e-9], [0., 0.]], dtype=np.longdouble)
    target = np.array([.3, -.2], dtype=np.longdouble)
    delta, metadata = newton.direction(D, D@target)
    np.testing.assert_allclose(D@delta, D@target, atol=1e-15)
    assert metadata['directions_dropped'] == 0


def test_exhaustion_is_not_reported_as_success():
    result = newton.solve(chart('PR'), maxiter=1)
    assert not result.success and result.status == 1


def test_actual_scipy_flat_value_zero_division_retains_last_point(monkeypatch):
    c = chart('ENTRY')
    original = c.centered
    # The real SciPy1.14.1 interpolation, not a mocked exception: a rounded
    # constant value and the nonzero derivative reproduce the observed crash.
    monkeypatch.setattr(c, 'centered', lambda z: (0., original(z)[1]))
    result = newton.solve(c)
    assert not result.success and result.status == 2 and result.nit == 0
    assert 'ZeroDivisionError' in result.line_search_error
    np.testing.assert_array_equal(result.x, np.zeros(c.dimension))
    np.testing.assert_array_equal(result.jac, c.vg(result.x)[1])

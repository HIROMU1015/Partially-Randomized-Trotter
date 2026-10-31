"""Invented tensors only. Actual optimizer iterations and molecular decode=0."""
import importlib
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from trottertracks import c_rcdf_comparator as m
from trottertracks import c1_general_k_adapter as general


def artificial(n):
    rng = np.random.default_rng(631+n)
    a = rng.normal(size=(3, n, n))*.02
    J = a+a.transpose(0, 2, 1)
    U = np.array([np.eye(n), scipy.linalg.block_diag(-np.eye(2), np.eye(n-2)),
                  np.eye(n)])
    U[0, :, 0] *= -1
    angle = np.pi-1e-12
    U[2, :2, :2] = [[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]]
    g = m.pr.corrected.codec.reconstruct(J, U)+1e-4*np.einsum(
        "pq,rs->pqrs", np.eye(n), np.eye(n))
    J, U, x, info = m.pr.prepare(g, J, U, ld=1)
    return g, J, U, x, info


def test_Z_equals_J_spin_half_is_not_penalty_half_and_offdiagonal_twice():
    J = np.array([[[1., -2.], [-2., 3.]], [[.2, .3], [.3, -.7]]])
    # Paper's spatial contraction uses Z=J. The spin lift is separately K=J/2.
    assert m.full_l2(J, np) == pytest.approx(
        sum(np.sum(np.diag(a)**2)+2*a[0, 1]**2 for a in J))
    for a in J:
        K = np.repeat(np.repeat(a/2, 2, axis=0), 2, axis=1)
        independent = [coefficient for _, coefficient
                       in general.general_k_coefficients(K)[1:]]
        np.testing.assert_allclose(m.pr.coefficients(a, np), independent, atol=1e-15)
    assert m.full_l2(J/2, np)*4 == m.full_l2(J, np)
    changed = J.copy()
    changed[0] *= 2
    spec = m.pr.Objective(2, 2, 1, 1., 1., .1)
    assert m.pr.smooth_mass(changed, spec, np) == m.pr.smooth_mass(J, spec, np)
    assert m.full_l2(changed, np) != m.full_l2(J, np)


def test_exact_mass_linear_map_row_cancellation_and_restricted_equivalence():
    def mass(a):
        return float(np.sum(np.abs(m.pr.coefficients(a, np))))
    for t in (-2., -1., -.5, 0., 1.):
        a = np.array([[1., t], [t, 1.]])
        assert mass(a) == pytest.approx(2*abs(1+t)+.5+abs(t))
    # On the fixed-entry-sign interval t<0, weighted entry L1 is affine.
    # The row-sum zero at -1 makes lambda nonaffine there.
    assert mass(np.array([[1., -1.], [-1., 1.]])) != pytest.approx(
        (mass(np.array([[1., -1.5], [-1.5, 1.]]))
         + mass(np.array([[1., -.5], [-.5, 1.]])))/2)
    nonnegative = np.array([[[1., .2], [.2, 3.]], [[.4, .1], [.1, .8]]])
    weights = np.array([[1.25, 1.5], [1.5, 1.25]])
    assert mass(nonnegative) == pytest.approx(np.sum(weights*abs(nonnegative)))
    assert mass(-nonnegative) == pytest.approx(np.sum(weights*abs(nonnegative)))
    # Smoothing is not merely a removable constant on the unrestricted class.
    spec = m.pr.Objective(2, 2, 1, 1., 1., .1, .2)
    J = np.array([np.zeros((2, 2)), [[1., -.3], [-.3, .2]]])
    def centered_square(t):
        return m.pr.smooth_mass(t*J, spec, np)**2-m.pr.smooth_mass(0*J, spec, np)**2
    assert centered_square(2) != pytest.approx(4*centered_square(1))


@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("strength", [0., .1])
def test_value_gradient_scaling_and_actual_ffsim_route_without_iterations(n, strength):
    from ffsim.linalg.util import df_tensors_from_params
    g, J, U, x, info = artificial(n)
    spec = m.initial_spec(J, ld=1, loss_scale=info["spec"]["loss_scale"],
                          strength=strength)
    fun = m.make_value_and_grad(spec)
    value, grad = fun(x, g)
    baseline = m.pr.make_value_and_grad(spec.baseline())
    bv, bg = baseline(x, g)
    if not strength:
        assert value == bv
        np.testing.assert_array_equal(grad, bg)
    else:
        # Rotation-coordinate gradient receives no direct penalty term.
        rotations = len(J)*n*(n-1)//2
        np.testing.assert_allclose(np.asarray(grad-bg)[:rotations], 0., atol=1e-12)
        rows, cols = np.triu_indices(n)
        direct_penalty_gradient = (
            np.where(rows == cols, 2., 4.)*J[:, rows, cols]
            *strength/spec.l2_scale).ravel()
        np.testing.assert_allclose(np.asarray(grad-bg)[rotations:],
                                   direct_penalty_gradient, atol=1e-10, rtol=1e-12)
    assert value == pytest.approx(1.+strength, abs=1e-8)
    if strength:
        assert spec.strength*spec.loss_scale/spec.l2_scale != strength

    def independent(z):
        a, u = df_tensors_from_params(z, len(J), n, None, real=True)
        f = .5*np.sum((g-m.pr.corrected.codec.reconstruct(a, u))**2)
        return f/spec.loss_scale+strength*float(m.full_l2(a, np))/spec.l2_scale
    assert float(value) == pytest.approx(independent(x), abs=1e-8)
    for k in (0, len(x)//2, len(x)-1):
        dx = np.zeros_like(x)
        dx[k] = 1e-7
        assert (independent(x+dx)-independent(x-dx))/(2e-7) == pytest.approx(
            float(grad[k]), abs=3e-4, rel=1e-4)
    d = importlib.import_module("ffsim.linalg.double_factorized_decomposition")
    original = d._make_compressed_value_and_grad

    def fake_optimizer(actual_fun, x0, **kw):
        np.testing.assert_array_equal(x0, x)
        assert kw == dict(method="L-BFGS-B", jac=True, callback=None, options=None)
        v, dg = actual_fun(x0)
        assert v == pytest.approx(value, abs=1e-8)
        np.testing.assert_allclose(dg, grad, atol=1e-10, rtol=1e-12)
        assert m.components(g, x0, spec)["combined_objective"] == pytest.approx(v, abs=1e-8)
        return scipy.optimize.OptimizeResult(
            x=x0.copy(), jac=dg, fun=v, success=True, status=0,
            message="ARTIFICIAL_ZERO_ITERATIONS", nit=0, nfev=1, njev=1)
    before, after = Mock(), Mock()
    with patch.object(scipy.optimize, "minimize", side_effect=fake_optimizer) as opt, patch.object(
            m.pr.corrected.codec, "roundtrip", side_effect=AssertionError("NO_SECOND_CODEC")):
        a, u, result = m.fit(g, J, U, x, spec,
                            before_optimizer=before, after_optimizer=after)
    opt.assert_called_once()
    before.assert_called_once()
    after.assert_called_once()
    assert result.nit == 0 and d._make_compressed_value_and_grad is original
    np.testing.assert_array_equal(a, J)
    np.testing.assert_array_equal(u, U)
    np.testing.assert_allclose(m.pr.corrected.codec.reconstruct(a, u),
                               m.pr.corrected.codec.reconstruct(J, U), atol=1e-12)
    diagnostic = m.components(g, x, spec)
    assert all(diagnostic[k] is None for k in (
        "PF_error", "RTE_error", "attenuation", "shots", "total_cost",
        "CA_achieved", "independent_novelty"))


@pytest.mark.parametrize("fault", ["nonfinite_J", "asymmetric", "complex",
                                  "zero_scale", "negative_strength", "nonfinite_x"])
def test_invalid_input_stops(fault):
    J = np.array([np.eye(2), np.eye(2)])
    if fault == "nonfinite_J":
        J[0, 0, 0] = np.nan
    if fault == "asymmetric":
        J[0, 0, 1] = 1.
    if fault == "complex":
        J = J.astype(complex)
    with pytest.raises(ValueError):
        if fault == "zero_scale":
            m.initial_spec(J*0., ld=1, loss_scale=1.)
        elif fault == "negative_strength":
            m.initial_spec(J, ld=1, loss_scale=1., strength=-1.)
        elif fault == "nonfinite_x":
            spec = m.initial_spec(J, ld=1, loss_scale=1.)
            m.make_value_and_grad(spec)(np.full(8, np.nan), np.zeros((2,)*4))
        else:
            m.initial_spec(J, ld=1, loss_scale=1.)


def test_hook_restoration_on_structure_mismatch_and_exception():
    d = importlib.import_module("ffsim.linalg.double_factorized_decomposition")
    original = d._make_compressed_value_and_grad
    spec = m.Objective(2, 2, 1, 1., 1.)
    with pytest.raises(ValueError, match="STRUCTURE"):
        with m.objective_context(spec):
            d._make_compressed_value_and_grad(3, 2, None)
    assert d._make_compressed_value_and_grad is original
    with pytest.raises(RuntimeError):
        with m.objective_context(spec):
            raise RuntimeError("ARTIFICIAL")
    assert d._make_compressed_value_and_grad is original


def test_passive_import():
    code = ("import sys;sys.path.insert(0,'src');"
            "from trottertracks import c_rcdf_comparator_pilot as m;"
            "assert m.MAX_CALLS==17;"
            "assert not any(x in sys.modules for x in ['numpy','scipy','jax','ffsim','qiskit'])")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)

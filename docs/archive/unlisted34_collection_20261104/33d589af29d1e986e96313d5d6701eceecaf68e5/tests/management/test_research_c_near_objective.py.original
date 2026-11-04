"""Invented tensors only; no molecular decoding or optimizer iterations."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks import c_near_objective as m


@pytest.fixture(autouse=True)
def no_saved_arrays_or_optimizer():
    with patch("numpy.load", side_effect=AssertionError("NO_SAVED_ARRAY_DECODE")), \
         patch("scipy.optimize.minimize", side_effect=AssertionError("NO_OPTIMIZER")):
        yield


@pytest.mark.parametrize("n", [1, 2, 4])
def test_restricted_equivalence_floor_bound_and_head_mask(n):
    rng = np.random.default_rng(917+n)
    a = rng.uniform(.1, 1., (3, n, n))
    J = a+a.transpose(0, 2, 1)
    spec = m.Objective(3, n, 1, 1., 1., .1)
    for sign in (1., -1.):
        d = m.mass_diagnostics(sign*J, spec)
        assert d["exact_entry_mass"] == pytest.approx(d["exact_tail_mass"])
        assert d["row_cancellation_gap"] == pytest.approx(0., abs=1e-14)
    zero = J*0
    assert m.mass(zero, spec, np) == pytest.approx(m.pr.smooth_mass(zero, spec, np))
    assert m.mass(zero, spec, np) == pytest.approx(spec.smoothing_mass_bound)
    signed = J*rng.choice([-1., 1.], (3, n, n))
    signed = (signed+signed.transpose(0, 2, 1))/2
    d = m.mass_diagnostics(signed, spec)
    assert d["exact_entry_mass"]-d["exact_tail_mass"] == pytest.approx(d["row_cancellation_gap"])
    assert d["row_cancellation_gap"] >= -1e-14
    for prefix in ("tail", "entry"):
        excess = d["smooth_"+prefix+"_mass"]-d["exact_"+prefix+"_mass"]
        assert -1e-14 <= excess <= spec.smoothing_mass_bound+1e-14
    altered = signed.copy()
    altered[0] += 100*np.eye(n)
    assert m.mass_diagnostics(altered, spec) == d


def test_identical_entry_magnitudes_distinct_signed_mapped_mass():
    positive = np.array([np.zeros((2, 2)), [[1., 1.], [1., 1.]]])
    negative = np.array([np.zeros((2, 2)), [[1., -1.], [-1., 1.]]])
    spec = m.Objective(2, 2, 1, 1., 1., .1)
    p, n = [m.mass_diagnostics(J, spec) for J in (positive, negative)]
    assert p["exact_entry_mass"] == n["exact_entry_mass"] == 5.5
    assert p["smooth_entry_mass"] == n["smooth_entry_mass"]
    assert p["exact_tail_mass"] == 5.5 and n["exact_tail_mass"] == 1.5
    assert n["row_cancellation_gap"] == 4.
    assert n["smooth_tail_mass"] < p["smooth_tail_mass"]


@pytest.mark.parametrize("n,rank,ld", [(2, 3, 1), (4, 8, 2)])
@pytest.mark.parametrize("strength", [0., .1])
def test_existing_ffsim_loss_gradient_and_zero_direct_U_penalty(n, rank, ld, strength):
    from ffsim.linalg.util import df_tensors_from_params
    rng = np.random.default_rng(302+n)
    a = rng.normal(size=(rank, n, n))*.05
    J = a+a.transpose(0, 2, 1)
    U = np.tile(np.eye(n), (rank, 1, 1))
    U[0, :, 0] *= -1  # Existing codec must preserve this projector gauge.
    g = m.pr.corrected.codec.reconstruct(J, U)+.01*np.einsum("pq,rs->pqrs",np.eye(n),np.eye(n))
    J, U, x, info = m.pr.prepare(g, J, U, ld=ld)
    spec = m.initial_spec(J, ld=ld, loss_scale=info["spec"]["loss_scale"], strength=strength)
    value, grad = m.make_value_and_grad(spec)(x, g)
    bv, bg = m.pr.make_value_and_grad(spec.baseline())(x, g)
    assert float(value) == pytest.approx(1.+strength)
    rotations = spec.rank*n*(n-1)//2
    np.testing.assert_allclose(np.asarray(grad-bg)[:rotations], 0., atol=1e-12)
    rows, cols = np.triu_indices(n)
    derivative = np.zeros_like(J)
    derivative[spec.ld:] = (m.entry_weights(n, np)*J[spec.ld:]
                           / np.hypot(J[spec.ld:], spec.entry_epsilon))
    derivative = derivative[:, rows, cols]*np.where(rows == cols, 1., 2.)
    expected = (2*strength*m.mass(J, spec, np)/spec.mass_scale**2*derivative).ravel()
    np.testing.assert_allclose(np.asarray(grad-bg)[rotations:], expected, atol=2e-11)

    def direct(y):
        a, u = df_tensors_from_params(y, spec.rank, n, None, real=True)
        loss = .5*np.sum((g-m.pr.corrected.codec.reconstruct(a, u))**2)
        return loss/spec.loss_scale+strength*(m.mass(a, spec, np)/spec.mass_scale)**2
    for k in (0, rotations, len(x)-1):
        dx = np.zeros_like(x)
        dx[k] = 1e-7
        assert (direct(x+dx)-direct(x-dx))/(2e-7) == pytest.approx(float(grad[k]), abs=2e-5, rel=2e-5)
    if strength == 0:
        assert value == bv
        np.testing.assert_array_equal(grad, bg)
    assert all(m.mass_diagnostics(J, spec)[k] is None for k in (
        "PF_error", "RTE_error", "native_price", "total_cost", "independent_novelty"))


def test_fixed_U_convex_penalty_and_shared_scale_not_endpoint_refitted():
    rng = np.random.default_rng(211)
    a, b = [rng.normal(size=(3, 2, 2)) for _ in range(2)]
    a, b = a+a.transpose(0, 2, 1), b+b.transpose(0, 2, 1)
    spec = m.initial_spec(a, ld=1, loss_scale=1., strength=.1)
    original_scale = spec.mass_scale
    t = .37
    p = lambda j: (m.mass(j, spec, np)/spec.mass_scale)**2
    assert p(t*a+(1-t)*b) <= t*p(a)+(1-t)*p(b)+1e-13
    assert spec.mass_scale == original_scale


@pytest.mark.parametrize("fault", ["asymmetric", "nonfinite", "complex", "empty_tail", "negative_strength"])
def test_invalid_input(fault):
    J = np.tile(np.eye(2), (3, 1, 1))
    if fault == "asymmetric": J[0, 0, 1] = 1.
    if fault == "nonfinite": J[0, 0, 0] = np.nan
    if fault == "complex": J = J.astype(complex)
    with pytest.raises(ValueError):
        m.initial_spec(J, ld=3 if fault == "empty_tail" else 1, loss_scale=1.,
                       strength=-.1 if fault == "negative_strength" else .1)


def test_passive_import_and_no_scientific_entry():
    code = ("import sys;sys.path.insert(0,'src');"
            "from trottertracks import c_near_objective as m;"
            "assert not hasattr(m,'execute');"
            "assert not any(x in sys.modules for x in ['numpy','scipy','jax','ffsim','qiskit'])")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)

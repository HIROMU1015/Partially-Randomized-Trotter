"""Artificial coefficient/JAX/real-fitter dispatch tests; no optimizer iterations."""
from dataclasses import asdict
import importlib
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_pr_aware_df as m
from trottertracks import c1_general_k_adapter as general
from trottertracks.c1_h6_initialization import reconstruct


def artificial():
    rng = np.random.default_rng(4301)
    a = rng.normal(size=(3, 2, 2)) * .02
    J = a + a.transpose(0, 2, 1)
    U = np.array([np.eye(2), -np.eye(2), [[0., 1.], [1., 0.]]])
    g = reconstruct(J, U) + .0001*np.einsum('pq,rs->pqrs', np.eye(2), np.eye(2))
    return g, J, U


def test_general_K_map_sign_spin_identity_and_smoothing():
    _, J, _ = artificial(); J[1] = [[1., -1.], [-1., 1.]]
    spec = m.Objective(3, 2, 1, 1., 1., .1)
    coeff = m.coefficients(J, np)
    for i, a in enumerate(J):
        K = np.repeat(np.repeat(a/2, 2, axis=0), 2, axis=1)
        old = general.general_k_coefficients(K)
        np.testing.assert_allclose(coeff[i], [x[1] for x in old[1:]], atol=1e-15, rtol=0)
        assert old[0][0] == ()
    assert np.all(coeff[1, :4] == 0)  # row cancellation; no abs(J) impersonation
    exact = np.sum(np.abs(coeff[1:]))
    smooth = m.smooth_mass(J, spec, np)
    assert exact <= smooth <= exact + coeff[1:].size * spec.epsilon
    empty = m.smooth_mass(np.zeros_like(J), spec, np)
    assert empty > 0  # smoothing is not an exact zero-tail certification


@pytest.mark.parametrize('gamma', [0., .1])
def test_JAX_value_gradient_and_real_fitter_x0_without_iterations(gamma):
    import jax.numpy as jnp
    from ffsim.linalg.util import df_tensors_from_params
    d = importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    g, J, U = artificial(); a, u, x, info = m.prepare(g, J, U, ld=1)
    spec = m.Objective(**{**info['spec'], 'gamma': gamma})
    fun = m.make_value_and_grad(spec); value, grad = fun(jnp.asarray(x), jnp.asarray(g))
    def direct(z):
        a, u = df_tensors_from_params(z, 3, 2, None, real=True)
        f = .5*np.sum((g-reconstruct(a,u))**2)
        return f/spec.loss_scale + gamma*(m.smooth_mass(a,spec,np)/spec.mass_scale)**2
    assert float(value) == pytest.approx(direct(x), abs=1e-9)
    if gamma:
        raw=m.make_value_and_grad(m.Objective(**{**asdict(spec),'gamma':0.}))
        _,base_grad=raw(jnp.asarray(x),jnp.asarray(g))
        np.testing.assert_allclose(np.asarray(grad-base_grad)[:3],0.,atol=1e-12)
    for k in range(len(x)):
        dx = np.zeros_like(x); dx[k] = 1e-7
        assert (direct(x+dx)-direct(x-dx))/(2e-7) == pytest.approx(float(grad[k]), rel=1e-4, abs=2e-4)
    original = d._make_compressed_value_and_grad
    before, after = Mock(), Mock()
    def fake_optimizer(actual_fun, x0, **kw):
        np.testing.assert_array_equal(x0,x)
        assert kw == dict(method='L-BFGS-B', jac=True, callback=None, options=None)
        v, dg = actual_fun(x0)
        assert v == pytest.approx(float(value), abs=1e-9)
        np.testing.assert_allclose(dg, grad, rtol=1e-12, atol=1e-10)
        diagnostic = m.components(g,x0,spec)  # must not double-normalize patched factory
        assert diagnostic['combined_objective'] == pytest.approx(v, abs=1e-9)
        assert all(diagnostic[k] is None for k in ('shots','total_cost','CA_achieved','attenuation','PF_error','RTE_error','synthesis_error','independent_novelty'))
        return scipy.optimize.OptimizeResult(x=x0.copy(), jac=dg, fun=v,
            success=True, status=0, message='MOCK_ZERO_ITERATIONS', nit=0, nfev=1, njev=1)
    with patch.object(scipy.optimize,'minimize',side_effect=fake_optimizer) as optimizer:
        aa, uu, result = m.fit(g,a,u,x,spec,before_optimizer=before,after_optimizer=after)
    optimizer.assert_called_once(); before.assert_called_once(); after.assert_called_once()
    assert result.nit == 0 and d._make_compressed_value_and_grad is original
    np.testing.assert_array_equal(aa,a); np.testing.assert_allclose(reconstruct(aa,uu),reconstruct(J,U),atol=1e-12)


@pytest.mark.parametrize('fault', ['rank', 'gamma', 'empty', 'epsilon', 'scale'])
def test_bad_objective_explicit_FAIL(fault):
    args=dict(rank=3,norb=2,ld=1,loss_scale=1.,mass_scale=1.,gamma=.1,epsilon=1e-8)
    if fault=='rank':args['rank']=0
    if fault=='gamma':args['gamma']=float('nan')
    if fault=='empty':args['ld']=3
    if fault=='epsilon':args['epsilon']=0.
    if fault=='scale':args['loss_scale']=-1.
    with pytest.raises(ValueError):m.Objective(**args)


def test_factory_restored_on_exception_and_wrong_structure():
    import ffsim
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition');old=d._make_compressed_value_and_grad
    spec=m.Objective(3,2,1,1.,1.,.1)
    with pytest.raises(ValueError,match='STRUCTURE'):
        with m.objective_context(spec):d._make_compressed_value_and_grad(4,2,None)
    assert d._make_compressed_value_and_grad is old

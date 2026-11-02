"""Qualified x0 forwarding on invented matrices only; optimizer iterations=0."""
import importlib
from pathlib import Path
import sys
from unittest.mock import Mock,patch

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_ffsim_corrected_codec as bridge


def artificial(n):
    rng=np.random.default_rng(716+n)
    A=rng.normal(size=(3,n,n))*.03;J=A+A.transpose(0,2,1)
    U=np.array([np.linalg.qr(rng.normal(size=(n,n)))[0] for _ in range(3)])
    if np.linalg.det(U[0])>0:U[0,:,0]*=-1  # Improper frame plus proper pi frames.
    Q=U[1];U[1]=Q@scipy.linalg.block_diag(-np.eye(2),np.eye(n-2))@Q.T
    angle=np.pi-1e-12;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    Q=U[2];U[2]=Q@scipy.linalg.block_diag(R,np.eye(n-2))@Q.T
    g=bridge.codec.reconstruct(J,U)+.0001*np.einsum('pq,rs->pqrs',np.eye(n),np.eye(n))
    a,u,x,_,_=bridge.codec.roundtrip(J,U)
    return g,a,u,x


@pytest.mark.parametrize('n',[2,4,6])
def test_single_qualified_x0_in_actual_ffsim_path(n):
    import ffsim
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    g,J,U,x=artificial(n)
    originals=(d._double_factorized_explicit_cholesky,d.df_tensors_to_params)
    before=Mock();after=Mock()
    def minimize(fun,x0,**kwargs):
        np.testing.assert_array_equal(x0,x)
        assert kwargs==dict(method='L-BFGS-B',jac=True,callback=None,options=None)
        value,gradient=fun(x0)
        from ffsim.linalg.util import df_tensors_from_params
        def loss(z):
            a,u=df_tensors_from_params(z,len(J),n,None,real=True)
            return .5*np.sum((g-bridge.codec.reconstruct(a,u))**2)
        assert value==pytest.approx(loss(x),abs=1e-12)
        for k in (0,len(x)-1):
            dx=np.zeros_like(x);dx[k]=1e-6
            assert gradient[k]==pytest.approx((loss(x+dx)-loss(x-dx))/(2e-6),abs=1e-9,rel=1e-4)
        return scipy.optimize.OptimizeResult(x=x0.copy(),jac=gradient,fun=value,
            success=True,status=0,message='MOCK_ZERO_ITERATIONS',nit=0,nfev=1,njev=1)
    forbidden=Mock(side_effect=AssertionError('NO_REPARAMETERIZATION'))
    with patch.object(bridge.codec,'roundtrip',forbidden),patch.object(scipy.linalg,'logm',forbidden),patch.object(
            scipy.optimize,'minimize',side_effect=minimize) as optimizer:
        a,u,res=bridge.fit(g,J,U,x,prepared_initial=True,before_optimizer=before,after_optimizer=after)
    optimizer.assert_called_once();before.assert_called_once();after.assert_called_once();forbidden.assert_not_called()
    assert before.call_args.args[2]['codec']['parameter_source']=='qualified_snapshot'
    assert res.nit==0
    np.testing.assert_array_equal(a,J);np.testing.assert_array_equal(u,U)
    assert (d._double_factorized_explicit_cholesky,d.df_tensors_to_params)==originals


def test_artificial_reencoding_is_not_bitwise_idempotent():
    # Reproduce the old wiring failure without reading any molecular export.
    g,J,U,x=artificial(4)
    reencoded=bridge.codec.roundtrip(J,U)[2]
    assert not np.array_equal(reencoded,x)
    optimizer=Mock(side_effect=AssertionError('NO_OPTIMIZER_IN_OLD_FAILURE'))
    with patch.object(scipy.optimize,'minimize',optimizer):
        with pytest.raises(ValueError,match='QUALIFIED_X0_CHANGED'):
            bridge.fit(g,J,U,x,before_optimizer=Mock(),after_optimizer=Mock())
    optimizer.assert_not_called()


@pytest.mark.parametrize('fault',['x_shape','x_dtype','x_nonfinite','x_value','J','U','U_nonfinite','hook_mutation'])
def test_prepared_snapshot_mismatch_STOPS_without_fallback(fault):
    import ffsim
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    g,J,U,x=artificial(4);original=(d._double_factorized_explicit_cholesky,d.df_tensors_to_params)
    if fault=='x_shape':x=x[:-1]
    if fault=='x_dtype':x=x.astype(np.float32)
    if fault=='x_nonfinite':x[0]=np.nan
    if fault=='x_value':x[0]+=.01
    if fault=='J':J[0,0,0]+=.01
    if fault=='U':U[0,:,0]*=-1
    if fault=='U_nonfinite':U[0,0,0]=np.nan
    def before(fun,z,state):
        if fault=='hook_mutation':z[0]+=.01
    optimizer=Mock(side_effect=AssertionError('NO_REAL_OPTIMIZER'))
    with patch.object(bridge.codec,'roundtrip',side_effect=AssertionError('NO_FALLBACK')),patch.object(
            scipy.optimize,'minimize',optimizer):
        with pytest.raises(ValueError):
            bridge.fit(g,J,U,x,prepared_initial=True,before_optimizer=before,after_optimizer=Mock())
        optimizer.assert_not_called()
    assert (d._double_factorized_explicit_cholesky,d.df_tensors_to_params)==original

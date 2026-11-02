"""Real fixed ffsim fitter, artificial arrays, mocked minimizer: zero iterations."""
import importlib
from pathlib import Path
import sys
from unittest.mock import Mock,patch
import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_ffsim_corrected_codec as b
from trottertracks import c1_h6_corrected_fit as m


def artificial(n,improper=False,angle=np.pi):
    from scipy.linalg import block_diag
    R=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    U=R if n==2 else block_diag(R,-np.eye(2),np.eye(2))
    U=np.array([U,np.eye(n),U.T])
    if improper:U[0,:,0]*=-1
    rng=np.random.default_rng(130+n);A=rng.normal(size=(3,n,n))*.04
    J=A+A.transpose(0,2,1)
    g=m.old.reconstruct(J,U)+.001*np.einsum('pq,rs->pqrs',np.eye(n),np.eye(n))
    return g,J,U


@pytest.mark.parametrize('n',[2,6])
@pytest.mark.parametrize('improper',[False,True])
@pytest.mark.parametrize('angle',[np.pi,np.pi-1e-12])
def test_actual_fitter_corrected_pack_x0_loss_gradient_and_restoration(n,improper,angle):
    from ffsim.linalg import util
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    g,J,U=artificial(n,improper,angle);a,u,x,report,_=b.codec.roundtrip(J,U)
    originals=(d._double_factorized_explicit_cholesky,d.df_tensors_to_params,util.df_tensors_to_params)
    observed={};before=Mock();after=Mock()
    def minimize(fun,x0,**kwargs):
        observed.update(kwargs);np.testing.assert_array_equal(x0,x)
        value,grad=fun(x0)
        assert value==pytest.approx(.5*np.sum((g-m.old.reconstruct(a,u))**2),abs=1e-12)
        reference,reference_grad=m.old.objective_gradient(g,x0,len(J))
        assert value==reference['objective'];np.testing.assert_array_equal(grad,reference_grad)
        # Independent directional finite differences, including rotation/J coordinates.
        for k in (0,n*(n-1)//2, len(x0)-1):
            dx=np.zeros_like(x0);dx[k]=1e-5
            def loss(z):
                aa,uu=util.df_tensors_from_params(z,len(J),n,None,real=True)
                return .5*np.sum((g-m.old.reconstruct(aa,uu))**2)
            assert (loss(x0+dx)-loss(x0-dx))/(2e-5)==pytest.approx(grad[k],abs=2e-9,rel=2e-5)
        return scipy.optimize.OptimizeResult(x=x0.copy(),jac=grad,fun=value,success=True,status=0,message='MOCK_NO_ITERATIONS',nit=0,nfev=1,njev=1)
    forbidden=Mock(side_effect=AssertionError('NO_OLD_LOGM_OR_INITIALIZER'))
    with patch.object(scipy.optimize,'minimize',side_effect=minimize) as optimizer,patch.object(scipy.linalg,'logm',forbidden):
        ja,ua,res=b.fit(g,J,U,x,before_optimizer=before,after_optimizer=after)
        optimizer.assert_called_once()
    assert observed==dict(method='L-BFGS-B',jac=True,callback=None,options=None)
    np.testing.assert_array_equal(ja,J);np.testing.assert_allclose(ua,u,atol=0,rtol=0)
    np.testing.assert_array_equal(before.call_args.args[1],x)
    assert before.call_args.args[2]['codec']['version']==b.codec.VERSION
    before.assert_called_once();after.assert_called_once();forbidden.assert_not_called()
    assert (d._double_factorized_explicit_cholesky,d.df_tensors_to_params,util.df_tensors_to_params)==originals


@pytest.mark.parametrize('failure',['wrong_x0','nonorthogonal','nonfinite_target','before_hook','optimizer'])
def test_failure_no_fallback_and_module_bindings_restored(failure):
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    g,J,U=artificial(2,True);x=b.codec.roundtrip(J,U)[2]
    initial,packing=d._double_factorized_explicit_cholesky,d.df_tensors_to_params
    if failure=='wrong_x0':x=x+.001
    if failure=='nonorthogonal':U[0,0,0]=.7
    if failure=='nonfinite_target':g[0,0,0,0]=np.nan
    before=Mock(side_effect=RuntimeError('BEFORE_STOP') if failure=='before_hook' else None)
    optimizer=Mock(side_effect=RuntimeError('MOCK_OPTIMIZER_STOP'))
    with patch.object(scipy.optimize,'minimize',optimizer):
        with pytest.raises((ValueError,RuntimeError)):b.fit(g,J,U,x,before_optimizer=before,after_optimizer=Mock())
        assert scipy.optimize.minimize is optimizer
    assert d._double_factorized_explicit_cholesky is initial and d.df_tensors_to_params is packing
    assert optimizer.call_count==(1 if failure=='optimizer' else 0)


@pytest.mark.parametrize('optimizer_success',[True,False])
def test_artificial_prepare_and_fit_worker_records_actual_parameters_no_iterations(tmp_path,optimizer_success):
    g,J,U=artificial(2,True);np.savez(tmp_path/'input.npz',g=g)
    np.savez(tmp_path/'initial.npz',J=J,U=U)
    c=dict(original_input='input.npz',initial_arrays='initial.npz',XDF_full='initial.npz',rank=3)
    m.prepare_worker(tmp_path,tmp_path/'prepare',c)
    m.io.save(tmp_path/'initial_sector_gate.json',dict(status='PASS'))
    def optimizer(fun,x0,**kwargs):
        value,jac=fun(x0)
        return scipy.optimize.OptimizeResult(x=x0,jac=jac,fun=value,success=optimizer_success,status=0 if optimizer_success else 2,message='ARTIFICIAL_NO_ITERATION',nit=0,nfev=1,njev=1)
    with patch.object(scipy.optimize,'minimize',side_effect=optimizer) as spy:
        m.fit_worker(tmp_path,tmp_path/'fit',c)
    assert spy.call_count==1
    assert m.io.read(tmp_path/'fit/optimizer.json')['success']==optimizer_success
    assert m.io.read(tmp_path/'fit/STOP.json')['scientific_units']==sum(m.FIT_LIMITS.values())
    initial=m.load_npz(tmp_path/'fit/initial_parameters.npz');final=m.load_npz(tmp_path/'fit/optimizer_parameters.npz')
    np.testing.assert_array_equal(initial['x'],final['x']);np.testing.assert_array_equal(initial['gradient'],final['jac'])
    assert m.io.read(tmp_path/'fit/final_value_gradient.json')['optimizer_jac_maxabs_difference']==0

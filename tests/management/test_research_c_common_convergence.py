"""Invented tensors only. Actual ffsim dispatch, optimizer always mocked."""
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

import numpy as np
import pytest
import scipy.optimize

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from trottertracks import c_common_convergence as m
from trottertracks import c_rcdf_comparator as rc


@pytest.mark.parametrize("role",["rc2","pr"])
@pytest.mark.parametrize("options",[None,m.OPTIONS])
def test_actual_ffsim_initial_x0_objective_and_last_hop_options(role,options):
    n,rank=4,3;rng=np.random.default_rng(532)
    a=rng.normal(size=(rank,n,n))*.1;J=a+a.transpose(0,2,1)
    U=np.repeat(np.eye(n)[None],rank,axis=0)
    U[0,:,0]*=-1;U[1,:2,:2]=-np.eye(2)
    J,U,x,_,_=rc.pr.corrected.codec.roundtrip(J,U)
    g=rc.pr.corrected.codec.reconstruct(J,U)+1e-4*np.einsum("pq,rs->pqrs",np.eye(n),np.eye(n))
    module=rc if role=="rc2" else rc.pr
    spec=(rc.initial_spec(J,ld=1,loss_scale=1.) if role=="rc2" else rc.pr.Objective(rank,n,1,1.,1.,.1))
    expected=module.make_value_and_grad(spec)(x,g)
    seen=[]
    def optimizer(fun,x0,**kwargs):
        np.testing.assert_array_equal(x0,x)
        assert kwargs==dict(method="L-BFGS-B",jac=True,callback=None,options=options)
        value,grad=fun(x0)
        assert value==pytest.approx(float(expected[0]));np.testing.assert_array_equal(grad,expected[1])
        seen.append(kwargs)
        return SimpleNamespace(x=x0.copy(),jac=grad,fun=value,success=True,status=0,message="MOCK_NO_ITERATIONS")
    with patch.object(scipy.optimize,"minimize",side_effect=optimizer) as original:
        result=m.fit(module,g,J,U,x,spec,options=options,
            before_optimizer=lambda *args:None,after_optimizer=lambda *args:None)
        assert scipy.optimize.minimize is original
    assert len(seen)==1;np.testing.assert_array_equal(result[2].x,x)


@pytest.mark.parametrize("n",[2,4])
def test_frobenius_normalized_tensor_null_projection(n):
    rank=3;rng=np.random.default_rng(55+n)
    grad=rng.normal(size=rank*n*n).astype(np.float64)
    out=m.first_order(grad,rank,n)
    split=rank*n*(n-1)//2;upper=grad[split:].reshape(rank,-1)
    rows,cols=np.triu_indices(n)
    G=np.zeros((rank,n,n));G[:,rows,cols]=upper
    for i in range(n):
        for j in range(i+1,n):G[:,j,i]=G[:,i,j]=upper[:,list(zip(rows,cols)).index((i,j))]/2
    independent=G.sum(axis=(1,2))/n;independent-=independent.mean()
    np.testing.assert_allclose(out["identity_tangent_coefficients"],independent,atol=1e-14)
    d=independent/np.linalg.norm(independent)
    delta=d[:,None,None]*np.ones((rank,n,n))/n
    assert np.linalg.norm(delta)==pytest.approx(1.)
    assert np.sum(G*delta)==pytest.approx(out["identity_tangent"]["value"])
    J=rng.normal(size=(rank,n,n));J=J+J.transpose(0,2,1)
    U=np.repeat(np.eye(n)[None],rank,axis=0)
    np.testing.assert_allclose(rc.pr.corrected.codec.reconstruct(J,U),
        rc.pr.corrected.codec.reconstruct(J+.03*delta,U),atol=1e-14)


def test_null_norm_can_fail_while_native_infinity_passes():
    g=np.zeros(128);g[48:]=np.repeat([4e-6]*4+[-4e-6]*4,10)
    out=m.first_order(g,8,4)
    assert out["parameter_gradient"]["status"]=="MET"
    assert out["identity_tangent"]["status"]=="NOT_MET"
    assert not out["convergence_criteria_met"]


@pytest.mark.parametrize("success,size,message,met,qualified",[
    (True,0.,"REL_REDUCTION",True,True),
    (True,.1,"REL_REDUCTION",False,False),
    (False,0.,"ABNORMAL_TERMINATION_IN_LNSRCH",True,False),
    (False,.1,"ITERATION_LIMIT",False,False),
    (True,1e-5,"REL_REDUCTION",False,False)])
def test_success_stopping_reason_and_common_criteria_separate(success,size,message,met,qualified):
    grad=np.zeros(12);grad[0]=size
    result=SimpleNamespace(x=np.zeros(12),jac=grad,fun=.1,success=success,status=0 if success else 2,message=message)
    report=m.terminal(.1,grad,result,3,2)
    assert report["convergence_criteria_met"]==met
    assert report["qualified"]==qualified
    assert report["solver_success"]==success
    assert not report["rigorous_stationarity"]


def test_nonfinite_and_mismatched_independent_gradient_stop_and_restore():
    result=SimpleNamespace(x=np.zeros(12),jac=np.zeros(12),fun=.1,success=True,status=0,message="mock")
    with pytest.raises(ValueError,match="MISMATCH"):m.terminal(.1,np.ones(12),result,3,2)
    with pytest.raises(ValueError,match="FINITE"):m.first_order(np.full(12,np.nan),3,2)
    original=scipy.optimize.minimize
    with pytest.raises(ValueError,match="DISPATCH"):
        with m.solver_options(m.OPTIONS):
            scipy.optimize.minimize(None,np.zeros(12),method="OTHER",bounds=[])
    assert scipy.optimize.minimize is original

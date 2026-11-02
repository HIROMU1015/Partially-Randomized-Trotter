"""Invented J/U only; real ffsim route with optimizer blocked at zero iterations."""
from pathlib import Path
import sys,importlib
import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_structural_support as m
from trottertracks import c_pr_aware_df as objective
from trottertracks.c1_h6_initialization import reconstruct


def fixture():
    rng=np.random.default_rng(18028);a=rng.normal(size=(8,4,4))*.03;J=a+a.transpose(0,2,1)
    U=np.array([np.linalg.qr(rng.normal(size=(4,4)))[0] for _ in range(8)])
    U[0,:,0]*=-1;U[1]=np.diag([-1.,-1.,1.,1.])
    g=reconstruct(J,U)+1e-4*np.einsum('pq,rs->pqrs',np.eye(4),np.eye(4))
    a,u,x,info=objective.prepare(g,J,U,ld=2)
    return g,a,u,x,info


def test_compact_layout_projection_and_ffsim_roundtrip():
    from ffsim.linalg.util import df_tensors_from_params
    layout=m.Support();g,J,U,x,_=fixture();projected=layout.project(x)
    assert layout.full_size==128 and layout.rotations==48 and layout.size==110
    assert np.count_nonzero(x-projected)==18
    with pytest.raises(ValueError,match='NONZERO_FORBIDDEN'):layout.reduce(x)
    np.testing.assert_array_equal(layout.expand(layout.reduce(projected)),projected)
    a,u=df_tensors_from_params(projected,8,4,None,real=True);layout.assess(a)
    np.testing.assert_array_equal(a[:2],J[:2]);np.testing.assert_allclose(u,U,atol=1e-12,rtol=0)
    assert np.linalg.norm(reconstruct(a,u)-reconstruct(J,U))>1e-5 # intentional change, NOT equivalence


def test_indefinite_general_J_support_and_rank_one_degeneracy():
    from trottertracks.c1_general_k_adapter import general_k_coefficients
    layout=m.Support();a=np.diag([-1.,2.,-3.,4.]);a[0,1]=a[1,0]=.2;a[1,2]=a[2,1]=-.4;a[2,3]=a[3,2]=.3
    J=np.repeat(a[None],8,axis=0);layout.assess(J)
    assert np.linalg.matrix_rank(a)==4 and min(np.linalg.eigvalsh(a))<0<max(np.linalg.eigvalsh(a))
    coeff=objective.coefficients(J,np)
    for x,c in zip(J,coeff):
        K=np.repeat(np.repeat(x/2,2,axis=0),2,axis=1)
        np.testing.assert_allclose(c,[v for _,v in general_k_coefficients(K)[1:]],atol=1e-15)
        assert np.count_nonzero(c)<=24
    w=np.array([1.,-2.,0.,0.]);layout.assess(np.repeat(np.outer(w,w)[None],8,axis=0))
    J[2,0,3]=J[2,3,0]=1e-30
    with pytest.raises(ValueError,match='FORBIDDEN'):layout.assess(J) # no numerical pruning


def test_restricted_gradient_analytic_and_finite_difference():
    import jax.numpy as jnp
    g,J,U,x,info=fixture();layout=m.Support();x=layout.project(x);spec=objective.Objective(**{**info['spec'],'gamma':.1})
    vg=objective.make_value_and_grad(spec)
    fun=lambda x:vg(jnp.asarray(x),jnp.asarray(g))
    reduced=m.pullback(fun,layout);y=layout.reduce(x);v,dg=reduced(y);vf,full=fun(x)
    assert v==pytest.approx(float(vf));np.testing.assert_array_equal(dg,np.asarray(full)[list(layout.kept)])
    for k in (0,11,47,48,67,109):
        d=np.zeros_like(y);d[k]=2e-7
        fd=(reduced(y+d)[0]-reduced(y-d)[0])/(4e-7)
        assert fd==pytest.approx(dg[k],rel=1e-4,abs=1e-3)


def test_actual_ffsim_uses_compact_optimizer_no_iterations_and_restores(monkeypatch):
    from ffsim.linalg.util import df_tensors_from_params
    g,J,U,x,info=fixture();layout=m.Support();x=layout.project(x);J,U=df_tensors_from_params(x,8,4,None,real=True)
    spec=objective.Objective(**{**info['spec'],'gamma':.1});seen=[];hooks=[]
    def fake(fun,y0,**kw):
        assert y0.shape==(110,);np.testing.assert_array_equal(y0,layout.reduce(x))
        assert kw==dict(method='L-BFGS-B',jac=True,callback=None,options=None)
        v,dg=fun(y0);seen.append((y0.copy(),dg.copy()))
        return scipy.optimize.OptimizeResult(x=y0.copy(),jac=dg.copy(),fun=v,success=True,status=0,message='MOCK_ZERO_ITERATIONS',nit=0,nfev=1,njev=1)
    monkeypatch.setattr(scipy.optimize,'minimize',fake)
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition');oldfactory=d._make_compressed_value_and_grad
    def before(fun,z,state):np.testing.assert_array_equal(z,x);hooks.append('before')
    def after(fun,result):
        _,full=fun(result.x);assert np.max(abs(np.asarray(full)[list(set(range(128))-set(layout.kept))]))>0
        hooks.append('after')
    a,u,res=m.fit(g,J,U,x,spec,layout,before_optimizer=before,after_optimizer=after)
    assert hooks==['before','after'] and len(seen)==1 and res.nit==0
    assert scipy.optimize.minimize is fake and d._make_compressed_value_and_grad is oldfactory
    np.testing.assert_array_equal(res.restricted_optimizer_x,seen[0][0]);np.testing.assert_array_equal(res.restricted_optimizer_jac,seen[0][1])
    np.testing.assert_array_equal(res.jac,layout.expand(seen[0][1]));layout.assess(a)
    np.testing.assert_allclose(reconstruct(a,u),reconstruct(J,U),atol=1e-12,rtol=0)


def test_optimizer_bridge_restores_on_failure(monkeypatch):
    def fail(*a,**k):raise RuntimeError('MOCK_BLOCK')
    monkeypatch.setattr(scipy.optimize,'minimize',fail);layout=m.Support()
    with pytest.raises(RuntimeError,match='MOCK_BLOCK'):
        with m.compact_optimizer(layout):scipy.optimize.minimize(lambda x:(0.,x),np.zeros(128),method='L-BFGS-B',jac=True,callback=None,options=None)
    assert scipy.optimize.minimize is fail


def test_real_prepare_fit_workers_zero_iterations(tmp_path,monkeypatch):
    from trottertracks import c_df_structural_execution as e
    g,J,U,x,info=fixture();np.savez(tmp_path/'input.npz',h=np.eye(4),g=g,constant=np.array(.31));np.savez(tmp_path/'initial.npz',J=J,U=U,x0=x);e.io.save(tmp_path/'scale.json',info)
    c=dict(input='input.npz',initial='initial.npz',scales='scale.json')
    e.worker(tmp_path,c,tmp_path/'prepare','prepare');e.io.save(tmp_path/'initial_gate.json',dict(status='SUPPORT_PASS'))
    def fake(fun,y,**kw):
        v,dg=fun(y);return scipy.optimize.OptimizeResult(x=y,jac=dg,fun=v,success=True,status=0,message='MOCK_ZERO_ITERATIONS',nit=0,nfev=1,njev=1)
    monkeypatch.setattr(scipy.optimize,'minimize',fake);e.worker(tmp_path,c,tmp_path/'fit','fit')
    assert e.io.read(tmp_path/'prepare/STOP.json')['scientific_units']==sum(e.PREPARE.values())
    assert e.io.read(tmp_path/'fit/STOP.json')['scientific_units']==sum(e.FIT.values())
    assert e.io.read(tmp_path/'fit/optimizer.json')['nit']==0
    with np.load(tmp_path/'fit/final.npz') as a:assert a['raw_optimizer_x'].shape==(110,) and a['x'].shape==(128,)

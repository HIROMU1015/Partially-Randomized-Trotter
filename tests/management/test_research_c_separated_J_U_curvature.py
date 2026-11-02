"""Invented tensors only. No saved H4 arrays/traces are numerically evaluated."""
from dataclasses import replace
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import sys
import numpy as np
import jax
import pytest
jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_separated_j_u_curvature as m
spec=importlib.util.spec_from_file_location('old_synthetic_tests',Path(__file__).with_name('test_research_c_separated_J_U.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
invented=helper.invented


@pytest.mark.parametrize('role',['pr','rc2'])
@pytest.mark.parametrize('n',[2,6])
def test_direction_normalization_chain_rule_and_chart_transport(role,n):
    from scipy.linalg import expm
    g,J,U,s=invented(role=role,n=n,rank=2,ld=1)
    U[0]=np.eye(n);U[0,:2,:2]=-np.eye(2);U[1,:,0]*=-1
    dimension=s.rank*n*(n-1)//2;d=np.linspace(-.4,.6,dimension);h=1e-6
    p=m.Coefficients(g,U,s);penalty=p.parts(p.pack(J))['penalty']
    eta=np.linspace(-.1,.15,dimension)
    # Independent SciPy exponential follows the documented upper-skew convention.
    indices=np.triu_indices(n,1)
    def independent(eta):
        K=np.zeros_like(U);K[:,indices[0],indices[1]]=eta.reshape(s.rank,-1)/np.sqrt(2)
        K=K-K.transpose(0,2,1)
        V=np.array([u@expm(k) for u,k in zip(U,K)])
        delta=m.linear.reconstruct(J,V)-g
        return .5*np.sum(delta*delta)/s.loss_scale+penalty,V
    value,V=independent(eta);ad,grad=m.old.u_value_gradient(eta,g,J,U,s,penalty)
    np.testing.assert_allclose(V,m.retract(U,eta),atol=3e-14)
    assert value==pytest.approx(ad,abs=3e-11)
    assert grad@d==pytest.approx((independent(eta+h*d)[0]-independent(eta-h*d)[0])/(2*h),abs=1e-7)
    _,physical=m.old.u_value_gradient(np.zeros(dimension),g,J,V,s,penalty)
    derivative=(independent(eta+h*d)[1]-independent(eta-h*d)[1])/(2*h)
    omega=V.transpose(0,2,1)@derivative
    coordinates=np.sqrt(2)*omega[:,indices[0],indices[1]].ravel()
    assert physical@coordinates==pytest.approx(grad@d,abs=2e-7)
    residual,D=m.residual_jacobian(eta,g,J,U,s)
    np.testing.assert_allclose(D.T@residual,grad,atol=3e-12,rtol=1e-10)
    np.testing.assert_allclose(D@d,(m.residual_jacobian(eta+h*d,g,J,U,s)[0]-m.residual_jacobian(eta-h*d,g,J,U,s)[0])/(2*h),atol=5e-8)
    _,plus=independent(h*d);_,minus=independent(-h*d)
    assert np.linalg.norm((plus-minus)/(2*h))==pytest.approx(np.linalg.norm(d),rel=1e-8)


def test_affine_coefficient_chart_gradient_exact_Hessian_and_objective():
    g,J,U,s=invented();p=m.Coefficients(g,U,s);c=m.CoefficientChart(p,J)
    z=np.linspace(-.02,.03,len(c.x0));direction=np.linspace(.1,-.2,len(z));x=c.x0+c.W@z
    f,gz=c.vg(z);v,gx=p.vg(x,independent=True)
    assert f==pytest.approx(v,abs=1e-13)
    np.testing.assert_allclose(gz,c.W.T@gx,atol=3e-13)
    np.testing.assert_allclose(c.hess(z),c.W.T@p.hess(x)@c.W,atol=1e-10)
    h=1e-6
    np.testing.assert_allclose((c.vg(z+h*direction)[1]-c.vg(z-h*direction)[1])/(2*h),c.hess(z)@direction,atol=1e-7)
    assert np.linalg.eigvalsh(c.hess(z)).min()>=-1e-10


@pytest.mark.parametrize('shape',[(2,8),(8,2),(8,8)])
def test_metric_keeps_every_direction_and_does_not_add_penalty(shape):
    rng=np.random.default_rng(7);B=rng.normal(size=shape);B[:,-1]=0
    W,s=m.metric(B)
    assert W.shape==(shape[1],shape[1]) and len(s)==shape[1]
    assert np.linalg.matrix_rank(W)==shape[1]
    assert np.linalg.norm(B@W,2)<=1+1e-12
    with pytest.raises(ValueError,match='METRIC'):m.metric(B,np.inf)


def test_J_head_free_flat_nonunique_and_exact_RC2_reuse():
    s=m.pr.Objective(3,1,2,1.,1.,.1);p=m.Coefficients(np.array([[[[.8]]]]),np.ones((3,1,1)),s)
    outputs=[]
    for x in [[1.,-.3,.2],[3.,-2.3,.2]]:
        J,report=m.solve_j(p,np.array(x)[:,None,None]);outputs.append(J)
        assert report['stationarity']['status']=='MET'
        assert not report['objective_regularization_added']
        assert report['hessian_calls_in_block']>0
    assert not np.allclose(outputs[0],outputs[1])
    g,J,U,s=invented(role='rc2');p=m.Coefficients(g,U,s)
    a,info=m.solve_j(p,J);b,previous=m.old.solve_j(p,J)
    np.testing.assert_array_equal(a,b);assert info['method']==previous['method']=='augmented_lstsq_gelsd'


@pytest.mark.parametrize('role',['pr','rc2'])
def test_U_same_budget_stiff_artificial_gradient_improvement_and_refit_rebound(role):
    g,J,U,s=invented(role=role,scale=2.1839632053132894e-9)
    p=m.Coefficients(g,U,s);J,_=m.old.solve_j(p,J)
    _,old=m.old.solve_u(g,J,U,s)
    V,new=m.solve_u(g,J,U,s)
    assert new['final_U_tangent_l2_at_fixed_J'] < old['final_U_tangent_l2_at_fixed_J']*.01
    assert new['nfev']<=25 and new['progress']['decrease']>=-1e-12
    np.testing.assert_allclose(V.transpose(0,2,1)@V,np.broadcast_to(np.eye(2),V.shape),atol=1e-12)
    J2,_=m.solve_j(m.Coefficients(g,V,s),J)
    row,_,_=m.state(g,J2,V,s)
    # Exposes the alternating coupling; never call a solved fixed-J step a joint solution.
    assert row['U_stationarity']['value']>new['final_U_tangent_l2_at_fixed_J']*1000


@pytest.mark.parametrize('role',['pr','rc2'])
def test_full_alternation_convergent_soft_and_unqualified_stiff(role):
    g,J,U,s=invented(role=role);events=[]
    _,_,r=m.optimize(g,J,U,s,record=lambda kind,label,*a:events.append((kind,label)))
    assert r['first_order_qualified'] and r['classification']=='BOTH_FIRST_ORDER_MET'
    for i,(kind,label) in enumerate(events):
        if kind=='U':assert ('J',label) in events[i+1:]
    assert all(b['progress']['status']=='PASS_WITHIN_ENGINEERING_ALLOWANCE' for b in r['blocks'])
    g,J,U,s=invented(role=role,scale=2.1839632053132894e-9)
    _,_,r=m.optimize(g,J,U,s,replace(m.DEFAULT,max_cycles=3))
    assert not r['first_order_qualified'] and r['classification'] in ('U_UNQUALIFIED','BOTH_UNQUALIFIED')
    assert r['completed_cycles']<=3


def test_J_raw_warning_and_false_success_do_not_set_stationarity():
    g,J,U,s=invented();p=m.Coefficients(g,U,s)
    def unchanged(fun,z,**kw):
        value,grad=fun(z)
        return SimpleNamespace(x=z,jac=grad,fun=value,success=True,status=0,message='false success')
    _,r=m.solve_j(p,J,solver=unchanged)
    assert r['success'] and r['stationarity']['status']=='NOT_MET'
    zero=np.zeros_like(J);g=np.zeros_like(g);p=m.Coefficients(g,U,s)
    def warn(fun,z,**kw):
        value,grad=fun(z);return SimpleNamespace(x=z,jac=grad,fun=value,success=False,status=2,message='warning with correct zero')
    _,r=m.solve_j(p,zero,solver=warn)
    assert not r['success'] and r['stationarity']['status']=='MET'


def test_nonfinite_mismatch_and_objective_increase_STOP_before_acceptance():
    g,J,U,s=invented();p=m.Coefficients(g,U,s);raw=[]
    def nan(fun,z,**kw):return SimpleNamespace(x=np.full_like(z,np.nan),jac=np.zeros_like(z),fun=np.nan,success=False,status=2,message='nonfinite')
    with pytest.raises(ValueError,match='NONFINITE_OR_MALFORMED'):
        m.solve_j(p,J,solver=nan,record=lambda *a:raw.append(a))
    assert len(raw)==1 and raw[0][0]['fun'] is None
    def mismatch(fun,z,**kw):
        value,grad=fun(z);return SimpleNamespace(x=z,jac=grad+1,fun=value,success=True,status=0,message='wrong gradient')
    with pytest.raises(ValueError,match='GRADIENT_MISMATCH'):m.solve_j(p,J,solver=mismatch)
    def increase(fun,z,**kw):
        z=z+100.;value,grad=fun(z);return SimpleNamespace(x=z,jac=grad,fun=value,success=True,status=0,message='increase')
    with pytest.raises(ValueError,match='OBJECTIVE_INCREASE'):m.solve_j(p,J,solver=increase)


@pytest.mark.parametrize('j,u,expected',[(True,True,'BOTH_FIRST_ORDER_MET'),(False,True,'J_UNQUALIFIED'),(True,False,'U_UNQUALIFIED'),(False,False,'BOTH_UNQUALIFIED')])
def test_explicit_classification(j,u,expected):
    assert m.classification(dict(J_stationarity=dict(status='MET' if j else 'NOT_MET'),U_stationarity=dict(status='MET' if u else 'NOT_MET')))==expected


def test_cycle_cap_stall_and_U_STOP_refit(monkeypatch):
    g,J,U,s=invented()
    for mode,cycles in [('limit',64),('stall',3),('U_stop',1)]:
        count=Counter();states=[]
        def state(*a):
            states.append(1);return dict(components=dict(objective=1000. if mode=='stall' else 1000.-len(states)),J_stationarity=dict(status='MET'),U_stationarity=dict(status='NOT_MET'),first_order_met=False),None,None
        monkeypatch.setattr(m,'state',state)
        monkeypatch.setattr(m,'solve_j',lambda p,J,*a,**kw:(J,dict(success=True,status=0)))
        monkeypatch.setattr(m,'solve_u',lambda g,J,U,*a,**kw:(U,dict(success=False,status=-1 if mode=='U_stop' else 0,final_U_tangent_l2_at_fixed_J=1.)))
        def call(k,label,fn):count[k]+=1;return fn()
        _,_,out=m.optimize(g,J,U,s,call=call)
        assert count==dict(J_solve=cycles+1,U_solve=cycles,state_diagnostic=cycles+2)
        assert out['status']==dict(limit='OUTER_LIMIT_UNQUALIFIED',stall='STALLED_NOT_STATIONARY',U_stop='U_SOLVER_STOP_UNQUALIFIED')[mode]


def test_passive_new_dispatch_and_call_budget():
    import subprocess
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_separated_j_u_curvature_pilot as m;assert not any(x in sys.modules for x in ('numpy','scipy','jax','ffsim'));assert m.MAX_CALLS==415 and sum(m.WORKER.values())==202;assert 'old_metric_diagnostic' not in m.WORKER;print('PASS')"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True,capture_output=True)


def test_U_inconsistent_gradient_nonfinite_and_bad_retraction_STOP(monkeypatch):
    g,J,U,s=invented();raw=[]
    def nan(fun,z,**kw):
        return SimpleNamespace(x=np.full_like(z,np.nan),grad=np.zeros_like(z),cost=np.nan,
            success=False,status=-1,message='nonfinite',nfev=1,njev=0)
    with pytest.raises(ValueError,match='NONFINITE_OR_MALFORMED'):
        m.solve_u(g,J,U,s,solver=nan,record=lambda *a:raw.append(a))
    assert len(raw)==1 and raw[0][0]['fun'] is None
    def wrong(fun,z,jac,**kw):
        residual=fun(z);D=jac(z)
        return SimpleNamespace(x=z,grad=D.T@residual+1,cost=.5*float(residual@residual),
            success=True,status=1,message='wrong gradient',nfev=1,njev=1)
    with pytest.raises(ValueError,match='GRADIENT_MISMATCH'):m.solve_u(g,J,U,s,solver=wrong)
    monkeypatch.setattr(m,'retract',lambda U,eta:U*np.array([1.01,1.]))
    with pytest.raises(ValueError,match='ORTHOGONALITY'):m.solve_u(g,J,U,s)


def artificial_study():
    """Fixed seed/scales, old-vs-selected controls; never saved molecular arrays."""
    rows=[]
    for role in ('pr','rc2'):
        for scale in (1.,2.1839632053132894e-9):
            g,J,U,s=invented(role=role,scale=scale)
            p=m.Coefficients(g,U,s);old_J,old_info=m.old.solve_j(p,J);new_J,new_info=m.solve_j(p,J)
            _,old_u=m.old.solve_u(g,old_J,U,s);_,new_u=m.solve_u(g,old_J,U,s)
            trajectories={}
            for name,algorithm in [('v1',m.old),('curvature_v2',m)]:
                _,_,a=algorithm.optimize(g,J,U,s)
                trajectories[name]=dict(status=a['status'],cycles=a['completed_cycles'],terminal=a['terminal'],
                    counts=dict(Counter(b['block'] for b in a['blocks'])),
                    raw_success=dict(Counter(str(b['success']) for b in a['blocks'])),
                    solver_nfev=sum(b.get('nfev',0) for b in a['blocks']))
            rows.append(dict(role=role,invented_seed=41,norb=2,rank=3,ld=1,loss_scale=scale,
                J_shared_start={name:dict(raw_status=info['status'],raw_success=info['success'],
                    objective=info['independent_objective'],stationarity=info['stationarity'])
                    for name,info in [('v1',old_info),('curvature_v2',new_info)]},
                U_shared_fixed_J={name:{k:info[k] for k in ('status','success','nfev',
                    'initial_U_tangent_l2','final_U_tangent_l2_at_fixed_J','independent_objective')}
                    for name,info in [('v1',old_u),('curvature_v2',new_u)]},
                trajectories=trajectories))
    return dict(schema='C_SEPARATED_CURVATURE_SYNTHETIC_COMPARISON_V1',rows=rows,
        actual_molecular_array_decodes=0,actual_scientific_calls=0,
        interpretation='Numerical technical controls: no H4 performance, global convergence, superiority or multi-start stability claim.')


if __name__=='__main__':
    import json
    target=Path(sys.argv[1])
    with target.open('x') as f:json.dump(artificial_study(),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

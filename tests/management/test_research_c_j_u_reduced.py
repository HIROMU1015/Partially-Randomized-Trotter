"""Invented tensors ONLY; no saved molecular arrays/trace reanalysis."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
import jax
jax.config.update('jax_enable_x64', True)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_j_u_reduced as m
from test_research_c_separated_J_U import invented


def at(g,J,U,spec,eta):
    o=m.ReducedOracle(g,J,U,spec)
    return o.evaluate(np.asarray(eta,dtype=float))


@pytest.mark.parametrize('role',['rc2','pr'])
def test_envelope_finite_difference_RESOLVES_inner_at_every_perturbed_U(role):
    g,J,U,s=invented(role=role)
    zero=np.zeros(3);direction=np.array([.3,-.7,.2]);h=2e-5
    count=Counter()
    def call(k,l,f):count[k]+=1;return f()
    o=m.ReducedOracle(g,J,U,s,call=call)
    p=o.evaluate(zero);pplus=o.evaluate(h*direction);pminus=o.evaluate(-h*direction)
    np.testing.assert_allclose(p['gradient']@direction,(pplus['value']-pminus['value'])/(2*h),atol=3e-7,rtol=1e-5)
    assert count==dict(inner_J_solve=6,reduced_state=3,envelope_gradient=3)
    assert o.evaluate(zero) is p and len(o.points)==3
    assert p['qualification']['rigorous_gradient_error_bound'] is None
    q=m.Coefficients(g,U,s);x=q.pack(p['J'])
    if role=='rc2':
        reference=np.linalg.solve(q.hess(x),q.A.T@g.ravel()/s.loss_scale)
        np.testing.assert_allclose(x,reference,atol=2e-14)
    else:
        assert np.linalg.eigvalsh(q.hess(x)).min()>-1e-12


def test_PR_only_invariant_gauge_allows_nonunique_smooth_envelope():
    g,J,U,s=invented(n=2,rank=3,ld=2)
    p=m.Coefficients(g,U,s);check=m.applicability(p)
    assert check['admitted'] and check['universal_nullity']==1
    N=np.zeros_like(J);N[0]=1.;N[1]=-1.
    x=p.pack(J);n=p.pack(N)
    np.testing.assert_allclose(p.A@n,0,atol=2e-15)
    np.testing.assert_allclose(p.T@n,0,atol=0)
    a=at(g,J,U,s,np.zeros(3));b=at(g,J+N,U,s,np.zeros(3))
    assert not np.allclose(a['J'],b['J'])
    np.testing.assert_allclose(a['value'],b['value'],atol=2e-12)
    np.testing.assert_allclose(a['gradient'],b['gradient'],atol=2e-7)
    # Extra head degeneracy: no smooth reduced gradient is silently assumed.
    U[1]=U[0]
    assert not m.applicability(m.Coefficients(g,U,s))['admitted']
    _,_,r=m.optimize(g,J,U,s)
    assert r['status']=='REDUCED_GRADIENT_UNQUALIFIED'
    assert not r['first_order_qualified'] and r['completed_cycles']==0


def test_nonunique_value_can_be_smooth_or_nonsmooth_not_a_DF_claim():
    # Convex inner minimization over [-1,1]. At u=0 all j minimize u*j.
    # Partial derivatives j differ -> Phi=-|u| has a kink.
    h=1e-5
    value=lambda u:min(-u,u)
    assert (value(h)-value(0))/h==-1.
    assert (value(0)-value(-h))/h==1.
    # Invariant gauge example has Phi(u)=u^2 despite nonunique (j0,j1).
    f=lambda j,u:.5*(j[0]+j[1]-u)**2+u*u
    assert f([0,.3],.3)==pytest.approx(f([4,-3.7],.3))


def test_inexact_inner_error_and_U_gradient_sensitivity():
    g,J,U,s=invented(role='rc2');p=m.Coefficients(g,U,s)
    a=at(g,J,U,s,np.zeros(3));x=p.pack(a['J'])
    d=np.linspace(-.7,.9,len(x));errors=[]
    for epsilon in (1e-2,1e-4,1e-6):
        y=x+epsilon*d;j=p.unpack(y);parts=p.parts(y)
        _,gu=m.old.u_value_gradient(np.zeros(3),g,j,U,s,parts['penalty'])
        gj=p.vg(y,independent=True)[1]
        mu=2*s.strength/s.l2_scale
        assert np.linalg.norm(y-x)<=np.linalg.norm(gj)/mu+1e-13
        errors.append(float(np.linalg.norm(gu-a['gradient'])))
    assert errors[2]<errors[1]<errors[0]


def test_cross_curvature_sensitivity_and_Schur_RC2():
    g,J,U,s=invented(role='rc2');p=m.Coefficients(g,U,s)
    a=at(g,J,U,s,np.zeros(3));x=p.pack(a['J']);d=np.array([.3,-.7,.2]);h=2e-5
    eta=np.zeros(3);z=np.r_[eta,x]
    # Cross curvature is computed on invented input only. Production uses no
    # Hessian inverse or pseudoinverse. Finite-difference the existing AD.
    def grad(z):return m.coupled_value_gradient(z,g,U,s)[1]
    H=np.column_stack([(grad(z+h*e)-grad(z-h*e))/(2*h) for e in np.eye(len(z))])
    Huu,Huj,Hjj=H[:3,:3],H[:3,3:],H[3:,3:]
    assert np.linalg.eigvalsh(Hjj).min()>0
    red=Huu-Huj@np.linalg.solve(Hjj,Huj.T)
    b=at(g,J,U,s,h*d);c=at(g,J,U,s,-h*d)
    fd=(b['gradient']-c['gradient'])/(2*h)
    np.testing.assert_allclose(red@d,fd,atol=2e-7,rtol=2e-5)
    sensitivity=(p.pack(b['J'])-p.pack(c['J']))/(2*h)
    np.testing.assert_allclose(-np.linalg.solve(Hjj,Huj.T@d),sensitivity,atol=3e-7)
    assert np.linalg.norm(Huj)>1e-2 and np.linalg.norm((Huu-red)@d)>1e-3


@pytest.mark.parametrize('role',['rc2','pr'])
def test_reduced_progress_orthogonality_and_physical_simultaneous_first_order(role):
    g,J,U,s=invented(role=role)
    a,V,r=m.optimize(g,J,U,s)
    assert r['status']=='BOTH_FIRST_ORDER_MET' and r['first_order_qualified']
    assert r['terminal']['J_stationarity']['status']==r['terminal']['U_stationarity']['status']=='MET'
    assert r['points']<=m.DEFAULT.max_points and r['inner_solves']==2*r['points']
    np.testing.assert_allclose(V.transpose(0,2,1)@V,np.broadcast_to(np.eye(2),V.shape),atol=2e-14)


def test_PR_inner_unqualified_NEVER_calls_BFGS_or_returns_reduced_gradient():
    g,J,U,s=invented()
    def forbidden(*a,**k):pytest.fail('outer after unqualified inner')
    _,_,r=m.optimize(g,J,U,s,replace(m.DEFAULT,j_maxiter=1),solver=forbidden)
    assert r['status']=='J_UNQUALIFIED' and not r['first_order_qualified']
    assert r['points']==1 and r['inner_solves']==2


def test_refinement_instability_and_false_success_do_not_qualify(monkeypatch):
    g,J,U,s=invented(role='rc2')
    # Deliberately impossible numerical stability requirement, no tolerance retry.
    _,_,r=m.optimize(g,J,U,s,replace(m.DEFAULT,refinement_U_l2=1e-30))
    assert r['status']=='REDUCED_GRADIENT_UNQUALIFIED'
    assert not r['first_order_qualified']


@pytest.mark.parametrize('mode',['points','outer'])
def test_reduced_also_has_nonconverged_examples_and_no_retry(mode):
    g,J,U,s=invented(role='rc2')
    policy=replace(m.DEFAULT,max_points=1) if mode=='points' else replace(m.DEFAULT,outer_maxiter=1)
    _,_,r=m.optimize(g,J,U,s,policy)
    assert not r['first_order_qualified']
    assert r['status'] in ('OUTER_LIMIT_UNQUALIFIED','U_UNQUALIFIED')
    assert r['retry']==r['resume']==0


@pytest.mark.parametrize('role',['rc2','pr'])
def test_joint_local_alternative_gradient_same_point_no_envelope(role):
    g,J,U,s=invented(role=role);p=m.Coefficients(g,U,s)
    z=np.r_[np.zeros(3),p.pack(J)];v,h=m.coupled_value_gradient(z,g,U,s)
    q,gj,gu=m.state(g,J,U,s)
    assert v==pytest.approx(q['components']['objective'],abs=1e-14)
    np.testing.assert_allclose(h,np.r_[gu,gj],atol=1e-12)
    d=np.linspace(-.4,.6,len(z));eps=1e-5
    fd=(m.coupled_value_gradient(z+eps*d,g,U,s)[0]-m.coupled_value_gradient(z-eps*d,g,U,s)[0])/(2*eps)
    assert fd==pytest.approx(h@d,abs=3e-9)


def artificial_comparison():
    from scipy.optimize import minimize
    rows=[]
    for role in ('rc2','pr'):
        g,J,U,s=invented(role=role)
        # Same original point/objective; stop the old fixed-J block method at
        # four cycles, then compare full registered reduced prototype.
        _,_,alternating=m.blocks.optimize(g,J,U,s,replace(m.blocks.DEFAULT,max_cycles=4))
        inner=[];counts=Counter()
        def record(kind,label,data,arrays):
            if kind=='inner_summary':inner.extend(data['solves'])
        def call(kind,label,fn):counts[kind]+=1;return fn()
        a,V,reduced=m.optimize(g,J,U,s,record=record,call=call)
        reduced['named_calls']=dict(counts)
        reduced['inner_evaluation_totals']={key:sum(b.get(key,0) for b in inner)
            for key in ('nfev','njev','nhev','value_gradient_calls_in_block','hessian_calls_in_block','linear_solves')}
        p=m.Coefficients(g,U,s);z=np.r_[np.zeros(3),p.pack(J)]
        joint=minimize(lambda z:m.coupled_value_gradient(z,g,U,s),z,jac=True,method='BFGS',
                       options=dict(maxiter=64,gtol=1e-7))
        Jc=p.unpack(joint.x[3:]);Uc=m.retract(U,joint.x[:3]);jc,ji=m.blocks.solve_j(m.Coefficients(g,Uc,s),Jc)
        coupled=m.state(g,jc,Uc,s)[0]
        rows.append(dict(role=role,seed=41,norb=2,rank=3,ld=1,scale=1.,
            alternating=alternating,reduced=reduced,
            coupled_alternative=dict(terminal=coupled,raw=m.old._raw(joint),post_J=ji),
            interpretation='Different classic evaluation budgets; no speedup, global order or H4 claim'))
    return dict(schema='C_J_U_REDUCED_ARTIFICIAL_V1',rows=rows,real_molecular_decodes=0,scientific_calls=0)


def test_fixed_J_TRF_stall_rebound_and_reduced_improvement():
    study=artificial_comparison()
    for row in study['rows']:
        a,b=row['alternating'],row['reduced']
        assert not a['first_order_qualified']
        assert b['first_order_qualified']
        assert b['terminal']['components']['objective']<a['terminal']['components']['objective']-1e-6
        assert b['terminal']['U_stationarity']['value']<a['terminal']['U_stationarity']['value']


def test_invalid_inputs_nonfinite_and_old_source_untouched():
    g,J,U,s=invented(role='rc2')
    o=m.ReducedOracle(g,J,U,s)
    with pytest.raises(ValueError,match='FINITE_BINARY64'):o.evaluate(np.ones(3,dtype=np.float32))
    with pytest.raises(ValueError,match='FINITE_BINARY64'):o.evaluate(np.full(3,np.nan))
    U=U*1.1
    with pytest.raises(ValueError,match='ORTHOGONALITY'):m.optimize(g,J,U,s)


def test_large_cross_curvature_causes_registered_alternating_stall_toy():
    # Analytic coupled quadratic, separate from the genuine DF comparison.
    # F=.5 a(j-u)^2+.5 b(j+u-2)^2. Hjj=a+b>0, Hju=b-a.
    # Exact fixed-j TRF steps nearly freeze, while Phi has modest curvature.
    from scipy.optimize import least_squares, minimize
    a,b=1e8,1e-3
    jstar=lambda u:((a-b)*u+2*b)/(a+b)
    F=lambda j,u:.5*a*(j-u)**2+.5*b*(j+u-2)**2
    gu=lambda j,u:a*(u-j)+b*(j+u-2)
    u=0.;changes=[];rebound=[]
    for _ in range(3):
        j=jstar(u);before=F(j,u)
        r=least_squares(lambda v:np.array([np.sqrt(a)*(j-v[0]),np.sqrt(b)*(j+v[0]-2)]),[u],
                        jac=lambda v:np.array([[-np.sqrt(a)],[np.sqrt(b)]]),gtol=1e-12)
        u=float(r.x[0]);fixed=abs(gu(j,u));j=jstar(u)
        changes.append(abs(before-F(j,u)));rebound.append([fixed,abs(gu(j,u))])
    assert all(change<m.DEFAULT.stall_atol for change in changes)
    assert all(after>m.DEFAULT.u_tangent_l2 for _,after in rebound)
    def reduced(v):
        j=jstar(v[0]);return F(j,v[0]),np.array([gu(j,v[0])])
    res=minimize(reduced,[0.],jac=True,method='BFGS',options=dict(gtol=1e-8))
    assert res.success and abs(res.x[0]-1.)<1e-5
    assert res.fun<F(j,u)/1e6


def test_false_outer_gradient_is_saved_before_correctness_STOP():
    from types import SimpleNamespace
    g,J,U,s=invented(role='rc2');records=[]
    def bad(fun,x,**kw):
        value,grad=fun(x)
        return SimpleNamespace(x=x,jac=grad+1.,fun=value,success=True,status=0,message='invented false derivative',nit=0,nfev=1,njev=1)
    with pytest.raises(ValueError,match='OUTER_RAW_ENDPOINT'):
        m.optimize(g,J,U,s,solver=bad,record=lambda *args:records.append(args))
    assert any(x[0]=='outer_solver_result' and 'jac' in x[3] for x in records)


def test_independent_U_gradient_mismatch_stops(monkeypatch):
    g,J,U,s=invented(role='rc2');original=m.blocks.residual_jacobian
    def bad(*a):
        r,D=original(*a);return r,D+1.
    monkeypatch.setattr(m.blocks,'residual_jacobian',bad)
    with pytest.raises(ValueError,match='SAME_POINT_VALUE_OR_GRADIENT'):
        m.optimize(g,J,U,s)


def test_full_point_budget_and_cache_are_explicit_without_extra_inner_solve():
    from types import SimpleNamespace
    g,J,U,s=invented(role='rc2');counts=Counter()
    def call(k,l,f):counts[k]+=1;return f()
    def trials(fun,x,**kw):
        for i in range(1,4):fun(x+i*np.array([.01,.02,-.01]))
        pytest.fail('must stop before fourth point')
    _,_,r=m.optimize(g,J,U,s,replace(m.DEFAULT,max_points=3),call=call,solver=trials)
    assert r['status']=='OUTER_LIMIT_UNQUALIFIED' and r['points']==3
    assert counts==dict(inner_J_solve=6,reduced_state=3,envelope_gradient=3,outer_driver=1)
    assert r['accepted_labels']==['0']  # evaluated trials are not accepted.


def test_degenerate_PR_joint_alternative_does_not_need_argmin_derivative():
    from scipy.optimize import minimize
    g,J,U,s=invented(rank=3,ld=2);U[1]=U[0]
    p=m.Coefficients(g,U,s)
    assert not m.applicability(p)['admitted']
    z=np.r_[np.zeros(3),p.pack(J)]
    result=minimize(lambda z:m.coupled_value_gradient(z,g,U,s),z,jac=True,method='BFGS',
                    options=dict(maxiter=64,gtol=1e-7))
    V=m.retract(U,result.x[:3]);j=p.unpack(result.x[3:])
    jr,_=m.blocks.solve_j(m.Coefficients(g,V,s),j)
    row=m.state(g,jr,V,s)[0]
    assert row['components']['objective']<p.parts(p.pack(J))['objective']-1e-3
    assert row['orthogonality_maxabs']<1e-12


if __name__=='__main__':
    import json
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--synthetic-report',type=Path,required=True);a=p.parse_args()
    with a.synthetic_report.open('x') as f:json.dump(artificial_comparison(),f,indent=2,allow_nan=False)

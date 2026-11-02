"""Invented arrays only; no saved molecular data may be opened by these tests."""
from decimal import Decimal,localcontext
from dataclasses import replace
from pathlib import Path
import json,sys,time
import numpy as np
import pytest
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_coefficient_point as m
from test_research_c_separated_J_U import invented


def decimal_tensor_gradient(chart,z):
    """80-digit product/adjoint reference; gamma=0 artificial test only."""
    _,J=chart.point(z);p=chart.p;n=p.spec.norb;rank=p.spec.rank
    assert n==2 and p.spec.gamma==0
    with localcontext() as ctx:
        ctx.prec=80;d=lambda x:Decimal.from_float(float(x))
        U=[[[d(v) for v in row] for row in a] for a in p.mapping.U]
        j=[[[d(v) for v in row] for row in a] for a in J]
        residual={}
        for a,b,c,e in np.ndindex((n,)*4):
            residual[a,b,c,e]=sum(j[t][i][k]*U[t][a][i]*U[t][b][i]*U[t][c][k]*U[t][e][k]
                for t in range(rank) for i in range(n) for k in range(n))-d(p.g[a,b,c,e])
        gx=[];slot=0
        for t in range(rank):
            for i,k in zip(*np.triu_indices(n)):
                v=Decimal(0)
                for (a,b,c,e),res in residual.items():
                    der=U[t][a][i]*U[t][b][i]*U[t][c][k]*U[t][e][k]
                    if i!=k:der+=U[t][a][k]*U[t][b][k]*U[t][c][i]*U[t][e][i]
                    v+=res*der/d(p.w[slot])/d(p.spec.loss_scale)
                gx.append(v);slot+=1
        return np.array([float(sum(d(chart.W[i,k])*gx[i] for i in range(len(gx)))) for k in range(len(gx))])


def failing_invented(seed):
    g,J,U,s=invented(seed=seed,n=4,rank=8,ld=2,scale=2.1839632053132894e-9)
    p=m.legacy.Coefficients(g,U,s);records=[]
    with pytest.raises(ValueError,match='J_POINT_CONSISTENCY_MISMATCH'):
        m.legacy.solve_j(p,J,record=lambda a,b:records.append((a,b)))
    return p,J,records[0][1],records[-1][0]


@pytest.mark.parametrize('seed',[41,42])
def test_reproduces_old_failure_and_new_point_agrees_without_relaxation(seed):
    p,J,raw,old=failing_invented(seed);chart=m.CoefficientChart(p,J);z=raw['z']
    assert not old['consistency']['chart_vs_physical_gradient']['PASS']
    assert chart.assess(z)['checks']['chart_vs_physical_gradient']['PASS']
    np.testing.assert_array_equal(chart.W,m.legacy.CoefficientChart(p,J).W)
    assert chart.policy.gradient_atol==1e-8 and chart.policy.gradient_rtol==1e-7
    assert chart.assess(z)['rigorous_gradient_error_bound'] is None


@pytest.mark.parametrize('scale',[1.,1e-9,1e-12])
def test_independent_Decimal_gradient_accuracy_not_only_internal_agreement(scale):
    g,J,U,s=invented(scale=scale);s=replace(s,gamma=0.)
    g=m.legacy.linear.reconstruct(J,U);p=m.legacy.Coefficients(g,U,s);chart=m.CoefficientChart(p,J)
    z=np.zeros(len(chart.x0));ref=decimal_tensor_gradient(chart,z)
    error=np.linalg.norm(chart.vg(z)[1]-ref)
    old_error=np.linalg.norm(chart.original.vg(z)[1]-ref)
    assert error<max(1e-15,old_error/30)


@pytest.mark.parametrize('seed',[41,42])
def test_general_signed_J_offdiagonal_chain_hessian_and_objective(seed):
    g,J,U,s=invented(seed=seed);p=m.legacy.Coefficients(g,U,s);chart=m.CoefficientChart(p,J)
    z=np.linspace(-.03,.04,len(chart.x0));d=np.linspace(.2,-.1,len(z));h=1e-5
    value,grad=chart.vg(z);x,j=chart.point(z)
    assert value==pytest.approx(p.vg(x,independent=True)[0],abs=1e-13)
    assert (chart.vg(z+h*d)[0]-chart.vg(z-h*d)[0])/(2*h)==pytest.approx(grad@d,abs=1e-9)
    np.testing.assert_allclose((chart.vg(z+h*d)[1]-chart.vg(z-h*d)[1])/(2*h),chart.hess(z)@d,atol=1e-8,rtol=1e-6)
    assert np.max(abs(chart.hess(z)-chart.hess(z).T))<1e-12
    np.testing.assert_allclose(chart.hess(z),chart.W.T@p.hess(x)@chart.W,atol=1e-11)


def test_gauge_no_new_constraints_det_sign_and_point_preserved():
    g,J,U,s=invented(rank=3,ld=2);p=m.legacy.Coefficients(g,U,s);a=m.CoefficientChart(p,J)
    N=np.zeros_like(J);N[0]=1.;N[1]=-1.
    assert p.vg(p.pack(J),independent=True)[0]==pytest.approx(p.vg(p.pack(J+N),independent=True)[0],abs=1e-13)
    before=J.copy();a.assess(np.zeros(len(a.x0)));np.testing.assert_array_equal(before,J)
    np.testing.assert_array_equal(a.W,a.original.W)
    flipped=U.copy();flipped[:,:,0]*=-1
    b=m.CoefficientChart(m.legacy.Coefficients(g,flipped,s),J)
    assert np.allclose(np.linalg.det(flipped),-np.linalg.det(U))
    assert a.vg(np.zeros(len(a.x0)))[0]==pytest.approx(b.vg(np.zeros(len(b.x0)))[0],abs=1e-13)
    np.testing.assert_allclose(a.independent(np.zeros(len(a.x0)))[1],b.independent(np.zeros(len(b.x0)))[1],atol=1e-13)


def test_false_gradient_is_not_accepted_and_RC2_backend_absent_STOP(monkeypatch):
    g,J,U,s=invented();p=m.legacy.Coefficients(g,U,s);chart=m.CoefficientChart(p,J)
    original=chart.vg;monkeypatch.setattr(chart,'vg',lambda z:(original(z)[0],original(z)[1]+.001))
    with pytest.raises(ValueError,match='NEW_POINT_CONSISTENCY'):chart.assess(np.zeros(len(chart.x0)))
    with pytest.raises(ValueError,match='ORIGINAL_METRIC_POINT_MISMATCH'):
        m.CoefficientChart(p,J+.01,original=chart.original)
    g,J,U,s=invented(role='rc2')
    with pytest.raises(ValueError,match='PR_ONLY'):m.CoefficientChart(m.legacy.Coefficients(g,U,s),J)
    monkeypatch.setattr(m.precision,'require_extended',lambda:(_ for _ in ()).throw(ValueError('EXTENDED_UNAVAILABLE')))
    g,J,U,s=invented()
    with pytest.raises(ValueError,match='EXTENDED_UNAVAILABLE'):m.CoefficientChart(m.legacy.Coefficients(g,U,s),J)


def test_new_chart_can_drive_existing_SciPy_Newton_without_new_optimizer():
    g,J,U,s=invented();p=m.legacy.Coefficients(g,U,s);chart=m.CoefficientChart(p,J);zero=np.zeros(len(chart.x0))
    res=minimize(chart.vg,zero,jac=True,hess=chart.hess,method='Newton-CG',options=dict(maxiter=1000,xtol=1e-12))
    assert chart.vg(res.x)[0]<chart.vg(zero)[0] and chart.assess(res.x)['status']=='NUMERICAL_POINT_CONSISTENCY_PASS'


def artificial_study():
    rows=[]
    for seed in (41,42):
        p,J,raw,info=failing_invented(seed);z=raw['z'];old=m.legacy.CoefficientChart(p,J);new=m.CoefficientChart(p,J,original=old)
        _,Jp=new.point(z);_,gj,res=m.precision.parts_gradient_J(p.g,Jp,p.mapping.U,p.spec)
        _,oldchart=old.vg(z);res64=m.legacy.linear.reconstruct(Jp,p.mapping.U)-p.g
        # A common actual-point residual, still binary64, isolates reassociation
        # from the subsequent gain of extended accumulation.
        c=m.legacy.pr.coefficients(Jp[p.spec.ld:],np).ravel();h=np.hypot(c,p.spec.epsilon);mass=np.sum(h)
        chart64=(p.A@old.W).T@res64.ravel()/p.spec.loss_scale+2*p.spec.gamma*mass/p.spec.mass_scale**2*((p.T@old.W).T@(c/h))
        gx64=p.mapping.adjoint(res64)/p.w/p.spec.loss_scale+2*p.spec.gamma*mass/p.spec.mass_scale**2*(p.T.T@(c/h))
        rows.append(dict(seed=seed,norb=4,rank=8,ld=2,invented_loss_scale=p.spec.loss_scale,
            legacy=info['consistency'],old_affine_vs_actual_extended_residual_max=float(np.max(abs(old.r0+old.A@z-res.ravel()))),
            extended_physical_only_gradient_difference=float(np.max(abs(old.W.T@gj-oldchart))),
            common_binary64_residual_chain_difference=float(np.max(abs(old.W.T@gx64-chart64))),new=new.assess(z)))
    accuracy=[]
    for scale in (1.,1e-9,1e-12):
        g,J,U,s=invented(scale=scale);s=replace(s,gamma=0.);g=m.legacy.linear.reconstruct(J,U)
        chart=m.CoefficientChart(m.legacy.Coefficients(g,U,s),J);z=np.zeros(len(chart.x0));ref=decimal_tensor_gradient(chart,z)
        accuracy.append(dict(loss_scale=scale,old_error=float(np.linalg.norm(chart.original.vg(z)[1]-ref)),new_error=float(np.linalg.norm(chart.vg(z)[1]-ref)),reference='Decimal80,exact supplied binary64 J/U array point,gamma0 invented tensor objective'))
    return dict(artificial_failure_reproductions=rows,independent_accuracy=accuracy,real_molecular_decodes=0,scientific_calls=0,rigorous_gradient_error_bound=None)

if __name__=='__main__':
    with Path(sys.argv[1]).open('x') as f:json.dump(artificial_study(),f,ensure_ascii=False,indent=2,allow_nan=False)

"""Analytic/Decimal/invented DF, NEVER saved molecular arrays."""
from decimal import Decimal,localcontext
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_inner_point_solver as inner
from trottertracks import c_pr_guarded_step as guarded
from test_research_c_separated_J_U import invented
from test_research_c_j_u_stability import decimal_physical_U
from test_research_c_pr_guarded_step import CoupledToy


def sensitivity_toy():
    # F=1+.5(a*j-b*u)^2/f*, a=1e-6,b=1,f*=1e-12.
    # At u=0: Hjj=1, Huj=-1e6, delta j=1e-8.
    a,b,scale,delta=1e-6,1.,1e-12,1e-8
    gu=lambda j,u:-b*(a*j-b*u)/scale
    gj=lambda j,u:a*(a*j-b*u)/scale
    h=1e-8;mixed=(gu(h,0)-gu(-h,0))/(2*h)
    with localcontext() as ctx:
        ctx.prec=80;dd=Decimal('1e-8');gap=dd*dd/2
    return dict(J_gradient=gj(delta,0),U_gradient_shift=abs(gu(delta,0)-gu(0,0)),
        exact_objective_gap=str(gap),binary64_F_equal=float(1+np.longdouble(delta)**2/2)==1.,
        mixed_Hessian_analytic=-a*b/scale,mixed_Hessian_FD=mixed,
        J_residual_PASS=gj(delta,0)<1e-5,U_stability_PASS=abs(gu(delta,0))<=1e-6)


def toy_inner_comparison():
    rows=[]
    for method,center in [('Newton-CG',False),('Newton-CG',True),('trust-exact',False),('trust-exact',True)]:
        def vg(x):
            j=np.longdouble(x[0]);v=j*j/2
            return float(v if center else 1+v),np.array([float(j)])
        opts=dict(maxiter=1000,**({'xtol':1e-12} if method=='Newton-CG' else {'gtol':1e-12}))
        r=minimize(vg,np.array([1e-8]),jac=True,hess=lambda x:np.ones((1,1)),method=method,options=opts)
        rows.append(dict(method=method,constant_centered_before_float=center,success=bool(r.success),status=int(r.status),message=str(r.message),
            J_endpoint=float(r.x[0]),independent_J_gradient=abs(float(r.x[0])),
            true_U_gradient_error=1e6*abs(float(r.x[0])),stability_PASS=1e6*abs(float(r.x[0]))<=1e-6))
    return rows


def test_small_J_and_equal_float_objective_not_small_U_error():
    r=sensitivity_toy()
    assert r['J_residual_PASS'] and not r['U_stability_PASS'] and r['binary64_F_equal']
    assert r['mixed_Hessian_FD']==pytest.approx(r['mixed_Hessian_analytic'],rel=1e-14)
    assert r['U_gradient_shift']==pytest.approx(.01)


def test_inner_improvement_and_remaining_failure_without_threshold_change():
    rows=toy_inner_comparison()
    assert rows[0]['success'] and not rows[0]['stability_PASS']
    assert not rows[1]['stability_PASS']  # centering alone is not a cure
    assert not rows[2]['stability_PASS']  # alternate solver alone is not a cure
    assert rows[3]['stability_PASS'] and rows[3]['J_endpoint']==0.


def test_real_Newton_precision_loss_does_not_equal_J_or_reduced_failure():
    g,J,U,s=invented(n=4,rank=8,ld=2,scale=2.1839632053132894e-9)
    _,i=inner.solve_j(inner.stable.Coefficients(g,U,s),J)
    assert i['status']==2 and not i['success'] and i['stationarity']['status']=='MET'
    assert i['consistency']['chart_vs_physical_gradient']['PASS']


def test_universal_gauge_and_nongauge_are_different():
    g,J,U,s=invented(n=2,rank=3,ld=2);N=np.zeros_like(J);N[0]=1.;N[1]=-1.
    base=inner.stable.precision.evaluate(g,J,U,s)
    gauge=inner.stable.precision.evaluate(g,J+N,U,s)
    np.testing.assert_allclose(base[0]['objective'],gauge[0]['objective'],atol=2e-13)
    np.testing.assert_allclose(base[2],gauge[2],atol=2e-13)
    N[1]=0.;other=inner.stable.precision.evaluate(g,J+N,U,s)
    assert abs(other[0]['objective']-base[0]['objective'])>.01


def test_same_array_residual_accuracy_against_Decimal_not_only_path_agreement():
    g,J,U,s=invented(scale=1e-12);g=inner.stable.linear.reconstruct(J,U)
    reference=decimal_physical_U(g,J,U,s)
    stable=inner.stable.precision.evaluate(g,J,U,s)[2]
    binary=inner.stable.previous.state(g,J,U,s)[2]
    assert np.linalg.norm(stable-reference)<1e-6<np.linalg.norm(binary-reference)


def study():
    comparisons=[]
    for mode in ('recover','all_unqualified','correctness'):
        for conservative in (True,False):
            o=CoupledToy(mode);p=o.evaluate(np.array([2.]));events=[]
            try:
                _,r=guarded.search(o.evaluate,p,conservative=conservative,record=lambda *x:events.append(dict(kind=x[0],**x[2])))
            except ValueError as e:r=dict(status='CORRECTNESS_FAIL',error=str(e))
            comparisons.append(dict(case=mode,policy='A_STOP' if conservative else 'B_REJECT',result=r,queries=o.queries,inner_diagnostics=o.details,events=events))
    g,J,U,s=invented(n=4,rank=8,ld=2,scale=2.1839632053132894e-9)
    _,i=inner.solve_j(inner.stable.Coefficients(g,U,s),J)
    g2,j2,u2,s2=invented(scale=1e-12);g2=inner.stable.linear.reconstruct(j2,u2)
    ref=decimal_physical_U(g2,j2,u2,s2)
    accurate=inner.stable.precision.evaluate(g2,j2,u2,s2)[2];binary=inner.stable.previous.state(g2,j2,u2,s2)[2]
    return dict(artificial_only=True,real_molecular_decodes=0,new_scientific_calls=0,
        analytic_sensitivity=sensitivity_toy(),inner_toy=toy_inner_comparison(),
        scale_amplification=[dict(loss_scale=t,J_gradient=1e-12*1e-8/t,U_gradient_shift=1e-6*1e-8/t,
            mixed_Hessian=-1e-6/t) for t in (1.,1e-9,1e-12)],
        real_invented_Newton={k:i[k] for k in ('status','success','message','nit','stationarity','consistency')},
        Decimal_reference=dict(stable_U_error_l2=float(np.linalg.norm(accurate-ref)),binary_U_error_l2=float(np.linalg.norm(binary-ref)),digits=80),
        outer_policy_comparison=comparisons,interpretation='Toy mechanisms and API behavior only;H4 cause and performance not established')


if __name__=='__main__':
    import json
    with Path(sys.argv[1]).open('x') as f:json.dump(study(),f,indent=2,allow_nan=False)

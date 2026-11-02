"""Invented arrays only: entry backend, same-array checks and reused solvers."""
from collections import Counter
from dataclasses import replace
from functools import partial
from unittest.mock import patch
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_near_stable as m
from test_research_c_separated_J_U import invented


@pytest.fixture(autouse=True)
def no_saved_molecular_decode():
    with patch('numpy.load',side_effect=AssertionError('NO_SAVED_ARRAYS')):yield


def problem(n=2,rank=3,ld=1,seed=41):
    g,J,U,_=invented(n=n,rank=rank,ld=ld,seed=seed)
    return g,J,U,m.near.initial_spec(J,ld=ld,loss_scale=1.,strength=.1)


@pytest.mark.parametrize('n,rank,ld',[(2,3,1),(4,8,2)])
def test_same_array_value_J_U_gradient_HVP_AD_and_refinement(n,rank,ld):
    g,J,U,s=problem(n,rank,ld)
    # Include det=-1 and pi charts without choosing a new logarithm branch.
    U[0]=np.eye(n);U[0,:2,:2]=-np.eye(2);U[-1,:,0]*=-1
    p=m.Coefficients(g,U,s)
    chart=m.inner.point.CoefficientChart(p,J)
    z=np.zeros(len(chart.x0));z+=np.linspace(-.002,.003,len(z))
    x,j=chart.point(z)
    value,grad=chart.vg(z);vi,gx,pull=chart.independent(z)
    assert value==vi
    np.testing.assert_allclose(grad,pull,atol=1e-12)
    row,vectors=chart.assess(z,vectors=True)
    assert row['checks']['chart_vs_physical_gradient']['PASS']
    d=np.linspace(-.1,.2,len(z));h=1e-6
    fd=(chart.vg(z+h*d)[0]-chart.vg(z-h*d)[0])/(2*h)
    assert fd==pytest.approx(grad@d,abs=2e-8)
    hv=(chart.vg(z+h*d)[1]-chart.vg(z-h*d)[1])/(2*h)
    np.testing.assert_allclose(hv,chart.hess(z)@d,atol=1e-7,rtol=1e-6)
    # Independent original B oracle/ffsim AD, no duplicated penalty oracle.
    _,_,packed,_,_=m.near.pr.corrected.codec.roundtrip(j,U)
    av,ag=m.near.make_value_and_grad(s)(packed,g)
    split=rank*n*(n-1)//2
    assert value==pytest.approx(float(av),abs=2e-12)
    np.testing.assert_allclose(gx,np.asarray(ag)[split:]/p.w,atol=3e-12)
    state,gj,gu,cg,_=m.state(g,j,U,s)
    assert state['forward_pullback_same_array']['PASS']
    # Local physical tangent vs standalone U finite differences.
    direction=np.linspace(-.2,.3,len(gu));zero=np.zeros_like(gu)
    def v(eta):
        V=m.stable.precision.geometry(U,eta)
        return m.parts_gradient_J(g,j,V,s)[0]['objective']
    assert (v(h*direction)-v(-h*direction))/(2*h)==pytest.approx(gu@direction,abs=2e-8)


def test_zero_small_coefficients_tail_mask_and_no_RC2_dispatch():
    g,J,U,s=problem();J[:]=0.;J[-1,0,0]=1e-18
    p=m.Coefficients(g,U,s);c=m.inner.point.CoefficientChart(p,J);z=np.zeros(len(c.x0))
    assert np.isfinite(c.hess(z)).all()
    assert m.near.mass(J,s,np)==pytest.approx(np.sum(np.hypot(p.penalty_coefficients(J),p.penalty_epsilon)))
    head=J.copy();head[0]+=np.eye(2)
    assert m.near.mass(head,s,np)==m.near.mass(J,s,np)
    assert not p.is_pr and p.is_entry
    with pytest.raises(ValueError,match='ENTRY_OBJECTIVE'):
        m.validate(m.near.pr.Objective(3,2,1,1.,1.,.1))
    with pytest.raises(ValueError,match='POSITIVE_STRENGTH'):
        m.validate(replace(s,strength=0.))


def test_entry_Newton_real_inner_raw_precision_loss_and_bad_jac_STOP():
    g,J,U,s=problem();p=m.Coefficients(g,U,s)
    from scipy.optimize import minimize
    records=[]
    def status2(*a,**k):
        out=minimize(*a,**k);out.success=False;out.status=2;out.message='synthetic precision loss';return out
    j,info=m.solve_j(p,J,solver=status2,record=lambda *a:records.append(a))
    assert not info['success'] and info['status']==2
    assert info['stationarity']['status']=='MET'
    assert info['hessian_calls_in_block']>0
    assert all(info['consistency'][k]['PASS'] for k in ('chart_vs_physical_gradient','chart_vs_physical_value','binary64_same_array_value'))
    assert records[0][1]['solver_jac'].dtype==np.float64
    def bad(*a,**k):
        out=minimize(*a,**k);out.jac+=1.;return out
    with pytest.raises(ValueError,match='CONSISTENCY_MISMATCH'):m.solve_j(p,J,solver=bad)


def test_reduced_three_solves_refinements_stability_cache_FD_and_admission(monkeypatch):
    g,J,U,s=problem();counts=Counter()
    def call(k,l,f):counts[k]+=1;return f()
    o=m.guarded.Oracle(g,J,U,s,backend=m.BACKEND,call=call)
    zero=np.zeros(3);d=np.array([.3,-.7,.2]);h=2e-5
    a=o.evaluate(zero);b=o.evaluate(h*d);c=o.evaluate(-h*d)
    assert (b['value']-c['value'])/(2*h)==pytest.approx(a['gradient']@d,abs=3e-7)
    assert o.evaluate(zero) is a
    assert counts==dict(inner_J_solve=9,precision_state=6,reduced_admission=3)
    assert all(a['qualification']['admission_gates'].values())
    assert a['qualification']['applicability']['condition']=='ENTRY_only_universal_head_gauge'
    assert a['qualification']['pair_diagnostics']['same_U_arrays']
    monkeypatch.setattr(m.BACKEND,'applicability',lambda *a:dict(admitted=False))
    bad=m.guarded.Oracle(g,J,U,s,backend=m.BACKEND)
    with pytest.raises(m.stable.Unqualified,match='REDUCED_GRADIENT_UNQUALIFIED'):bad.evaluate(zero)


def test_PR_default_stays_bitwise_identical_dispatch():
    g,J,U,s=invented();p=m.stable.Coefficients(g,U,s)
    c=m.inner.point.CoefficientChart(p,J)
    z=np.zeros(len(c.x0));v,gz=c.vg(z)
    # Independent frozen implementation loaded as isolated module, never patched.
    import importlib.util,subprocess
    root=Path(__file__).resolve().parents[2]
    raw=subprocess.check_output(['git','show','2459e018ea2276c359d6fe96253fbcad0bfd65bb:src/trottertracks/c_pr_coefficient_point.py'],cwd=root,text=True)
    namespace={'__name__':'trottertracks._frozen_chart','__package__':'trottertracks'}
    exec(compile(raw,'frozen_chart','exec'),namespace)
    old=namespace['CoefficientChart'](p,J)
    assert v==old.vg(z)[0]
    np.testing.assert_array_equal(gz,old.vg(z)[1]);np.testing.assert_array_equal(c.hess(z),old.hess(z))
    for a,b in zip(c.independent(z),old.independent(z)):np.testing.assert_array_equal(a,b)


def test_actual_entry_Armijo_uses_same_admissions_and_physics():
    g,J,U,s=problem();calls=Counter();physical=[]
    def call(k,l,f):calls[k]+=1;return f()
    def assess(label,j,u):physical.append(label);return dict(budget_PASS=True,artificial=True)
    r=m.trajectory(g,J,U,s,assess,step=replace(m.control.STEP,max_updates=1),
        maximum_total_points=8,call=call)
    assert r['accepted_U_updates']==1
    assert r['history'][-1]['qualification']['status']=='NUMERICAL_PILOT_ADMITTED'
    assert physical==['initial','update001']
    assert calls['inner_J_solve']==3*r['point_evaluations']
    assert calls['independent_terminal_assessment']==1
    assert r['true_reduced_gradient_qualification'] is None


def test_total_trial_budget_is_enforced_without_more_inner_solves():
    g,J,U,s=problem()
    r=m.trajectory(g,J,U,s,lambda *a:dict(budget_PASS=True),maximum_total_points=1)
    assert r['status']=='MAXIMUM_TOTAL_U_POINTS' and r['point_evaluations']==1
    assert r['accepted_U_updates']==0


def test_four_orbital_tiny_loss_scale_same_array_consistency():
    g,J,U,_=problem(n=4,rank=8,ld=2)
    s=m.near.initial_spec(J,ld=2,loss_scale=2.1839632053132894e-9,strength=.1)
    rng=np.random.default_rng(206)
    g=m.stable.linear.reconstruct(J,U)+rng.normal(size=(4,)*4)*1e-6
    p=m.Coefficients(g,U,s);c=m.inner.point.CoefficientChart(p,J)
    z=np.linspace(-.001,.002,len(c.x0))
    assert c.assess(z)['checks']['chart_vs_physical_gradient']['PASS']
    j,info=m.solve_j(p,J,replace(m.inner.DEFAULT,j_maxiter=3))
    assert info['consistency']['chart_vs_physical_gradient']['PASS']
    assert info['consistency']['binary64_same_array_value']['PASS']
    assert info['progress']['status']=='PASS_WITHIN_ENGINEERING_ALLOWANCE'


@pytest.mark.parametrize('failure',['stability','J','correctness','physics'])
def test_entry_expected_unqualified_vs_correctness_and_physics_STOP(failure,monkeypatch):
    g,J,U,s=problem()
    if failure=='stability':
        policy=m.control.DEFAULT
        real=m.BACKEND.state;count=[0]
        def shifted(*a,**k):
            row,gj,gu,cg,b=real(*a,**k);count[0]+=1
            return row,gj,gu+count[0]*1e-4,cg+count[0]*1e-4,b
        monkeypatch.setattr(m.BACKEND,'state',shifted)
    elif failure=='J':policy=replace(m.control.DEFAULT,j_gradient_infinity=1e-20)
    else:policy=m.control.DEFAULT
    if failure=='correctness':
        monkeypatch.setattr(m.BACKEND,'state',lambda *a,**k:(_ for _ in ()).throw(ValueError('ENTRY_CORRECTNESS')))
        with pytest.raises(ValueError,match='ENTRY_CORRECTNESS'):
            m.trajectory(g,J,U,s,lambda *a:dict(budget_PASS=True),maximum_total_points=2)
    else:
        r=m.trajectory(g,J,U,s,lambda *a:dict(budget_PASS=failure!='physics'),policy=policy,maximum_total_points=2)
        assert r['accepted_U_updates']==0
        if failure=='physics':assert r['status']=='INITIAL_DF_BUDGET_NOT_MET'
        else:assert r['status'] in ('J_UNQUALIFIED','REDUCED_GRADIENT_UNQUALIFIED','BACKTRACKING_NUMERICAL_QUALIFICATION_EXHAUSTED')

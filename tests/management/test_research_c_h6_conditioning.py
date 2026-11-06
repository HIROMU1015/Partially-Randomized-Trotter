"""Invented ill-conditioned maps/arrays only; no saved molecular inputs."""
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
import pytest
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_h6_fit_conditioning as fix
from trottertracks.c1_h6_initialization import reconstruct
from trottertracks.c_near_transfer import System,objective_specs


def invented(n=6,rank=14,ld=4,angle=1e-8):
    rng=np.random.default_rng(1229)
    u=np.stack([np.linalg.qr(a)[0] for a in rng.normal(size=(rank,n,n))])
    for frame in range(1,ld):
        a=rng.normal(size=(n,n));u[frame]=u[0]@expm(angle*(a-a.T))
    j=rng.normal(size=(rank,n,n))*.05;j=(j+j.transpose(0,2,1))/2
    target=j.copy();target[ld:]=0.
    g=reconstruct(target,u)
    specs=objective_specs(g,j,u,System(n,rank,ld,1.,'a'*64),strength=.1,epsilon=1e-8)
    return g,j,u,specs


@pytest.mark.parametrize('arm',['PR','ENTRY'])
def test_verified_SVD_separates_weak_nongauge_from_true_extra_null(arm):
    g,j,u,specs=invented();b=fix.backend(specs[arm]);p=b.Coefficients(g,u,specs[arm])
    report=fix.head_report(p)
    assert not report['original_float64_report']['admitted']
    assert report['admitted'] and report['rank_gap_factor']==1024
    assert report['SVD_reconstruction_Frobenius']<1e-10
    u[:4]=u[0];p=b.Coefficients(g,u,specs[arm]);assert not fix.head_report(p)['admitted']


@pytest.mark.parametrize('arm',['PR','ENTRY'])
def test_full_physical_objective_and_gradient_preserved_in_new_coordinates(arm):
    g,j,u,specs=invented(n=2,rank=3,ld=2,angle=.01)
    p=fix.backend(specs[arm]).Coefficients(g,u,specs[arm]);chart=fix.Chart(p,j,fix.inner.DEFAULT)
    assert chart.dimension==len(p.w)-1
    assert p._h6_metric.gauge_map_residual<1e-10
    metric=p._h6_metric
    assert metric.scaling_condition_limit==1/np.sqrt(np.finfo(float).eps)
    assert np.all(np.isfinite(metric.W))
    z=np.zeros(chart.dimension);z[0]=.001
    value,gradient=chart.vg(z);centered,_=chart.centered(z)
    assert centered==pytest.approx(value-float(chart.origin_primal[-1]),abs=1e-12)
    np.testing.assert_allclose(gradient,chart.independent(z)[2],atol=1e-8,rtol=1e-7)
    # Search Hessian BLAS assembly is checked against the original LD formula.
    reference=fix.inner.point.CoefficientChart.hess(chart,z)
    np.testing.assert_allclose(chart.hess(z),reference,atol=1e-10,rtol=1e-10)
    d=np.random.default_rng(51).normal(size=chart.dimension);d/=np.linalg.norm(d)
    h=1e-6
    fd=(chart.centered(z+h*d)[0]-chart.centered(z-h*d)[0])/(2*h)
    assert fd==pytest.approx(gradient@d,abs=2e-7,rel=1e-5)


@pytest.mark.parametrize('arm',['PR','ENTRY'])
def test_invented_solver_retains_original_J_and_U_stability_gates(arm):
    g,j,u,specs=invented(n=2,rank=3,ld=2,angle=.01)
    b=fix.backend(specs[arm]);p=b.Coefficients(g,u,specs[arm]);last=None
    for _ in range(3):
        j,info=fix.solve_j(p,j)
        assert info['stationarity']['threshold']==1e-5
        assert info['stationarity']['status']=='MET'
        assert info['nongauge_directions_dropped']==0 and info['known_gauge_coordinates_fixed']==1
        assert info['objective_unchanged'] and not info['thresholds_changed']
        state=b.state(g,j,u,specs[arm]);now=state[2]
        if last is not None:assert np.linalg.norm(now-last)<1e-6
        last=now


def test_extra_null_rejected_before_solver_without_correctness_FAIL():
    g,j,u,specs=invented();u[:4]=u[0];events=[];counts=[]
    def call(k,l,f):counts.append(k);return f()
    from functools import partial
    from trottertracks.c_pr_guarded_step import Oracle
    o=Oracle(g,j,u,specs['PR'],fix.inner.DEFAULT,backend=fix.backend(specs['PR']),
        call=call,record=lambda *x:events.append(x))
    with pytest.raises(fix.inner.stable.Unqualified,match='REDUCED_GRADIENT_UNQUALIFIED'):
        o.evaluate(np.zeros(210))
    assert counts==['inner_J_solve'] and len(o.points)==1
    assert events[0][0]=='early_structural_rejection'
    result=fix.trajectory(g,j,u,specs['PR'],lambda *a:pytest.fail('physical call after structural rejection'))
    assert result['inner_solves']==result['refinement_solves']==0
    assert result['coefficient_setup_and_solve_macro_calls']==1


@pytest.mark.parametrize('arm',['PR','ENTRY'])
def test_six_orbital_rank14_three_solves_actual_oracle_and_unchanged_gates(arm):
    g,j,u,specs=invented(angle=.03)
    from trottertracks.c_pr_guarded_step import Oracle
    o=Oracle(g,j,u,specs[arm],fix.inner.DEFAULT,backend=fix.backend(specs[arm]))
    point=o.evaluate(np.zeros(210))
    assert point['qualification']['status']=='NUMERICAL_PILOT_ADMITTED'
    assert all(point['qualification']['admission_gates'].values())
    assert point['qualification']['latest_refinement_U_shift_l2']<=1e-6
    assert len(point['inner'])==3
    assert all(v['known_gauge_coordinates_fixed']==3 for v in point['inner'])
    assert point['state']['J_stationarity']['threshold']==1e-5


def test_corrupt_returned_solver_gradient_is_not_accepted(monkeypatch):
    import scipy.optimize
    original=scipy.optimize.minimize
    def corrupt(*args,**kwargs):
        value=original(*args,**kwargs);value.jac=value.jac+1.;return value
    monkeypatch.setattr(scipy.optimize,'minimize',corrupt)
    g,j,u,specs=invented(n=2,rank=3,ld=2,angle=.01)
    p=fix.backend(specs['PR']).Coefficients(g,u,specs['PR'])
    with pytest.raises(ValueError,match='CONDITIONED_INNER_POINT_CONSISTENCY'):
        fix.solve_j(p,j)


def test_bad_SVD_vectors_cannot_reduce_uncertainty_guard(monkeypatch):
    import scipy.linalg
    original=scipy.linalg.svd
    def corrupt(*args,**kwargs):
        result=original(*args,**kwargs)
        if isinstance(result,tuple):
            u,s,v=result;u=u.copy();u[:,0]+=.001;return u,s,v
        return result
    monkeypatch.setattr(scipy.linalg,'svd',corrupt)
    g,j,u,specs=invented()
    p=fix.backend(specs['PR']).Coefficients(g,u,specs['PR'])
    assert not fix.head_report(p)['admitted']


@pytest.mark.parametrize('arm',['PR','ENTRY'])
def test_near_gauge_cancellation_remains_consistent_without_direction_truncation(arm):
    g,j,u,specs=invented()
    # Large cancelling coefficients at nearly identical head frames mimic the
    # precision risk of the failed run, without reading a molecular array.
    rng=np.random.default_rng(87);delta=rng.normal(size=j[0].shape)*1e4
    delta=(delta+delta.T)/2;j[0]+=delta;j[1]-=delta
    p=fix.backend(specs[arm]).Coefficients(g,u,specs[arm]);chart=fix.Chart(p,j,fix.inner.DEFAULT)
    z=np.zeros(chart.dimension)
    direct=chart.vg(z)[1];independent=chart.independent(z)[2]
    check=fix.inner.point.compare(direct,independent,atol=1e-8,rtol=1e-7)
    assert check['PASS']
    assert p._h6_metric.limited_directions>0
    assert chart.W.shape[1]==len(p.w)-(specs[arm].ld-1)
    assert np.linalg.matrix_rank(np.asarray(chart.W,float))==chart.dimension


@pytest.mark.parametrize('arm',['PR','ENTRY'])
def test_centered_cost_is_exactly_zero_at_the_actual_decoded_origin(arm):
    g,j,u,specs=invented(n=2,rank=3,ld=2,angle=.01)
    # Choose a coefficient whose actual binary64 pack/unpack moves one ulp.
    bits=np.float64(1e5).view(np.uint64)
    candidates=(bits+np.arange(256,dtype=np.uint64)).view(np.float64)
    restored=(candidates*np.sqrt(2.))/np.sqrt(2.)
    value=candidates[np.flatnonzero(restored!=candidates)[0]]
    j[0,0,1]=j[0,1,0]=value
    p=fix.backend(specs[arm]).Coefficients(g,u,specs[arm]);chart=fix.Chart(p,j,fix.inner.DEFAULT)
    zero=np.zeros(chart.dimension)
    assert chart.point(zero)[1][0,0,1]!=j[0,0,1]
    assert chart.centered(zero)[0]==0.
    np.testing.assert_array_equal(chart.origin_J,chart.point(zero)[1])

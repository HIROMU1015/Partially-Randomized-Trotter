"""Artificial inner/outer wiring; never read saved molecular arrays."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import json,sys
import numpy as np
import pytest
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_inner_point_solver as m
from test_research_c_separated_J_U import invented


def test_real_Newton_pre_step_raw_jac_semantics_and_independent_endpoint():
    s=m.stable.pr.Objective(2,1,1,1.,1.,0.)
    p=m.stable.Coefficients(np.array([[[[.9]]]]),np.ones((2,1,1)),s)
    records=[];J,info=m.solve_j(p,np.array([.1,.2])[:,None,None],replace(m.DEFAULT,j_maxiter=1),record=lambda a,b:records.append((a,b)))
    assert not info['success'] and info['status']==1
    assert info['consistency']['last_step_start']['PASS']
    assert not info['consistency']['raw_jac_at_endpoint']['PASS']
    assert info['stationarity']['status']=='MET'
    assert records[0][1]['solver_jac'].dtype==np.float64
    assert info['hessian_calls_in_block']>0 and info['independent_adjoint_calls_in_block']==2


def test_every_VG_Hessian_call_uses_opt_in_chart_and_no_legacy_fallback(monkeypatch):
    g,J,U,s=invented();p=m.stable.Coefficients(g,U,s);counts=Counter();real=m.point.CoefficientChart
    class Spy(real):
        def vg(self,z):counts['vg']+=1;return super().vg(z)
        def hess(self,z):counts['hess']+=1;return super().hess(z)
        def independent(self,z):counts['independent']+=1;return super().independent(z)
    monkeypatch.setattr(m.point,'CoefficientChart',Spy)
    monkeypatch.setattr(m.point.legacy,'solve_j',lambda *a,**k:pytest.fail('legacy optimizer dispatch'))
    out,info=m.solve_j(p,J)
    assert counts['vg']==info['value_gradient_calls_in_block'] and counts['hess']==info['hessian_calls_in_block']>0
    assert counts['independent']==2 and info['method']==m.VERSION
    assert info['progress']['status']=='PASS_WITHIN_ENGINEERING_ALLOWANCE'
    np.testing.assert_array_equal(out,out.transpose(0,2,1))
    assert info['metric_directions_dropped']==0 and not info['thresholds_changed']


@pytest.mark.parametrize('seed',[41,42])
def test_artificial_four_orbital_inner_entire_solver_consistency(seed):
    g,J,U,s=invented(seed=seed,n=4,rank=8,ld=2,scale=2.1839632053132894e-9)
    p=m.stable.Coefficients(g,U,s);out,info=m.solve_j(p,J)
    assert info['consistency']['chart_vs_physical_gradient']['PASS']
    assert info['consistency']['chart_vs_physical_value']['PASS']
    assert info['progress']['decrease']>=-info['progress']['allowance']


def test_forged_raw_gradient_preserved_then_explicit_STOP():
    g,J,U,s=invented();p=m.stable.Coefficients(g,U,s);records=[]
    def bad(*args,**kwargs):
        r=minimize(*args,**kwargs);r.jac=r.jac+1.;return r
    with pytest.raises(ValueError,match='PR_INNER_POINT_CONSISTENCY'):
        m.solve_j(p,J,solver=bad,record=lambda a,b:records.append((a,b)))
    assert records[0][1] is not None and records[-1][0]['consistency_record']
    assert not records[-1][0]['consistency']['raw_jac_at_endpoint']['PASS']


def test_nonfinite_endpoint_and_RC2_are_not_fallbacks():
    g,J,U,s=invented();p=m.stable.Coefficients(g,U,s)
    def bad(*args,**kwargs):
        r=minimize(*args,**kwargs);r.x[:]=np.nan;return r
    with pytest.raises(ValueError,match='NONFINITE_OR_MALFORMED'):m.solve_j(p,J,solver=bad)
    g,J,U,s=invented(role='rc2')
    with pytest.raises(ValueError,match='PR_ONLY'):m.solve_j(m.stable.Coefficients(g,U,s),J)
    with pytest.raises(ValueError,match='PR_ONLY'):m.PRReducedOracle(g,J,U,s)


def test_reduced_FD_cache_three_new_inner_calls_and_no_old_module_patch():
    g,J,U,s=invented();counts=Counter();records=[]
    old_solver=m.stable.blocks.solve_j;old_eval=m.stable.StableReducedOracle.evaluate
    def call(k,l,f):counts[k]+=1;return f()
    oracle=m.PRReducedOracle(g,J,U,s,call=call,record=lambda *r:records.append(r))
    zero=np.zeros(3);d=np.array([.3,-.7,.2]);h=2e-5
    a=oracle.evaluate(zero);b=oracle.evaluate(zero+h*d);c=oracle.evaluate(zero-h*d)
    assert (b['value']-c['value'])/(2*h)==pytest.approx(a['gradient']@d,abs=3e-7)
    assert counts==dict(inner_J_solve=9,precision_state=6,reduced_admission=3)
    assert oracle.evaluate(zero) is a
    assert all(x['method']==m.VERSION for point in oracle.points for x in point['inner'])
    assert m.stable.blocks.solve_j is old_solver and m.stable.StableReducedOracle.evaluate is old_eval


def test_reduced_SciPy_outer_interface_uses_new_inner_without_new_outer_implementation():
    g,J,U,s=invented();o=m.PRReducedOracle(g,J,U,s)
    r=minimize(o.fun,np.zeros(3),jac=True,method='BFGS',options=dict(maxiter=1,gtol=1e-7))
    assert len(o.points)>1 and r.nit==1
    assert all(i['method']==m.VERSION for p in o.points for i in p['inner'])


@pytest.mark.parametrize('mode',['limit','stationarity','structure'])
def test_reduced_unqualified_STOP_never_restarts(mode,monkeypatch):
    g,J,U,s=invented();policy=m.DEFAULT
    if mode=='stationarity':policy=replace(policy,j_gradient_infinity=1e-20)
    if mode=='limit':policy=replace(policy,max_points=1)
    if mode=='structure':
        monkeypatch.setattr(m.stable.previous,'applicability',lambda *a:dict(admitted=False,condition='artificial_extra_null'))
    oracle=m.PRReducedOracle(g,J,U,s,policy)
    with pytest.raises(m.stable.Unqualified):
        oracle.evaluate(np.zeros(3))
        oracle.evaluate(np.ones(3)*.001)


def artificial_study():
    rows=[]
    for seed,n,rank,ld,scale in [(41,2,3,1,1.),(41,4,8,2,2.1839632053132894e-9),(42,4,8,2,2.1839632053132894e-9)]:
        g,J,U,s=invented(seed=seed,n=n,rank=rank,ld=ld,scale=scale)
        p=m.stable.Coefficients(g,U,s);_,info=m.solve_j(p,J)
        rows.append(dict(seed=seed,norb=n,rank=rank,ld=ld,loss_scale=scale,
            solver={k:info[k] for k in ('method','success','status','message','nit','nfev','njev','nhev','initial_objective','independent_objective','stationarity','consistency','value_gradient_calls_in_block','hessian_calls_in_block','wall_seconds','CPU_seconds')}))
    return dict(artificial_only=True,rows=rows,real_molecular_decodes=0,scientific_calls=0)

if __name__=='__main__':
    with Path(sys.argv[1]).open('x') as f:json.dump(artificial_study(),f,ensure_ascii=False,indent=2,allow_nan=False)

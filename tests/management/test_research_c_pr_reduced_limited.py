"""Invented tensors only: opt-in three-solve admission and limited BFGS."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_reduced_limited as m
from test_research_c_separated_J_U import invented


def test_all_three_same_array_solves_cache_and_FD_without_old_patch():
    g,J,U,s=invented();counts=Counter();records=[]
    old=m.stable.StableReducedOracle.evaluate
    o=m.Oracle(g,J,U,s,call=lambda k,l,f:(counts.update([k]),f())[1],record=lambda *a:records.append(a))
    h=2e-5;d=np.array([.3,-.7,.2]);a=o.evaluate(np.zeros(3))
    b,c=o.evaluate(h*d),o.evaluate(-h*d)
    assert (b['value']-c['value'])/(2*h)==pytest.approx(a['gradient']@d,abs=3e-7)
    assert o.evaluate(np.zeros(3)) is a
    assert counts==dict(inner_J_solve=9,precision_state=6,reduced_admission=3)
    assert all(i['method']==m.inner.VERSION for p in o.points for i in p['inner'])
    assert all(a['qualification']['admission_gates'].values())
    assert a['qualification']['rigorous_gradient_error_bound'] is None
    assert m.stable.StableReducedOracle.evaluate is old


def test_precision_loss_preserved_but_not_sufficient_for_admission(monkeypatch):
    g,J,U,s=invented();real=m.inner.solve_j
    def precision(*a,**k):
        from scipy.optimize import minimize
        def stop(*args,**kw):
            r=minimize(*args,**kw);r.success=False;r.status=2;r.message='artificial precision loss';return r
        return real(*a,**k,solver=stop)
    monkeypatch.setattr(m.inner,'solve_j',precision)
    p=m.Oracle(g,J,U,s).evaluate(np.zeros(3))
    q=p['qualification']
    assert all(not x['success'] and x['status']==2 for x in q['raw_solver_flags'])
    assert q['raw_success_is_not_an_admission_gate'] and all(q['admission_gates'].values())
    monkeypatch.setattr(m.stable.previous,'applicability',lambda *a:dict(admitted=False,condition='invented_extra_null'))
    o=m.Oracle(g,J,U,s)
    with pytest.raises(m.stable.Unqualified,match='REDUCED_GRADIENT_UNQUALIFIED'):o.evaluate(np.zeros(3))
    assert o.points[0]['qualification']['admission_gates']['final_inner_J']
    assert not o.points[0]['qualification']['admission_gates']['rank_nullspace']


@pytest.mark.parametrize('fault',['inner_J','state_J','stability','structure'])
def test_separate_qualification_STOP_and_no_outer_or_retry(fault,monkeypatch):
    g,J,U,s=invented();solve=m.inner.solve_j;state=m.stable.state;count=Counter()
    if fault=='inner_J':
        def bad(*a,**k):
            j,i=solve(*a,**k);i['stationarity']=dict(i['stationarity'],status='NOT_MET');return j,i
        monkeypatch.setattr(m.inner,'solve_j',bad)
    if fault in ('state_J','stability'):
        def badstate(*a,**k):
            row,gj,gu,chart,binary=state(*a,**k);count['state']+=1
            if fault=='state_J':row['J_stationarity']=dict(row['J_stationarity'],status='NOT_MET')
            elif count['state']==2:gu=gu+.001
            return row,gj,gu,chart,binary
        monkeypatch.setattr(m.stable,'state',badstate)
    if fault=='structure':monkeypatch.setattr(m.stable.previous,'applicability',lambda *a:dict(admitted=False))
    def forbidden(*a,**k):pytest.fail('BFGS after unqualified initial point')
    _,_,r=m.optimize(g,J,U,s,solver=forbidden)
    assert r['status']==('J_UNQUALIFIED' if fault in ('inner_J','state_J') else 'REDUCED_GRADIENT_UNQUALIFIED')
    assert r['inner_solves']==3 and r['refinement_solves']==2 and r['completed_U_updates']==0
    assert r['terminal_kind']=='unqualified_initial_diagnostic'
    assert r['outer_solver_invocations']==0
    assert r['DF_budget'] is r['PR_performance'] is None
    assert r['retry']==r['resume']==0


def test_real_artificial_BFGS_accepts_at_most_two_updates():
    g,J,U,s=invented();counts=Counter()
    j,u,r=m.optimize(g,J,U,s,call=lambda k,l,f:(counts.update([k]),f())[1])
    assert 1<=r['completed_U_updates']<=2
    assert r['advancement']=='ACCEPTED_U_ADVANCE_OBSERVED'
    assert r['outer_solver_invocations']==1
    assert r['distinct_U_points']<=8
    assert counts['inner_J_solve']==3*r['distinct_U_points']
    assert counts['precision_state']==2*r['distinct_U_points']
    assert counts['reduced_admission']==r['distinct_U_points']
    np.testing.assert_allclose(u.transpose(0,2,1)@u,np.broadcast_to(np.eye(2),u.shape),atol=1e-13)


def test_new_point_budget_does_not_turn_trial_into_accepted_endpoint():
    g,J,U,s=invented();records=[]
    def requests(fun,x,**kw):
        for i in range(1,5):fun(x+i*np.array([.001,.002,-.001]))
        pytest.fail('point cap must stop')
    _,_,r=m.optimize(g,J,U,s,replace(m.DEFAULT,max_points=3),solver=requests,record=lambda *a:records.append(a))
    assert r['status']=='OUTER_LIMIT_UNQUALIFIED' and r['distinct_U_points']==3
    assert r['accepted_labels']==['0'] and r['terminal_label']=='0' and r['stop_trial_label']=='2'
    assert r['outer_raw'] is None and r['completed_U_updates']==0


def test_callback_update_cap_is_enforced_even_for_injected_longer_driver():
    from scipy.optimize import minimize
    g,J,U,s=invented()
    def longer(*args,**kwargs):
        kwargs['options']=dict(kwargs['options'],maxiter=3)
        return minimize(*args,**kwargs)
    _,_,r=m.optimize(g,J,U,s,replace(m.DEFAULT,outer_maxiter=1),solver=longer)
    assert r['status']=='LIMITED_UPDATE_TARGET_REACHED' and r['completed_U_updates']==1
    assert r['terminal_label']!=r['stop_trial_label']


def test_inner_consistency_FAIL_keeps_raw_then_stops(monkeypatch):
    g,J,U,s=invented();real=m.inner.solve_j;records=[]
    def bad(*a,**k):
        from scipy.optimize import minimize
        def forged(*args,**kw):
            r=minimize(*args,**kw);r.jac=r.jac+1.;return r
        return real(*a,**k,solver=forged)
    monkeypatch.setattr(m.inner,'solve_j',bad)
    with pytest.raises(ValueError,match='PR_INNER_POINT_CONSISTENCY'):
        m.optimize(g,J,U,s,record=lambda *a:records.append(a))
    assert any(x[0]=='inner_raw' and x[3] is not None for x in records)
    assert not any(x[0]=='accepted' for x in records)


@pytest.mark.parametrize('fault',['jac','fun'])
def test_outer_false_raw_saved_before_FAIL(fault):
    g,J,U,s=invented();records=[]
    def forged(fun,x,**kw):
        value,grad=fun(x)
        return SimpleNamespace(x=x,jac=grad+(1. if fault=='jac' else 0.),fun=value+(1. if fault=='fun' else 0.),
            success=True,status=0,message='invented bad endpoint',nit=0,nfev=1,njev=1)
    with pytest.raises(ValueError,match='OUTER_RAW_ENDPOINT'):
        m.optimize(g,J,U,s,solver=forged,record=lambda *a:records.append(a))
    assert any(x[0]=='outer_solver_result' and x[3] is not None for x in records)


def test_RC2_rejected_and_no_physical_oracle_fallback():
    g,J,U,s=invented(role='rc2')
    with pytest.raises(ValueError,match='PR_ONLY'):m.optimize(g,J,U,s)


def test_correctness_failure_after_accepted_update_does_not_export_fallback(monkeypatch):
    g,J,U,s=invented();real=m.Oracle.evaluate;records=[];accepted=[]
    def evaluate(self,eta):
        if accepted and eta.tobytes() not in self.cache:raise ValueError('ARTIFICIAL_LATER_TRIAL_FAIL')
        return real(self,eta)
    def record(*args):
        records.append(args)
        if args[0]=='accepted':accepted.append(args)
    def requests(fun,x,callback,**kw):
        y=x+.001*np.array([.3,-.7,.2]);v,h=fun(y)
        # Choose the descending sign on invented input; no new production rule.
        if v>fun(x)[0]:y=x-.001*np.array([.3,-.7,.2]);v,h=fun(y)
        callback(SimpleNamespace(x=y,fun=v))
        fun(y+.001)
        pytest.fail('correctness failure must escape')
    monkeypatch.setattr(m.Oracle,'evaluate',evaluate)
    with pytest.raises(ValueError,match='ARTIFICIAL_LATER_TRIAL_FAIL'):
        m.optimize(g,J,U,s,solver=requests,record=record)
    assert len(accepted)==1 and not any(x[0]=='limited_terminal' for x in records)


@pytest.mark.parametrize('seed',[41,42])
def test_artificial_four_orbital_rank8_full_refinement_route(seed):
    g,J,U,s=invented(seed=seed,n=4,rank=8,ld=2,scale=2.1839632053132894e-9)
    o=m.Oracle(g,J,U,s)
    try:o.evaluate(np.zeros(48))
    except m.stable.Unqualified:pass
    assert len(o.points)==1 and len(o.points[0]['inner'])==3
    q=o.points[0]['qualification']
    assert q['refinement_U_allowance']==1e-6 and q['rigorous_gradient_error_bound'] is None
    assert all(i['consistency']['chart_vs_physical_gradient']['PASS'] for i in o.points[0]['inner'])
    assert (q['status']=='NUMERICAL_PILOT_ADMITTED')==all(q['admission_gates'].values())

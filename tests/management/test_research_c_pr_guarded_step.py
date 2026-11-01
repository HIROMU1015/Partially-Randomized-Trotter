"""Artificial exact/inexact reduced toy and real invented DF oracle."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_guarded_step as m
from test_research_c_separated_J_U import invented


class CoupledToy:
    """F=.5(j-u)^2+.5u^2, j*=u. Controlled refinement accuracy.

    Large first trial: both J residuals pass1e-5, but U shift8e-6>1e-6.
    Smaller trials can pass. No fake objective or gradient is returned.
    """
    def __init__(self,mode='recover'):self.mode=mode;self.queries=[];self.details=[]
    def evaluate(self,x):
        u=float(x[0]);self.queries.append(u)
        if len(self.queries)>1 and self.mode=='correctness':raise ValueError('ARTIFICIAL_CORRECTNESS_FAIL')
        delta=8e-6 if len(self.queries)>1 and (self.mode=='all_unqualified' or u<1.2) else 1e-8
        self.details.append(dict(u=u,refine1_J_residual=delta,refine2_J_residual=0.,U_shift=delta,
            F_refine1=.5*u*u+.5*delta*delta,F_refine2=.5*u*u))
        if delta>1e-6:raise m.stable.Unqualified('REDUCED_GRADIENT_UNQUALIFIED')
        return dict(value=.5*u*u,gradient=np.array([u]),eta=x.copy(),
                    qualification=dict(status='NUMERICAL_PILOT_ADMITTED'))


@pytest.mark.parametrize('conservative',[True,False])
def test_policy_A_vs_B_qualification_not_fake_values(conservative):
    toy=CoupledToy();initial=toy.evaluate(np.array([2.]));events=[]
    p,r=m.search(toy.evaluate,initial,conservative=conservative,record=lambda *x:events.append(x))
    if conservative:
        assert r['status']=='REDUCED_GRADIENT_UNQUALIFIED' and r['accepted_U_updates']==0
        assert len(toy.queries)==2
    else:
        assert r['status']=='QUALIFIED_ARMIJO_STEP_ACCEPTED' and r['accepted_U_updates']==1
        assert len(toy.queries)==3 and p['value']<initial['value']
        assert p['value']<=r['Armijo_bound'] and r['strong_Wolfe_claim'] is False
    failed=[e[2] for e in events if e[0]=='guarded_trial' and e[2]['status']=='REDUCED_GRADIENT_UNQUALIFIED']
    assert failed and failed[0]['objective'] is None and not failed[0]['gradient_used']


def test_all_unqualified_has_exact_registered_trial_limit_and_no_accept():
    toy=CoupledToy('all_unqualified');p0=toy.evaluate(np.array([2.]));p,r=m.search(toy.evaluate,p0)
    assert p is p0 and r['status']=='GUARDED_TRIAL_LIMIT_NO_ACCEPT'
    assert r['trials']==8 and len(toy.queries)==9 and r['accepted_U_updates']==0


def test_correctness_exception_does_not_backtrack():
    toy=CoupledToy('correctness');p0=toy.evaluate(np.array([2.]))
    with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS_FAIL'):m.search(toy.evaluate,p0)
    assert len(toy.queries)==2


def test_actual_SciPy_BFGS_propagates_missing_qualified_trial():
    from scipy.optimize import minimize
    toy=CoupledToy()
    def fg(x):
        p=toy.evaluate(x);return p['value'],p['gradient']
    with pytest.raises(m.stable.Unqualified,match='REDUCED_GRADIENT_UNQUALIFIED'):
        minimize(fg,np.array([2.]),jac=True,method='BFGS',options=dict(maxiter=2,gtol=1e-7))
    assert len(toy.queries)==2


def test_qualified_higher_objective_is_rejected_without_fiction():
    p0=dict(value=1.,eta=np.array([1.]),gradient=np.array([1.]),qualification={'status':'NUMERICAL_PILOT_ADMITTED'})
    events=[]
    def actual(x):return dict(p0,eta=x,value=2.)
    p,r=m.search(actual,p0,replace(m.SEARCH,max_trials=2),record=lambda *x:events.append(x))
    assert p is p0 and r['trials']==2 and r['status']=='GUARDED_TRIAL_LIMIT_NO_ACCEPT'
    assert all(e[2]['objective']==2. for e in events if e[0]=='guarded_trial')


@pytest.mark.parametrize('fault',['qualification','nonfinite','identity'])
def test_invalid_oracle_contract_is_correctness_FAIL(fault):
    p0=dict(value=1.,eta=np.array([1.]),gradient=np.array([1.]),qualification={'status':'NUMERICAL_PILOT_ADMITTED'})
    def bad(x):
        p=dict(p0,eta=x)
        if fault=='qualification':p['qualification']={'status':'REDUCED_GRADIENT_UNQUALIFIED'}
        if fault=='nonfinite':p['gradient']=np.array([np.nan])
        if fault=='identity':p['eta']=x+1.
        return p
    with pytest.raises(ValueError):m.search(bad,p0)


def test_initial_unqualified_and_trial_budget_are_distinct(monkeypatch):
    g,J,U,s=invented()
    monkeypatch.setattr(m.stable.previous,'applicability',lambda *a:dict(admitted=False))
    _,_,r=m.optimize(g,J,U,s)
    assert r['status']=='REDUCED_GRADIENT_UNQUALIFIED' and r['trials']==0
    assert r['inner_solves']==3 and r['terminal_kind']=='unqualified_initial_diagnostic'


def test_real_invented_DF_step_preserves_full_admission_and_counts():
    g,J,U,s=invented();counts={}
    def call(k,l,f):counts[k]=counts.get(k,0)+1;return f()
    j,u,r=m.optimize(g,J,U,s,call=call)
    assert r['status']=='QUALIFIED_ARMIJO_STEP_ACCEPTED' and r['accepted_U_updates']==1
    assert r['outer_BFGS_invocations']==0 and r['distinct_U_points']<=9
    assert counts==dict(inner_J_solve=3*r['distinct_U_points'],precision_state=2*r['distinct_U_points'],reduced_admission=r['distinct_U_points'])
    assert all(r['terminal_qualification']['admission_gates'].values())
    q=r['terminal_qualification'];d=q['pair_diagnostics']
    assert max(d['physical_shift_l2'],d['chart_shift_l2'])==pytest.approx(q['latest_refinement_U_shift_l2'])
    assert d['delta_J_frobenius']>=0 and d['mixed_Hessian'] is None
    np.testing.assert_allclose(u.transpose(0,2,1)@u,np.broadcast_to(np.eye(2),u.shape),atol=2e-14)


def test_RC2_and_zero_direction_not_silent_fallbacks():
    g,J,U,s=invented(role='rc2')
    with pytest.raises(ValueError,match='PR_ONLY'):m.optimize(g,J,U,s)
    p=dict(value=0.,eta=np.zeros(1),gradient=np.zeros(1),qualification={'status':'NUMERICAL_PILOT_ADMITTED'})
    _,r=m.search(lambda x:pytest.fail('zero direction query'),p)
    assert r['status']=='NO_QUALIFIED_DESCENT_DIRECTION'

"""Artificial small-step acceptance and failure cases; no saved molecules."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_small_step as m
from test_research_c_separated_J_U import invented


def point(x, value=None):
    value=.5*float(x[0])**2 if value is None else value
    return dict(eta=x.copy(),gradient=x.copy(),value=value,
        state=dict(value_same_array={'maxabs':0.},binary64_AD_value_same_array={'maxabs':0.}),
        qualification=dict(status='NUMERICAL_PILOT_ADMITTED',objective_refinement=dict(before=value,after=value)))


def test_one_registered_small_query_remains_provisional_before_independent_assessment():
    p0=point(np.array([2.]));queries=[]
    def evaluate(eta):queries.append(eta.copy());return point(eta)
    p,r=m.propose(evaluate,p0)
    assert len(queries)==1 and r['alpha']==pytest.approx(.0625)
    assert r['status']=='PROVISIONAL_SMALL_ARMIJO_PASS' and r['accepted_U_updates']==0
    assert p['value']+r['comparison_scale']<=r['Armijo_bound']
    assert r['certified_descent'] is None and not r['strong_Wolfe_claim']


@pytest.mark.parametrize('mode',['unqualified','no_decrease','below_scale','correctness'])
def test_no_fallback_or_second_trial(mode):
    p0=point(np.array([2.]));queries=[]
    def evaluate(eta):
        queries.append(eta)
        if mode=='unqualified':raise m.stable.Unqualified('REDUCED_GRADIENT_UNQUALIFIED')
        if mode=='correctness':raise ValueError('ARTIFICIAL_CORRECTNESS_FAIL')
        return point(eta,value=p0['value']+(1. if mode=='no_decrease' else -1e-12))
    if mode=='correctness':
        with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS_FAIL'):m.propose(evaluate,p0)
    else:
        p,r=m.propose(evaluate,p0)
        assert p is p0 and r['accepted_U_updates']==0
        if mode=='unqualified':assert r['objective'] is None and not r['gradient_used']
        else:assert r['status']=='SMALL_STEP_REJECTED_ARMIJO_OR_VALUE_SCALE'
    assert len(queries)==1


@pytest.mark.parametrize('mode',['qualification','identity','nonfinite'])
def test_trial_correctness_mismatch_escapes(mode):
    p0=point(np.array([2.]))
    def bad(eta):
        p=point(eta)
        if mode=='qualification':p['qualification']['status']='J_UNQUALIFIED'
        if mode=='identity':p['eta']=eta+1
        if mode=='nonfinite':p['gradient'][:]=np.nan
        return p
    with pytest.raises(ValueError):m.propose(bad,p0)


@pytest.mark.parametrize('mode',['zero','nonpositive','nonfinite'])
def test_no_direction_or_scale_does_not_query(mode):
    p0=point(np.array([0. if mode=='zero' else 2.]))
    if mode=='nonpositive':p0['value']=0.
    if mode=='nonfinite':p0['gradient'][:]=1e308
    def never(x):pytest.fail('unexpected query')
    if mode=='nonfinite':
        with np.errstate(over='ignore'):
            with pytest.raises(ValueError,match='NONFINITE_SMALL_STEP_SLOPE'):m.propose(never,p0)
    else:
        _,r=m.propose(never,p0);assert r['trials']==0


def test_actual_invented_DF_accepts_once_after_independent_state():
    g,J,U,s=invented();events=[];counts={}
    def call(k,l,f):counts[k]=counts.get(k,0)+1;return f()
    j,u,r=m.optimize(g,J,U,s,call=call,record=lambda *e:events.append(e))
    assert r['status']=='QUALIFIED_SMALL_ARMIJO_STEP_ACCEPTED' and r['accepted_U_updates']==1
    assert r['terminal_label']=='1' and r['terminal_kind']=='accepted_U_update'
    assert counts==dict(inner_J_solve=6,precision_state=4,reduced_admission=2,independent_terminal_assessment=1)
    q=r['independent_terminal_assessment'];assert q['J_numerically_qualified']
    assert all(r['terminal_qualification']['admission_gates'].values())
    assert q['state']['J_stationarity']['status']=='MET'
    assert [e[0] for e in events].index('independent_terminal_assessment')<[e[0] for e in events].index('accepted_U_update')
    arrays=next(e[3] for e in events if e[0]=='independent_terminal_state')
    np.testing.assert_array_equal(j,arrays['J']);np.testing.assert_array_equal(u,arrays['U'])
    assert r['DF_budget'] is r['PR_performance'] is None
    assert r['outer_solver_invocations']==r['outer_BFGS_invocations']==0


@pytest.mark.parametrize('mode',['unqualified','mismatch'])
def test_independent_terminal_failure_never_commits_update(monkeypatch,mode):
    g,J,U,s=invented();events=[];real=m.assess
    def assess(*a,**k):
        q=real(*a,**k)
        if mode=='mismatch':raise ValueError('INDEPENDENT_TERMINAL_STATE_MISMATCH')
        return dict(q,status='POST_UPDATE_J_UNQUALIFIED',J_numerically_qualified=False)
    monkeypatch.setattr(m,'assess',assess)
    if mode=='mismatch':
        with pytest.raises(ValueError,match='INDEPENDENT_TERMINAL_STATE_MISMATCH'):
            m.optimize(g,J,U,s,record=lambda *e:events.append(e))
    else:
        j,u,r=m.optimize(g,J,U,s,record=lambda *e:events.append(e))
        assert r['status']=='POST_UPDATE_J_UNQUALIFIED' and r['accepted_U_updates']==0
        assert r['terminal_label']=='0' and r['terminal_kind']=='qualified_initial_no_update'
        np.testing.assert_array_equal(u,U)
    assert not any(e[0]=='accepted_U_update' for e in events)


def test_real_assessment_checks_gradient_and_does_not_hide_U_unconvergence(monkeypatch):
    g,J,U,s=invented();oracle=m.Oracle(g,J,U,s,m.DEFAULT)
    p=oracle.evaluate(np.zeros(s.rank*s.norb*(s.norb-1)//2));q=m.assess(g,p,U,s)
    assert q['U_first_order_met']==(q['state']['U_stationarity']['status']=='MET')
    bad=dict(p,gradient=p['gradient']+1.)
    with pytest.raises(ValueError,match='INDEPENDENT_TERMINAL_STATE_MISMATCH'):m.assess(g,bad,U,s)


def test_precision_loss_is_separate_from_J_and_reduced_admission(monkeypatch):
    g,J,U,s=invented();real=m.Oracle.evaluate
    def evaluate(self,eta):
        p=real(self,eta)
        p['qualification']['raw_solver_flags'][0]=dict(success=False,status=2,message='artificial precision loss flag')
        return p
    monkeypatch.setattr(m.Oracle,'evaluate',evaluate)
    _,_,r=m.optimize(g,J,U,s)
    assert r['accepted_U_updates']==1
    assert all(p['raw_solver_flags'][0]['status']==2 for p in r['point_qualifications'])
    assert all(p['admission_gates']['final_independent_J'] for p in r['point_qualifications'])


def test_policy_and_RC2_do_not_silently_change_scope():
    with pytest.raises(ValueError,match='SMALL_STEP_POLICY'):replace(m.STEP,fraction=0.)
    g,J,U,s=invented(role='rc2')
    with pytest.raises(ValueError,match='PR_ONLY'):m.optimize(g,J,U,s)

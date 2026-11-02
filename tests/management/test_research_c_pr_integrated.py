"""Synthetic recentered trajectories; no saved molecular values."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_integrated as m
from test_research_c_separated_J_U import invented


def physical(label,J,U):
    return dict(budget_PASS=True,exact_tail_lambda_Ha=float(np.sum(abs(m.stable.pr.coefficients(J[1:],np)))))


def test_genuine_multistep_recenter_changes_direction_preserves_cold_J(monkeypatch):
    g,J,U,s=invented();events=[];calls={};bases=[];seeds=[]
    real=m.small.Oracle.__init__
    def init(self,g,j,u,*a,**k):bases.append(u.copy());seeds.append(j.copy());real(self,g,j,u,*a,**k)
    monkeypatch.setattr(m.small.Oracle,'__init__',init)
    def call(k,l,f):calls[k]=calls.get(k,0)+1;return f()
    r=m.trajectory(g,J,U,s,physical,max_updates=3,call=call,record=lambda *x:events.append(x))
    assert r['optimization_accepted_updates']>=2 and r['physically_qualified_updates']==r['optimization_accepted_updates']
    assert all(np.array_equal(x,J) for x in seeds)
    assert not np.array_equal(bases[1],bases[2])
    values=[x['state']['components']['objective'] for x in r['history']]
    assert all(b<a for a,b in zip(values,values[1:]))
    assert r['inner_solves']==calls['inner_J_solve'] and r['refinement_solves']==2*r['point_evaluations']
    directions=[x[3]['direction'] for x in events if x[0]=='small_trial_attempt']
    assert not np.array_equal(directions[0],directions[1])
    np.testing.assert_allclose(r['terminal']['U'].transpose(0,2,1)@r['terminal']['U'],np.broadcast_to(np.eye(2),U.shape),atol=1e-12)
    assert r['best']['row']['physical']['budget_PASS'] and r['certified_descent'] is None


def test_physical_failure_stops_future_updates_and_keeps_previous(monkeypatch):
    g,J,U,s=invented();count=[]
    def assessment(label,j,u):
        count.append(label);return dict(physical(label,j,u),budget_PASS=label=='B_initial')
    r=m.trajectory(g,J,U,s,assessment,max_updates=3)
    assert r['status']=='ACCEPTED_CANDIDATE_DF_BUDGET_NOT_MET'
    assert count==['B_initial','update001'] and r['optimization_accepted_updates']==1
    assert r['physically_qualified_updates']==0 and r['best']['row']['id']=='B_initial' and not r['can_performance']


@pytest.mark.parametrize('mode',['all_unqualified','first_unqualified','correctness'])
def test_registered_trial_grid_not_unlimited(monkeypatch,mode):
    g,J,U,s=invented();real=m.small.Oracle.evaluate;seen=[]
    def evaluate(self,eta):
        p=real(self,eta)
        if np.any(eta):
            seen.append(eta.copy())
            if mode=='correctness':raise ValueError('SYNTHETIC_CORRECTNESS_FAIL')
            if mode=='all_unqualified' or len(seen)==1:raise m.stable.Unqualified('REDUCED_GRADIENT_UNQUALIFIED')
        return p
    monkeypatch.setattr(m.small.Oracle,'evaluate',evaluate)
    if mode=='correctness':
        with pytest.raises(ValueError,match='SYNTHETIC_CORRECTNESS_FAIL'):m.trajectory(g,J,U,s,physical,max_updates=2)
        assert len(seen)==1
    else:
        r=m.trajectory(g,J,U,s,physical,max_updates=2)
        if mode=='all_unqualified':assert len(seen)==3 and r['status']=='FINITE_STEP_GRID_NO_ACCEPT'
        else:assert r['optimization_accepted_updates']>=1


def test_physical_initial_NOT_MET_has_no_outer_trial():
    g,J,U,s=invented();r=m.trajectory(g,J,U,s,lambda *a:dict(budget_PASS=False),max_updates=2)
    assert r['status']=='B_INITIAL_DF_BUDGET_NOT_MET' and not r['trials'] and not r['can_performance']

"""Synthetic replay, finite scope and fresh-only statistics; no real H4 runs."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from compiled_price_artificial import fixture, ROOT, P
from trottertracks import p2_direct_pilot as Q
import run_p2_direct_pilot as runner


def test_plan_has_exact_replay_and_fresh_scope_with_frozen_carrier_seeds():
    plan = Q.make_plan(ROOT)
    assert plan == Q.make_plan(ROOT)
    assert plan['compile_budget'] == 20 and plan['trajectory_paths'] == 8
    assert [(r['m'],r['index']) for r in plan['replay']] == [(0,0),(6,0)]
    seed = P.trial_seed_function(ROOT)
    for cell,row in zip(plan['cells'],plan['menus']['H4']['candidates']):
        assert cell['n'] == 4 and cell['cohort'] == 'D_P2_DIRECT_PILOT_V1'
        assert cell['seeds'] == [[seed(row['candidate'],Q.COHORT,i,j,0)
            for j in range(row['candidate']['q'])] for i in range(4)]
    assert sum(c['n']*2 for c in plan['cells']) + len(plan['replay'])*2 == 20


@pytest.mark.parametrize('negative',[False,True])
def test_saved_synthetic_requests_restore_without_any_sampler(monkeypatch,negative):
    s,ctx,c,cell,plan,requests = fixture(2,5,negative=negative)
    from trotterlib import rte
    from trotterlib.df_rte_circuit import DFRTEEventPreparation
    def forbidden(*args,**kwargs):
        raise AssertionError('Replay must not use any sampler')
    monkeypatch.setattr(rte,'sample_rte_events',forbidden)
    monkeypatch.setattr(rte,'iter_sample_rte_events',forbidden)
    monkeypatch.setattr(DFRTEEventPreparation,'sample_occurrence_request',forbidden)
    ctx.h['require'] = lambda ok,reason: None if ok else (_ for _ in ()).throw(ValueError(reason))
    s.functions(ROOT,P.CARRIER,'scripts/h46_pr_pilot_repair.py',['canonical_tau'],ctx.h)
    saved = dict(events=[[e.to_dict() for e in r.rte_occurrence.events] for r in requests],
        seeds=[r.seed for r in requests], laws=[dict(steps=5, orders=list(r.rte_distribution.orders),
        probabilities=list(r.rte_distribution.order_probabilities)) for r in requests],conditional=False)
    restored,seeds,laws = Q.restore_requests(ctx,c,saved)
    assert seeds == saved['seeds'] and laws == saved['laws']
    assert [[e.to_dict() for e in r.rte_occurrence.events] for r in restored] == saved['events']
    bad = deepcopy(saved)
    bad['events'][0][0]['rotation_angle'] += .001
    with pytest.raises(ValueError,match='METADATA'):
        Q.restore_requests(ctx,c,bad)


def test_replay_comparison_rejects_any_required_difference():
    keys = ('event_digest','seeds','C','S','fingerprints','compiled','compiler')
    reference = {k: 'same' for k in keys}
    assert all(Q.compare_replay(reference,reference).values())
    for k in keys:
        bad = dict(reference,**{k:'different'})
        with pytest.raises(ValueError,match=k):
            Q.compare_replay(bad,reference)


def artificial_rows(plan):
    return [dict(phase='fresh', cell_id=c['id'], candidate_id=c['candidate_id'], index=i,
        cohort=Q.COHORT,conditional=False,C=[10+i,11+2*i],S=[8,7],
        compiler=plan['compiler']['settings_hash'],event_digest='toy'+str(i),fingerprints=['x','y'])
        for c in plan['cells'] for i in range(4)]


def test_fresh_statistics_exact_and_development_never_pooled():
    plan = Q.make_plan(ROOT)
    rows = artificial_rows(plan)
    old = dict(rows[0],phase='replay',cohort=P.COHORT,C=[999,999])
    result = Q.fresh_summary(plan,rows+[old],'TEST')
    for r in result:
        assert r['n_paths'] == 4
        assert r['X']['mean_D'] == Fraction(7,2)
        assert r['Y']['sample_variance_D'] == Fraction(20,3)
        assert r['XY_covariance_D'] == Fraction(10,3)
        assert r['S']['sample_variance'] == [0,0]
        assert r['C']['sample_variance'] == [Fraction(5,3),Fraction(20,3)]
        assert r['uncertainty']['total_CI'] is None
    with pytest.raises(ValueError,match='INCOMPLETE'):
        Q.fresh_summary(plan,rows[:-1],'TEST')
    bad=deepcopy(rows)
    bad[0]['cohort']=P.COHORT
    with pytest.raises(ValueError,match='WRONG'):
        Q.fresh_summary(plan,bad,'TEST')


def test_sample_size_is_only_future_design():
    plan=Q.make_plan(ROOT)
    summary=Q.fresh_summary(plan,artificial_rows(plan),'TEST')
    structural=[dict(m=m,axis=a,mean_S=100.,numerical_enclosure=[99.99,100.01])
        for m in (0,6) for a in ('cosine','sine')]
    result=Q.planning(summary,structural)
    assert all(not r['confidence_claim'] and r['proposed_n_paths'] >= 4 for r in result)
    assert 'LAW/RARE UNRESOLVED' in result[0]['assumptions']
    assert isinstance(Q.json_safe(summary)[0]['X']['mean_D'],str)


def test_failed_replay_gate_does_not_launch_fresh(monkeypatch,tmp_path):
    plan=Q.make_plan(ROOT)
    fake=SimpleNamespace(context=lambda system:SimpleNamespace())
    monkeypatch.setattr(runner,'preflight',lambda *a:plan)
    monkeypatch.setattr(runner.P,'load_c1_backend',lambda root:dict(r_price_phase1_science=SimpleNamespace(Backend=object)))
    monkeypatch.setattr(runner.P,'execution_plan',lambda *a:plan)
    monkeypatch.setattr(runner.F,'make_backend_class',lambda *a:lambda *a:fake)
    seen=[]
    def isolated(jobs,out):
        seen.extend(jobs)
        assert all(j[0]=='replay' for j in jobs)
        return [dict(ok=False,exitcode=0,error='synthetic mismatch')]
    monkeypatch.setattr(runner,'isolated',isolated)
    monkeypatch.setattr(runner.sys,'argv',['run','--execute','--source','TEST','--plan',str(tmp_path/'plan'),
        '--out',str(tmp_path/'out'),'--private',str(tmp_path/'private')])
    (tmp_path/'plan').write_text('{}')
    with pytest.raises(ValueError,match='FRESH_NOT_STARTED'):
        runner.main()
    assert len(seen)==2
    stop=json.loads((tmp_path/'out/STOP.json').read_text())
    assert stop['fresh']==[] and stop['status']=='STOP'

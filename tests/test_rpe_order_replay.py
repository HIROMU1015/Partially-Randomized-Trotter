"""Artificial integration of W planning, live P state and causal replay clocks.

Every numerical value below is artificial.  No archived outcome, real signal,
scientific DP or compiler is opened or executed.
"""
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_priority_search as P
from trottertracks import rpe_schedule_search as W
from trottertracks.rpe_candidate_ordering import Selector
import check_rpe_priority_search as CP
import replay_rpe_candidate_ordering as R

FIXED_W = types.ModuleType('_fixed_wide_d8e806265')
exec(compile(subprocess.check_output(['git', 'show', 'd8e806265:src/trottertracks/rpe_schedule_search.py'], cwd=ROOT),
             'd8e806265:src/trottertracks/rpe_schedule_search.py', 'exec'), FIXED_W.__dict__)


def family(ld=5):
    return W.Family(ld, [dict(m=m, c=F(1), q=1, J=1) for m in range(13)])


def candidate(cid='A', ld=5, kappa='1', scale=1):
    c = CP.config(cid, ld, kappa, scale)
    c['kappa_value'] = kappa
    return c


def vector(c):
    return tuple(n for n in c['caps'] for axis in O.AXES)


def proposal(c, decrement=1):
    v = list(vector(c));v[0] -= decrement;v[1] -= decrement;v = tuple(v)
    return dict(configuration_id=c['configuration_id'], vector=v, candidate=c['name'], method='PR_PAULI',
                cost=O.cost(v, c['prices']), cls='UPDATE', origin=dict(kind='ARTIFICIAL_PROPOSAL'))


def record(status=O.PASS, value=F(1, 2)):
    return dict(status=status, MSE_lower=value, MSE_upper=value, source='ARTIFICIAL_INITIAL', calls=0)


class HiddenValue(Mapping):
    def __init__(self, value):
        self.value, self.reads = dict(value), 0
    def __iter__(self):
        return iter(self.value)
    def __len__(self):
        return len(self.value)
    def __getitem__(self, key):
        self.reads += 1
        return self.value[key]


class ArtificialIndex(dict):
    """An explicit, deterministic synthetic oracle with no filesystem access."""
    def __init__(self, function, seconds=1.):
        super().__init__();self.function = function;self.seconds = seconds;self.requests = []
    def get(self, key, default=None):
        self.requests.append(key)
        if key not in self:
            lo, up = self.function(*key)
            self[key] = dict(seconds=self.seconds, value=dict(status=O.PASS if up <= 1 else O.FAIL,
                                                           MSE_lower=lo, MSE_upper=up))
        return super().get(key, default)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.no_archive = patch.object(Path, 'read_bytes', side_effect=AssertionError('No real archive reads in synthetic test'))
        self.no_archive.start();self.addCleanup(self.no_archive.stop)

    def make_joint(self, cands, seeds, families=None, sched=None, index=None, *, budget=100, slots=1,
                   lookahead=1, wide=None, batch=12, max_cycles=1, policy='ROUND05'):
        if index is None:
            index = ArtificialIndex(lambda cid, v: (F(1, 2), F(1, 2)))
        clock = R.ReplayClock(index, seeds, cands, slots, budget)
        joint = R.JointReplay(cands, seeds, families or {}, sched or {}, clock,
            dict(CP.SEARCH, max_steps_per_method=batch, lookahead=lookahead, lookahead_mode='HYBRID'),
            F(1), CP.none_generator, policy, wide=wide, batch=batch, max_cycles=max_cycles)
        return joint, clock

    def test_outer_init_refine_and_floor_match_fixed_round05_planner(self):
        fam = family();cands = [candidate('K1'), candidate('K4', kappa='4'), candidate('K16', kappa='16')]
        for c in cands:
            c['prices'] = [F(1)] * 26
        cases = [([], dict(max_kappa='64', per_decision=2)),
                 ([dict(family=5, kappa=F(k), status='FEASIBLE', best=F(cost), cid='K'+str(k))
                   for k, cost in ((1, 1400), (4, 1000), (16, 1100))], dict(max_kappa='16')),
                 ([dict(family=5, kappa=F(1), status='FEASIBLE', best=F(10), cid='K1')], dict(max_kappa='64'))]
        for points, settings in cases:
            with self.subTest(points=points):
                joint, clock = self.make_joint(cands, {}, {5: fam}, wide=settings)
                joint.points = deepcopy(points)
                floors = [(p['family'], fam.R(p['kappa']), sum(joint.by_id[p['cid']]['prices'])) for p in points]
                def floor(ld, Rv):
                    lower = [price for family_id, old, price in floors
                             if family_id == ld and all(a <= b for a, b in zip(old, Rv))]
                    return max(lower) if lower else None
                expected, decisions, report = FIXED_W.plan_wide({5: fam}, deepcopy(points), settings, floor)
                self.assertEqual(joint.outer(), expected)
                self.assertEqual(joint.plans[-1]['report'], report)
                if not points:
                    self.assertGreater(len(expected), settings['per_decision'])
                    self.assertEqual([x['reason'] for x in expected], ['INIT_FLOOR_SCHEDULE']+['INIT_FAR_SCHEDULE']*3)
                elif points[0]['best'] == 10:
                    self.assertEqual(report['families'][5]['upper_basis'], 'PROVEN_COST_FLOOR')
                else:
                    self.assertTrue(any(x['reason'].startswith('REFINE') for x in expected))

    def test_A_shot_path_matches_normal_P_and_keeps_explore_counter(self):
        cands = [candidate('A', 5, '2', 1), candidate('B', 5, '1', F(11, 10)), candidate('C', 3, '1', 2)]
        base = sum(w/n for w,n in zip(CP.W, CP.CAPS))
        truth = CP.mse_fn({'A':F(1,2)/base, 'B':F(3,5)/base, 'C':F(7,10)/base})
        seeds = CP.seeded(cands, truth)
        expected = P.run_priority('PR_PAULI', deepcopy(cands), deepcopy(seeds), O.FixtureBackend(truth, F(1)),
            dict(CP.SEARCH, max_steps_per_method=12, lookahead=1, lookahead_mode='HYBRID'), F(1), CP.none_generator, 100)
        joint, clock = self.make_joint(cands, seeds, index=ArtificialIndex(truth), batch=12)
        result = joint.run()
        fields = lambda rows: [(r['configuration_id'], r['vector'], r['cls'], r['status'], r['step']) for r in rows]
        self.assertEqual(fields(result['adopted_trace']), fields(expected['trace']))
        self.assertIn('EXPLORE', [r['cls'] for r in result['adopted_trace']])
        self.assertEqual(len(result['plans']), 1)

    def test_A_emits_complete_fixed_W_prefix_before_shot_action(self):
        fam=family();cands=[candidate('A',kappa='2')]+[candidate('K'+str(k),kappa=str(k))for k in(1,4,16)]
        seeds={('A',vector(cands[0])):record()};settings=dict(max_kappa='16',per_decision=2)
        sched={c['configuration_id']:(5,fam.R(F(c['kappa_value'])),F(c['kappa_value']))for c in cands}
        joint,clock=self.make_joint(cands,seeds,{5:fam},sched,wide=settings,batch=1)
        prior_R=fam.R(2);floor_price=sum(cands[0]['prices'])
        def floor(ld,Rv):
            return floor_price if all(a<=b for a,b in zip(prior_R,Rv)) else None
        expected,_,_=FIXED_W.plan_wide({5:fam},deepcopy(joint.points),settings,floor)
        ids=['S:5:'+','.join(map(str,fam.R(p['kappa'])))for p in expected]
        result=joint.run()
        self.assertGreater(len(ids),settings['per_decision'])
        self.assertEqual([d['selected']for d in result['decisions'][:len(ids)]],ids)
        self.assertTrue(all(d['kind']=='NEW_SCHEDULE'for d in result['decisions'][:len(ids)]))
        self.assertEqual(result['decisions'][len(ids)]['kind'],'SHOT_Q3')

    def test_unknown_schedule_strictly_stops_before_any_Q3(self):
        c = candidate(kappa='4');fam = family();seeds = {('A', vector(c)):record()}
        joint, clock = self.make_joint([c], seeds, {5:fam}, {'A':(5,fam.R(4),F(4))}, wide=dict(max_kappa='4'))
        result = joint.run()
        self.assertIn('PREPARATION', result['censor']['reason'])
        self.assertEqual(result['Q3_launched'], 0)
        self.assertEqual(result['revealed_records'], 0)
        self.assertEqual(len(result['decisions']), 1)

    def test_mapped_but_unprepared_schedule_still_censors(self):
        a = candidate('A', kappa='4');b = candidate('B', kappa='1');b['ready'] = False
        fam = family();seeds = {('A', vector(a)):record()}
        joint, clock = self.make_joint([a,b], seeds, {5:fam},
            {'A':(5,fam.R(4),F(4)), 'B':(5,fam.R(1),F(1))}, wide=dict(max_kappa='4'))
        result = joint.run()
        self.assertIn('PREPARATION', result['censor']['reason'])
        self.assertNotIn('B', joint.active)
        self.assertEqual(result['Q3_launched'], 0)

    def test_unknown_demanded_Q3_stops_after_one_selection(self):
        c = candidate();seeds = {('A', vector(c)):record()}
        joint, clock = self.make_joint([c], seeds, index={})
        result = joint.run()
        self.assertEqual(result['unknown_Q3_requests'], 1)
        self.assertEqual(len(result['decisions']), 1)
        self.assertEqual(result['revealed_records'], 0)
        self.assertEqual(clock.now, 0)

    def test_unknown_speculation_stops_without_revealing_running_demand(self):
        c = candidate();p1,p2,p3 = [proposal(c,i) for i in (1,2,3)]
        secret = HiddenValue(record())
        index = {(p1['configuration_id'],p1['vector']):dict(seconds=3.,value=secret)}
        clock = R.ReplayClock(index, {}, [c], 4, 10);clock.demand = p1
        with self.assertRaises(R.Censored) as censored:
            clock.prefetch([p1,p2,p3])
        self.assertIn('SPECULATION', censored.exception.reason)
        self.assertEqual(len(clock.finishes), 1)
        self.assertEqual(clock.unknown_requests, [R.qid(p2)])
        self.assertEqual(secret.reads, 0)
        self.assertFalse(clock.values)
        self.assertFalse(any(e['id']==R.qid(p3) for e in clock.events))

    def test_completed_unused_speculation_updates_best_all_only(self):
        c = candidate();p1,p2 = [proposal(c,i) for i in (1,2)]
        seeds = {('A',vector(c)):record()}
        index = {(p1['configuration_id'],p1['vector']):dict(seconds=5.,value=record()),
                 (p2['configuration_id'],p2['vector']):dict(seconds=1.,value=record())}
        clock = R.ReplayClock(index, seeds, [c], 2, 10);clock.demand=p1;clock.prefetch([p1,p2])
        answer = clock.answer(p1);clock.adopt(dict(configuration_id='A',vector=p1['vector']))
        joint = R.JointReplay([c],seeds,{}, {},clock,dict(CP.SEARCH,lookahead=1),F(1),CP.none_generator,'ROUND05')
        joint.knowledge[('A',p1['vector'])]=answer
        result=joint.project()
        self.assertEqual(result['completed_not_adopted'],1)
        self.assertEqual(result['best_adopted'],p1['cost'])
        self.assertEqual(result['best_all_completed'],p2['cost'])
        self.assertLess(result['best_all_completed'],result['best_adopted'])

    def test_declared_initial_records_do_not_open_archive_values(self):
        c=candidate();cap=vector(c);secret=HiddenValue(record(O.FAIL,F(20)))
        index={('A',cap):dict(seconds=1000000.,value=secret)}
        seeds={('A',cap):record()};clock=R.ReplayClock(index,seeds,[c],2,10)
        answer=clock.answer(dict(configuration_id='A',vector=cap))
        self.assertEqual(answer['status'],O.PASS)
        self.assertEqual(answer['calls'],0)
        self.assertEqual(secret.reads,0)
        self.assertEqual(len(clock.observations),1)
        self.assertIsNone(clock.observations[0]['seconds'])
        self.assertFalse(clock.opened)

    def test_service_time_and_MSE_stay_private_until_completion(self):
        c=candidate();p1,p2=[proposal(c,i) for i in (1,2)];secret=HiddenValue(record())
        clock=R.ReplayClock({('A',p1['vector']):dict(seconds=13.,value=secret)}, {}, [c], 2, 10)
        action=dict(id=R.qid(p2),kind='SHOT_Q3',a_rank=[0],method='PR_PAULI',configuration_id='A',cost=p2['cost'],
            profile=clock.profile,preparation=dict(signal_reuse=True,price_reuse=True,requires_calibration=False))
        def prediction():
            return Selector('IMPROVEMENT_PER_TIME_V1').choose([action],method_best=F(10000),tau=F(1),
                now=clock.now,observations=clock.observations)['trace']['ranks'][0]['timing']['q3_seconds']
        self.assertIsNone(prediction());clock.demand=p1;clock.launch(p1)
        self.assertIsNone(prediction());self.assertEqual(secret.reads,0)
        clock.advance(12.)
        self.assertIsNone(prediction());self.assertEqual(secret.reads,0)
        clock.advance(13.)
        self.assertEqual(prediction(),13.)
        self.assertGreater(secret.reads,0)

    def test_missing_service_time_is_not_zero_or_archive_mean(self):
        c=candidate();p=proposal(c);secret=HiddenValue(record())
        clock=R.ReplayClock({('A',p['vector']):dict(seconds=None,value=secret)}, {}, [c], 1, 10)
        clock.demand=p
        with self.assertRaises(R.Censored)as censored:
            clock.answer(p)
        self.assertIn('SERVICE_TIME_MISSING',censored.exception.reason)
        self.assertEqual(secret.reads,0)
        self.assertEqual(clock.now,0)
        self.assertFalse(clock.finishes)

    def test_budget_includes_speculation_and_free_adoption_after_full(self):
        c=candidate();p1,p2,p3=[proposal(c,i) for i in (1,2,3)]
        index={('A',p['vector']):dict(seconds=1.,value=record()) for p in (p1,p2,p3)}
        clock=R.ReplayClock(index,{},[c],3,2);clock.demand=p1;clock.prefetch([p1,p2,p3])
        self.assertEqual(len(clock.finishes),2)
        clock.answer(p1);clock.answer(p2)
        self.assertEqual(len(clock.finishes),2)
        self.assertEqual(len(clock.values),2)
        with self.assertRaises(R.BudgetStop):
            clock.answer(p3)
        self.assertEqual(len(clock.finishes),2)

    def test_shot_only_budget_stop_is_propagated_without_replanning(self):
        c=candidate();seeds={('A',vector(c)):record()}
        joint,clock=self.make_joint([c],seeds,budget=1,batch=8,max_cycles=3)
        result=joint.run()
        self.assertEqual(result['stop'],'Q3_BUDGET_EXHAUSTED')
        self.assertEqual(result['Q3_launched'],1)
        self.assertEqual(result['completed'],1)
        self.assertEqual(result['adopted'],1)
        self.assertEqual(len(result['plans']),1)
        self.assertEqual(result['unknown_Q3_requests'],0)

    def test_unsolved_point_status_and_no_duplicate_outer_stop_plan(self):
        c=candidate();fam=family();seeds={('A',vector(c)):record(O.FAIL,F(2))}
        joint,clock=self.make_joint([c],seeds,{5:fam},{'A':(5,fam.R(1),F(1))},wide=dict(max_kappa='1'),max_cycles=3)
        result=joint.run()
        self.assertEqual(joint.points[0]['status'],'NO_FEASIBLE_FOUND')
        self.assertEqual(joint.points[0]['min_lower_over_tau'],2)
        self.assertEqual(len(result['plans']),1)

    def test_first_divergence_has_same_pool_and_common_constraints(self):
        a=candidate('A',kappa='4');b=candidate('B',kappa='1');fam=family()
        seeds={('A',vector(a)):record()}
        result=R.compare([a,b],seeds,{5:fam},{'A':(5,fam.R(4),F(4)),'B':(5,fam.R(1),F(1))},
            ArtificialIndex(lambda cid,v:(F(1,2),F(1,2))),dict(CP.SEARCH),F(1),CP.none_generator,
            slots=1,lookahead=1,budget=4,wide=dict(max_kappa='4'))
        divergence=result['first_divergence']
        self.assertIsNotNone(divergence)
        self.assertEqual(divergence['index'],0)
        self.assertTrue(divergence['same_pool'])
        self.assertEqual(divergence['A']['kind'],'NEW_SCHEDULE')
        self.assertEqual(divergence['B']['kind'],'SHOT_Q3')
        for policy in result['policies'].values():
            self.assertLessEqual(policy['Q3_launched'],4)
            self.assertEqual(policy['physical_scientific_calls'],0)
        self.assertEqual(result['conditions']['DP_workers'],8)
        self.assertFalse(result['conditions']['prefix_reuse'])
        self.assertFalse(result['conditions']['shot_exchange'])


if __name__=='__main__':
    unittest.main(verbosity=2)

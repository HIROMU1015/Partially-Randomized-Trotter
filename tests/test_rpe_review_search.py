"""Artificial exploration under the optimizer's exact active price quotes."""
from fractions import Fraction
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_priority_search as P
from trottertracks import rpe_schedule_search as W


def candidate(cid, price, ld):
    return dict(method='PR_PAULI',family='PR',configuration_id=cid,name=cid,ready=True,
                key=dict(L_D=ld,formula='S2',delta_over_h='1'),prices=[price]*26,caps=[4]*13)


class ReviewSearch(unittest.TestCase):
    def exercise(self, enabled, upper=Fraction(1,2), refresh=None):
        candidates=[candidate('leader',1,1),candidate('blocked',2,2),candidate('other',Fraction(3,2),3)]
        known={('leader',(2,)*26):dict(status=O.PASS,MSE_upper=upper,source='FIXTURE')}
        asks=[];standby_seen=[];parked=[]
        class Backend:
            def answer(_, proposal):
                asks.append(proposal['configuration_id'])
                return dict(status=O.FAIL,MSE_upper=Fraction(3),source='ARTIFICIAL',calls=1)
        result=P.run_priority('PR_PAULI',candidates,known,Backend(),
            dict(self_price_floor=enabled,seed_rules=[],decrement_steps=[1],large_step_headroom='1/2',
                 max_steps_per_method=2,explore_every=1,kappa_neighbourhood=True),
            Fraction(1),lambda *args:None,8,
            prefetch_state=lambda:dict(capacity=4),
            standby=lambda choice,ps:standby_seen.extend(p['configuration_id'] for p in ps),
            retire_priced_out=lambda rows:parked.extend(rows),refresh_ready=refresh)
        return result,asks+standby_seen,parked

    def test_equal_one_shot_floor_parked_for_demand_standby_and_resume(self):
        enabled,offered,parked=self.exercise(True)
        old,old_offered,_=self.exercise(False)
        self.assertIn('blocked',old_offered)
        self.assertIn('other',offered)
        self.assertNotIn('blocked',offered)
        self.assertIn('blocked',parked)
        self.assertEqual(enabled['self_price_floor_deferred']['blocked']['floor'],52)
        self.assertNotIn('self_price_floor_deferred',old)

    def test_price_refresh_and_missing_accepted_bound_do_not_permanently_veto(self):
        responses=iter([([],{}),([candidate('leader',4,1)],{})])
        _,offered,parked=self.exercise(True,refresh=lambda:next(responses,([],{})))
        self.assertIn('blocked',parked)
        self.assertIn('blocked',offered)
        _,offered,parked=self.exercise(True,upper=None)
        self.assertIn('blocked',offered)
        self.assertFalse(parked)


    def test_explore_prefers_unvisited_family_over_cheaper_probe_in_known_family(self):
        lead=candidate('leader',1,1)
        familiar=candidate('known_family',2,1)
        new=candidate('new_family',3,2)
        new['key']['delta_over_h']='2'
        known={('leader',(2,)*26):dict(status=O.PASS,MSE_upper=Fraction(1,2),source='FIXTURE')}
        def first(review_order):
            asks=[]
            class Backend:
                def answer(_,proposal):
                    asks.append(proposal['configuration_id'])
                    return dict(status=O.FAIL,MSE_upper=Fraction(3),source='ARTIFICIAL',calls=1)
            P.run_priority('PR_PAULI',[lead,familiar,new],dict(known),Backend(),
                dict(review_search_order=review_order,seed_rules=[],decrement_steps=[1],
                     max_steps_per_method=1,explore_every=1,large_step_headroom='1/2'),
                Fraction(1),lambda *args:None,1)
            return asks
        self.assertEqual(first(False),['known_family'])
        self.assertEqual(first(True),['new_family'])

    def test_exchange_only_for_leading_configuration_after_decrements(self):
        leader=candidate('leader',4,1)
        leader['prices']=[1]*24+[20,20]
        other=candidate('other',8,2)
        other['prices']=[2]*24+[40,40]
        base=(1,)*24+(2,2)
        known={(c['configuration_id'],base):dict(status=O.PASS,MSE_upper=Fraction(1,2),source='FIXTURE')
               for c in (leader,other)}
        asks=[]
        class Backend:
            def answer(_,proposal):
                asks.append((proposal['configuration_id'],proposal['origin']['kind']))
                return dict(status=O.FAIL,MSE_upper=Fraction(7,8),source='ARTIFICIAL',calls=1)
        P.run_priority('PR_PAULI',[leader,other],known,Backend(),
            dict(review_search_order=True,seed_rules=[],decrement_steps=[1],max_steps_per_method=20,large_step_headroom='1/2',
                 small_and_deferred='STOP',small_gain_fraction='0',explore_every=2,
                 allocation_exchange=dict(enabled=True,max_moves=1,add_steps=[1])),
            Fraction(1),lambda *args:None,3)
        self.assertIn(('leader','COST_DECREASING_ROUND_EXCHANGE'),asks)
        self.assertNotIn(('other','COST_DECREASING_ROUND_EXCHANGE'),asks)
        self.assertLess(asks.index(('leader','DECREMENT_FROM_CONFIGURATION_INCUMBENT')),
                        asks.index(('leader','COST_DECREASING_ROUND_EXCHANGE')))


class WidthFloor(unittest.TestCase):
    def test_width_planner_uses_valid_model_incumbent_and_leaves_unmatched_family_open(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import run_rpe_schedule_search as entry
        rows=W.family_rows(Fraction(3,2),Fraction(1,10),Fraction(1,10))
        supported=W.Family('pauli',rows)
        families={'pauli':supported,'unproven':W.Family('unproven',rows)}
        driver=entry.WidthDriver.__new__(entry.WidthDriver)
        driver.spec=dict(search_overrides=dict(common_pauli_price_floor=True))
        driver.problem=dict(accuracy=dict(epsilon_E_decimal='1'))
        driver.state=dict(families={key:dict(configuration={'family':key},method='PR_PAULI' if key=='pauli' else 'PR_DF')
                                    for key in families},points=[])
        driver.completion_api=lambda:SimpleNamespace(common_pauli_model_terms=lambda configs:{'pauli':[(10,2)]*26})
        point=dict(family='pauli',kappa=supported.floor_rep(),best=780,best_MSE_upper=Fraction(1,2),
                   best_N_X=[3]*13,best_N_Y=[3]*13,status='FEASIBLE')
        driver.state['points']=[point]
        floor,cutoff=driver.model_floor(families)
        self.assertEqual(cutoff,780);self.assertIsNone(floor('unproven',supported.floor_R))
        proposals,_,report=W.plan_wide(families,[point],dict(max_kappa=1000),floor,price_floor_best=cutoff)
        self.assertEqual(report['families']['pauli']['upper_basis'],'PROVEN_COST_FLOOR')
        self.assertEqual(report['families']['unproven']['upper_basis'],'RANGE_LIMIT')
        self.assertTrue(all(floor('pauli',supported.R(p['kappa']))<cutoff for p in proposals if p['family']=='pauli'))
        point['best']=779  # stale price provenance cannot set the proof threshold
        self.assertEqual(driver.model_floor(families),(None,None))


if __name__=='__main__':
    unittest.main(verbosity=2)

"""Synthetic ordering/timing contracts; no quantum data, kernel or real run."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trottertracks.rpe_candidate_ordering import Selector, estimate_seconds


def action(identity='local', *, rank=0, kind='SHOT_Q3', cost=90, width=.2, **kwargs):
    value = dict(id=identity, kind=kind, a_rank=[rank], method='ARTIFICIAL', L_D=1, width=width,
                 cost=None if cost is None else F(cost), configuration_id=identity, parent_cost=F(100),
                 parent_MSE_upper=F(1, 2), untried=False, profile={'workers': 2}, q3_seconds=2.,
                 preparation=dict(signal_reuse=True, price_reuse=True, requires_calibration=False))
    value.update(kwargs)
    return value


def observation(identity, *, finished=3, seconds=4, **kwargs):
    value = dict(id=identity, kind='SHOT_Q3', method='ARTIFICIAL', configuration_id='prior',
                 status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', MSE_upper=F(1, 3),
                 finished_at=finished, seconds=seconds, profile={'workers': 2})
    value.update(kwargs)
    return value


class OrderingTests(unittest.TestCase):
    def choose(self, selector, actions, **kwargs):
        return selector.choose(actions, method_best=F(100), tau=F(1), now=kwargs.pop('now', 0), **kwargs)

    def test_a_is_exact_stable_a_rank_even_when_b_features_favor_other_action(self):
        actions = [action('second', rank=1, cost=1), action('first', rank=0, cost=99), action('tie', rank=0)]
        result = self.choose(Selector(), actions)
        self.assertIs(result['chosen'], actions[1])
        self.assertFalse(result['trace']['changed_from_a_rank'])

    def test_cheap_new_schedule_beats_local_shot_without_using_floor(self):
        actions = [action(), action('new', kind='NEW_SCHEDULE', rank=1, cost=20, untried=True, price_floor=F(1))]
        result = self.choose(Selector('IMPROVEMENT_PER_TIME_V1'), actions)
        self.assertEqual(result['chosen']['id'], 'new')
        self.assertTrue(result['trace']['changed_from_a_rank'])
        self.assertEqual(result['trace']['ranks'][1]['immediate_saving'], 80)
        self.assertEqual(result['trace']['ranks'][1]['pass_rate']['value'], F(1, 2))

    def test_expensive_calibration_reverses_the_same_price_order(self):
        actions = [action(), action('new', kind='NEW_SCHEDULE', rank=1, cost=20, untried=True,
            preparation=dict(signal_reuse=True, price_reuse=False, requires_calibration=True,
                             seconds=60, source='ARTIFICIAL_FIXED_CALIBRATION_PRIOR'))]
        result = self.choose(Selector('IMPROVEMENT_PER_TIME_V1'), actions)
        self.assertEqual(result['chosen']['id'], 'local')
        row = result['trace']['ranks'][1]
        self.assertEqual(row['timing']['total_seconds'], 62)
        self.assertTrue(row['preparation']['requires_calibration'])
        self.assertEqual(row['timing']['preparation_source'], 'ARTIFICIAL_FIXED_CALIBRATION_PRIOR')

    def test_unknown_price_floor_is_never_a_point_prediction_and_is_explored(self):
        actions = [action(), action('unknown', kind='NEW_SCHEDULE', rank=1, cost=None, price_floor=F(1),
                                  preparation=dict(requires_calibration=True), q3_seconds=None, untried=True)]
        selector = Selector('IMPROVEMENT_PER_TIME_V1', explore_every=4)
        chosen = [self.choose(selector, actions) for _ in range(4)]
        self.assertEqual([x['chosen']['id'] for x in chosen], ['local', 'local', 'local', 'unknown'])
        row = chosen[-1]['trace']['ranks'][1]
        self.assertIsNone(row['immediate_saving'])
        self.assertIsNone(row['timing']['total_seconds'])
        self.assertIn('CALIBRATION_REQUIRED_TIME_UNKNOWN', row['timing']['unknown_reasons'])

    def test_unmeasured_times_are_null_and_use_explicit_cost_fallback(self):
        actions = [action(q3_seconds=None), action('cheap', rank=1, cost=20, q3_seconds=None)]
        result = self.choose(Selector('IMPROVEMENT_PER_TIME_V1'), actions)
        self.assertEqual(result['chosen']['id'], 'cheap')
        self.assertIn('FALLBACK', result['trace']['reason'])
        self.assertTrue(all(r['timing']['q3_seconds'] is None for r in result['trace']['ranks']))

    def test_fresh_untried_shots_do_not_starve_unknown_schedules(self):
        selector = Selector('IMPROVEMENT_PER_TIME_V1', explore_every=4)
        unknown = action('unknown', kind='NEW_SCHEDULE', rank=2, cost=None, untried=True, q3_seconds=None)
        for step in range(4):
            result = self.choose(selector, [action('fresh-shot-'+str(step), untried=True), unknown])
        self.assertEqual(result['chosen']['id'], 'unknown')

    def test_future_timing_and_result_do_not_change_pre_choice_features(self):
        actions = [action(q3_seconds=None), action('cheap', rank=1, cost=20, q3_seconds=None)]
        future = [observation('future', finished=10, seconds=100, status='FAIL')]
        clean = self.choose(Selector('IMPROVEMENT_PER_TIME_V1'), actions)
        hidden = self.choose(Selector('IMPROVEMENT_PER_TIME_V1'), actions, observations=future)
        self.assertEqual(clean, hidden)
        visible = self.choose(Selector('IMPROVEMENT_PER_TIME_V1'), actions, now=10, observations=future)
        self.assertEqual(visible['trace']['ranks'][0]['timing']['q3_seconds'], 100)
        self.assertEqual(visible['trace']['ranks'][0]['pass_rate']['value'], F(1, 3))

    def test_own_held_out_record_and_different_profile_are_not_training_data(self):
        current = action(q3_seconds=None)
        records = [observation('local', finished=0, seconds=999, status='FAIL'),
                   observation('different-profile', finished=0, seconds=3, profile={'workers': 8})]
        self.assertIsNone(estimate_seconds(current, now=1, observations=records)['q3_seconds'])
        result = self.choose(Selector('IMPROVEMENT_PER_TIME_V1'), [current], observations=records, now=1)
        self.assertNotIn('local', result['trace']['ranks'][0]['pass_rate']['observation_ids'])

    def test_completed_configuration_group_precedes_method_group_and_no_double_count(self):
        current = action(q3_seconds=None)
        records = [observation('same', configuration_id='local', seconds=6), observation('other', seconds=50),
                   observation('preparation-plus-q3', kind='NEW_SCHEDULE', seconds=1000)]
        timing = estimate_seconds(current, now=3, observations=records)
        self.assertEqual(timing['q3_seconds'], 6)
        self.assertEqual(timing['timing_observation_ids'], ['same'])
        self.assertEqual(timing['preparation_seconds'], 0)
        self.assertIsNone(estimate_seconds(current, now=3, observations=records[2:])['q3_seconds'])

    def test_exact_width_representations_remain_opaque_and_distinct(self):
        actions = [action('binary64', width=.2), action('rational', rank=1, width='1/5')]
        before = deepcopy(actions)
        result = self.choose(Selector('IMPROVEMENT_PER_TIME_V1'), actions)
        self.assertEqual(actions, before)
        self.assertEqual([r['width'] for r in result['trace']['ranks']], [.2, '1/5'])
        self.assertEqual([r['id'] for r in result['trace']['ranks']], ['binary64', 'rational'])

    def test_resume_preserves_periodic_exploration_and_never_prunes_failed_scores(self):
        actions = [action(), action('unknown', rank=1, cost=None, q3_seconds=None, untried=True)]
        selector = Selector('IMPROVEMENT_PER_TIME_V1')
        for _ in range(3):
            self.choose(selector, actions)
        restored = Selector.restore(selector.snapshot())
        self.assertEqual(self.choose(selector, actions), self.choose(restored, actions))
        result = self.choose(restored, actions, now=5, observations=[observation('failed', status='FAIL')])
        self.assertEqual(len(result['trace']['ranks']), 2)
        self.assertTrue(all(r['pass_rate']['value'] > 0 for r in result['trace']['ranks']))


if __name__ == '__main__':
    unittest.main()

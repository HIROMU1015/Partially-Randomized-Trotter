"""Artificial scalar/law fixtures only; never read saved H6 science values."""
from copy import deepcopy
from fractions import Fraction
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h6_confirmation_sample_design as d
import run_h6_confirmation_s_stats as runner


def policy():
    return runner.c.read(runner.ROOT/runner.POLICY)


def plan():
    targets = []
    for q, R, eps, pf, sign, mean in [(2,8,.05,'B0I_p7_q1_2nd',-1,'11'),
                                      (8,32,.0005,'B0I_p11_q4_4th',1,'9')]:
        targets.append(dict(scope='H6_CONFIRM_P4_Q'+str(q), epsilon=eps, fixed_PF_id=pf,
            expected_direction=sign, cell=dict(id=f'B2_p4_q{q}_2nd_r4_K2',prefix=4,q=q,r=4,K=2,R=R),
            fixed_PF_G_exact='10', saved_n8=dict(n=8,mean_exact=mean,paired_SE=.05),
            law=dict(finite_taylor_order=2,orders=[0,2],order_probabilities=[.99,.01]),
            original_law_sha256='ARTIFICIAL_LAW'))
    return dict(binding=dict(artificial=True),targets=targets)


class ArtificialSampleDesign(unittest.TestCase):
    def test_positive_and_negative_margins_reuse_same_H8_body(self):
        plus = d.signed_normal_scenario('9', .5, 8, '10', 1)
        minus = d.signed_normal_scenario('11', .5, 8, '10', -1)
        original = d.shared.normal_scenario('9', .5, 8, '10')
        self.assertEqual(plus['planning_n'], minus['planning_n'])
        self.assertEqual(plus['pilot_SD'], minus['pilot_SD'])
        self.assertEqual(plus['normal_ceiling_n'], original['normal_plugin_n'])
        self.assertEqual(minus['saved_PR_mean_exact'], '11')
        self.assertEqual(minus['signed_margin_exact'], '-1')

    def test_wrong_direction_tie_and_invalid_SE_rejected(self):
        for mean, se, sign in [('10', .1, 1),('11', .1, 1),('9', .1, -1),('9', 0, 1),('9',float('nan'),1)]:
            with self.subTest(mean=mean,se=se,sign=sign),self.assertRaises(ValueError):
                d.signed_normal_scenario(mean,se,8,'10',sign)

    def test_minimum_two_preserves_unrounded_normal_ceiling(self):
        result = d.signed_normal_scenario('9', .0001, 8, '10', 1)
        self.assertEqual(result['normal_ceiling_n'], 1)
        self.assertEqual(result['planning_n'], 2)

    def test_exact_negative_fraction_and_translation_invariance(self):
        one = d.signed_normal_scenario('2/3', .1, 8, '1/3', -1)
        two = d.signed_normal_scenario('5/3', .1, 8, '4/3', -1)
        self.assertEqual(Fraction(one['signed_margin_exact']), Fraction(-1,3))
        self.assertEqual(one['planning_n'], two['planning_n'])

    def test_variance_scenarios_monotone_and_normal_power_reaches_target(self):
        cases = [d.signed_normal_scenario('9', .5, 8, '10', 1,sd_multiplier=f) for f in (1,2,4)]
        self.assertEqual([c['planning_n'] for c in cases],sorted(c['planning_n'] for c in cases))
        for case in cases:
            self.assertGreaterEqual(d.planning_power(case['planning_n'],case), .8-1e-12)
            self.assertFalse(case['actual_power_certified'])

    def test_order2_visibility_reuses_exact_independent_occurrence_formula(self):
        result = d.order2_visibility(.01,8,48)
        self.assertAlmostEqual(result['probability_at_least_one'],1-(1-.01)**(8*48))
        tiny = d.order2_visibility(1e-20,8,48)
        self.assertGreater(tiny['probability_at_least_one'],0)

    def test_visibility_threshold_is_minimal_and_not_mean_precision(self):
        pi = d.order2_visibility(.01,8,48)['order2_trajectory_probability']
        n = d.shared.detection_n(pi,.025)
        self.assertGreaterEqual(d.order2_visibility(.01,8,n)['probability_at_least_one'],.975)
        self.assertLess(d.order2_visibility(.01,8,n-1)['probability_at_least_one'],.975)

    def test_recommendation_uses_all_three_criteria_and_rounds_up(self):
        self.assertEqual(d.recommended_n(48,56,61,8),64)
        self.assertEqual(d.recommended_n(48,2,20,8),48)
        self.assertIsNone(d.recommended_n(48,2,None,8))
        with self.assertRaises(ValueError):d.recommended_n(48,2,20,0)

    def test_artificial_two_target_pass_has_six_scenarios_and_no_final_seal(self):
        result = d.analyze(plan(),policy())
        self.assertEqual(sum(len(c['scenarios']) for c in result['candidates']),6)
        self.assertEqual(result['saved_scalar_passes'],1)
        self.assertEqual([t['expected_direction'] for t in result['candidates']],[-1,1])
        self.assertFalse(result['actual_power_certified'])
        self.assertFalse(result['visibility_is_cost_mean_precision'])
        self.assertIsNone(result['final_scientific_n'])
        self.assertIsNone(result['final_seed_registry'])
        self.assertFalse(result['next_stage_authorized'])

    def test_K4_changed_target_and_changed_policy_rejected(self):
        for key, value in [('K',4),('q',16),('R',64),('prefix',5)]:
            changed=plan();changed['targets'][0]['cell'][key]=value
            with self.assertRaises(ValueError):d.analyze(changed,policy())
        changed=plan();changed['targets'][0]['law']['orders']=[0,2,4]
        with self.assertRaises(ValueError):d.analyze(changed,policy())
        p=policy();p['family_alpha']=.1
        with self.assertRaises(ValueError):d.analyze(plan(),p)

    def test_unapproved_or_science_grant_rejected_before_any_input(self):
        with patch.object(runner.c,'read',side_effect=AssertionError('NO_INPUT_READ')):
            for grant in [dict(approved_by_user=False),dict(approved_by_user=True,runtime_authorization=True,
                one_shot=True,stage=runner.SCOPE,science_sampling_compile_authorized=True,retry=False,next_stage_authorized=False)]:
                with self.assertRaisesRegex(ValueError,'NUMERICAL_ONLY_APPROVAL'):
                    runner.run({},grant)

    def test_artificial_one_shot_runner_preserves_marker_and_reservation(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            root=Path(directory);runner.c.save(root/'plan',plan())
            spec=dict(output=str(root/'output'),lock=str(root/'marker'),SOURCE='a'*40,
                      calls=dict(saved_scalar_pass=1),plan=str(root/'plan'))
            with patch.object(runner,'verify'):
                self.assertEqual(runner.run(spec,dict(artificial=True)),0)
                with self.assertRaisesRegex(ValueError,'UNUSED_OUTPUT_ONE_SHOT'):
                    runner.run(spec,dict(artificial=True))
            self.assertTrue((root/'marker').exists())
            self.assertTrue((root/'output/saved_scalar_pass_1_reserved.json').exists())
            terminal=runner.c.read(root/'output/terminal.json')
            self.assertEqual(terminal['actual_calls'],dict(saved_scalar_pass=1))
            self.assertFalse(terminal['scientific_execution_approved'])


if __name__ == '__main__':unittest.main()

"""Artificial scalar checks; no H-chain inputs, RNG or quantum compiler."""
from fractions import Fraction
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h8_confirmation_sample_design as d


class SampleDesign(unittest.TestCase):
    def test_normal_formula_matches_handbook_example(self):
        x = d.normal_scenario('9', 1/math.sqrt(8), 8, '10', family_alpha=.05,
                              comparisons=1, power=.9)
        self.assertEqual(x['normal_plugin_n'], 9)
        self.assertFalse(x['actual_power_certified'])

    def test_variance_inflation_and_family_correction_never_reduce_n(self):
        values = [d.normal_scenario('9', .5, 8, '10', sd_multiplier=x)['normal_plugin_n'] for x in (1, 2, 4)]
        self.assertEqual(values, sorted(values))
        one = d.normal_scenario('9', .5, 8, '10', comparisons=1)['normal_plugin_n']
        self.assertLessEqual(one, values[0])

    def test_nonpositive_margin_and_zero_unobserved_variance_are_not_saved(self):
        for mean, se in [('10', 1), ('11', 1), ('9', 0), ('9', float('nan'))]:
            with self.subTest(mean=mean,se=se), self.assertRaises(ValueError):
                d.normal_scenario(mean, se, 8, '10')

    def test_exact_fraction_margin(self):
        x=d.normal_scenario('1/3', .1, 8, '2/3')
        self.assertEqual(Fraction(x['margin_exact']), Fraction(1,3))

    def test_tiny_probabilities_do_not_cancel_to_zero(self):
        self.assertAlmostEqual(d.at_least_one(1e-20, 64), 64e-20, delta=1e-32)

    def test_visibility_is_not_mean_precision(self):
        value=d.visibility(.001,1e-9,64,48)
        self.assertGreater(value['n_for_four_visibility_97_5_percent'], 1000000)
        self.assertIsNone(value['cost_bound'])
        self.assertFalse(value['population_mean_certified'])

    def test_detection_size_is_minimal(self):
        n=d.detection_n(.01,.025)
        self.assertGreaterEqual(d.at_least_one(.01,n), .975)
        self.assertLess(d.at_least_one(.01,n-1), .975)

    def test_zero_and_certain_event(self):
        self.assertIsNone(d.detection_n(0,.025))
        self.assertEqual(d.detection_n(1,.025),1)
        self.assertEqual(d.at_least_one(1,0),0)
        self.assertEqual(d.at_least_one(0,48),0)

    def test_zero_count_bound_is_not_zero(self):
        p=d.zero_count_probability_upper(48,.025)
        self.assertAlmostEqual((1-p)**48,.025)

    def test_invalid_law_and_trial_counts_rejected(self):
        for p,n in [(-.1,4),(1.1,4),(.1,-1),(.1,2.5)]:
            with self.subTest(p=p,n=n),self.assertRaises(ValueError): d.at_least_one(p,n)
        with self.assertRaises(ValueError): d.visibility(.9,.2,8,48)

    def test_two_fixed_cohorts_budget(self):
        c=d.calls(48)
        self.assertEqual((c['unconditional_trajectory'],c['wrapper_compile'],c['RTE_event']), (96,192,3456))
        self.assertEqual(c['occurrence'],864)
        self.assertEqual(c['new_signal'],0)


if __name__ == '__main__': unittest.main()

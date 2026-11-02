"""Synthetic design arithmetic and refusal checks; no molecular execution."""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
d = load('design', 'src/trotterlib/h46_confirmation_design.py')
k = load('existing', 'src/trotterlib/higher_pf_readiness.py')

class DesignTests(unittest.TestCase):
    def test_minimum_and_expanded_families(self):
        self.assertEqual((len(d.MINIMAL), len(d.PRIMARY_PAIRS)), (3, 2))
        self.assertEqual((len(d.EXPANDED), len(d.EXPANDED_PAIRS)), (7, 7))
    def test_zero_time_tail_closure(self):
        p = d.paired_law_strata([0.], [32], 1, 1.)['probabilities']
        self.assertEqual(p, {'s0': 1., 's2_single': 0., 's2_multi': 0., 's4': 0.})
    def test_tiny_rare_mass_retained(self):
        p = d.paired_law_strata([1.], [32], 8, .9)['probabilities']
        self.assertGreater(p['s4'], 0); self.assertGreater(p['s2_multi'], 0)
        self.assertAlmostEqual(sum(p.values()), 1.)
    def test_negative_time_same_order_law(self):
        self.assertEqual(d.paired_law_strata([1.], [32], 1, .9), d.paired_law_strata([-1.], [32], 1, .9))
    def test_schedule_rejected(self):
        with self.assertRaises(ValueError): d.paired_law_strata([1.], [31], 1, 1.)
    def test_precision_matches_standard_normal_formula(self):
        from statistics import NormalDist
        z = NormalDist().inv_cdf(.9875)
        self.assertEqual(d.difference_precision_n(4., 5., 2., family_size=2, halfwidth_fraction=.5), math.ceil(z*z*9.))
    def test_four_times_variance_requires_four_times_continuous_n(self):
        n = d.difference_precision_n(4., 5., .2, family_size=2, halfwidth_fraction=.5)
        n4 = d.difference_precision_n(4., 5., .2, family_size=2, halfwidth_fraction=.5, variance_factor=4)
        self.assertLessEqual(abs(n4 - 4*n), 3)
    def test_endpoint_allocation_satisfies_every_pair_requirement(self):
        pilots = {x: {'mean': v, 'variance': 4., 'pilot_n': 8} for x, v in [('a', 10.), ('b', 11.), ('c', 14.)]}
        result = d.design(pilots, ['a','b','c'], [('a','b'),('b','c')], k.planning_equal_n_difference)
        for pair in result['comparisons']:
            for side in ['left','right']: self.assertGreaterEqual(result['endpoint_allocation_plugin'][pair[side]], pair['equal_n_pair_plugin'])
        self.assertIsNone(result['adopted_n'])
    def test_unknown_variance_not_filled(self):
        p = {'a': {'mean': 10., 'variance': None, 'pilot_n': 8}, 'b': {'mean': 11., 'variance': 1., 'pilot_n': 8}}
        with self.assertRaises(ValueError): d.design(p, ['a','b'], [('a','b')], k.planning_equal_n_difference)
    def test_zero_observed_gap_not_silently_resolved(self):
        p = {x: {'mean': 10., 'variance': 1., 'pilot_n': 8} for x in ['a','b']}
        with self.assertRaises(ValueError): d.design(p, ['a','b'], [('a','b')], k.planning_equal_n_difference)
    def test_resource_counts_and_parallel_makespan(self):
        r = {'a': {'seconds_per_trajectory_proxy': 60., 'peak_worker_RSS_bytes': 2**30}}
        p = d.resource_projection({'a': 3}, r, 2)
        self.assertEqual((p['trajectories'],p['cost_wrapper_axes'],p['LPT_wall_minutes_proxy']), (3,6,2.))
    def test_invalid_and_unimplemented_dispatch_both_refused(self):
        with self.assertRaisesRegex(ValueError,'EXPLICIT'): d.refuse_unapproved({'status':'DRAFT'}, {'valid':False})
        with self.assertRaisesRegex(ValueError,'ADAPTER'): d.refuse_unapproved({'status':'SEALED_APPROVED'}, {'valid':True})
    def test_cli_defaults_to_scope(self):
        z = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/prepare_h46_confirmation_design.py')], capture_output=True, text=True)
        self.assertEqual(z.returncode, 0); self.assertIn('"native_execution_supported": false', z.stdout)
    def test_cli_execute_refused_without_reading_input(self):
        z = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/prepare_h46_confirmation_design.py'), '--execute', '--pilot', '/DOES_NOT_EXIST'], capture_output=True, text=True)
        self.assertEqual(z.returncode, 2); self.assertIn('APPROVAL_AND_NATIVE_ADAPTER', z.stderr)
    def test_both_families_and_no_adopted_n(self):
        cli = load('cli', 'scripts/prepare_h46_confirmation_design.py')
        pilots = {key: {'mean': 10.+i, 'variance': 4., 'pilot_n': 8} for i, key in enumerate(d.EXPANDED)}
        resources = {key: {'seconds_per_trajectory_proxy': 10., 'peak_worker_RSS_bytes': 2**30} for key in d.EXPANDED}
        out = cli.build({'pilots': pilots, 'resources': resources, 'worker_scenarios': [2,4], 'input_result_commit': 'SYNTHETIC'})
        self.assertEqual(len(out['cases']), 14); self.assertIsNone(out['adopted_n'])
        self.assertFalse(out['confirmation_started'])
        for case in out['cases']:
            for n in case['reporting_guard_allocation_plugin'].values(): self.assertGreaterEqual(n, 8)

if __name__ == '__main__': unittest.main()

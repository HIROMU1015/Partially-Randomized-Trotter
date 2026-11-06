"""Synthetic scalar/metadata tests only; no saved scientific values are loaded."""
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import statistics
import sys
import unittest

import analysis_kernels as A


class SyntheticPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ['O2_PREFLIGHT_INPUT_ROOT'])
        cls.specs = json.loads((Path(__file__).parent / 'SOURCE_SPECS.json').read_text())
        def path(name):
            v = cls.specs[name]
            return root / v['commit'] / v['path'], v['sha256']
        A.verified_bytes(*path('allocation'))
        cls.rte = A.standalone_module(*path('rte'), 'o2_preflight_fixed_rte')
        cls.plan = A.standalone_module(*path('plan'), 'o2_preflight_fixed_plan')
        cls.tau = A.selected_functions(*path('canonical_tau'),
                  ['require', 'canonical_tau'], {'math':math})['canonical_tau']
        cls.stats = A.selected_functions(*path('paired'),
                    ['require','close','rational','frac','moments','paired'],
                    {'math':math, 'Fraction':Fraction, 'statistics':statistics})
        cls.cost = A.selected_functions(*path('cost_for'), ['fraction','cost_for'],
                   {'math':math, 'Fraction':Fraction})['cost_for']
        def forbidden(*args, **kwargs):
            raise AssertionError('Scientific sampling is forbidden in preflight')
        for name in ['sample_rte_events','iter_sample_rte_events',
                     'sample_event_mean_operator','make_rte_config']:
            setattr(cls.rte, name, forbidden)

    def terms(self, weights=(1.,), counts=(4,), K=2, T=.25):
        return A.normalization_terms(weights, counts, T=T, q=2, K=K, lambda_R=.5,
              finite_distribution=self.rte.finite_rte_distribution,
              canonical_tau=type(self).tau)

    def test_fixed_formula_structure(self):
        for label, length in [('2nd',1),('4th',3),('4th(m5_best)',11)]:
            w = self.plan.weights(label)
            self.assertEqual(len(w), length)
            self.assertEqual(w, w[::-1])
            self.assertLess(abs(math.fsum(w)-1), 1e-12)
        self.assertLess(self.plan.weights('4th(m5_best)')[5],0)

    def test_source_integer_allocation(self):
        self.assertEqual(self.plan.allocation((1.,1.,1.),5),(2,2,1))
        self.assertEqual(sum(self.plan.allocation((1.,-1.,1.),7)),7)
        with self.assertRaises(ValueError):
            self.plan.allocation((1.,1.,1.),2)

    def test_k0_closed_form_synthetic(self):
        actual = self.terms(K=0)['exact']
        tau = .5*(.25/2/4)
        self.assertAlmostEqual(actual, 2*4*.5*math.log1p(tau*tau), places=13)

    def test_k0_and_k2_are_distinct(self):
        self.assertGreater(self.terms(K=2)['exact'],self.terms(K=0)['exact'])
        self.assertEqual(self.terms(K=2)['integer_leading'],
                         2*self.terms(K=0)['integer_leading'])

    def test_negative_time_preserves_normalization(self):
        self.assertEqual(self.terms(T=.25)['exact'],self.terms(T=-.25)['exact'])

    def test_even_cutoff_required(self):
        with self.assertRaises(ValueError):
            self.terms(K=1)

    def test_integer_continuous_distinction_synthetic(self):
        value = self.terms(weights=(1.,-1.,1.),counts=(3,2,2))
        self.assertGreater(value['integer_leading'],value['continuous'])
        self.assertEqual(value['events_per_trajectory'],14)

    def test_proportional_synthetic_leading_equality(self):
        value = self.terms(weights=(1.,-1.,1.),counts=(2,2,2))
        self.assertAlmostEqual(value['integer_leading'],value['continuous'])

    def test_paired_covariance_and_fixed_N(self):
        value = A.paired_metric_cost([(2,2),(4,4)],10,20,
                paired=self.stats['paired'],cost_for=type(self).cost,cohort='SYNTHETIC')
        self.assertEqual(value['G_exact'],'90')
        self.assertEqual(value['G_SE'],30)
        self.assertEqual(value['cohort'],'SYNTHETIC')

    def test_mean_only_SE_stays_unknown(self):
        value = A.mean_only_metric_cost('3/2','5/2',10,20)
        self.assertEqual(value['G_exact'],'65')
        self.assertIsNone(value['G_SE'])
        self.assertFalse(value['N_reoptimized'])

    def test_saved_zero_is_allowed_but_missing_is_rejected(self):
        self.assertEqual(A.mean_only_metric_cost('0','0',3,4)['G_exact'],'0')
        with self.assertRaises(ValueError):
            A.mean_only_metric_cost(None,'0',3,4)

    def test_N_must_be_integer_and_unchanged(self):
        with self.assertRaises(ValueError):
            A.mean_only_metric_cost('3','4',1.5,2)
        endpoint=dict(candidate=dict(method='PR'),N_cos=11,N_sin=7)
        before=json.dumps(endpoint,sort_keys=True)
        type(self).cost(endpoint,dict(unit_statistics=self.stats['paired']([(1,3),(3,1)])))
        self.assertEqual(before,json.dumps(endpoint,sort_keys=True))

    def test_no_hardware_package_import(self):
        self.assertNotIn('qiskit',sys.modules)
        self.assertNotIn('trotterlib',sys.modules)


if __name__ == '__main__':
    unittest.main()

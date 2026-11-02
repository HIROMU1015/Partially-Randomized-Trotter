"""Scalar artificial fixtures only; never imports Qiskit or scientific runners."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
PATH=ROOT/'artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13/analyze_saved_cost.py'
spec=importlib.util.spec_from_file_location('saved_cost_analysis',PATH)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


class ScalarAnalysisTests(unittest.TestCase):
    def test_max_partition(self):
        p=A.max_order_probabilities(2,.1,.01)
        self.assertAlmostEqual(p['0'],.7921)
        self.assertAlmostEqual(p['2'],.188)
        self.assertAlmostEqual(p['4'],.0199)
        self.assertAlmostEqual(math.fsum(p.values()),1.)

    def test_multinomial_partition(self):
        h=A.histogram_probabilities(2,.1,.01)
        self.assertAlmostEqual(math.fsum(h.values()),1.)
        self.assertAlmostEqual(h[(1,1)],.002)
        self.assertAlmostEqual(h[(2,0)],.01)
        p=A.max_order_probabilities(2,.1,.01)
        self.assertAlmostEqual(sum(v for (n2,n4),v in h.items() if n4),p['4'])

    def test_k2_has_no_order4_mass(self):
        h=A.histogram_probabilities(16,.001)
        self.assertTrue(all(n4==0 for n2,n4 in h))
        self.assertEqual(A.max_order_probabilities(16,.001)['4'],0.)

    def test_small_order4_survives_cancellation(self):
        p=A.max_order_probabilities(16,.001,1e-18)
        self.assertGreater(p['4'],0.)
        self.assertTrue(math.isclose(p['4'],16e-18,rel_tol=1e-14))

    def test_unobserved_mean_remains_null(self):
        m=A.mixture({'0':.9,'2':.1},{'0':[10,10]},baseline=20)
        self.assertIsNone(m['full_plugin_mean'])
        self.assertIsNone(m['strata']['2']['mean'])
        self.assertAlmostEqual(m['unknown_common_conditional_mean_break_even'],110)
        self.assertFalse(m['population_mean_certified'])
        self.assertIsNone(m['cost_tail_bound'])

    def test_reweighted_plugin_is_not_pooling_or_bound(self):
        m=A.mixture({'0':.9,'2':.1},{'0':[10,10],'2':[30]})
        self.assertAlmostEqual(m['full_plugin_mean'],12)
        self.assertIsNone(m['strata']['2']['standard_error'])
        self.assertFalse(m['population_mean_certified'])

    def test_large_integer_cost_mean(self):
        s=A.stats([10**14,10**14+4])
        self.assertEqual(s['mean'],10**14+2)
        self.assertEqual(s['standard_error'],2.)

    def test_invalid_probability_domain(self):
        for R,p2,p4 in [(0,.1,0),(1,-.1,0),(2,.9,.1),(True,.1,0)]:
            with self.assertRaises(ValueError):A.max_order_probabilities(R,p2,p4)

    def test_scalar_codec_rejects_array_decoding(self):
        node,scalar=A.scalar_codec({'nodes':{'a':{'type':'ndarray','shape':[2],'data':'unused'}},'roots':{}})
        with self.assertRaises(ValueError):scalar({'ref':'a'})
        self.assertEqual(scalar({'float_hex':'0x1.0p+0'}),1.)

    def test_no_scientific_import(self):
        self.assertFalse(any(k in sys.modules for k in ('numpy','qiskit','scipy','pyscf','cupy')))


if __name__=='__main__':unittest.main()

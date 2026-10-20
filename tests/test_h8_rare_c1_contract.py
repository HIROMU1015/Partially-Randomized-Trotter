"""Planning/contract tests only; imports no molecular provider or RTE law."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h8_rare_c1_contract as p


class PlanningTests(unittest.TestCase):
    def test_chi2_two_closed_form(self):
        for x in (.01, 1., 10., 80.):
            self.assertAlmostEqual(p.chi2_cdf(x, 2), -math.expm1(-x/2), places=13)

    def test_chi2_one_closed_form(self):
        for x in (.01, 1., 10., 80.):
            self.assertAlmostEqual(p.chi2_cdf(x, 1), math.erf(math.sqrt(x/2)), places=13)

    def test_chi2_four_closed_form(self):
        for x in (.01, 1., 10., 80.):
            self.assertAlmostEqual(p.chi2_cdf(x, 4), 1-math.exp(-x/2)*(1+x/2), places=13)

    def test_minimality_and_separate_paths(self):
        d = p.sample_design()
        self.assertEqual(d['n_per_condition'], 43)
        self.assertEqual(d['trajectories_total'], 86)
        self.assertEqual([r['n'] for r in d['rows']], list(range(2, 44)))
        self.assertTrue(all(not r['passed'] for r in d['rows'][:-1]))
        self.assertTrue(d['rows'][-1]['passed'])
        self.assertGreater(d['endpoints'][0]['CI_upper_over_sample_variance'], 1.5)
        self.assertLess(d['endpoints'][1]['CI_upper_over_sample_variance'], 1.5)

    def test_inverse_quantiles_with_other_cdf(self):
        for df in (1, 2, 41, 42):
            for prob in (.05, .95):
                q = p.quantile(prob, df)
                self.assertAlmostEqual(p.independent_cdf(q, df), prob, places=11)

    def test_budget_scope(self):
        self.assertEqual(p.budget(2, 43)['RTE_event'], 344)
        self.assertEqual(p.budget(16, 43)['RTE_event'], 2752)
        for q in (2, 16):
            b = p.budget(q, 43)
            self.assertEqual(b['wrapper_compile'], 86)
            self.assertEqual(b['control_probe'], 4)
            self.assertEqual(b['event_oracle'], 1)
        with self.assertRaisesRegex(ValueError, 'FIXED_PILOT_SCOPE'):
            p.budget(8, 43)

    def test_task_ids_are_unique_and_fixed_scope(self):
        tasks = p.task_plan('B2_p7_q16_2nd_r4_K4', 16, 43)
        self.assertEqual(len({t['id'] for t in tasks}), 43)
        self.assertEqual([t['replica'] for t in tasks], list(range(43)))
        self.assertEqual({t['stratum'] for t in tasks}, {'S2_SINGLE'})

    def fixture(self):
        c = dict(approved=False, runtime_authorization=False, C1_scientific_execution_authorized=False,
                 candidate='B2_p8_q2_2nd_r4_K4', q=2, strata=['S2_SINGLE'],
                 n_by_stratum=dict(S2_SINGLE=43, S2_MULTI=None, S4_PRESENT=None),
                 frozen_scientific_budget=p.budget(2,43), validation_task_ids=[p.task_plan('B2_p8_q2_2nd_r4_K4',2,43)[0]['id']],
                 production_authorized=False, cost_cache_enabled=False, valid_grant_issued=False,
                 numeric_model_approved=False, live_session_library_verified=False, fork_safety_verified=False,
                 original_binding=dict(rank=28), existing_S0_n=48)
        return c

    def test_contract_only_gate(self):
        self.assertTrue(p.check_contract(self.fixture(), 43))

    def test_reject_execution_flags(self):
        for key in ('approved','runtime_authorization','C1_scientific_execution_authorized'):
            c = self.fixture(); c[key] = True
            with self.assertRaisesRegex(ValueError,'NO_EXECUTION'):p.check_contract(c,43)

    def test_reject_scope_and_forged_gates(self):
        for key, value in [('strata',['S2_MULTI']),('cost_cache_enabled',True),
                           ('numeric_model_approved',True),('fork_safety_verified',True),
                           ('live_session_library_verified',True),('validation_task_ids',[])]:
            c=self.fixture();c[key]=value
            with self.assertRaises(ValueError):p.check_contract(c,43)

    def test_no_real_science_imports(self):
        self.assertFalse(any(x == 'numpy' or x.startswith(('qiskit','trotterlib.')) for x in sys.modules))


if __name__ == '__main__':
    unittest.main()

"""PR-Pauli 4th(m5_best), K=2: provider signals equal an explicit independent composition (artificial only)."""
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import verify_pr_pauli_m5_k2_composition as V  # noqa: E402


class M5K2Composition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = V.run()

    def test_specified_weights_and_allocation(self):
        w = V.spec_weights(); s = self.report['spec_checks']
        self.assertTrue(s['palindrome']); self.assertEqual(s['negative_pieces'], [5]); self.assertLess(abs(s['cubic_sum']), 1e-15)
        self.assertEqual(self.report['allocation_R_11_3000_mismatches'], [])
        for r in self.report['rounds']:
            self.assertTrue(r['weights_equal_spec'] and r['allocation_equals_reference'] and r['piece_events_equal'])
            self.assertEqual(r['formula_id'], V.SPEC_FORMULA_ID)
        self.assertEqual(V.allocation_ref(w, 11), [1] * 11)

    def test_signals_and_normalizer_agree_including_negative_pieces(self):
        rows = self.report['rounds']
        self.assertEqual(len(rows), 21)
        for r in rows:
            self.assertTrue(r['gate_pass'], r['m'])
            self.assertTrue(r['agree'], (r['case'], r['m'], r['differences']))
            self.assertEqual(len(r['negative_pieces']), 1)
            neg = r['negative_pieces'][0]
            self.assertLess(neg['signed_time'], 0); self.assertLess(neg['tau'], 0); self.assertGreater(neg['b'], 1)
        self.assertGreater(max(abs(r['negative_pieces'][0]['tau']) for r in rows), 1.5)
        self.assertGreater(max(r['R_outer'] for r in rows), 1000)
        self.assertGreater(max(r['q'] for r in rows), 8)
        for x in self.report['event_law']:
            self.assertLess(x['law_mean_times_b_minus_polynomial'], 1e-13)
            self.assertLess(abs(x['probability_sum_minus_one']), 1e-13)

    def test_deliberate_errors_are_detected(self):
        for r in self.report['rounds']:
            self.assertTrue(r['mutations_detected'], (r['case'], r['m'], r['mutation_differences']))
        names = {k for r in self.report['rounds'] for k in r['mutation_differences']}
        self.assertEqual(names, {'head_order_reversed', 'negative_piece_sign_dropped', 'allocation_shifted',
                                 'tail_before_head', 'normalizer_without_lambda'})

    def test_noncommuting_system(self):
        self.assertTrue(all(v > .5 for v in self.report['noncommuting'].values()))

    def test_separate_question_approximation_order(self):
        a = self.report['approximation_separate_question']
        self.assertLess(abs(a['4th(m5_best)']['observed_orders'][-1] - 4), .3)
        self.assertLess(abs(a['2nd']['observed_orders'][-1] - 2), .3)

    def test_normalizer_budget_and_non_dyadic_step(self):
        n = self.report['normalizer_budget']
        self.assertTrue(all('Finite-normalizer scope exhausted' in s for s in n['one_per_round']['statuses']))
        self.assertTrue(all(s == 'LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED' for s in n['one_per_piece']['statuses']))
        self.assertGreater(n['one_per_piece']['distinct_normalizers'], n['one_per_round']['budget'])
        self.assertEqual(self.report['non_dyadic_step_refusal']['outcome'], 'REFUSED: Foreign B whole-schedule identity')


if __name__ == '__main__':
    unittest.main(verbosity=2)

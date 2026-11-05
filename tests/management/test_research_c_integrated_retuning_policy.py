"""Invented already-accounted summaries; no H4 artifacts or science calls."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[2] / 'src/trottertracks/c_integrated_retuning_policy.py'
SPEC = importlib.util.spec_from_file_location('integrated_retuning_policy', PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def invented():
    return [dict(r=r, candidate_sha256='invented', split='selection', samples=32,
                 correctness='PASS', eligible=True, total_rz=value)
            for r, value in ((16, 120.), (32, 100.), (64, 130.))]


class SelectionTests(unittest.TestCase):
    def test_selection_uses_existing_total_and_preserves_input(self):
        rows = invented(); before = copy.deepcopy(rows)
        self.assertEqual(m.select_r(rows, 'invented')['selected_r'], 32)
        self.assertEqual(rows, before)

    def test_common_r_tie_and_distinct_final_policy(self):
        rows = invented()
        for x in rows: x['total_rz'] = 100.
        self.assertEqual(m.select_r(rows, 'invented')['selected_r'], 32)
        self.assertEqual(m.final_pairs('A', 'A', 32), (('A', 32),))
        self.assertEqual(m.final_pairs('A', 'A', 16), (('A', 32), ('A', 16)))
        self.assertEqual(m.final_pairs('A', 'B', 32), (('A', 32), ('B', 32)))

    def test_fair_policy_opportunities(self):
        for rows in (invented()[:2], invented() + invented()[:1]):
            with self.assertRaises(ValueError): m.select_r(rows, 'invented')

    def test_final_cohort_and_mixed_candidate_rejected(self):
        for field, value in (('split', 'final'), ('samples', 64), ('candidate_sha256', 'other')):
            rows = invented(); rows[0][field] = value
            with self.assertRaises(ValueError): m.select_r(rows, 'invented')

    def test_ineligible_is_not_free_and_cannot_win(self):
        rows = invented(); rows[1].update(eligible=False, total_rz=None)
        self.assertEqual(m.select_r(rows, 'invented')['selected_r'], 16)
        rows[1]['total_rz'] = 0.
        with self.assertRaises(ValueError): m.select_r(rows, 'invented')

    def test_correctness_failure_does_not_remove_a_policy(self):
        rows = invented(); rows[0]['correctness'] = 'FAIL'
        with self.assertRaises(ValueError): m.select_r(rows, 'invented')

    def test_no_qualified_policy_has_no_retuned_winner(self):
        rows = invented()
        for x in rows: x.update(eligible=False, total_rz=None)
        self.assertIsNone(m.select_r(rows, 'invented')['selected_r'])

    def test_invalid_totals_and_unregistered_final_policy(self):
        for value in (float('nan'), float('inf'), -1., 0., True):
            rows = invented(); rows[0]['total_rz'] = value
            with self.assertRaises(ValueError): m.select_r(rows, 'invented')
        for value in (8, 128, True):
            with self.assertRaises(ValueError): m.final_pairs('A', 'A', value)


if __name__ == '__main__':
    unittest.main()

"""Synthetic data only. Never loads molecular results or compiler modules."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    'hybrid_rte', Path(__file__).parents[1] / 'src/trotterlib/pauli_hybrid_rte_comparison.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
INVENTORY_SPEC = importlib.util.spec_from_file_location(
    'inventory', Path(__file__).parents[1] / 'scripts/inventory_pauli_hybrid_rte.py')
INVENTORY = importlib.util.module_from_spec(INVENTORY_SPEC)
INVENTORY_SPEC.loader.exec_module(INVENTORY)


def row(cid, group, G, cohort='exploration', status='CONDITIONAL_GUARD_ELIGIBLE'):
    return dict(molecule='synthetic', epsilon_sig='synthetic', candidate_id=cid, group=group,
                G_exact=G, cohort=cohort, eligibility=status)


def minimum(rows, group='Hybrid', cohort='exploration'):
    return MODULE.saved_class_minimum(rows, molecule='synthetic', epsilon='synthetic',
                                      group=group, cohort=cohort)


class SyntheticPreparationChecks(unittest.TestCase):
    def test_both_saved_CSV_null_spellings_stay_missing(self):
        self.assertIsNone(INVENTORY.saved_csv_value('null'))
        self.assertIsNone(INVENTORY.saved_csv_value(''))
        self.assertEqual(INVENTORY.saved_csv_value('0'), '0')
        x = dict(G_exact=None, eligibility='BIAS_INELIGIBLE')
        self.assertEqual(INVENTORY.availability(x), 'BIAS_INELIGIBLE')

    def test_cohorts_do_not_mix(self):
        rows = [row('explore', 'Hybrid', '2'), row('confirm', 'Hybrid', '1', 'confirmation')]
        self.assertEqual(minimum(rows)['selected']['candidate_id'], 'explore')
        self.assertEqual(minimum(rows, cohort='confirmation')['selected']['candidate_id'], 'confirm')

    def test_null_never_becomes_zero_or_infinity(self):
        rows = [row('missing', 'Hybrid', None), row('finite', 'Hybrid', '7/3')]
        result = minimum(rows)
        self.assertEqual(result['G_exact'], '7/3')
        self.assertEqual(result['unresolved_ids'], ['missing'])
        self.assertFalse(result['all_registered_parameters_resolved'])

    def test_exact_ties_and_ineligible_prices(self):
        rows = [row('b', 'Hybrid', '4/2'), row('a', 'Hybrid', '2'),
                row('bad', 'Hybrid', '1/100', status='BIAS_INELIGIBLE')]
        self.assertEqual(minimum(rows)['ties'], ['a', 'b'])
        self.assertEqual(minimum(rows)['selected']['candidate_id'], 'a')

    def test_ratio_orientation_and_missing_confirmation(self):
        rows = [row('h', 'Hybrid', '3'), row('r', 'RTE', '2')]
        self.assertEqual(MODULE.hybrid_over_rte(minimum(rows), minimum(rows, 'RTE')), '3/2')
        missing = minimum(rows, 'RTE', 'confirmation')
        self.assertEqual(missing['status'], 'CONFIRMATION_MISSING')
        self.assertIsNone(MODULE.hybrid_over_rte(minimum(rows), missing))

    def test_ratio_rejects_mixed_cohorts(self):
        rows = [row('h', 'Hybrid', '3'), row('r', 'RTE', '2', 'confirmation')]
        with self.assertRaises(ValueError):
            MODULE.hybrid_over_rte(minimum(rows), minimum(rows, 'RTE', 'confirmation'))

    def test_ineligible_confirmed_subset_does_not_disqualify_registered_family(self):
        rows = [row('explore', 'Hybrid', '2'),
                row('old_confirmed', 'Hybrid', None, 'confirmation', 'BIAS_INELIGIBLE')]
        self.assertEqual(minimum(rows, cohort='confirmation')['status'], 'CONFIRMATION_MISSING')


if __name__ == '__main__':
    unittest.main()

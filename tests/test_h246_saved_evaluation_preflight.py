"""Artificial admission checks; no real saved-value science."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('preflight', Path(__file__).parents[1] /
    'scripts/audit_h246_saved_evaluation_preflight.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class PolicyAdmissionTests(unittest.TestCase):
    def test_no_adoption_rejected(self):
        for value in (None, {}, {'adopted': False}):
            with self.assertRaisesRegex(audit.AdmissionError, 'NOT_ADOPTED'):
                audit.require_accounting_policy(value)

    def test_flag_without_fixed_rules_rejected(self):
        with self.assertRaisesRegex(audit.AdmissionError, 'INCOMPLETE'):
            audit.require_accounting_policy({'adopted': True})

    def test_legacy_adoption_cannot_move_to_new_series(self):
        policy = {k: 'artificial' for k in ('adoption_evidence', 'epsilon_grid',
            'axis_allocation', 'alpha_axis', 'shot_integer_SOURCE', 'bias_policy',
            'allowance_policy', 'normalization_policy', 'headroom_guard',
            'integer_overflow_policy')}
        policy.update(adopted=True, applicable_series='LEGACY_H2_R_PER_PIECE')
        with self.assertRaisesRegex(audit.AdmissionError, 'SERIES_MISMATCH'):
            audit.require_accounting_policy(policy)

    def test_invalid_cohort_and_incomplete_domain_rejected(self):
        task = dict(id='ARTIFICIAL', stage='H2_DESIGN_PILOT',
                    axes=['cosine', 'sine'], candidate=dict(system='H2',
                    K=4, r_policy='ABS_TIME_PER_OUTER', r_value=32))
        with self.assertRaisesRegex(audit.AdmissionError, 'EXPLORATION_ONLY'):
            audit.cohort_metadata([task], 'H2')
        task['stage'] = 'H2_PR_EXPLORATION'
        task['candidate']['id'] = 'ARTIFICIAL'
        with self.assertRaisesRegex(audit.AdmissionError, 'TWELVE_GROUPS'):
            audit.cohort_metadata([task], 'H2')

    def test_pairing_and_outer_law_are_required(self):
        task = dict(id='ARTIFICIAL', stage='H2_PR_EXPLORATION',
                    axes=['cosine', 'sine'], candidate=dict(system='H2',
                    K=4, r_policy='ABS_TIME_PER_OUTER', r_value=32))
        for field, value in [('axes', ['cosine']), ('r_policy', 'PER_PIECE')]:
            changed = copy.deepcopy(task)
            (changed if field == 'axes' else changed['candidate'])[field] = value
            with self.assertRaisesRegex(audit.AdmissionError, 'EXPLORATION_ONLY'):
                audit.cohort_metadata([changed], 'H2')


if __name__ == '__main__':
    unittest.main()

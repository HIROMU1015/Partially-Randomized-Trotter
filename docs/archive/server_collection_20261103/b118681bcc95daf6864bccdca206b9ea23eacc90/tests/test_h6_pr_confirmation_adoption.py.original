"""Reject misleading adoption metadata without any scientific computation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'h6_adoption', ROOT / 'scripts/check_h6_pr_confirmation_adoption.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class AdoptionMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle = ROOT / check.BUNDLE
        cls.adoption = check.load((bundle / 'adoption_record.json').read_bytes())
        cls.completion = check.load((bundle / 'completion_status.json').read_bytes())
        cls.saved = check.load(check.blob(check.RESULT, check.SAVED + '/result.json'))

    def rejects(self, reason, *, adoption=None, completion=None):
        with self.assertRaisesRegex(ValueError, reason):
            check.check_adoption(adoption or self.adoption, completion or self.completion, self.saved)

    def test_current_scoped_adoption(self):
        check.check_adoption(self.adoption, self.completion, self.saved)

    def test_reject_source_substitution(self):
        row = deepcopy(self.adoption)
        row['SOURCE'] = check.BASE
        self.rejects('FIXED_COMMIT_IDENTITY', adoption=row)

    def test_reject_extra_scope(self):
        row = deepcopy(self.completion)
        row['scopes'].append(deepcopy(row['scopes'][0]))
        self.rejects('EXACT_SCOPE_KEYS', completion=row)

    def test_reject_n48_copied_to_q8(self):
        row = deepcopy(self.completion)
        row['scopes'][1]['n'] = 48
        self.rejects('FIXED_CELL_AND_N', completion=row)

    def test_reject_reproduced_as_necessarily_cheaper(self):
        row = deepcopy(self.completion)
        row['scopes'][0]['PR_CHEAPER'] = True
        self.rejects('DIRECTION_SEPARATE_FROM_CHEAPER', completion=row)

    def test_reject_reversed_expected_direction(self):
        row = deepcopy(self.completion)
        row['scopes'][0]['expected_direction'] = 1
        self.rejects('DIRECTION_SEPARATE_FROM_CHEAPER', completion=row)

    def test_reject_union_cohort_or_pooling(self):
        for update in [{'cohort': 'OLD_CONFIRMATION_UNION_N32'}, {'old_samples_pooled': True}]:
            with self.subTest(update=update):
                row = deepcopy(self.completion)
                row['scopes'][0].update(update)
                self.rejects('NO_COHORT_POOLING', completion=row)

    def test_reject_h8_k4_support(self):
        row = deepcopy(self.completion)
        row['scopes'][1]['event_order_counts']['4'] = 0
        self.rejects('K2_SAVED_COUNTS', completion=row)

    def test_reject_new_ci_or_population_guarantee(self):
        for update in [{'confidence_interval': ['1', '2']}, {'population_conclusion': 'GUARANTEED'}]:
            with self.subTest(update=update):
                row = deepcopy(self.completion)
                row['scopes'][1].update(update)
                self.rejects('SAVED_PRIMARY_LITERAL', completion=row)

    def test_reject_se_transcription_change(self):
        row = deepcopy(self.completion)
        row['scopes'][1]['paired_SE'] = '0'
        self.rejects('SAVED_PRIMARY_LITERAL', completion=row)

    def test_reject_final_freeze_or_science_permission(self):
        for field in ['final_evidence_freeze', 'new_science_authorized', 'new_numerical_analysis', 'push']:
            with self.subTest(field=field):
                row = deepcopy(self.adoption)
                row[field] = True
                self.rejects('NO_SCOPE_EXPANSION', adoption=row)

    def test_reject_common_core_or_mechanism_promotion(self):
        for field in ['COMMON_CORE_READY', 'M_CORE_new_cohort', 'M_EXPL_new_cohort']:
            with self.subTest(field=field):
                row = deepcopy(self.completion)
                row[field] = 'COMPLETE'
                self.rejects('NO_CORE_OR_MECHANISM_PROMOTION', completion=row)

    def test_reject_tie_as_reproduced(self):
        row = deepcopy(self.adoption)
        row['tie'] = 'REPRODUCED'
        self.rejects('TIE_NOT_REPRODUCED', adoption=row)


if __name__ == '__main__':
    unittest.main()

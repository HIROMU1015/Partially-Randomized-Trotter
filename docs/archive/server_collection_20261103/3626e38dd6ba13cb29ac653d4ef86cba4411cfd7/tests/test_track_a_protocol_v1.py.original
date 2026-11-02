"""Artificial metadata mutations; no molecular data or scientific imports."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('protocol_v1', ROOT / 'scripts/check_track_a_protocol_v1.py')
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


class ProtocolV1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        files = (*k.DOCS, k.MATRIX, *(k.BUNDLE + '/' + p for p in (
            'adoption_record.json', 'evidence_catalog.json', 'requirements.json',
            'matrix_change_log.json', 'evidence_update_status.json')))
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self):
        return k.validate(self.root, verify_git=False, verify_links=False)

    def json_mutate(self, name, change):
        path = self.root / k.BUNDLE / name
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def matrix_mutate(self, change):
        path = self.root / k.MATRIX
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        fields = list(rows[0])
        change(rows)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    def test_current_formal_structure(self):
        self.assertEqual(self.check()['status'], 'PASS')

    def test_reject_execution_authorization(self):
        self.json_mutate('adoption_record.json', lambda d: d.update(new_science_authorized=True))
        with self.assertRaisesRegex(ValueError, 'NOT_EXECUTION'):
            self.check()

    def test_reject_push_authorization(self):
        self.json_mutate('adoption_record.json', lambda d: d.update(push_authorized=True))
        with self.assertRaisesRegex(ValueError, 'NOT_EXECUTION'):
            self.check()

    def test_reject_automatic_migration(self):
        self.matrix_mutate(lambda rows: rows[0].update(automatic_application_to_existing_contracts='TRUE'))
        with self.assertRaisesRegex(ValueError, 'POLICY_NOT_EXECUTION'):
            self.check()

    def test_reject_unconditional_conditional_decision(self):
        self.json_mutate('adoption_record.json', lambda d: next(
            x for x in d['decisions'] if x['id'] == 'AD08').update(status='ADOPTED_BASIC_POLICY'))
        with self.assertRaisesRegex(ValueError, 'ADOPTION_PARTITION'):
            self.check()

    def test_reject_m5_adoption(self):
        self.json_mutate('adoption_record.json', lambda d: next(
            x for x in d['decisions'] if x['id'] == 'AD03').update(status='ADOPTED_BASIC_POLICY'))
        with self.assertRaisesRegex(ValueError, 'ADOPTION_PARTITION'):
            self.check()

    def test_reject_missing_row(self):
        self.matrix_mutate(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, '88_UNIQUE'):
            self.check()

    def test_reject_duplicate_row(self):
        self.matrix_mutate(lambda rows: rows.append(rows[0].copy()))
        with self.assertRaisesRegex(ValueError, '88_UNIQUE'):
            self.check()

    def test_reject_phase0_as_confirmation(self):
        self.matrix_mutate(lambda rows: next(r for r in rows if r['series_id'] == 'H6_TOL_ONLY_DF19'
                          and r['requirement_id'] == 'P_FIXED').update(status='COMPLETE'))
        with self.assertRaisesRegex(ValueError, 'STATUS_UNCHANGED'):
            self.check()

    def test_reject_premature_final_n(self):
        self.json_mutate('evidence_update_status.json', lambda d: d['H6'].update(final_n=48))
        with self.assertRaisesRegex(ValueError, 'SEAL_STILL_NULL'):
            self.check()

    def test_reject_recovery_as_explanation(self):
        self.json_mutate('evidence_update_status.json', lambda d: d['A82'].update(numeric_explanation_validated=True))
        with self.assertRaisesRegex(ValueError, 'NO_NUMERIC_COMPLETION'):
            self.check()

    def test_reject_64_complete_explanations(self):
        self.json_mutate('evidence_update_status.json', lambda d: d['A82'].update(historical_complete_sides=64))
        with self.assertRaisesRegex(ValueError, 'NO_NUMERIC_COMPLETION'):
            self.check()

    def test_reject_model_as_direct(self):
        self.matrix_mutate(lambda rows: next(r for r in rows if r['series_id'] == 'H6_TOL_ONLY_DF19')
                           .update(cost_evidence='ALL_DIRECT'))
        with self.assertRaisesRegex(ValueError, 'MODEL_NOT_DIRECT'):
            self.check()

    def test_reject_legacy_as_tol_only(self):
        self.matrix_mutate(lambda rows: next(r for r in rows if r['series_id'] == 'H4_LEGACY_DF12'
                          and r['requirement_id'] == 'C_DF').update(DF_policy='TOL_ONLY_1E-8'))
        with self.assertRaisesRegex(ValueError, 'LEGACY_NOT_TOL_ONLY'):
            self.check()

    def test_reject_unknown_evidence(self):
        self.matrix_mutate(lambda rows: rows[0].update(evidence_ids='E_ABSENT'))
        with self.assertRaisesRegex(ValueError, 'EVIDENCE_REFERENCE'):
            self.check()

    def test_reject_wrong_commit_binding(self):
        self.matrix_mutate(lambda rows: rows[0].update(evidence_commits='0' * 40))
        with self.assertRaisesRegex(ValueError, 'COMMIT_BINDING'):
            self.check()

    def test_reject_signal_and_analysis_source_conflation(self):
        self.json_mutate('evidence_update_status.json', lambda d: d['A82'].update(
            D0_SOURCE=d['A82']['original_signal_SOURCE']))
        with self.assertRaisesRegex(ValueError, 'ROLE_IDENTITY'):
            self.check()

    def test_reject_union_as_independent_confirmation(self):
        self.matrix_mutate(lambda rows: next(r for r in rows if r['series_id'] == 'H8_TOL_ONLY_DF28'
                          and r['requirement_id'] == 'P_FIXED').update(cohort='OLD_UNION_N32'))
        with self.assertRaisesRegex(ValueError, 'FIXED_P_NOT_UNION'):
            self.check()


if __name__ == '__main__':
    unittest.main()

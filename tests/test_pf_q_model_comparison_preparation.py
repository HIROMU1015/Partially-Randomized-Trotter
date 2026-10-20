"""Metadata and synthetic publication checks; no scientific execution."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_pf_q_model_comparison as p


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((p.OUT / 'axis_correspondence.json').read_text())['rows']
        cls.predictions = json.loads((p.OUT / 'saved_prediction_records.json').read_text())['rows']
        cls.models = json.loads((p.OUT / 'model_registry.json').read_text())

    def test_missing_values_are_never_promoted_to_comparable(self):
        self.assertEqual(p.classify(False, True, False, [])[0], 'NOT_COMPARABLE')
        self.assertEqual(p.classify(True, False, False, [])[0], 'NOT_COMPARABLE')
        self.assertEqual(p.classify(True, True, True, [])[0], 'NOT_COMPARABLE')
        self.assertEqual(p.classify(True, True, False, ['missing gate'])[0], 'CONDITIONAL_COMPARISON')
        self.assertEqual(p.classify(True, True, False, [])[0], 'COMPARABLE')

    def test_canonical_digest_keeps_original_json_numeric_types(self):
        raw = b'{ "x": 0.8, "count": 4 }\n'
        expected = hashlib.sha256(b'{"count":4,"x":0.8}').hexdigest()
        detail = dict(source_sha256=hashlib.sha256(raw).hexdigest(), canonical_json_sha256=expected)
        self.assertEqual(p.canonical_json_digest(raw), expected)
        self.assertEqual(p.model_digest_kind(expected, detail), 'CANONICAL_JSON_SHA256')
        self.assertEqual(p.model_digest_kind(detail['source_sha256'], detail), 'FILE_SHA256')
        self.assertEqual(p.model_digest_kind('wrong', detail), 'MISMATCH')
        self.assertEqual(p.model_digest_kind('', detail), 'UNDECLARED')

    def test_original_saved_file_and_canonical_identities_remain_distinct(self):
        h8 = [m for m in self.models if m['system'] == 'H8_DF28']
        self.assertEqual(len(h8), 4)
        self.assertTrue(all(m['source_sha256'] != m['canonical_json_sha256'] for m in h8))
        self.assertEqual(Counter(r['model_digest_kind'] for r in self.predictions),
                         {'FILE_SHA256': 192, 'CANONICAL_JSON_SHA256': 258})
        self.assertTrue(all(r['original_literal_verified'] and not r['evaluated_this_task'] for r in self.predictions))

    def test_788_unique_axes_form_complete_pairs(self):
        self.assertEqual(len({r['task_id'] for r in self.rows}), 788)
        pairs = defaultdict(set)
        for r in self.rows:
            pairs[(r['system'], r['task_id'].rsplit(':', 1)[0])].add(r['axis'])
        self.assertEqual(len(pairs), 394)
        self.assertTrue(all(axes == {'cosine', 'sine'} for axes in pairs.values()))
        self.assertEqual(Counter(r['system'] for r in self.rows),
                         {'H2_DF3':72,'H4_DF10':122,'H6_DF19':214,'H8_DF28':314,'H4_LEGACY_DF12':66})

    def test_saved_prediction_absence_and_same_series_model_absence_are_separate(self):
        self.assertEqual(Counter(r['comparison_class'] for r in self.rows),
                         {'CONDITIONAL_COMPARISON':450,'NOT_COMPARABLE':338})
        missing = [r for r in self.rows if r['model_id'] and not r['saved_prediction_literals']]
        self.assertEqual(Counter(r['system'] for r in missing), {'H6_DF19':22,'H8_DF28':56})
        self.assertEqual(sum(not r['model_id'] for r in self.rows), 260)
        self.assertTrue(all(not r['saved_prediction_literals'] for r in self.rows if not r['model_id']))

    def test_legacy_limits_are_preserved_and_excluded_axes_remain_excluded(self):
        self.assertEqual(sum(r['original_global_final_gate_missing'] for r in self.rows), 572)
        flagged = [r for r in self.rows if r['requires_saved_classical_signal_reconciliation']]
        self.assertEqual(len(flagged), 20)
        self.assertTrue(all(r['system']=='H2_DF3' and r['PF']=='4th' and r['prefix']=='0' for r in flagged))
        excluded = list(csv.DictReader(io.StringIO((p.OUT/'excluded_48_axes.csv').read_text())))
        self.assertEqual(len(excluded), 48)
        self.assertFalse({r['task_id'] for r in self.rows} & {':'.join(p.key(r)) for r in excluded})

    def test_training_and_validation_history_is_explicit(self):
        points = json.loads((p.OUT/'model_used_points.json').read_text())['rows']
        self.assertEqual(Counter(r['role'] for r in points),
                         {'TRAIN_ANCHOR':28,'TRAIN':22,'DECLARED_VALIDATION':4,'HOLDOUT':2,'PRIOR_DIRECT_VALIDATION':18})
        self.assertFalse(any(r['point_in_788_axis_set'] for r in points))
        self.assertFalse(any(r['prior_model_point_roles'] for r in self.rows))

    def test_csv_json_and_original_direct_rz_literals_agree(self):
        csv_rows = list(csv.DictReader(io.StringIO((p.OUT/'axis_correspondence.csv').read_text())))
        old = json.loads((ROOT/p.P2/'results01/combined_observation_index.json').read_text())['rows']
        old = {r['task_id']:r for r in old}
        self.assertEqual(len(csv_rows), 788)
        for current, csv_row in zip(self.rows, csv_rows):
            self.assertEqual(current['task_id'], csv_row['task_id'])
            self.assertEqual(current['direct_RZ_literal'], csv_row['direct_RZ_literal'])
            self.assertEqual(current['direct_RZ_literal'], str(old[current['task_id']]['RZ']))
            self.assertEqual(len(current['direct_result_sha256']), 64)
            self.assertFalse(current['model_evaluated_this_task'] or current['residuals_computed'] or current['cost_reaccounted'])

    def test_h6_priority_candidate_is_covered_on_both_axes(self):
        selected = [r for r in self.rows if r['task_id'].startswith('H6_DF19:B0_p11_q4_2nd:')]
        self.assertEqual({r['axis'] for r in selected}, {'cosine','sine'})
        self.assertTrue(all(r['comparison_class']=='CONDITIONAL_COMPARISON' for r in selected))

    def mock_publication(self, first):
        names = [f'demo/{i}.json.gz' for i in range(80)]
        content = {n:gzip.compress(json.dumps(first if i==0 else {}).encode()) for i,n in enumerate(names)}
        scope = dict(cumulative_unapproved_paths=names,last_approved_publication_commit='approved')
        def git(*args):
            if args[0]=='diff':return '\0'.join(names).encode()+b'\0'
            if args[0]=='show':return content[args[1].split(':',1)[1]]
            raise AssertionError(args)
        with patch.object(p,'fixed',return_value=scope), patch.object(p.inventory,'git',side_effect=git):
            return p.audit_publication()

    def test_gzip_scanner_detects_secret_without_republishing_value(self):
        synthetic_token = 'ghp_' + 'X'*30
        result = self.mock_publication({'note':synthetic_token,'client_secret':'synthetic-sensitive-value'})
        self.assertEqual(result['status'], 'STOP_PUBLICATION_ISSUE')
        self.assertEqual({r['kind'] for r in result['findings']}, {'GITHUB_TOKEN','NONEMPTY_CREDENTIAL_FIELD'})
        self.assertNotIn(synthetic_token, json.dumps(result))
        self.assertNotIn('synthetic-sensitive-value', json.dumps(result))
        self.assertEqual(result['files_checked'], 80)

    def test_runtime_hashes_and_scientific_grants_are_typed_metadata(self):
        result = self.mock_publication({'runtime_authorization':{'approved_by_user':True},
                                       'runtime_files':{'/home/example/runtime': 'a'*64}})
        self.assertEqual(result['status'], 'PASS_SCOPED_PUBLICATION_REVIEW')
        self.assertEqual(result['findings'], [])
        self.assertEqual(result['files'][0]['metadata_counts']['home_path_hash_keys'], 1)

    def test_jsonl_locator_transcribes_without_evaluation(self):
        doc = [{'updated':{'axis_costs':{'cosine':{'RZ':'123/7'}}}}]
        self.assertEqual(p.locate(doc,'JSONL_RECORD:1/updated/axis_costs/cosine')['RZ'], '123/7')


if __name__ == '__main__':
    unittest.main()

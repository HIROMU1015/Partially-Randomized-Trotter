"""Metadata-only regression and deliberate-corruption checks; no science run."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('higher_pf_dataset', ROOT/'scripts/h46_higher_pf_dataset.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.sources = d.get_sources()
        cls.reuse = json.loads((ROOT/d.BUNDLE/'saved_reuse_metric_projection_4.json').read_bytes())
        cls.conditions = json.loads((ROOT/d.BUNDLE/'conditions.json').read_bytes())
        cls.data = d.assemble(cls.sources, cls.reuse, cls.conditions)

    def reject(self, change, message):
        sources, reuse, conditions = copy.deepcopy((self.sources, self.reuse, self.conditions))
        change(sources, reuse, conditions)
        with self.assertRaisesRegex(d.DatasetError, message):
            d.assemble(sources, reuse, conditions)

    def test_exact_saved_counts(self):
        self.assertEqual(self.data['counts'], d.COUNTS)

    def test_missing_candidate_rejected(self):
        self.reject(lambda s,r,c:s['domain']['rows'].pop(), 'CANDIDATE_SETS')

    def test_duplicate_candidate_rejected(self):
        self.reject(lambda s,r,c:s['domain']['rows'].append(s['domain']['rows'][0]), 'DUPLICATE_CANDIDATE')

    def test_duplicate_axis_rejected(self):
        self.reject(lambda s,r,c:s['active_cost'].append(s['active_cost'][0]), 'DUPLICATE_ACTIVE_COST')

    def test_duplicate_eligibility_rejected(self):
        self.reject(lambda s,r,c:s['eligibility'].append(s['eligibility'][0]), 'DUPLICATE_ELIGIBILITY')

    def test_cross_epsilon_join_rejected(self):
        self.reject(lambda s,r,c:s['eligibility'][0].update(epsilon='0.0004'), 'ENDPOINT_DOMAIN')

    def test_changed_saved_bias_rejected(self):
        self.reject(lambda s,r,c:s['eligibility'][0].update(bias='0'), 'ELIGIBILITY_VALUE_TRANSCRIPTION')

    def test_zero_shot_imputation_rejected(self):
        self.reject(lambda s,r,c:s['eligibility'][0].update(shots='0'), 'ELIGIBILITY_VALUE_TRANSCRIPTION')

    def test_changed_formula_rejected(self):
        self.reject(lambda s,r,c:s['domain']['rows'][0].update(formula='new_2'), 'CANDIDATE_IDENTITY')

    def test_wrong_source_rejected(self):
        self.reject(lambda s,r,c:s['signal'].update(SOURCE='0'*40), 'SIGNAL_SOURCE')

    def test_wrong_compiler_rejected(self):
        self.reject(lambda s,r,c:s['active_cost'][0].update(compiler_settings_hash='0'*64), 'COST_COMPILER')

    def test_wrong_preparation_rejected(self):
        self.reject(lambda s,r,c:s['active_cost'][0].update(preparation_hash='0'*64), 'COST_PREPARATION')

    def test_changed_RZ_rejected(self):
        self.reject(lambda s,r,c:s['new_cost'][0].update(rz_count='0'), 'RZ_TRANSCRIPTION')

    def test_changed_cost_receipt_rejected(self):
        self.reject(lambda s,r,c:s['new_cost'][0].update(cost_receipt_sha256='0'*64), 'NEW_COST_RECEIPT')

    def test_reuse_not_new_label_rejected(self):
        self.reject(lambda s,r,c:s['active_cost'][-1].update(measurement='DIRECT_MEASURED_NEW_RUN01'), 'REUSE_LABEL')

    def test_saved_receipt_not_verified_rejected(self):
        self.reject(lambda s,r,c:r['rows'][0].update(private_receipt_byte_verified=False), 'REUSE_RECEIPT_NOT_VERIFIED')

    def test_missing_saved_CX_rejected(self):
        self.reject(lambda s,r,c:r['rows'][0]['metrics'].pop('cx_count'), 'REUSE_METRIC_FIELDS')

    def test_saved_metrics_are_integers(self):
        self.reject(lambda s,r,c:r['rows'][0]['metrics'].update(cx_count='81568'), 'METRIC_TYPE')

    def test_old_STOP_not_rewritten(self):
        def change(s,r,c):
            row=next(x for x in s['readiness']['reuse_axes'] if x['eligible_in_current_five_eps'])
            row['old_STOP_preserved']=False
        self.reject(change, 'OLD_STOP_PRESERVED')

    def test_native_FAIL_rejected(self):
        self.reject(lambda s,r,c:s['native_gates'][0].update(PASS='False'), 'SAVED_NATIVE_PASS')

    def test_phase_alignment_rejected(self):
        self.reject(lambda s,r,c:s['native_gates'][0].update(phase_alignment='True'), 'SAVED_NATIVE_PASS')

    def test_exit_failure_rejected(self):
        self.reject(lambda s,r,c:s['completion'][0].update(native_exit=1), 'SAVED_COMPLETION')

    def test_retry_rejected(self):
        self.reject(lambda s,r,c:s['completion'][0]['completion'].update(retry=True), 'SAVED_STOP')

    def test_adoption_not_implicit(self):
        self.reject(lambda s,r,c:c.update(AD03='ADOPTED'), 'AD03')

    def test_shot_policy_not_implicit(self):
        self.reject(lambda s,r,c:c.update(formal_shot_policy_adopted=True), 'FORMAL_SHOTS')

    def test_legacy_input_cannot_replace_tol(self):
        self.reject(lambda s,r,c:c['systems']['H4']['binding'].update(rank=12), 'INPUT_BINDING')

    def test_measured_partition_by_system(self):
        rows=self.data['measured_gate_costs']
        self.assertEqual(len([r for r in rows if r['system']=='H4']),40)
        self.assertEqual(len([r for r in rows if r['system']=='H6']),56)
        self.assertEqual(len([r for r in rows if r['measurement']=='DIRECT_MEASURED_NEW_RUN01']),92)

    def test_all_PR_and_inactive_kept_with_nulls(self):
        for row in self.data['candidates']:
            self.assertIsNone(row['n_E'])
            self.assertIsNone(row['shots'])
            self.assertIsNone(row['total_RZ'])
        self.assertEqual(len([r for r in self.data['axis_provenance'] if r['cost_key'] is None]),80)

    def test_missing_compiled_hash_is_unknown(self):
        rows=[r for r in self.data['measured_gate_costs'] if r['system']=='H4' and r['measurement']=='DIRECT_MEASURED_SAVED_REUSE']
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['compiled_SHA'] is None for r in rows))

    def test_materialized_dataset_and_v2_preservation(self):
        result=d.verify()
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['new_science_calls'],0)

    def test_json_csv_null_encoding(self):
        self.assertEqual(d.csv_bytes([{'id':'x','missing':None}]),b'id,missing\nx,\n')
        self.assertIn(b'null',d.json_bytes({'missing':None}))


if __name__=='__main__':
    unittest.main()

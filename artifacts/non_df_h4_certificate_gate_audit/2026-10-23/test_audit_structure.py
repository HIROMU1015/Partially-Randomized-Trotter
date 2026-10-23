"""Meaningful contract/schema refusal checks on audit products; stdlib only."""
import ast
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def read(name):return json.loads((HERE/name).read_text())

class Structure(unittest.TestCase):
    def test_exact40_unique_axis_registry(self):
        rows=read('certificate_gate_matrix.json')['rows']
        actual={(r['case'],r['epsilon_sig'],r['axis']) for r in rows}
        expected={(c,e,a) for c in ('pauli_jw_PF','pauli_jw_PR','df_native_PF','df_native_PR')
                  for e in (.05,.01,.005,.001,.0005) for a in ('cosine','sine')}
        self.assertEqual(len(rows),40);self.assertEqual(actual,expected)

    def test_csv_json_statuses_are_identical(self):
        with (HERE/'certificate_gate_matrix.csv').open(newline='') as f:csvrows=list(csv.DictReader(f))
        rows=read('certificate_gate_matrix.json')['rows'];self.assertEqual(len(csvrows),len(rows))
        for a,b in zip(rows,csvrows):
            self.assertEqual(a['case'],b['case'])
            for key,value in a['gates'].items():self.assertEqual(value,b[key])

    def test_scopes_and_status_vocab_are_complete(self):
        record=read('certificate_gate_matrix.json');catalog=read('gate_catalog.json')
        keys={g['id'] for g in catalog['gates']}
        for row in record['rows']:
            self.assertEqual(set(row['gates']),keys)
            self.assertTrue(set(row['gates'].values())<=set(record['status_vocabulary']))
            self.assertFalse(row['formal_certificate_accepted'])
            self.assertIsNone(row['old_N']);self.assertIsNone(row['old_G'])

    def test_flags_never_imply_acceptance_or_compiler_proof(self):
        for row in read('certificate_gate_matrix.json')['rows']:
            self.assertEqual(row['status'],'TARGET_MET')
            self.assertEqual(row['gates']['compiler_transfer'],'MISSING_EVIDENCE')
            self.assertEqual(row['gates']['certificate_acceptance_implementation'],'MISSING_EVIDENCE')
            self.assertFalse(row['formal_certificate_accepted'])

    def test_DF_norm_and_binary_readout_are_separate(self):
        for row in read('certificate_gate_matrix.json')['rows']:
            self.assertEqual(row['gates']['measurement_binary_range'],'VERIFIED')
            if row['case'].startswith('df_native'):
                self.assertEqual(row['gates']['encoded_event_norm'],'MISSING_EVIDENCE')

    def test_IID_remains_explicit_not_new_RNG_proof_gate(self):
        record=read('certificate_gate_matrix.json')
        self.assertTrue(record['no_new_unconditional_RNG_independence_requirement'])
        for row in record['rows']:
            self.assertEqual(row['gates']['conditional_statistics'],'ASSUMPTION_DEPENDENT')
            if row['case'].endswith('_PR'):self.assertEqual(row['gates']['sampling_law'],'ASSUMPTION_DEPENDENT')

    def test_no_new_scientific_condition_violation_claim(self):
        self.assertEqual(read('gate_catalog.json')['FAILED_CONDITION_findings_in_real_H4'],[])
        self.assertFalse(read('static_flag_path.json')['actual_H4_range_violation_established'])

    def test_minimal_proposal_is_inactive(self):
        p=read('minimal_next_contract_DRAFT.json')
        self.assertEqual(p['kind'],'H4_CERTIFICATE_GATE_MINIMAL_DIAGNOSTIC_DRAFT_NOT_EXECUTABLE')
        for flag in ('execution_authorized','grant_valid','dispatch_enabled'):self.assertIs(p[flag],False)
        self.assertIsNone(p['grant_id']);self.assertIsNone(p['approval_file'])
        self.assertTrue(p['mandatory_STOP']);self.assertTrue(p['certificate_acceptance_and_N_G_separate'])

    def test_local_only_future_budget_preserves40_saved_axes(self):
        phase=next(x for x in read('minimal_next_contract_DRAFT.json')['phases'] if x['id']=='L')
        budget=phase['science_call_budget_max']
        self.assertEqual(budget['unique_forward_local_gate_enclosures'],148)
        self.assertEqual(budget['inverse_local_gate_enclosures'],148)
        for name in ('full_reference','full_law_mean','bias_reevaluation','dense_256_matrix_exponential','sampling','RNG_replay','compile','N_G_accounting'):
            self.assertEqual(budget[name],0)

    def test_git_and_source_references_remain_bound(self):
        for item in read('source_references.json').values():
            self.assertEqual(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest(),item['SHA256'])
            self.assertIn('/blob/'+item['SOURCE']+'/',item['url'])
            self.assertIn('#L'+str(item['line']),item['url'])

    def test_supplemental_auditor_has_no_scientific_imports(self):
        tree=ast.parse((HERE/'audit_saved_evidence.py').read_text());forbidden={'numpy','scipy','qiskit','flint','trotterlib'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                self.assertFalse(forbidden&{x.name.split('.')[0] for x in node.names})
            if isinstance(node,ast.ImportFrom):self.assertNotIn((node.module or '').split('.')[0],forbidden)

    def test_saved_join_scope_and_legacy_native_gap(self):
        saved=read('legacy_compiler_evidence.json');matrix=read('certificate_gate_matrix.json')
        self.assertEqual(saved['trajectories_verified'],514)
        self.assertEqual(saved['compiled_circuit_serializations_in_known_case_roots'],[])
        self.assertEqual(matrix['original_H4_native_completion'],'UNKNOWN_UNRECOVERED_RETAINED_SEPARATELY')
        self.assertEqual(matrix['new_C_native_completion'],'EXISTING_PASS_SEPARATE')

def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Structure)
    with (HERE/'structure_tests.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),scientific_calls=0)
    with (HERE/'structure_tests.json').open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(record));raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__=='__main__':main()

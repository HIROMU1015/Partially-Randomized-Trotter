"""Artificial metadata only; no molecular circuit, scientific import or execution."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h2_runtime_cost_reuse as c


def fixture():
    return dict(candidate=c.PR_CANDIDATES[0], task=c.PROBE_TASKS[0], axis='cosine',
        cohort='EXPLORATION', runtime_id=c.PY312, compiler_hash=c.COMPILER,
        compiler_source_sha256='1'*64, settings=deepcopy(c.SETTINGS),
        input_sha256='2'*64, wrapper_raw_sha256='3'*64, compiled_raw_sha256='4'*64,
        receipt_sha256='5'*64, RZ=12, unit='compiled_RZ_per_full_Hadamard_wrapper_axis',
        correctness=dict(scope='SINGLE_SAVED_INPUT_STATE_CT01', strictly_below=1e-7,
                         PASS=True, phase_alignment=False, receipt_sha256='6'*64))


class MetadataTests(unittest.TestCase):
    def test_saved_integer_copied_without_measurement(self):
        row=fixture(); self.assertEqual(c.join_saved_cost(row, deepcopy(row)), row)

    def test_no_authorization_from_complete_evidence(self):
        p=c.preflight()
        self.assertFalse(p['execution_ready']); self.assertFalse(p['science_authorized'])
        self.assertEqual(p['COMPARE'], 'BLOCKED'); self.assertEqual(set(p['actual_calls'].values()), {0})

    def test_unexpected_field_rejected(self):
        row=fixture(); row['all_states']=True
        with self.assertRaises(c.r.EvidenceError): c.join_saved_cost(row, fixture())

    def test_operator_scope_promotion_rejected_even_in_both_packets(self):
        row=fixture(); row['correctness']['scope']='ALL_STATE_OPERATOR'
        with self.assertRaises(c.r.EvidenceError): c.join_saved_cost(row, deepcopy(row))

    def test_threshold_relaxation_rejected(self):
        row=fixture(); row['correctness']['strictly_below']=1e-6
        with self.assertRaises(c.r.EvidenceError): c.join_saved_cost(row, deepcopy(row))

    def test_boolean_setting_not_integer(self):
        row=fixture(); row['settings']['optimization_level']=True
        with self.assertRaises(c.r.EvidenceError): c.join_saved_cost(row, deepcopy(row))

    def test_invalid_sha_rejected(self):
        row=fixture(); row['compiled_raw_sha256']='z'*64
        with self.assertRaises(c.r.EvidenceError): c.join_saved_cost(row, deepcopy(row))

    def test_guard_no_phase_fit(self):
        value=dict(policy=c.GUARD, unchanged_gate_stream=True, state_fit=False,
            raw_validation_threshold_changed=False, passes=[dict(policy=c.GUARD,
                unchanged_gate_stream=True,input_or_output_state_fit=False,postcompile_phase_correction=False)])
        c.guard_receipt(value)
        value['passes'][0]['postcompile_phase_correction']=True
        with self.assertRaises(c.r.EvidenceError):c.guard_receipt(value)

    def test_missing_guard_pass_rejected(self):
        with self.assertRaises(c.r.EvidenceError):
            c.guard_receipt(dict(policy=c.GUARD,unchanged_gate_stream=True,state_fit=False,
                raw_validation_threshold_changed=False,passes=[]))

    def test_fixture_budget_exact(self):
        caps=dict(compile=4,transpile=4,simulate=8,sampling=0)
        ledger=c.FixtureLedger(caps,fixture='ARTIFICIAL_ONLY')
        for name,n in caps.items():
            for _ in range(n):ledger.take(name)
        self.assertEqual(ledger.finish(),caps)

    def test_budget_overrun_preserves_stop(self):
        ledger=c.FixtureLedger(dict(compile=1),fixture='ARTIFICIAL_ONLY');ledger.take('compile')
        with self.assertRaises(c.r.EvidenceError):ledger.take('compile')
        self.assertEqual(ledger.used['compile'],1)
        with self.assertRaises(c.r.EvidenceError):ledger.take('compile')

    def test_sampling_forbidden(self):
        ledger=c.FixtureLedger(dict(sampling=0),fixture='ARTIFICIAL_ONLY')
        with self.assertRaises(c.r.EvidenceError):ledger.take('sampling')

    def test_incomplete_budget_rejected(self):
        ledger=c.FixtureLedger(dict(compile=4),fixture='ARTIFICIAL_ONLY')
        with self.assertRaises(c.r.EvidenceError):ledger.finish()

    def test_real_execution_fixture_rejected(self):
        with self.assertRaises(c.r.EvidenceError):c.FixtureLedger(dict(compile=4),fixture='H2_REAL')

    def test_existing_scientific_validator_denied(self):
        with self.assertRaises(c.r.EvidenceError):c.deny_scientific_validation('simulate_statevector')

    def test_four_axes_not_134(self):
        rows=[dict(task=t,axis=a) for t in c.PROBE_TASKS for a in ('cosine','sine')]
        old=dict(prospective_exact_calls=dict(compile=4,transpile=4,simulate=8,
            setup=0,prepare=0,sampling=0,lowering=0))
        d=c.probe_draft(rows,old)
        self.assertFalse(d['proof_of_134_axes']);self.assertFalse(d['science_authorized'])
        self.assertIsNone(d['grant']);self.assertFalse(d['execution_entry_implemented'])
        rows.append(dict(task='UNAPPROVED',axis='cosine'))
        # Unselected rows do not expand the four-axis contract.
        self.assertEqual(len(c.probe_draft(rows,old)['axes']),4)

    def test_probe_missing_axis_rejected(self):
        with self.assertRaises(c.r.EvidenceError):c.probe_draft([],dict(prospective_exact_calls={}))

    def test_probe_budget_change_rejected(self):
        rows=[dict(task=t,axis=a) for t in c.PROBE_TASKS for a in ('cosine','sine')]
        old=dict(prospective_exact_calls=dict(compile=134,transpile=134,simulate=268,
            setup=0,prepare=0,sampling=0,lowering=0))
        with self.assertRaises(c.r.EvidenceError):c.probe_draft(rows,old)

    def test_no_scientific_imports(self):
        self.assertFalse(any(k.split('.')[0] in ('numpy','qiskit','scipy','symengine',
                                               'trotterlib','trottertracks') for k in sys.modules))


def mutation_test(field):
    def test(self):
        expected=fixture();row=deepcopy(expected)
        if field=='RZ':row[field]=13
        elif field=='settings':row[field]['seed_transpiler']=18
        elif field=='correctness':row[field]['phase_alignment']=True
        else:row[field]='substituted'
        with self.assertRaises(c.r.EvidenceError):c.join_saved_cost(row,expected)
    return test


for field in fixture():
    setattr(MetadataTests,'test_reject_substitution_'+field,mutation_test(field))


if __name__=='__main__':unittest.main()

"""Synthetic receipt tampering/boundary tests, no molecular data."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_h46_m5_det_cost_results as a


class AuditTests(unittest.TestCase):
    def gate(self):
        return dict(check='compiled_vector_cosine', PASS=True, error=1e-8, strictly_below=1e-7, phase_alignment=False)

    def receipt(self):
        return dict(axis='cosine', status='DIRECT_MEASURED_CT01_SINGLE_SAVED_INPUT_STATE', compiler_settings_hash=a.c.design.COMPILER_HASH,
                    native_policy=a.c.design.NATIVE_POLICY, metrics={key: 1 for key in a.METRICS})

    def test_original_threshold_and_strict_boundary(self):
        a.validate_gate(self.gate())
        for value in (1e-7, float('nan'), float('inf'), -1., True):
            row=self.gate();row['error']=value
            with self.assertRaises(ValueError):a.validate_gate(row)

    def test_PASS_or_phase_adjustment_cannot_override_receipt(self):
        for field, value in [('PASS', False), ('phase_alignment', True), ('strictly_below', 1e-6)]:
            row=self.gate();row[field]=value
            with self.assertRaises(ValueError):a.validate_gate(row)

    def test_native_threshold_stays_1e_minus_9(self):
        row=self.gate();row.update(check='native_low', strictly_below=1e-9, error=1e-10);a.validate_gate(row)
        row['error']=1e-9
        with self.assertRaises(ValueError):a.validate_gate(row)

    def test_integer_metrics_not_bool_missing_or_negative(self):
        a.validate_receipt(self.receipt(), 'cosine')
        for value in (True, 0, -1, 1.5):
            row=self.receipt();row['metrics']['rz_count']=value
            with self.assertRaises(ValueError):a.validate_receipt(row, 'cosine')

    def test_mixed_axis_or_compiler_or_native_policy_rejected(self):
        for field, value in [('axis', 'sine'), ('compiler_settings_hash', 'other'), ('native_policy', 'other'), ('status', 'UNVALIDATED')]:
            row=self.receipt();row[field]=value
            with self.assertRaises(ValueError):a.validate_receipt(row, 'cosine')


if __name__ == '__main__':unittest.main()

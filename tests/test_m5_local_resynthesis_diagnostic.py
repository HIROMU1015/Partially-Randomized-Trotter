"""Author checks of the synthetic diagnostic; no molecular inputs."""
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import m5_local_resynthesis_diagnostic as diagnostic


class LocalResynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.direct = diagnostic.direct_probe()
        cls.euler = diagnostic.euler_probe()

    def test_all_source_mirror_cases_match_installed_kernel_bits(self):
        self.assertEqual(len(self.direct), len(diagnostic.direct_cases())*3)
        self.assertTrue(all(r['source_mirror_bit_identity'] for r in self.direct))

    def test_each_probe_is_labelled_by_simplification_mode(self):
        self.assertEqual({r['mode'] for r in self.direct}, set(diagnostic.MODES))
        self.assertTrue(all(math.isfinite(r['raw_frobenius']) for r in self.direct))

    def test_subthreshold_RZ_loss_is_not_fixed_by_scalar_sum(self):
        row = next(r for r in self.direct if r['case'] == 'rz_'+(2.**-40).hex() and r['mode'] == 'default')
        self.assertEqual(row['gates'], [])
        self.assertGreater(row['raw_frobenius'], 8e-13)
        self.assertEqual(row['scalar_addition_roundoff'], 0.)
        self.assertEqual(row['raw_frobenius'], row['exact_scalar_sum_only_raw_frobenius'])

    def test_disabling_simplification_resolves_selected_small_RZ_probe(self):
        row = next(r for r in self.direct if r['case'] == 'rz_'+(2.**-40).hex() and r['mode'] == 'no_simplify')
        self.assertLess(row['raw_frobenius'], 2e-15)
        self.assertGreater(len(row['gates']), 0)

    def test_wrap_can_lose_sub_ulp_angle_without_simplification(self):
        self.assertEqual(diagnostic.wrap(2.**-55, -1.), 0.)
        self.assertNotEqual(2.**-55, 0.)

    def test_exact_sum_uses_unmodified_signed_binary64_increments(self):
        trace = diagnostic.trace_zsxx((.6, .37, -.29, -.21), 'default')
        expected = Fraction.from_float(-.21)+sum((Fraction.from_float(float.fromhex(v)) for v in trace['phase_updates_hex']), Fraction())
        self.assertEqual(trace['exact'], expected)
        self.assertNotEqual(Fraction.from_float(trace['phase']), expected)

    def test_reference_precision_stability(self):
        result = diagnostic.reference_checks()
        self.assertEqual(result['checked_cases'], len(diagnostic.direct_cases()))
        self.assertLess(result['max_element_difference'], 1e-75)

    def test_Euler_extraction_kept_separate_from_simplification(self):
        self.assertEqual(len(self.euler), len(diagnostic.direct_cases()))
        self.assertTrue(all(set(r['modes']) == set(diagnostic.MODES) for r in self.euler))
        self.assertLess(max(r['f64_vs_high_precision_Euler_raw'] for r in self.euler), 4e-15)
        row = next(r for r in self.euler if r['case'] == 'rz_'+(2.**-40).hex())
        # Separation of mechanisms, not a new molecular acceptance threshold.
        self.assertLess(row['f64_Euler_vs_same_input_raw'], row['modes']['default']['raw_vs_same_input']*1e-6)
        self.assertGreater(row['modes']['default']['raw_vs_same_input'], 8e-13)

    def test_coherent_actual_pass_loss_survives_exact_outer_batch(self):
        rows = diagnostic.coherent_probe()
        self.assertEqual([r['independent_runs'] for r in rows], [1,16,256,4096])
        self.assertTrue(all(r['output_nonbarrier_gates'] == 0 for r in rows))
        self.assertTrue(all(r['phase_increment_count'] == r['independent_runs'] for r in rows))
        self.assertGreater(rows[-1]['raw_basis1'], 3e-9)
        self.assertTrue(all(not r['H6_causal_proof'] for r in rows))

    def test_unknown_gate_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'UNREGISTERED_SYNTHETIC_GATE'):
            diagnostic.gate(diagnostic.context(), 'unrecognized', [])

    def test_default_CLI_does_no_calculation(self):
        script = Path(diagnostic.__file__)
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
        self.assertIn('ARTIFICIAL_SCOPE_ONLY', result.stdout)

    def test_existing_output_refused_before_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'kept.json'
            output.write_text('preserved')
            result = subprocess.run([sys.executable, diagnostic.__file__, '--run-artificial', '--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('NEW_EXCLUSIVE_OUTPUT_REQUIRED', result.stderr)
            self.assertEqual(output.read_text(), 'preserved')


if __name__ == '__main__':
    unittest.main()

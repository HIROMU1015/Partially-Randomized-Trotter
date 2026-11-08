"""Synthetic arithmetic/bookkeeping only; no saved scientific inputs loaded."""
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/diagnose_pr_reference_post_v0_precision.py"
spec = importlib.util.spec_from_file_location("v1_prepared", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PreparedDiagnosticTests(unittest.TestCase):
    def test_ties_even_sign_and_carry(self):
        r = m.rounder(2)
        self.assertEqual([r(x) for x in (0.125, 0.375, -0.375, 1.875, -0.0)], [0, 2, -2, 8, 0])
        for x in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                r(x)

    def test_known_chunks_and_unknown_guard(self):
        row = m.components(0.375, 3, {-2: F(1, 100)}, F(3, 2))
        self.assertEqual(row["selected_indices"], (-3, -2))
        self.assertEqual(row["Q"], 0)
        self.assertEqual(row["G_known"], F(1, 100))
        self.assertEqual(row["missing_indices"], (-3,))
        self.assertIsNone(row["logical_upper"])
        self.assertEqual(m.local_probe(row, F(1, 50))["relation"],
                         "POSITIVE_RESIDUAL_FOR_MISSING_GUARDS_ONLY")
        self.assertEqual(m.local_probe(row, F(1, 100))["relation"],
                         "KNOWN_ADDITIVE_CERTIFICATE_EXHAUSTS_LOCAL_ALLOCATION")

    def test_complete_logical_bound_and_quantization(self):
        row = m.components(0.375, 1, {-1: F(1, 100)}, F(3, 2))
        self.assertEqual(row["delta"], F(1, 8))
        self.assertEqual(row["Q"], F(3, 32))
        self.assertEqual(row["logical_upper"], F(3, 32) + F(1, 100))

    def test_finite_selection_rule_uses_input_exponent(self):
        self.assertEqual(m.new_precisions([float(2**-38).hex()]), (37, 38))
        self.assertEqual(m.new_precisions([(0.125).hex()]), ())
        with self.assertRaises(ValueError):
            m.new_precisions([])

    def test_joint_probe_keeps_unqualified_remainder(self):
        row = m.components(0.375, 3, {-2: F(1, 100)}, F(3, 2))
        condition = dict(joint_synthesis_error_retaining_saved_N_inclusive_upper=0.5,
                         axis={"X": dict(parts=dict(rotation=2, frames=3, scalar_and_wrapper=1))})
        probe = m.central_probe([row, row], condition, "X")
        self.assertIsNone(probe["complete_central_upper"])
        self.assertEqual(probe["central_known_certificate_load"], F(1, 50))
        self.assertEqual(probe["residual_for_missing_other_emitter_shell"], F(12, 25))
        self.assertEqual(probe["remaining_role_count_bounds"], dict(frames=3, scalar_and_wrapper=1))
        self.assertEqual(probe["remaining_errors"], "UNQUALIFIED_NOT_ZERO")

    def test_role_key_coverage_never_zero_fills(self):
        self.assertIsNone(m.role_envelope([], {})["logical_upper"])
        self.assertIsNone(m.role_envelope(["a", "b"], {"a": F(1, 10)})["logical_upper"])
        self.assertEqual(m.role_envelope(["a", "b"], {"a": F(1, 10), "b": F(1, 5)})[
            "logical_upper"], F(1, 5))

    def test_noncentral_count_is_positive_not_signed_price_correction(self):
        env = [dict(arm="toy", count_group="D", logical_upper=F(1, 10)),
               dict(arm="toy", count_group="frames", logical_upper=None)]
        result = m.noncentral_load(env, "toy", dict(D=3, frames=2))
        self.assertEqual(result["known_noncentral_certificate_load"], F(3, 10))
        self.assertEqual(result["missing_role_envelopes"], ["frames"])
        self.assertIsNone(result["full_joint_error_upper"])
        with self.assertRaises(ValueError):
            m.noncentral_load(env, "toy", dict(D=-3, frames=2))


if __name__ == "__main__":
    unittest.main()

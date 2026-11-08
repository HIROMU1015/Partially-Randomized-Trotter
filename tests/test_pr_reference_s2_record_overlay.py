"""Artificial metadata tests only; no saved guard/matrix or scientific run."""
from copy import deepcopy
import hashlib
import unittest

from trottertracks.pr_reference_s2_record_overlay import SingleSavedRecordOverlay


def fixture():
    literal = (0.125).hex()
    spec = literal + ":10^-13:pygridsynth2.0.0:strict:seed17"
    word = "WHT"
    return dict(theta_hex=literal, spec=spec,
        key=hashlib.sha256(spec.encode()).hexdigest(), precision_exponent=-13,
        status="PASS", guard_method="exact cyclotomic word matrix + outward interval strict Frobenius",
        strict_error_upper="artificial label, not a certified approximation",
        sequence=word, sequence_sha256=hashlib.sha256(word.encode()).hexdigest(),
        T=1, Tdagger=0, global_W=1)


class OverlayTests(unittest.TestCase):
    def test_same_positive_word_and_complete_negative_adjoint(self):
        p = SingleSavedRecordOverlay(fixture())
        a, b = p.provide(0.125), p.provide(-0.125)
        self.assertEqual(a.positive_sequence, b.positive_sequence)
        self.assertEqual(b.signed_sequence(), "tHWWWWWWW")
        self.assertEqual(a.backend_spec, b.backend_spec)

    def test_unknown_and_nonfinite_literals_rejected(self):
        p = SingleSavedRecordOverlay(fixture())
        for x in (0.25, 0.0, float("nan"), float("inf"), 1):
            with self.assertRaises(ValueError): p.provide(x)

    def test_provenance_and_W_corruption_rejected(self):
        for key, value in (("sequence", "WH"), ("spec", "fake -11"),
                           ("global_W", 0), ("status", "FAIL")):
            r = deepcopy(fixture()); r[key] = value
            with self.assertRaises(ValueError): SingleSavedRecordOverlay(r)

    def test_original_exponent_and_saved_guard_retained(self):
        r = fixture(); s = SingleSavedRecordOverlay(r).provide(0.125).summary()
        self.assertEqual(s["backend_precision_exponent"], -13)
        self.assertEqual(s["transport_exponent"], -11)
        self.assertFalse(s["transport_is_backend_spec"])
        self.assertEqual(s["saved_strict_guard"], r["strict_error_upper"])
        self.assertFalse(s["combined_strict_guard_newly_evaluated"])

    def test_original_record_mutation_cannot_change_selection(self):
        r = fixture(); p = SingleSavedRecordOverlay(r); r["sequence"] = ""
        self.assertEqual(p.provide(0.125).positive_sequence, "WHT")


if __name__ == "__main__": unittest.main()

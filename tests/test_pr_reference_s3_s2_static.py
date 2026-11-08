"""Artificial formal-token tests, no saved guard, circuit or matrix calls."""
import importlib.util
from pathlib import Path
import unittest

p = Path(__file__).resolve().parents[1] / "scripts/audit_pr_reference_s3_s2_static.py"
spec = importlib.util.spec_from_file_location("s3_static", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FormalTokenTests(unittest.TestCase):
    def test_same_complete_word_inverse_cancels(self):
        self.assertEqual(m.reduce_tokens([("word", 0, "including-W", 1), ("word", 0, "including-W", -1)]), [])

    def test_different_word_W_or_wire_does_not_cancel(self):
        self.assertTrue(m.reduce_tokens([("word", 0, "W0", 1), ("word", 0, "W1", -1)]))
        self.assertTrue(m.reduce_tokens([("word", 0, "same", 1), ("word", 1, "same", -1)]))

    def test_native_Clifford_identities_keep_control_wire_order(self):
        self.assertEqual(m.reduce_tokens([("sx", 0, 1), ("sx", 0, 2), ("sx", 0, 1)]), [])
        self.assertEqual(m.reduce_tokens([("cx", 0, 1), ("cx", 0, 1)]), [])
        self.assertTrue(m.reduce_tokens([("cx", 0, 1), ("cx", 1, 0)]))

    def test_word_level_pair_success_and_missing_never_identity(self):
        words = {(0.125).hex(): dict(sequence_sha256="exact_W_word")}
        result = m.pair_syntax([("rz", [0], [0.125])], [("rz", [0], [-0.125])], words)
        self.assertEqual(result["status"], "INSTRUCTION_PRODUCT_ADJOINT_PROVED")
        self.assertEqual(result["native_global_phase_relation"], "UNQUALIFIED_NOT_ZERO")
        self.assertEqual(m.pair_syntax([("rz", [0], [0.125])], [], {})["status"], "CURRENT_WORD_METADATA_MISSING")

    def test_noncancelled_product_is_not_refutation(self):
        words = {(0.125).hex(): dict(sequence_sha256="exact_W_word")}
        result = m.pair_syntax([("rz", [0], [0.125])], [("sx", [0], [])], words)
        self.assertEqual(result["status"], "NOT_PROVED_BY_LOCAL_REWRITES")
        self.assertTrue(result["nonidentity_remainder_is_not_operator_inequality_proof"])


if __name__ == "__main__":
    unittest.main()

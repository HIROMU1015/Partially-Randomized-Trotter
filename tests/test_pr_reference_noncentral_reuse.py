"""Artificial metadata and accounting; no saved error/T arithmetic."""
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys
import unittest

p = Path(__file__).resolve().parents[1] / "scripts/audit_pr_reference_noncentral_reuse.py"
spec = importlib.util.spec_from_file_location("noncentral_audit", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(p.parent))
import diagnose_pr_reference_noncentral_saved as prepared


class MetadataAndTemplateTests(unittest.TestCase):
    def test_same_word_different_record_is_not_alternative(self):
        row = dict(theta_hex="literal", sequence_sha256="same", status="PASS",
                   strict_spec=True, guard_method_documented=True)
        out = m.index_matches({"literal": []}, [row, dict(row, key="other-spec")],
                              {"literal": dict(sequence_sha256="same")})
        self.assertFalse(out[0]["distinct_documented_alternative_word"])

    def test_unqualified_or_different_literal_does_not_count(self):
        row = dict(theta_hex="other-literal", sequence_sha256="new", status="PASS",
                   strict_spec=True, guard_method_documented=True)
        old = {"literal": dict(sequence_sha256="old")}
        self.assertFalse(m.index_matches({"literal": []}, [row], old)[0]["distinct_documented_alternative_word"])
        row.update(theta_hex="literal", guard_method_documented=False)
        self.assertFalse(m.index_matches({"literal": []}, [row], old)[0]["distinct_documented_alternative_word"])
        row["guard_method_documented"] = True
        self.assertTrue(m.index_matches({"literal": []}, [row], old)[0]["distinct_documented_alternative_word"])

    def test_weighted_context_keeps_key_coupling(self):
        contexts = [dict(a=2, b=1), dict(a=0, b=2)]
        guards = dict(a=F(1, 100), b=F(1, 5))
        value = m.weighted_template_max(contexts, guards)
        self.assertEqual(value, F(2, 5))
        self.assertLess(value, 3 * max(guards.values()))

    def test_unknown_is_not_zero_and_negative_count_is_rejected(self):
        self.assertIsNone(m.weighted_template_max([dict(missing=1)], {}))
        self.assertIsNone(m.weighted_template_max([], {}))
        with self.assertRaises(ValueError):
            m.weighted_template_max([dict(a=-1)], dict(a=F(1)))

    def test_restricting_basis_alphabet_does_not_change_D_or_products(self):
        a = {k: F(1) for k in ("left_all", "right_all", "left_no0", "right_no0", "boundary_control", "cz", "z", "cswap", "block:0:full", *(f"block:{j}:half" for j in range(4)))}
        a.update({"basis:tail": F(1, 10), "basis_inverse:tail": F(1, 10),
                  "basis:outside": F(1, 5), "basis_inverse:outside": F(1, 5)})
        full = prepared.weighted_parts(2, 1, a, a, a, {k: ("tail", "outside") for k in ("Bprime", "reference")})
        tail = prepared.weighted_parts(2, 1, a, a, a, {k: ("tail",) for k in ("Bprime", "reference")})
        self.assertEqual(tail["Bprime"]["D"], F(5))
        self.assertEqual(tail["reference"]["D"], F(16))
        self.assertLess(tail["Bprime"]["frames"], full["Bprime"]["frames"])
        self.assertEqual(tail["Bprime"]["products"], full["Bprime"]["products"])
        missing = dict(a, cz=None)
        self.assertIsNone(prepared.weighted_parts(2, 1, missing, a, a, {k: ("tail",) for k in ("Bprime", "reference")}))

    def test_discrete_guard_T_frontier_is_not_one_selected_price(self):
        offers = [dict(id="low_T", guard=F(1, 5), T=10),
                  dict(id="low_error", guard=F(1, 10), T=20),
                  dict(id="dominated", guard=F(3, 10), T=30)]
        self.assertEqual(prepared.pareto_ids(offers), ["low_T", "low_error"])


if __name__ == "__main__":
    unittest.main()

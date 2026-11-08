"""Synthetic-only coverage, arithmetic preservation and S2 exclusion checks."""
import ast
import copy
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/diagnose_pr_reference_noncentral_s1_coverage_fix.py"
spec = importlib.util.spec_from_file_location("s1fix", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    b = {k: F(1) for k in ("left_all", "right_all", "left_no0", "right_no0", "boundary_control", "cz")}
    ref = {k: F(2) for k in ("block:0:full", "z", "cswap", *(f"block:{j}:half" for j in range(4)))}
    inv = {k: F(3) for k in ("block:0:full", *(f"block:{j}:half" for j in range(4)))}
    for a in (b, ref):
        a.update({"basis:tail": F(1, 10), "basis_inverse:tail": F(1, 5),
                  "basis:unused": F(7), "basis_inverse:unused": F(9)})
    return b, ref, inv


class CoverageFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, cls.arms, cls.evidence = m.load_legacy()

    def case(self, b, ref, inv, arm, ids=("tail",)):
        return m.scoped_parts(4, 2, b, ref, inv, {a: ids for a in ("Bprime", "reference")}, arm, self.arms)

    def test_unused_missing_and_variant_independence(self):
        b, ref, inv = fixture()
        b["basis:unused"] = ref["basis_inverse:unused"] = None
        for arm in ("Bprime", "reference"):
            self.assertIsNone(self.case(b, ref, inv, arm, ("tail", "unused"))["parts"])
            self.assertIsNotNone(self.case(b, ref, inv, arm)["parts"])

    def test_required_basis_missing_and_no_cross_arm_propagation(self):
        b, ref, inv = fixture()
        b["basis:tail"] = None
        self.assertIsNone(self.case(b, ref, inv, "Bprime")["parts"])
        self.assertIsNotNone(self.case(b, ref, inv, "reference")["parts"])
        b, ref, inv = fixture()
        ref["basis_inverse:tail"] = None
        self.assertIsNone(self.case(b, ref, inv, "reference")["parts"])
        self.assertIsNotNone(self.case(b, ref, inv, "Bprime")["parts"])

    def test_every_required_D_product_CSWAP_dependency(self):
        for arm in ("Bprime", "reference"):
            required = m.required_templates(arm, ("tail",))
            for mapping, names in required.items():
                for name in names:
                    with self.subTest(arm=arm, mapping=mapping, template=name):
                        b, ref, inv = fixture()
                        maps = dict(b=b, ref=ref, inv=inv)
                        maps[mapping][name] = None
                        result = self.case(b, ref, inv, arm)
                        self.assertIsNone(result["parts"])
                        self.assertTrue(result["missing_templates"])
                        other = "reference" if arm == "Bprime" else "Bprime"
                        self.assertIsNotNone(self.case(b, ref, inv, other)["parts"])

    def test_absent_template_is_not_zero_filled(self):
        b, ref, inv = fixture()
        del b["cz"]
        result = self.case(b, ref, inv, "Bprime")
        self.assertIsNone(result["parts"])
        self.assertEqual(result["missing_templates"][0]["reason"], "template_absent")

    def test_unchanged_arithmetic_against_original_on_complete_toy_inputs(self):
        original = next(n for n in ast.parse(m.fixed_blob(m.SOURCE_COMMIT, m.OLD_SOURCE)).body
                        if isinstance(n, ast.FunctionDef) and n.name == "weighted_parts")
        ns = dict(F=F)
        exec(compile(ast.Module(body=[copy.deepcopy(original)], type_ignores=[]), "old_synthetic_only", "exec"), ns)
        b, ref, inv = fixture()
        ids = {a: ("tail",) for a in ("Bprime", "reference")}
        old = ns["weighted_parts"](4, 2, b, ref, inv, ids)
        for arm in old:
            self.assertEqual(self.case(b, ref, inv, arm)["parts"], old[arm])

    def test_Fraction_hex_context_max_and_unknown_guard(self):
        key = (0.125).hex()
        batches = [dict(records=[dict(name="basis:tail", lowering=[("rz", [0], [0.125]), ("rz", [1], [-0.125])])]),
                   dict(records=[dict(name="basis:tail", lowering=[("rz", [0], [0.125])])])]
        self.assertEqual(self.ns["template_weights"](batches, {key: F(1, 7)})["basis:tail"], F(2, 7))
        self.assertIsNone(self.ns["template_weights"](batches, {})["basis:tail"])

    def test_S2_not_loaded_and_namespace_separated(self):
        self.assertFalse({"main", "pareto_ids", "index_matches"} & self.ns.keys())
        self.assertIn("s1_coverage_fix", str(m.OUT))
        self.assertNotIn("noncentral_saved_numeric", str(m.OUT))
        self.assertEqual(set(self.evidence["unchanged_arm_expression_AST_sha256"]), {"Bprime", "reference"})


if __name__ == "__main__":
    unittest.main()

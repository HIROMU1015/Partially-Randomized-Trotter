"""Artificial scalar tests only. No stored research inputs are read."""
import importlib.util
from pathlib import Path
from fractions import Fraction as F
import unittest

spec = importlib.util.spec_from_file_location(
    "cost_structure_preparation",
    Path(__file__).resolve().parents[1] / "scripts/diagnose_pr_reference_cost_structure.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CostAttributionTests(unittest.TestCase):
    def test_common_price_is_charged_for_extra_shots(self):
        d = module.split_cost(2, 3, ("7", "7"), ("7", "7"))
        self.assertEqual(d["change"], (F(7), F(7)))
        self.assertEqual(d["Bprime_shot_anchor"]["price"], (F(0), F(0)))
        self.assertEqual(d["Bprime_shot_anchor"]["shots"], d["change"])
        total = module.aggregate_splits([d, d])
        self.assertEqual(total["change"], (F(14), F(14)))
        self.assertEqual(total["reference_shot_anchor"]["shots"], total["change"])

    def test_component_extrema_are_not_independent(self):
        d = module.split_cost(2, 3, ("10", "12"), ("8", "11"))
        self.assertEqual(d["change"], (F(0), F(13)))
        # Same baseline endpoint enters both terms: independent extrema
        # are wider and must never replace the tighter direct enclosure.
        summed = module.add_intervals(d["reference_shot_anchor"].values())
        self.assertNotEqual(summed, d["change"])
        self.assertLess(summed[0], d["change"][0])

    def test_signed_fusion_and_negative_shot_difference(self):
        d = module.split_cost(3, 2, ("-4", "-4"), ("-3", "-2"))
        self.assertEqual(d["change"], (F(6), F(8)))
        self.assertEqual(d["reference_shot_anchor"]["shots"], (F(4), F(4)))

    def test_missing_role_or_reversed_bound_is_rejected(self):
        with self.assertRaises(ValueError):
            module.role_prices({"parts": {}}, False)
        with self.assertRaises(ValueError):
            module.split_cost(1, 1, (2, 1), (1, 2))
        with self.assertRaises(ValueError):
            module.split_cost(1.0, 1, (1, 1), (1, 1))


if __name__ == "__main__":
    unittest.main()

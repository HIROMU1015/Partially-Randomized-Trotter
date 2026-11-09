"""Artificial arithmetic/schema fixtures only. No saved scientific values read."""
import copy
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/diagnose_pr_reference_v6p.py"
spec = importlib.util.spec_from_file_location("v6p_prepared", PATH)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def toy_costs():
    """Toy prices deliberately have ranges and a signed fusion correction."""
    c = v.reused("cost")
    rows, prices = [], []
    for m in range(18):
        nb, nr = (626, 630) if m == 17 else (2, 2)
        identity = dict(m=m, q=131072 if m == 17 else 1, r=32 if m == 17 else 1, bits=88)
        p = {**identity, "shots_per_axis": nb, "price_per_axis": {}}
        for axis in "XY":
            role_b = {r: (F(0), F(0)) for r in c.ROLES}
            role_r = copy.deepcopy(role_b)
            role_b["D_unfused"] = F(20), F(20)
            role_r["D_unfused"] = F(10), F(10)
            role_r["D_boundary_fusion_difference"] = F(-2), F(-2)
            role_r["RTE_basis_frames"] = F(5), F(7)
            role_r["RTE_central_rotation"] = F(3), F(3)
            pb, pr = c.add_intervals(role_b.values()), c.add_intervals(role_r.values())
            rows.append({**identity, "axis": axis, "Bprime_shots": nb, "reference_shots": nr,
                         **c.split_cost(nb, nr, pb, pr),
                         "roles": {r: c.split_cost(nb, nr, role_b[r], role_r[r]) for r in c.ROLES}})
            p["price_per_axis"][axis] = dict(zip(
                ("Bprime_RZ_lower", "Bprime_RZ_upper", "reference_RZ_lower", "reference_RZ_upper"),
                (*pb, *pr)))
        prices.append(p)
    totals = c.aggregate_splits(rows)
    d0 = {"rows": rows, "totals": totals}
    price_data = {"rows": prices, "exact_Bprime_total_fractions":
                  dict(zip(("RZ_lower", "RZ_upper"), totals["Bprime_cost"])),
                  "exact_reference_RZ_lower_fraction": totals["reference_cost"][0]}
    return d0, price_data


def toy_saved():
    """Structural stand-ins; no real disk, phase, radius or price is used."""
    d0, prices = toy_costs()
    s3, s4, bs, original = [], [], [], []
    for price in prices["rows"]:
        identity = {k: price[k] for k in ("m", "q", "r", "bits")}
        ref = {"all_support_sufficient": True, "native_raw_disk_upper": .002,
               "phase_generation_extra_raw_disk_upper": .03}
        s3.append({**identity, "arms": {"reference": ref},
                   "old_source_shared_raw_disk_upper": .001,
                   "phase_arithmetic": {"exact_prefix_absolute_upper": 13.,
                                        "normalization_vs_controlled_source_upper": .03125,
                                        "phase_error_upper": .125}})
        s4.append({**identity, "arms": {"reference": {
            "alpha_axis": "1/720", "success_conditions_closed": True,
            "shots_per_axis": 630 if identity["m"] == 17 else 2}}})
        bs.append({**identity, "shots_per_axis": price["shots_per_axis"],
                   "raw_radius_lower": .625, "statistical_phase_lower": .125})
        original.append({**identity, "PF_phase_upper": .02, "raw_radius_lower": .75,
                         "raw_extra_disk_upper": .125,
                         "finite_bit_proposal": {"bits_per_uniform_draw": 88}})
    return {"stage3": {"rows": s3}, "stage4": {"rows": s4,
            "full_failure_probability": "36*(1/720)=0.05", "energy_accuracy_Ha": 1.5936e-4,
            "phase_budget": ".4 rad"}, "rounds": [{"name": "CN_S2_h0.02_ld3", "rows": original}],
            "bprime": {"rows": bs, "interval_bits": 192, "alpha_per_axis": "1/720",
            "failure_total": "36/720=.05", "energy_accuracy_Ha": 1.5936e-4, "phase_cap_rad": .4},
            "prices": prices, "d0": d0}


class V6PPreparationTests(unittest.TestCase):
    def test_import_has_no_git_data_or_quantum_calls(self):
        with patch("subprocess.check_output", side_effect=AssertionError("Unexpected input read")):
            fresh_spec = importlib.util.spec_from_file_location("v6p_import_probe", PATH)
            fresh = importlib.util.module_from_spec(fresh_spec)
            fresh_spec.loader.exec_module(fresh)
        self.assertNotIn("qiskit", fresh.__dict__)

    def test_interval_helpers_are_pinned_definitions(self):
        iv = v.reused("iv")
        x = iv.I('0.1')
        self.assertLessEqual(F.from_float(iv.lower(x)), F(1, 10))
        self.assertGreaterEqual(F.from_float(iv.upper(x)), F(1, 10))
        self.assertEqual(iv.mp.iv.prec, 192)

    def test_normalization_is_retained_and_candidate_is_outward(self):
        M, norm = 13.0, 0.03125
        result = v.candidate_bound(M, norm)
        exact = F.from_float(norm) + 2 * F(2) ** -53 * F.from_float(M)
        self.assertGreaterEqual(F.from_float(result["phase_upper"]), exact)
        self.assertEqual(result["normalization_upper"], norm)
        self.assertGreater(result["phase_upper"], norm)

    def test_raw_components_and_monotonic_shots(self):
        old = v.bound_and_shots((0.75, 0.125), .02, .001, .002, .03)
        new = v.bound_and_shots((0.75, 0.125), .02, .001, .002, .003)
        self.assertGreaterEqual(new["raw_radius_lower"], old["raw_radius_lower"])
        self.assertGreaterEqual(new["statistical_phase_lower"], old["statistical_phase_lower"])
        self.assertLessEqual(new["shots_per_axis"], old["shots_per_axis"])
        self.assertGreaterEqual(old["other_raw_disk_upper"], .003)

    def test_integer_shots_are_sufficient_for_upper_endpoint(self):
        result = v.shot_ceiling(.625, .125)
        n = result["shots_per_axis"]
        lo, hi = result["required_shots_interval"]
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(F(n), hi)
        self.assertLess(F(n - 1), hi)

    def test_invalid_disks_and_nonfinite_bounds_stop(self):
        for phase in (1.0, -1.0, float('nan'), float('inf')):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                v.bound_and_shots((.5, 0.), .02, .01, .01, phase)
        with self.assertRaises(ValueError):
            v.shot_ceiling(.5, 0.)

    def test_negative_shot_change_keeps_endpoint_correlation(self):
        d0, prices = toy_costs()
        result = v.recost(d0, prices, 600)
        # 34 unchanged cells at 2 shots, then 2 terminal cells at 600 shots.
        expected = F(34 * 2 * 16 + 2 * 600 * 16), F(34 * 2 * 18 + 2 * 600 * 18)
        self.assertEqual(result["conditional_reference_total"], expected)
        naive = v.reused("check").add((result["saved_reference_total"], result["delta_to_saved_reference"]))
        self.assertNotEqual(naive, expected)
        unchanged = [x for x in result["rows"] if x["m"] < 17]
        self.assertTrue(all(x["shots_per_axis"] == 2 for x in unchanged))

    def test_signed_fusion_and_no_change_cost_identity(self):
        d0, prices = toy_costs()
        new = v.recost(d0, prices, 600)
        for row in new["rows"]:
            if row["m"] == 17:
                self.assertEqual(row["roles"]["D_boundary_fusion_difference"], (F(-1200), F(-1200)))
        old = v.recost(d0, prices, 630)
        self.assertEqual(old["conditional_reference_total"], old["saved_reference_total"])
        self.assertEqual(old["delta_to_saved_reference"], (F(0), F(0)))

    def test_missing_role_coverage_and_saved_N_are_rejected(self):
        for defect in ("role", "cell", "N"):
            d0, prices = toy_costs()
            if defect == "role":
                del d0["rows"][-1]["roles"]["CSWAP"]
            elif defect == "cell":
                d0["rows"].pop()
            else:
                d0["rows"][-1]["reference_shots"] = 629
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                v.recost(d0, prices, 600)

    def test_campaign_schema_binding_uses_artificial_values_only(self):
        # Isolate schema/accounting from shot arithmetic with explicit stand-ins.
        with patch.object(v, "bound_and_shots", side_effect=[
                {"shots_per_axis": 630}, {"shots_per_axis": 600}]), \
                patch.object(v, "shot_ceiling", return_value={"shots_per_axis": 626}):
            result = v.campaign(toy_saved())
        self.assertFalse(result["new_phase_provider_implemented_or_qualified"])
        self.assertEqual(result["next_action"], "MANDATORY_STOP")
        self.assertEqual(result["conditional_reference"]["shots_per_axis"], 600)

    def test_old_N_mismatch_stops_before_candidate_and_recost(self):
        with patch.object(v, "bound_and_shots", return_value={"shots_per_axis": 629}), \
                patch.object(v, "shot_ceiling", return_value={"shots_per_axis": 626}), \
                patch.object(v, "candidate_bound", side_effect=AssertionError("Too early")), \
                patch.object(v, "recost", side_effect=AssertionError("Too early")):
            with self.assertRaisesRegex(ValueError, "Old N reproduction"):
                v.campaign(toy_saved())


if __name__ == "__main__":
    unittest.main()

"""Artificial/mock tests only. Never load real molecular signal/cost values."""
import copy
import ast
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from h46_higher_pf_adapter import kernel, load_rows, SIGNAL, NATIVE, GROUP_LABEL, PREFIX
from run_h46_higher_pf_comparison import authorize, compare_rows, claim_grant
from d0_frozen_shot_rule_v1 import u_aware_shots
k = kernel()


def fake_row(q=1, label="2nd", system="ARTIFICIAL_H4", base=10):
    identity = {"SOURCE": "synthetic-source", "input_sha256": "synthetic-input",
                "preparation_hash": "synthetic-preparation", "compiler_settings_hash": k.PROFILE,
                "series": k.SERIES, "control": "ordinary", "formula_id": k.FORMULAS[label],
                "system": system, "prefix": 3, "q": q, "stage": "EXPLORATION"}
    records = []
    for replica in range(3):
        for axis in ("cosine", "sine"):
            records.append({**identity, "replica_id": str(replica), "axis": axis,
                            "event_digest": "fake-event-"+str(replica), "seed_namespace": "synthetic-explore",
                            "forced_event": False, "cost_kind": "MEASURED",
                            "rz_count": base+replica+(5 if axis == "sine" else 0), "cx_count": 3+replica})
    return {"candidate_id": system+"_"+label+"_q"+str(q), "identity": identity,
            "total_signed_bias": {"real": 0., "imag": 0.}, "log_B": 0.,
            "signal_evidence": "ARTIFICIAL_ONLY", "cost_kind": "MEASURED", "records": records}


class CoefficientTests(unittest.TestCase):
    def test_frozen_allocation_function_on_artificial_signed_weights(self):
        # Execute only the original pure function AST, never scientific imports.
        tree = ast.parse((ROOT/"scripts/h46_new2_core.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "allocation")
        scope = {"math": math}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "frozen-allocation-only", "exec"), scope)
        counts = scope["allocation"]([1., -1., 1.], 32)
        self.assertEqual(counts, (11, 11, 10))  # largest-remainder ties use generation order
        self.assertEqual(sum(s["rte_steps"] for s in k.signed_schedule([1.,-1.,1.],8,.8,counts)), 256)

    def test_decimal_vs_binary_rational(self):
        dec = k.moment_audit(["0.1", "0.8", "0.1"], "DECIMAL_LITERAL")
        binary = k.moment_audit([.1, .8, .1], "BINARY64_RATIONAL")
        self.assertEqual(dec["first_exact"], "1")
        self.assertNotEqual(binary["first_exact"], "1")
        self.assertFalse(dec["algebraic_formula_adopted"])

    def test_palindrome_negative_and_rounding_defect(self):
        rounded = k.moment_audit(["0.4", "0.45", "-0.7", "0.45", "0.4"], "DECIMAL_LITERAL")
        self.assertTrue(rounded["palindrome"] and rounded["negative_time"])
        self.assertEqual(rounded["first_exact"], "1")
        self.assertFalse(rounded["rational_order_conditions_met"])

    def test_exact_algebraic_family_identity(self):
        a, b = Fraction(7, 19), Fraction(3, 13)
        polynomial = k.exact_family_polynomial(a)
        self.assertEqual(sum(v*b**i for i, v in enumerate(polynomial)), 2*a**3+2*b**3+(1-2*a-2*b)**3)

    def test_formula_identities_are_distinct(self):
        self.assertEqual(len(set(k.FORMULAS.values())), 3)
        self.assertNotIn("new2-projected-not-adopted", k.FORMULAS.values())

    def test_nonfinite_or_misrepresented_coefficients(self):
        for values, mode in (([float("nan")], "BINARY64_RATIONAL"), ([1.], "DECIMAL_LITERAL"), ([], "DECIMAL_LITERAL")):
            with self.assertRaises(ValueError): k.moment_audit(values, mode)

    def test_signed_q_schedule(self):
        schedule = k.signed_schedule([1., -1., 1.], 4, .8, [11, 10, 11])
        self.assertEqual([(s["outer"], s["piece"]) for s in schedule], [(q,p) for q in range(4) for p in range(3)])
        self.assertEqual(sum(s["rte_steps"] for s in schedule), 128)
        self.assertEqual(schedule[1]["signed_time"], -.2)
        self.assertAlmostEqual(sum(s["signed_time"] for s in schedule), .8)

    def test_total32_per_outer_not_per_run(self):
        with self.assertRaises(ValueError): k.signed_schedule([1.], 4, .8, [8])

    def test_mock_control_and_scalar_relative_phase(self):
        # Mock controlled U: phase belongs in ancilla-one branch, not global.
        t, scalar = .2, 1.7
        phase = complex(math.cos(-scalar*t), math.sin(-scalar*t))
        control_zero, control_one = (1., 0.), (0., phase)
        self.assertEqual(control_zero[0], 1.)
        self.assertNotEqual(control_one[1], 1.)
        self.assertAlmostEqual(abs(control_one[1]), 1.)
        # Reuse the unchanged, already tested Gaussian path via a static guard.
        native = (ROOT/"scripts/h46_new2_native.py").read_text()
        self.assertIn("XXPlusYYGate", native)
        self.assertIn("control_policy='ordinary'", native.replace('"', "'"))
        self.assertNotIn("UnitaryGate(", native)


class CostTests(unittest.TestCase):
    def test_shot_integer_math_and_axis_weights(self):
        row = fake_row()
        row["total_signed_bias"]["imag"] = .0001
        out = k.evaluate_row(row, u_aware_shots)
        n = {a: out["shots"]["axes"][a]["shots"] for a in ("real", "imag")}
        self.assertNotEqual(n["real"], n["imag"])
        expected = [n["real"]*(10+i)+n["imag"]*(15+i) for i in range(3)]
        self.assertEqual(out["cost"]["paired_values"]["RZ"], expected)
        self.assertEqual(Fraction(out["mean_exact"]), sum(expected)/3)

    def test_normalization_and_ceil(self):
        row = fake_row(); row["log_B"] = math.log(2)
        out = k.evaluate_row(row, u_aware_shots)
        h = .001/math.sqrt(2)-1e-9
        expected = math.ceil(2*math.exp(math.log(2)+1e-12)**2/h**2*math.log(2/.025))
        self.assertEqual(out["shots"]["axes"]["real"]["shots"], expected)

    def test_ineligible_is_not_missing_cost(self):
        row = fake_row(); row["total_signed_bias"]["real"] = .1
        out = k.evaluate_row(row, u_aware_shots)
        self.assertEqual(out["eligibility"], "INELIGIBLE")
        self.assertIsNone(out["cost"])

    def test_unknown_bias_is_undetermined(self):
        row = fake_row(); row["total_signed_bias"]["real"] = None
        self.assertEqual(k.evaluate_row(row, u_aware_shots)["eligibility"], "UNDETERMINED")

    def test_logdomain_shots_are_not_integers(self):
        row = fake_row(); row["log_B"] = 500
        out = k.evaluate_row(row, u_aware_shots)
        self.assertIsNone(out["cost"])
        self.assertIsNone(out["shots"]["N_total"])

    def test_missing_is_not_zero(self):
        row = fake_row(); row.update(cost_kind="MISSING", records=[])
        self.assertIsNone(k.evaluate_row(row, u_aware_shots)["mean_exact"])
        row["records"] = fake_row()["records"]
        with self.assertRaises(ValueError): k.evaluate_row(row, u_aware_shots)

    def test_model_does_not_get_measured_SE_or_winner(self):
        row = fake_row(); row.update(cost_kind="MODELED", records=[], model={"real": 3, "imag": 4, "evidence": "fake-model"})
        out = k.evaluate_row(row, u_aware_shots)
        self.assertGreater(out["model_only_cost"], 0)
        self.assertIsNone(out["mean_exact"])
        self.assertIsNone(out["model_SE"])

    def test_source_input_compiler_control_rejections(self):
        for key in ("SOURCE", "input_sha256", "preparation_hash", "compiler_settings_hash", "control", "series", "formula_id", "q", "stage"):
            row = fake_row(); row["records"][0][key] = "WRONG"
            with self.subTest(key=key), self.assertRaises(ValueError): k.evaluate_row(row, u_aware_shots)

    def test_paired_event_seed_axes_and_forced_rejections(self):
        for key, value in (("event_digest", "WRONG"), ("seed_namespace", "WRONG"), ("forced_event", True), ("rz_count", -1)):
            row = fake_row(); row["records"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): k.evaluate_row(row, u_aware_shots)
        for records in (fake_row()["records"][:-1], fake_row()["records"]+[fake_row()["records"][0]]):
            row = fake_row(); row["records"] = records
            with self.assertRaises(ValueError): k.evaluate_row(row, u_aware_shots)

    def test_nonfinite_inputs(self):
        for key in ("bias", "log", "cost"):
            row = fake_row()
            if key == "bias": row["total_signed_bias"]["real"] = float("nan")
            elif key == "log": row["log_B"] = float("inf")
            else: row["records"][0]["rz_count"] = float("nan")
            with self.subTest(key=key), self.assertRaises(ValueError): k.evaluate_row(row, u_aware_shots)

    def test_exact_tie_chooses_smaller_q(self):
        rows = [k.evaluate_row(fake_row(q=q), u_aware_shots) for q in k.Q_GRID]
        self.assertTrue(k.select_q(rows)["full_grid_winner"].endswith("q1"))

    def test_incomplete_grid_never_declares_full_winner(self):
        rows = [fake_row(q=q) for q in k.Q_GRID]
        rows[-1].update(cost_kind="MISSING", records=[])
        selected = k.select_q([k.evaluate_row(r, u_aware_shots) for r in rows])
        self.assertIsNone(selected["full_grid_winner"])
        self.assertIsNotNone(selected["available_subset_point_best"])
        with self.assertRaises(ValueError): k.select_q([k.evaluate_row(r, u_aware_shots) for r in rows[:-1]])

    def test_runner_end_to_end_artificial_24_groups(self):
        rows = [fake_row(q, label, system, base=10+q) for system in ("ARTIFICIAL_H4", "ARTIFICIAL_H6")
                for label in k.FORMULAS for q in k.Q_GRID]
        out = compare_rows(rows, independent_candidates=True)
        self.assertEqual(len(out["rows"]), 24)
        self.assertEqual(len(out["selections"]), 6)
        self.assertTrue(all(x["full_grid_winner"].endswith("q1") for x in out["selections"]))
        self.assertTrue(out["mandatory_STOP"])
        self.assertEqual(out["native_compile_calls"], 0)
        self.assertFalse(out["next_phase_started"])


class StatisticsTests(unittest.TestCase):
    def test_variance_SE_and_ratio(self):
        x, y = k.sample_summary([10, 20, 30]), k.sample_summary([20, 40, 60])
        self.assertEqual(x["variance"], 100.)
        self.assertAlmostEqual(x["SE"], math.sqrt(100/3))
        out = k.comparison(x, y, independent_candidates=True)
        self.assertEqual(out["ratio"], .5)
        self.assertEqual(out["difference"], -20)
        self.assertIsNone(out["rigorous_interval"])
        with self.assertRaises(ValueError): k.comparison(x, y, independent_candidates=False)

    def test_n1_and_zero_denominator_defer(self):
        out = k.comparison(k.sample_summary([1]), k.sample_summary([0]), independent_candidates=True)
        self.assertEqual(out["status"], "STATISTICAL_CONCLUSION_DEFERRED")
        self.assertFalse(k.sample_summary([1,1])["population_mean_certified"])

    def test_namespace_stage_candidate_and_collision(self):
        args = ("artificial-campaign", "EXPLORATION", "candidateA", 0, 0, 0, "fake-input")
        seed = k.seed_for(*args)
        self.assertEqual(seed, k.seed_for(*args))
        confirmation = k.seed_for(args[0], "CONFIRMATION", *args[2:])
        self.assertNotEqual(seed, confirmation)
        with self.assertRaises(ValueError): k.seed_for(*args, occupied={seed})

    def test_confirmation_freeze_overlap_and_tampering(self):
        selection = k.freeze_selection(["A", "B"], [1,2,3], "fake-evidence")
        sha = selection["selection_sha256"]
        self.assertTrue(k.check_confirmation(selection, ["A","B"], [4,5,6], preregistered_selection_sha256=sha)["namespace_disjoint"])
        for candidates, seeds in ((["A"], [4,5]), (["A","B"], [3,4]), (["A","B"], [4,4])):
            with self.assertRaises(ValueError): k.check_confirmation(selection, candidates, seeds, preregistered_selection_sha256=sha)
        tampered = copy.deepcopy(selection); tampered["candidates"] = ["C"]
        with self.assertRaises(ValueError): k.check_confirmation(tampered, ["C"], [4], preregistered_selection_sha256=sha)

    def test_planning_has_no_adopted_sample_count(self):
        self.assertEqual(k.planning_n(100, absolute_mcse=2)["n_plugin"], 25)
        self.assertIsNone(k.planning_n(100, relative_mcse=.1, reference_mean=20)["adopted_n"])
        self.assertTrue(k.planning_n(0, absolute_mcse=2)["zero_variance_pilot_is_not_constant_cost_proof"])
        result = k.planning_equal_n_difference(100,100,10,alpha=.05,power=.8,comparisons=3)
        self.assertIsNone(result["adopted_n"])
        self.assertFalse(result["coverage_or_power_guarantee"])


class AuthorizationTests(unittest.TestCase):
    def test_one_shot_consumption_never_refunded(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp)/"own-ledger"
            marker = claim_grant(ledger, {"grant_id": "ARTIFICIAL_ONE_SHOT"})
            self.assertEqual(json.loads(marker.read_text())["state"], "CONSUMED_NO_RETRY")
            with self.assertRaises(FileExistsError): claim_grant(ledger, {"grant_id": "ARTIFICIAL_ONE_SHOT"})

    def test_saved_adapter_projection_with_only_synthetic_values(self):
        def fake_saved(root, path, commit=None):
            if path.parent == SIGNAL:
                system = path.name[:2]
                return {"records": [{"prefix": PREFIX[system], "label": label, "q": q,
                                     "finite_rows": [{"cutoff": 4, "total_signed_bias": {"real": 0., "imag": 0.}, "log_B": 0.}]}
                                    for label in GROUP_LABEL.values() for q in k.Q_GRID]}
            system, short = path.name.split("_")[:2]
            row = fake_row(label=GROUP_LABEL[short])
            return {"SOURCE": "synthetic-source", "preparation_hash": "synthetic-preparation",
                    "records": [{"sample": i//2, "axis": r["axis"], "event_digest": r["event_digest"],
                                 "metrics": {"rz_count": r["rz_count"], "cx_count": r["cx_count"]}}
                                for i, r in enumerate(row["records"])]}
        with patch("h46_higher_pf_adapter.inventory", return_value={"ARTIFICIAL": True}), \
             patch("h46_higher_pf_adapter.checked_json", side_effect=fake_saved):
            rows, meta = load_rows()
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(bool(r["records"]) for r in rows), 6)
        self.assertEqual(sum(r["cost_kind"] == "MISSING" for r in rows), 18)
        self.assertTrue(meta["ARTIFICIAL"])
        self.assertEqual(len(compare_rows(rows)["selections"]), 6)

    def test_invalid_grant_before_real_input_load(self):
        with patch("h46_higher_pf_adapter.load_rows", side_effect=AssertionError("REAL_INPUT_MUST_NOT_LOAD")):
            with self.assertRaisesRegex(ValueError, "EXPLICIT_NUMERICAL"):
                authorize({"status": "UNAPPROVED"}, {"valid": False})

    def test_wrong_phase_and_binding_rejected(self):
        with self.assertRaisesRegex(ValueError, "SCOPE"):
            authorize({"status": "SEALED_APPROVED", "phase": "C"}, {"valid": True})
        with self.assertRaisesRegex(ValueError, "BINDING"):
            authorize({"status": "SEALED_APPROVED", "phase": "B_SAVED_ONLY"}, {"valid": True, "phase": "B_SAVED_ONLY"})

    def test_cli_default_and_draft_refusal_no_output_or_marker(self):
        runner = ROOT/"scripts/run_h46_higher_pf_comparison.py"
        out = subprocess.run([sys.executable, "-B", str(runner)], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(out.stdout)["status"], "READINESS_ONLY")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/"plan.json").write_text('{"status":"UNAPPROVED"}')
            (path/"grant.json").write_text('{"valid":false}')
            result = subprocess.run([sys.executable,"-B",str(runner),"--execute","--plan",str(path/"plan.json"),
                                     "--grant",str(path/"grant.json"),"--output",str(path/"output")], capture_output=True,text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("EXPLICIT_NUMERICAL", result.stderr)
            self.assertFalse((path/"output").exists())
            self.assertFalse((ROOT/".higher_pf_execution_ledger").exists())


if __name__ == "__main__":
    unittest.main()

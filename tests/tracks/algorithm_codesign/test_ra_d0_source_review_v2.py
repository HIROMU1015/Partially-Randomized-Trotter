"""V2 contracts and off-domain fixtures. Never optimize the saved R1 table."""
from dataclasses import replace
from fractions import Fraction as F
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from trottertracks.algorithm_codesign.ra_d0.exact import DENOMINATOR, quantize, log_interval
from trottertracks.algorithm_codesign.ra_d0.lp import LP, dual_lower, strict_witness
from trottertracks.algorithm_codesign.ra_d0.numerical import (
    membership_allowance, membership_certificate, common_certificate,
    build_numerical_lp, certify_nominal, coalesce_to_B3,
)
from trottertracks.algorithm_codesign.ra_d0.backend import nominal, evaluate, OPTIONS
from trottertracks.algorithm_codesign.ra_d0.guard import BudgetGuard, ResourceCap, TechnicalFailure, CAPS
from trottertracks.algorithm_codesign.ra_d0.freeze import (
    query_recipe, write_freeze, load_freeze, coverage_contexts, classification,
)
from trottertracks.algorithm_codesign.ra_d0.engine import OneShotEngine
from trottertracks.algorithm_codesign.ra_d0.launch import (
    validate_authorization, verify_launch, AUTH_PATH, OUTPUT_PATH, RECEIPT_PATH,
)
from trottertracks.algorithm_codesign.ra_d0.table import extract_saved_table
from trottertracks.algorithm_codesign.ra_d0.semantics import audit_semantics

ELL = log_interval(10560)[1]
N = 10**7  # off-domain fixture, no saved R1 timing or cost used
OLD = "artifacts/track_b_ra_d0_preparation/2026-10-06"


def fixture():
    return {"target": ["1", "0", "0", "0"],
            "columns": [{"id": "O0:"+ep, "D_intervals": [["1", "1"]]+[["0", "0"]]*3,
                         "costs": {"T": "1", "CX": "1", "1Q": "1"}, "d_upper": "0", "workspace_peak": 1}
                        for ep in ("1e-3", "1e-4", "1e-6")],
            "B0_saved_profiles": [{"arm": arm, "epsilon": "1e-3", "memberships": [
                {"column_id": "O0:1e-3", "ideal_weight_interval": ["1", "1"],
                 "saved_midpoint_weight": "1"}],
                "original_profile": {"coefficient_and_strict_synthesis_bias_upper": "0",
                                     "implemented_B": "1", "workspace_qubits_beyond_2_system": 1,
                                     "E_native_cost": {"T": "1", "CX": "1", "1Q": "1"}}}
                for arm in ("ordinary", "PTSC_K0", "A")]}


def point(lp):
    nq = len(lp.labels)
    v = [F(0)]*len(lp.c)
    v[0], v[nq] = F(1), F(1)
    if len(lp.c) > nq+5:
        v[nq+5] = F(1)
    return v


def artificial_backend(lp, guard, call_id, baseline, permit=None):
    if lp.domain != "SYNTHETIC":
        raise AssertionError("real data rejected by artificial backend")
    with guard.lp_call(call_id, baseline):
        return {"status": 0, "nominal_primal": point(lp),
                "dual_certificate": dual_lower(lp, [F(0)]*len(lp.A), [F(0)]*len(lp.H))}


def artificial_engine(output, backend=artificial_backend, caps=None):
    table = {"domain": "SYNTHETIC", "tables": {x: fixture() for x in ("1/8", "1/4")}}
    grid = {"grids": {x: {"anchor_shots": [N], "points": [
        {"n": N, "tag": "PRIMARY_ANCHOR"}, {"n": N+1, "tag": "COVERAGE_GRID"}]}
        for x in ("1/8", "1/4")}}
    return OneShotEngine(table, grid, output, {"fixture": "artificial_identity"},
                         synthetic=True, backend=backend, synthetic_caps=caps)


def protected_blob(path):
    # Sparse checkout does not copy old evidence. Read the fixed Git blob for
    # absent paths; physical paths, when present, must match the same identity.
    if path.lower().endswith(".npz"):
        raise AssertionError("NPZ identity is forbidden")
    physical = ROOT/path
    return physical.read_bytes() if physical.exists() else subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "0ddf67756516e08f85fed1b987459a5e862676b7:"+path])


class SourceReviewV2Tests(unittest.TestCase):
    def test_saved_anchor_and_ideal_are_separate(self):
        table = json.loads((ROOT/OLD/"candidate_table_v1.json").read_text())
        members = table["tables"]["1/8"]["B0_saved_profiles"][0]["memberships"]
        self.assertTrue(any(F(m["saved_midpoint_weight"]) != F(m["ideal_weight_interval"][0]) for m in members))
        self.assertTrue(audit_semantics(table)["status"])

    def test_B1_embedding_B2_then_B3_preserves_numerical_certificate(self):
        data = fixture()
        lp1 = build_numerical_lp(data, N, "T", ELL, "B1", representation="ordinary")
        cert1 = certify_nominal(data, lp1, point(lp1), N, ELL, "B1", "T")
        self.assertTrue(cert1["certified"])
        lp2 = build_numerical_lp(data, N, "T", ELL, "B2")
        cert2 = certify_nominal(data, lp2, point(lp2), N, ELL, "B2", "T")
        self.assertTrue(cert2["certified"])
        labels, q = coalesce_to_B3(data, lp2.labels, list(map(F, cert2["q_exact"])))
        cert3 = common_certificate(data, labels, q, F(cert2["y_exact"]), N, ELL)
        self.assertTrue(cert3["certified"])
        self.assertEqual(cert2["resources"], cert3["resources"])
        self.assertEqual(cert2["xi"], cert3["xi"])

    def test_B2_alias_mixture_sum_is_dyadic_and_mean_preserving(self):
        data = fixture()
        lp = build_numerical_lp(data, N, "T", ELL, "B2")
        q = [F(0)]*9; q[0], q[3] = F(1, 4), F(3, 4)
        z = {"ordinary": F(1, 4), "PTSC_K0": F(3, 4), "A": F(0)}
        self.assertTrue(membership_certificate(data, lp.labels, q, F(1), "B2", z)["certified"])
        labels, merged = coalesce_to_B3(data, lp.labels, q)
        self.assertEqual(merged, [1, 0, 0])
        self.assertTrue(common_certificate(data, labels, merged, F(1), N, ELL)["certified"])

    def test_B1_allowance_exact_formula_and_endpoint_verification(self):
        self.assertEqual(membership_allowance(2), F(4, 2**60))
        data = fixture(); data["B0_saved_profiles"][0]["memberships"][0]["ideal_weight_interval"] = [str(1-F(1, 2**60)), str(1+F(1, 2**60))]
        labels = [("ordinary", "O0:"+ep) for ep in ("1e-3", "1e-4", "1e-6")]
        c = membership_certificate(data, labels, [F(1), F(0), F(0)], F(1), "B1")
        self.assertTrue(c["certified"])
        self.assertEqual(F(c["groups"][0]["residual_upper"]), F(1, 2**60))

    def test_membership_over_fixed_allowance_rejected(self):
        data = fixture(); labels = [("ordinary", "O0:1e-3")]
        c = membership_certificate(data, labels, [F(1)], F(1)+F(4, 2**60), "B1")
        self.assertFalse(c["certified"])

    def test_B1_requires_one_representation(self):
        c = membership_certificate(fixture(), [("ordinary", "O0:1e-3"), ("A", "O0:1e-3")], [F(1, 2)]*2, F(1), "B1")
        self.assertFalse(c["certified"])

    def test_z_witness_can_have_non_dyadic_denominator(self):
        data = fixture(); lp = build_numerical_lp(data, N, "T", ELL, "B2")
        q, _ = quantize([F(1, 3), 0, 0, F(2, 3), 0, 0, 0, 0, 0], 1)
        z = {"ordinary": F(1, 3), "PTSC_K0": F(2, 3), "A": F(0)}
        self.assertTrue(membership_certificate(data, lp.labels, q, F(1), "B2", z)["certified"])
        self.assertNotEqual(DENOMINATOR % 3, 0)

    def test_z_must_be_nonnegative_and_sum_to_y(self):
        lp = build_numerical_lp(fixture(), N, "T", ELL, "B2")
        for z in ({"ordinary": 1, "PTSC_K0": -1, "A": 1}, {"ordinary": 1, "PTSC_K0": 1, "A": 0}):
            self.assertFalse(membership_certificate(fixture(), lp.labels, [F(1)]+[F(0)]*8, F(1), "B2", z)["certified"])

    def test_quantized_B2_fixture_and_confidence(self):
        data = fixture(); lp = build_numerical_lp(data, N, "CX", ELL, "B2")
        c = certify_nominal(data, lp, point(lp), N, ELL, "B2", "CX")
        self.assertTrue(c["certified"]); self.assertEqual(c["xi"], "0")
        self.assertEqual(c["membership"]["latent_z"], {"ordinary": "1", "PTSC_K0": "0", "A": "0"})
        self.assertEqual(F(c["objective_upper"]), 2*N)

    def test_B2_outer_lower_le_certified_primal(self):
        data = fixture(); inner = build_numerical_lp(data, N, "T", ELL, "B2")
        cert = certify_nominal(data, inner, point(inner), N, ELL, "B2", "T")
        outer = replace(build_numerical_lp(data, N, "T", ELL, "B2", robust=False), domain="SYNTHETIC")
        result = evaluate(outer, BudgetGuard("SYNTHETIC"), "outer:artificial", "B2")
        self.assertLessEqual(result["dual_certificate"]["lower"], F(cert["objective_upper"]))

    def test_workspace_is_peak_not_average(self):
        data = fixture(); data["columns"][0]["workspace_peak"] = 2
        lp = build_numerical_lp(data, N, "T", ELL, "B2")
        self.assertFalse(certify_nominal(data, lp, point(lp), N, ELL, "B2", "T")["certified"])

    def test_negative_nominal_q_not_clipped(self):
        data = fixture(); lp = build_numerical_lp(data, N, "T", ELL, "B2")
        v = point(lp); v[1] = -F(1, 10**20)
        self.assertFalse(certify_nominal(data, lp, v, N, ELL, "B2", "T")["certified"])

    def test_registered_denominator_fixed_for_law(self):
        data = fixture(); lp = build_numerical_lp(data, N, "T", ELL, "B3")
        self.assertFalse(common_certificate(data, lp.labels, [F(1, 3), F(2, 3), F(0)], F(1), N, ELL)["certified"])
        self.assertEqual(DENOMINATOR, 2**60)

    def test_Phase_A_B3_rejected_before_call(self):
        g = BudgetGuard("SYNTHETIC"); g.current_phase = "BUDGET_FREEZE"
        with self.assertRaises(TechnicalFailure):
            with g.lp_call("forbidden", "B3"):
                self.fail("B3 was entered")
        self.assertEqual(g.main_calls, 0)

    def test_freeze_tamper_rejected_before_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = artificial_engine(directory)
            path, checksum = engine.budget_stage("P1_ANCHORS", [("1/8", N, "PRIMARY_ANCHOR")])
            calls = engine.guard.main_calls
            path.write_bytes(path.read_bytes()+b" ")
            with self.assertRaises(TechnicalFailure):
                engine.compare_stage(path, checksum)
            self.assertEqual(engine.guard.main_calls, calls)

    def test_freeze_input_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory)
            path, checksum = e.budget_stage("P1", [("1/8", N, "PRIMARY_ANCHOR")])
            with self.assertRaises(TechnicalFailure):
                load_freeze(path, checksum, {"wrong": "input"})

    def test_freeze_before_B3_and_complete_artificial_workflow(self):
        phases = []
        with tempfile.TemporaryDirectory() as directory:
            def recording(lp, guard, call_id, baseline, permit=None):
                if baseline == "B3":
                    freezes = list(Path(directory).glob("*/budget_freeze.json"))
                    self.assertTrue(freezes)
                    self.assertNotEqual(guard.current_phase, "BUDGET_FREEZE")
                phases.append((guard.current_phase, baseline))
                return artificial_backend(lp, guard, call_id, baseline, permit)
            e = artificial_engine(directory, recording)
            result = e.run()
            self.assertEqual(result["classification"], "D0_NO_REGISTERED_WITNESS")
            self.assertEqual(result["completed_queries"], 12)
            self.assertTrue(result["mandatory_STOP"])
            self.assertEqual(result["resource_usage"]["main_LP_calls"], 36)
            self.assertEqual([s["stage"] for s in result["stage_freezes"]], ["P1_ANCHORS", "P2_CONDITIONAL_COVERAGE"])
            rows = [json.loads(r) for r in gzip.decompress((Path(directory)/"paired_queries.jsonl.gz").read_bytes()).splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all("certificate_hashes" in row for row in rows))
            self.assertTrue(all(b == "B2" for p, b in phases if p == "BUDGET_FREEZE"))

    def test_uncertified_minimum_stops_before_B3(self):
        with tempfile.TemporaryDirectory() as directory:
            def bad(lp, guard, call_id, baseline, permit=None):
                r = artificial_backend(lp, guard, call_id, baseline, permit)
                r["nominal_primal"][0] = -F(1)
                return r
            e = artificial_engine(directory, bad); r = e.run()
            self.assertEqual(r["classification"], "D0_TECHNICAL_INCONCLUSIVE")
            self.assertEqual(r["technical_reason"], "TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION")
            self.assertEqual(r["completed_queries"], 0)
            self.assertFalse(list(Path(directory).glob("*/budget_freeze.json")))

    def test_budget_vectors_never_cartesian_mix(self):
        vectors = [{"source": name, "resources": {"T": str(t), "CX": str(cx), "1Q": str(q)}}
                   for name, t, cx, q in (("p", 1, 2, 3), ("q", 4, 5, 6))]
        budgets, queries = query_recipe("fixture", 7, "PRIMARY_ANCHOR", vectors)
        self.assertEqual(len(queries), 6)
        t_caps = {tuple(q["caps"].values()) for q in queries if q["objective"] == "T"}
        self.assertEqual(t_caps, {("2", "3"), ("5", "6")})
        self.assertEqual(len(budgets), 2)

    def test_complete_vector_exact_dedup(self):
        v = {"T": "1/2", "CX": "2", "1Q": "3"}
        b, q = query_recipe("fixture", 7, "PRIMARY_ANCHOR", [
            {"source": "a", "resources": v}, {"source": "b", "resources": v | {"T": "2/4"}}])
        self.assertEqual(len(b), 1); self.assertEqual(len(q), 3)
        self.assertEqual(b[0]["sources"], ["a", "b"])

    def test_maximum_12_vectors_36_queries(self):
        vs = [{"source": str(j), "resources": {"T": j, "CX": j+1, "1Q": j+2}} for j in range(12)]
        b, q = query_recipe("fixture", 7, "PRIMARY_ANCHOR", vs)
        self.assertEqual((len(b), len(q)), (12, 36))
        with self.assertRaises(TechnicalFailure):
            query_recipe("fixture", 7, "PRIMARY_ANCHOR", vs+vs[:1])

    def test_incomplete_vector_rejected(self):
        with self.assertRaises(TechnicalFailure):
            query_recipe("fixture", 7, "PRIMARY_ANCHOR", [{"source": "bad", "resources": {"T": 1, "CX": 2}}])

    def test_main_and_auxiliary_bounds_from_737_points(self):
        self.assertEqual(737*(3+12*3*2), CAPS["main_calls"])
        self.assertEqual(CAPS["main_calls"], 55275)
        self.assertEqual(2*CAPS["main_calls"], CAPS["recipe_total_calls"])
        self.assertEqual(CAPS["recipe_total_calls"], 110550)
        self.assertEqual(CAPS["hard_total_calls"], 111000)

    def test_anchor_first_skips_all_coverage_when_both_succeed(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory); seen = []
            def budget(stage, points):
                seen.append((stage, points)); return stage, "fixture"
            def compare(stage, _):
                e.rows = [{"x": x, "n": N, "tag": "PRIMARY_ANCHOR", "strict_witness": True} for x in ("1/8", "1/4")]
            e.budget_stage, e.compare_stage = budget, compare
            self.assertEqual(e.stages(), "D0_STRONG_DEGREE_LOCAL_SIGNAL")
            self.assertEqual(len(seen), 1)

    def test_coverage_only_missing_anchor_context(self):
        self.assertEqual(coverage_contexts({"1/8": True, "1/4": False}), ["1/4"])
        self.assertEqual(coverage_contexts({"1/8": False, "1/4": True}), ["1/8"])
        self.assertEqual(coverage_contexts({}), ["1/8", "1/4"])

    def test_coverage_only_cannot_become_strong(self):
        rows = [{"x": x, "tag": "COVERAGE_GRID", "strict_witness": True} for x in ("1/8", "1/4")]
        self.assertEqual(classification(rows, True), "D0_LOCAL_DEGREE_LOCAL_SIGNAL")

    def test_prefix_witness_or_no_witness_cannot_classify(self):
        for rows in ([], [{"x": "1/8", "tag": "PRIMARY_ANCHOR", "strict_witness": True}]):
            self.assertEqual(classification(rows, False), "D0_TECHNICAL_INCONCLUSIVE")
            self.assertEqual(classification(rows, True, technical_failure=True), "D0_TECHNICAL_INCONCLUSIVE")

    def test_B3_only_feasibility_is_not_strict_witness(self):
        self.assertFalse(strict_witness({"certified": True, "objective_upper": "1"}, None))
        self.assertFalse(strict_witness({"certified": True, "objective_upper": "1"}, {"lower": "1"}))
        self.assertTrue(strict_witness({"certified": True, "objective_upper": "1"}, {"lower": "2"}))

    def test_exact_Farkas_required_after_infeasible_flag(self):
        lp = LP([F(1)], [[F(1)], [F(-1)]], [F(0), F(-1)], [], [], [F(2)])
        guard = BudgetGuard("SYNTHETIC")
        result = evaluate(lp, guard, "infeasible:artificial", "B2")
        self.assertEqual(result["status"], "CERTIFIED_INFEASIBLE")
        self.assertTrue(result["Farkas"]["certified_infeasible"])
        self.assertEqual((guard.main_calls, guard.aux_calls), (1, 1))

    def test_failed_Farkas_cannot_become_negative_result(self):
        lp = LP([F(1)], [[F(1)]], [F(0)], [], [], [F(2)])
        with patch("trottertracks.algorithm_codesign.ra_d0.backend.nominal", side_effect=[
            {"status": 2, "message": "artificial"}, {"status": 0, "nominal_primal": [F(0)]*2}]):
            with self.assertRaises(TechnicalFailure):
                evaluate(lp, BudgetGuard("SYNTHETIC"), "fake", "B2")

    def test_auxiliary_acquisition_failure_is_technical(self):
        lp = LP([F(1)], [], [], [], [], [F(2)])
        with patch("trottertracks.algorithm_codesign.ra_d0.backend.nominal", side_effect=[
            {"status": 2, "message": "artificial"}, {"status": 1, "message": "timeout"}]):
            with self.assertRaises(TechnicalFailure):
                evaluate(lp, BudgetGuard("SYNTHETIC"), "fake", "B2")

    def test_failed_certificate_returned_vectors_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            def failed(lp, guard, call_id, baseline, permit=None):
                raise TechnicalFailure("artificial certificate failure", {"returned_primal": ["1", "0"]})
            result = artificial_engine(directory, failed).run()
            self.assertEqual(result["classification"], "D0_TECHNICAL_INCONCLUSIVE")
            raw = gzip.decompress((Path(directory)/"certificates.jsonl.gz").read_bytes())
            self.assertEqual(json.loads(raw)["value"]["returned_primal"], ["1", "0"])

    def test_retry_and_second_auxiliary_rejected(self):
        g = BudgetGuard("SYNTHETIC")
        with g.lp_call("once", "B2"):
            pass
        with g.lp_call("once", "B2", True):
            pass
        for auxiliary in (False, True):
            with self.assertRaises(TechnicalFailure):
                with g.lp_call("once", "B2", auxiliary):
                    pass
        self.assertEqual((g.main_calls, g.aux_calls), (1, 1))

    def test_cap_refuses_before_backend_and_keeps_count(self):
        g = BudgetGuard("SYNTHETIC", {"main_calls": 1})
        with g.lp_call("first", "B2"):
            pass
        with self.assertRaises(ResourceCap):
            with g.lp_call("second", "B2"):
                self.fail("entered over cap")
        self.assertEqual(g.main_calls, 1)

    def test_cap_hit_engine_records_technical_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, caps={"main_calls": 1})
            result = e.run()
            self.assertEqual(result["classification"], "D0_TECHNICAL_INCONCLUSIVE")
            self.assertIn("D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP", result["technical_reason"])
            self.assertEqual(result["resource_usage"]["main_LP_calls"], 1)
            self.assertTrue(result["mandatory_STOP"])

    def test_per_call_wall_overrun_is_technical(self):
        g = BudgetGuard("SYNTHETIC", {"per_LP_wall_seconds": .01})
        with self.assertRaises(ResourceCap):
            with g.lp_call("sleep:artificial", "B2"):
                time.sleep(.015)

    def test_registered_caps_cannot_be_overridden(self):
        with self.assertRaises(PermissionError):
            BudgetGuard(synthetic_caps={"wall_seconds": 1})

    def test_output_cap_and_terminal_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            g = BudgetGuard("SYNTHETIC", {"output_MiB": 1})
            with self.assertRaises(ResourceCap):
                g.write(Path(directory)/"too_big.json", {"padding": "x"*(1024**2)})
            g.write(Path(directory)/"failure.json", {"mandatory_STOP": True}, terminal=True)
            self.assertFalse((Path(directory)/"too_big.json").exists())

    def test_solver_thread_and_time_options_fixed(self):
        self.assertEqual((OPTIONS["threads"], OPTIONS["parallel"], OPTIONS["time_limit"]), (1, False, 2.))
        self.assertEqual((CAPS["processes"], CAPS["threads"], CAPS["retries"]), (1, 1, 0))

    def test_registered_backend_rejected_without_authorization(self):
        lp = replace(LP([F(1)], [], [], [], [], [F(2)]), domain="REGISTERED_SAVED_TABLE")
        g = BudgetGuard()
        with self.assertRaises(PermissionError):
            nominal(lp, g, "registered:must-not-run", "B2")
        self.assertEqual(g.main_calls, 0)

    def test_real_table_cannot_enter_synthetic_engine(self):
        table = json.loads((ROOT/OLD/"candidate_table_v1.json").read_text())
        with self.assertRaises(PermissionError):
            OneShotEngine(table, {}, "/unused", {}, synthetic=True)

    def test_source_review_launch_denied_before_marker(self):
        self.assertFalse((ROOT/AUTH_PATH).exists())
        with self.assertRaises(FileNotFoundError):
            verify_launch(ROOT)
        self.assertFalse((ROOT/OUTPUT_PATH/"one_shot_consumed.json").exists())

    def test_future_authorization_must_be_direct_child_and_path_limited(self):
        source, head = "a"*40, "b"*40
        auth = {"status": "APPROVED_FOR_ONE_RA_D0_RUN", "development_execution_authorized": True,
                "science_execution_authorized": False, "runs": 1, "retries": 0, "mandatory_STOP": True,
                "source_commit": source, "explicit_execution_instruction": "synthetic validation fixture only",
                "contract_sha256": "c"*64}
        validate_authorization(auth, head, [source], [AUTH_PATH, RECEIPT_PATH], "c"*64)
        for parents, paths in ((["d"*40], [AUTH_PATH]), ([source], [AUTH_PATH, "src/not-allowed.py"])):
            with self.assertRaises(PermissionError):
                validate_authorization(auth, head, parents, paths, "c"*64)
        with self.assertRaises(PermissionError):
            validate_authorization(auth | {"retries": 1}, head, [source], [AUTH_PATH], "c"*64)

    def test_static_columns_sign_controls_and_old_table_identity(self):
        raw = (ROOT/"artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json").read_bytes()
        table = extract_saved_table(raw)
        self.assertEqual(table, json.loads((ROOT/OLD/"candidate_table_v1.json").read_text()))
        for t in table["tables"].values():
            self.assertEqual((len(t["columns"]), len(t["sign_checks"])), (21, 9))
            self.assertTrue(all(s["cost_error_equal"] for s in t["sign_checks"]))

    def test_old_R1_result_marker_authorization_source_hashes_unchanged(self):
        manifest_path = "artifacts/track_b_rte_reallocation_r1_source/2026-10-06/source_manifest_v1.json"
        manifest = json.loads(protected_blob(manifest_path))
        expected = dict(manifest["critical_sha256"])
        expected.update({
            "artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json": "f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e",
            "artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/one_shot_consumed.json": "f25000ee5e3b94eb499b4a89bb28814e3de1aea641b249bcbb9d953d427c2ced",
            "artifacts/track_b_rte_reallocation_r1_source/2026-10-06/authorization.json": "9d30dd6cbe5b4d47d4b9080db656564fec6503d9e77a4f08442a39c8f30a582f"})
        for path, sha in expected.items():
            self.assertEqual(hashlib.sha256(protected_blob(path)).hexdigest(), sha, path)


if __name__ == "__main__":
    unittest.main()

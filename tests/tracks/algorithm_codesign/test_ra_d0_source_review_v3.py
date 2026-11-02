"""Off-domain regression for certified-infeasible minimum/point handling."""
from dataclasses import replace
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from test_ra_d0_source_review_v2 import (
    ROOT, ELL, N, fixture, artificial_backend, artificial_engine, point,
)
from trottertracks.algorithm_codesign.ra_d0.backend import evaluate, OPTIONS
from trottertracks.algorithm_codesign.ra_d0.engine import OneShotEngine
from trottertracks.algorithm_codesign.ra_d0.exact import DENOMINATOR
from trottertracks.algorithm_codesign.ra_d0.freeze import load_freeze, classification, identity
from trottertracks.algorithm_codesign.ra_d0.guard import BudgetGuard, TechnicalFailure, CAPS
from trottertracks.algorithm_codesign.ra_d0.lp import LP
from trottertracks.algorithm_codesign.ra_d0.numerical import build_numerical_lp

BASE = "4012cbd167ec10f143fbfddc89feff4dec54bf2b"
LOW = 17  # artificial confidence task; not a saved R1 shot point


def exact_infeasible(lp, guard, call_id, baseline, permit=None):
    """Actual solver + exact ray check, only for the artificial low-n LP."""
    if lp.domain != "SYNTHETIC":
        raise AssertionError("registered domain refused by fixture")
    result = evaluate(lp, guard, call_id, baseline)
    if result["status"] != "CERTIFIED_INFEASIBLE":
        raise AssertionError("artificial low-n LP must be certified infeasible")
    return result


def selective_backend(lp, guard, call_id, baseline, permit=None):
    if f":{LOW}:" in call_id:
        return exact_infeasible(lp, guard, call_id, baseline)
    return artificial_backend(lp, guard, call_id, baseline)


class SourceReviewV3Tests(unittest.TestCase):
    def test_exact_Farkas_minimum_is_normal_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, exact_infeasible)
            path, checksum = e.budget_stage("ARTIFICIAL", [("1/8", LOW, "COVERAGE_GRID")])
            body = load_freeze(path, checksum, e.inputs); p = body["points"][0]
            self.assertEqual(p["point_status"], "B2_POINT_CERTIFIED_INFEASIBLE")
            self.assertEqual(p["certified_infeasible_objectives"], ["T", "CX", "1Q"])
            self.assertTrue(all(m["status"] == "B2_CERTIFIED_INFEASIBLE_AT_N" for m in p["minima"]))
            self.assertEqual((e.guard.main_calls, e.guard.aux_calls), (3, 3))
            full = [json.loads(r)["value"] for r in gzip.decompress((Path(directory)/"certificates.jsonl.gz").read_bytes()).splitlines()]
            self.assertTrue(all(r["solver"]["Farkas"]["certified_infeasible"] for r in full))
            self.assertTrue(all(r["solver"]["full_auxiliary_primal"] for r in full))

    def test_certified_infeasible_has_no_budget_or_paired_query(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, exact_infeasible)
            path, checksum = e.budget_stage("ARTIFICIAL", [("1/8", LOW, "PRIMARY_ANCHOR")])
            p = load_freeze(path, checksum, e.inputs)["points"][0]
            self.assertEqual((p["budget_vectors"], p["queries"]), ([], []))
            calls = e.guard.main_calls
            e.compare_stage(path, checksum)
            self.assertEqual(e.guard.main_calls, calls)
            self.assertFalse(e.point_outcomes[0]["B3_feasibility_executed"])
            self.assertFalse(e.rows)

    def test_same_n_saved_B0_cannot_reenable_skipped_point(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, exact_infeasible)
            # This artificial saved reproduction anchor is deliberately distinct
            # from the LP law. It supplies no mathematical inclusion assumption.
            for data in e.table["tables"].values():
                for profile in data["B0_saved_profiles"]:
                    profile["original_profile"]["implemented_B"] = "1/1000000"
            path, checksum = e.budget_stage("ARTIFICIAL", [("1/8", LOW, "COVERAGE_GRID")])
            p = load_freeze(path, checksum, e.inputs)["points"][0]
            self.assertEqual(len(p["same_n_feasible_B0_profile_IDs"]), 3)
            self.assertEqual((p["budget_vectors"], p["queries"]), ([], []))

    def test_coverage_skips_low_n_and_freezes_next_ready_point(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, selective_backend)
            path, checksum = e.budget_stage("P2", [("1/8", LOW, "COVERAGE_GRID"), ("1/8", N, "COVERAGE_GRID")])
            body = load_freeze(path, checksum, e.inputs)
            self.assertEqual([p["point_status"] for p in body["points"]], ["B2_POINT_CERTIFIED_INFEASIBLE", "BUDGET_READY"])
            self.assertEqual((body["number_of_B2_certified_infeasible_points"], body["number_of_budget_ready_points"]), (1, 1))
            e.compare_stage(path, checksum)
            self.assertEqual([p["n"] for p in e.point_outcomes], [LOW, N])
            self.assertEqual(len(e.rows), 3)
            self.assertTrue(all(r["n"] == N for r in e.rows))

    def test_full_run_infeasible_anchors_continue_to_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, selective_backend)
            for g in e.grid["grids"].values():
                g["anchor_shots"] = [LOW]
                g["points"] = [{"n": LOW, "tag": "PRIMARY_ANCHOR"}, {"n": N, "tag": "COVERAGE_GRID"}]
            result = e.run()
            self.assertEqual(result["classification"], "D0_NO_REGISTERED_WITNESS")
            self.assertEqual(result["B2_certified_infeasible_points"], 2)
            self.assertEqual(result["completed_budget_ready_points"], 2)
            self.assertEqual(result["certified_comparable_queries"], 6)
            self.assertEqual(len(result["stage_freezes"]), 2)
            self.assertTrue(result["mandatory_STOP"])

    def test_all_infeasible_domain_does_not_claim_negative_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, exact_infeasible)
            for g in e.grid["grids"].values():
                g["anchor_shots"] = [LOW]; g["points"] = [{"n": LOW, "tag": "PRIMARY_ANCHOR"}]
            result = e.run()
            self.assertEqual(result["classification"], "D0_NO_REGISTERED_WITNESS")
            self.assertEqual((result["completed_queries"], result["certified_comparable_queries"]), (0, 0))
            self.assertEqual(result["B2_certified_infeasible_points"], 2)
            self.assertIn("not negative evidence", result["NO_REGISTERED_WITNESS_scope"])

    def test_uncertified_Farkas_acquisition_still_technical(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, evaluate)
            with patch("trottertracks.algorithm_codesign.ra_d0.backend.nominal", side_effect=[
                {"status": 2, "message": "artificial infeasible"}, {"status": 1, "message": "artificial timeout"}]):
                result = e.run()
            self.assertEqual(result["classification"], "D0_TECHNICAL_INCONCLUSIVE")
            self.assertEqual(result["technical_reason"], "Farkas acquisition failure")
            self.assertFalse(result["stage_freezes"])

    def test_failed_exact_ray_still_technical(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, evaluate)
            with patch("trottertracks.algorithm_codesign.ra_d0.backend.check_farkas_output", return_value={"certified_infeasible": False}), patch(
                "trottertracks.algorithm_codesign.ra_d0.backend.nominal", side_effect=[
                    {"status": 2, "message": "artificial"}, {"status": 0, "nominal_primal": [F(0)]}]):
                result = e.run()
            self.assertEqual(result["classification"], "D0_TECHNICAL_INCONCLUSIVE")
            self.assertEqual(result["technical_reason"], "Farkas exact ray verification failure")

    def test_forged_status_without_verified_ray_cannot_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            def forged(*args):
                return {"status": "CERTIFIED_INFEASIBLE", "Farkas": {"certified_infeasible": True}, "full_auxiliary_primal": []}
            e = artificial_engine(directory, forged)
            result = e.run()
            self.assertEqual(result["classification"], "D0_TECHNICAL_INCONCLUSIVE")
            self.assertEqual(result["technical_reason"], "TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION")
            self.assertFalse(result["stage_freezes"])

    def test_later_technical_minimum_failure_still_stops_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            def second_fails(lp, guard, call_id, baseline, permit=None):
                if call_id.endswith(':minimum:T'):
                    return exact_infeasible(lp, guard, call_id, baseline)
                raise TechnicalFailure('artificial later objective failure')
            e = artificial_engine(directory, second_fails)
            for g in e.grid['grids'].values():
                g['anchor_shots'] = [LOW]
            result = e.run()
            self.assertEqual(result['classification'], 'D0_TECHNICAL_INCONCLUSIVE')
            self.assertEqual(result['technical_reason'], 'artificial later objective failure')
            self.assertFalse(result['stage_freezes'])

    def test_feasible_nominal_failed_certificate_still_technical(self):
        with tempfile.TemporaryDirectory() as directory:
            def invalid(lp, guard, call_id, baseline, permit=None):
                result = artificial_backend(lp, guard, call_id, baseline)
                result["nominal_primal"][0] = -F(1)
                return result
            result = artificial_engine(directory, invalid).run()
            self.assertEqual(result["classification"], "D0_TECHNICAL_INCONCLUSIVE")
            self.assertEqual(result["technical_reason"], "TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION")

    def test_infeasible_anchor_and_descriptive_feasibility_are_not_witnesses(self):
        rows = [{"x": x, "tag": "PRIMARY_ANCHOR", "strict_witness": False,
                 "classification": "B3_ONLY_FEASIBLE_DESCRIPTIVE"} for x in ("1/8", "1/4")]
        self.assertEqual(classification(rows, True), "D0_NO_REGISTERED_WITNESS")

    def test_other_comparable_anchors_can_still_be_strong(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, selective_backend)
            for g in e.grid["grids"].values():
                g["anchor_shots"] = [LOW, N]
            def controlled_compare(path, checksum):
                e.compare_original(path, checksum)
                # Synthetic control-flow witness flag only; no resource claim.
                for row in e.rows:
                    row["strict_witness"] = row["n"] == N
            e.compare_original = e.compare_stage; e.compare_stage = controlled_compare
            result = e.run()
            self.assertEqual(result["classification"], "D0_STRONG_DEGREE_LOCAL_SIGNAL")
            self.assertEqual(result["B2_certified_infeasible_points"], 2)
            self.assertEqual(len(result["stage_freezes"]), 1)

    def test_skipped_coverage_never_upgrades_to_strong(self):
        rows = [{"x": x, "tag": "COVERAGE_GRID", "strict_witness": True} for x in ("1/8", "1/4")]
        self.assertEqual(classification(rows, True), "D0_LOCAL_DEGREE_LOCAL_SIGNAL")

    def test_skipped_point_cannot_carry_budgets_even_with_rehashed_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory, exact_infeasible)
            path, checksum = e.budget_stage("P2", [("1/8", LOW, "COVERAGE_GRID")])
            body = load_freeze(path, checksum, e.inputs)
            body["points"][0]["budget_vectors"] = [{"resources": {"T": "1", "CX": "1", "1Q": "1"}, "sources": ["forged"]}]
            raw = (json.dumps(body,sort_keys=True)+"\n").encode(); path.write_bytes(raw)
            with self.assertRaises(TechnicalFailure):
                load_freeze(path, hashlib.sha256(raw).hexdigest(), e.inputs)

    def test_minimum_status_mismatch_in_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            e = artificial_engine(directory)
            path, checksum = e.budget_stage("P1", [("1/8", N, "PRIMARY_ANCHOR")])
            body = load_freeze(path, checksum, e.inputs)
            body["points"][0]["minima"][0]["status"] = "B2_CERTIFIED_INFEASIBLE_AT_N"
            raw = (json.dumps(body,sort_keys=True)+"\n").encode(); path.write_bytes(raw)
            with self.assertRaises(TechnicalFailure):
                load_freeze(path, hashlib.sha256(raw).hexdigest(), e.inputs)

    def test_B3_not_called_for_infeasible_point_and_batch_freeze_kept(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            def recording(lp, guard, call_id, baseline, permit=None):
                events.append((guard.current_phase, baseline, call_id))
                if baseline == "B3":
                    self.assertTrue(list(Path(directory).glob('*/budget_freeze.json')))
                return selective_backend(lp, guard, call_id, baseline, permit)
            e = artificial_engine(directory, recording)
            e.compare_stage(*e.budget_stage("P2", [("1/8", LOW, "COVERAGE_GRID"), ("1/8", N, "COVERAGE_GRID")]))
            self.assertTrue(all(b == "B2" for phase,b,_ in events if phase == "BUDGET_FREEZE"))
            self.assertEqual(sum(b == "B3" for _,b,_ in events), 3)

    def test_call_caps_and_backend_and_denominator_unchanged(self):
        old=root_blob('artifacts/track_b_ra_d0_source_review_v2/2026-10-07/execution_contract_v2.json')
        self.assertEqual(CAPS, json.loads(old)['resource_caps'])
        self.assertEqual((CAPS['main_calls'], CAPS['recipe_total_calls'], CAPS['hard_total_calls']), (55275,110550,111000))
        self.assertEqual(DENOMINATOR,2**60)
        for path in ('backend.py','guard.py','exact.py','numerical.py','lp.py','table.py','grid.py','semantics.py'):
            p='src/trottertracks/algorithm_codesign/ra_d0/'+path
            self.assertEqual((ROOT/p).read_bytes(),root_blob(p),p)

    def test_candidate_sign_and_existing_80_tests_unchanged(self):
        p='artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json'
        raw=(ROOT/p).read_bytes();self.assertEqual(raw,root_blob(p))
        table=json.loads(raw)
        self.assertTrue(all(len(t['columns'])==21 for t in table['tables'].values()))
        self.assertEqual(sum(len(t['sign_checks']) for t in table['tables'].values()),18)
        for name in ('test_ra_d0_preparation.py','test_ra_d0_source_review_v2.py'):
            p='tests/tracks/algorithm_codesign/'+name
            self.assertEqual((ROOT/p).read_bytes(),root_blob(p))

    def test_old_R1_protected_hashes_unchanged(self):
        p='artifacts/track_b_ra_d0_source_review_v2/2026-10-07/provenance_audit_v2.json'
        hashes=json.loads(root_blob(p))['protected_sha256']
        for path,expected in hashes.items():
            self.assertFalse(path.lower().endswith('.npz'))
            raw=(ROOT/path).read_bytes() if (ROOT/path).exists() else root_blob(path)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),expected,path)


def root_blob(path):
    return subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+path])


if __name__ == '__main__':
    unittest.main()

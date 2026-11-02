"""Result inventory failure controls; no matrix, coefficients or score evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from trottertracks.algorithm_codesign.synthesis_placement.wrapper_result import validate_result

ROOT = Path(__file__).resolve().parents[3]


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads((ROOT/"artifacts/track_b_sp1_wrapper_source/2026-10-06/contract_v1.json").read_text())
        self.receipt = {"source_commit": "1"*40, "authorization_commit": "2"*40,
            "runs": 1, "retries": 0, "mandatory_STOP": True, "next_stage_authorized": False,
            "status": "SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW",
            "synthesis_calls": 0, "sampling_calls": 0, "GPU_query_use": 0,
            "rows": [], "summary": {"research_GO": None, "automatic_next_stage": None}}
        for t in self.contract["domain"]["templates"]:
            for n in self.contract["domain"]["sizes"]:
                for m in self.contract["domain"]["masks"]:
                    duplicate = ("NONE" if m == "R" else "D" if m == "DR" else None) if t in ("A", "B") else None
                    self.receipt["rows"].append({"template": t, "n": n, "mask": m,
                        "classification": "NO_MATERIAL_SEPARATION", "duplicate_of_mask": duplicate,
                        "count_as_independent_positive": False,
                        "axes": [{"axis": a, "status": "ELIGIBLE", "shots_sufficient": 1} for a in ("Re", "Im")]})

    def test_structural_complete_inventory_accepts_no_research_go(self):
        validate_result(self.receipt, self.contract)

    def test_missing_or_duplicated_row_cannot_claim_complete(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(-1, rows[0])):
            result = deepcopy(self.receipt)
            mutation(result["rows"])
            with self.assertRaises(ValueError):
                validate_result(result, self.contract)

    def test_cap_hit_cannot_remain_eligible(self):
        self.receipt["rows"][0]["axes"][0]["shots_sufficient"] = 1000000001
        with self.assertRaises(ValueError):
            validate_result(self.receipt, self.contract)

    def test_duplicate_gain_cannot_be_an_independent_positive(self):
        row = self.receipt["rows"][3]  # A DR duplicates D.
        row.update(classification="MATERIAL_GAIN", count_as_independent_positive=True)
        with self.assertRaises(ValueError):
            validate_result(self.receipt, self.contract)

    def test_mislabeled_axis_and_automatic_go_are_rejected(self):
        original = deepcopy(self.receipt)
        self.receipt["rows"][0]["axes"][1]["axis"] = "Re"
        with self.assertRaises(ValueError):
            validate_result(self.receipt, self.contract)
        original["summary"]["research_GO"] = True
        with self.assertRaises(ValueError):
            validate_result(original, self.contract)

    def test_partial_failure_preserves_stop_without_complete_inventory(self):
        self.receipt.update(status="INCONCLUSIVE_MANDATORY_STOP_NO_RETRY", rows=[])
        validate_result(self.receipt, self.contract)
        self.receipt["mandatory_STOP"] = False
        with self.assertRaises(ValueError):
            validate_result(self.receipt, self.contract)


if __name__ == "__main__":
    unittest.main()

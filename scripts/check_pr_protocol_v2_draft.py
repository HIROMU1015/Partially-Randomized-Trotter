#!/usr/bin/env python3
"""Check draft state, preserved evidence and document joins; no science imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = "artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/"
PROTO = "docs/research/track_a_pr_resource_evaluation_protocol_v2_DRAFT.md"
V1 = "docs/research/track_a_pr_resource_evaluation_protocol_v1.md"
PAPER = "artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/"
OLD = "artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/"


def check(root=ROOT):
    root = Path(root)
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    def j(path):
        return json.loads((root / path).read_text(), parse_float=str)
    def c(path):
        with (root / path).open(newline="") as stream:
            return list(csv.DictReader(stream))
    def sha(path):
        return hashlib.sha256((root / path).read_bytes()).hexdigest()
    def bound(x):
        require(sha(x["path"]) == x["sha256"], "preserved/bound bytes changed: " + x["path"])
    def visit(x):
        if isinstance(x, dict):
            if "path" in x and "sha256" in x:
                bound(x)
            for value in x.values():
                visit(value)
        elif isinstance(x, list):
            for value in x:
                visit(value)
    protection = j(B + "preservation.json")
    for record in protection["files"]:
        bound(record)
    require(any(x["path"] == V1 for x in protection["files"]), "v1 not protected")
    draft = j(B + "draft_status.json")
    require(draft["status"] == "DRAFT_V2_NOT_ADOPTED_NO_EXECUTION_AUTHORIZATION", "draft silently adopted")
    require(draft["formal_adoption_by_user_required"] is True, "user adoption bypassed")
    for key in ["H2_paper_main_table_member", "new_numerical_analysis", "private_files_read_or_published", "existing_contracts_changed", "new_grant_issued", "final_evidence_freeze", "detailed_scientific_review_started", "COMMON_CORE_READY_automatically_granted"]:
        require(draft[key] is False, "forbidden draft scope: " + key)
    require(draft["new_scientific_calls"] == 0, "new science forbidden")
    manifest = j(B + "protocol_manifest.json")
    require(manifest["status"] == "DRAFT_NOT_ADOPTED" and manifest["formal_adoption_pending"] is True, "manifest adoption drift")
    require(manifest["H2_paper_inclusion_authorized"] is False, "H2 paper scope mixed")
    require(manifest["new_scientific_calculations"] is False, "calculation authorization drift")
    require(manifest["logical_protocol_base"] == "9036d8a87ad1c38e83b9ff654b412dde7041e9d3", "wrong v1 base")
    visit(manifest)
    require((root / B / "H468_independent_confirmations.csv").read_bytes() == (root / PAPER / "main_table.csv").read_bytes(), "paper confirmation values changed or H2 appended")
    stages = {(r["system"], r["stage"]): r for r in c(B + "stage_status.csv")}
    require(len(stages) == 17, "stage inventory drift")
    for stage in ["F", "FP", "D", "A", "S_STATS"]:
        require(stages[("H2", stage)]["status"] == "COMPLETE_MANDATORY_STOP", "H2 completed stage lost")
    require(stages[("H2", "P")]["status"] == "NOT_EXECUTED_NOT_SEALED", "H2 P falsely completed")
    require(stages[("H8", "S_STATS")]["status"] == "NO_SEPARATE_COMPLETION_VERIFIED", "H8 S_STATS inferred")
    require(stages[("H8", "P")]["status"] == "SAVED_POINT_RESULTS_ADOPTED_PUBLIC_PR_AUDIT_PARTIAL", "H8 public audit gap lost")
    projection = j(B + "H2_saved_condition_projection.json")
    require(projection["rank"] == 3 and projection["sector"]["dimension"] == 4, "old H2 input mixed")
    require(projection["A_scientific_execution"] is False, "historical H2 A state overwritten")
    history = j(B + "H2_status_history.json")
    require(history["commit"] == "7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb" and history["H2_stages"]["A"] == "NOT_EXECUTED_PREPARED_ONLY", "initial H2 A history lost")
    current = j(B + "H2_latest_condition_projection.json")
    require(current["source_result_commit"] == manifest["H2_fixed_results_snapshot"] == "829d697ae7905b3ea90a61da9d2d4da15d8b757a", "latest H2 selection lost")
    require(current["A"]["status"] == current["S_STATS"]["status"] == "COMPLETE_MANDATORY_STOP", "current H2 completed stages lost")
    require(current["A"]["n_per_candidate"] == 8 and current["A"]["FP_PR_cost_pooled_with_A"] is False, "H2 A/FP cohorts mixed")
    pd = current["P_draft_saved"]
    require(pd["status"] == "PENDING_USER_ADOPTION_AND_FINAL_P_SEAL", "H2 P draft adopted")
    for key in ["P_n_final", "P_SOURCE", "P_seeds", "P_grant"]:
        require(pd[key] is None, "H2 P unsealed field filled by inference")
    require(pd["P_science_authorized"] is False and pd["P_executed"] is False, "H2 P execution invented")
    require(pd["recommended_scenario"]["n_fixed"] == {"B2_p1_q1_2nd_r2_K4": 80, "B2_p1_q4_2nd_r2_K4": 12}, "H2 saved proposed n changed")
    require(current["S_STATS"]["population_mean_certified"] is False and current["S_STATS"]["actual_power_certified"] is False, "H2 design promoted to guarantee")
    require(all(r["unobserved_strata_zero_filled"] is False and r["unseen_cost"] is None for r in current["S_STATS"]["saved_rare_information"]), "H2 unseen strata zero filled")
    require(projection["A_plan_without_seed_values"]["cohort"] == "EXPLORATION_N8" and projection["A_plan_without_seed_values"]["n"] == 8, "A cohort drift")
    require(projection["A_plan_without_seed_values"]["FP_PR_cost_pooled_with_A"] is False, "H2 FP n1 pooled into A")
    for key in ["SOURCE", "candidates", "n", "seeds", "scientific_call_budget"]:
        require(projection["P_contract_saved"][key] is None, "H2 P unsealed field filled by inference")
    sampling = j(B + "sampling_and_seed_status.json")
    require(sampling["H2"]["FP_PR_n_per_candidate"] == 1 and sampling["H2"]["activation_refresh_required"] is True, "H2 seed/native diagnostic scope")
    require(sampling["H2"]["FP_A_pooled"] is False and sampling["H2"]["P_n"] is None and sampling["H2"]["P_n_adopted"] is False, "H2 pooling or n inference")
    require(sampling["H2"]["A_status"] == "COMPLETE_MANDATORY_STOP" and sampling["H2"]["A_seed_activation_refresh_completed"] is True, "H2 A completion/seed refresh lost")
    registries = j(B + "candidate_registrations.json")["registries"]
    expected = {"H2_D70": 70, "H2_A66": 66, "H4_TOL_D": 190, "H4_TOL_A": 50, "H6_strengthened_S2_S4": 136, "H6_new_A16": 16, "H8_strengthened184": 184, "H8_new_A16": 16}
    for key, size in expected.items():
        registry = registries[key]
        rows = registry.get("cells", registry.get("rows"))
        require(len(rows) == size and len({x["id"] for x in rows}) == size, "registered domain drift: " + key)
    inputs = j(B + "system_conditions.json")
    require([x["system"] for x in inputs["systems"]] == ["H2", "H4", "H6", "H8"], "system inventory")
    require([x["actual_rank"] for x in inputs["systems"]] == [3, 10, 19, 28], "DF rank drift")
    require(all(x["all_input_procedures_identical"] is False for x in inputs["systems"]), "input equivalence inferred")
    require(inputs["H4_legacy"]["separate_from_current_P"] is True, "H4 DF12/DF10 mixed")
    for row in c(B + "paper_alignment.csv"):
        bound(row)
        require(row["status"] == "CONSISTENT_WITH_SCOPE_SEPARATION", "paper alignment drift")
    require(len(c(B + "changes.csv")) == 10, "change inventory")
    receipts = j(B + "github_read_receipt.json")
    require(len(receipts["files"]) == 67 and len({x["label"] for x in receipts["files"]}) == 67, "public read inventory")
    require(all(x["fixed_local_blob_matches_remote"] and x["read_extent"] == "FULL_FILE" for x in receipts["files"]), "public blob mismatch")
    require(all(r["evidence_label"] in {x["label"] for x in receipts["files"]} for r in stages.values()), "unresolved stage evidence")
    proto = (root / PROTO).read_text()
    for token in ["DRAFT_V2_NOT_ADOPTED_NO_EXECUTION_AUTHORIZATION", "H2をv1対象外から追加対象として提案", "当初のA未完了", "A66完了", "S_STATS完了", "提案n80/12", "NOT_REPRODUCED", "PF優位方向", "REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS", "後続Mechanism", "H2はこの表に入らない", "mandatory STOP"]:
        require(token in proto, "protocol scope text missing: " + token)
    # Check all six P rows against saved manuscript values, without cost arithmetic.
    saved = j(OLD + "main_table_proposed.json")["rows"]
    lines = [s for s in proto.splitlines() if s.startswith("|H") and "B2_" in s]
    require(len(lines) == 6, "P table must remain six H468 comparisons")
    def dot(token):
        return str(token).removeprefix("0")
    for row, line in zip(saved, lines):
        cols = line.split("|")[1:-1]
        expected = [f"{row['system']} DF{row['DF_rank']}", dot(row["epsilon"]), row["PR_id"], f"{row['L_D']} / {dot(row['delta'])}", str(row["n"]), row["PF_id"], row["saved_direction_status"], row["lower_saved_point_cost"]]
        require(cols == expected, "P saved direction/cohort token mismatch")
    stop = j(B + "MANDATORY_STOP.json")
    require(stop["status"] == "MANDATORY_STOP" and stop["v2_adopted"] is False and stop["next_scientific_stage_authorized"] is False, "STOP boundary drift")
    return {"status": "PASS_STRUCTURE_ONLY_NOT_INDEPENDENT_SCIENTIFIC_REVIEW", "public_files_read": 67, "protected_files": len(protection["files"]), "candidate_domains": 8, "stage_rows": 17, "paper_P_rows_unchanged": 6, "v2_adopted": False, "new_scientific_calls": 0, "new_numerical_analysis": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="store_true")
    args = parser.parse_args()
    text = json.dumps(check(), ensure_ascii=False, indent=2) + "\n"
    if args.receipt:
        (ROOT / B / "structural_validation.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

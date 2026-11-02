#!/usr/bin/env python3
"""Read-only manuscript structure checks; no scientific imports or arithmetic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = "artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/"
OLD = "artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/"
FIG = "artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/"
BASE = "a77141ba5ff0752d6409e32fda9e4649d0919fdb"


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
    def digest(path):
        return hashlib.sha256((root / path).read_bytes()).hexdigest()
    def bound(item):
        require(digest(item["path"]) == item["sha256"], "bound document/hash changed: " + item["path"])

    preservation = j(B + "preservation.json")
    require(preservation["base_commit"] == BASE, "wrong accepted base")
    for item in preservation["files"]:
        bound(item)
    original = [x for x in preservation["files"] if x["path"].startswith(FIG)]
    require(len(original) == 34, "original Figure2 inventory")
    require((root / B / "main_table.csv").read_bytes() == (root / OLD / "main_table_proposed.csv").read_bytes(), "main-table numeric bytes changed")
    rows = j(OLD + "main_table_proposed.json")["rows"]
    require(len(rows) == 6, "six comparison scope")
    require(rows[0]["saved_direction_status"] == "NOT_REPRODUCED" and rows[0]["lower_saved_point_cost"] == "PF", "H4 coarse non-reproduction")
    require(rows[2]["saved_direction_status"] == "REPRODUCED" and rows[2]["lower_saved_point_cost"] == "PF", "H6 coarse PF-favouring reproduction")
    table = (root / B / "main_table.md").read_text()
    tablerows = [line.split("|")[1:-1] for line in table.splitlines() if line.startswith("|H")][:6]
    expected = [[str(r[k]) if k != "window" else f"{r['L_D']} / {r['delta']}"
                 for k in ["comparison_id", "epsilon", "window", "n", "G_PR_exact", "paired_SE", "G_PF_exact", "saved_direction_status", "lower_saved_point_cost"]] for r in rows]
    require(tablerows == expected, "manuscript table saved-token/direction mismatch")
    layout = j(B + "layout_decision.json")
    require(layout["accepted_base_commit"] == BASE and layout["main_table"] == "SIX_FIXED_INDEPENDENT_CONFIRMATION_COMPARISONS", "layout/main scope")
    require(layout["Figure2A"] == "LEGACY_EXPLORATORY_MECHANISM" and layout["Figure2B"] == "H8_INDEPENDENT_N48_MECHANISM", "Figure2 cohort relabelling")
    require(layout["H8_B_is_same_evidence_as_Table1"] is True, "H8 double counting")
    manifest = j(B + "publication_manifest.json")
    require(manifest["base_commit"] == BASE, "manifest base")
    for key in ["new_numerical_analysis", "private_files_opened_or_published", "H2_included", "COMMON_CORE_READY", "input_procedures_fully_identical", "size_causal_claim", "population_mean_certified", "final_evidence_freeze", "detailed_scientific_review_started"]:
        require(manifest[key] is False, "forbidden scientific scope: " + key)
    require(manifest["new_scientific_calls"] == 0 and manifest["main_table"]["values_recomputed"] is False, "new scientific processing")
    for key in ["methods", "input_exceptions", "H8_public_evidence", "prior_paper_manifest", "prior_CEM", "current_claim_overlay", "prior_evidence_ledger", "display_evidence_linkage", "documentation_evidence_ledger"]:
        bound(manifest[key])
    for key in ["csv", "manuscript_text", "immutable_source"]:
        bound(manifest["main_table"][key])
    figure = manifest["figure2"]
    require(figure["original_files"] == 34 and figure["changed"] is False, "Figure2 alteration")
    require(figure["A"]["H4_policy"] == "LEGACY_FIXED_DF12", "H4 legacy relabelled")
    require(figure["A82_original_note"] == "HISTORICAL_RENDER_TIME_ONLY" and figure["A82_current"] == "LATER_SAVED_EXPLANATION_COMPLETE_ALL32_64_AXES", "stale A82 status")
    require(figure["new_factors_transferred_to_H4_H6_P"] is False, "new P factor transfer")
    for panel in ["A", "B"]:
        for item in figure[panel]["assets"]:
            bound(item)
    for key in ["caption", "old_display_plan", "old_axis_table", "completed_factor_table"]:
        bound(figure[key])
    caption = (root / B / "figure2_caption.md").read_text()
    for token in ["legacy fixed DF12", "exploratory n8", "n = 48", "Historical A82 annotation", "1ff6ac0b6ffed6231dac5f655275b620598d0201", "not additional independent evidence", "unknown cost and variance", "detailed independent-PR execution receipts"]:
        require(token in caption, "caption loses scope: " + token)
    methods = (root / B / "methods.md").read_text()
    for token in ["scipy.linalg.eigh", "eigsh", "one binary64 normalization", "1e-10", "1e-9", "original provider/input witnesses were not re-audited", "We do not infer complete procedural identity or attribute between-system differences solely to molecular size", "No new ratios", "not proof of a later execution grant"]:
        require(token in methods, "Methods loses exception: " + token)
    evidence = {x["item_id"]: x for x in c(B + "h8_public_evidence_status.csv")}
    require(len(evidence) == 14, "H8 evidence item inventory")
    require(evidence["H8_FIXED_PF_NATIVE"]["status"] == "PUBLIC_SELECTED_PF_RECEIPT_VERIFIED", "H8 PF receipt scope lost")
    require(evidence["H8_PR_NATIVE"]["status"] == "REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS", "H8 PR audit gap erased")
    require(evidence["H8_ORIGINAL_PROFILE"]["status"] == "NOT_VERIFIED_FROM_PUBLIC_ORIGINAL_BYTES", "H8 original profile incorrectly verified")
    require(evidence["H8_ACTUAL_SEED_EXCLUSION"]["status"] == "NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS", "H8 actual seed proof invented")
    exceptions = c(B + "input_exceptions.csv")
    require(len(exceptions) == 8 and len({x["exception_id"] for x in exceptions}) == 8, "exception inventory")
    require(exceptions[3]["status"] == "DECLARED_SOLVER_DIFFERENCE" and exceptions[4]["status"] == "DECLARED_NUMERICAL_GATE_DIFFERENCE", "solver difference erased")
    require(exceptions[7]["status"] == "LATER_EXPLANATION_COMPLETE_ARTWORK_UNCHANGED", "A82 stale exception")
    claims = {x["id"]: x for x in c(OLD + "claim_evidence_matrix.csv")}
    ledger = {x["evidence_row_id"]: x for x in c(OLD + "evidence_ledger_v3.csv")}
    links = c(B + "evidence_linkage.csv")
    require(len(links) == 6 and len({x["comparison_id"] for x in links}) == 6, "linkage duplication")
    for link, row in zip(links, rows):
        require(link["comparison_id"] == row["comparison_id"] and link["cohort"] == row["cohort"], "linkage cohort mismatch")
        require(link["CEM_id"] in claims, "unknown CEM claim")
        require(link["result_commit"] == row["result_commit"] and link["science_SOURCE"] == row["SOURCE"], "linkage SOURCE mismatch")
        for key, value in [("ledger_mean_id", "G_PR_exact"), ("ledger_SE_id", "paired_SE"), ("ledger_PF_id", "G_PF_exact")]:
            require(ledger[link[key]]["value_literal"] == str(row[value]), "ledger/table value mismatch")
        require(link["figure_alias"] == (row["mechanism_core_alias"] or "NOT_IN_ORIGINAL_FIGURE2"), "Figure2 same-cohort alias mismatch")
    remote = j(B + "public_read_receipt.json")
    require(len(remote["files"]) == 17 and remote["private_files_opened_or_published"] is False, "public receipt scope")
    labels = {x["label"] for x in remote["files"]}
    require(all(x["local_fixed_blob_matches_remote"] is True and x["read_extent"] == "FULL_FILE" for x in remote["files"]), "incomplete public read")
    require(all(x["public_source_label"] in labels for x in evidence.values()), "unknown public evidence reference")
    require(all(x["public_source_label"] in labels | {"BASE_ACCEPTED_MAIN_TABLE"} for x in exceptions), "unknown input exception reference")
    docs_ledger = c(B + "documentation_evidence_ledger.csv")
    require(len(docs_ledger) == 17, "documentation ledger inventory")
    for receipt in remote["files"]:
        record = next(x for x in docs_ledger if x["evidence_id"] == receipt["label"])
        require(record["sha256"] == receipt["sha256"] and record["git_blob_sha"] == receipt["git_blob_sha"], "documentation ledger SOURCE binding")
    stop = j(B + "MANDATORY_STOP.json")
    require(stop["status"] == "MANDATORY_STOP" and stop["new_grant_issued"] is False and stop["next_scientific_stage_authorized"] is False and stop["final_evidence_freeze"] is False, "STOP boundary changed")
    return {"status": "PASS_DOCUMENT_STRUCTURE_ONLY_NOT_INDEPENDENT_SCIENTIFIC_REVIEW", "base_commit": BASE,
            "protected_files": len(preservation["files"]), "original_Figure2_files": 34, "main_table_rows": 6,
            "public_files_read": 17, "input_exceptions": 8, "H8_evidence_items": 14, "CEM_ledger_linkages": 6,
            "new_scientific_calls": 0, "new_numerical_analysis": False, "final_evidence_freeze": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="store_true", help="save new documentation-only structural receipt")
    args = parser.parse_args()
    result = json.dumps(check(), ensure_ascii=False, indent=2) + "\n"
    if args.receipt:
        (ROOT / B / "structural_validation.json").write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()

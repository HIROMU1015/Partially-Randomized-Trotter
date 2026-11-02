#!/usr/bin/env python3
"""Transcribe fixed public evidence; never import or execute scientific runners.

JSON decimal tokens are read as strings. No fractions, statistics, shot accounting,
or plotting are evaluated. --build writes only the new integration directory;
--check checks transcriptions, lineage, exclusions, and preserved upstream bytes.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "artifacts/resource_applicability/"
BUNDLE = BASE + "track_a_h468_paper_evidence_integration/2026-10-19/"
H4 = BASE + "track_a_h4_tol_p_completed/2026-10-19/"
H6 = BASE + "track_a_h6_pr_confirmation_science/2026-10-19/"
ADOPT = BASE + "track_a_h6_pr_confirmation_adoption/2026-10-19/"
H8 = BASE + "track_a_h8_new_pr_confirmation_results/2026-10-19/"
MECH = BASE + "track_a_mechanism_a82_factor_results/2026-10-19/"
FIG = BASE + "track_a_mechanism_figure2_a82_readiness/2026-10-19/"
H8_REPORT = "docs/research/track_a_h8_new_pr_confirmation_results_20261019.md"
LIMITS = ("FINITE_SAMPLE_POINT;NO_POPULATION_MEAN_GUARANTEE;UNOBSERVED_RARE_COST_UNKNOWN;"
          "NO_COHORT_POOLING;EMPIRICAL_BIAS_ALLOWANCE;ANALYTIC_SHOTS;COMPILED_RZ_NOT_FT;"
          "P0;NO_GLOBAL_OPTIMUM;INPUT_RULE_EQUIVALENCE_NOT_VERIFIED;NOT_FINAL_FREEZE")
H8_AUDIT = "REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS"


def read_json(path):
    return json.loads((ROOT / path).read_text(), parse_float=str)


def read_csv(path):
    with (ROOT / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def json_text(data):
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def pointer(data, path):
    for part in path.strip("/").split("/"):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def literal(value):
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def make_rows():
    receipt = read_json(BUNDLE + "github_verification.json")
    commits = {r["label"]: r["sha"] for r in receipt["commits"]}
    rows = []
    p4, p6 = read_json(H4 + "result.json"), read_json(H6 + "result.json")
    projection = read_json(H4 + "comparison_projection.json")["rows"]
    old_manifest = read_csv(ADOPT + "paper_data_manifest.csv")
    old_by_id = {r["record_id"]: r for r in old_manifest}
    core = {r["contrast_id"]: r for r in read_csv(MECH + "data/main_table.csv")}
    for system, items in [("H4", p4["comparisons"]), ("H6", p6["scopes"])]:
        for i, item in enumerate(items):
            h4 = system == "H4"
            source = H4 + "result.json" if h4 else H6 + "result.json"
            prefix = f"/comparisons/{i}" if h4 else f"/scopes/{i}"
            prim = item["result"] if h4 else item["primary"]
            fields = {"epsilon": prefix + "/epsilon", "PR_id": prefix + "/cell/id",
                      "n": prefix + ("/n" if h4 else "/primary/n"),
                      "G_PR_exact": prefix + ("/result/G_exact" if h4 else "/primary/mean_exact"),
                      "paired_SE": prefix + ("/result/paired_SE" if h4 else "/primary/paired_SE"),
                      "G_PF_exact": prefix + ("/PF_G_exact" if h4 else "/primary/fixed_PF_G_exact"),
                      "PF_id": prefix + ("/PF_id" if h4 else "/fixed_PF_id"),
                      "saved_direction_status": prefix + ("/result/status" if h4 else "/primary/reproduction_status")}
            data = p4 if h4 else p6
            row = {k: pointer(data, v) for k, v in fields.items()}
            row.update({"comparison_id": f"{system}_DF{10 if h4 else 19}_P_{i+1}",
                        "system": system, "geometry": "linear_1Angstrom", "basis": "STO-3G",
                        "DF_policy": "tol_only_1e-8", "DF_rank": 10 if h4 else 19,
                        "T": "0.8", "P": 0, "L_D": item["cell"]["prefix"],
                        "delta": ["0.8", "0.1"][i] if h4 else ["0.4", "0.1"][i],
                        "q": item["cell"]["q"], "r": item["cell"]["r"], "K": item["cell"]["K"],
                        "bias_mode": "USUAL_SIGNED", "allowance_factor": "1",
                        "cohort": "INDEPENDENT_H4_CONFIRMATION" if h4 else prim["cohort"],
                        "scope": p4["scope"] if h4 else item["scope"],
                        "lower_saved_point_cost": ("PF" if i == 0 else "PR"),
                        "PR_CHEAPER": i == 1,
                        "saved_reduction_percent_exact": None if h4 else prim["PR_reduction_percent_exact"],
                        "saved_PF_minus_PR_exact": projection[i]["PF_minus_PR_exact"] if h4 else None,
                        "ratio_status": "NOT_SAVED_NO_NEW_RATIO" if h4 else "SAVED_REDUCTION_ONLY_NO_NEW_DECOMPOSITION",
                        "PF_cost_class": "DIRECT_MEASURED", "PR_cost_class": "PAIRED_TRAJECTORY_MEASURED_RZ_ANALYTIC_SHOTS",
                        "event_order_counts": projection[i]["RTE_event_order_counts"] if h4 else item["event_order_counts"],
                        "trajectory_strata": item["strata"] if h4 else None,
                        "rare_event_source": H4 + "comparison_projection.json" if h4 else source,
                        "rare_event_locator": f"/rows/{i}/RTE_event_order_counts" if h4 else prefix + "/event_order_counts",
                        "audit_status": "PUBLIC_COMPLETION_RECEIPTS_VERIFIED",
                        "population_mean_certified": False, "unseen_cost": None, "unseen_variance": None,
                        "cohorts_pooled": False, "mechanism_core_alias": None,
                        "mechanism_for_this_cohort": "NOT_COMPUTED_NO_TRANSFER_FROM_OLD32",
                        "result_commit": commits[system + "_P" if h4 else "H6_confirmation"],
                        "SOURCE": data["SOURCE"], "source_file": source, "source_sha256": sha(source),
                        "source_record": prefix, "field_pointers": fields,
                        "interpretation_limits": LIMITS})
            rows.append(row)
    h8rows = [(i, r) for i, r in enumerate(read_csv(H8 + "comparisons.csv"), 1) if r["mode"] == "primary"]
    for i, (record, item) in enumerate(h8rows):
        old = old_by_id[["C1_CONFIRMATION_PR", "C3_CONFIRMATION_PR"][i]]
        alias = ["H8_CONFIRM_1", "H8_CONFIRM_3"][i]
        fields = {"epsilon": "epsilon", "PR_id": "candidate", "n": "n", "G_PR_exact": "mean_exact",
                  "paired_SE": "paired_SE", "G_PF_exact": "fixed_PF_G_exact", "PF_id": "fixed_PF_id",
                  "saved_direction_status": "point_decision", "saved_reduction_percent_exact": "PR_reduction_percent_exact"}
        row = {k: item[v] for k, v in fields.items()}
        row.update({"comparison_id": f"H8_DF28_P_{i+1}", "system": "H8", "geometry": "linear_1Angstrom",
                    "basis": "STO-3G", "DF_policy": "tol_only_1e-8", "DF_rank": 28, "T": "0.8", "P": 0,
                    "L_D": old["p"], "delta": ["0.4", "0.05"][i], "q": old["q"], "r": old["r"], "K": old["K"],
                    "bias_mode": "USUAL_SIGNED", "allowance_factor": "1", "cohort": old["PR_cohort"], "scope": item["scope"],
                    "lower_saved_point_cost": "PR", "PR_CHEAPER": True, "saved_PF_minus_PR_exact": None,
                    "ratio_status": "EXISTING_CORE_SAME_COHORT_ALIAS", "PF_cost_class": item["PF_price_evidence"],
                    "PR_cost_class": "PAIRED_TRAJECTORY_MEASURED_RZ_ANALYTIC_SHOTS",
                    "event_order_counts": {"0": [384, 3072][i], "2": 0, "4": 0},
                    "trajectory_strata": None, "rare_event_source": H8_REPORT,
                    "rare_event_locator": "観測eventはp8でTaylor0が384、p7で3072、Taylor2/4はいずれも0だった。",
                    "audit_status": H8_AUDIT, "population_mean_certified": False,
                    "unseen_cost": None, "unseen_variance": None, "cohorts_pooled": False,
                    "mechanism_core_alias": alias, "mechanism_for_this_cohort": "EXISTING_SAME_COHORT_FACTORS_SAVED",
                    "result_commit": commits["H8_confirmation"], "SOURCE": old["science_SOURCE"],
                    "source_file": H8 + "comparisons.csv", "source_sha256": sha(H8 + "comparisons.csv"),
                    "source_record": record, "field_pointers": fields, "interpretation_limits": LIMITS,
                    "saved_core_ratios": {key: core[alias][key] for key in ["R_G", "R_N", "R_Ceff"]}})
        rows.append(row)
    return rows, commits


def render():
    rows, commits = make_rows()
    output = {}
    def put(name, value):
        output[name] = value if isinstance(value, str) else json_text(value)
    put("main_table_proposed.json", {"status": "PROPOSAL_NOT_FINAL_FREEZE", "rows": rows,
        "numeric_policy": "COPY_SAVED_TOKENS_ONLY_NO_NEW_STATISTICS_OR_RATIOS",
        "cohorts_pooled": False, "H8_already_in_mechanism32": True})
    table_fields = ["comparison_id", "system", "geometry", "basis", "DF_rank", "DF_policy", "T", "P", "epsilon", "L_D", "delta", "PR_id", "PF_id", "n", "G_PR_exact", "paired_SE", "G_PF_exact", "saved_direction_status", "lower_saved_point_cost", "saved_reduction_percent_exact", "PF_cost_class", "cohort", "audit_status", "mechanism_core_alias", "result_commit", "source_file", "source_record", "interpretation_limits"]
    put("main_table_proposed.csv", csv_text([{k: r.get(k) for k in table_fields} for r in rows]))
    md = "# 論文主表案：固定独立確認6条件\n\n保存値の転記のみ。費用単位はanalytic shots × compiled RZ。全行でlinear 1 Å/STO-3G/tol-only 1e-8、T=.8、P=0、通常signed bias・倍率1。SEは同一trajectory両軸の費用から保存された値。\n\n|系/rank|ε|L_D/δ|n|PR平均G（exact）|paired SE（保存）|PF G（exact）|方向再現|低費用|\n|---|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        md += f"|{r['system']}/{r['DF_rank']}|{r['epsilon']}|{r['L_D']}/{r['delta']}|{r['n']}|{r['G_PR_exact']}|{r['paired_SE']}|{r['G_PF_exact']}|{r['saved_direction_status']}|{r['lower_saved_point_cost']}|\n"
    md += "\n候補ID・PF対照・SOURCE・cohort・field単位出典は隣接CSV/JSONを参照。H6粗精度のREPRODUCEDはPF優位の再現。H8のPOINT_REPRODUCEDは公開報告どおりPR低費用の点順位再現。母平均の優位性、実power、CI、未観測費用は認定しない。H8の詳細完了監査は公開receiptで再検証できていない。H8の2行は既存Mechanism32内H8_CONFIRM_1/3と同一証拠であり追加独立証拠として数えない。H4 DF12 legacy・探索n8・保存FP n32は本表に混ぜない。新H4 Pの削減率は未保存のため空欄を維持する。\n"
    put("main_table_proposed.md", md)

    # Preserve every original row/cell, appending only four H4 method records.
    original = read_csv(ADOPT + "paper_data_manifest.csv")
    manifest_rows = [dict(r) for r in original]
    for i, row in enumerate(rows[:2], 1):
        for role in ["CONFIRMATION_PR", "FIXED_PF"]:
            pr = role == "CONFIRMATION_PR"
            r = dict.fromkeys(original[0], "null")
            r.update({"record_id": f"H4TOL10_P{i}_{role}", "record_type": "H4_TOL10_FIXED_CONFIRMATION",
                "panel": "PROPOSED_MAIN_P6", "system": "H4", "geometry": row["geometry"], "basis": row["basis"],
                "DF_policy": row["DF_policy"], "DF_rank": "10", "T": "0.8", "P": "0", "epsilon": row["epsilon"],
                "bias_mode": row["bias_mode"], "allowance_factor": "1", "alpha_total": "0.05", "alpha_axis": "0.025",
                "deterministic_comparison_scope": "D190_REGISTERED_BEST_FIXED_BEFORE_P",
                "PR_cohort": row["cohort"] if pr else "SAVED_DETERMINISTIC_FIXED_CONTROL",
                "sample_size": str(row["n"]) if pr else "NOT_APPLICABLE",
                "sample_role": "INDEPENDENT_FIXED_CANDIDATE_CONFIRMATION" if pr else "FIXED_DETERMINISTIC_CONTROL",
                "role": role, "candidate_id": row["PR_id"] if pr else row["PF_id"],
                "p": str(row["L_D"]) if pr else ["5", "7"][i-1], "q": str(row["q"]),
                "r": str(row["r"]) if pr else "null", "K": str(row["K"]) if pr else "null", "outer_PF_order": "2",
                "saved_cost_exact": row["G_PR_exact"] if pr else row["G_PF_exact"],
                "saved_SE": row["paired_SE"] if pr else "null", "cost_class": "DIRECT_MEASURED",
                "saved_cost_status": row["saved_direction_status"], "eligibility": "FIXED_CONTRACT_NOT_REEVALUATED",
                "missing": "REDUCTION_RATIO_NOT_SAVED", "unresolved": "POPULATION_RARE_SUPPORT_INPUT_RULE_EQUIVALENCE",
                "full_domain_state": "D190_FIXED_CONTROL_P_TWO_CELLS_ONLY", "registered_domain": "H4_TOL10_A50_D190",
                "boundary_context": "FIXED_CONFIRMATION_NOT_NEW_SEARCH;BOUNDARY_NOT_RECLASSIFIED",
                "result_commit": row["result_commit"], "source_file": row["source_file"],
                "source_record": row["source_record"], "source_field": row["field_pointers"]["G_PR_exact" if pr else "G_PF_exact"],
                "source_file_sha256": row["source_sha256"], "metadata_source_commit": row["result_commit"],
                "metadata_source_file": row["source_file"], "metadata_source_record": row["source_record"],
                "science_SOURCE": row["SOURCE"], "acquisition_SOURCE": row["SOURCE"] if pr else commits["H4_D_A_SOURCE"],
                "compiler": "Qiskit 1.3.0;rz/sx/x/cx;optimization_level=1;seed_transpiler=17;backend/coupling=None",
                "compiler_settings_hash": read_json(H4 + "result.json")["binding"]["compiler"],
                "profile_evidence_kind": "PUBLIC_P_BINDING_AND_D190_RESULT;NO_CROSS_SYSTEM_IDENTITY_ASSERTION",
                "figure_table_ids": "PROPOSED_MAIN_P6;FIGURE2_UNCHANGED", "interpretation_limits": LIMITS})
            manifest_rows.append({k: literal(v) if v is not None else "null" for k, v in r.items()})
    put("paper_data_manifest.csv", csv_text(manifest_rows))

    cem_original = read_csv(ADOPT + "claim_evidence_matrix.csv")
    cem = [dict(r) for r in cem_original]
    for row in rows[:2] + rows[4:]:
        r = dict.fromkeys(cem_original[0], "")
        r.update({"id": "CE_" + row["comparison_id"],
            "claim_candidate": f"{row['system']} ε={row['epsilon']}の固定独立標本では{row['lower_saved_point_cost']}が低費用。方向判定{row['saved_direction_status']}。",
            "fixed_commits": row["result_commit"], "result_files": row["source_file"], "SOURCE": row["SOURCE"],
            "conditions": f"{row['geometry']};{row['basis']};tol-only DF{row['DF_rank']};T=.8;P=0;L_D={row['L_D']};delta={row['delta']};n={row['n']};{row['PR_id']};PF={row['PF_id']}",
            "existing_value_quote": f"G_PR={row['G_PR_exact']};paired_SE={row['paired_SE']};G_PF={row['G_PF_exact']}",
            "measured_or_model": "PAIRED_MEASURED_RZ_AND_DIRECT_PF;ANALYTIC_SHOTS",
            "search_or_confirmation": row["scope"] + ";" + row["cohort"], "unresolved": LIMITS,
            "evidence_strength": "ADOPTED_FINITE_POINT_ONLY;" + row["audit_status"], "audit_links": row["rare_event_source"]})
        cem.append(r)
    current_mech = read_json(MECH + "claim_evidence_matrix_update_DRAFT.json")["rows"]
    for item, new_id in zip(current_mech[6:], ["CE_M_LEGACY_H4_P", "CE_M_A82_FACTORS"]):
        r = dict.fromkeys(cem_original[0], "")
        r.update({"id": new_id, "claim_candidate": item["proposed_claim"], "fixed_commits": commits["Mechanism"],
                  "result_files": MECH + "claim_evidence_matrix_update_DRAFT.json", "SOURCE": commits["Mechanism_FACTOR_SOURCE"],
                  "conditions": item["supported_scope"], "unresolved": item["unresolved"],
                  "evidence_strength": "SCOPED_PRIOR_EVIDENCE_NOT_TRANSFERRED_TO_NEW_H4_H6_P", "search_or_confirmation": "SEE_FIXED_COHORT_IN_SOURCE"})
        cem.append(r)
    put("claim_evidence_matrix.csv", csv_text(cem))
    put("claim_evidence_matrix_current_overlay.json", {"status": "SCOPED_INTEGRATION_NOT_FINAL_SCIENTIFIC_REVIEW",
        "historical_H6_matrix_rows_preserved": 8,
        "current_mechanism_claims": [{k: r[k] for k in ["id", "proposed_claim", "supported_scope", "unresolved", "status", "new_evidence"]} for r in current_mech],
        "H4_legacy_CE7_applies_only_to": "DF12_LEGACY_P_NOT_DF10_P",
        "new_adopted_point_claim_ids": ["CE_" + r["comparison_id"] for r in rows[:2] + rows[4:]],
        "H6_existing_point_claim_ids": ["CE_H6P_Q2", "CE_H6P_Q8"],
        "supersession_rule": "CE1-CE6 historic cells retained; overlay adds current scoped Mechanism evidence. New H4 DF10 result supersedes no DF12 numeric cells.",
        "universal_PR_advantage_claim": False, "scaling_claim": False, "population_mean_claim": False})

    ledger = read_csv(ADOPT + "evidence_ledger_v2.csv")
    for row in rows:
        for key, locator in row["field_pointers"].items():
            r = dict.fromkeys(ledger[0], "")
            r.update({"evidence_row_id": "INT_" + row["comparison_id"] + "_" + key,
                "study_id": row["scope"], "metric": key, "value_literal": literal(row[key]),
                "unit": "SAVED_LITERAL", "comparison_scope": row["cohort"], "baseline_denominator": row["PF_id"],
                "cost_evidence": row["PF_cost_class"], "scientific_guarantee": LIMITS,
                "evidence_id": row["comparison_id"], "source_line": str(row["source_record"]),
                "source_excerpt": "EXACT_SAVED_FIELD_TOKEN", "source_field": locator,
                "SOURCE_commit": row["SOURCE"], "result_commit": row["result_commit"],
                "evidence_path": row["source_file"], "evidence_file_sha256": row["source_sha256"],
                "reported_precision": "COPIED_NO_NUMERICAL_RECOMPUTATION"})
            ledger.append(r)
        r = dict.fromkeys(ledger[0], "")
        r.update({"evidence_row_id": "INT_" + row["comparison_id"] + "_RARE", "study_id": row["scope"],
                  "metric": "event_order_counts", "value_literal": literal(row["event_order_counts"]),
                  "unit": "SAVED_EVENT_COUNTS", "comparison_scope": row["cohort"], "scientific_guarantee": LIMITS,
                  "source_field": row["rare_event_locator"], "SOURCE_commit": row["SOURCE"],
                  "result_commit": row["result_commit"], "evidence_path": row["rare_event_source"],
                  "evidence_file_sha256": sha(row["rare_event_source"]), "reported_precision": "COPIED_NO_RAW_REAGGREGATION"})
        ledger.append(r)
    for identifier, path, field, commit, source in [
        ("H4_PUBLIC_COMPLETION", H4 + "completion_audit.json", "/status", commits["H4_P"], rows[0]["SOURCE"]),
        ("H6_PUBLIC_COMPLETION", H6 + "result.json", "/status", commits["H6_confirmation"], rows[2]["SOURCE"]),
        ("MECHANISM_COMPLETION", MECH + "paper_manifest.json", "/status", commits["Mechanism"], commits["Mechanism_FACTOR_SOURCE"]),
        ("H8_PUBLIC_AUDIT_GAP", H8_REPORT, "TEXT_CONTAINS", commits["H8_confirmation"], rows[4]["SOURCE"]),
    ]:
        value = "共通入口の追記は自動承認審査の公開拒否に従いローカルに保全し、この結果公開へ含めない。" if field == "TEXT_CONTAINS" else pointer(read_json(path), field)
        r = dict.fromkeys(ledger[0], "")
        r.update({"evidence_row_id": "INT_" + identifier, "study_id": identifier, "metric": "PUBLIC_EVIDENCE_STATUS_LITERAL",
                  "value_literal": value, "unit": "SAVED_LITERAL", "comparison_scope": "PUBLIC_RECEIPT_AVAILABILITY_NOT_NEW_REEXECUTION",
                  "scientific_guarantee": LIMITS, "source_field": field, "source_excerpt": value,
                  "SOURCE_commit": source, "result_commit": commit, "evidence_path": path,
                  "evidence_file_sha256": sha(path), "reported_precision": "COPIED_SAVED_STATUS_OR_REPORT_TEXT"})
        ledger.append(r)
    put("evidence_ledger_v3.csv", csv_text(ledger))

    status_rows = []
    for row in rows:
        status_rows.append({"comparison_id": row["comparison_id"], "point_adoption": "ADOPTED_FIXED_FINITE_POINT",
            "direction": row["saved_direction_status"], "lower_point_cost": row["lower_saved_point_cost"],
            "M_INPUT": "PARTIAL_INPUT_RULE_EQUIVALENCE_NOT_VERIFIED", "M_COST": "FIXED_COMPARISON_DIRECT_ONLY",
            "M_CORE_M_EXPL": row["mechanism_for_this_cohort"], "TECHNICAL_PUBLIC": row["audit_status"],
            "COMMON_CORE_READY": "NOT_ASSERTED", "population_mean": "NOT_CERTIFIED",
            "unobserved_rare_cost_variance": "UNKNOWN_NOT_ZERO", "original_profile_bytes": "NOT_VERIFIED" if row["system"] == "H8" else "SEE_FIXED_PUBLIC_BINDING_NOT_CROSS_SYSTEM_EQUIVALENCE"})
    put("protocol_scope_status.csv", csv_text(status_rows))
    put("adoption_record.json", {"status": "ADOPTED_SCOPED_PAPER_INTEGRATION_NOT_FINAL_FREEZE", "date": "2026-10-19",
        "user_instruction": "H4独立確認Pの完了結果を正式採用しH4/H6/H8の既存論文証拠を統合",
        "publication_authorization": "新規統合資料も専用branchへpushする",
        "H4_DF10_P": [{k: r[k] for k in ["comparison_id", "epsilon", "PR_id", "PF_id", "n", "saved_direction_status", "lower_saved_point_cost"]} for r in rows[:2]],
        "accepted_fixed_commits": {k: commits[k] for k in ["H4_D", "H4_A", "H4_S_STATS", "H4_P", "H6_confirmation", "H6_adoption", "H8_confirmation", "Mechanism", "Figure2", "Protocol"]},
        "new_scientific_calls": 0, "new_statistics": False, "raw_samples_reaggregated": False,
        "H2_included": False, "final_evidence_freeze": False, "detailed_scientific_review_started": False,
        "H8_public_audit": H8_AUDIT, "COMMON_CORE_READY": False})
    put("figure2_update_proposal.json", {"status": "LAYOUT_AND_CAPTION_PROPOSAL_ONLY", "original_commit": commits["Figure2"],
        "original_files_changed": False, "new_numeric_processing_run": False,
        "recommended_A": "Retain original as labelled legacy/exploratory mechanism panel; H4 is DF12, never relabel DF10.",
        "recommended_B": "Retain H8 independent n48 two comparisons; aliases H8_CONFIRM_1/3 already in existing32.",
        "new_main_table": "main_table_proposed.csv (six fixed comparisons, exact saved means/SE)",
        "caption_addendum": "A82 explanation completed at Mechanism commit; original annotation documents render-time state.",
        "if_new_confirmatory_A_requested": {"status": "REQUIRES_SEPARATE_NUMERICAL_AUTHORIZATION_NOT_RUN",
            "missing_comparison_ids": [r["comparison_id"] for r in rows[:4]],
            "minimum_scope": "Bind correct fixed P per-axis saved aggregates and PF shots/axes; separately approve any missing aggregate recovery and M_CORE/M_EXPL factor/ratio/log calculations. H4 G ratio/reduction is not saved. Do not transfer old n8/n32 factors or invent error bars.",
            "additional_sampling_or_compile": "NOT_IMPLIED_OR_AUTHORIZED",
            "H8": "Only copy existing same-cohort saved factors; public audit gap persists."},
        "review_decisions_needed": ["Main P6 table and legacy/exploration Figure2 layout", "Explicit H4/H6/H8 input solver/normalization exceptions", "Whether to authorize missing numerical factor work", "H8 public audit disclosure policy; do not publish restricted receipts"]})
    put("paper_manifest.json", {"schema": "H468_PAPER_EVIDENCE_INTEGRATION_V1", "status": "DOCUMENT_INTEGRATION_NOT_FINAL_FREEZE",
        "base_commit": commits["Mechanism"], "fixed_commits": commits,
        "paper_data_manifest": {"path": BUNDLE + "paper_data_manifest.csv", "rows": len(manifest_rows), "historical_prefix_rows": len(original), "appended_H4_DF10_method_rows": 4, "H6_H8_reused_not_duplicated": True},
        "claim_evidence_matrix": {"path": BUNDLE + "claim_evidence_matrix.csv", "rows": len(cem), "historical_prefix_rows": len(cem_original), "current_overlay": BUNDLE + "claim_evidence_matrix_current_overlay.json"},
        "evidence_ledger": {"path": BUNDLE + "evidence_ledger_v3.csv", "rows": len(ledger), "historical_prefix_rows": 256},
        "proposed_main_table": {"path": BUNDLE + "main_table_proposed.json", "comparisons": 6, "sampling_cohorts_pooled": False},
        "existing_mechanism": {"manifest": MECH + "paper_manifest.json", "comparisons": 32, "axis_factors": 64,
            "H8_overlap_aliases": {"H8_DF28_P_1": "H8_CONFIRM_1", "H8_DF28_P_2": "H8_CONFIRM_3"},
            "H4_H6_latest_P_factors": "NOT_COMPUTED_NOT_INHERITED", "adding_32_and_6_as_independent_evidence": False},
        "H4_legacy": {"DF_rank": 12, "result_commit": "8a18b277663bf2f8998f79e9b8c0a9e61d2b184c", "path": FIG + "data/H4_P_confirmation_separate.csv", "scope": "LEGACY14_NOT_DF10_P"},
        "H4_A50": "SAVED_FP_N32_AND_49_NEW_N8_SEPARATE_NO_POOLING", "H4_D190": "FIVE_SAVED_WINNERS_REUSED_NO_RESELECTION",
        "S_STATS": "SAVED_PLANNING_NOT_POPULATION_OR_ACTUAL_POWER_GUARANTEE", "original_Figure2": "UNCHANGED_34_FILES",
        "protocol": "ADOPTED_V1_DOCUMENT_ONLY;SCOPE_GATES_NOT_AUTOMATIC_COMMON_CORE_READY",
        "protocol_status": BUNDLE + "protocol_scope_status.csv", "H8_public_audit": H8_AUDIT,
        "upstream_preservation": BUNDLE + "upstream_preservation.json", "github_verification": BUNDLE + "github_verification.json",
        "science_execution": False, "new_statistics": False, "final_evidence_freeze": False})
    put("MANDATORY_STOP.json", {"status": "MANDATORY_STOP", "stage": "H468_DOCUMENT_EVIDENCE_INTEGRATION",
        "new_scientific_calls": 0, "new_statistics": False, "grant_issued": False,
        "next_scientific_stage_authorized": False, "final_evidence_freeze": False,
        "next_user_decisions": "Layout/input exceptions/H8 audit availability; any numerical figure update requires separate scope and authorization"})
    return output


def validate(root=ROOT):
    """Structural checks only. Never read trajectory raw costs or recompute a mean."""
    root = Path(root)
    def j(path):
        return json.loads((root / path).read_text(), parse_float=str)
    def c(path):
        with (root / path).open(newline="") as stream:
            return list(csv.DictReader(stream))
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    protected = j(BUNDLE + "upstream_preservation.json")["files"]
    for item in protected:
        require(hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"], "upstream bytes changed: " + item["path"])
    remote = j(BUNDLE + "github_verification.json")
    require(len(remote["commits"]) == 19 and len(remote["files"]) == 47, "remote verification coverage")
    require(not remote["source_tree_truncated"] and all(r["verified"] for r in remote["source_checks"]), "SOURCE verification")
    for item in remote["files"]:
        require(item["remote_blob_matches_fixed_local_object"], "remote blob mismatch")
    require(len(remote["figure2_remote_tree"]) == 34, "Figure2 file inventory")
    table = j(BUNDLE + "main_table_proposed.json")
    rows = table["rows"]
    require(len(rows) == 6 and len({r["comparison_id"] for r in rows}) == 6, "six unique fixed comparisons")
    expected = [("H4", 10, 1108, "NOT_REPRODUCED", "PF"), ("H4", 10, 78, "REPRODUCED", "PR"),
                ("H6", 19, 48, "REPRODUCED", "PF"), ("H6", 19, 104, "REPRODUCED", "PR"),
                ("H8", 28, "48", "POINT_REPRODUCED", "PR"), ("H8", 28, "48", "POINT_REPRODUCED", "PR")]
    for row, exp in zip(rows, expected):
        require(tuple(row[k] for k in ["system", "DF_rank", "n", "saved_direction_status", "lower_saved_point_cost"]) == exp, "fixed cohort/rank/direction mismatch")
        require(row["DF_policy"] == "tol_only_1e-8" and not row["cohorts_pooled"], "cohort pooling or input drift")
        require(not row["population_mean_certified"] and row["unseen_cost"] is None and row["unseen_variance"] is None, "population or zero-fill claim")
        require(row["PF_cost_class"] == "DIRECT_MEASURED", "primary PF evidence class")
        require(hashlib.sha256((root / row["source_file"]).read_bytes()).hexdigest() == row["source_sha256"], "table source hash")
        source = c(row["source_file"])[row["source_record"]-1] if row["system"] == "H8" else j(row["source_file"])
        for key, locator in row["field_pointers"].items():
            value = source[locator] if row["system"] == "H8" else pointer(source, locator)
            require(value == row[key], "saved token changed: " + key)
        if row["system"] == "H4":
            require(row["saved_reduction_percent_exact"] is None and row["ratio_status"] == "NOT_SAVED_NO_NEW_RATIO", "new H4 ratio forbidden")
            winners = j(BASE + "track_a_h4_tol_d_completed/2026-10-19/result.json")["winners"]
            saved = next(x["best"] for x in winners if x["epsilon"] == row["epsilon"])
            require(saved["cell_id"] == row["PF_id"] and saved["G_RZ_exact"] == row["G_PF_exact"], "D190 PF control changed")
        if row["system"] == "H8":
            require(row["audit_status"] == H8_AUDIT, "H8 public audit gap erased")
            core = {x["contrast_id"]: x for x in c(MECH + "data/main_table.csv")}[row["mechanism_core_alias"]]
            require(core["G_PR"] == row["G_PR_exact"] and core["G_PF"] == row["G_PF_exact"] and core["saved_SE"] == row["paired_SE"], "H8 same-cohort alias mismatch")
            require(all(core[k] == v for k, v in row["saved_core_ratios"].items()), "existing H8 factors changed")
        else:
            require(row["mechanism_for_this_cohort"] == "NOT_COMPUTED_NO_TRANSFER_FROM_OLD32", "mechanism cohort transfer")
        if row["system"] == "H8":
            require(row["rare_event_locator"] in (root / H8_REPORT).read_text(), "H8 rare observation quote missing")
        else:
            require(pointer(j(row["rare_event_source"]), row["rare_event_locator"]) == row["event_order_counts"], "rare event counts changed")
    for new, old, count, key in [("paper_data_manifest.csv", "paper_data_manifest.csv", 1572, "record_id"),
                                 ("claim_evidence_matrix.csv", "claim_evidence_matrix.csv", 8, "id"),
                                 ("evidence_ledger_v3.csv", "evidence_ledger_v2.csv", 256, "evidence_row_id")]:
        n, o = c(BUNDLE + new), c(ADOPT + old)
        require(n[:count] == o and len(o) == count, "historical prefix changed: " + new)
        require(len({r[key] for r in n}) == len(n), "duplicate identifier: " + new)
    require(len(c(BUNDLE + "paper_data_manifest.csv")) == 1576, "duplicate H6/H8 or missing H4 method records")
    ledger = c(BUNDLE + "evidence_ledger_v3.csv")[256:]
    for r in ledger:
        require(hashlib.sha256((root / r["evidence_path"]).read_bytes()).hexdigest() == r["evidence_file_sha256"], "ledger source hash")
        if r["metric"] != "event_order_counts":
            if r["source_field"] == "TEXT_CONTAINS":
                require(r["value_literal"] in (root / r["evidence_path"]).read_text(), "ledger report quote changed")
                continue
            source = c(r["evidence_path"])[int(r["source_line"])-1] if r["evidence_path"].endswith(".csv") else j(r["evidence_path"])
            value = source[r["source_field"]] if isinstance(source, dict) and not r["source_field"].startswith("/") else pointer(source, r["source_field"])
            require(literal(value) == r["value_literal"], "ledger value not exact saved token")
    require(all(r["COMMON_CORE_READY"] == "NOT_ASSERTED" for r in c(BUNDLE + "protocol_scope_status.csv")), "unjustified protocol gate")
    for name in ["adoption_record.json", "paper_manifest.json", "MANDATORY_STOP.json"]:
        require(j(BUNDLE + name)["final_evidence_freeze"] is False, "final freeze forbidden")
    require(j(BUNDLE + "adoption_record.json")["H2_included"] is False, "H2 mixing forbidden")
    proposal = j(BUNDLE + "figure2_update_proposal.json")
    require(proposal["original_files_changed"] is False and proposal["new_numeric_processing_run"] is False, "Figure2 numeric work forbidden")
    # Deterministic outputs must match all fields, not only the cells above.
    for name, text in render().items():
        require((root / BUNDLE / name).read_text() == text, "transcription output drift: " + name)
    return {"status": "PASS_STRUCTURAL_ONLY_NOT_INDEPENDENT_SCIENTIFIC_REVIEW", "protected_files": len(protected),
            "remote_commits": 19, "remote_files": 47, "SOURCE_hash_checks": len(remote["source_checks"]),
            "Figure2_protected_files": 34, "main_table_rows": 6, "paper_manifest_rows": 1576,
            "claim_rows": len(c(BUNDLE + "claim_evidence_matrix.csv")), "ledger_rows": len(c(BUNDLE + "evidence_ledger_v3.csv")),
            "new_scientific_calls": 0, "new_statistics": False, "final_evidence_freeze": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="write new transcription bundle only")
    parser.add_argument("--check", action="store_true", help="read-only structural check")
    args = parser.parse_args()
    if not args.build and not args.check:
        parser.error("choose --build or --check")
    if args.build:
        for name, text in render().items():
            (ROOT / BUNDLE / name).write_text(text)
    result = validate()
    if args.build:
        (ROOT / BUNDLE / "structural_validation.json").write_text(json_text(result))
    print(json_text(result), end="")


if __name__ == "__main__":
    main()

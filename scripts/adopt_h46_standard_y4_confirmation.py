#!/usr/bin/env python3
"""Adopt fixed public evidence by literal transcription; no scientific imports.

Uses the earlier paper integration's token/CSV/pointer helpers. Fixed Git blobs
are inputs; --build writes only this adoption bundle. --check is read-only.
No numerical result, interval, shot count, rank, or statistic is evaluated here.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import subprocess

from integrate_h468_paper_evidence import csv_text, json_text, literal, pointer

ROOT = Path(__file__).resolve().parents[1]
BASE = "artifacts/resource_applicability/"
OUT = BASE + "track_a_h46_standard_y4_adoption/2026-10-24/"
RESULT = "4d117041469fb8eebf2ad55405c0f7d1b70dea7d"
SCIENCE = "a2a53698a72da7280d217adc7931938121ed732d"
NUMERICS = "c028f1c2885afdba65ef8c62c075738aadd7323d"
S2 = "3eca46148070ba4d4b0f5815ef45943407cc959e"
FAILED = "4ce444eb5b7cd1dad7cf61f149aa182949ad60a1"
BATCH = "eebdbfcbd0d5f5608d50bfd91c9cd526c68f0d23"
CNG = "d2536c4f959bfb4edd7d2deeceb155585fb53eb0"
PAPER = "f75a7c25d584b69d90de454731a7f343a421e059"
THREE = "6dafed86cc82149d55b4e6395382953d5f3631d7"
DISPLAY = "1923fce8f7461513d1bc47b5061e87cd70917e87"
REC = BASE + "track_a_h46_standard_y4_confirmation_recovery/2026-10-24/"
PRIOR = BASE + "track_a_h246_pf_common_adopted/2026-10-22/"
RUN = BASE + "track_a_h246_reversal_batch_run01/2026-10-24/"
THREE_PATH = "docs/data_collection/2026-10-23_three_environment_integration/"
STATUS = "FORMALLY_ADOPTED_CONDITIONAL_INDEPENDENT_COST_DIRECTION"
CLAIMS = ["CE_H4_STANDARD_Y4_FIXED_SHOT_DIRECTION", "CE_H6_STANDARD_Y4_FIXED_SHOT_DIRECTION"]
LIMITS = [
    "CONDITIONAL_ON_SAVED_SIGNAL_REFERENCE_EMPIRICAL_ALLOWANCE_ANALYTIC_SHOTS",
    "NORMAL_BONFERRONI_APPROXIMATE_NOT_RIGOROUS_COVERAGE",
    "NO_POPULATION_MEAN_RANK_CERTIFICATE_OR_RARE_TAIL_BOUND",
    "FINITE_SAMPLES_AND_POST_SELECTION_LIMITATIONS",
    "DIFFERENT_Q_FIXED_CONFIGURATIONS_NOT_PURE_ORDER_CAUSAL_EFFECT",
    "SAME_Q_CONTROLS_SHOW_NO_FULL_PR_REVERSAL",
    "NO_GLOBAL_OPTIMUM_GENERALIZATION_OR_UNIVERSAL_PR_ADVANTAGE",
    "COMPILED_RZ_NOT_END_TO_END_FAULT_TOLERANT_COST",
    "NOT_EXTERNAL_RESEARCHER_OR_FACILITY_REPLICATION",
    "NOT_FINAL_PAPER_CLAIM_NOVELTY_OR_RESEARCH_GO",
]
ZERO_CALLS = dict.fromkeys([
    "sampling", "circuit_generation", "compile", "simulation", "Hamiltonian_reference_signal",
    "analytic_shots", "G_ratio_difference_variance_SE_CI_reanalysis", "ranking",
    "candidate_search", "scientific_grants_issued", "scientific_grants_consumed",
    "old_SOURCE_execution",
], 0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


@lru_cache(maxsize=None)
def git_bytes(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_saved(raw):
    return json.loads(raw, parse_float=str)


def csv_saved(raw):
    return list(csv.DictReader(io.StringIO(raw.decode())))


def append_csv(raw, rows):
    """Keep the entire historical byte prefix, including its original header."""
    fields = next(csv.reader(io.StringIO(raw.decode())))
    ordered = [{k: row.get(k, "") for k in fields} for row in rows]
    require(raw.endswith(b"\n"), "historical CSV lacks final newline")
    return raw + csv_text(ordered).split("\n", 1)[1].encode()


def specifications():
    specs = {"result_" + n: (RESULT, REC + n) for n in [
        "REPORT.md", "results.json", "independent_groups_4.csv", "independent_comparisons_2.csv",
        "quoted_full_and_exploration_8.csv", "execution_contract.json", "end_audit.json", "STOP.json"]}
    specs.update({"prior_" + n: (PAPER, PRIOR + n) for n in [
        "paper_manifest.json", "paper_data_manifest.csv", "claim_evidence_matrix_v2.csv",
        "evidence_ledger_v4.csv", "source_catalog.json"]})
    specs.update({"batch_" + n: (BATCH, RUN + n) for n in [
        "primary_directions_15.csv", "primary_comparisons_30.csv", "same_q_comparisons_8.csv",
        "same_q_directions_4.csv", "REPORT.md"]})
    specs.update({"three_" + n: (THREE, THREE_PATH + n) for n in [
        "README.md", "claim_evidence_overlay.csv", "research_index.csv"]})
    for n in ["REPORT.md", "end_audit.json", "STOP.json"]:
        specs["failed_" + n] = (FAILED, BASE + "track_a_h46_standard_y4_confirmation_stop/2026-10-24/" + n)
    specs["old_H6_adoption"] = (RESULT, "docs/research/track_a_h6_pr_confirmation_adoption_20261019.md")
    specs["S2_original_result"] = (S2, BASE + "track_a_h246_reconstructed_confirmation/2026-10-22/fixed_evaluation/results.json")
    specs["CNG_REPORT"] = (CNG, BASE + "track_a_h246_order_cost_analysis_run01/2026-10-24/REPORT.md")
    return specs


def source_catalog():
    refs = {}
    for key, (commit, path) in specifications().items():
        raw = git_bytes(commit, path)
        refs[key] = {"commit": commit, "path": path, "sha256": digest(raw),
                     "url": f"https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/{commit}/{path}"}
    return refs


def check_source_joins(data, groups, comparisons, contract, controls):
    require(data["SOURCE"] == contract["SOURCE"] == SCIENCE, "wrong science SOURCE")
    require(data["input_manifest_sha256"] == contract["input_manifest_sha256"], "input binding")
    require(data["package_sha256"] == contract["contract_sha256"], "contract binding")
    require(contract["scientific_numerics_SOURCE"] == NUMERICS, "numeric SOURCE")
    require(contract["reused_S2_result_commit"] == S2, "S2 original cohort")
    require(data["cohort_pooling"] is False and data["new_shots"] == 0, "pooling or new shots")
    require(len(groups) == 4 and len(comparisons) == 2 and len(controls) == 4, "incomplete fixed family")
    for system in ["H4", "H6"]:
        for formula, name in [("2nd", "S2"), ("4th", "Y4")]:
            g, = [g for g in groups if g["system"] == system and g["formula"] == formula]
            saved = data["results"][system]["groups"][name]
            expected = {"candidate": saved["candidate"], "q": str(saved["q"]),
                        "n": str(saved["statistics"]["n"]), "G_mean": saved["statistics"]["mean"],
                        "G_SE": saved["statistics"]["SE"], "epsilon": "0.001", "cohort": "CONFIRMATION",
                        "prefix": str(contract["groups"][system]["prefix"]),
                        "origin": "REUSED_INDEPENDENT_S2" if name == "S2" else "NEW_INDEPENDENT_STANDARD_Y4",
                        "SOURCE": NUMERICS if name == "S2" else SCIENCE}
            require(all(g[k] == v for k, v in expected.items()), "group token/cohort/epsilon/q/prefix mismatch")
        c, = [r for r in comparisons if r["system"] == system]
        saved = data["results"][system]["comparison"]
        for field, key in [("difference_S2_minus_Y4", "difference"), ("difference_SE", "difference_SE"),
                           ("ratio_Y4_over_S2", "ratio_Y4_over_S2"), ("decision", "decision")]:
            require(c[field] == saved[key], "comparison token mismatch")
        require([c["approximate_interval_low"], c["approximate_interval_high"]] == saved["approximate_interval"], "interval token mismatch")
        require(c["family_size"] == "2" and c["family_alpha"] == "0.05", "family rule")
        require(c["interval_is_rigorous"] == "False" and c["rare_tail_bound"] == "", "guarantee escalation")
        require(saved["decision"] == "REPRODUCED_APPROXIMATE" and saved["population_rank"] == "NOT_CERTIFIED", "saved decision changed")
        for q, direction in [("4", "Y4_LOWER"), ("8", "S2_LOWER")]:
            ctl, = [r for r in controls if r["system"] == system and r["q"] == q]
            require(ctl["epsilon"] == "0.001" and ctl["Full_lower"] == ctl["PR_lower"] == direction
                    and ctl["reversal"] == "False", "same-q negative control changed")


def render():
    refs = source_catalog()
    raw = {k: git_bytes(r["commit"], r["path"]) for k, r in refs.items()}
    data = json_saved(raw["result_results.json"])
    contract = json_saved(raw["result_execution_contract.json"])
    groups = csv_saved(raw["result_independent_groups_4.csv"])
    comparisons = csv_saved(raw["result_independent_comparisons_2.csv"])
    controls = csv_saved(raw["batch_same_q_directions_4.csv"])
    check_source_joins(data, groups, comparisons, contract, controls)
    files = {}

    def put(name, value):
        files[name] = value if isinstance(value, bytes) else json_text(value).encode()

    limits = ";".join(LIMITS)
    conditions = ("linear H4/H6; adjacent1.0Angstrom;STO-3G;tol-only DF1e-8;rank10/19;T=.8;"
                  "epsilon_signal=.001;alpha_axis=.025;P=0;USUAL_SIGNED;empirical allowance factor1;"
                  "saved reference/signal/analytic shots;symmetric-directional;full cos/sin Hadamard wrapper;"
                  "without state preparation;compiled RZ;PR K4 ABS_TIME_PER_OUTER32;prefix3/5;S2 q8/Y4 q4")
    adopted = []
    new_cem = []
    for system, claim in zip(["H4", "H6"], CLAIMS):
        selected = [r for r in groups if r["system"] == system]
        comp, = [r for r in comparisons if r["system"] == system]
        adopted.append({"claim_id": claim, "system": system, "groups_saved": selected,
                        "comparison_saved": comp, "result_saved": data["results"][system],
                        "fixed_group_binding": contract["groups"][system],
                        "Full_and_exploration_saved": [r for r in data["quoted_saved_endpoints"] if r["system"] == system],
                        "same_q_controls_saved": [r for r in controls if r["system"] == system]})
        new_cem.append(dict(id=claim,
            claim_candidate=f"{system}・ε=.001の固定構成では独立PR費用標本でも標準Yoshida4/q4がS2/q8より低費用となる方向を再現した。保存FullはS2/q8低費用。同一q対照ではFull/PR逆転なし。",
            fixed_commits=";".join([RESULT, S2, BATCH]),
            result_files=refs["result_results.json"]["url"] + "#/results/" + system,
            SOURCE=SCIENCE + ";" + NUMERICS,
            conditions=conditions + ";" + literal(contract["groups"][system]),
            existing_value_quote=literal({"groups": selected, "comparison": comp}),
            measured_or_model="INDEPENDENT_PAIRED_COMPILED_RZ_PRICES_TIMES_FIXED_SAVED_ANALYTIC_SHOTS",
            search_or_confirmation="INDEPENDENT_S2_REUSED_AND_NEW_STANDARD_Y4;EXPLORATION_SEPARATE;NO_POOLING",
            unresolved=limits, evidence_strength=STATUS + ";SAVED_DECISION_REPRODUCED_APPROXIMATE",
            audit_links=";".join(refs[k]["url"] for k in ["result_end_audit.json", "result_execution_contract.json", "batch_same_q_directions_4.csv"])))

    record = {"schema": "H46_STANDARD_Y4_CONDITIONAL_ADOPTION_V1", "date": "2026-10-24", "status": STATUS,
              "authority": {"kind": "CURRENT_USER_INSTRUCTION", "historical_adoption_receipt_created": False,
                            "scope": "H4/H6標準Yoshida4独立確認結果を、固定shot政策下の条件付き費用方向再現として正式採用し研究証拠へ統合する"},
              "SOURCE": SCIENCE, "numeric_SOURCE": NUMERICS, "result_commit": RESULT,
              "input_manifest_sha256": data["input_manifest_sha256"], "contract_sha256": data["package_sha256"],
              "conditions": conditions, "compiler_profile": "Qiskit1.3.0;rz/sx/x/cx;opt1;seed17",
              "family_policy_saved": data["policy"], "family_size_is_not_trajectory_count": True,
              "groups_and_comparisons_saved": adopted, "new_Y4_trajectories": {"H4": 45, "H6": 32, "total_acquisition_only": 77},
              "reused_S2_trajectories": {"H4": 25, "H6": 23, "total_acquisition_only": 48},
              "cohort_pooling": False, "independence_scope": "INDEPENDENT_SAMPLING_COHORTS_NOT_EXTERNAL_REPLICATION",
              "saved_result_decision": "REPRODUCED_APPROXIMATE", "adopted_claim_ids": CLAIMS,
              "not_adopted": LIMITS, "rare_tail_bound": None, "ratio_interval": None,
              "failed_attempt_result_commit": FAILED, "success_is_separate_recovery": True,
              "old_H6_adoption_is_different_S2_K2_scope": refs["old_H6_adoption"],
              "m5_best_confirmation_transferred_to_standard_Y4": False,
              "new_scientific_calls": ZERO_CALLS, "scientific_execution_authorized": False,
              "final_evidence_freeze": False, "detailed_scientific_review_started": False,
              "old_strict_COMPARE": "BLOCKED", "mandatory_STOP": True}
    put("adoption_record.json", record)
    put("claim_evidence_matrix_v3.csv", append_csv(raw["prior_claim_evidence_matrix_v2.csv"], new_cem))

    # Each immutable source/pointer is one evidence unit. Roles reuse these IDs.
    ledger, roles = [], []
    def evidence(identifier, source_key, field, role, source=SCIENCE):
        ref = refs[source_key]
        obj = json_saved(raw[source_key]) if source_key.endswith(".json") else None
        value = (obj if field == "/" else pointer(obj, field)) if obj is not None else (
            csv_saved(raw[source_key]) if source_key.endswith(".csv") else "SEE_FIXED_REPORT")
        ledger.append(dict(evidence_row_id=identifier, study_id="H46_STANDARD_Y4_FIXED_SHOT_CONFIRMATION",
            metric=role, value_literal=literal(value), unit="SAVED_LITERAL", comparison_scope=conditions,
            baseline_denominator="S2_INDEPENDENT_FIXED_Q8;NOT_OLD_M5_BEST", cost_evidence="SAVED_RECEIPT_OR_STATISTIC",
            scientific_guarantee=limits, evidence_id=identifier, source_line="", source_excerpt="LITERAL_SAVED_VALUE_NO_REANALYSIS",
            source_field=field, SOURCE_commit=source, result_commit=ref["commit"], evidence_path=ref["url"],
            evidence_file_sha256=ref["sha256"], reported_precision="EXACT_SAVED_TOKENS"))
        roles.append({"role": role, "evidence_id": identifier, "source_id": source_key, "source_field": field,
                      "identity": ref["commit"] + ":" + ref["path"] + "#" + field,
                      "new_independent_sample_count_from_reference": "0"})
    for system in ["H4", "H6"]:
        for name in ["S2", "Y4"]:
            evidence(f"H46Y4_{system}_{name}", "result_results.json", f"/results/{system}/groups/{name}",
                     "REUSED_S2_GROUP" if name == "S2" else "NEW_STANDARD_Y4_GROUP", NUMERICS if name == "S2" else SCIENCE)
        evidence(f"H46Y4_{system}_COMPARISON", "result_results.json", f"/results/{system}/comparison", "DIFFERENCE_RATIO_APPROXIMATE_INTERVAL")
    for identifier, key, field, role, source in [
        ("H46Y4_POLICY", "result_results.json", "/policy", "FIXED_TWO_COMPARISON_FAMILY", NUMERICS),
        ("H46Y4_QUOTES", "result_results.json", "/quoted_saved_endpoints", "SAVED_FULL_AND_EXPLORATION_SEPARATE", BATCH),
        ("H46Y4_CONTRACT", "result_execution_contract.json", "/", "SOURCE_INPUT_CONTRACT_BINDING", SCIENCE),
        ("H46Y4_END", "result_end_audit.json", "/", "SUCCESS_ACQUISITION_AND_END_AUDIT", SCIENCE),
        ("H46Y4_STOP", "result_STOP.json", "/", "SUCCESS_MANDATORY_STOP", SCIENCE),
        ("H46Y4_FAILED_END", "failed_end_audit.json", "/", "SEPARATE_FAILED_ATTEMPT_AUDIT", "a94735946e8cf15ffea5c93ae99000f7ab83d23a"),
        ("H46Y4_FAILED_STOP", "failed_STOP.json", "/", "SEPARATE_FAILED_ATTEMPT_STOP", "a94735946e8cf15ffea5c93ae99000f7ab83d23a"),
        ("H46Y4_SAME_Q", "batch_same_q_directions_4.csv", "/", "SAME_Q_NEGATIVE_CONTROLS", BATCH),
    ]:
        evidence(identifier, key, field, role, source)
    put("evidence_ledger_v5.csv", append_csv(raw["prior_evidence_ledger_v4.csv"], ledger))
    put("evidence_roles.csv", csv_text(roles).encode())
    put("source_catalog.json", {"schema": "FIXED_SOURCE_REFERENCE_OVERLAY_V1", "sources": refs,
        "base_catalog": refs["prior_source_catalog.json"], "reuse_rule": "Commit/path/pointer identifies evidence; aliases and views add zero samples"})

    # Retain all original paper-manifest rows. Add reference records, not new data.
    new_data = [dict(record_id="H46Y4_" + k, record_type="CONDITIONAL_ADOPTION_SOURCE_REFERENCE",
                     path=r["url"], sha256=r["sha256"], result_commit=r["commit"],
                     scientific_values_recomputed="False", independent_confirmation="True" if k.startswith("result_") else "False")
                for k, r in refs.items() if k.startswith(("result_", "batch_same_q", "failed_"))]
    put("paper_data_manifest.csv", append_csv(raw["prior_paper_data_manifest.csv"], new_data))
    prior = json_saved(raw["prior_paper_manifest.json"])
    put("paper_manifest.json", {"schema": "H46_STANDARD_Y4_PAPER_EVIDENCE_ADOPTION_OVERLAY_V1", "status": STATUS,
        "parent_paper_manifest": refs["prior_paper_manifest.json"], "three_environment_parent": refs["three_README.md"],
        "claim_evidence_matrix": OUT + "claim_evidence_matrix_v3.csv", "evidence_ledger": OUT + "evidence_ledger_v5.csv",
        "paper_data_manifest": OUT + "paper_data_manifest.csv", "source_catalog": OUT + "source_catalog.json",
        "evidence_roles": OUT + "evidence_roles.csv", "adoption_record": OUT + "adoption_record.json",
        "three_environment_overlay": OUT + "three_environment_overlay.json", "adopted_claim_ids": CLAIMS,
        "historical_CEM_rows": 21, "new_CEM_rows": 2, "historical_ledger_rows": 391, "new_ledger_rows": len(ledger),
        "main_table": prior["main_table"], "Figure2_original_files": prior["Figure2_original_files"],
        "main_table_changed": False, "Figure2_changed": False, "final_paper_claim_adoption": False,
        "final_evidence_freeze": False, "population_mean_guarantee": False,
        "original_result_decision": "REPRODUCED_APPROXIMATE", "limits": LIMITS,
        "new_scientific_calls": ZERO_CALLS, "mandatory_STOP": True})
    put("three_environment_overlay.json", {"schema": "THREE_ENVIRONMENT_ADOPTION_DELTA_V1",
        "parent_index": refs["three_README.md"], "parent_CEM_overlay": refs["three_claim_evidence_overlay.csv"],
        "parent_49_series_unchanged": refs["three_research_index.csv"], "historical_claim_rows_unchanged": 29,
        "new_claims": [{"claim_id": claim, "canonical_CEM": OUT + "claim_evidence_matrix_v3.csv",
                        "canonical_evidence_ids": [f"H46Y4_{system}_S2", f"H46Y4_{system}_Y4", f"H46Y4_{system}_COMPARISON", "H46Y4_SAME_Q"]}
                       for system, claim in zip(["H4", "H6"], CLAIMS)],
        "old_SUP_PR_VS_PR_RECONSTRUCTED": "Original comparison unchanged; only its S2 component is reused here",
        "no_new_environment_or_external_replication": True, "independent_evidence_increment_from_indexing": 0,
        "mandatory_STOP": True})

    protected = list(refs.values()) + [prior["main_table"]] + prior["Figure2_original_files"]
    protected += [{"commit": SCIENCE, "path": r["path"], "sha256": r["sha256"]}
                  for r in json_saved(raw["result_end_audit.json"])["scientific_SOURCE"]]
    receipts = []
    for ref in protected:
        original = git_bytes(ref["commit"], ref["path"])
        require(digest(original) == ref["sha256"], "fixed reference SHA mismatch: " + ref["path"])
        local = ROOT / ref["path"]
        if local.exists():
            require(local.read_bytes() == original, "protected original changed: " + ref["path"])
        receipts.append({"commit": ref["commit"], "path": ref["path"], "sha256": ref["sha256"],
                         "check": "FIXED_BLOB_AND_WORKTREE_BYTES" if local.exists() else "FIXED_BLOB_ONLY_NOT_MATERIALIZED"})
    put("preservation_audit.json", {"status": "PASS", "references": receipts,
        "private_raw_runtime_grant_ledger_reaudit": "NOT_PERFORMED;REFERENCE_EXISTING_END_AUDIT_ONLY",
        "private_originals_written": False, "prior_grants_markers_STOP_reused_or_modified": False,
        "new_scientific_calls": ZERO_CALLS, "shared_environment_or_other_user_changes": 0})
    put("STOP.json", {"status": "ADOPTION_INTEGRATION_COMPLETE_MANDATORY_STOP", "mandatory_STOP": True,
        "adoption_status": STATUS, "source_result_commit": RESULT, "new_scientific_calls": ZERO_CALLS,
        "final_evidence_freeze": False, "detailed_scientific_review_started": False,
        "old_strict_COMPARE": "BLOCKED", "further_work_authorized": False})
    return files


def validate(root=ROOT):
    expected = render()
    for name, content in expected.items():
        actual = (root / OUT / name).read_bytes()
        # Scientific tokens/rows matter; presentation whitespace does not.
        parse = json_saved if name.endswith(".json") else csv_saved
        require(parse(actual) == parse(content), "adoption transcription/reference/guard changed: " + name)
    return {"status": "PASS", "generated_files": len(expected), "new_scientific_calls": ZERO_CALLS,
            "scope": "AUTHOR_TECHNICAL_CHECK_NOT_INDEPENDENT_SCIENTIFIC_REVIEW"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.build:
        target = ROOT / OUT
        target.mkdir(parents=True, exist_ok=True)
        for name, content in render().items():
            (target / name).write_bytes(content)
    print(json_text(validate()), end="")


if __name__ == "__main__":
    main()

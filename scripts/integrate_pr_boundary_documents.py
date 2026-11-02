#!/usr/bin/env python3
"""Literal public-document overlay and static checks; no scientific arithmetic.

--build writes only two documentation CSVs. --check reads fixed Git objects and
documents. Never imports scientific runners, opens private raw, or evaluates G.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

from adopt_h46_standard_y4_confirmation import (
    append_csv, csv_saved, git_bytes, json_saved, require,
)
from integrate_h468_paper_evidence import csv_text, json_text

ROOT = Path(__file__).resolve().parents[1]
BASE = "731b3d6f3572c6aa7950a579b68251fec9a5218b"
SYNTHESIS = "6e3436a16b13e539b0faa2a0a3286f6ab3549543"
RESULT = "202d25eb34be365aaa6638c6353e25f51acebe42"
PUBLIC = "analysis/pr_boundary_execution_20261027/"
OUT = "docs/data_collection/2026-10-27_pr_boundary_addendum/"
URL = "https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/"
H8 = "862d732fba0a5f353c522a9219dec78ae2f21a4b"
H8E = "artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/new_pr_cost_points.csv"
H8C = "artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv"
ADOPT = "artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/"
KEYS = ["BND_H4_R16_R64", "BND_H4_R128_R64", "BND_H6_R11_R16",
        "BND_H4_R32_EXPLORATION_ONLY", "BND_H8_Q16_REUSE_GAP",
        "BND_EXECUTION_PROVENANCE", "BND_COHORT_CELL_SCOPE"]
IDS = {"f85eb7b56ba5131d1eb1": KEYS[0], "3a55c1dac3947ae23c76": KEYS[1],
       "0e0e9fb6da33cb6ab2ab": KEYS[2], "d9cb4e923530349e940a": KEYS[3]}
STATUS = "ADDITIONAL_SAVED_EVIDENCE;PAPER_PLACEMENT_PROPOSED;NO_NEW_FORMAL_ADOPTION"
LIMIT = ("FINITE_SAVED_SAMPLE;EMPIRICAL_ALLOWANCE;NO_POPULATION_RANK_OR_TAIL_BOUND;"
         "NO_GLOBAL_OPTIMUM;NO_COHORT_POOLING;NOT_END_TO_END_QPE_RPE;NOT_EXTERNAL_REPLICATION")
FIELDS = "evidence_id matrix_id system geometry basis DF_rank_policy T epsilon family formula delta prefix q R_outer_or_r K candidate_id stage cohort n N_cos N_sin C_cos C_sin log_B G G_exact SE reference_id reference_cohort reference_G reference_SE relative_G_change_percent diagnostic supports does_not_support SOURCE_commit SOURCE_status result_commit saved_row public_url adoption_status paper_placement saved_fields_json".split()


def j(commit, path):
    return json_saved(git_bytes(commit, path))


def one(rows, **keys):
    found = [r for r in rows if all(r.get(k) == v for k, v in keys.items())]
    require(len(found) == 1, "non-unique fixed selector: " + str(keys))
    return found[0]


def make_row(eid, matrix_id, commit, path, selector, original, **values):
    r = dict.fromkeys(FIELDS, "not_applicable")
    r.update(evidence_id=eid, matrix_id=matrix_id, result_commit=commit,
             saved_row=selector, public_url=URL + commit + "/" + path,
             adoption_status=STATUS, paper_placement="Supplementary proposal;not adopted",
             saved_fields_json=json.dumps(original, ensure_ascii=False, sort_keys=True))
    r.update(values)
    return r


def evidence_rows():
    rows = csv_saved(git_bytes(RESULT, PUBLIC + "RESULTS.csv"))
    summary = j(RESULT, PUBLIC + "EXECUTION_SUMMARY.json")
    comparison = {r["id"]: r for r in summary["limited_comparisons"]}
    output = []
    for line, r in enumerate(rows, 2):
        if r["epsilon"] != "0.0005":
            continue
        mid = IDS.get(r["id"], KEYS[2] if r["system"] == "H6" else KEYS[0])
        if r["id"] == "45f9fc0a53c1a37c7345":
            mid = "BND_H4_R16_R64;BND_H4_R128_R64"
        confirmation = "CONFIRMATION" in r["cohort"]
        stage = "SAVED_REFERENCE_" if r["record_kind"] == "SAVED_BASELINE" else "NEW_BOUNDARY_"
        stage += "CONFIRMATION" if confirmation else "EXPLORATION"
        support = ("H6 R11 confirmation point lower;combined-SE INCONCLUSIVE;exploration separate"
                   if mid == KEYS[2] else "H4 R16/R128 higher G than fixed R64;no global optimum")
        if r["record_kind"] == "SAVED_BASELINE":
            support = "Previously saved reference;not reacquired;R32 is exploration n9 only"
        source = summary["execution_SOURCE"] if r["record_kind"] == "NEW_BOUNDARY_RESULT" else (
            "cab73373593ba7c113c1af018b1dc3077c8e78fa" if r["R_outer"] == "64" else
            "155529e20a8fac39e00786c40b57830b71203c4f")
        rel = comparison[r["id"]]["relative_G_change_percent"] if confirmation and r["id"] in comparison else "not_applicable"
        output.append(make_row("BND_ROW_" + str(line), mid, RESULT, PUBLIC + "RESULTS.csv",
            "line=" + str(line) + ";id=" + r["id"] + ";epsilon=0.0005;cohort=" + r["cohort"], r,
            **{k: r[k] or "not_applicable" for k in ["system", "T", "epsilon", "formula", "prefix", "q", "K", "cohort", "n", "N_cos", "N_sin", "C_cos", "C_sin", "G", "G_exact", "reference_id", "reference_cohort", "reference_G", "reference_SE", "diagnostic"]},
            geometry="linear;adjacent1.00Angstrom", basis="STO-3G", DF_rank_policy="tol_only1e-8;rank" + ("10" if r["system"] == "H4" else "19;weighted-Hermitian completion"),
            family="DF-native partial PR", delta="2.0", R_outer_or_r=r["R_outer"],
            candidate_id=r["id"], stage=stage, log_B=r["log_B"], SE=r["G_SE"],
            relative_G_change_percent=rel, supports=support, does_not_support=LIMIT,
            SOURCE_commit=source, SOURCE_status="New SOURCE identifier quoted from public summary;full private copy not re-audited or published" if r["record_kind"] == "NEW_BOUNDARY_RESULT" else "Public original SOURCE snapshot/provenance;not a claim that result commit equals original execution commit;not re-executed"))
    e = one(csv_saved(git_bytes(H8, H8E)), cell_id="B2_p7_q16_2nd_r4_K4", epsilon="0.001", bias_mode="USUAL_SIGNED", factor="1.0")
    c = one(csv_saved(git_bytes(H8, H8C)), candidate="B2_p7_q16_2nd_r4_K4", epsilon="0.0005", mode="primary")
    h8 = dict(system="H8", geometry="linear;adjacent1.00Angstrom", basis="STO-3G",
        DF_rank_policy="tol_only1e-8;rank28", T="0.8", family="DF-native partial PR",
        formula="S2", delta="0.05", prefix="7", q="16", R_outer_or_r="r=4;not m5 R_outer", K="4",
        candidate_id="B2_p7_q16_2nd_r4_K4", SOURCE_status="Public original SOURCE reference;full private runtime not re-audited", does_not_support="No new q8/q16 independent comparison at epsilon0.001;no new accounting;not evidence of zero cost or ineligibility")
    output.append(make_row("BND_H8_SAVED_E8", KEYS[4], H8, H8E,
        "line=928;cell_id=B2_p7_q16_2nd_r4_K4;epsilon=0.001;bias_mode=USUAL_SIGNED;factor=1.0", e,
        **h8, epsilon=e["epsilon"], stage="EXISTING_EXPLORATION", cohort=e["cohort"], n=e["n_used"],
        N_cos="unknown/original_signal_not_recovered", N_sin="unknown/original_signal_not_recovered",
        G="not_saved_decimal_no_conversion", G_exact=e["G_RZ_exact"], SE=e["G_RZ_SE"],
        supports="Existing q16 epsilon0.001 exploratory cost may be quoted", SOURCE_commit=e["SOURCE"]))
    output.append(make_row("BND_H8_SAVED_C48", KEYS[4], H8, H8C,
        "line=4;candidate=B2_p7_q16_2nd_r4_K4;epsilon=0.0005;mode=primary", c,
        **h8, epsilon=c["epsilon"], stage="EXISTING_CONFIRMATION", cohort=c["scope"], n=c["n"],
        N_cos=c["shots_cosine"], N_sin=c["shots_sine"], G="not_saved_decimal_no_conversion",
        G_exact=c["mean_exact"], SE=c["paired_SE"], reference_id=c["fixed_PF_id"], reference_G=c["fixed_PF_G_exact"],
        supports="Existing independent n48 at epsilon0.0005 only;no extra independent evidence",
        SOURCE_commit="745a674546f2cb6258526fbfde1d9a00601689f8"))
    output.append(make_row("BND_H8_MISSING_SIGNAL", KEYS[4], RESULT, PUBLIC + "EXECUTION_SUMMARY.json",
        "/H8_status", {"H8_status": summary["H8_status"], "H8_new_science": summary["H8_new_science"]},
        **h8, epsilon="0.001", stage="ACCOUNTING_NOT_ESTABLISHED", cohort="NO_NEW_C48_ACCOUNTING",
        N_cos="unknown/original_signal_not_recovered", N_sin="unknown/original_signal_not_recovered",
        G="not_established", G_exact="not_established", SE="not_established",
        diagnostic=summary["H8_status"], supports="A_signal missing;no C48 epsilon0.001 G/SE or new q8 comparison",
        SOURCE_commit="not_applicable_no_new_H8_execution"))
    output.append(make_row("BND_EXECUTION_END", KEYS[5], RESULT, PUBLIC + "EXECUTION_SUMMARY.json",
        "/actual;/correctness_checks;/controller_exit_code;/finalizer_exit_code;/mandatory_STOP", summary,
        system="H4/H6;H8 stopped", stage="PUBLIC_REPORTED_EXECUTION_AND_END_AUDIT",
        n=str(summary["actual"]["trajectories"]), diagnostic=summary["status"],
        supports="Public report records completed fixed scope,joined workers and exit0;author static checks only",
        does_not_support="Not a new raw audit,CI reproduction,external review or proof of scientific superiority",
        SOURCE_commit=summary["execution_SOURCE"] + ";" + summary["accounting_SOURCE"],
        SOURCE_status="Identifiers quoted from public summary;full SOURCE/raw locally preserved;not opened or published in this task"))
    return output


def matrix_rows(evidence):
    fields = list(csv_saved(git_bytes(SYNTHESIS, "EVIDENCE_MATRIX.csv"))[0])
    claims = [
        "H4 R16/K2 confirmation G higher than fixed R64/K2 (+19.983370567096262%);finite boundary contrast",
        "H4 R128/K2 confirmation G higher than fixed R64/K2 (+11.197670305697669%);finite boundary contrast",
        "H6 R11/K6 confirmation point G lower than R16 (-0.172094224366903%);INCONCLUSIVE",
        "H4 R32/K2 is saved exploration n9 only;not independent C evidence",
        "H8 q16 E8 epsilon0.001 and C48 epsilon0.0005 exist;C48 epsilon0.001 accounting not established",
        "Fixed H46 execution completed;public report/end-audit summary available;private SOURCE not re-audited",
        "Original33/40,seven-extended33/40,T2-limited E8/10 and C8/10 remain distinct saved evidence",
    ]
    summary = j(RESULT, PUBLIC + "EXECUTION_SUMMARY.json")
    extra = {"seven_extended_summary": j("bb97a93d65992fdb53e642dde1584dd7d5b811d7", "analysis/h46_seven_cost_readiness_20261027/results/extended_point_summary.json"),
             "original_PR_lower_cost_cell_count": summary["original_PR_lower_cost_cell_count"],
             "original_cell_count": summary["original_cell_count"],
             "limited_counts_literal_from_public_REPORT": "E8/10;C8/10;T2 only;not a new40-cell ranking"}
    out = []
    for mid, claim in zip(KEYS, claims):
        selected = [r for r in evidence if mid in r["matrix_id"].split(";")]
        r = dict.fromkeys(fields, "not_applicable")
        systems = "H8" if mid == KEYS[4] else "H6" if mid == KEYS[2] else "H4" if mid in [KEYS[0], KEYS[1], KEYS[3]] else "H4/H6;H8 gap separately"
        r.update(claim_id=mid, block="B_ADDENDUM_20261027", study_id="PR_BOUNDARY_FIXED_SCOPE_20261027",
            claim=claim, adoption_status=STATUS, adoption_authority="Documentation integration only;no new formal claim authority",
            system=systems, geometry="linear;adjacent1.00Angstrom", basis="STO-3G",
            Hamiltonian_state_input_identity="Saved study H/state/reference;H6 weighted-Hermitian completion;no regeneration or private re-audit",
            DF_rank_policy="tol_only1e-8;H4 rank10;H6 rank19;H8 rank28", T="0.8" if systems == "H8" else "2.0;counts scope explicitly separate" if mid == KEYS[6] else "2.0",
            epsilon="0.001/0.0005 separate" if systems == "H8" else "0.0005 primary;all5 saved endpoints referenced",
            PF_formula_q_delta="S2/q16/delta0.05" if systems == "H8" else "m5_best/q1/delta2;11 signed pieces",
            method_families="DF-native partial PR;fixed PF references only;Pauli-JW/B3 not integrated",
            partition_r_K="H8 p7/r4/K4" if systems == "H8" else "H4 p3/q1/K2 R16/32/64/128;H6 p7/q1/K6 R11/16;ABS_TIME_PER_OUTER",
            random_law_control_compiler="Original finite-RTE;PCG64/NumPy;symmetric_directional;actual adjoint;full cosine/+Im sine wrapper;P0;Qiskit1.3.0 rz/sx/x/cx opt1 seed17",
            cost_unit_shot_accounting="Compiled RZ per analytic shot;G=N_cos*C_cos+N_sin*C_sin;no new arithmetic",
            comparison_domain="Fixed finite candidates and original saved references;not full-domain reranking",
            exploration_confirmation_selection="E/C separate;fixed3 all eligible confirmed;references not reacquired;no top-up or pooling",
            n_N_SE="See literal rows;3 new physical candidates,not 5eps independent repetitions;paired axes are one trajectory",
            saved_result=claim, saved_fields_json=json.dumps(extra if mid == KEYS[6] else selected, ensure_ascii=False, sort_keys=True),
            scientific_limits=LIMIT + ";R32 E9 only;H6 INCONCLUSIVE;H8 no new .001 accounting",
            SOURCE_commit="Public summary SOURCE identifiers only;complete copies locally preserved,not re-audited",
            result_commit=RESULT, document_path=OUT + "REPORT.md",
            result_locator=OUT + "EVIDENCE_ADDENDUM.csv#matrix_id=" + mid,
            paper_placement="Existing section4/discussion+Supplementary boundary table proposal;no new chapter or adopted central claim",
            new_scientific_computation_required="No for literal integration;new analysis/graphs/confirmation require separate explicit scope")
        out.append(r)
    return out


def views():
    evidence = evidence_rows()
    return {OUT + "EVIDENCE_ADDENDUM.csv": csv_text(evidence).encode(),
            "EVIDENCE_MATRIX.csv": append_csv(git_bytes(SYNTHESIS, "EVIDENCE_MATRIX.csv"), matrix_rows(evidence))}


def check(writing_receipt=False):
    expected = views()
    for name, raw in expected.items():
        require((ROOT / name).read_bytes() == raw, "literal transcription differs: " + name)
    evidence = csv_saved(expected[OUT + "EVIDENCE_ADDENDUM.csv"])
    require(len(evidence) == 15 and len({r["evidence_id"] for r in evidence}) == 15, "evidence ID/schema inventory")
    for r in evidence:
        require(list(r) == FIELDS and all(v not in [None, ""] for v in r.values()), "empty CSV field")
    matrix = csv_saved(expected["EVIDENCE_MATRIX.csv"])
    require(len(matrix) == 50 and len({r["claim_id"] for r in matrix}) == 50, "matrix IDs")
    require([r["claim_id"] for r in matrix[-7:]] == KEYS, "overlay order")
    report = (ROOT / OUT / "REPORT.md").read_text()
    saved = csv_saved(git_bytes(RESULT, PUBLIC + "RESULTS.csv"))
    for candidate in ["f85eb7b56ba5131d1eb1", "45f9fc0a53c1a37c7345", "3a55c1dac3947ae23c76",
                      "0e0e9fb6da33cb6ab2ab", "efbb07c848b303a6abab"]:
        r = one([r for r in saved if "CONFIRMATION" in r["cohort"]], id=candidate, epsilon="0.0005")
        # Compare saved decimal tokens in the same Markdown table row; no evaluation.
        row = next(line for line in report.splitlines() if line.startswith("|") and candidate in line)
        for value in [r["cohort"], r["n"], r["N_cos"] + " / " + r["N_sin"], r["C_cos"],
                      r["G"] + " ± " + r["G_SE"]]:
            require(value in row, "report saved-token mismatch: " + candidate + ":" + value)
    for token in ["INCONCLUSIVE", "+19.98337%", "+11.19767%", "−0.17209%",
                  "R32", "探索n9", "33/40", "E8/10", "C8/10", "未成立", "完全SOURCE/rawを再監査・公開していない"]:
        require(token in report, "report loses declared scope: " + token)
    manifest = json.loads((ROOT / OUT / "INTEGRATION_MANIFEST.json").read_text())
    for r in manifest["public_reads"] + manifest["existing_public_file_reads"]:
        got = subprocess.check_output(["git", "rev-parse", r["ref"] + ":" + r["path"]], cwd=ROOT, text=True).strip()
        require(got == r["git_blob_sha"], "remote/local public blob mismatch: " + r.get("label", r["path"]))
    require(manifest["new_scientific_calls"] == 0 and manifest["new_numerical_reanalysis"] is False and manifest["new_formal_adoption"] is False, "scope")
    require(manifest["mandatory_STOP"] is True and manifest["full_private_SOURCE_reaudited"] is False, "STOP/private scope")
    # All inherited tracked bytes outside the documented allowlist remain intact.
    allowed = set(manifest["modified_existing_paths"])
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", BASE], cwd=ROOT).split(b"\0")
    count = 0
    for item in tree:
        if not item:
            continue
        meta, path_bytes = item.split(b"\t", 1)
        path = path_bytes.decode()
        if path in allowed:
            continue
        raw = (ROOT / path).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        require(blob == meta.split()[2].decode(), "protected file changed: " + path)
        count += 1
    for name in ["RESEARCH_SYNTHESIS.md", "PAPER_OUTLINE.md", "FIGURE_TABLE_PLAN.md"]:
        require((ROOT / name).read_bytes().startswith(git_bytes(SYNTHESIS, name)), "historical proposal overwritten: " + name)
    for name in allowed - {"artifacts/validation_manifest.json"}:
        current = (ROOT / name).read_bytes()
        old = git_bytes(BASE, name)
        require(current.startswith(old) or current.endswith(old), "not append/prepend-only: " + name)
    old = j(BASE, "artifacts/validation_manifest.json")
    new = json.loads((ROOT / "artifacts/validation_manifest.json").read_text(), parse_float=str)
    new.pop("pr_boundary_evidence_integration_20261027")
    require(old == new, "old validation manifest entries changed")
    # New local and pinned GitHub Markdown links resolve; no URL probing executes science.
    link_count = 0
    docs = manifest["new_markdown_paths"] + ["RESEARCH_SYNTHESIS.md", "PAPER_OUTLINE.md", "FIGURE_TABLE_PLAN.md"]
    fixed = re.compile(re.escape(URL) + r"([0-9a-f]{40})/([^\s)\]>]+)")
    for name in docs:
        text = (ROOT / name).read_text()
        for commit, path in fixed.findall(text):
            subprocess.check_output(["git", "cat-file", "-e", commit + ":" + unquote(path.split("#")[0])], cwd=ROOT)
            link_count += 1
        for target in re.findall(r"\[[^\]\n]*\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            destination = ((ROOT / name).parent / unquote(target.split("#")[0])).resolve()
            require(destination.exists() or (writing_receipt and destination == ROOT / OUT / "STATIC_VALIDATION.json"),
                    "broken local link: " + target)
    # Only the new suffixes/content are checked for accidental internal metadata.
    for name in manifest["publication_paths"]:
        if writing_receipt and name == OUT + "STATIC_VALIDATION.json" and not (ROOT / name).exists():
            continue
        raw = (ROOT / name).read_bytes()
        if name in ["EVIDENCE_MATRIX.csv", "RESEARCH_SYNTHESIS.md", "PAPER_OUTLINE.md", "FIGURE_TABLE_PLAN.md"]:
            raw = raw[len(git_bytes(SYNTHESIS, name)):]
        elif name in allowed:
            old = git_bytes(BASE, name)
            if raw.startswith(old):
                raw = raw[len(old):]
            elif raw.endswith(old):
                raw = raw[:-len(old)]
            else:
                raw = json.dumps(json.loads(raw)["pr_boundary_evidence_integration_20261027"]).encode()
        text = raw.decode()
        require(re.search(r"/(?:home|tmp|mnt|srv|Users)/|events_PRIVATE\.json|seed_(?:digest|registry)\s*[:=]|PRIVATE_NEXT[0-9]+_", text) is None,
                "private metadata in new publication content: " + name)
    allowed_imports = {"argparse", "ast", "csv", "hashlib", "io", "json", "pathlib", "re", "subprocess",
                      "urllib.parse", "adopt_h46_standard_y4_confirmation", "integrate_h468_paper_evidence"}
    module = ast.parse(Path(__file__).read_text())
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            require(all(x.name in allowed_imports for x in node.names), "scientific import forbidden")
        if isinstance(node, ast.ImportFrom):
            require(node.module in allowed_imports, "scientific import forbidden")
    return dict(status="PASS_STATIC_DOCUMENT_CHECKS_ONLY", evidence_rows=len(evidence),
        original_matrix_rows=43, added_matrix_rows=7, adopted_CEM_rows_preserved=23,
        protected_inherited_files=count, public_files_read=len(manifest["public_reads"]),
        existing_public_discovery_files_verified=len(manifest["existing_public_file_reads"]),
        pinned_markdown_references_checked=link_count, original_result_bytes_changed=0,
        new_scientific_calls=0, new_numerical_reanalysis=False, new_formal_adoption=False,
        full_private_SOURCE_reaudited=False, independent_external_review=False, mandatory_STOP=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="write only literal documentation CSV views")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--receipt", action="store_true", help="save static-check summary in addendum")
    args = parser.parse_args()
    if args.build:
        for name, raw in views().items():
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    result = check(writing_receipt=args.receipt)
    if args.receipt:
        (ROOT / OUT / "STATIC_VALIDATION.json").write_text(json_text(result))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

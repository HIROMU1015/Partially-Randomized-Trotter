#!/usr/bin/env python3
"""Read-only SP-1 saved-field audit. Stdlib only; no science module imports.

Never invokes a runner, synthesizer, channel/matrix evaluator or shot calculator.
Checks stored identities, inventory, interval predicates and saved-field arithmetic.
Prints a fresh report; does not write, replace or repair any evidence file.
"""
from collections import Counter
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "0d01ed9a332ebc5b66ed08acf56214a9b9c0236d"
AUTHORIZATION = "9630e06122172af238ac5219bec6dfc8b01ca837"
RESULT_SHA = "9a874f027b2927dcfde44cfb3b3e08a577cdf4b71aa16760a9d58c46a9464c48"
MARKER_SHA = "fd1064efdacde90953c73eb0092e6f3bd22ab283457bfe5cf63e862986f101f8"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError("SP-1 saved-field audit: "+message)


def enclosure(value):
    lo, hi = F(value["lo"]), F(value["hi"])
    require(lo <= hi, "reversed saved enclosure")
    return lo, hi


def overlap(a, b):
    require(max(a[0], b[0]) <= min(a[1], b[1]), "saved arithmetic enclosures disagree")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def audit():
    cp = "artifacts/track_b_sp1_wrapper_source/2026-10-06/contract_v1.json"
    raw_contract = (ROOT/cp).read_bytes()
    require(digest(raw_contract) == "f95931dbde2cfbba23c7585e851a26e3f3217d5820343f88b155b9a3a45093d1",
            "fixed contract changed")
    contract = json.loads(raw_contract)
    directory = ROOT/contract["result_directory"]
    result_bytes = (directory/"result.json").read_bytes()
    marker_bytes = (directory/"one_shot_consumed.json").read_bytes()
    require(digest(result_bytes) == RESULT_SHA and digest(marker_bytes) == MARKER_SHA,
            "raw one-shot result/marker changed")
    result, marker = json.loads(result_bytes), json.loads(marker_bytes)
    auth_bytes = (ROOT/contract["authorization_path"]).read_bytes()
    auth = json.loads(auth_bytes)
    require(result["source_commit"] == SOURCE and result["authorization_commit"] == AUTHORIZATION,
            "wrong execution source/HEAD")
    require(all(result[k] == v for k, v in marker.items()), "marker/result receipt mismatch")
    require(result["authorization_sha256"] == digest(auth_bytes), "authorization identity mismatch")
    require(auth_bytes == git("show", AUTHORIZATION+":"+contract["authorization_path"]),
            "authorization differs from committed A")
    require(auth["status"] == "APPROVED_FOR_ONE_SP1_RUN" and auth["science_execution_authorized"] is True,
            "separate authorization missing")
    require(auth["source_commit"] == SOURCE and auth["source_review_decision"] == "PASS_FOR_SEPARATE_SP1_AUTHORIZATION",
            "source review binding missing")
    require(digest(auth["explicit_execution_instruction"].encode()) == auth["explicit_execution_instruction_sha256"],
            "explicit instruction identity mismatch")
    require(digest((ROOT/auth["review_receipt"]["path"]).read_bytes()) == auth["review_receipt"]["sha256"],
            "instruction receipt changed")
    require(git("show", "-s", "--format=%P", AUTHORIZATION).decode().strip() == SOURCE,
            "A is not direct child of S")
    require(set(git("diff", "--name-only", SOURCE, AUTHORIZATION).decode().splitlines()) ==
            {contract["authorization_path"], contract["optional_receipt_path"]}, "A changed forbidden path")
    manifest_bytes = (ROOT/contract["source_manifest_path"]).read_bytes()
    require(manifest_bytes == git("show", SOURCE+":"+contract["source_manifest_path"]), "source manifest changed")
    manifest = json.loads(manifest_bytes)
    for path, expected in manifest["critical_sha256"].items():
        require(digest((ROOT/path).read_bytes()) == expected == digest(git("show", SOURCE+":"+path)),
                "critical source/input changed: "+path)
    old_marker = ROOT/"artifacts/track_b_sp05_economics_result/2026-10-06/v1/one_shot_consumed.json"
    require(digest(old_marker.read_bytes()) == manifest["protected_SP05_marker_sha256"], "old marker changed")
    for key in ("stored_input_sha256", "tool_identity_sha256", "contract_sha256"):
        expected = (contract["saved_input"]["sha256"] if key == "stored_input_sha256" else
                    contract["tool_identity"]["sha256"] if key == "tool_identity_sha256" else digest(raw_contract))
        require(result[key] == expected, "receipt identity mismatch: "+key)
    for document in (result, marker, auth):
        require(document["runs"] == 1 and document["retries"] == 0 and document["mandatory_STOP"] is True
                and document["next_stage_authorized"] is False, "one-shot STOP discipline mismatch")
    require(result["status"] == "SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW" and "failure" not in result,
            "this frozen complete-result audit cannot claim partial evidence complete")
    require(all(result[k] == 0 for k in ("synthesis_calls", "sampling_calls", "GPU_query_use")),
            "prohibited work recorded")
    require(result["actual_compiled_wrapper_T_claim"] is False and result["cost_metric"] == contract["metric"]["name"],
            "metric scope mismatch")
    tool = json.loads((ROOT/contract["tool_identity"]["path"]).read_bytes())
    require(result["runtime_identity"]["python"] == tool["python"] and
            result["runtime_identity"]["packages_match"] is True and
            result["runtime_identity"]["source_py_tree_sha256"] == tool["source_py_tree_sha256"], "runtime receipt mismatch")
    raw_saved = json.loads((ROOT/contract["saved_input"]["path"]).read_bytes())
    original_sequences = {s["key"]: s for s in raw_saved["synthesis_rows"]}
    sequences = result["stored_sequence_library"]
    for key, sequence in sequences.items():
        require(sequence == original_sequences[key], "saved primitive record changed")
        require(digest(sequence["sequence"].encode()) == sequence["sequence_sha256"] and
                sequence["sequence"].count("T")+sequence["sequence"].count("t") == sequence["T_count"] and
                sequence["error_pass"] is True, "sequence/count/error guard mismatch")
    require(set(sequences) == set(result["static_plan"]["stored_synthesis_keys_needed"]), "saved key inventory mismatch")
    specs = result["native_spec_library"]
    for identity, spec in specs.items():
        require(digest(json.dumps(spec, sort_keys=True).encode()) == identity, "native spec identity mismatch")
        coefficients = [F(g) for g in spec["coefficients_exact"]]
        probabilities = [F(p) for p in spec["canonical_probabilities_exact"]]
        gamma = sum(abs(g) for g in coefficients)
        require(sum(coefficients) == sum(probabilities) == 1 and
                probabilities == [abs(g)/gamma for g in coefficients], "stored canonical coefficients/probabilities inconsistent")
        lo, hi = enclosure(spec["gamma"])
        require(lo <= gamma <= hi, "primitive gamma serialization mismatch")
        for key, count in zip(spec["stored_sequence_keys"], spec["T_counts_additive"], strict=True):
            require(count == (sequences[key]["T_count"] if key is not None else 0), "additive T count mismatch")
    static = json.loads((ROOT/contract["static_fusion_audit_path"]).read_bytes())
    require(result["fusion_audit"] == static, "static fusion ledger changed")
    require(all(static[k] == 0 for k in ("fusion_candidate_count", "actual_fusion_count",
                                      "cross_role_fusion_opportunities")), "unexpected fusion")
    static_paths = {p["path_id"]: p for p in static["paths"]}
    rows = result["rows"]
    expected = {(t, n, m) for t in contract["domain"]["templates"] for n in contract["domain"]["sizes"]
                for m in contract["domain"]["masks"]}
    require(len(rows) == 48 and {(r["template"], r["n"], r["mask"]) for r in rows} == expected,
            "48 unique mask rows missing")
    by_key = {(r["template"], r["n"], r["mask"]): r for r in rows}
    cap_axes, eligible_axes, numeric_axes = 0, 0, 0
    for row in rows:
        require(len(row["axes"]) == 2 and {a["axis"] for a in row["axes"]} == {"Re", "Im"}, "axis inventory")
        baseline = by_key[(row["template"], row["n"], "NONE")]
        require(row["diagnostic_point_only"]["ideal_signal"] == baseline["diagnostic_point_only"]["ideal_signal"] and
                row["diagnostic_point_only"]["used_for_primary_shots_or_selection"] is False, "target/diagnostic scope mismatch")
        require(len(row["paths"]) == (2 if row["template"] == "C" else 1), "outer path count mismatch")
        for path in row["paths"]:
            native = static_paths[path["path_id"]]["post_fusion_native_sequence"]
            require(path["probability"] == static_paths[path["path_id"]]["probability"] and
                    path["outer_weight"] == static_paths[path["path_id"]]["outer_weight"], "outer path receipt mismatch")
            ids = path["native_spec_ids_in_order"]
            require(len(ids) == path["native_rotations"] == len(native) == 2*row["n"], "native inventory mismatch")
            for entry, identity in zip(native, ids, strict=True):
                spec = specs[identity]
                roles = {e["role"] for e in entry["lineage"]}
                require(len(roles) == 1 and spec["generator"] == entry["generator"], "native generator/lineage mismatch")
                selected = row["mask"] == "DR" or next(iter(roles)) == row["mask"]
                require(spec["placement_selected"] is selected, "placement role mismatch")
        for axis in row["axes"]:
            require(axis["alpha_exact"] == "1/1920" and axis["epsilon_axis_lower"] == contract["metric"]["epsilon_axis_lower"],
                    "confidence allocation changed")
            enclosure(axis["second_moment"])
            bias = F(axis["synthesis_bias_upper"])+F(axis["coefficient_numeric_bias_upper"])
            require(F(row["diagnostic_point_only"]["axis_residual"][axis["axis"]]) <= bias+F(contract["diagnostics"]["residual_slack"]),
                    "stored point residual inconsistent with stored guard")
            require(F(row["diagnostic_point_only"]["trace_residual"]) <= F(contract["diagnostics"]["residual_slack"]), "trace guard")
            if axis["status"] == "ELIGIBLE":
                require(type(axis["shots_sufficient"]) is int and 1 <= axis["shots_sufficient"] <= 10**9,
                        "eligible shots invalid")
                eligible_axes += 1
            elif axis["status"] == "SHOT_CAP_EXCEEDED":
                require(axis["shots_sufficient"] > 10**9, "model cap count clipped")
                cap_axes += 1
            else:
                numeric_axes += 1
        require(row["eligible"] is all(a["status"] == "ELIGIBLE" for a in row["axes"]), "eligibility mismatch")
        if not row["eligible"]:
            require(row["G_T_add"] is None and row["G_T_add_over_NONE"] is None and
                    row["classification"] == row["axes"][0]["status"], "ineligible row used as winner")
        else:
            cost = enclosure(row["axes"][0]["expected_C_T_add"])
            shots = sum(a["shots_sufficient"] for a in row["axes"])
            overlap(enclosure(row["G_T_add"]), (shots*cost[0], shots*cost[1]))
            ratio = enclosure(row["G_T_add_over_NONE"])
            g, base = enclosure(row["G_T_add"]), enclosure(baseline["G_T_add"])
            overlap(ratio, (g[0]/base[1], g[1]/base[0]))
            label = ("BASELINE" if row["mask"] == "NONE" else
                     "MATERIAL_GAIN" if ratio[1] <= F(19,20) else
                     "MATERIAL_LOSS" if ratio[0] >= F(21,20) else
                     "NO_MATERIAL_SEPARATION" if F(19,20) < ratio[0] <= ratio[1] < F(21,20) else
                     "NUMERIC_INCONCLUSIVE")
            require(row["classification"] == label, "saved classification predicate mismatch")
        duplicate = row["duplicate_of_mask"]
        require(row["count_as_independent_positive"] is
                (row["classification"] == "MATERIAL_GAIN" and row["mask"] != "NONE" and duplicate is None),
                "duplicate positive counting mismatch")
        if duplicate is not None:
            paired = by_key[(row["template"], row["n"], duplicate)]
            for key in ("G_T_add", "G_T_add_over_NONE", "axes", "paths", "diagnostic_point_only"):
                require(row[key] == paired[key], "duplicate mask saved values disagree")
    require(sum(len(r["axes"]) for r in rows) == 96, "96 axes missing")
    counts = dict(Counter(r["classification"] for r in rows))
    unique_gains = sum(r["count_as_independent_positive"] for r in rows)
    c_gains = sum(r["template"] == "C" and r["count_as_independent_positive"] for r in rows)
    require(result["summary"]["eligible_rows"] == sum(r["eligible"] for r in rows) and
            result["summary"]["independent_material_gain_rows"] == unique_gains and
            result["summary"]["placement_C_material_gain_rows"] == c_gains and
            result["summary"]["research_GO"] is None and result["summary"]["automatic_next_stage"] is None,
            "summary/automatic continuation mismatch")
    resources = result["resources"]
    require(0 <= resources["wall_seconds"] < contract["caps"]["wall_seconds"] and
            0 <= resources["cpu_seconds"] < contract["caps"]["cpu_seconds"] and
            resources["peak_RSS_KiB"] <= contract["caps"]["RSS_MiB"]*1024 and resources["processes"] == 1,
            "registered technical cap exceeded")
    payload_bytes = len(result_bytes)+len(marker_bytes)
    require(payload_bytes <= contract["caps"]["output_bytes"], "registered output cap exceeded")
    c_rows = [{"n": r["n"], "mask": r["mask"], "ratio_interval": r["G_T_add_over_NONE"],
               "classification": r["classification"], "shots_per_axis": r["axes"][0]["shots_sufficient"]}
              for r in rows if r["template"] == "C"]
    return {"schema": "track_b_sp1_saved_field_audit_v1", "status": "PASS_SAVED_FIELD_AUDIT",
        "source_commit": SOURCE, "authorization_commit": AUTHORIZATION,
        "raw_result_sha256": RESULT_SHA, "one_shot_marker_sha256": MARKER_SHA,
        "audit_script_sha256": digest(Path(__file__).read_bytes()),
        "critical_source_identities_match": len(manifest["critical_sha256"]),
        "raw_result_and_marker_preserved": True, "authorization_only_direct_child_verified": True,
        "wrappers": 12, "mask_rows": 48, "axis_rows": 96, "path_mask_records": sum(len(r["paths"]) for r in rows),
        "path_axis_records": 2*sum(len(r["paths"]) for r in rows),
        "primitive_sequences_verified": len(sequences), "native_specs_verified": len(specs),
        "eligible_rows": sum(r["eligible"] for r in rows), "eligible_axes": eligible_axes,
        "model_shot_cap_rows": counts.get("SHOT_CAP_EXCEEDED", 0), "model_shot_cap_axes": cap_axes,
        "other_ineligible_axes": numeric_axes, "classification_counts": counts,
        "independent_material_gain_rows": unique_gains, "placement_C_material_gain_rows": c_gains,
        "placement_C_selective_only_material_gain_rows": sum(r["template"] == "C" and
            r["mask"] in ("D", "R") and r["classification"] == "MATERIAL_GAIN" for r in rows),
        "C_rows": c_rows, "resources": resources, "resource_snapshot_scope": result["resource_snapshot_scope"],
        "registered_science_output_bytes": payload_bytes, "technical_failure_recorded": False,
        "synthesis_calls": 0, "trajectory_sampling": 0, "GPU_query_use": 0,
        "science_recomputed_by_audit": False, "classification_changed": False,
        "audit_scope": "stored identities/inventory/predicates/canonical coefficient consistency and saved-field interval arithmetic only; no matrix, guard, Bernstein or resource recalculation",
        "runs": 1, "retries": 0, "mandatory_STOP": True, "research_GO": None,
        "next_stage_authorized": False}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))

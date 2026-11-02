#!/usr/bin/env python3
"""Audit fixed R1 saved fields only; never import the science implementation.

No synthesis, angle evaluation, circuit construction, signal, trajectory,
Hamiltonian, NPZ, GPU, objective search, or scientific reclassification.
Bernstein shots and strict matrix guards are checked as stored certificates;
their logarithms and matrices are not evaluated again.
"""
import collections
import csv
import hashlib
import io
import json
import subprocess
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "d43d64a821a0249a0dfab12a2472bd3a72fdee74"
AUTHORIZATION = "411f08f768244fe87b600d82308c3851847fe9e4"
PREPARATION = ROOT / "artifacts/track_b_rte_reallocation_r1_source/2026-10-06"
OUTPUT = ROOT / "artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1"
RESULT_DIGEST = "f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e"
MARKER_DIGEST = "f25000ee5e3b94eb499b4a89bb28814e3de1aea641b249bcbb9d953d427c2ced"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def git(*args, binary=False):
    value = subprocess.check_output(["git", "-C", str(ROOT), *args])
    return value if binary else value.decode().strip()


def display(value):
    return format(float(F(value)), ".12g")


def pair(value):
    lo, hi = F(value["lo"]), F(value["hi"])
    require(0 <= lo <= hi, "invalid saved nonnegative interval")
    return lo, hi


def identity(row):
    return tuple(row[k] for k in ("context", "x", "sigma", "controlled", "epsilon"))


def validate_ratio(saved, numerator, denominator):
    lo, hi = pair(numerator)
    a, b = pair(denominator)
    if a <= 0:
        require(saved["status"] == "ZERO_OR_UNRESOLVED_BASELINE"
                and saved["ratio"] is None, "zero-baseline handling mismatch")
        return
    expected = lo / b, hi / a
    require(pair(saved["ratio"]) == expected, "stored ratio arithmetic mismatch")
    status = ("STRICTLY_LOWER" if expected[1] < 1 else
              "STRICTLY_HIGHER" if expected[0] > 1 else "OVERLAP_OR_EQUAL")
    require(saved["status"] == status, "stored ratio label mismatch")
    require(saved["materiality_or_research_GO"] is False, "unexpected ratio GO")


def audit():
    contract_bytes = (PREPARATION / "contract_v2.json").read_bytes()
    contract = json.loads(contract_bytes)
    auth = load(ROOT / contract["authorization_path"])
    manifest_bytes = (PREPARATION / "source_manifest_v1.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    schema = load(PREPARATION / "result_schema_v1.json")
    plan = load(ROOT / contract["key_inventory_path"])
    raw = (OUTPUT / "result.json").read_bytes()
    marker_bytes = (OUTPUT / "one_shot_consumed.json").read_bytes()
    result, marker = json.loads(raw), json.loads(marker_bytes)
    require(digest(raw) == RESULT_DIGEST, "original result changed")
    require(digest(marker_bytes) == MARKER_DIGEST, "original marker changed")
    require(set(schema["required"]).issubset(result), "missing result fields")
    require(result["status"] == "R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW",
            "this audit is for the saved complete run, not a replacement result")
    require("failure" not in result, "unexpected saved failure")
    require(result["source_commit"] == SOURCE and result["authorization_commit"] == AUTHORIZATION,
            "source/authorization binding mismatch")
    require(git("show", "-s", "--format=%P", AUTHORIZATION).split() == [SOURCE],
            "authorization is not a direct child")
    require(set(git("diff", "--name-only", SOURCE, AUTHORIZATION).splitlines()) ==
            {contract["authorization_path"], contract["optional_receipt_path"]},
            "authorization path allowlist mismatch")
    require(auth["status"] == "APPROVED_FOR_ONE_R1_RUN"
            and auth["science_execution_authorized"] is True
            and auth["source_commit"] == SOURCE and bool(auth["explicit_execution_instruction"]),
            "missing explicit approval")
    require(auth["explicit_execution_instruction"] ==
            "source `d43d64a821a0249a0dfab12a2472bd3a72fdee74` の固定契約でR1を一回だけ実行し、終了後はmandatory STOPしてください。",
            "user instruction mismatch")
    require(manifest_bytes == git("show", SOURCE + ":" + contract["source_manifest_path"], binary=True),
            "source manifest changed")
    for name in ("runs", "retries", "mandatory_STOP"):
        require(auth[name] == result[name] == marker[name], "one-shot policy mismatch: " + name)
    require(result["runs"] == 1 and result["retries"] == 0 and result["mandatory_STOP"] is True,
            "one-shot/STOP violation")
    require(result["next_stage_authorized"] is False and marker["next_stage_authorized"] is False
            and result["science_GO"] is False and result["DF_wrapper_authorized"] is False
            and result["molecule_GPU_trajectory_calls"] == 0, "unexpected scientific authorization")
    for key, value in marker.items():
        require(result[key] == value, "marker/result receipt mismatch: " + key)
    receipt_hashes = {
        "contract_sha256": digest(contract_bytes),
        "authorization_sha256": digest((ROOT / contract["authorization_path"]).read_bytes()),
        "tool_identity_sha256": digest((ROOT / contract["tool_identity"]["path"]).read_bytes()),
        "key_inventory_sha256": digest((ROOT / contract["key_inventory_path"]).read_bytes()),
        "one_shot_marker_sha256": digest(marker_bytes),
    }
    for name, expected in receipt_hashes.items():
        require(result[name] == expected, "receipt hash mismatch: " + name)
    require(auth["contract_sha256"] == receipt_hashes["contract_sha256"], "auth contract mismatch")
    for path, expected in manifest["critical_sha256"].items():
        require(not path.lower().endswith(".npz"), "NPZ access excluded")
        require(digest((ROOT / path).read_bytes()) == expected, "critical source changed: " + path)
    protected = manifest["prior_R0_R05_SP_BS_evidence_git_blob_sha256_unchanged"]
    for path, expected in protected.items():
        require(not path.lower().endswith(".npz"), "NPZ access excluded")
        require(digest(git("show", "HEAD:" + path, binary=True)) == expected,
                "protected evidence changed: " + path)
    tool = load(ROOT / contract["tool_identity"]["path"])
    require(result["runtime_identity"]["packages_match"] is True
            and result["runtime_identity"]["python"] == tool["python"]
            and result["runtime_identity"]["source_py_tree_sha256"] == tool["source_py_tree_sha256"],
            "saved runtime identity mismatch")
    synthesis = result["synthesis_rows"]
    require([row["key"] for row in synthesis] == plan["keys"], "synthesis inventory mismatch")
    require(len(synthesis) == result["synthesis_attempts"] == result["pygridsynth_invocations"]
            == result["synthesis_calls"] == result["planned_synthesis_keys"] == 126,
            "attempt/call/key counts mismatch")
    require(result["last_attempted_synthesis_key"] == plan["keys"][-1], "last key mismatch")
    for row in synthesis:
        require(set(schema["synthesis_rows_required"]).issubset(row), "missing synthesis fields")
        sequence = row["sequence"]
        require(set(sequence).issubset(set("HTtSXW")), "unexpected synthesis alphabet")
        require(digest(sequence.encode()) == row["sequence_sha256"], "sequence digest mismatch")
        require(row["T_count"] == sequence.count("T") + sequence.count("t")
                and row["Tdagger_count"] == sequence.count("t")
                and row["global_W_count"] == sequence.count("W")
                and row["one_qubit_count"] == len(sequence) - sequence.count("W"),
                "sequence counts mismatch")
        require(row["epsilon"] in contract["native_operator_epsilons"]
                and row["key"] == row["angle_key"] + ":epsilon:" + row["epsilon"], "key identity mismatch")
        require(row["error_pass"] is True
                and 0 <= F(row["strict_operator_error_upper"]) <= F(row["epsilon"]), "saved strict guard failed")
        require(len(sequence) <= contract["caps"]["sequence_characters"], "sequence cap exceeded")
        for field, cap in (("wall_seconds", "per_key_wall_seconds"), ("cpu_seconds", "per_key_cpu_seconds")):
            require(0 <= row[field] < contract["caps"][cap], "per-key cap exceeded")
        require(row["peak_RSS_KiB"] <= 1024 * contract["caps"]["RSS_MiB"], "per-key RSS cap exceeded")
    usage = result["resource_usage"]
    require(usage["processes"] == 1 and usage["wall_seconds"] < contract["caps"]["wall_seconds"]
            and usage["cpu_seconds"] < contract["caps"]["cpu_seconds"]
            and usage["peak_RSS_KiB"] <= 1024 * contract["caps"]["RSS_MiB"], "saved resource cap exceeded")
    require(len(raw) < contract["caps"]["output_bytes"], "result payload cap exceeded")

    rows = result["resource_rows"]
    expected_ids = {
        (context, x, sigma, controlled, eps, arm)
        for context in contract["domain"]["contexts"]
        for x in contract["domain"]["x"] for sigma in contract["domain"]["sigma"]
        for controlled in (False, True) for eps in contract["native_operator_epsilons"]
        for arm in contract["arms_by_context"][context]
    }
    by_id = {identity(row) + (row["arm"],): row for row in rows}
    require(len(rows) == len(by_id) == contract["planned_resource_rows"] == 264
            and set(by_id) == expected_ids, "resource row inventory mismatch")
    event_count = 0
    for row in rows:
        require(set(schema["resource_rows_required"]).issubset(row), "missing resource fields")
        profile = row["profile"]
        require(set(schema["profile_required"]).issubset(profile), "missing profile fields")
        require(row["science_GO"] is False and row["mandatory_STOP"] is True, "row GO/STOP violation")
        require(row["evidence_role"] == ("primary_native_controlled" if
                row["context"] == "distinct_basis" and row["controlled"] else "control_or_diagnostic"),
                "evidence-role mismatch")
        B = F(profile["implemented_B"])
        events = profile["events"]
        event_count += len(events)
        require(events and B > 0, "invalid event population")
        require(sum(F(e["implemented_coefficient"]) for e in events) == B
                and sum(F(e["canonical_probability_exact"]) for e in events) == 1,
                "canonical normalization mismatch")
        lo = sum(pair(e["ideal_coefficient"])[0] for e in events)
        hi = sum(pair(e["ideal_coefficient"])[1] for e in events)
        require(pair(profile["ideal_B"]) == (lo, hi)
                and pair(profile["ideal_B_squared"]) == (lo * lo, hi * hi)
                and F(profile["implemented_weight_second_moment"]) == B * B
                and F(profile["corrected_weight_range"]) == B,
                "saved B/moment arithmetic mismatch")
        coefficient_bias = F(0)
        synthesis_bias = F(0)
        for event in events:
            a = F(event["implemented_coefficient"])
            elo, ehi = pair(event["ideal_coefficient"])
            require(a == (elo + ehi) / 2 and F(event["canonical_probability_exact"]) == a / B,
                    "midpoint probability mismatch")
            cost = event["native_cost"]
            require(all(type(cost[k]) is int and cost[k] >= 0 for k in ("T", "CX", "1Q", "IR_gate_count")),
                    "invalid saved native cost")
            require(len(cost["IR_sha256"]) == 64 and F(cost["strict_event_error_upper"]) >= 0,
                    "invalid saved IR/error certificate")
            coefficient_bias += max(abs(a - elo), abs(a - ehi))
            synthesis_bias += a * F(cost["strict_event_error_upper"])
        require(F(profile["coefficient_L1_bias_upper"]) == coefficient_bias
                and coefficient_bias <= F(contract["finite_confidence"]["coefficient_bias_cap"])
                and F(profile["coefficient_and_joint_operator_error_upper"]) == coefficient_bias + synthesis_bias
                and F(profile["coefficient_and_strict_synthesis_bias_upper"]) == coefficient_bias + 2 * synthesis_bias
                and profile["coherent_measurement_synthesis_factor"] == 2, "stored bias arithmetic mismatch")
        for name in ("T", "CX", "1Q"):
            require(F(profile["E_native_cost"][name]) == sum(
                F(e["canonical_probability_exact"]) * e["native_cost"][name] for e in events),
                "expected native cost mismatch")
        require(profile["workspace_qubits_beyond_2_system"] == int(row["controlled"]), "workspace mismatch")
        confidence = row["finite_confidence"]
        if not row["controlled"]:
            require(confidence["status"] == "ORDINARY_DIAGNOSTIC_NO_COHERENT_TASK", "ordinary task pooling")
            continue
        require(confidence["status"] == "ELIGIBLE_COMMON_FINITE_CONFIDENCE_TASK", "unexpected saved ineligible task")
        n = confidence["sufficient_shots_per_axis"]
        require(type(n) is int and 0 < n <= contract["finite_confidence"]["shot_cap_per_axis"]
                and confidence["sufficient_shots_total"] == 2 * n, "saved shot count mismatch")
        bias = F(profile["coefficient_and_strict_synthesis_bias_upper"])
        require(F(confidence["epsilon_stat_axis_lower"]) == F(contract["finite_confidence"]["epsilon_axis"]) - bias
                and F(confidence["bias_upper"]) == bias
                and F(confidence["variance_upper"]) == B * B
                and F(confidence["centered_range_upper"]) == 2 * B, "confidence input arithmetic mismatch")
        for name in ("T", "CX"):
            require(F(confidence["G_" + name]) == 2 * n * F(profile["E_native_cost"][name]), "saved G arithmetic mismatch")
        require(F(confidence["G_1Q_with_Hadamard_preparation_readout"]) == n * (2 * F(profile["E_native_cost"]["1Q"]) + 5),
                "saved 1Q preparation/readout accounting mismatch")
        require(confidence["exact_signal_used"] is False
                and confidence["statistical_estimation_shots_executed"] == 0, "unexpected oracle/shots")

    comparisons = result["comparison_rows"]
    require(len(comparisons) == contract["planned_comparison_groups"] == 72
            and {identity(c) for c in comparisons} == {key[:-1] for key in expected_ids}, "comparison inventory mismatch")
    comparison_pairs = 0
    for row in comparisons:
        key = identity(row)
        a = by_id[key + ("A",)]
        require(row["scientific_classification"] is None, "unexpected scientific reclassification")
        expected_baselines = set(contract["arms_by_context"][row["context"]]) - {"A"}
        require({p["baseline"] for p in row["A_vs_registered_baselines"]} == expected_baselines,
                "baseline inventory mismatch")
        for saved in row["A_vs_registered_baselines"]:
            comparison_pairs += 1
            baseline = by_id[key + (saved["baseline"],)]
            validate_ratio(saved["B_squared"], a["profile"]["ideal_B_squared"], baseline["profile"]["ideal_B_squared"])
            for name in ("T", "CX", "1Q"):
                x, y = a["profile"]["E_native_cost"][name], baseline["profile"]["E_native_cost"][name]
                validate_ratio(saved["native_expected_cost"][name], {"lo": x, "hi": x}, {"lo": y, "hi": y})
            for name in ("G_T", "G_CX"):
                if not row["controlled"]:
                    require(saved[name]["status"] == "TASK_NOT_JOINTLY_ELIGIBLE" and saved[name]["ratio"] is None,
                            "ordinary row has coherent-task ratio")
                else:
                    x, y = a["finite_confidence"][name], baseline["finite_confidence"][name]
                    validate_ratio(saved[name], {"lo": x, "hi": x}, {"lo": y, "hi": y})
    report = {
        "schema": "track_b_R1_saved_field_audit_v1", "status": "PASS_SAVED_FIELDS_ONLY",
        "source_commit": SOURCE, "authorization_commit": AUTHORIZATION,
        "result_sha256": RESULT_DIGEST, "one_shot_marker_sha256": MARKER_DIGEST,
        "receipt_hashes": receipt_hashes,
        "reviewed_critical_hashes_verified": len(manifest["critical_sha256"]),
        "protected_prior_evidence_hashes_verified": len(protected),
        "synthesis_rows_verified": len(synthesis), "resource_rows_verified": len(rows),
        "saved_event_records_verified": event_count, "comparison_groups_verified": len(comparisons),
        "comparison_pairs_verified": comparison_pairs,
        "controlled_tasks_verified": sum(row["controlled"] for row in rows),
        "primary_distinct_basis_controlled_resource_rows": sum(row["evidence_role"] == "primary_native_controlled" for row in rows),
        "resource_usage_as_saved": usage, "original_result_bytes": len(raw),
        "scope": "saved identities, inventories, exact rational bookkeeping and saved labels only",
        "not_recomputed": ["strict matrix guard", "native circuits/IR hashes", "Bernstein logarithm or sufficient-shot ceiling", "finite first-moment matrices or signals"],
        "science_runner_invocations_in_audit": 0, "synthesis_calls_in_audit": 0,
        "scientific_reclassification": False, "new_candidates_or_cells": 0,
        "molecule_DF_NPZ_trajectory_GPU_operations": 0,
        "runs": 1, "retries": 0, "mandatory_STOP": True, "next_stage_authorized": False,
    }
    return result, report


def summaries(result):
    buckets = {}
    primary = []
    for row in result["comparison_rows"]:
        for saved in row["A_vs_registered_baselines"]:
            metrics = {"B_squared": saved["B_squared"], "E_T": saved["native_expected_cost"]["T"],
                       "E_CX": saved["native_expected_cost"]["CX"], "E_1Q": saved["native_expected_cost"]["1Q"],
                       "G_T": saved["G_T"], "G_CX": saved["G_CX"]}
            key = (row["context"], row["controlled"], saved["baseline"])
            bucket = buckets.setdefault(key, {name: [] for name in metrics})
            for name, value in metrics.items():
                bucket[name].append(value)
            if row["context"] == "distinct_basis" and row["controlled"]:
                primary.append({**{k: row[k] for k in ("context", "x", "sigma", "controlled", "epsilon")},
                                "baseline": saved["baseline"],
                                "saved_comparison": saved})
    output = []
    for (context, controlled, baseline), metrics in buckets.items():
        values = {}
        for name, saved in metrics.items():
            ratios = [v["ratio"] for v in saved if v["ratio"] is not None]
            bounds = {"lo": str(min(F(v["lo"]) for v in ratios)),
                      "hi": str(max(F(v["hi"]) for v in ratios))} if ratios else None
            values[name] = {"saved_status_counts": dict(collections.Counter(v["status"] for v in saved)),
                            "range_of_saved_ratios": bounds,
                            "range_display_only": {k: display(v) for k, v in bounds.items()} if bounds else None}
        output.append({"context": context, "controlled": controlled, "baseline": baseline, "metrics": values})
    return {"schema": "track_b_R1_saved_descriptive_summary_v1", "result_sha256": RESULT_DIGEST,
            "display_rounding_only": "12 significant digits; original exact fractions and interval labels govern",
            "summary_method": "counts and extrema of stored comparisons; no new objective or scientific classification",
            "buckets": output, "all_primary_comparisons": primary,
            "controlled_task_status_counts": dict(collections.Counter(
                row["finite_confidence"]["status"] for row in result["resource_rows"] if row["controlled"])),
            "research_GO": False, "mandatory_STOP": True, "next_stage_authorized": False}


def scalar_csv(result):
    columns = ["context", "x", "sigma", "controlled", "epsilon", "arm", "evidence_role",
               "ideal_B_squared_lo_display", "ideal_B_squared_hi_display", "implemented_second_moment_display",
               "E_T_display", "E_CX_display", "E_1Q_display", "workspace_qubits_beyond_2_system",
               "description_entries", "expected_index_draws_display", "enumerated_evaluator_events",
               "CTS_collection_word_labels", "CTS_collection_Pauli_multiplications", "acquisition_wall_seconds_diagnostic",
               "coefficient_bias_upper_display", "coherent_measurement_bias_upper_display", "finite_confidence_status",
               "sufficient_shots_per_axis", "G_T_display", "G_CX_display", "G_1Q_with_Hadamard_preparation_readout_display"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in result["resource_rows"]:
        p, c, f = row["profile"], row["C_classical"], row["finite_confidence"]
        out = {k: row[k] for k in columns[:7]}
        out.update(ideal_B_squared_lo_display=display(p["ideal_B_squared"]["lo"]),
                   ideal_B_squared_hi_display=display(p["ideal_B_squared"]["hi"]),
                   implemented_second_moment_display=display(p["implemented_weight_second_moment"]),
                   workspace_qubits_beyond_2_system=p["workspace_qubits_beyond_2_system"],
                   description_entries=c["coefficient_description_entries"],
                   expected_index_draws_display=display(c["expected_I0_involution_index_draws"]) if c["expected_I0_involution_index_draws"] is not None else "",
                   enumerated_evaluator_events=c["enumerated_evaluator_events"],
                   CTS_collection_word_labels=c["CTS_collection_word_labels"],
                   CTS_collection_Pauli_multiplications=c["CTS_collection_Pauli_multiplications"],
                   acquisition_wall_seconds_diagnostic=c["coefficient_and_event_acquisition_wall_seconds_diagnostic"],
                   coefficient_bias_upper_display=display(p["coefficient_L1_bias_upper"]),
                   coherent_measurement_bias_upper_display=display(p["coefficient_and_strict_synthesis_bias_upper"]),
                   finite_confidence_status=f["status"],
                   sufficient_shots_per_axis=f.get("sufficient_shots_per_axis", ""))
        for name in ("T", "CX", "1Q"):
            out["E_" + name + "_display"] = display(p["E_native_cost"][name])
        for name in ("G_T", "G_CX", "G_1Q_with_Hadamard_preparation_readout"):
            out[name + "_display"] = display(f[name]) if name in f else ""
        writer.writerow(out)
    return stream.getvalue()


def main():
    result, report = audit()
    report["audit_script_sha256"] = digest(Path(__file__).read_bytes())
    outputs = {"saved_field_audit_v1.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
               "descriptive_summary_v1.json": json.dumps(summaries(result), ensure_ascii=False, indent=2) + "\n",
               "resource_rows_display_v1.csv": scalar_csv(result)}
    for name, text in outputs.items():
        with (OUTPUT / name).open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(json.dumps({"status": report["status"], "synthesis_rows": report["synthesis_rows_verified"],
                      "resource_rows": report["resource_rows_verified"], "comparison_pairs": report["comparison_pairs_verified"],
                      "science_runner_calls": 0, "mandatory_STOP": True}))


if __name__ == "__main__":
    main()

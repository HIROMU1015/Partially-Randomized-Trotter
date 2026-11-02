"""Fixed-blob metadata audit and synthetic boundary checks only.

Never imports a scientific provider, reads NPZ, evaluates a saved shot/cost
formula, samples trajectories/measurements, builds circuits, or compiles.
All real result numbers are inspected as saved fields, not recomputed.
"""
from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "41ba315b2ad369825577ad671c79b69c77e1f3cc"
H4 = "217c971dd537b1cc7fbc13742bc0f6c263ff7f45"
H6 = "87b6b9f46b7e8ba9abe132ee21478c114b6f46f7"
H8 = "c31ef424175c02379eafebe8fc6c7548c1d9aabe"
GEO = "d0726e946f89f8c30a08d383fb8ff48b363629ec"
D0 = "fa75c7997e135da7b7a0c4e23949e47252bd7a2e"
EPS = "3156b036fb4a0015e09cce2cb73785d7819dcb75"
OLD = "e098c54c78f589055082f9cfc2b13de50c90ca94"
PREFIX = "src/trottertracks/resource_applicability/"
OUT = ROOT / "artifacts/resource_applicability/pr_alpha_contract_audit/2026-10-18"

# Deliberate allowlist. No worktree/process discovery or recursive disk scan.
SPECS = {
    "geometry_signal": (GEO, PREFIX + "h4_geometry/signal.py"),
    "geometry_circuits": (GEO, PREFIX + "h4_geometry/circuits.py"),
    "geometry_identity": (GEO, PREFIX + "h4_geometry/identity.py"),
    "h4_b0i_accounting": ("98cdb35cd08469714f20d6cc3b982d9c80908f46", "artifacts/resource_applicability/track_a_h4_b0i_map_results/2026-10-14/analyze_saved_results.py"),
    "h4_native_adapter": (H4, PREFIX + "h4_common_panel_port_v1.py"),
    "h4_native_contract": (H4, PREFIX + "h48_preflight_contract_v1.py"),
    "h6_contract": (H6, PREFIX + "h6_matched_contract_v1.py"),
    "h8_contract": (H8, PREFIX + "h8_resource_map_contract_v1.py"),
    "h8_preflight_contract": (H8, PREFIX + "h8_signal_preflight_contract_v1.py"),
    "h4_native_shot": (H4, PREFIX + "ax2b_numerical_accounting.py"),
    "h6_native_shot": (H6, PREFIX + "ax2b_numerical_accounting.py"),
    "h8_native_shot": (H8, PREFIX + "ax2b_numerical_accounting.py"),
    "native_headroom": (H8, PREFIX + "ax2a_preparation.py"),
    "native_signal": (H8, PREFIX + "h6_matched_port_v1.py"),
    "native_accounting": (H8, PREFIX + "h6_matched_accounting_v1.py"),
    "h4_native_signal": (H4, PREFIX + "h6_matched_port_v1.py"),
    "h6_native_signal": (H6, PREFIX + "h6_matched_port_v1.py"),
    "h4_native_accounting": (H4, PREFIX + "h6_matched_accounting_v1.py"),
    "h6_native_accounting": (H6, PREFIX + "h6_matched_accounting_v1.py"),
    "h6_b0i_accounting": ("ac6d28e35cf01398f169ce59577182a38da9b959", "scripts/h6_b0i_family_accounting_v1.py"),
    "native_sampling": (H8, PREFIX + "ax2b_molecular_ports_v3.py"),
    "native_builder": (H8, PREFIX + "ax2b_native_df_v5.py"),
    "h8_cost": (H8, PREFIX + "h8_fast_cost_port_v1.py"),
    "rte_law": (H8, "src/trotterlib/rte.py"),
    "repeated_request": (H8, "src/trotterlib/df_partial_s2_repeated.py"),
    "rte_circuit": (H8, "src/trotterlib/df_rte_circuit.py"),
    "rte_builder": (H8, "src/trotterlib/df_rte_qiskit.py"),
    "d0_shot": (D0, "scripts/d0_frozen_shot_rule_v1.py"),
    "d0_accounting": (D0, "scripts/d0_saved_value_reanalysis_v1.py"),
    "h6_strengthened_accounting": ("2261a3ad1d6c23521474b31204572a6c3a51258b", "scripts/track_a_v1_reaccount.py"),
    "h8_noncancellation": (EPS, "src/trotterlib/h8_noncancellation.py"),
    "h8_epsilon_accounting": (EPS, "src/trotterlib/h8_epsilon_boundary.py"),
    "h4_geometry_result": ("7c4483378047b57f514c47dad98905e3f2a9a756", "artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/saved_candidates_v1.json"),
    "h4_native_result": ("0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41", "artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/signal_metadata.json"),
    "d0_result": (BASE, "artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/candidate_accounting_v1.json"),
    "h6_strengthened_result": (BASE, "artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/candidate_accounting.json.gz"),
    "h8_six_axis_result": (BASE, "artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/accounting_rows.jsonl.gz"),
    "h8_epsilon_result": (BASE, "artifacts/resource_applicability/track_a_h8_epsilon_boundary_run01/2026-10-18/accounting_rows.jsonl.gz"),
    "h8_epsilon_inputs": (EPS, "references/h8_epsilon_boundary_inputs_v1.json"),
    "cross_system_inventory": (BASE, "artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/study_inventory_v1.json"),
    "old_rpe_source": (OLD, "src/trotterlib/rpe_hadamard_failure_validation.py"),
    "old_rpe_three_round": (OLD, "docs/rpe_hadamard_failure_validation.md"),
    "old_rpe_four_round": (OLD, "docs/rpe_four_round_phase_validation.md"),
    "old_rpe_long_round": (OLD, "docs/rpe_delta_round_schedule_validation.md"),
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def function(source, name):
    matches = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError("FUNCTION_IDENTITY:" + name)
    return matches[0]


def same_function(left, right, name):
    return ast.dump(function(left, name), include_attributes=False) == ast.dump(function(right, name), include_attributes=False)


def pure_functions(source, names, namespace):
    # Extract only previously inspected pure functions; ignore all module code.
    module = ast.Module(body=[function(source, n) for n in names], type_ignores=[])
    exec(compile(module, "<fixed-pure-functions>", "exec"), namespace)
    return namespace


def decoded(raw, path):
    data = gzip.decompress(raw) if path.endswith(".gz") else raw
    if ".jsonl" in path:
        return [json.loads(line) for line in data.splitlines() if line]
    return json.loads(data)


def accounting_metadata(rows):
    evidence, alpha, method, states, integers = (Counter() for _ in range(5))
    missing = 0
    errors = []
    for index, row in enumerate(rows):
        method[row.get("method", "UNKNOWN")] += 1
        a = row.get("accounting")
        if a is None:
            missing += 1
            continue
        evidence[a.get("evidence_kind")] += 1
        alpha[str(a.get("alpha_axis"))] += 1
        states[a.get("eligibility_under_declared_allowance")] += 1
        if a.get("alpha_axis") != .025 or a.get("evidence_kind") != "EMPIRICAL":
            errors.append([index, "ALPHA_OR_EVIDENCE"])
        for key in ("allowance_certification_claimed_by_caller", "certificate_verified_by_this_function", "familywise_winner_certified"):
            if a.get(key) is not False:
                errors.append([index, key])
        if a.get("interpretation") != "CONDITIONAL_DIAGNOSTIC":
            errors.append([index, "INTERPRETATION"])
        for axis in ("real", "imag"):
            axis_row = a["axes"][axis]
            integers[axis_row["integer_status"]] += 1
            if axis_row["integer_status"] == "LOG_DOMAIN_ONLY" and axis_row["shots"] is not None:
                errors.append([index, "LOG_ONLY_HAS_INTEGER"])
        if a.get("cost_acquisition_eligible"):
            if a["eligibility_under_declared_allowance"] != "ELIGIBLE" or type(a["N_total"]) is not int:
                errors.append([index, "ELIGIBILITY_STATE"])
            for axis in a["axes"].values():
                if type(axis["shots"]) is not int or axis["shots"] <= 0:
                    errors.append([index, "INVALID_SAVED_SHOT_TYPE"])
    return dict(rows=len(rows), methods=dict(method), accounting_missing=missing,
                evidence_kind=dict(evidence), alpha_axis=dict(alpha), eligibility=dict(states),
                integer_status=dict(integers), errors=errors,
                operation="SAVED_FIELD_SCHEMA_AND_CLASSIFICATION_ONLY_NO_RECALCULATION")


def synthetic_checks(blobs):
    ns = pure_functions(blobs["d0_shot"].decode(), ("axis_headroom", "_number", "u_aware_shots"),
                        dict(math=math, sys=sys, AXES=("real", "imag")))
    shots = ns["u_aware_shots"]
    results = []

    def check(name, assertion):
        if not assertion:
            raise AssertionError(name)
        results.append(dict(name=name, passed=True))

    bias = dict(real=0., imag=0.)
    allowance = dict(real=0., imag=0.)
    def call(**overrides):
        kw = dict(epsilon=math.sqrt(2), log_B_upper=0., evidence_kind="EMPIRICAL", evidence_ref="SYNTHETIC_ONLY")
        kw.update(overrides)
        return shots(bias, allowance, **kw)

    unknown = call(evidence_kind="UNKNOWN", evidence_ref=None)
    check("unknown_does_not_produce_finite_shots", unknown["N_total"] is None and unknown["eligibility_under_declared_allowance"] == "UNDETERMINED")
    try:
        call(evidence_ref=None)
    except ValueError as exc:
        check("empirical_requires_reference", str(exc) == "NUMERICAL_EVIDENCE_REFERENCE_REQUIRED")
    else:
        raise AssertionError("empirical_requires_reference")
    certified = call(evidence_kind="CERTIFIED")
    check("caller_certification_never_verifies_certificate", certified["allowance_certification_claimed_by_caller"] is True and certified["certificate_verified_by_this_function"] is False)
    empirical = call()
    check("empirical_diagnostic_not_winner_certificate", empirical["interpretation"] == "CONDITIONAL_DIAGNOSTIC" and empirical["familywise_winner_certified"] is False)
    check("zero_headroom_is_ineligible", ns["axis_headroom"](1., 1., 0.)["status"] == "INELIGIBLE")
    check("negative_headroom_can_remain_undetermined", ns["axis_headroom"](1., .9, .2)["status"] == "UNDETERMINED")
    check("clear_excess_bias_is_ineligible", ns["axis_headroom"](1., 1.2, .1)["status"] == "INELIGIBLE")
    logonly = call(log_B_upper=100.)
    check("log_domain_is_not_finite_integer_budget", logonly["N_total"] is None and all(a["integer_status"] == "LOG_DOMAIN_ONLY" and a["shots"] is None for a in logonly["axes"].values()))
    check("minimum_one_shot", all(a["shots"] == 1 for a in call(epsilon=100.)["axes"].values()))
    check("ceil_sufficient_at_synthetic_integer_boundary", all(a["shots"] == 2 for a in call(alpha_axis=2 / math.e)["axes"].values()))
    try:
        call(normalization_upper=2.)
    except ValueError as exc:
        check("normalization_log_binding_rejected", str(exc) == "NORMALIZATION_LOG_BINDING")
    else:
        raise AssertionError("normalization_log_binding_rejected")
    mixed = shots(dict(real=2., imag=0.), allowance, epsilon=math.sqrt(2), log_B_upper=0.,
                  evidence_kind="EMPIRICAL", evidence_ref="SYNTHETIC_ONLY")
    check("one_bad_axis_prevents_complex_budget", mixed["N_total"] is None and mixed["eligibility_under_declared_allowance"] == "INELIGIBLE")
    for invalid in (0., 1., -1., math.nan):
        try:
            call(alpha_axis=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid_alpha")
    check("invalid_alpha_rejected", True)
    metric_ns = pure_functions(blobs["d0_accounting"].decode(), ("rational", "weighted_metric"),
                              dict(Fraction=Fraction, statistics=statistics, math=math,
                                   METRICS=("RZ",), AXIS_CHANNEL={"cosine":"real", "sine":"imag"}))
    pairs = [dict(cosine={"RZ":1}, sine={"RZ":3}), dict(cosine={"RZ":3}, sine={"RZ":1})]
    metric = metric_ns["weighted_metric"](pairs, dict(cosine=1, sine=1), "RZ")
    check("paired_covariance_is_retained", metric["G_standard_error"] == 0. and metric["paired_samples"] == 2 and metric["wrapper_axes"] == 4)
    check("paired_se_is_not_confidence_interval", metric["uncertainty_scope"] == "PAIRED_SAMPLE_SE_NOT_CI")
    check("cost_winner_not_certified", metric["population_winner_certified"] is False)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blobs, catalog = {}, []
    for key, (commit, path) in SPECS.items():
        raw = git("show", commit + ":" + path)
        blobs[key] = raw
        catalog.append(dict(id=key, commit=commit, path=path, bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest(),
                            git_blob=git("rev-parse", commit + ":" + path).decode().strip(),
                            url="https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/" + commit + "/" + path))
    static = []
    for key in ("h4_native_shot", "h6_native_shot", "h8_native_shot"):
        for name in ("_number", "u_aware_shots"):
            static.append(dict(name=key + ":" + name + "_matches_frozen_D0",
                               passed=same_function(blobs[key].decode(), blobs["d0_shot"].decode(), name)))
    static.append(dict(name="native_headroom_matches_frozen_D0", passed=same_function(
        blobs["native_headroom"].decode(), blobs["d0_shot"].decode(), "axis_headroom")))
    for key in ("h4_native_signal", "h6_native_signal"):
        static.append(dict(name=key + ":compact_signal_matches_H8", passed=same_function(
            blobs[key].decode(), blobs["native_signal"].decode(), "compact_signal")))
    for key in ("h4_native_accounting", "h6_native_accounting"):
        static.append(dict(name=key + ":precision_rows_matches_H8", passed=same_function(
            blobs[key].decode(), blobs["native_accounting"].decode(), "precision_rows")))
    if not all(x["passed"] for x in static):
        raise AssertionError("SOURCE_FUNCTION_MISMATCH")
    metadata = {}
    d0 = decoded(blobs["d0_result"], SPECS["d0_result"][1])
    for scope, rows in d0.items():
        metadata[scope] = accounting_metadata(rows)
    for key in ("h6_strengthened_result", "h8_epsilon_result"):
        obj = decoded(blobs[key], SPECS[key][1])
        metadata[key] = accounting_metadata(obj["rows"] if isinstance(obj, dict) else obj)
    geometry = decoded(blobs["h4_geometry_result"], SPECS["h4_geometry_result"][1])
    metadata["h4_geometry_result"] = dict(axis_alpha=geometry["scope"]["axis_alpha"],
        saved_candidates=len(geometry["candidates"]),
        trajectory_count=geometry["scope"]["stochastic_trajectories"],
        normalizer_field="saved.normalization", explicit_u_subtraction=False,
        per_record_distribution_parameters_present=False,
        operation="SAVED_SCOPE_AND_SCHEMA_ONLY_NO_RECALCULATION")
    if geometry["scope"]["axis_alpha"] != .025:
        raise AssertionError("GEOMETRY_ALPHA")
    six = decoded(blobs["h8_six_axis_result"], SPECS["h8_six_axis_result"][1])
    six_groups = {}
    for kind in sorted({r["bias_kind"] for r in six}):
        updated = [r["updated"] for r in six if r["bias_kind"] == kind]
        flags = {key:dict(Counter(str(r.get(key, "NOT_EMBEDDED")) for r in updated))
                 for key in ("population_winner_certified", "rigorous_bias_guaranteed",
                             "population_mean_guaranteed", "rigorous_bias_guarantee")}
        if any(r.get(key) is True for r in updated for key in flags):
            raise AssertionError("SIX_AXIS_CERTIFICATION_PROMOTION")
        embedded = [r for r in updated if "accounting" in r]
        embedded_summary = accounting_metadata(embedded) if embedded else None
        if embedded_summary is not None and embedded_summary["errors"]:
            raise AssertionError("SIX_AXIS_EMBEDDED_ACCOUNTING")
        six_groups[kind] = dict(rows=len(updated), full_accounting_embedded=len(embedded),
            flags=flags, embedded_accounting=embedded_summary)
    metadata["h8_six_axis_result"] = dict(rows=len(six), groups=six_groups,
        guard_information="usual signed projection requires fixed parent accounting/source; noncancellation embeds accounting",
        saved_shots_present=all("saved_shots" in r for r in six),
        operation="SAVED_FIELD_SCHEMA_ONLY_NO_RECALCULATION")
    h6_rows = decoded(blobs["h6_strengthened_result"], SPECS["h6_strengthened_result"][1])["rows"]
    missing_rows = [r for r in h6_rows if r.get("accounting") is None]
    metadata["h6_strengthened_result"]["missing_classification"] = dict(Counter(
        "/".join((r["method"],r["bias_rule"],str(r["eligible"]),str(r.get("G_exact"))))
        for r in missing_rows))
    if any(r["eligible"] is not None or r.get("G_exact") is not None for r in missing_rows):
        raise AssertionError("UNRESOLVED_HAS_FINITE_COST")
    failures = {k:v["errors"] for k,v in metadata.items() if v.get("errors")}
    if failures:
        raise AssertionError(json.dumps(failures))
    checks = synthetic_checks(blobs)
    result = dict(schema="PR_ALPHA_AUDIT_METADATA_V1", date_JST="2026-10-18", base_commit=BASE,
        source_catalog=catalog, static_function_checks=static, saved_metadata=metadata,
        synthetic_checks=checks, scientific_execution=dict(new_signal=0, new_sampling=0,
            new_binomial_probability=0, saved_shot_recalculation=0, saved_cost_reaccounting=0,
            circuit_build=0, compile=0, NPZ_load=0),
        author_checks_only=True, independent_review=False,
        interpretation="TECHNICAL_AUDIT_NOT_UNCONDITIONAL_SCIENTIFIC_CERTIFICATE",
        next_stage_authorized=False, mandatory_stop=True)
    (OUT / "technical_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(status="PASS_METADATA_AND_SYNTHETIC_CHECKS", source_files=len(catalog),
                         static_checks=len(static), synthetic_checks=len(checks),
                         scopes={k:v.get("rows",v.get("saved_candidates")) for k,v in metadata.items()})))


if __name__ == "__main__":
    main()

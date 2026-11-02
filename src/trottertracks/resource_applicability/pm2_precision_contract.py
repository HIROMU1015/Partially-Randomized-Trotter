"""PM-2 preparation only: fixed saved JSON inventory, no precision evaluation.

No molecular/science imports, directory discovery, embedded-path access, sweep,
ranking, plotting, or production analysis entry point is provided here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

EVIDENCE_COMMIT = "194cc604b90c56a0e7e949b91b064a4bcfc846da"
STATUS = "PM2_PRECISION_CONTRACT_FROZEN_ANALYSIS_NOT_AUTHORIZED"
COMPLETE = "PM2_PRECISION_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW"
FAILURE = "IMPLEMENTATION_GATE_FAILED"
METRICS = ("rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size")
AXES = {"real": "cosine", "imag": "sine"}
INPUTS = {
    "m1_signal": (
        "artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_a_result_v1.json",
        "1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086"),
    "m1_compile": (
        "artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json",
        "71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4"),
    "pm1": (
        "artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result.json",
        "9305857873602d6bc4f45fbc78c4903911d083156620df01e9b23f00e7fdf05b"),
    "m2": (
        "artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json",
        "f41a92beb57e59cddc8c063b061c40acd4da50cb76ac0698efc2bce004937931"),
}
COMMON_FIVE = (
    "B2-rank3-q1-r4-K2", "B2-rank3-q1-r8-K2", "B0-rank6-q1-r0-K0",
    "B1-rank12-q1-r0-K0", "B3-rank0-q8-r32-K4",
)
ZERO_ACTIONS = (
    "precision_sweep", "precision_work_evaluations", "P_envelope_evaluations",
    "rankings", "new_signal_evaluations", "trajectory_sampling", "circuit_build",
    "compile", "molecular_data_access", "runtime_cache_access", "ground_state_solves",
    "quantum_shots", "gpu_operations",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def verified_bytes(current, committed, expected):
    require(hashlib.sha256(current).hexdigest() == expected, "saved input SHA mismatch")
    require(current == committed, "saved input differs from evidence blob")


def load_inputs(root):
    """Exactly four allowed JSON files. No paths contained in them are followed."""
    values, audit = {}, {}
    for name, (relative, expected) in INPUTS.items():
        require(relative.endswith(".json") and ".runtime" not in Path(relative).parts,
                "non-text input forbidden")
        data = (root / relative).read_bytes()
        blob = subprocess.check_output(["git", "show", f"{EVIDENCE_COMMIT}:{relative}"], cwd=root)
        verified_bytes(data, blob, expected)
        values[name] = json.loads(data)
        audit[name] = {"path": relative, "bytes": len(data), "sha256": expected,
                       "evidence_commit": EVIDENCE_COMMIT, "commit_blob_identical": True}
    return values, audit


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def pair_inventory(axes, random):
    expected = 32 if random else 1
    pairs = []
    for axis in AXES.values():
        record = axes[axis]
        require(record["status"] == "complete", "incomplete axis")
        require(not record["trajectory_records_truncated"], "missing saved cost samples")
        rows = record["retained_trajectory_records"]
        require(len(rows) == expected, "unexpected saved trajectory count")
        pairs.append([(r["trajectory_index"], r["trajectory_seed"]) for r in rows])
        for metric in METRICS:
            require(finite_nonnegative(record["metric_statistics"][metric]["mean"]), "invalid compiled mean")
    require(pairs[0] == pairs[1], "cosine/sine trajectory pairing mismatch")
    require(len(set(pairs[0])) == expected, "duplicate trajectory identities")
    return {"sample_count": expected, "paired_axis_identities_identical": True,
            "complete_saved_samples": True, "pair_identity_fingerprint": fingerprint(pairs[0])}


def ledger_entry(dataset, signal, axes, signal_source, cost_source, index, cost_index, candidate=None):
    candidate = candidate or signal["candidate"]
    require(candidate["candidate_fingerprint"] == signal["candidate_fingerprint"], "candidate fingerprint mismatch")
    for axis in AXES:
        require(finite_nonnegative(signal["axis_bias"][axis]), "invalid saved bias")
    require(finite_nonnegative(signal["normalization_multiplier"]) and signal["normalization_multiplier"] >= 1,
            "invalid saved normalization")
    pairing = pair_inventory(axes, candidate["method"] in {"B2", "B3"})
    return {
        "dataset": dataset, "candidate_id": candidate["candidate_id"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "parameters": {k: candidate[k] for k in ("method", "rank", "q", "r", "K", "T", "delta")},
        "signal_source": signal_source, "compile_source": cost_source,
        "signal_row_index": index, "compile_row_index": cost_index,
        "signal_record_fingerprint": fingerprint(signal),
        "reference_accuracy_eligible": signal["accuracy_eligible"],
        "saved_field_coverage": {"axis_bias": True, "normalization": True, "axis_metric_means": True,
                                  "source_values_copied_or_reevaluated": False},
        "cost_sample_inventory": pairing,
        "cost_uncertainty_source": "saved_paired_32_samples" if pairing["sample_count"] == 32 else "deterministic_exact",
    }


def candidate_inventory(values):
    """Identity/field coverage only. Never reevaluate shots, costs, or a winner."""
    signals = values["m1_signal"]["signal_records"]
    costs = values["m1_compile"]["compile_map"]
    require(len(signals) == len(costs) == 210, "M1 domain must contain all 210 records")
    lookup = {r["candidate"]["candidate_id"]: (i, r) for i, r in enumerate(costs)}
    require(len(lookup) == 210, "duplicate M1 candidate")
    development = []
    for index, signal in enumerate(signals):
        cost_index, cost = lookup[signal["candidate_id"]]
        require(cost["candidate"] == signal["candidate"], "signal/cost candidate mismatch")
        require(cost["signal_record_fingerprint"] == fingerprint(signal), "signal/cost record mismatch")
        require(cost["axis_shots"] == signal["axis_shots"], "saved axis shots mismatch")
        development.append(ledger_entry("development", signal, cost["compiled_axes"], "m1_signal", "m1_compile", index, cost_index))
    require(values["pm1"]["status"] == "PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW", "PM1 completion required")
    new = values["pm1"]["candidate_records"]
    require(len(new) == 8, "PM1 domain must contain eight records")
    for index, row in enumerate(new):
        require(row["signal_cost_candidate_fingerprint"] == row["candidate"]["candidate_fingerprint"], "PM1 cost identity mismatch")
        development.append(ledger_entry("development", row["signal"], row["axes"], "pm1", "pm1", index, index, row["candidate"]))
    require(len({r["candidate_id"] for r in development}) == 218, "duplicate development candidate")
    require(len({r["candidate_fingerprint"] for r in development}) == 218, "duplicate development fingerprint")
    require(sum(r["reference_accuracy_eligible"] for r in development) == 214, "reference eligibility changed")
    all_candidates = [r["candidate"] for r in signals] + [r["candidate"] for r in new]
    require({r["hamiltonian_hash"] for r in all_candidates} ==
            {"de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424"}, "development Hamiltonian mismatch")
    require(len({r["state_vector_hash"] for r in all_candidates}) == 1, "development state mismatch")

    require(values["m2"]["status"] == "TRANSFER_SUPPORTED", "saved M2 status changed")
    transfer = []
    rows = values["m2"]["candidate_results"]
    require(len(rows) == 5 and {r["candidate_id"] for r in rows} == set(COMMON_FIVE), "M2 fixed-five domain changed")
    for index, row in enumerate(rows):
        signal = row["signal"]
        require(row["execution_candidate_fingerprint"] == signal["candidate_fingerprint"], "M2 candidate mismatch")
        require(all(finite_nonnegative(signal["axis_bias"][a]) for a in AXES), "invalid M2 saved bias")
        require(finite_nonnegative(signal["normalization_multiplier"]) and signal["normalization_multiplier"] >= 1,
                "invalid M2 normalization")
        require(all(finite_nonnegative(row["axis_one_shot_compiled_means"][a][m])
                    for a in AXES.values() for m in METRICS), "invalid M2 compiled mean")
        paired = row["compiled"]["paired_trajectory_rows"]
        expected = 32 if row["method"] in {"B2", "B3"} else 1
        require(len(paired) == expected, "M2 saved sample count changed")
        require(len({(r["trajectory_index"], r["trajectory_seed"]) for r in paired}) == expected, "M2 duplicate trajectory")
        require(all(r["axes"]["cosine"]["shared_evolution_fingerprint"] ==
                    r["axes"]["sine"]["shared_evolution_fingerprint"] for r in paired), "M2 unpaired axis evolution")
        transfer.append({"dataset": "transfer_fixed_five", "candidate_id": row["candidate_id"],
            "candidate_fingerprint": row["execution_candidate_fingerprint"],
            "development_candidate_fingerprint": row["development_candidate_fingerprint"],
            "parameters": {k: signal["candidate"][k] for k in ("method", "rank", "q", "r", "K", "T", "delta")},
            "signal_source": "m2", "compile_source": "m2",
            "signal_row_index": index, "compile_row_index": index, "signal_record_fingerprint": fingerprint(signal),
            "reference_accuracy_eligible": signal["accuracy_eligible"],
            "saved_field_coverage": {"axis_bias": True, "normalization": True, "axis_metric_means": True,
                                      "source_values_copied_or_reevaluated": False},
            "cost_sample_inventory": {"sample_count": expected, "paired_axis_identities_identical": True,
                                      "complete_saved_samples": True},
            "cost_uncertainty_source": "saved_paired_32_samples" if expected == 32 else "deterministic_exact"})
    return {"development": development, "transfer_fixed_five": transfer}


def contract_settings():
    return {
        "schema_version": "track_a_pm2_precision_contract_v1", "status": STATUS,
        "analysis_label": "POSTHOC_SAVED_VALUES_ONLY", "evidence_commit": EVIDENCE_COMMIT,
        "RQ": "Accuracy eligibility versus measurement-inclusive resource choice in the fixed DF-prefix second-order finite grid",
        "domains": {"development": 218, "reference_eligible_development": 214, "transfer_fixed_five": 5,
                    "retain_original_ineligible": True, "cross_geometry_optimum_comparison": False},
        "epsilon_design": {"reference": 0.05, "minimum": 0.005, "maximum": 0.1,
            "log_uniform_base_points": 301, "reference_inserted_exactly": True,
            "base_point_definition": "0.005*(0.1/0.005)**(i/300), i=0..300; endpoints exact; insert 0.05; sort and deduplicate",
            "rationale": "ten times tighter to twice looser than the original coherent-signal task, not selected from resulting rankings",
            "adaptive_precision_sampling": False, "continuous_coverage_claim": False,
            "epsilon_min": "sqrt(2)*max(axis_bias)", "eligibility": "epsilon>epsilon_min AND all(epsilon/sqrt(2)-axis_bias>0)",
            "equality_boundary_eligible": False},
        "accounting": {"alpha_axis": 0.025, "axis_map": AXES,
            "axis_shots": "ceil(2*B**2/(epsilon/sqrt(2)-b_axis)**2*log(2/alpha_axis))",
            "work": "sum_axis(N_axis*(saved_axis_compiled_mean+P))",
            "ineligible_axis_shots": None, "ineligible_matched_work": None,
            "missing_is_zero": False, "fixed_C_eff_allowed": False,
            "shot_rule": "sufficient corrected Hoeffding bound, not minimum achievable shots or task impossibility",
            "reference_reproduction": "exact integer axis shots and eligibility; work relative tolerance 1e-12 / absolute 1e-6"},
        "metrics": {"primary": "rz_count", "point_pareto": list(METRICS),
                    "strict_dominance": "all <= and at least one <; retain ties", "research_GO_threshold": None},
        "P_design": {"minimum": 0, "maximum": None, "unit": "common RZ-equivalent preparation cost per shot",
            "primary_only": True, "method": "analytic affine lower envelope at each fixed epsilon, all P>=0, retain boundary ties",
            "P_grid": False, "actual_preparation_compile": False, "candidate_specific_preparation": False},
        "uncertainty": {"sample_source": "saved paired cosine/sine trajectory costs only",
            "sample_covariance_denominator": "n-1 (unbiased sample variance/covariance)",
            "work_SE": "sqrt((N_real**2*scc+N_imag**2*sss+2*N_real*N_imag*scs)/n)",
            "P_changes_sampling_SE": False, "engineering_interval": "point +/- 2SE",
            "deterministic_variance": "exact nonrandom cost; absent SE is not missing stochastic variance",
            "formal_CI": False, "familywise_winner_guarantee": False,
            "interval_ranking_rule": "overlap does not resolve a winner; point frontier remains descriptive"},
        "future_outputs": ["summary.json", "precision_ledger.csv", "eligibility_boundaries.csv", "P_envelope.csv",
            "representative_decomposition.csv", "claim_audit.json", "report.md", "manifest.json"],
        "output_columns": {
            "precision_ledger.csv": ["dataset", "candidate_id", "candidate_fingerprint", "epsilon", "epsilon_min",
                "accuracy_eligible", "axis_bias_real", "axis_bias_imag", "normalization", "allowance_real", "allowance_imag",
                "N_real", "N_imag", "N_total", "primary_RZ_P0", "primary_SE", "point_frontier",
                "rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size", "missing_reason"],
            "eligibility_boundaries.csv": ["dataset", "candidate_id", "epsilon_min", "strict_boundary", "inside_display_range"],
            "P_envelope.csv": ["dataset", "epsilon", "candidate_id", "G_RZ_P0", "N_total", "P_min", "P_max", "boundary_tie"],
            "representative_decomposition.csv": ["dataset", "epsilon", "method", "candidate_id", "N_total", "normalization_squared",
                "bias_real", "bias_imag", "remaining_headroom_real", "remaining_headroom_imag", "mean_RZ_cosine", "mean_RZ_sine", "G_RZ_P0"],
        },
        "representatives": "all exact point-minimum ties per method at every fixed epsilon; no handpicked favorable examples",
        "missing_policy": "null in JSON; MISSING in CSV; no eligible registered candidate is not method impossibility",
        "claim_audit_required": ["PM0_attribution_corrections_preserved", "PM1_pure_bias_decomposition_missing",
            "M2_only_original_five", "no_general_optimum", "no_energy_RPE_total_cost", "posthoc_not_preregistered_science"],
        "future_analysis_caps": {"cpu_processes": 1, "BLAS_threads": 1, "epsilon_points": 302,
            "candidate_epsilon_records": 67346, "additional_candidates": 0, "new_signal_evaluations": 0,
            "new_cost_samples": 0, "circuit_build_compile": 0, "protected_data_access": 0,
            "gpu_operations": 0, "quantum_shots": 0},
        "future_result_statuses": [COMPLETE, FAILURE],
        "permissions": {"preparation_input_inventory": True, "precision_analysis": False, "new_science": False,
                        "automatic_research_decision": False, "next_stage": False},
        "mandatory_stop": True, "next_stage_authorized": False, "research_decision": None,
        "source_freeze_required_before_analysis": True,
    }


def plan_schema():
    settings = contract_settings()
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "additionalProperties": False, "required": list(settings),
            "properties": {k: {"const": v} for k, v in settings.items()}}


def reserved_result_schema():
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "status", "analysis_label", "preparation_manifest_sha256",
                     "input_identity", "domain_counts", "reference_reproduction", "output_files",
                     "failure_reason", "new_science_counts", "mandatory_stop", "next_stage_authorized", "research_decision"],
        "properties": {
            "schema_version": {"const": "track_a_pm2_precision_result_v1"},
            "status": {"enum": [COMPLETE, FAILURE]}, "analysis_label": {"const": "POSTHOC_SAVED_VALUES_ONLY"},
            "preparation_manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "input_identity": {"type": "object"}, "domain_counts": {"const": {"development": 218, "transfer_fixed_five": 5}},
            "reference_reproduction": {"type": ["object", "null"]}, "output_files": {"type": "array"},
            "failure_reason": {"type": ["string", "null"]},
            "new_science_counts": {"type": "object", "additionalProperties": False,
                "required": list(ZERO_ACTIONS[4:]), "properties": {k: {"const": 0} for k in ZERO_ACTIONS[4:]}},
            "mandatory_stop": {"const": True}, "next_stage_authorized": {"const": False}, "research_decision": {"const": None}},
        "allOf": [{"if": {"properties": {"status": {"const": COMPLETE}}},
                   "then": {"properties": {"failure_reason": {"const": None}, "reference_reproduction": {
                       "type": "object", "required": ["passed"], "properties": {"passed": {"const": True}}}}},
                   "else": {"properties": {"failure_reason": {"type": "string", "minLength": 1}}}}]}


def build_bundle(root):
    values, audit = load_inputs(root)
    inventory = candidate_inventory(values)
    settings = contract_settings()
    return {
        "contract_settings_v1.json": settings,
        "plan_schema_v1.json": plan_schema(), "reserved_result_schema_v1.json": reserved_result_schema(),
        "input_identity_v1.json": audit,
        "candidate_inventory_v1.json": {"evidence_commit": EVIDENCE_COMMIT,
            "inventory_fingerprint": fingerprint(inventory), "counts": {k: len(v) for k, v in inventory.items()}, **inventory},
        "preparation_audit_v1.json": {"status": STATUS, "input_files_verified": 4,
            "development_candidates": 218, "reference_eligible_development": 214, "transfer_candidates": 5,
            "counts": {k: 0 for k in ZERO_ACTIONS}, "analysis_authorized": False,
            "mandatory_stop": True, "next_stage_authorized": False, "research_decision": None,
            "provenance": "local uncommitted preparation; not immutable CI or independent reproduction"},
    }

"""Zero-compute contract for the PR-2 M2 held-out transfer.

The module reads only the committed M1-B1 result and its committed validation.
It fixes five development-selected configurations, transfer decision rules, a
future seed schedule, and resource caps.  It deliberately has no molecular,
NumPy, Qiskit, circuit-building, or compiler dependency and never opens the
held-out snapshot named in the plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pr2_matched_accuracy_m2_transfer_zero_compute_plan_v1"
STATUS = "M2_TRANSFER_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED"
SERIES_ID = "pr2-rebaseline-de7a5492-v1"

M1_B1_RESULT_RELATIVE = (
    "artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/"
    "pr2_matched_accuracy_m1_b1_compile_map_result_v2.json"
)
M1_B1_RESULT_SHA256 = (
    "71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4"
)
M1_B1_RESULT_FINGERPRINT = (
    "504d9c9089726800a291a8259e87b2d37c1fdea046263db6c9582bb659c77975"
)
M1_B1_VALIDATION_RELATIVE = (
    "artifacts/pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/"
    "pr2_matched_accuracy_m1_b1_result_validation_v1.json"
)
M1_B1_VALIDATION_SHA256 = (
    "c9a05babed99cd1e80eaec5b58e47f25d74513c7ba2e5a00775cfd5959c37a0f"
)
M1_B1_VALIDATION_FINGERPRINT = (
    "c3cf1c084ebfe343d576236de2803c9c69855e0247ca6a2d628c496ee0546214"
)
M1_B1_EVIDENCE_COMMIT = "8e0814e70c14ecf526444fac8a2142799610dc96"

HELD_OUT_RELATIVE_LITERAL = (
    "artifacts/pr2_s0_s1_validation/2026-09-28/"
    "h4_1p30_rank12_held_out_v1.npz"
)

METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)
PRIMARY_METRIC = "rz_count"
AXES = ("cosine", "sine")
TRANSFER_STATUSES = (
    "TRANSFER_SUPPORTED",
    "TRANSFER_NOT_SUPPORTED",
    "TRANSFER_INCONCLUSIVE",
    "IMPLEMENTATION_GATE_FAILED",
)
MATERIALITY_RATIO = 1.10
MAJOR_UNDERESTIMATE_FRACTION = 0.10
RATIO_INTERVAL_Z = 2.0
TRAJECTORIES_PER_RANDOM_CELL = 32
MASTER_SEED = 2026100401
MAXIMUM_WORKERS = 5

FROZEN_CANDIDATES = (
    {
        "candidate_id": "B2-rank3-q1-r4-K2",
        "candidate_fingerprint": "ff8e17f270d298339d53b9f47b8fc13189a91c30105498e405a642bfe5bf29ec",
        "method": "B2",
        "rank": 3,
        "q": 1,
        "r": 4,
        "K": 2,
        "role": "development_actual_pareto_partial",
    },
    {
        "candidate_id": "B2-rank3-q1-r8-K2",
        "candidate_fingerprint": "2a527a84ef1481d55ffd576634b4a4841282871ce6d6872ca72767971c7255f2",
        "method": "B2",
        "rank": 3,
        "q": 1,
        "r": 8,
        "K": 2,
        "role": "development_actual_pareto_partial",
    },
    {
        "candidate_id": "B0-rank6-q1-r0-K0",
        "candidate_fingerprint": "4893d918556d0daeec8dae3a1a5c6f97c598ab6d7efb7e4c7d714ee6400b6d52",
        "method": "B0",
        "rank": 6,
        "q": 1,
        "r": 0,
        "K": 0,
        "role": "development_best_discard_reference",
    },
    {
        "candidate_id": "B1-rank12-q1-r0-K0",
        "candidate_fingerprint": "5118c41031227fe7642bc243c875f815afae86eb716e4c1b31c31f7ea2c48c7c",
        "method": "B1",
        "rank": 12,
        "q": 1,
        "r": 0,
        "K": 0,
        "role": "development_best_full_deterministic_reference",
    },
    {
        "candidate_id": "B3-rank0-q8-r32-K4",
        "candidate_fingerprint": "9feb7b75a61651992917ebc41e6b410eb6c6942213237eb566c049cd884377a8",
        "method": "B3",
        "rank": 0,
        "q": 8,
        "r": 32,
        "K": 4,
        "role": "development_best_random_dominant_reference",
    },
)

COMPILER_IDENTITY = {
    "qiskit_version": "1.3.0",
    "basis_gates": ["rz", "sx", "x", "cx"],
    "optimization_level": 1,
    "transpiler_seed": 17,
    "coupling_map": None,
    "backend_name": None,
    "layout_method": None,
    "routing_method": None,
}

ZERO_SCIENCE_COUNTER_NAMES = (
    "development_npz_loads",
    "held_out_path_resolutions",
    "held_out_path_stats",
    "held_out_hash_reads",
    "held_out_npz_loads",
    "signal_evaluations",
    "trajectory_samples",
    "occurrence_samples",
    "circuits_built",
    "compiler_invocations",
    "full_wrappers_compiled",
    "quantum_shots_executed",
    "molecular_calculations",
    "candidate_searches",
    "gpu_queries",
    "gpu_allocations",
    "gpu_kernels",
)


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_inputs(
    result: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    result_sha256: str,
    validation_sha256: str,
) -> None:
    _require(result_sha256 == M1_B1_RESULT_SHA256, "M1-B1 result SHA-256 differs")
    _require(
        validation_sha256 == M1_B1_VALIDATION_SHA256,
        "M1-B1 validation SHA-256 differs",
    )
    _require(
        result.get("schema_version") == "pr2_matched_accuracy_m1_b1_result_v2",
        "unexpected M1-B1 result schema",
    )
    _require(
        result.get("status") == "M1_B1_COMPILE_MAP_COMPLETE_AWAITING_REVIEW",
        "M1-B1 result is not complete and awaiting review",
    )
    _require(
        result.get("result_fingerprint") == M1_B1_RESULT_FINGERPRINT,
        "M1-B1 result fingerprint differs",
    )
    _require(result.get("held_out_accessed") is False, "M1-B1 accessed held-out")
    _require(result.get("transfer_executed") is False, "M1-B1 executed transfer")
    _require(
        validation.get("schema_version")
        == "pr2_matched_accuracy_m1_b1_result_validation_v1",
        "unexpected M1-B1 validation schema",
    )
    _require(
        validation.get("validation_fingerprint") == M1_B1_VALIDATION_FINGERPRINT,
        "M1-B1 validation fingerprint differs",
    )
    review = validation.get("external_research_review", {})
    _require(
        review.get("decision") == "CONTINUE_RESOURCE_STUDY",
        "M1-B1 review did not select CONTINUE_RESOURCE_STUDY",
    )
    _require(
        validation.get("integrity", {}).get("all_gates_passed") is True,
        "M1-B1 validation integrity gates did not all pass",
    )


def _record_index(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = result.get("compile_map")
    _require(isinstance(records, list) and len(records) == 210, "invalid M1-B1 map")
    index: dict[str, Mapping[str, Any]] = {}
    for record in records:
        _require(isinstance(record, Mapping), "invalid M1-B1 record")
        candidate = record.get("candidate")
        _require(isinstance(candidate, Mapping), "missing M1-B1 candidate")
        identifier = str(candidate.get("candidate_id"))
        _require(identifier not in index, "duplicate M1-B1 candidate id")
        index[identifier] = record
    return index


def _development_prediction(record: Mapping[str, Any]) -> dict[str, Any]:
    axes = record["compiled_axes"]
    return {
        "accuracy_eligible": bool(record["accuracy_eligible"]),
        "axis_shots": {
            "real": int(record["axis_shots"]["real"]),
            "imag": int(record["axis_shots"]["imag"]),
        },
        "signal_record_fingerprint": str(record["signal_record_fingerprint"]),
        "axis_one_shot_compiled_cost": {
            axis: {
                metric: float(axes[axis]["metric_statistics"][metric]["mean"])
                for metric in METRICS
            }
            for axis in AXES
        },
        "work_by_metric": {
            metric: float(record["matched_accuracy_compiled_work_no_state_preparation"][metric])
            for metric in METRICS
        },
    }


def frozen_candidate_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    index = _record_index(result)
    frozen: list[dict[str, Any]] = []
    for expected in FROZEN_CANDIDATES:
        record = index.get(str(expected["candidate_id"]))
        _require(record is not None, f"missing frozen candidate: {expected['candidate_id']}")
        candidate = record["candidate"]
        for field in ("candidate_fingerprint", "method", "rank", "q", "r", "K"):
            _require(
                candidate.get(field) == expected[field],
                f"frozen candidate field differs: {expected['candidate_id']} {field}",
            )
        _require(record.get("accuracy_eligible") is True, "development candidate is ineligible")
        configuration = {
            "method": candidate["method"],
            "method_semantics": candidate["method_semantics"],
            "mode": candidate["mode"],
            "rank": candidate["rank"],
            "T": candidate["T"],
            "q": candidate["q"],
            "delta": candidate["delta"],
            "r": candidate["r"],
            "K": candidate["K"],
            "outer_formula": candidate["outer_formula"],
            "identity_policy": candidate["identity_policy"],
            "coefficient_atol": candidate["coefficient_atol"],
            "wrapper_semantics": candidate["wrapper_semantics"],
            "compiler_identity": candidate["compiler_identity"],
            "seed_policy": candidate["seed_policy"],
        }
        frozen.append(
            {
                **dict(expected),
                "transfer_configuration": configuration,
                "transfer_configuration_fingerprint": fingerprint(configuration),
                "development_prediction": _development_prediction(record),
            }
        )
    _require(len(frozen) == 5, "transfer contract does not contain exactly five candidates")
    return frozen


def trajectory_seed(configuration_fingerprint: str, trajectory_index: int) -> int:
    if not 0 <= trajectory_index < TRAJECTORIES_PER_RANDOM_CELL:
        raise ValueError("trajectory index outside the frozen transfer range")
    identity = {
        "policy": "pr2_m2_transfer_configuration_trajectory_sha256_v1",
        "master_seed": MASTER_SEED,
        "transfer_configuration_fingerprint": configuration_fingerprint,
        "trajectory_index": trajectory_index,
    }
    return int.from_bytes(hashlib.sha256(canonical_json(identity)).digest()[:8], "big")


def predicted_work(
    development_axis_cost: Mapping[str, Mapping[str, float]],
    held_out_axis_shots: Mapping[str, int],
    metric: str,
) -> float:
    if metric not in METRICS:
        raise ValueError("unsupported compiled metric")
    return (
        int(held_out_axis_shots["real"])
        * float(development_axis_cost["cosine"][metric])
        + int(held_out_axis_shots["imag"])
        * float(development_axis_cost["sine"][metric])
    )


def underestimate_fraction(*, predicted: float, actual: float) -> float:
    if not math.isfinite(predicted) or predicted <= 0.0:
        raise ValueError("predicted work must be finite and positive")
    if not math.isfinite(actual) or actual < 0.0:
        raise ValueError("actual work must be finite and non-negative")
    return max(0.0, actual / predicted - 1.0)


def materiality_ratio_interval(
    *,
    b2_work: float,
    b2_standard_error: float,
    endpoint_work: float,
    endpoint_standard_error: float,
) -> dict[str, float]:
    if b2_work <= 0.0 or endpoint_work <= 0.0:
        raise ValueError("ratio work values must be positive")
    ratio = b2_work / endpoint_work
    relative = math.sqrt(
        (b2_standard_error / b2_work) ** 2
        + (endpoint_standard_error / endpoint_work) ** 2
    )
    standard_error = ratio * relative
    return {
        "point": ratio,
        "standard_error": standard_error,
        "lower_2se": max(0.0, ratio - RATIO_INTERVAL_Z * standard_error),
        "upper_2se": ratio + RATIO_INTERVAL_Z * standard_error,
    }


def classify_transfer(
    candidate_results: Sequence[Mapping[str, Any]],
    *,
    implementation_gate_passed: bool = True,
) -> dict[str, Any]:
    """Apply the frozen terminal decision to already-computed candidate summaries."""
    if not implementation_gate_passed:
        return {"status": "IMPLEMENTATION_GATE_FAILED", "primary_ratio": None}
    by_id = {str(item["candidate_id"]): item for item in candidate_results}
    expected_ids = {str(item["candidate_id"]) for item in FROZEN_CANDIDATES}
    if set(by_id) != expected_ids:
        raise ValueError("classification requires exactly the five frozen candidates")
    eligible_b2 = [
        item
        for item in candidate_results
        if item["method"] == "B2" and item["accuracy_eligible"] is True
    ]
    if not eligible_b2:
        return {"status": "TRANSFER_NOT_SUPPORTED", "primary_ratio": None}
    endpoints = [
        item
        for item in candidate_results
        if item["method"] in {"B0", "B1", "B3"}
        and item["accuracy_eligible"] is True
    ]
    if not endpoints:
        return {"status": "TRANSFER_INCONCLUSIVE", "primary_ratio": None}
    usable_b2 = [item for item in eligible_b2 if not item["major_cost_underestimate"]]
    if not usable_b2:
        return {"status": "TRANSFER_NOT_SUPPORTED", "primary_ratio": None}
    best_b2 = min(usable_b2, key=lambda item: float(item["primary_work"]))
    best_endpoint = min(endpoints, key=lambda item: float(item["primary_work"]))
    ratio = materiality_ratio_interval(
        b2_work=float(best_b2["primary_work"]),
        b2_standard_error=float(best_b2["primary_standard_error"]),
        endpoint_work=float(best_endpoint["primary_work"]),
        endpoint_standard_error=float(best_endpoint["primary_standard_error"]),
    )
    if any(item["point_six_metric_pareto"] for item in usable_b2):
        status = "TRANSFER_SUPPORTED"
    elif ratio["upper_2se"] <= MATERIALITY_RATIO:
        status = "TRANSFER_SUPPORTED"
    elif ratio["lower_2se"] > MATERIALITY_RATIO:
        status = "TRANSFER_NOT_SUPPORTED"
    else:
        status = "TRANSFER_INCONCLUSIVE"
    return {
        "status": status,
        "primary_ratio": ratio,
        "best_b2_candidate_id": best_b2["candidate_id"],
        "best_endpoint_candidate_id": best_endpoint["candidate_id"],
    }


def build_plan(
    *,
    result: Mapping[str, Any],
    validation: Mapping[str, Any],
    result_sha256: str,
    validation_sha256: str,
    source_commit: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    validate_inputs(
        result,
        validation,
        result_sha256=result_sha256,
        validation_sha256=validation_sha256,
    )
    _require(
        len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit),
        "source commit must be a full lowercase Git object name",
    )
    candidates = frozen_candidate_records(result)
    for item in candidates:
        if item["method"] in {"B2", "B3"}:
            item["future_trajectory_seeds"] = [
                trajectory_seed(item["transfer_configuration_fingerprint"], index)
                for index in range(TRAJECTORIES_PER_RANDOM_CELL)
            ]
        else:
            item["future_trajectory_seeds"] = []
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "series_id": SERIES_ID,
        "status": STATUS,
        "source_commit": source_commit,
        "source_hashes": dict(sorted(source_hashes.items())),
        "input_identity": {
            "m1_b1_evidence_commit": M1_B1_EVIDENCE_COMMIT,
            "m1_b1_result_path": M1_B1_RESULT_RELATIVE,
            "m1_b1_result_sha256": result_sha256,
            "m1_b1_result_fingerprint": M1_B1_RESULT_FINGERPRINT,
            "m1_b1_validation_path": M1_B1_VALIDATION_RELATIVE,
            "m1_b1_validation_sha256": validation_sha256,
            "m1_b1_validation_fingerprint": M1_B1_VALIDATION_FINGERPRINT,
            "m1_b1_research_decision": "CONTINUE_RESOURCE_STUDY",
        },
        "provisional_claim": {
            "text": (
                "For the H4 1.00 Angstrom development condition, matched-accuracy "
                "evaluation and actual full-wrapper compilation changed the design "
                "choice relative to fixed-q/proxy comparison, and an intermediate "
                "DF-prefix partial-randomization region remained on the resource frontier."
            ),
            "scope": "H4 linear 1.00 Angstrom, STO-3G, DF rank 12, 8 qubits, T=0.8",
            "general_optimum_claimed": False,
            "rank3_q1_general_optimum_claimed": False,
            "r4_vs_r8_exact_winner_claimed": False,
        },
        "held_out_target": {
            "system": "H4 linear",
            "geometry_angstrom": 1.30,
            "basis": "STO-3G",
            "df_rank": 12,
            "sector_qubits": 8,
            "repository_relative_path_literal": HELD_OUT_RELATIVE_LITERAL,
            "path_resolved_during_planning": False,
            "path_statted_during_planning": False,
            "file_hashed_during_planning": False,
            "npz_loaded_during_planning": False,
            "byte_identity_verification_deferred_until_authorized_execution": True,
        },
        "frozen_candidates": candidates,
        "accuracy_and_shot_rule": {
            "complex_accuracy": 0.05,
            "axis_accuracy_formula": "0.05/sqrt(2)",
            "axis_failure_allocation": 0.025,
            "corrected_mean_formula": "nu_axis=B_total*mu_axis",
            "axis_bias_formula": "abs(nu_axis-exact_target_axis)",
            "axis_allowance_formula": "0.05/sqrt(2)-axis_bias",
            "axis_shots_formula": "ceil(2*B_total^2/allowance^2*log(2/0.025))",
            "held_out_shots_recomputed": True,
            "method_rank_q_r_K_retuned": False,
            "nonpositive_allowance_is_ineligible": True,
            "development_eligible_held_out_ineligible_is_accuracy_misclassification": True,
        },
        "compiled_cost_rule": {
            "metrics": list(METRICS),
            "primary_metric": PRIMARY_METRIC,
            "primary_formula": "N_real*E[C_cosine_rz]+N_imag*E[C_sine_rz]",
            "secondary_metrics": [metric for metric in METRICS if metric != PRIMARY_METRIC],
            "point_frontier_uses_all_six_metrics": True,
            "state_preparation_sensitivity_formula": "G_rz(P)=G_rz(0)+(N_real+N_imag)*P, P>=0",
            "state_preparation_lower_envelope_is_secondary": True,
        },
        "prediction_and_underestimate_rule": {
            "prediction": (
                "held-out analytic axis shots multiplied by the corresponding "
                "development one-shot expected compiled cost"
            ),
            "relative_underestimate_formula": "max(0,actual/predicted-1)",
            "major_underestimate_primary_metric": PRIMARY_METRIC,
            "major_underestimate_threshold_fraction": MAJOR_UNDERESTIMATE_FRACTION,
            "major_underestimate_comparison": "strictly_greater_than_threshold",
            "secondary_metric_underestimates_reported_but_not_terminal": True,
            "major_underestimate_disqualifies_that_B2_candidate_from_support": True,
        },
        "materiality_and_uncertainty_rule": {
            "point_ratio_formula": "min_eligible_B2_G_rz/min_eligible_B0_B1_B3_G_rz",
            "materially_competitive_if_ratio_at_most": MATERIALITY_RATIO,
            "ratio_interval_method": "delta_method_independent_candidate_2SE_engineering_interval",
            "ratio_interval_z": RATIO_INTERVAL_Z,
            "formal_confidence_interval_claimed": False,
        },
        "terminal_decision_rule": {
            "allowed_statuses": list(TRANSFER_STATUSES),
            "supported": (
                "at least one accuracy-eligible, non-majorly-underestimated B2 is on "
                "the point six-metric Pareto frontier, or the upper 2SE primary ratio "
                "is <=1.10"
            ),
            "not_supported": (
                "no B2 remains accuracy-eligible, every eligible B2 has a major primary "
                "cost underestimate, or no usable B2 is point-Pareto and the lower 2SE "
                "primary ratio is >1.10"
            ),
            "inconclusive": (
                "required endpoints are unavailable or the 2SE ratio straddles 1.10 "
                "without a usable point-Pareto B2"
            ),
            "implementation_gate_failed": "any source, identity, numerical, or resource gate fails",
            "runner_authorizes_next_stage": False,
            "mandatory_stop_after_terminal_status": True,
        },
        "seed_policy": {
            "name": "pr2_m2_transfer_configuration_trajectory_sha256_v1",
            "master_seed": MASTER_SEED,
            "axis_excluded_from_random_trajectory": True,
            "same_random_trajectory_shared_by_cosine_and_sine": True,
            "seeds_fixed_before_held_out_access": True,
        },
        "resource_caps": {
            "candidate_count": 5,
            "random_candidate_count": 3,
            "deterministic_or_discard_candidate_count": 2,
            "trajectories_per_random_candidate": TRAJECTORIES_PER_RANDOM_CELL,
            "random_full_wrappers": 3 * TRAJECTORIES_PER_RANDOM_CELL * len(AXES),
            "deterministic_or_discard_full_wrappers": 2 * len(AXES),
            "total_full_wrappers": 196,
            "maximum_process_workers": MAXIMUM_WORKERS,
            "blas_threads_per_worker": 1,
            "additional_trajectories": 0,
            "candidate_searches": 0,
            "held_out_snapshot_loads": 1,
        },
        "prohibitions": {
            "candidate_addition_removal_or_replacement": True,
            "held_out_rank_q_r_K_search": True,
            "development_r4_r8_winner_refinement": True,
            "additional_96_trajectories": True,
            "adaptive_trajectory_extension": True,
            "new_geometry_molecule_basis_or_pf": True,
            "h5_h6_h12_or_long_rpe": True,
            "automatic_research_direction_decision": True,
        },
        "authorization": {
            "zero_compute_plan_generation_authorized": True,
            "held_out_access_authorized": False,
            "signal_evaluation_authorized": False,
            "trajectory_sampling_authorized": False,
            "circuit_build_authorized": False,
            "compile_authorized": False,
            "transfer_execution_authorized": False,
            "next_stage_authorized": False,
            "automatic_next_stage": None,
        },
        "planning_input_reads": {
            "m1_b1_result_json": 1,
            "m1_b1_validation_json": 1,
        },
        "zero_science_counters": {name: 0 for name in ZERO_SCIENCE_COUNTER_NAMES},
    }
    body["plan_fingerprint"] = fingerprint(body)
    validate_plan(body)
    return body


def validate_plan(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "unexpected plan schema")
    _require(payload.get("status") == STATUS, "unexpected plan status")
    body = {key: value for key, value in payload.items() if key != "plan_fingerprint"}
    _require(fingerprint(body) == payload.get("plan_fingerprint"), "plan fingerprint mismatch")
    candidates = payload.get("frozen_candidates")
    _require(isinstance(candidates, list) and len(candidates) == 5, "candidate count differs")
    identity_fields = ("candidate_id", "candidate_fingerprint", "method", "rank", "q", "r", "K", "role")
    expected = [tuple(item[field] for field in identity_fields) for item in FROZEN_CANDIDATES]
    actual = [tuple(item[field] for field in identity_fields) for item in candidates]
    _require(actual == expected, "candidate identity or order differs")
    _require(sum(item["method"] in {"B2", "B3"} for item in candidates) == 3, "random count differs")
    for item in candidates:
        seeds = item["future_trajectory_seeds"]
        expected_count = TRAJECTORIES_PER_RANDOM_CELL if item["method"] in {"B2", "B3"} else 0
        _require(len(seeds) == expected_count, "trajectory seed count differs")
        _require(len(seeds) == len(set(seeds)), "trajectory seeds collide within candidate")
    all_seeds = [seed for item in candidates for seed in item["future_trajectory_seeds"]]
    _require(len(all_seeds) == len(set(all_seeds)), "trajectory seeds collide across candidates")
    caps = payload["resource_caps"]
    _require(caps["total_full_wrappers"] == 196, "wrapper cap differs")
    _require(caps["additional_trajectories"] == 0, "extension trajectories enabled")
    decision = payload["terminal_decision_rule"]
    _require(tuple(decision["allowed_statuses"]) == TRANSFER_STATUSES, "statuses differ")
    _require(decision["runner_authorizes_next_stage"] is False, "runner authorizes next stage")
    authorization = payload["authorization"]
    _require(authorization["held_out_access_authorized"] is False, "held-out access authorized")
    _require(authorization["transfer_execution_authorized"] is False, "transfer execution authorized")
    _require(authorization["automatic_next_stage"] is None, "automatic next stage exists")
    _require(all(value == 0 for value in payload["zero_science_counters"].values()), "science counter nonzero")
    target = payload["held_out_target"]
    for field in (
        "path_resolved_during_planning",
        "path_statted_during_planning",
        "file_hashed_during_planning",
        "npz_loaded_during_planning",
    ):
        _require(target[field] is False, f"held-out planning access recorded: {field}")


def write_json_artifact(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

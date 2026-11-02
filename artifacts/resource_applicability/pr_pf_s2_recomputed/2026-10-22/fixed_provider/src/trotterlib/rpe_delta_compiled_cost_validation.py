"""Compiled RTE-block cost projection for shortlisted RPE delta schedules.

This module deliberately stops one level below a complete Hadamard
interrogation.  It reweights a validated local connected-cluster calibration
for every ``(r_m, K_m)`` selected by the round-schedule validation and sums the
central RTE-block cost over repetitions and provisional shots.  Deterministic
DF sweeps, their boundaries, control overhead, the ancilla wrapper, and state
preparation are not included.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .rpe_delta_round_schedule_validation import (
    validate_rpe_delta_round_schedule_payload,
)
from .rte import finite_rte_distribution
from .rte_connected_cluster_cost_validation import (
    validate_connected_cluster_calibration_payload,
    validate_connected_cluster_k4_calibration_payload,
    validate_connected_cluster_transfer_payload,
)
from .rte_cost_angle_invariance_validation import (
    validate_rte_cost_angle_invariance_payload,
)


SCHEMA_VERSION = "rpe_delta_compiled_rte_block_cost_validation_v1"
METHOD = "angle_validated_connected_cluster_round_schedule_projection_v1"
METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)
ORDERS = (0, 2)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _parameter_key(pattern: Sequence[int]) -> str:
    return f"k{len(pattern)}:" + ",".join(str(value) for value in pattern)


def iid_local_window_form(
    event_count: int,
    order_probabilities: Mapping[int, float],
) -> dict[str, float]:
    """Expected 1--3-event window multiplicities for an IID event sequence."""
    length = int(event_count)
    if length < 1:
        raise ValueError("event_count must be positive.")
    probabilities = {order: float(order_probabilities[order]) for order in ORDERS}
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities.values()):
        raise ValueError("Order probabilities must be finite and non-negative.")
    if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-12):
        raise ValueError("Order probabilities must sum to one.")
    result: dict[str, float] = {}
    for window_length in range(1, min(3, length) + 1):
        multiplicity = length - window_length + 1
        for pattern in itertools.product(ORDERS, repeat=window_length):
            result[_parameter_key(pattern)] = float(
                multiplicity
                * math.prod(probabilities[order] for order in pattern)
            )
    return result


def _statistics_lookup(calibration: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(stratum["parameter_key"]): stratum
        for output in calibration["production"].values()
        for stratum in output["strata"].values()
    }


def _k4_statistics_lookup(
    k4_validation: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(stratum["parameter_key"]): stratum
        for stratum in k4_validation["k4_calibration"]["strata"].values()
    }


def _k4_iid_form(
    event_count: int,
    order_probabilities: Mapping[int, float],
) -> tuple[dict[str, float], float]:
    """Return supported K4 terms and omitted >=2-rare window mass."""
    length = int(event_count)
    if length < 4:
        return {}, 0.0
    multiplicity = length - 3
    form: dict[str, float] = {}
    retained_probability = 0.0
    for pattern in itertools.product(ORDERS, repeat=4):
        probability = math.prod(order_probabilities[value] for value in pattern)
        if sum(value == 2 for value in pattern) <= 1:
            form[_parameter_key(pattern)] = float(multiplicity * probability)
            retained_probability += probability
    return form, float(multiplicity * max(0.0, 1.0 - retained_probability))


def _evaluate_form(
    form: Mapping[str, float],
    lookup: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> tuple[float, float]:
    missing = set(form).difference(lookup)
    if missing:
        raise ValueError(f"Calibration is missing local parameters: {sorted(missing)}")
    mean = math.fsum(
        coefficient * float(lookup[key]["metric_statistics"][metric]["mean"])
        for key, coefficient in form.items()
    )
    variance = math.fsum(
        (
            coefficient
            * float(
                lookup[key]["metric_statistics"][metric]["standard_error"]
            )
        )
        ** 2
        for key, coefficient in form.items()
    )
    return float(mean), float(math.sqrt(variance))


def _probabilities(tau: float, cutoff: int) -> tuple[dict[int, float], dict[str, Any]]:
    if cutoff not in (0, 2):
        raise ValueError("This projection supports only selected K_m=0 or 2.")
    if cutoff == 0:
        return {0: 1.0, 2: 0.0}, {
            "orders": [0],
            "order_probabilities": [1.0],
            "finite_taylor_order": 0,
            "dimensionless_step_time": float(tau),
        }
    distribution = finite_rte_distribution(float(tau), 2)
    probabilities = {
        int(order): float(probability)
        for order, probability in zip(
            distribution.orders,
            distribution.order_probabilities,
            strict=True,
        )
    }
    return probabilities, distribution.to_dict()


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)


def validate_rpe_delta_compiled_cost(
    schedule: Mapping[str, Any],
    calibration: Mapping[str, Any],
    angle_invariance: Mapping[str, Any],
    *,
    transfer_validations: Sequence[Mapping[str, Any]] = (),
    k4_validation: Mapping[str, Any] | None = None,
    k4_angle_invariance: Mapping[str, Any] | None = None,
    k4_minimum_event_count: int = 32,
    model_relative_tolerance: float = 0.05,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the central RTE-block cost of each feasible delta schedule."""
    validate_rpe_delta_round_schedule_payload(schedule)
    validate_connected_cluster_calibration_payload(calibration)
    validate_rte_cost_angle_invariance_payload(angle_invariance)
    for transfer in transfer_validations:
        validate_connected_cluster_transfer_payload(transfer)
        if str(transfer["calibration_fingerprint"]) != str(
            calibration["calibration_fingerprint"]
        ):
            raise ValueError("Transfer validation uses a different calibration.")
    if (k4_validation is None) != (k4_angle_invariance is None):
        raise ValueError("K4 validation and K4 angle validation must be supplied together.")
    if k4_validation is not None:
        validate_connected_cluster_k4_calibration_payload(k4_validation)
        validate_rte_cost_angle_invariance_payload(k4_angle_invariance)  # type: ignore[arg-type]
        if not k4_validation["summary"]["primary_point_tolerance_passed"]:
            raise ValueError("The K4 point-error validation did not pass.")
        if not k4_angle_invariance["summary"]["sampled_metric_invariance_passed"]:  # type: ignore[index]
            raise ValueError("The K4 angle-invariance validation did not pass.")
    k4_threshold = int(k4_minimum_event_count)
    if k4_threshold < 4:
        raise ValueError("k4_minimum_event_count must be at least four.")

    tolerance = float(model_relative_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("model_relative_tolerance must be finite and non-negative.")
    if not angle_invariance["summary"]["sampled_metric_invariance_passed"]:
        raise ValueError("The angle-invariance prerequisite did not pass.")

    system = schedule["system"]
    condition = calibration["condition"]
    for key in ("hamiltonian_hash", "partition_hash", "preparation_hash"):
        if str(system[key]) != str(condition[key]):
            raise ValueError(f"Schedule/calibration {key} mismatch.")
    if int(system["ld"]) != int(condition["ld"]):
        raise ValueError("Schedule/calibration L_D mismatch.")
    angle_configuration = angle_invariance["configuration"]
    if int(angle_configuration["ld"]) != int(condition["ld"]):
        raise ValueError("Angle-invariance/calibration L_D mismatch.")
    if angle_configuration["compiler"] != condition["compiler"]:
        raise ValueError("Angle-invariance/calibration compiler mismatch.")
    if not {1, 2, 3}.issubset(
        {int(value) for value in angle_configuration["cluster_lengths"]}
    ):
        raise ValueError("Angle validation must cover cluster lengths 1--3.")
    angle_hamiltonian = angle_invariance["hamiltonian"]
    for angle_key, condition_key in (
        ("hash", "hamiltonian_hash"),
        ("partition_hash", "partition_hash"),
        ("preparation_hash", "preparation_hash"),
    ):
        if str(angle_hamiltonian[angle_key]) != str(condition[condition_key]):
            raise ValueError(
                f"Angle-invariance/calibration {condition_key} mismatch."
            )

    calibrated_short_step = float(condition["short_step_time"])
    sampled_times = tuple(float(value) for value in angle_configuration["short_step_times"])
    if not any(_same_float(calibrated_short_step, value) for value in sampled_times):
        raise ValueError("Angle validation omits the calibrated short-step time.")

    lookup = _statistics_lookup(calibration)
    k4_lookup = (
        None if k4_validation is None else _k4_statistics_lookup(k4_validation)
    )
    k4_sampled_times = (
        ()
        if k4_angle_invariance is None
        else tuple(
            float(value)
            for value in k4_angle_invariance["configuration"]["short_step_times"]
        )
    )
    if k4_validation is not None and k4_angle_invariance is not None:
        k4_configuration = k4_validation["configuration"]
        if int(k4_configuration["ld"]) != int(condition["ld"]):
            raise ValueError("K4/calibration L_D mismatch.")
        if k4_configuration["compiler"] != condition["compiler"]:
            raise ValueError("K4/calibration compiler mismatch.")
        if not _same_float(
            float(k4_configuration["short_step_time"]), calibrated_short_step
        ):
            raise ValueError("K4/calibration short-step mismatch.")
        if str(k4_validation["source_calibration_fingerprint"]) != str(
            calibration["calibration_fingerprint"]
        ):
            raise ValueError("K4 validation uses a different calibration.")
        for key in ("hamiltonian_hash", "partition_hash", "preparation_hash"):
            if str(k4_validation["k4_calibration"][key]) != str(condition[key]):
                raise ValueError(f"K4/calibration {key} mismatch.")
        k4_angle_configuration = k4_angle_invariance["configuration"]
        if int(k4_angle_configuration["ld"]) != int(condition["ld"]):
            raise ValueError("K4 angle-invariance/calibration L_D mismatch.")
        if k4_angle_configuration["compiler"] != condition["compiler"]:
            raise ValueError("K4 angle-invariance/calibration compiler mismatch.")
        if 4 not in {
            int(value) for value in k4_angle_configuration["cluster_lengths"]
        }:
            raise ValueError("K4 angle validation must cover cluster length four.")
        k4_angle_hamiltonian = k4_angle_invariance["hamiltonian"]
        for angle_key, condition_key in (
            ("hash", "hamiltonian_hash"),
            ("partition_hash", "partition_hash"),
            ("preparation_hash", "preparation_hash"),
        ):
            if str(k4_angle_hamiltonian[angle_key]) != str(condition[condition_key]):
                raise ValueError(
                    f"K4 angle-invariance/calibration {condition_key} mismatch."
                )
    projections: list[dict[str, Any]] = []
    all_selected_times: set[float] = set()
    for candidate in schedule["analytic_round_schedules"]:
        delta = float(candidate["delta_time"])
        aggregate_form: dict[str, float] = {}
        aggregate_k4_form: dict[str, float] = {}
        maximum_omitted_k4_window_mass = 0.0
        rounds = []
        for round_item in candidate["rounds"]:
            rte_steps = int(round_item["r_m"])
            q_m = int(round_item["q_m"])
            cutoff = int(round_item["K_m"])
            shots = int(round_item["total_axis_shots"])
            short_step = delta / rte_steps
            all_selected_times.add(short_step)
            if not any(_same_float(short_step, value) for value in sampled_times):
                raise ValueError(
                    f"Angle validation omits selected short-step {short_step}."
                )
            probabilities, distribution = _probabilities(
                float(round_item["tau_m"]), cutoff
            )
            form = iid_local_window_form(rte_steps, probabilities)
            use_k4 = k4_lookup is not None and rte_steps >= k4_threshold
            if use_k4 and not any(
                _same_float(short_step, value) for value in k4_sampled_times
            ):
                raise ValueError(
                    f"K4 angle validation omits selected short-step {short_step}."
                )
            k4_form, omitted_k4_window_mass = (
                _k4_iid_form(rte_steps, probabilities) if use_k4 else ({}, 0.0)
            )
            maximum_omitted_k4_window_mass = max(
                maximum_omitted_k4_window_mass, omitted_k4_window_mass
            )
            per_occurrence = {}
            per_interrogation = {}
            shot_weighted = {}
            for metric in METRICS:
                mean, standard_error = _evaluate_form(form, lookup, metric)
                if k4_form:
                    correction, correction_se = _evaluate_form(
                        k4_form, k4_lookup, metric  # type: ignore[arg-type]
                    )
                    mean += correction
                    standard_error = math.hypot(standard_error, correction_se)
                per_occurrence[metric] = {
                    "mean": mean,
                    "calibration_standard_error": standard_error,
                }
                per_interrogation[metric] = float(q_m * mean)
                shot_weighted[metric] = float(shots * q_m * mean)
            aggregate_weight = shots * q_m
            for key, coefficient in form.items():
                aggregate_form[key] = (
                    aggregate_form.get(key, 0.0) + aggregate_weight * coefficient
                )
            for key, coefficient in k4_form.items():
                aggregate_k4_form[key] = (
                    aggregate_k4_form.get(key, 0.0)
                    + aggregate_weight * coefficient
                )
            rounds.append(
                {
                    "round_index": int(round_item["round_index"]),
                    "q_m": q_m,
                    "r_m": rte_steps,
                    "K_m": cutoff,
                    "short_step_time": short_step,
                    "total_axis_shots": shots,
                    "order_distribution": distribution,
                    "connected_cluster_maximum_window": 4 if use_k4 else 3,
                    "omitted_k4_two_or_more_order2_expected_window_count": (
                        omitted_k4_window_mass
                    ),
                    "rte_occurrence_compiled_cost": per_occurrence,
                    "rte_blocks_per_interrogation_additive_proxy": per_interrogation,
                    "shot_weighted_rte_block_cost_proxy": shot_weighted,
                }
            )
        totals = {}
        for metric in METRICS:
            mean, standard_error = _evaluate_form(aggregate_form, lookup, metric)
            if aggregate_k4_form:
                correction, correction_se = _evaluate_form(
                    aggregate_k4_form, k4_lookup, metric  # type: ignore[arg-type]
                )
                mean += correction
                standard_error = math.hypot(standard_error, correction_se)
            totals[metric] = {
                "mean": mean,
                "calibration_standard_error": standard_error,
                "calibration_relative_95_half_width": (
                    None if mean == 0.0 else 1.96 * standard_error / abs(mean)
                ),
            }
        projections.append(
            {
                "delta_time": delta,
                "round_count": len(rounds),
                "rounds": rounds,
                "aggregate_prediction_form": dict(sorted(aggregate_form.items())),
                "aggregate_k4_prediction_form": dict(
                    sorted(aggregate_k4_form.items())
                ),
                "maximum_omitted_k4_two_or_more_order2_expected_window_count": (
                    maximum_omitted_k4_window_mass
                ),
                "total_shot_weighted_rte_block_cost_proxy": totals,
            }
        )

    rankings: dict[str, Any] = {}
    for metric in METRICS:
        ordered = sorted(
            (
                (
                    float(item["delta_time"]),
                    float(item["total_shot_weighted_rte_block_cost_proxy"][metric]["mean"]),
                )
                for item in projections
            ),
            key=lambda item: item[1],
        )
        best = ordered[0][1]
        rankings[metric] = {
            "best_delta": ordered[0][0],
            "rows": [
                {
                    "delta_time": delta,
                    "cost": cost,
                    "relative_overhead_to_best": cost / best - 1.0,
                }
                for delta, cost in ordered
            ],
        }

    primary = rankings["rz_count"]
    near_best = [
        row["delta_time"]
        for row in primary["rows"]
        if row["relative_overhead_to_best"] <= tolerance + 1e-15
    ]
    transfer_summaries = [
        {
            "validation_fingerprint": item["validation_fingerprint"],
            "holdout_lengths": list(item["configuration"]["holdout_lengths"]),
            "holdout_zero_sample_count_per_length": int(
                item["configuration"]["holdout_zero_sample_count_per_length"]
            ),
            "holdout_single_rare_sample_count_per_position": int(
                item["configuration"][
                    "holdout_single_rare_sample_count_per_position"
                ]
            ),
            "summary": dict(item["summary"]),
        }
        for item in transfer_validations
    ]
    k4_evidence = None
    if k4_validation is not None and k4_angle_invariance is not None:
        k4_evidence = {
            "validation_fingerprint": k4_validation["validation_fingerprint"],
            "summary": dict(k4_validation["summary"]),
            "angle_invariance_fingerprint": k4_angle_invariance[
                "validation_fingerprint"
            ],
            "angle_invariance_summary": dict(k4_angle_invariance["summary"]),
            "angle_invariance_coverage": dict(k4_angle_invariance["coverage"]),
        }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "scope": {
            "central_rte_blocks_only": True,
            "deterministic_df_sweeps_included": False,
            "deterministic_rte_boundary_cost_included": False,
            "controlled_evolution_overhead_included": False,
            "ancilla_hadamard_axis_measurement_wrapper_included": False,
            "state_preparation_included": False,
            "q_greater_than_8_full_circuit_compilation_performed": False,
            "final_one_shot_compiled_cost_evaluated": False,
            "final_total_cost_evaluation_performed": False,
        },
        "system": dict(system),
        "configuration": {
            "primary_metric": "rz_count",
            "model_relative_tolerance": tolerance,
            "calibrated_short_step_time": calibrated_short_step,
            "sampled_angle_short_step_times": list(sampled_times),
            "selected_schedule_short_step_times": sorted(all_selected_times),
            "connected_cluster_base_maximum_window": 3,
            "conditional_k4_minimum_event_count": (
                None if k4_lookup is None else k4_threshold
            ),
            "repetition_rule": (
                "multiply_one_rte_occurrence_prediction_by_q_m; cross_partial_s2_"
                "boundary_effects_are_not_included"
            ),
        },
        "evidence": {
            "schedule_content_fingerprint": schedule["content_fingerprint"],
            "calibration_fingerprint": calibration["calibration_fingerprint"],
            "angle_invariance_fingerprint": angle_invariance[
                "validation_fingerprint"
            ],
            "angle_invariance_summary": dict(angle_invariance["summary"]),
            "angle_invariance_coverage": dict(angle_invariance["coverage"]),
            "transfer_validations": transfer_summaries,
            "conditional_k4_validation": k4_evidence,
        },
        "candidate_projections": projections,
        "rankings": rankings,
        "summary": {
            "angle_invariance_passed": True,
            "all_metrics_best_delta": {
                metric: rankings[metric]["best_delta"] for metric in METRICS
            },
            "primary_best_delta": primary["best_delta"],
            "primary_near_best_deltas_within_model_tolerance": near_best,
            "primary_rows": primary["rows"],
            "shortlist_separated_at_rte_block_proxy_level": len(near_best) == 1,
            "interpretation": (
                "the_central_rte_block_proxy_favors_one_delta_but_full_controlled_"
                "hadamard_one_shot_cost_remains_unvalidated"
            ),
            "next_action": (
                "validate_a_controlled_partial_s2_repetition_proxy_for_the_selected_"
                "round_specific_r_and_K_values_before_final_delta_selection"
            ),
        },
        "provenance": dict(provenance or {}),
    }
    payload["content_fingerprint"] = _fingerprint(payload)
    validate_rpe_delta_compiled_cost_payload(payload)
    return payload


def validate_rpe_delta_compiled_cost_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("method") != METHOD:
        raise ValueError("Unsupported RPE delta compiled-cost artifact.")
    unsigned = dict(payload)
    fingerprint = unsigned.pop("content_fingerprint", None)
    if fingerprint != _fingerprint(unsigned):
        raise ValueError("RPE delta compiled-cost content_fingerprint mismatch.")
    scope = payload.get("scope", {})
    forbidden = (
        "deterministic_df_sweeps_included",
        "deterministic_rte_boundary_cost_included",
        "controlled_evolution_overhead_included",
        "ancilla_hadamard_axis_measurement_wrapper_included",
        "state_preparation_included",
        "q_greater_than_8_full_circuit_compilation_performed",
        "final_one_shot_compiled_cost_evaluated",
        "final_total_cost_evaluation_performed",
    )
    if any(scope.get(key) is not False for key in forbidden):
        raise ValueError("RPE delta compiled-cost scope overclaims its coverage.")
    if scope.get("central_rte_blocks_only") is not True:
        raise ValueError("RPE delta compiled-cost scope must be central-RTE-only.")
    summary = payload.get("summary", {})
    if summary.get("angle_invariance_passed") is not True:
        raise ValueError("Angle invariance must pass before cost projection.")
    projections = payload.get("candidate_projections", ())
    if not projections:
        raise ValueError("At least one delta projection is required.")
    deltas = {float(item["delta_time"]) for item in projections}
    for metric in METRICS:
        ranking = payload.get("rankings", {}).get(metric, {})
        rows = ranking.get("rows", ())
        if {float(item["delta_time"]) for item in rows} != deltas:
            raise ValueError(f"Incomplete ranking for {metric}.")
        if not rows or float(rows[0]["delta_time"]) != float(ranking["best_delta"]):
            raise ValueError(f"Invalid best-delta ranking for {metric}.")


def write_rpe_delta_compiled_cost_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_rpe_delta_compiled_cost_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

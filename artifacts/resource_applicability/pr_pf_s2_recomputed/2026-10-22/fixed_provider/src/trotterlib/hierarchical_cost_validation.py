"""Held-out validation for the hierarchical compiled-cost model.

The module deliberately keeps two validation layers separate:

* a scalar one-/two-/three-event cluster model for an RTE occurrence; and
* an affine repeated-step model for controlled partial-S2 circuits.

All calibration and holdout estimates use disjoint master seeds.  The payloads
are local validation evidence, not final resource estimates.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
from .df_partial_s2_repeated_cost import (
    CompiledRepeatedPartialS2CostEstimate,
    estimate_monte_carlo_compiled_repeated_partial_s2_cost,
)
from .rte import (
    CircuitCost,
    CompilerSettings,
    make_rte_config,
    require_integer_count,
    step_taylor_truncation_residual_bound,
)
from .rte_compiled_cost import (
    CompiledEventCostEstimate,
    CompiledSequenceCostEstimate,
    TranspiledCircuitCostCache,
    estimate_compiled_occurrence_cost,
    estimate_exact_compiled_event_cost,
    estimate_monte_carlo_compiled_event_cost,
)


HIERARCHICAL_COST_VALIDATION_SCHEMA_VERSION = (
    "hierarchical_compiled_cost_validation_v1"
)
RTE_CLUSTER_EXTENSION_METHOD = "independent_c1_c2_c3_cluster_holdout_v1"
CONTROLLED_REPETITION_METHOD = "controlled_q1_q2_affine_q4_holdout_v1"
CONTROLLED_REPETITION_GENERAL_METHOD = "controlled_q1_q2_affine_holdout_v2"
_METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)


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


def _cost_values(cost: CircuitCost) -> dict[str, float]:
    return {metric: float(getattr(cost, metric)) for metric in _METRICS}


def _zero_cost_values() -> dict[str, float]:
    return {metric: 0.0 for metric in _METRICS}


def _compiler_payload(compiler: CompilerSettings) -> dict[str, Any]:
    return {
        "basis_gates": list(compiler.basis_gates),
        "backend_name": compiler.backend_name,
        "coupling_map": (
            None
            if compiler.coupling_map is None
            else [list(edge) for edge in compiler.coupling_map]
        ),
        "optimization_level": compiler.optimization_level,
        "layout_method": compiler.layout_method,
        "routing_method": compiler.routing_method,
        "transpiler_seed": compiler.transpiler_seed,
        "qiskit_version": compiler.qiskit_version,
    }


def _same_short_step_distribution(left: Any, right: Any) -> bool:
    """Compare a fixed short-step distribution up to floating-point noise."""
    return (
        left.finite_taylor_order == right.finite_taylor_order
        and left.orders == right.orders
        and math.isclose(
            left.dimensionless_step_time,
            right.dimensionless_step_time,
            rel_tol=1e-14,
            abs_tol=1e-15,
        )
        and all(
            math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-15)
            for a, b in zip(
                left.order_probabilities,
                right.order_probabilities,
                strict=True,
            )
        )
    )


def _explicit_cutoff_tolerance(tau: float, cutoff: int) -> float:
    residual = step_taylor_truncation_residual_bound(tau, cutoff)
    if not math.isfinite(residual):
        raise ValueError("The requested finite-Taylor residual overflowed.")
    if residual == 0.0:
        return math.ulp(0.0)
    # A small relative margin avoids a second arithmetic path rounding back
    # above nextafter(residual, +inf), while retaining the requested cutoff.
    return max(math.nextafter(residual, math.inf), residual * (1.0 + 1e-12))


def _prediction_result(
    *,
    prediction: float,
    prediction_standard_error: float,
    actual: float,
    actual_standard_error: float,
) -> dict[str, float | None]:
    difference = prediction - actual
    combined_standard_error = math.hypot(
        prediction_standard_error,
        actual_standard_error,
    )
    return {
        "prediction": float(prediction),
        "prediction_standard_error": float(prediction_standard_error),
        "actual": float(actual),
        "actual_standard_error": float(actual_standard_error),
        "prediction_minus_actual": float(difference),
        "absolute_relative_error": (
            None if actual == 0.0 else float(abs(difference / actual))
        ),
        "combined_standard_error": float(combined_standard_error),
        "absolute_z_score": (
            None
            if combined_standard_error == 0.0
            else float(abs(difference) / combined_standard_error)
        ),
        "relative_95_percent_upper": (
            None
            if actual == 0.0
            else float(
                (abs(difference) + 1.96 * combined_standard_error) / abs(actual)
            )
        ),
    }


def _event_estimate_payload(
    estimate: CompiledEventCostEstimate,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "estimate_kind": estimate.estimate_kind,
        "expected_cost": _cost_values(estimate.expected_cost),
        "standard_error": (
            None
            if estimate.standard_error is None
            else _cost_values(estimate.standard_error)
        ),
        "sample_count": estimate.sample_count,
        "enumerated_event_count": estimate.enumerated_event_count,
        "seed": estimate.seed,
        "event_probability_sum": estimate.event_probability_sum,
        "unique_compiled_circuit_count": estimate.unique_compiled_circuit_count,
        "elapsed_seconds": float(elapsed_seconds),
    }


def _sequence_estimate_payload(
    estimate: CompiledSequenceCostEstimate,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "sample_count": estimate.sample_count,
        "sequence_length": estimate.event_count_per_sample,
        "seed": estimate.seed,
        "full_expected_cost": _cost_values(estimate.sequence_expected_cost),
        "full_standard_error": (
            None
            if estimate.sequence_standard_error is None
            else _cost_values(estimate.sequence_standard_error)
        ),
        "full_minus_isolated_sum": _cost_values(
            estimate.nonadditive_difference
        ),
        "difference_standard_error": (
            None
            if estimate.difference_standard_error is None
            else _cost_values(estimate.difference_standard_error)
        ),
        "event_stream_rolling_digest": estimate.event_stream_rolling_digest,
        "unique_sequence_circuit_count": estimate.unique_sequence_circuit_count,
        "unique_compiled_circuit_count": estimate.unique_compiled_circuit_count,
        "elapsed_seconds": float(elapsed_seconds),
    }


def _required_standard_error(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, float]:
    standard_error = payload.get("standard_error")
    if not isinstance(standard_error, Mapping):
        raise RuntimeError(f"{label} does not contain a standard error.")
    return standard_error


def _required_full_standard_error(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, float]:
    standard_error = payload.get("full_standard_error")
    if not isinstance(standard_error, Mapping):
        raise RuntimeError(f"{label} does not contain a standard error.")
    return standard_error


def _maximum(
    holdouts: Sequence[Mapping[str, Any]],
    model: str,
    field: str,
) -> float | None:
    values = [
        metric_models[model][field]
        for holdout in holdouts
        for metric_models in holdout["metrics"].values()
        if metric_models[model][field] is not None
    ]
    return None if not values else float(max(values))


def validate_rte_cluster_extension(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    finite_taylor_order: int,
    compiler: CompilerSettings,
    calibration_sample_count: int,
    holdout_sequence_lengths: Sequence[int],
    holdout_sample_count: int,
    seed: int,
    maximum_exact_events: int = 1_000,
    maximum_samples: int = 10_000,
    cache_maximum_entries: int = 8_192,
    maximum_untranspiled_circuit_size: int = 100_000,
    maximum_planned_instruction_applications: int = 1_000_000_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate independent C1/C2/C3 estimates and hold out longer sequences."""
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    reference_rte_steps = require_integer_count(
        reference_rte_steps,
        name="reference_rte_steps",
        minimum=1,
    )
    finite_taylor_order = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    calibration_sample_count = require_integer_count(
        calibration_sample_count,
        name="calibration_sample_count",
        minimum=2,
    )
    holdout_sample_count = require_integer_count(
        holdout_sample_count,
        name="holdout_sample_count",
        minimum=2,
    )
    maximum_samples = require_integer_count(
        maximum_samples,
        name="maximum_samples",
        minimum=max(calibration_sample_count, holdout_sample_count),
    )
    holdout_lengths = tuple(
        sorted(
            require_integer_count(value, name="holdout_length", minimum=4)
            for value in holdout_sequence_lengths
        )
    )
    if not holdout_lengths or len(set(holdout_lengths)) != len(holdout_lengths):
        raise ValueError("holdout_sequence_lengths must be nonempty and unique.")
    if not math.isfinite(reference_delta_time) or reference_delta_time <= 0.0:
        raise ValueError("reference_delta_time must be finite and positive.")

    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
    )
    if preparation.is_deterministic_only:
        raise ValueError("RTE cluster validation requires a non-empty tail.")
    short_step_time = float(reference_delta_time) / reference_rte_steps
    tau = preparation.exact_rte_lambda_r * short_step_time

    def make_config(sequence_length: int):
        return make_rte_config(
            preparation.rte_preparation.symbolic_tail,
            evolution_time=short_step_time * sequence_length,
            rte_steps=sequence_length,
            truncation_tolerance=_explicit_cutoff_tolerance(
                tau,
                finite_taylor_order,
            ),
            finite_taylor_order=finite_taylor_order,
            seed=seed,
        )

    _config, distribution = make_config(1)
    supported_orders = tuple(
        order
        for order, probability in zip(
            distribution.orders,
            distribution.order_probabilities,
            strict=True,
        )
        if probability > 0.0
    )
    component_count = len(preparation.rte_preparation.symbolic_tail.components)
    event_space_size = sum(
        component_count ** (order + 1) for order in supported_orders
    )
    cache = TranspiledCircuitCostCache(maximum_entries=cache_maximum_entries)

    point_started = time.perf_counter()
    if event_space_size <= maximum_exact_events:
        c1_estimate = estimate_exact_compiled_event_cost(
            preparation.rte_preparation,
            distribution,
            compiler,
            max_events=maximum_exact_events,
            maximum_planned_instruction_applications=(
                maximum_planned_instruction_applications
            ),
            cache=cache,
        )
    else:
        c1_estimate = estimate_monte_carlo_compiled_event_cost(
            preparation.rte_preparation,
            distribution,
            compiler,
            sample_count=calibration_sample_count,
            seed=seed + 1,
            maximum_samples=maximum_samples,
            maximum_planned_instruction_applications=(
                maximum_planned_instruction_applications
            ),
            cache=cache,
        )
    c1_payload = _event_estimate_payload(
        c1_estimate,
        elapsed_seconds=time.perf_counter() - point_started,
    )
    c1 = c1_payload["expected_cost"]
    c1_se = (
        _zero_cost_values()
        if c1_payload["standard_error"] is None
        else _required_standard_error(c1_payload, label="C1")
    )

    calibrations: dict[int, dict[str, Any]] = {}
    for length in (2, 3):
        config, current_distribution = make_config(length)
        if not _same_short_step_distribution(current_distribution, distribution):
            raise RuntimeError("Fixed short-step event distributions differ.")
        point_started = time.perf_counter()
        estimate = estimate_compiled_occurrence_cost(
            preparation.rte_preparation,
            config,
            distribution,
            compiler,
            sequence_sample_count=calibration_sample_count,
            seed=seed + length,
            maximum_samples=maximum_samples,
            maximum_rte_steps=max(16, *holdout_lengths),
            maximum_untranspiled_circuit_size=(
                maximum_untranspiled_circuit_size
            ),
            maximum_planned_instruction_applications=(
                maximum_planned_instruction_applications
            ),
            cache=cache,
        )
        calibrations[length] = _sequence_estimate_payload(
            estimate,
            elapsed_seconds=time.perf_counter() - point_started,
        )

    holdout_payloads: dict[int, dict[str, Any]] = {}
    for length in holdout_lengths:
        config, current_distribution = make_config(length)
        if not _same_short_step_distribution(current_distribution, distribution):
            raise RuntimeError("Fixed short-step event distributions differ.")
        point_started = time.perf_counter()
        estimate = estimate_compiled_occurrence_cost(
            preparation.rte_preparation,
            config,
            distribution,
            compiler,
            sequence_sample_count=holdout_sample_count,
            seed=seed + 100 + length,
            maximum_samples=maximum_samples,
            maximum_rte_steps=max(16, *holdout_lengths),
            maximum_untranspiled_circuit_size=(
                maximum_untranspiled_circuit_size
            ),
            maximum_planned_instruction_applications=(
                maximum_planned_instruction_applications
            ),
            cache=cache,
        )
        holdout_payloads[length] = _sequence_estimate_payload(
            estimate,
            elapsed_seconds=time.perf_counter() - point_started,
        )

    c2 = calibrations[2]["full_expected_cost"]
    c2_se = _required_full_standard_error(calibrations[2], label="C2")
    c3 = calibrations[3]["full_expected_cost"]
    c3_se = _required_full_standard_error(calibrations[3], label="C3")
    coefficients: dict[str, Any] = {}
    for metric in _METRICS:
        mu1 = c1[metric]
        mu2 = c2[metric] - 2.0 * mu1
        mu3 = c3[metric] + mu1 - 2.0 * c2[metric]
        coefficients[metric] = {
            "mu1": float(mu1),
            "mu1_standard_error": float(c1_se[metric]),
            "mu2": float(mu2),
            "mu2_standard_error": float(
                math.hypot(c2_se[metric], 2.0 * c1_se[metric])
            ),
            "mu3": float(mu3),
            "mu3_standard_error": float(
                math.sqrt(
                    c3_se[metric] ** 2
                    + c1_se[metric] ** 2
                    + 4.0 * c2_se[metric] ** 2
                )
            ),
        }

    holdout_results: list[dict[str, Any]] = []
    for length in holdout_lengths:
        actuals = holdout_payloads[length]["full_expected_cost"]
        actual_ses = _required_full_standard_error(
            holdout_payloads[length],
            label=f"C{length}",
        )
        metric_results: dict[str, Any] = {}
        for metric in _METRICS:
            naive = length * c1[metric]
            naive_se = length * c1_se[metric]
            pair = (2 - length) * c1[metric] + (length - 1) * c2[metric]
            pair_se = math.hypot(
                (length - 2) * c1_se[metric],
                (length - 1) * c2_se[metric],
            )
            triple = (3 - length) * c2[metric] + (length - 2) * c3[metric]
            triple_se = math.hypot(
                (length - 3) * c2_se[metric],
                (length - 2) * c3_se[metric],
            )
            metric_results[metric] = {
                "naive": _prediction_result(
                    prediction=naive,
                    prediction_standard_error=naive_se,
                    actual=actuals[metric],
                    actual_standard_error=actual_ses[metric],
                ),
                "pair": _prediction_result(
                    prediction=pair,
                    prediction_standard_error=pair_se,
                    actual=actuals[metric],
                    actual_standard_error=actual_ses[metric],
                ),
                "triple": _prediction_result(
                    prediction=triple,
                    prediction_standard_error=triple_se,
                    actual=actuals[metric],
                    actual_standard_error=actual_ses[metric],
                ),
            }
        holdout_results.append(
            {
                "sequence_length": length,
                "sample_count": holdout_payloads[length]["sample_count"],
                "seed": holdout_payloads[length]["seed"],
                "metrics": metric_results,
            }
        )

    primary = [
        holdout["metrics"]["rz_count"] for holdout in holdout_results
    ]
    payload: dict[str, Any] = {
        "schema_version": HIERARCHICAL_COST_VALIDATION_SCHEMA_VERSION,
        "validation_method": RTE_CLUSTER_EXTENSION_METHOD,
        "final_cost_evaluation_performed": False,
        "acceptance_policy": {
            "primary_metric": "rz_count",
            "relative_tolerance": 0.05,
            "require_relative_95_percent_upper": True,
        },
        "hamiltonian": {
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": hamiltonian.n_blocks,
            "metadata": dict(hamiltonian.metadata),
            "preparation_hash": preparation.preparation_hash,
            "partition_hash": preparation.partition_hash,
        },
        "configuration": {
            "ld": ld,
            "reference_delta_time": float(reference_delta_time),
            "reference_rte_steps": reference_rte_steps,
            "short_step_time": short_step_time,
            "dimensionless_short_step_time": tau,
            "finite_taylor_order": finite_taylor_order,
            "component_count": component_count,
            "event_space_size": event_space_size,
            "calibration_sample_count": calibration_sample_count,
            "holdout_sample_count": holdout_sample_count,
            "holdout_sequence_lengths": list(holdout_lengths),
            "seed": seed,
            "compiler": _compiler_payload(compiler),
        },
        "c1": c1_payload,
        "calibration": {str(key): value for key, value in calibrations.items()},
        "holdout": {str(key): value for key, value in holdout_payloads.items()},
        "cluster_coefficients": coefficients,
        "holdout_predictions": holdout_results,
        "summary": {
            "naive_maximum_absolute_relative_error": _maximum(
                holdout_results, "naive", "absolute_relative_error"
            ),
            "pair_maximum_absolute_relative_error": _maximum(
                holdout_results, "pair", "absolute_relative_error"
            ),
            "triple_maximum_absolute_relative_error": _maximum(
                holdout_results, "triple", "absolute_relative_error"
            ),
            "triple_maximum_absolute_z_score": _maximum(
                holdout_results, "triple", "absolute_z_score"
            ),
            "primary_metric_maximum_absolute_relative_error": max(
                result["triple"]["absolute_relative_error"] for result in primary
            ),
            "primary_metric_maximum_relative_95_percent_upper": max(
                result["triple"]["relative_95_percent_upper"] for result in primary
            ),
            "primary_metric_point_tolerance_passed": all(
                result["triple"]["absolute_relative_error"] <= 0.05
                for result in primary
            ),
            "primary_metric_95_percent_tolerance_passed": all(
                result["triple"]["relative_95_percent_upper"] <= 0.05
                for result in primary
            ),
        },
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "shared_transpile_cache": {
                "maximum_entries": cache.maximum_entries,
                "final_hits": cache.hit_count,
                "final_misses": cache.miss_count,
                "final_bypasses": cache.bypass_count,
                "final_evictions": cache.eviction_count,
            },
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def _repeated_estimate_payload(
    estimate: CompiledRepeatedPartialS2CostEstimate,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    def optional(cost: CircuitCost | None) -> dict[str, float] | None:
        return None if cost is None else _cost_values(cost)

    return {
        "repetition_count": estimate.repetition_count,
        "sample_count": estimate.sample_count,
        "seed": estimate.master_seed,
        "expected_cost": _cost_values(estimate.expected_cost),
        "standard_error": optional(estimate.standard_error),
        "matched_per_step_expected_cost": optional(
            estimate.matched_per_step_expected_cost
        ),
        "matched_per_step_standard_error": optional(
            estimate.matched_per_step_standard_error
        ),
        "cross_step_nonadditive_difference": optional(
            estimate.cross_step_nonadditive_difference
        ),
        "cross_step_difference_standard_error": optional(
            estimate.cross_step_difference_standard_error
        ),
        "primitive_additive_expected_cost": optional(
            estimate.primitive_additive_expected_cost
        ),
        "boundary_optimization_difference": optional(
            estimate.boundary_optimization_difference
        ),
        "trajectory_seed_digest": estimate.sampled_trajectory_seed_digest,
        "circuit_semantics_rolling_digest": (
            estimate.circuit_semantics_rolling_digest
        ),
        "unique_trajectory_circuit_count": (
            estimate.unique_trajectory_circuit_count
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }


def validate_controlled_repetition_extension(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    finite_taylor_order: int,
    compiler: CompilerSettings,
    calibration_sample_count: int,
    holdout_sample_count: int,
    holdout_repetition_count: int = 4,
    seed: int,
    maximum_samples: int = 10_000,
    cache_maximum_entries: int = 8_192,
    maximum_untranspiled_circuit_size: int = 500_000,
    maximum_planned_instruction_applications: int = 2_000_000_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit q=1,2 controlled costs and hold out an unused q >= 3."""
    started = time.perf_counter()
    calibration_sample_count = require_integer_count(
        calibration_sample_count,
        name="calibration_sample_count",
        minimum=2,
    )
    holdout_sample_count = require_integer_count(
        holdout_sample_count,
        name="holdout_sample_count",
        minimum=2,
    )
    holdout_repetition_count = require_integer_count(
        holdout_repetition_count,
        name="holdout_repetition_count",
        minimum=3,
    )
    maximum_samples = require_integer_count(
        maximum_samples,
        name="maximum_samples",
        minimum=max(calibration_sample_count, holdout_sample_count),
    )
    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
    )
    if preparation.is_deterministic_only:
        raise ValueError("Controlled repetition validation requires an RTE tail.")
    step_time = float(reference_delta_time)
    short_step_time = step_time / reference_rte_steps
    tau = preparation.exact_rte_lambda_r * short_step_time
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=step_time,
        rte_steps=reference_rte_steps,
        truncation_tolerance=_explicit_cutoff_tolerance(
            tau,
            finite_taylor_order,
        ),
        finite_taylor_order=finite_taylor_order,
        seed=seed,
    )
    cache = TranspiledCircuitCostCache(maximum_entries=cache_maximum_entries)
    estimates: dict[int, dict[str, Any]] = {}
    for repetition_count, sample_count, current_seed in (
        (1, calibration_sample_count, seed + 1),
        (2, calibration_sample_count, seed + 2),
        (
            holdout_repetition_count,
            holdout_sample_count,
            seed + 100 + holdout_repetition_count,
        ),
    ):
        point_started = time.perf_counter()
        estimate = estimate_monte_carlo_compiled_repeated_partial_s2_cost(
            preparation,
            step_time,
            repetition_count,
            config,
            distribution,
            compiler,
            sample_count=sample_count,
            seed=current_seed,
            maximum_samples=maximum_samples,
            controlled=True,
            ancilla_qubit=hamiltonian.n_qubits,
            construction_policy="boundary_optimized",
            evaluation_mode="selected_only",
            maximum_untranspiled_circuit_size=(
                maximum_untranspiled_circuit_size
            ),
            maximum_planned_instruction_applications=(
                maximum_planned_instruction_applications
            ),
            cache=cache,
        )
        estimates[repetition_count] = _repeated_estimate_payload(
            estimate,
            elapsed_seconds=time.perf_counter() - point_started,
        )

    c1 = estimates[1]["expected_cost"]
    c1_se = _required_standard_error(estimates[1], label="controlled q=1")
    c2 = estimates[2]["expected_cost"]
    c2_se = _required_standard_error(estimates[2], label="controlled q=2")
    holdout_cost = estimates[holdout_repetition_count]["expected_cost"]
    holdout_se = _required_standard_error(
        estimates[holdout_repetition_count],
        label=f"controlled q={holdout_repetition_count}",
    )
    prediction_results: dict[str, Any] = {}
    for metric in _METRICS:
        c1_coefficient = 2.0 - holdout_repetition_count
        c2_coefficient = holdout_repetition_count - 1.0
        prediction = (
            c1_coefficient * c1[metric] + c2_coefficient * c2[metric]
        )
        prediction_se = math.hypot(
            c1_coefficient * c1_se[metric],
            c2_coefficient * c2_se[metric],
        )
        prediction_results[metric] = _prediction_result(
            prediction=prediction,
            prediction_standard_error=prediction_se,
            actual=holdout_cost[metric],
            actual_standard_error=holdout_se[metric],
        )

    payload: dict[str, Any] = {
        "schema_version": HIERARCHICAL_COST_VALIDATION_SCHEMA_VERSION,
        "validation_method": (
            CONTROLLED_REPETITION_METHOD
            if holdout_repetition_count == 4
            else CONTROLLED_REPETITION_GENERAL_METHOD
        ),
        "final_cost_evaluation_performed": False,
        "acceptance_policy": {
            "primary_metric": "rz_count",
            "relative_tolerance": 0.05,
            "require_relative_95_percent_upper": True,
        },
        "hamiltonian": {
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": hamiltonian.n_blocks,
            "metadata": dict(hamiltonian.metadata),
            "preparation_hash": preparation.preparation_hash,
            "partition_hash": preparation.partition_hash,
        },
        "configuration": {
            "ld": ld,
            "reference_delta_time": step_time,
            "reference_rte_steps": reference_rte_steps,
            "short_step_time": short_step_time,
            "dimensionless_short_step_time": tau,
            "finite_taylor_order": finite_taylor_order,
            "calibration_repetition_counts": [1, 2],
            "holdout_repetition_counts": [holdout_repetition_count],
            "calibration_sample_count": calibration_sample_count,
            "holdout_sample_count": holdout_sample_count,
            "seed": seed,
            "controlled": True,
            "ancilla_qubit": hamiltonian.n_qubits,
            "evaluation_mode": "selected_only",
            "compiler": _compiler_payload(compiler),
        },
        "estimates": {str(key): value for key, value in estimates.items()},
        f"q{holdout_repetition_count}_affine_holdout_prediction": {
            "metrics": prediction_results
        },
        "summary": {
            "maximum_absolute_relative_error": max(
                result["absolute_relative_error"]
                for result in prediction_results.values()
                if result["absolute_relative_error"] is not None
            ),
            "maximum_absolute_z_score": max(
                (
                    result["absolute_z_score"]
                    for result in prediction_results.values()
                    if result["absolute_z_score"] is not None
                ),
                default=None,
            ),
            "primary_metric_absolute_relative_error": (
                prediction_results["rz_count"]["absolute_relative_error"]
            ),
            "primary_metric_relative_95_percent_upper": (
                prediction_results["rz_count"]["relative_95_percent_upper"]
            ),
            "primary_metric_point_tolerance_passed": (
                prediction_results["rz_count"]["absolute_relative_error"] <= 0.05
            ),
            "primary_metric_95_percent_tolerance_passed": (
                prediction_results["rz_count"]["relative_95_percent_upper"]
                <= 0.05
            ),
        },
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "shared_transpile_cache": {
                "maximum_entries": cache.maximum_entries,
                "final_hits": cache.hit_count,
                "final_misses": cache.miss_count,
                "final_bypasses": cache.bypass_count,
                "final_evictions": cache.eviction_count,
            },
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_hierarchical_cost_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != HIERARCHICAL_COST_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Unsupported hierarchical cost validation schema.")
    if payload.get("validation_method") not in (
        RTE_CLUSTER_EXTENSION_METHOD,
        CONTROLLED_REPETITION_METHOD,
        CONTROLLED_REPETITION_GENERAL_METHOD,
    ):
        raise ValueError("Unsupported hierarchical cost validation method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("validation_fingerprint must be a SHA-256 hex string.")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Hierarchical cost validation fingerprint mismatch.")


def write_hierarchical_cost_validation(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_hierarchical_cost_payload(payload)
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

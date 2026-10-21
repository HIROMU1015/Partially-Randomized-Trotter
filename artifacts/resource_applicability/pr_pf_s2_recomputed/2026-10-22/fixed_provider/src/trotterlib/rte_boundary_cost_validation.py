"""Expectation-level cluster validation for compiled RTE boundary costs."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
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
)


RTE_BOUNDARY_COST_VALIDATION_SCHEMA_VERSION = "rte_boundary_cost_validation_v1"
RTE_BOUNDARY_COST_VALIDATION_METHOD = (
    "iid_fixed_short_step_compiled_cost_cluster_expansion_v1"
)
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


def _explicit_cutoff_tolerance(tau: float, cutoff: int) -> float:
    residual = step_taylor_truncation_residual_bound(tau, cutoff)
    if not math.isfinite(residual):
        raise ValueError("The requested finite-Taylor residual overflowed.")
    if residual == 0.0:
        return math.ulp(0.0)
    return math.nextafter(residual, math.inf)


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


def _cost_values(cost: CircuitCost) -> dict[str, float]:
    return {metric: float(getattr(cost, metric)) for metric in _METRICS}


def _same_short_step_distribution(left: Any, right: Any) -> bool:
    """Compare the physical finite-event distribution up to roundoff."""
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
        "event_probability_sum": estimate.event_probability_sum,
        "unique_compiled_circuit_count": estimate.unique_compiled_circuit_count,
        "cache": {
            "hits": estimate.transpile_cache_hit_count,
            "misses": estimate.transpile_cache_miss_count,
            "bypasses": estimate.transpile_cache_bypass_count,
            "evictions": estimate.transpile_cache_eviction_count,
        },
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
        "full_expected_cost": _cost_values(estimate.sequence_expected_cost),
        "full_standard_error": (
            None
            if estimate.sequence_standard_error is None
            else _cost_values(estimate.sequence_standard_error)
        ),
        "isolated_event_sum_expected_cost": _cost_values(
            estimate.additive_expected_cost
        ),
        "isolated_event_sum_standard_error": (
            None
            if estimate.additive_standard_error is None
            else _cost_values(estimate.additive_standard_error)
        ),
        "full_minus_isolated_sum": _cost_values(
            estimate.nonadditive_difference
        ),
        "difference_standard_error": (
            None
            if estimate.difference_standard_error is None
            else _cost_values(estimate.difference_standard_error)
        ),
        "unique_sequence_circuit_count": (
            estimate.unique_sequence_circuit_count
        ),
        "unique_compiled_circuit_count": estimate.unique_compiled_circuit_count,
        "cache": {
            "hits": estimate.transpile_cache_hit_count,
            "misses": estimate.transpile_cache_miss_count,
            "bypasses": estimate.transpile_cache_bypass_count,
            "evictions": estimate.transpile_cache_eviction_count,
        },
        "seed": estimate.seed,
        "elapsed_seconds": float(elapsed_seconds),
    }


def _prediction_result(
    *,
    prediction: float,
    prediction_standard_error: float,
    actual: float,
    actual_standard_error: float,
) -> dict[str, float | None]:
    error = prediction - actual
    combined_standard_error = math.hypot(
        prediction_standard_error,
        actual_standard_error,
    )
    return {
        "prediction": float(prediction),
        "prediction_standard_error": float(prediction_standard_error),
        "actual": float(actual),
        "actual_standard_error": float(actual_standard_error),
        "prediction_minus_actual": float(error),
        "signed_relative_error": (
            None if actual == 0.0 else float(error / actual)
        ),
        "absolute_relative_error": (
            None if actual == 0.0 else float(abs(error / actual))
        ),
        "combined_standard_error": float(combined_standard_error),
        "absolute_z_score": (
            None
            if combined_standard_error == 0.0
            else float(abs(error) / combined_standard_error)
        ),
    }


def _maximum_relative_error(
    holdouts: Sequence[Mapping[str, Any]],
    model: str,
) -> float | None:
    values = [
        metric_results[model]["absolute_relative_error"]
        for holdout in holdouts
        for metric_results in holdout["metrics"].values()
        if metric_results[model]["absolute_relative_error"] is not None
    ]
    return None if not values else float(max(values))


def _maximum_z_score(
    holdouts: Sequence[Mapping[str, Any]],
    model: str,
) -> float | None:
    values = [
        metric_results[model]["absolute_z_score"]
        for holdout in holdouts
        for metric_results in holdout["metrics"].values()
        if metric_results[model]["absolute_z_score"] is not None
    ]
    return None if not values else float(max(values))


def validate_rte_boundary_cost_model(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    finite_taylor_order: int,
    compiler: CompilerSettings,
    calibration_sample_count: int = 300,
    holdout_sequence_lengths: Sequence[int] = (4, 6),
    holdout_sample_count: int = 300,
    seed: int = 20260823,
    coefficient_atol: float = 1e-12,
    maximum_exact_events: int = 1_000,
    maximum_samples: int = 10_000,
    cache_maximum_entries: int = 4_096,
    maximum_untranspiled_circuit_size: int = 100_000,
    maximum_planned_instruction_applications: int = 1_000_000_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate one- through three-event clusters and test longer sequences.

    Every sequence uses the same short-step time and finite event distribution.
    Pair and triple calibrations and each holdout use independent master seeds.
    Version 1 is restricted to ``K=0`` so a sampled event has one DF basis and
    the fragment collision probability has an unambiguous interpretation.
    """
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    reference_rte_steps = require_integer_count(
        reference_rte_steps,
        name="reference_rte_steps",
        minimum=4,
    )
    finite_taylor_order = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    if finite_taylor_order != 0:
        raise ValueError("Boundary validation v1 currently requires K=0.")
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
            require_integer_count(value, name="holdout_sequence_length", minimum=4)
            for value in holdout_sequence_lengths
        )
    )
    if not holdout_lengths or len(set(holdout_lengths)) != len(holdout_lengths):
        raise ValueError("holdout_sequence_lengths must be nonempty and unique.")
    reference_delta_time = float(reference_delta_time)
    if not math.isfinite(reference_delta_time) or reference_delta_time <= 0.0:
        raise ValueError("reference_delta_time must be finite and positive.")

    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=coefficient_atol,
    )
    if preparation.is_deterministic_only:
        raise ValueError("Boundary validation requires a non-empty RTE tail.")
    short_step_time = reference_delta_time / reference_rte_steps
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

    _single_config, distribution = make_config(1)
    cache = TranspiledCircuitCostCache(maximum_entries=cache_maximum_entries)

    exact_started = time.perf_counter()
    single_event = estimate_exact_compiled_event_cost(
        preparation.rte_preparation,
        distribution,
        compiler,
        max_events=maximum_exact_events,
        maximum_planned_instruction_applications=(
            maximum_planned_instruction_applications
        ),
        cache=cache,
    )
    single_payload = _event_estimate_payload(
        single_event,
        elapsed_seconds=time.perf_counter() - exact_started,
    )

    calibration_estimates: dict[int, CompiledSequenceCostEstimate] = {}
    calibration_payloads: dict[str, Any] = {}
    for length in (2, 3):
        config, current_distribution = make_config(length)
        if not _same_short_step_distribution(current_distribution, distribution):
            raise RuntimeError("Fixed short-step distributions are inconsistent.")
        current_seed = seed + length
        point_started = time.perf_counter()
        estimate = estimate_compiled_occurrence_cost(
            preparation.rte_preparation,
            config,
            distribution,
            compiler,
            sequence_sample_count=calibration_sample_count,
            seed=current_seed,
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
        calibration_estimates[length] = estimate
        calibration_payloads[str(length)] = _sequence_estimate_payload(
            estimate,
            elapsed_seconds=time.perf_counter() - point_started,
        )

    holdout_estimates: dict[int, CompiledSequenceCostEstimate] = {}
    holdout_estimate_payloads: dict[str, Any] = {}
    for length in holdout_lengths:
        config, current_distribution = make_config(length)
        if not _same_short_step_distribution(current_distribution, distribution):
            raise RuntimeError("Fixed short-step distributions are inconsistent.")
        current_seed = seed + 100 + length
        point_started = time.perf_counter()
        estimate = estimate_compiled_occurrence_cost(
            preparation.rte_preparation,
            config,
            distribution,
            compiler,
            sequence_sample_count=holdout_sample_count,
            seed=current_seed,
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
        holdout_estimates[length] = estimate
        holdout_estimate_payloads[str(length)] = _sequence_estimate_payload(
            estimate,
            elapsed_seconds=time.perf_counter() - point_started,
        )

    pair_estimate = calibration_estimates[2]
    triple_estimate = calibration_estimates[3]
    if pair_estimate.sequence_standard_error is None:
        raise RuntimeError("Pair calibration did not report a standard error.")
    if triple_estimate.sequence_standard_error is None:
        raise RuntimeError("Triple calibration did not report a standard error.")
    pair_full = _cost_values(pair_estimate.sequence_expected_cost)
    pair_se = _cost_values(pair_estimate.sequence_standard_error)
    triple_full = _cost_values(triple_estimate.sequence_expected_cost)
    triple_se = _cost_values(triple_estimate.sequence_standard_error)
    single = _cost_values(single_event.expected_cost)

    cluster_metrics: dict[str, Any] = {}
    for metric in _METRICS:
        mu1 = single[metric]
        mu2 = pair_full[metric] - 2.0 * mu1
        mu2_se = pair_se[metric]
        mu3 = triple_full[metric] - 3.0 * mu1 - 2.0 * mu2
        mu3_se = math.hypot(triple_se[metric], 2.0 * mu2_se)
        cluster_metrics[metric] = {
            "single_event_mu1": mu1,
            "pair_boundary_mu2": float(mu2),
            "pair_boundary_mu2_standard_error": float(mu2_se),
            "triple_residual_mu3": float(mu3),
            "triple_residual_mu3_standard_error": float(mu3_se),
            "pair_relative_to_single_event": (
                None if mu1 == 0.0 else float(mu2 / mu1)
            ),
            "triple_relative_to_single_event": (
                None if mu1 == 0.0 else float(mu3 / mu1)
            ),
        }

    holdout_results: list[dict[str, Any]] = []
    for length in holdout_lengths:
        estimate = holdout_estimates[length]
        if estimate.sequence_standard_error is None:
            raise RuntimeError("Holdout estimate did not report a standard error.")
        actual_values = _cost_values(estimate.sequence_expected_cost)
        actual_se_values = _cost_values(estimate.sequence_standard_error)
        metric_results: dict[str, Any] = {}
        for metric in _METRICS:
            mu1 = cluster_metrics[metric]["single_event_mu1"]
            mu2 = cluster_metrics[metric]["pair_boundary_mu2"]
            mu2_se = cluster_metrics[metric][
                "pair_boundary_mu2_standard_error"
            ]
            mu3 = cluster_metrics[metric]["triple_residual_mu3"]
            actual = actual_values[metric]
            actual_se = actual_se_values[metric]
            naive = length * mu1
            pair_prediction = naive + (length - 1) * mu2
            pair_prediction_se = (length - 1) * mu2_se
            triple_prediction = pair_prediction + (length - 2) * mu3
            # In terms of the independent calibration means C2 and C3,
            # C_triple(L)=(3-L)C2+(L-2)C3.
            triple_prediction_se = math.hypot(
                (3 - length) * pair_se[metric],
                (length - 2) * triple_se[metric],
            )
            metric_results[metric] = {
                "naive_additive": _prediction_result(
                    prediction=naive,
                    prediction_standard_error=0.0,
                    actual=actual,
                    actual_standard_error=actual_se,
                ),
                "pair_corrected": _prediction_result(
                    prediction=pair_prediction,
                    prediction_standard_error=pair_prediction_se,
                    actual=actual,
                    actual_standard_error=actual_se,
                ),
                "triple_corrected": _prediction_result(
                    prediction=triple_prediction,
                    prediction_standard_error=triple_prediction_se,
                    actual=actual,
                    actual_standard_error=actual_se,
                ),
            }
        holdout_results.append(
            {
                "sequence_length": length,
                "sample_count": estimate.sample_count,
                "seed": estimate.seed,
                "metrics": metric_results,
            }
        )

    fragment_probabilities: dict[str, float] = defaultdict(float)
    fragment_component_counts: dict[str, int] = defaultdict(int)
    for component in preparation.rte_preparation.symbolic_tail.components:
        key = component.df_fragment_id or component.basis_id or "unassigned"
        fragment_probabilities[key] += component.probability
        fragment_component_counts[key] += 1
    fragment_rows = [
        {
            "fragment_id": fragment_id,
            "probability": float(probability),
            "component_count": fragment_component_counts[fragment_id],
        }
        for fragment_id, probability in sorted(
            fragment_probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    collision_probability = float(
        math.fsum(value * value for value in fragment_probabilities.values())
    )

    payload: dict[str, Any] = {
        "schema_version": RTE_BOUNDARY_COST_VALIDATION_SCHEMA_VERSION,
        "validation_method": RTE_BOUNDARY_COST_VALIDATION_METHOD,
        "final_cost_evaluation_performed": False,
        "acceptance_threshold_decided": False,
        "scope": {
            "purpose": (
                "test whether expectation-level pair and triple boundary "
                "corrections predict longer compiled RTE event sequences"
            ),
            "fixed_short_step_distribution": True,
            "independent_calibration_and_holdout_seeds": True,
            "cluster_formula": (
                "C_L = L*mu1 + (L-1)*mu2 + (L-2)*mu3 + higher clusters"
            ),
            "limitations": (
                "K=0, uncontrolled, no coupling map, one H4 split and one "
                "compiler context"
            ),
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
            "reference_delta_time": reference_delta_time,
            "reference_rte_steps": reference_rte_steps,
            "short_step_time": short_step_time,
            "dimensionless_short_step_time": tau,
            "finite_taylor_order": finite_taylor_order,
            "lambda_r": preparation.exact_rte_lambda_r,
            "component_count": len(
                preparation.rte_preparation.symbolic_tail.components
            ),
            "calibration_sequence_lengths": [1, 2, 3],
            "calibration_sample_count": calibration_sample_count,
            "holdout_sequence_lengths": list(holdout_lengths),
            "holdout_sample_count": holdout_sample_count,
            "seed": seed,
            "compiler": _compiler_payload(compiler),
        },
        "fragment_distribution": {
            "fragment_count": len(fragment_rows),
            "rows": fragment_rows,
            "same_fragment_boundary_probability": collision_probability,
        },
        "single_event_exact": single_payload,
        "calibration_sequence_estimates": calibration_payloads,
        "holdout_sequence_estimates": holdout_estimate_payloads,
        "cluster_coefficients": {"metrics": cluster_metrics},
        "holdout_predictions": holdout_results,
        "summary": {
            "naive_maximum_absolute_relative_error": _maximum_relative_error(
                holdout_results,
                "naive_additive",
            ),
            "pair_maximum_absolute_relative_error": _maximum_relative_error(
                holdout_results,
                "pair_corrected",
            ),
            "triple_maximum_absolute_relative_error": _maximum_relative_error(
                holdout_results,
                "triple_corrected",
            ),
            "naive_maximum_absolute_z_score": _maximum_z_score(
                holdout_results,
                "naive_additive",
            ),
            "pair_maximum_absolute_z_score": _maximum_z_score(
                holdout_results,
                "pair_corrected",
            ),
            "triple_maximum_absolute_z_score": _maximum_z_score(
                holdout_results,
                "triple_corrected",
            ),
            "interpretation_status": (
                "pilot_measurement_only_no_proxy_acceptance_threshold_yet"
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


def validate_rte_boundary_cost_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RTE_BOUNDARY_COST_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Unsupported RTE boundary cost validation schema.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("validation_fingerprint must be a SHA-256 hex string.")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("RTE boundary cost validation fingerprint mismatch.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")


def write_rte_boundary_cost_validation(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_rte_boundary_cost_payload(payload)
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

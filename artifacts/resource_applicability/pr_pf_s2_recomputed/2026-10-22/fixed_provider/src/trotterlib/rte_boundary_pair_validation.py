"""Stratified validation of pair-boundary compiled-cost corrections."""

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
    CompilerSettings,
    make_rte_config,
    require_integer_count,
    step_taylor_truncation_residual_bound,
)
from .rte_compiled_cost import (
    TranspiledCircuitCost,
    TranspiledCircuitCostCache,
    estimate_compiled_occurrence_cost,
)


RTE_BOUNDARY_PAIR_VALIDATION_SCHEMA_VERSION = "rte_boundary_pair_validation_v1"
RTE_BOUNDARY_PAIR_VALIDATION_METHOD = (
    "fixed_short_step_binary_and_fragment_pair_stratification_v1"
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


def _metric_vector(cost: Any) -> tuple[float, ...]:
    return tuple(float(getattr(cost, metric)) for metric in _METRICS)


class _OnlineMetricVector:
    def __init__(self) -> None:
        self.count = 0
        self._mean = [0.0] * len(_METRICS)
        self._m2 = [0.0] * len(_METRICS)

    def update(self, values: Sequence[float]) -> None:
        if len(values) != len(_METRICS):
            raise ValueError("Metric vector length mismatch.")
        self.count += 1
        for index, raw_value in enumerate(values):
            value = float(raw_value)
            delta = value - self._mean[index]
            self._mean[index] += delta / self.count
            self._m2[index] += delta * (value - self._mean[index])

    def mean(self, metric: str) -> float:
        return float(self._mean[_METRICS.index(metric)])

    def standard_error(self, metric: str) -> float | None:
        if self.count < 2:
            return None
        index = _METRICS.index(metric)
        variance = self._m2[index] / (self.count - 1)
        return float(math.sqrt(max(0.0, variance) / self.count))

    def payload(self) -> dict[str, Any]:
        return {
            "sample_count": self.count,
            "metrics": {
                metric: {
                    "mean_boundary_correction": self.mean(metric),
                    "standard_error": self.standard_error(metric),
                }
                for metric in _METRICS
            },
        }


def _event_fragment_id(event: Any) -> str:
    identifiers = tuple(
        identifier for identifier in event.df_fragment_ids if identifier is not None
    )
    if len(identifiers) != 1:
        raise ValueError("Pair validation v1 requires one fragment per K=0 event.")
    return identifiers[0]


class _PairObserver:
    def __init__(self) -> None:
        self.overall = _OnlineMetricVector()
        self.same = _OnlineMetricVector()
        self.different = _OnlineMetricVector()
        self.by_pair: dict[tuple[str, str], _OnlineMetricVector] = {}

    def __call__(
        self,
        request: Any,
        sequence_cost: TranspiledCircuitCost,
        event_costs: tuple[TranspiledCircuitCost, ...],
    ) -> None:
        if len(request.events) != 2 or len(event_costs) != 2:
            raise ValueError("Pair observer requires exactly two events.")
        left = _event_fragment_id(request.events[0])
        right = _event_fragment_id(request.events[1])
        sequence = _metric_vector(sequence_cost)
        first = _metric_vector(event_costs[0])
        second = _metric_vector(event_costs[1])
        correction = tuple(
            sequence_value - first_value - second_value
            for sequence_value, first_value, second_value in zip(
                sequence,
                first,
                second,
                strict=True,
            )
        )
        self.overall.update(correction)
        (self.same if left == right else self.different).update(correction)
        self.by_pair.setdefault((left, right), _OnlineMetricVector()).update(
            correction
        )


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
        "holdout_actual": float(actual),
        "holdout_standard_error": float(actual_standard_error),
        "prediction_minus_holdout": float(difference),
        "absolute_relative_error": (
            None if actual == 0.0 else float(abs(difference / actual))
        ),
        "combined_standard_error": float(combined_standard_error),
        "absolute_z_score": (
            None
            if combined_standard_error == 0.0
            else float(abs(difference) / combined_standard_error)
        ),
    }


def _required_standard_error(
    accumulator: _OnlineMetricVector,
    metric: str,
    *,
    label: str,
) -> float:
    result = accumulator.standard_error(metric)
    if result is None:
        raise RuntimeError(f"{label} has fewer than two samples.")
    return result


def _estimate_payload(estimate: Any, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "sample_count": estimate.sample_count,
        "seed": estimate.seed,
        "sequence_length": estimate.event_count_per_sample,
        "event_stream_rolling_digest": estimate.event_stream_rolling_digest,
        "unique_sequence_circuit_count": estimate.unique_sequence_circuit_count,
        "cache": {
            "hits": estimate.transpile_cache_hit_count,
            "misses": estimate.transpile_cache_miss_count,
            "bypasses": estimate.transpile_cache_bypass_count,
            "evictions": estimate.transpile_cache_eviction_count,
        },
        "elapsed_seconds": float(elapsed_seconds),
    }


def validate_rte_boundary_pairs(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    finite_taylor_order: int,
    compiler: CompilerSettings,
    calibration_sample_count: int = 1_500,
    holdout_sample_count: int = 1_500,
    calibration_seed: int = 20263823,
    holdout_seed: int = 20263824,
    coefficient_atol: float = 1e-12,
    maximum_samples: int = 10_000,
    cache_maximum_entries: int = 8_192,
    maximum_untranspiled_circuit_size: int = 100_000,
    maximum_planned_instruction_applications: int = 1_000_000_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate same/different-fragment corrections and test a new seed."""
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    reference_rte_steps = require_integer_count(
        reference_rte_steps,
        name="reference_rte_steps",
        minimum=2,
    )
    finite_taylor_order = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    if finite_taylor_order != 0:
        raise ValueError("Pair validation v1 currently requires K=0.")
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
    if calibration_seed == holdout_seed:
        raise ValueError("Calibration and holdout seeds must be different.")
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
        raise ValueError("Pair validation requires a non-empty RTE tail.")
    short_step_time = reference_delta_time / reference_rte_steps
    tau = preparation.exact_rte_lambda_r * short_step_time
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=2.0 * short_step_time,
        rte_steps=2,
        truncation_tolerance=_explicit_cutoff_tolerance(
            tau,
            finite_taylor_order,
        ),
        finite_taylor_order=finite_taylor_order,
        seed=calibration_seed,
    )

    fragment_probabilities: dict[str, float] = defaultdict(float)
    for component in preparation.rte_preparation.symbolic_tail.components:
        fragment_id = component.df_fragment_id or component.basis_id or "unassigned"
        fragment_probabilities[fragment_id] += component.probability
    same_probability = float(
        math.fsum(value * value for value in fragment_probabilities.values())
    )
    different_probability = 1.0 - same_probability

    cache = TranspiledCircuitCostCache(maximum_entries=cache_maximum_entries)
    calibration_observer = _PairObserver()
    point_started = time.perf_counter()
    calibration_estimate = estimate_compiled_occurrence_cost(
        preparation.rte_preparation,
        config,
        distribution,
        compiler,
        sequence_sample_count=calibration_sample_count,
        seed=calibration_seed,
        maximum_samples=maximum_samples,
        maximum_rte_steps=2,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        maximum_planned_instruction_applications=(
            maximum_planned_instruction_applications
        ),
        cache=cache,
        sample_observer=calibration_observer,
    )
    calibration_payload = _estimate_payload(
        calibration_estimate,
        time.perf_counter() - point_started,
    )

    holdout_observer = _PairObserver()
    point_started = time.perf_counter()
    holdout_estimate = estimate_compiled_occurrence_cost(
        preparation.rte_preparation,
        config,
        distribution,
        compiler,
        sequence_sample_count=holdout_sample_count,
        seed=holdout_seed,
        maximum_samples=maximum_samples,
        maximum_rte_steps=2,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        maximum_planned_instruction_applications=(
            maximum_planned_instruction_applications
        ),
        cache=cache,
        sample_observer=holdout_observer,
    )
    holdout_payload = _estimate_payload(
        holdout_estimate,
        time.perf_counter() - point_started,
    )

    if calibration_observer.same.count < 2:
        raise RuntimeError("Calibration contains fewer than two same-fragment pairs.")
    if calibration_observer.different.count < 2:
        raise RuntimeError("Calibration contains fewer than two different pairs.")

    model_results: dict[str, Any] = {}
    for metric in _METRICS:
        same_mean = calibration_observer.same.mean(metric)
        different_mean = calibration_observer.different.mean(metric)
        same_se = _required_standard_error(
            calibration_observer.same,
            metric,
            label="same-fragment calibration",
        )
        different_se = _required_standard_error(
            calibration_observer.different,
            metric,
            label="different-fragment calibration",
        )
        holdout_actual = holdout_observer.overall.mean(metric)
        holdout_se = _required_standard_error(
            holdout_observer.overall,
            metric,
            label="pair holdout",
        )
        same_only_prediction = same_probability * same_mean
        same_only_se = same_probability * same_se
        binary_prediction = (
            same_probability * same_mean
            + different_probability * different_mean
        )
        binary_se = math.hypot(
            same_probability * same_se,
            different_probability * different_se,
        )
        model_results[metric] = {
            "same_fragment_only": _prediction_result(
                prediction=same_only_prediction,
                prediction_standard_error=same_only_se,
                actual=holdout_actual,
                actual_standard_error=holdout_se,
            ),
            "same_vs_different": _prediction_result(
                prediction=binary_prediction,
                prediction_standard_error=binary_se,
                actual=holdout_actual,
                actual_standard_error=holdout_se,
            ),
        }

    pair_rows = []
    for (left, right), accumulator in sorted(
        calibration_observer.by_pair.items(),
        key=lambda item: (-item[1].count, item[0]),
    ):
        exact_probability = fragment_probabilities[left] * fragment_probabilities[right]
        pair_rows.append(
            {
                "left_fragment_id": left,
                "right_fragment_id": right,
                "same_fragment": left == right,
                "sample_count": accumulator.count,
                "empirical_probability": (
                    accumulator.count / calibration_sample_count
                ),
                "exact_iid_probability": float(exact_probability),
                "metrics": {
                    metric: {
                        "conditional_mean_boundary_correction": accumulator.mean(
                            metric
                        ),
                        "conditional_standard_error": accumulator.standard_error(
                            metric
                        ),
                        "exact_probability_weighted_contribution": float(
                            exact_probability * accumulator.mean(metric)
                        ),
                    }
                    for metric in _METRICS
                },
            }
        )

    def maximum_result(model: str, field: str) -> float | None:
        values = [
            models[model][field]
            for models in model_results.values()
            if models[model][field] is not None
        ]
        return None if not values else float(max(values))

    payload: dict[str, Any] = {
        "schema_version": RTE_BOUNDARY_PAIR_VALIDATION_SCHEMA_VERSION,
        "validation_method": RTE_BOUNDARY_PAIR_VALIDATION_METHOD,
        "final_cost_evaluation_performed": False,
        "acceptance_threshold_decided": False,
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
            "calibration_sample_count": calibration_sample_count,
            "holdout_sample_count": holdout_sample_count,
            "calibration_seed": calibration_seed,
            "holdout_seed": holdout_seed,
            "compiler": _compiler_payload(compiler),
        },
        "fragment_distribution": {
            "probabilities": dict(sorted(fragment_probabilities.items())),
            "same_fragment_probability": same_probability,
            "different_fragment_probability": different_probability,
        },
        "calibration": {
            "estimate": calibration_payload,
            "overall": calibration_observer.overall.payload(),
            "same_fragment": calibration_observer.same.payload(),
            "different_fragment": calibration_observer.different.payload(),
            "directed_fragment_pair_rows": pair_rows,
        },
        "holdout": {
            "estimate": holdout_payload,
            "overall": holdout_observer.overall.payload(),
            "same_fragment": holdout_observer.same.payload(),
            "different_fragment": holdout_observer.different.payload(),
        },
        "holdout_model_comparison": {"metrics": model_results},
        "summary": {
            "same_fragment_only_maximum_absolute_relative_error": maximum_result(
                "same_fragment_only",
                "absolute_relative_error",
            ),
            "same_vs_different_maximum_absolute_relative_error": maximum_result(
                "same_vs_different",
                "absolute_relative_error",
            ),
            "same_fragment_only_maximum_absolute_z_score": maximum_result(
                "same_fragment_only",
                "absolute_z_score",
            ),
            "same_vs_different_maximum_absolute_z_score": maximum_result(
                "same_vs_different",
                "absolute_z_score",
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


def validate_rte_boundary_pair_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RTE_BOUNDARY_PAIR_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Unsupported RTE boundary pair validation schema.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("validation_fingerprint must be a SHA-256 hex string.")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("RTE boundary pair validation fingerprint mismatch.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")


def write_rte_boundary_pair_validation(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_rte_boundary_pair_payload(payload)
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

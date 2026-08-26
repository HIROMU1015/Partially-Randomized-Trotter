"""Taylor-order-stratified validation of compiled RTE cluster costs.

The ordinary K=2 distribution can make order-2 events too rare for an IID
pilot to observe.  This module samples every order pattern conditionally,
fits an order-dependent one-/two-/three-event cluster model, and recombines
the conditional expectations with the exact analytic order probabilities.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .df_hamiltonian import DFHamiltonian
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
from .df_rte_circuit import DFRTEEventSequenceCircuitRequest
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .rte import (
    CompilerSettings,
    RTEFiniteDistribution,
    _make_event,
    make_rte_config,
    require_integer_count,
    step_taylor_truncation_residual_bound,
)
from .rte_compiled_cost import (
    CompiledMetricAccumulator,
    TranspiledCircuitCostCache,
)


ORDER_STRATIFIED_COST_SCHEMA_VERSION = "rte_order_stratified_cost_validation_v1"
ORDER_STRATIFIED_COST_METHOD = "conditional_order_cluster_l4_l6_holdout_v1"
PAIRED_CLUSTER_SCHEMA_VERSION = "rte_order_stratified_paired_cluster_v1"
PAIRED_CLUSTER_METHOD = "paired_local_window_residual_l4_l6_v1"
PAIRED_K4_L8_SCHEMA_VERSION = "rte_order_stratified_paired_k4_l8_v1"
PAIRED_K4_L8_METHOD = "paired_local_window_k1_k4_residual_l8_v1"
_METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)
_SUPPORTED_ORDERS = (0, 2)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _pattern_key(pattern: Sequence[int]) -> str:
    return ",".join(str(value) for value in pattern)


def _parse_pattern(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _base_key(length: int, pattern: Sequence[int]) -> str:
    return f"{length}:{_pattern_key(pattern)}"


def _derived_seed(
    master_seed: int,
    *,
    role: str,
    length: int,
    pattern: Sequence[int],
) -> int:
    encoded = _canonical_json(
        {
            "master_seed": master_seed,
            "role": role,
            "length": length,
            "pattern": list(pattern),
        }
    ).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _explicit_cutoff_tolerance(tau: float, cutoff: int) -> float:
    residual = step_taylor_truncation_residual_bound(tau, cutoff)
    if not math.isfinite(residual):
        raise ValueError("The requested finite-Taylor residual overflowed.")
    if residual == 0.0:
        return math.ulp(0.0)
    return max(math.nextafter(residual, math.inf), residual * (1.0 + 1e-12))


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


def _statistics_payload(statistics: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "mean": float(item.mean),
            "unbiased_sample_variance": (
                None
                if item.unbiased_sample_variance is None
                else float(item.unbiased_sample_variance)
            ),
            "standard_error": (
                None if item.standard_error is None else float(item.standard_error)
            ),
            "minimum": float(item.minimum),
            "maximum": float(item.maximum),
        }
        for name, item in statistics
    }


def _sample_count_for_pattern(
    pattern: Sequence[int],
    *,
    common_sample_count: int,
    single_rare_sample_count: int,
    multi_rare_sample_count: int,
) -> int:
    rare_count = sum(order == 2 for order in pattern)
    if rare_count == 0:
        return common_sample_count
    if rare_count == 1:
        return single_rare_sample_count
    return multi_rare_sample_count


def _compile_length_strata(task: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    hamiltonian = task["hamiltonian"]
    length = int(task["length"])
    role = str(task["role"])
    compiler = task["compiler"]
    partition = split_df_hamiltonian_by_ld(hamiltonian, int(task["ld"]))
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
    )
    short_step_time = float(task["short_step_time"])
    tau = preparation.exact_rte_lambda_r * short_step_time
    _config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=short_step_time * length,
        rte_steps=length,
        truncation_tolerance=_explicit_cutoff_tolerance(tau, 2),
        finite_taylor_order=2,
        seed=int(task["master_seed"]),
    )
    if distribution.orders != _SUPPORTED_ORDERS:
        raise RuntimeError("The stratified K=2 validator requires orders (0, 2).")

    components = preparation.rte_preparation.symbolic_tail.components
    component_probability_sum = math.fsum(item.probability for item in components)
    component_probabilities = np.asarray(
        [item.probability / component_probability_sum for item in components],
        dtype=float,
    )
    order_indices = {
        order: distribution.orders.index(order) for order in distribution.orders
    }
    order_probabilities = dict(
        zip(
            distribution.orders,
            distribution.order_probabilities,
            strict=True,
        )
    )
    cache = TranspiledCircuitCostCache(
        maximum_entries=int(task["cache_maximum_entries"])
    )
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    strata: dict[str, Any] = {}
    for pattern in itertools.product(_SUPPORTED_ORDERS, repeat=length):
        sample_count = _sample_count_for_pattern(
            pattern,
            common_sample_count=int(task["common_sample_count"]),
            single_rare_sample_count=int(task["single_rare_sample_count"]),
            multi_rare_sample_count=int(task["multi_rare_sample_count"]),
        )
        seed = _derived_seed(
            int(task["master_seed"]),
            role=role,
            length=length,
            pattern=pattern,
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        accumulator = CompiledMetricAccumulator(weighted=False)
        digest = hashlib.sha256()
        unique_circuits: set[str] = set()
        stratum_started = time.perf_counter()
        for _ in range(sample_count):
            events = []
            for order in pattern:
                indices = rng.choice(
                    len(components),
                    size=order + 1,
                    p=component_probabilities,
                )
                event = _make_event(
                    indices,
                    components,
                    distribution,
                    order_indices[order],
                )
                events.append(event)
                encoded = json.dumps(
                    event.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            request = DFRTEEventSequenceCircuitRequest(
                events=tuple(events),
                component_specs=preparation.rte_preparation.component_specs,
                controlled=False,
                ancilla_qubit=None,
                cancel_adjacent_equal_bases=True,
                tail_id=preparation.rte_preparation.symbolic_tail.tail_id,
                tail_hash=preparation.rte_preparation.symbolic_tail.tail_hash,
                occurrence_rte_steps=length,
            )
            built = builder.build_sequence(request)
            if built.circuit.size() > int(task["maximum_circuit_size"]):
                raise ValueError("A conditional order circuit exceeded the size limit.")
            cost, cache_key, _cached = cache.get_or_transpile(
                built.circuit,
                compiler,
                circuit_fingerprint=built.circuit_fingerprint,
            )
            unique_circuits.add(cache_key)
            accumulator.update(cost)
        statistics = accumulator.finalize()
        pattern_probability = math.prod(order_probabilities[item] for item in pattern)
        strata[_pattern_key(pattern)] = {
            "order_pattern": list(pattern),
            "rare_order_count": sum(order == 2 for order in pattern),
            "pattern_probability": float(pattern_probability),
            "sample_count": sample_count,
            "seed": seed,
            "forced_order_counts": {
                "0": sample_count * sum(order == 0 for order in pattern),
                "2": sample_count * sum(order == 2 for order in pattern),
            },
            "metric_statistics": _statistics_payload(statistics),
            "event_stream_rolling_digest": digest.hexdigest(),
            "unique_compiled_circuit_count": len(unique_circuits),
            "elapsed_seconds": float(time.perf_counter() - stratum_started),
        }
    probability_sum = math.fsum(
        item["pattern_probability"] for item in strata.values()
    )
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-14):
        raise RuntimeError("Conditional order-pattern probabilities do not sum to one.")
    return {
        "role": role,
        "sequence_length": length,
        "preparation_hash": preparation.preparation_hash,
        "partition_hash": preparation.partition_hash,
        "distribution": distribution.to_dict(),
        "strata": strata,
        "pattern_probability_sum": float(probability_sum),
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "cache_hits": cache.hit_count,
            "cache_misses": cache.miss_count,
            "cache_evictions": cache.eviction_count,
        },
    }


def _add_form(
    target: dict[str, float],
    source: Mapping[str, float],
    *,
    scale: float = 1.0,
) -> None:
    for key, coefficient in source.items():
        updated = target.get(key, 0.0) + scale * coefficient
        if abs(updated) <= 1e-15:
            target.pop(key, None)
        else:
            target[key] = float(updated)


def _cluster_prediction_forms() -> dict[int, dict[str, dict[str, float]]]:
    singleton: dict[str, dict[str, float]] = {}
    for pattern in itertools.product(_SUPPORTED_ORDERS, repeat=1):
        singleton[_pattern_key(pattern)] = {_base_key(1, pattern): 1.0}

    pair: dict[str, dict[str, float]] = {}
    for pattern in itertools.product(_SUPPORTED_ORDERS, repeat=2):
        form = {_base_key(2, pattern): 1.0}
        for order in pattern:
            _add_form(form, singleton[_pattern_key((order,))], scale=-1.0)
        pair[_pattern_key(pattern)] = form

    triple: dict[str, dict[str, float]] = {}
    for pattern in itertools.product(_SUPPORTED_ORDERS, repeat=3):
        form = {_base_key(3, pattern): 1.0}
        for order in pattern:
            _add_form(form, singleton[_pattern_key((order,))], scale=-1.0)
        _add_form(form, pair[_pattern_key(pattern[:2])], scale=-1.0)
        _add_form(form, pair[_pattern_key(pattern[1:])], scale=-1.0)
        triple[_pattern_key(pattern)] = form

    result: dict[int, dict[str, dict[str, float]]] = {}
    for length in (4, 6):
        result[length] = {}
        for pattern in itertools.product(_SUPPORTED_ORDERS, repeat=length):
            form: dict[str, float] = {}
            for order in pattern:
                _add_form(form, singleton[_pattern_key((order,))])
            for start in range(length - 1):
                _add_form(form, pair[_pattern_key(pattern[start : start + 2])])
            for start in range(length - 2):
                _add_form(form, triple[_pattern_key(pattern[start : start + 3])])
            result[length][_pattern_key(pattern)] = form
    return result


def _calibration_lookup(
    calibration: Mapping[int, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        _base_key(length, _parse_pattern(pattern)): stratum
        for length, payload in calibration.items()
        for pattern, stratum in payload["strata"].items()
    }


def _evaluate_form(
    form: Mapping[str, float],
    lookup: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> tuple[float, float]:
    mean = math.fsum(
        coefficient * lookup[key]["metric_statistics"][metric]["mean"]
        for key, coefficient in form.items()
    )
    variance = math.fsum(
        (
            coefficient
            * lookup[key]["metric_statistics"][metric]["standard_error"]
        )
        ** 2
        for key, coefficient in form.items()
    )
    return float(mean), float(math.sqrt(variance))


def _comparison(
    *,
    prediction: float,
    prediction_standard_error: float,
    actual: float,
    actual_standard_error: float,
) -> dict[str, float | None]:
    difference = prediction - actual
    combined = math.hypot(prediction_standard_error, actual_standard_error)
    return {
        "prediction": float(prediction),
        "prediction_standard_error": float(prediction_standard_error),
        "actual": float(actual),
        "actual_standard_error": float(actual_standard_error),
        "prediction_minus_actual": float(difference),
        "absolute_relative_error": (
            None if actual == 0.0 else float(abs(difference / actual))
        ),
        "absolute_z_score": None if combined == 0.0 else float(abs(difference) / combined),
        "combined_standard_error": float(combined),
        "pointwise_normal_relative_95_upper_diagnostic": (
            None
            if actual == 0.0
            else float((abs(difference) + 1.96 * combined) / abs(actual))
        ),
    }


def _maximum_present(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else float(max(present))


def _weighted_actual(
    strata: Mapping[str, Any],
    patterns: Sequence[str],
    weights: Sequence[float],
    metric: str,
) -> tuple[float, float]:
    mean = math.fsum(
        weight * strata[pattern]["metric_statistics"][metric]["mean"]
        for pattern, weight in zip(patterns, weights, strict=True)
    )
    variance = math.fsum(
        (
            weight
            * strata[pattern]["metric_statistics"][metric]["standard_error"]
        )
        ** 2
        for pattern, weight in zip(patterns, weights, strict=True)
    )
    return float(mean), float(math.sqrt(variance))


def _aggregate_prediction_form(
    prediction_forms: Mapping[str, Mapping[str, float]],
    patterns: Sequence[str],
    weights: Sequence[float],
) -> dict[str, float]:
    form: dict[str, float] = {}
    for pattern, weight in zip(patterns, weights, strict=True):
        _add_form(form, prediction_forms[pattern], scale=weight)
    return form


def _aggregate_result(
    *,
    patterns: Sequence[str],
    raw_weights: Sequence[float],
    prediction_forms: Mapping[str, Mapping[str, float]],
    calibration_lookup: Mapping[str, Mapping[str, Any]],
    holdout_strata: Mapping[str, Any],
) -> dict[str, Any]:
    total = math.fsum(raw_weights)
    if total <= 0.0:
        raise ValueError("Aggregate stratum probability must be positive.")
    weights = tuple(weight / total for weight in raw_weights)
    form = _aggregate_prediction_form(prediction_forms, patterns, weights)
    metrics: dict[str, Any] = {}
    for metric in _METRICS:
        prediction, prediction_se = _evaluate_form(
            form,
            calibration_lookup,
            metric,
        )
        actual, actual_se = _weighted_actual(
            holdout_strata,
            patterns,
            weights,
            metric,
        )
        metrics[metric] = _comparison(
            prediction=prediction,
            prediction_standard_error=prediction_se,
            actual=actual,
            actual_standard_error=actual_se,
        )
    return {
        "unnormalized_probability": float(total),
        "normalized_pattern_weights": {
            pattern: float(weight)
            for pattern, weight in zip(patterns, weights, strict=True)
        },
        "prediction_form": dict(sorted(form.items())),
        "metrics": metrics,
    }


def validate_order_stratified_k2_cost_model(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    compiler: CompilerSettings,
    common_sample_count: int,
    single_rare_sample_count: int,
    multi_rare_sample_count: int,
    seed: int,
    maximum_workers: int = 1,
    cache_maximum_entries: int = 8_192,
    maximum_circuit_size: int = 100_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an order-dependent triple cluster model at K=2."""
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    reference_rte_steps = require_integer_count(
        reference_rte_steps,
        name="reference_rte_steps",
        minimum=1,
    )
    for name, value in (
        ("common_sample_count", common_sample_count),
        ("single_rare_sample_count", single_rare_sample_count),
        ("multi_rare_sample_count", multi_rare_sample_count),
    ):
        require_integer_count(value, name=name, minimum=2)
    maximum_workers = require_integer_count(
        maximum_workers,
        name="maximum_workers",
        minimum=1,
    )
    if not math.isfinite(reference_delta_time) or reference_delta_time <= 0.0:
        raise ValueError("reference_delta_time must be finite and positive.")

    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
    )
    if preparation.is_deterministic_only:
        raise ValueError("Order-stratified validation requires a non-empty tail.")
    short_step_time = float(reference_delta_time) / reference_rte_steps
    tau = preparation.exact_rte_lambda_r * short_step_time
    _config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=short_step_time,
        rte_steps=1,
        truncation_tolerance=_explicit_cutoff_tolerance(tau, 2),
        finite_taylor_order=2,
        seed=seed,
    )
    tasks = []
    for role, lengths in (("calibration", (1, 2, 3)), ("holdout", (4, 6))):
        for length in lengths:
            tasks.append(
                {
                    "hamiltonian": hamiltonian,
                    "ld": ld,
                    "short_step_time": short_step_time,
                    "compiler": compiler,
                    "length": length,
                    "role": role,
                    "master_seed": seed,
                    "common_sample_count": common_sample_count,
                    "single_rare_sample_count": single_rare_sample_count,
                    "multi_rare_sample_count": multi_rare_sample_count,
                    "cache_maximum_entries": cache_maximum_entries,
                    "maximum_circuit_size": maximum_circuit_size,
                }
            )

    outputs = []
    if maximum_workers == 1:
        outputs = [_compile_length_strata(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(maximum_workers, len(tasks)),
            mp_context=context,
        ) as executor:
            futures = [executor.submit(_compile_length_strata, task) for task in tasks]
            for future in as_completed(futures):
                outputs.append(future.result())

    preparation_hashes = {preparation.preparation_hash}
    preparation_hashes.update(item["preparation_hash"] for item in outputs)
    if len(preparation_hashes) != 1:
        raise RuntimeError("Order-stratified worker preparation hashes differ.")
    partition_hashes = {preparation.partition_hash}
    partition_hashes.update(item["partition_hash"] for item in outputs)
    if len(partition_hashes) != 1:
        raise RuntimeError("Order-stratified worker partition hashes differ.")

    calibration = {
        item["sequence_length"]: item
        for item in outputs
        if item["role"] == "calibration"
    }
    holdout = {
        item["sequence_length"]: item
        for item in outputs
        if item["role"] == "holdout"
    }
    if set(calibration) != {1, 2, 3} or set(holdout) != {4, 6}:
        raise RuntimeError("Required calibration or holdout lengths are missing.")

    forms = _cluster_prediction_forms()
    lookup = _calibration_lookup(calibration)
    holdout_results: dict[str, Any] = {}
    for length in (4, 6):
        strata = holdout[length]["strata"]
        all_patterns = tuple(sorted(strata))
        probabilities = tuple(strata[key]["pattern_probability"] for key in all_patterns)
        zero_patterns = tuple(
            key for key in all_patterns if strata[key]["rare_order_count"] == 0
        )
        one_patterns = tuple(
            key for key in all_patterns if strata[key]["rare_order_count"] == 1
        )
        multi_patterns = tuple(
            key for key in all_patterns if strata[key]["rare_order_count"] >= 2
        )
        holdout_results[str(length)] = {
            "full_distribution": _aggregate_result(
                patterns=all_patterns,
                raw_weights=probabilities,
                prediction_forms=forms[length],
                calibration_lookup=lookup,
                holdout_strata=strata,
            ),
            "zero_order2_condition": _aggregate_result(
                patterns=zero_patterns,
                raw_weights=tuple(1.0 for _ in zero_patterns),
                prediction_forms=forms[length],
                calibration_lookup=lookup,
                holdout_strata=strata,
            ),
            "exactly_one_order2_condition": _aggregate_result(
                patterns=one_patterns,
                raw_weights=tuple(
                    strata[key]["pattern_probability"] for key in one_patterns
                ),
                prediction_forms=forms[length],
                calibration_lookup=lookup,
                holdout_strata=strata,
            ),
            "two_or_more_order2_condition": _aggregate_result(
                patterns=multi_patterns,
                raw_weights=tuple(
                    strata[key]["pattern_probability"] for key in multi_patterns
                ),
                prediction_forms=forms[length],
                calibration_lookup=lookup,
                holdout_strata=strata,
            ),
        }

    primary_results = [
        holdout_results[str(length)][scope]["metrics"]["rz_count"]
        for length in (4, 6)
        for scope in (
            "full_distribution",
            "exactly_one_order2_condition",
            "two_or_more_order2_condition",
        )
    ]
    all_scope_results = [
        value
        for length in (4, 6)
        for aggregate in holdout_results[str(length)].values()
        for value in aggregate["metrics"].values()
    ]
    payload: dict[str, Any] = {
        "schema_version": ORDER_STRATIFIED_COST_SCHEMA_VERSION,
        "validation_method": ORDER_STRATIFIED_COST_METHOD,
        "final_cost_evaluation_performed": False,
        "acceptance_policy": {
            "primary_metric": "rz_count",
            "required_scopes": [
                "full_distribution",
                "exactly_one_order2_condition",
                "two_or_more_order2_condition",
            ],
            "relative_tolerance": 0.05,
            "normal_approximation_is_diagnostic_only": True,
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
            "finite_taylor_order": 2,
            "distribution": distribution.to_dict(),
            "sample_allocation": {
                "zero_order2_pattern": common_sample_count,
                "exactly_one_order2_pattern": single_rare_sample_count,
                "two_or_more_order2_pattern": multi_rare_sample_count,
            },
            "seed": seed,
            "compiler": _compiler_payload(compiler),
        },
        "calibration": {str(key): value for key, value in calibration.items()},
        "holdout": {str(key): value for key, value in holdout.items()},
        "holdout_predictions": holdout_results,
        "summary": {
            "primary_maximum_absolute_relative_error": _maximum_present(
                [value["absolute_relative_error"] for value in primary_results]
            ),
            "primary_maximum_absolute_z_score": _maximum_present(
                [value["absolute_z_score"] for value in primary_results]
            ),
            "primary_maximum_pointwise_normal_95_upper_diagnostic": _maximum_present(
                [
                    value["pointwise_normal_relative_95_upper_diagnostic"]
                    for value in primary_results
                ]
            ),
            "primary_point_tolerance_passed": all(
                value["absolute_relative_error"] is not None
                and value["absolute_relative_error"] <= 0.05
                for value in primary_results
            ),
            "primary_diagnostic_95_tolerance_passed": all(
                value["pointwise_normal_relative_95_upper_diagnostic"] is not None
                and value["pointwise_normal_relative_95_upper_diagnostic"] <= 0.05
                for value in primary_results
            ),
            "all_scopes_metrics_maximum_absolute_relative_error": _maximum_present(
                [value["absolute_relative_error"] for value in all_scope_results]
            ),
            "all_scopes_metrics_maximum_absolute_z_score": _maximum_present(
                [value["absolute_z_score"] for value in all_scope_results]
            ),
        },
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "maximum_workers": min(maximum_workers, len(tasks)),
            "worker_seconds": {
                f"{item['role']}_L{item['sequence_length']}": item["performance"]
                for item in outputs
            },
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_order_stratified_cost_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != ORDER_STRATIFIED_COST_SCHEMA_VERSION:
        raise ValueError("Unsupported order-stratified cost schema.")
    if payload.get("validation_method") != ORDER_STRATIFIED_COST_METHOD:
        raise ValueError("Unsupported order-stratified validation method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")
    distribution = payload.get("configuration", {}).get("distribution", {})
    if distribution.get("orders") != [0, 2]:
        raise ValueError("The serialized order distribution must be [0, 2].")
    probabilities = distribution.get("order_probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != 2:
        raise ValueError("Two serialized order probabilities are required.")
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-14):
        raise ValueError("Serialized order probabilities must sum to one.")
    all_seeds = []
    for role in ("calibration", "holdout"):
        lengths = payload.get(role)
        if not isinstance(lengths, Mapping):
            raise ValueError(f"{role} strata are missing.")
        for length, length_payload in lengths.items():
            expected_patterns = 2 ** int(length)
            strata = length_payload.get("strata")
            if not isinstance(strata, Mapping) or len(strata) != expected_patterns:
                raise ValueError(f"{role} L={length} pattern count is invalid.")
            if not math.isclose(
                float(length_payload.get("pattern_probability_sum")),
                1.0,
                abs_tol=1e-14,
            ):
                raise ValueError("Pattern probabilities must sum to one.")
            for key, stratum in strata.items():
                pattern = _parse_pattern(key)
                if list(pattern) != stratum.get("order_pattern"):
                    raise ValueError("Stratum key and order pattern differ.")
                if any(order not in _SUPPORTED_ORDERS for order in pattern):
                    raise ValueError("A stratum contains an unsupported order.")
                all_seeds.append(stratum.get("seed"))
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Conditional stratum seeds must be unique.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("validation_fingerprint must be a SHA-256 hex string.")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Order-stratified cost fingerprint mismatch.")


def write_order_stratified_cost_validation(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_order_stratified_cost_payload(payload)
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


def _sample_statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("At least two paired samples are required.")
    variance = float(np.var(array, ddof=1))
    return {
        "mean": float(np.mean(array)),
        "unbiased_sample_variance": variance,
        "standard_error": float(math.sqrt(variance / array.size)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _cost_vector(cost: Any) -> np.ndarray:
    return np.asarray([float(getattr(cost, metric)) for metric in _METRICS])


def _connected_window_prediction(
    windows: Mapping[tuple[int, int], np.ndarray],
    *,
    sequence_length: int,
    maximum_cluster_length: int,
) -> np.ndarray:
    """Sum interval-connected coefficients through the requested cluster length."""
    if maximum_cluster_length < 1 or maximum_cluster_length > sequence_length:
        raise ValueError("maximum_cluster_length must lie within the sequence.")
    prediction = np.zeros(len(_METRICS), dtype=float)
    for start in range(sequence_length):
        prediction += windows[(start, 1)]
    for window_length in range(2, maximum_cluster_length + 1):
        for start in range(sequence_length - window_length + 1):
            connected = (
                windows[(start, window_length)]
                - windows[(start, window_length - 1)]
                - windows[(start + 1, window_length - 1)]
            )
            if window_length > 2:
                connected += windows[(start + 1, window_length - 2)]
            prediction += connected
    return prediction


def _compile_paired_length_strata(task: Mapping[str, Any]) -> dict[str, Any]:
    """Compile full and local windows from each identical sampled trajectory."""
    started = time.perf_counter()
    hamiltonian = task["hamiltonian"]
    length = int(task["length"])
    compiler = task["compiler"]
    partition = split_df_hamiltonian_by_ld(hamiltonian, int(task["ld"]))
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
    )
    short_step_time = float(task["short_step_time"])
    tau = preparation.exact_rte_lambda_r * short_step_time
    _config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=short_step_time * length,
        rte_steps=length,
        truncation_tolerance=_explicit_cutoff_tolerance(tau, 2),
        finite_taylor_order=2,
        seed=int(task["master_seed"]),
    )
    components = preparation.rte_preparation.symbolic_tail.components
    probability_sum = math.fsum(item.probability for item in components)
    component_probabilities = np.asarray(
        [item.probability / probability_sum for item in components],
        dtype=float,
    )
    order_indices = {
        order: distribution.orders.index(order) for order in distribution.orders
    }
    order_probabilities = dict(
        zip(distribution.orders, distribution.order_probabilities, strict=True)
    )
    cache = TranspiledCircuitCostCache(
        maximum_entries=int(task["cache_maximum_entries"]),
        persistent_path=task.get("persistent_cache_path"),
    )
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )

    def compile_window(events: Sequence[Any]) -> tuple[np.ndarray, str]:
        request = DFRTEEventSequenceCircuitRequest(
            events=tuple(events),
            component_specs=preparation.rte_preparation.component_specs,
            controlled=False,
            ancilla_qubit=None,
            cancel_adjacent_equal_bases=True,
            tail_id=preparation.rte_preparation.symbolic_tail.tail_id,
            tail_hash=preparation.rte_preparation.symbolic_tail.tail_hash,
            occurrence_rte_steps=len(events),
        )
        built = builder.build_sequence(request)
        if built.circuit.size() > int(task["maximum_circuit_size"]):
            raise ValueError("A paired local-window circuit exceeded the size limit.")
        cost, cache_key, _cached = cache.get_or_transpile(
            built.circuit,
            compiler,
            circuit_fingerprint=built.circuit_fingerprint,
        )
        return _cost_vector(cost), cache_key

    maximum_cluster_length = int(task.get("maximum_cluster_length", 3))
    if maximum_cluster_length < 1 or maximum_cluster_length > length:
        raise ValueError("Invalid paired maximum cluster length.")
    multi_rare_sample_count = task.get("multi_rare_sample_count")
    requested_patterns = task.get("patterns")
    if requested_patterns is not None:
        patterns = [tuple(int(order) for order in pattern) for pattern in requested_patterns]
    elif multi_rare_sample_count is None:
        patterns = [(0,) * length]
        patterns.extend(
            tuple(2 if position == rare_position else 0 for position in range(length))
            for rare_position in range(length)
        )
    else:
        patterns = list(itertools.product(_SUPPORTED_ORDERS, repeat=length))
    strata: dict[str, Any] = {}
    for pattern in patterns:
        if len(pattern) != length or any(order not in _SUPPORTED_ORDERS for order in pattern):
            raise ValueError("A paired order pattern is invalid.")
        rare_count = sum(order == 2 for order in pattern)
        if rare_count == 0:
            sample_count = int(task["common_sample_count"])
        elif rare_count == 1:
            sample_count = int(task["single_rare_sample_count"])
        else:
            sample_count = int(multi_rare_sample_count)
        seed = _derived_seed(
            int(task["master_seed"]),
            role="paired_holdout",
            length=length,
            pattern=pattern,
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        actual_samples = [[] for _ in _METRICS]
        predicted_samples = [[] for _ in _METRICS]
        residual_samples = [[] for _ in _METRICS]
        digest = hashlib.sha256()
        unique_circuits: set[str] = set()
        stratum_started = time.perf_counter()
        for _ in range(sample_count):
            events = []
            for order in pattern:
                indices = rng.choice(
                    len(components),
                    size=order + 1,
                    p=component_probabilities,
                )
                event = _make_event(
                    indices,
                    components,
                    distribution,
                    order_indices[order],
                )
                events.append(event)
                encoded = json.dumps(
                    event.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)

            windows: dict[tuple[int, int], np.ndarray] = {}
            for window_length in range(1, maximum_cluster_length + 1):
                for start in range(length - window_length + 1):
                    values, cache_key = compile_window(
                        events[start : start + window_length]
                    )
                    windows[(start, window_length)] = values
                    unique_circuits.add(cache_key)
            actual, cache_key = compile_window(events)
            unique_circuits.add(cache_key)
            predicted = _connected_window_prediction(
                windows,
                sequence_length=length,
                maximum_cluster_length=maximum_cluster_length,
            )
            residual = predicted - actual
            for metric_index in range(len(_METRICS)):
                actual_samples[metric_index].append(actual[metric_index])
                predicted_samples[metric_index].append(predicted[metric_index])
                residual_samples[metric_index].append(residual[metric_index])

        strata[_pattern_key(pattern)] = {
            "order_pattern": list(pattern),
            "rare_order_count": sum(order == 2 for order in pattern),
            "pattern_probability": float(
                math.prod(order_probabilities[order] for order in pattern)
            ),
            "sample_count": sample_count,
            "seed": seed,
            "forced_order_counts": {
                "0": sample_count * sum(order == 0 for order in pattern),
                "2": sample_count * sum(order == 2 for order in pattern),
            },
            "actual_statistics": {
                metric: _sample_statistics(actual_samples[index])
                for index, metric in enumerate(_METRICS)
            },
            "local_prediction_statistics": {
                metric: _sample_statistics(predicted_samples[index])
                for index, metric in enumerate(_METRICS)
            },
            "paired_prediction_minus_actual_statistics": {
                metric: _sample_statistics(residual_samples[index])
                for index, metric in enumerate(_METRICS)
            },
            "event_stream_rolling_digest": digest.hexdigest(),
            "unique_compiled_circuit_count": len(unique_circuits),
            "elapsed_seconds": float(time.perf_counter() - stratum_started),
        }
    return {
        "sequence_length": length,
        "preparation_hash": preparation.preparation_hash,
        "partition_hash": preparation.partition_hash,
        "distribution": distribution.to_dict(),
        "strata": strata,
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "cache_hits": cache.hit_count,
            "cache_misses": cache.miss_count,
            "cache_evictions": cache.eviction_count,
            "persistent_cache_hits": cache.persistent_hit_count,
            "persistent_cache_writes": cache.persistent_write_count,
        },
    }


def _aggregate_paired_condition(
    strata: Mapping[str, Any],
    patterns: Sequence[str],
) -> dict[str, Any]:
    total_probability = math.fsum(
        float(strata[pattern]["pattern_probability"]) for pattern in patterns
    )
    if total_probability <= 0.0:
        raise ValueError("Paired conditional probability must be positive.")
    weights = [
        float(strata[pattern]["pattern_probability"]) / total_probability
        for pattern in patterns
    ]
    metrics: dict[str, Any] = {}
    for metric in _METRICS:
        actual = math.fsum(
            weight * strata[pattern]["actual_statistics"][metric]["mean"]
            for pattern, weight in zip(patterns, weights, strict=True)
        )
        prediction = math.fsum(
            weight
            * strata[pattern]["local_prediction_statistics"][metric]["mean"]
            for pattern, weight in zip(patterns, weights, strict=True)
        )
        residual = math.fsum(
            weight
            * strata[pattern]["paired_prediction_minus_actual_statistics"][metric][
                "mean"
            ]
            for pattern, weight in zip(patterns, weights, strict=True)
        )
        residual_se = math.sqrt(
            math.fsum(
                (
                    weight
                    * strata[pattern][
                        "paired_prediction_minus_actual_statistics"
                    ][metric]["standard_error"]
                )
                ** 2
                for pattern, weight in zip(patterns, weights, strict=True)
            )
        )
        metrics[metric] = {
            "prediction": float(prediction),
            "actual": float(actual),
            "prediction_minus_actual": float(residual),
            "paired_residual_standard_error": float(residual_se),
            "absolute_relative_error": (
                None if actual == 0.0 else float(abs(residual) / abs(actual))
            ),
            "absolute_z_score": (
                None if residual_se == 0.0 else float(abs(residual) / residual_se)
            ),
            "pointwise_normal_relative_95_upper_diagnostic": (
                None
                if actual == 0.0
                else float((abs(residual) + 1.96 * residual_se) / abs(actual))
            ),
        }
    return {
        "patterns": list(patterns),
        "conditional_pattern_weights": {
            pattern: float(weight)
            for pattern, weight in zip(patterns, weights, strict=True)
        },
        "metrics": metrics,
    }


def validate_paired_order_stratified_k2_cluster_model(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    compiler: CompilerSettings,
    common_sample_count: int,
    single_rare_sample_count: int,
    seed: int,
    multi_rare_sample_count: int | None = None,
    maximum_workers: int = 1,
    cache_maximum_entries: int = 16_384,
    maximum_circuit_size: int = 100_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Directly estimate local-cluster truncation residuals on paired circuits."""
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    reference_rte_steps = require_integer_count(
        reference_rte_steps, name="reference_rte_steps", minimum=1
    )
    common_sample_count = require_integer_count(
        common_sample_count, name="common_sample_count", minimum=2
    )
    single_rare_sample_count = require_integer_count(
        single_rare_sample_count, name="single_rare_sample_count", minimum=2
    )
    if multi_rare_sample_count is not None:
        multi_rare_sample_count = require_integer_count(
            multi_rare_sample_count,
            name="multi_rare_sample_count",
            minimum=2,
        )
    maximum_workers = require_integer_count(
        maximum_workers, name="maximum_workers", minimum=1
    )
    if not math.isfinite(reference_delta_time) or reference_delta_time <= 0.0:
        raise ValueError("reference_delta_time must be finite and positive.")
    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian, partition, identity_policy="extract_identity_phase"
    )
    if preparation.is_deterministic_only:
        raise ValueError("Paired validation requires a non-empty RTE tail.")
    short_step_time = float(reference_delta_time) / reference_rte_steps
    tasks = [
        {
            "hamiltonian": hamiltonian,
            "ld": ld,
            "short_step_time": short_step_time,
            "compiler": compiler,
            "length": length,
            "master_seed": seed,
            "common_sample_count": common_sample_count,
            "single_rare_sample_count": single_rare_sample_count,
            "multi_rare_sample_count": multi_rare_sample_count,
            "cache_maximum_entries": cache_maximum_entries,
            "maximum_circuit_size": maximum_circuit_size,
        }
        for length in (4, 6)
    ]
    if maximum_workers == 1:
        outputs = [_compile_paired_length_strata(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(maximum_workers, len(tasks)), mp_context=context
        ) as executor:
            outputs = [future.result() for future in as_completed(
                executor.submit(_compile_paired_length_strata, task) for task in tasks
            )]
    preparation_hashes = {preparation.preparation_hash}
    preparation_hashes.update(item["preparation_hash"] for item in outputs)
    partition_hashes = {preparation.partition_hash}
    partition_hashes.update(item["partition_hash"] for item in outputs)
    if len(preparation_hashes) != 1 or len(partition_hashes) != 1:
        raise RuntimeError("Paired validation worker preparations differ.")
    holdout = {item["sequence_length"]: item for item in outputs}
    results: dict[str, Any] = {}
    for length in (4, 6):
        strata = holdout[length]["strata"]
        zero = tuple(key for key, value in strata.items() if value["rare_order_count"] == 0)
        one = tuple(key for key, value in strata.items() if value["rare_order_count"] == 1)
        current_results = {
            "zero_order2_condition": _aggregate_paired_condition(strata, zero),
            "exactly_one_order2_condition": _aggregate_paired_condition(strata, one),
        }
        if multi_rare_sample_count is not None:
            multi = tuple(
                key
                for key, value in strata.items()
                if value["rare_order_count"] >= 2
            )
            current_results["two_or_more_order2_condition"] = (
                _aggregate_paired_condition(strata, multi)
            )
        results[str(length)] = current_results
    required_conditions = [
        "zero_order2_condition",
        "exactly_one_order2_condition",
    ]
    if multi_rare_sample_count is not None:
        required_conditions.append("two_or_more_order2_condition")
    primary = [
        results[str(length)][scope]["metrics"]["rz_count"]
        for length in (4, 6)
        for scope in required_conditions
    ]
    payload: dict[str, Any] = {
        "schema_version": PAIRED_CLUSTER_SCHEMA_VERSION,
        "validation_method": PAIRED_CLUSTER_METHOD,
        "final_cost_evaluation_performed": False,
        "acceptance_policy": {
            "primary_metric": "rz_count",
            "required_conditions": required_conditions,
            "relative_tolerance": 0.05,
            "normal_approximation_is_diagnostic_only": True,
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
            "finite_taylor_order": 2,
            "common_sample_count": common_sample_count,
            "single_rare_sample_count_per_position": single_rare_sample_count,
            "multi_rare_sample_count_per_pattern": multi_rare_sample_count,
            "seed": seed,
            "compiler": _compiler_payload(compiler),
        },
        "holdout": {str(key): value for key, value in holdout.items()},
        "paired_residual_results": results,
        "summary": {
            "primary_maximum_absolute_relative_error": _maximum_present(
                [value["absolute_relative_error"] for value in primary]
            ),
            "primary_maximum_absolute_z_score": _maximum_present(
                [value["absolute_z_score"] for value in primary]
            ),
            "primary_maximum_pointwise_normal_95_upper_diagnostic": _maximum_present(
                [
                    value["pointwise_normal_relative_95_upper_diagnostic"]
                    for value in primary
                ]
            ),
            "primary_point_tolerance_passed": all(
                value["absolute_relative_error"] is not None
                and value["absolute_relative_error"] <= 0.05
                for value in primary
            ),
            "primary_diagnostic_95_tolerance_passed": all(
                value["pointwise_normal_relative_95_upper_diagnostic"] is not None
                and value["pointwise_normal_relative_95_upper_diagnostic"] <= 0.05
                for value in primary
            ),
        },
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "maximum_workers": min(maximum_workers, len(tasks)),
            "worker_seconds": {
                f"L{item['sequence_length']}": item["performance"]
                for item in outputs
            },
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_paired_cluster_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != PAIRED_CLUSTER_SCHEMA_VERSION:
        raise ValueError("Unsupported paired-cluster schema.")
    if payload.get("validation_method") != PAIRED_CLUSTER_METHOD:
        raise ValueError("Unsupported paired-cluster method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")
    seeds = []
    multi_sample_count = payload.get("configuration", {}).get(
        "multi_rare_sample_count_per_pattern"
    )
    for length in ("4", "6"):
        length_payload = payload.get("holdout", {}).get(length)
        if not isinstance(length_payload, Mapping):
            raise ValueError(f"Paired L={length} holdout is missing.")
        strata = length_payload.get("strata")
        expected_count = 2 ** int(length) if multi_sample_count is not None else int(length) + 1
        if not isinstance(strata, Mapping) or len(strata) != expected_count:
            raise ValueError(f"Paired L={length} strata are incomplete.")
        for key, stratum in strata.items():
            if _parse_pattern(key) != tuple(stratum.get("order_pattern", [])):
                raise ValueError("Paired stratum key and pattern differ.")
            seeds.append(stratum.get("seed"))
        expected_conditions = {
            "zero_order2_condition",
            "exactly_one_order2_condition",
        }
        if multi_sample_count is not None:
            expected_conditions.add("two_or_more_order2_condition")
        if set(payload.get("paired_residual_results", {}).get(length, {})) != (
            expected_conditions
        ):
            raise ValueError(f"Paired L={length} result scopes are incomplete.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Paired stratum seeds must be unique.")
    fingerprint = payload.get("validation_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Paired-cluster fingerprint mismatch.")


def write_paired_cluster_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_paired_cluster_payload(payload)
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


def validate_paired_k4_l8_residual(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    compiler: CompilerSettings,
    common_sample_count: int,
    single_rare_sample_count: int,
    seed: int,
    maximum_workers: int = 3,
    cache_maximum_entries: int = 32_768,
    persistent_cache_path: str | Path | None = None,
    checkpoint_directory: str | Path | None = None,
    maximum_circuit_size: int = 200_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure the paired L=8 residual after retaining K1 through K4."""
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    reference_rte_steps = require_integer_count(
        reference_rte_steps, name="reference_rte_steps", minimum=1
    )
    common_sample_count = require_integer_count(
        common_sample_count, name="common_sample_count", minimum=2
    )
    single_rare_sample_count = require_integer_count(
        single_rare_sample_count, name="single_rare_sample_count", minimum=2
    )
    maximum_workers = require_integer_count(
        maximum_workers, name="maximum_workers", minimum=1
    )
    if not math.isfinite(reference_delta_time) or reference_delta_time <= 0.0:
        raise ValueError("reference_delta_time must be finite and positive.")

    sequence_length = 8
    maximum_cluster_length = 4
    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian, partition, identity_policy="extract_identity_phase"
    )
    if preparation.is_deterministic_only:
        raise ValueError("Paired validation requires a non-empty RTE tail.")
    short_step_time = float(reference_delta_time) / reference_rte_steps
    patterns = [(0,) * sequence_length]
    patterns.extend(
        tuple(2 if position == rare else 0 for position in range(sequence_length))
        for rare in range(sequence_length)
    )
    common_task = {
        "hamiltonian": hamiltonian,
        "ld": ld,
        "short_step_time": short_step_time,
        "compiler": compiler,
        "length": sequence_length,
        "master_seed": seed,
        "common_sample_count": common_sample_count,
        "single_rare_sample_count": single_rare_sample_count,
        "multi_rare_sample_count": None,
        "maximum_cluster_length": maximum_cluster_length,
        "cache_maximum_entries": cache_maximum_entries,
        "persistent_cache_path": (
            None if persistent_cache_path is None else str(persistent_cache_path)
        ),
        "maximum_circuit_size": maximum_circuit_size,
    }
    tasks = [{**common_task, "patterns": (pattern,)} for pattern in patterns]
    checkpoint_root = (
        None if checkpoint_directory is None else Path(checkpoint_directory)
    )
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    def checkpoint_identity(pattern: Sequence[int]) -> dict[str, Any]:
        return {
            "method": PAIRED_K4_L8_METHOD,
            "preparation_hash": preparation.preparation_hash,
            "partition_hash": preparation.partition_hash,
            "ld": ld,
            "reference_delta_time": float(reference_delta_time),
            "reference_rte_steps": reference_rte_steps,
            "sequence_length": sequence_length,
            "maximum_cluster_length": maximum_cluster_length,
            "order_pattern": list(pattern),
            "sample_count": (
                common_sample_count if 2 not in pattern else single_rare_sample_count
            ),
            "seed": seed,
            "compiler": _compiler_payload(compiler),
            "maximum_circuit_size": maximum_circuit_size,
        }

    def checkpoint_path(pattern: Sequence[int]) -> Path | None:
        if checkpoint_root is None:
            return None
        return checkpoint_root / f"pattern_{_pattern_key(pattern).replace(',', '-')}.json"

    def load_checkpoint(task: Mapping[str, Any]) -> dict[str, Any] | None:
        pattern = tuple(task["patterns"][0])
        path = checkpoint_path(pattern)
        if path is None or not path.exists():
            return None
        envelope = json.loads(path.read_text(encoding="utf-8"))
        expected = _fingerprint(checkpoint_identity(pattern))
        if envelope.get("schema_version") != PAIRED_K4_L8_SCHEMA_VERSION:
            raise ValueError(f"Unsupported paired K4 checkpoint: {path}")
        if envelope.get("task_fingerprint") != expected:
            raise ValueError(f"Paired K4 checkpoint fingerprint mismatch: {path}")
        result = envelope.get("result")
        if not isinstance(result, Mapping):
            raise ValueError(f"Paired K4 checkpoint result is missing: {path}")
        return dict(result)

    def write_checkpoint(task: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        pattern = tuple(task["patterns"][0])
        path = checkpoint_path(pattern)
        if path is None:
            return
        envelope = {
            "schema_version": PAIRED_K4_L8_SCHEMA_VERSION,
            "task_fingerprint": _fingerprint(checkpoint_identity(pattern)),
            "result": result,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                envelope,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    outputs: list[dict[str, Any]] = []
    pending: list[Mapping[str, Any]] = []
    for task in tasks:
        cached = load_checkpoint(task)
        if cached is None:
            pending.append(task)
        else:
            outputs.append(cached)
    if maximum_workers == 1:
        for task in pending:
            result = _compile_paired_length_strata(task)
            write_checkpoint(task, result)
            outputs.append(result)
    elif pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(maximum_workers, len(pending)), mp_context=context
        ) as executor:
            futures = {
                executor.submit(_compile_paired_length_strata, task): task
                for task in pending
            }
            for future in as_completed(futures):
                result = future.result()
                write_checkpoint(futures[future], result)
                outputs.append(result)

    preparation_hashes = {preparation.preparation_hash}
    preparation_hashes.update(item["preparation_hash"] for item in outputs)
    partition_hashes = {preparation.partition_hash}
    partition_hashes.update(item["partition_hash"] for item in outputs)
    if len(preparation_hashes) != 1 or len(partition_hashes) != 1:
        raise RuntimeError("Paired K4 worker preparations differ.")
    strata: dict[str, Any] = {}
    for item in outputs:
        if item["sequence_length"] != sequence_length:
            raise RuntimeError("Paired K4 worker sequence length differs.")
        overlap = set(strata).intersection(item["strata"])
        if overlap:
            raise RuntimeError("Paired K4 worker strata overlap.")
        strata.update(item["strata"])
    expected_patterns = {_pattern_key(pattern) for pattern in patterns}
    if set(strata) != expected_patterns:
        raise RuntimeError("Paired K4 worker strata are incomplete.")

    zero = tuple(key for key, value in strata.items() if value["rare_order_count"] == 0)
    one = tuple(key for key, value in strata.items() if value["rare_order_count"] == 1)
    results = {
        "zero_order2_condition": _aggregate_paired_condition(strata, zero),
        "exactly_one_order2_condition": _aggregate_paired_condition(strata, one),
    }
    primary = [results[scope]["metrics"]["rz_count"] for scope in results]
    all_metrics = [
        results[scope]["metrics"][metric]
        for scope in results
        for metric in _METRICS
    ]
    payload: dict[str, Any] = {
        "schema_version": PAIRED_K4_L8_SCHEMA_VERSION,
        "validation_method": PAIRED_K4_L8_METHOD,
        "final_cost_evaluation_performed": False,
        "acceptance_policy": {
            "primary_metric": "rz_count",
            "required_conditions": list(results),
            "relative_tolerance": 0.05,
            "normal_approximation_is_diagnostic_only": True,
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
            "finite_taylor_order": 2,
            "sequence_length": sequence_length,
            "maximum_cluster_length": maximum_cluster_length,
            "common_sample_count": common_sample_count,
            "single_rare_sample_count_per_position": single_rare_sample_count,
            "seed": seed,
            "compiler": _compiler_payload(compiler),
            "checkpoint_directory": (
                None if checkpoint_root is None else str(checkpoint_root)
            ),
            "persistent_cache_path": (
                None if persistent_cache_path is None else str(persistent_cache_path)
            ),
        },
        "holdout": {
            "sequence_length": sequence_length,
            "strata": strata,
        },
        "paired_residual_results": results,
        "summary": {
            "primary_maximum_absolute_relative_error": _maximum_present(
                [value["absolute_relative_error"] for value in primary]
            ),
            "primary_maximum_absolute_z_score": _maximum_present(
                [value["absolute_z_score"] for value in primary]
            ),
            "primary_maximum_pointwise_normal_95_upper_diagnostic": _maximum_present(
                [
                    value["pointwise_normal_relative_95_upper_diagnostic"]
                    for value in primary
                ]
            ),
            "all_metrics_maximum_absolute_relative_error": _maximum_present(
                [value["absolute_relative_error"] for value in all_metrics]
            ),
            "primary_point_tolerance_passed": all(
                value["absolute_relative_error"] is not None
                and value["absolute_relative_error"] <= 0.05
                for value in primary
            ),
        },
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "maximum_workers": min(maximum_workers, len(tasks)),
            "reused_checkpoint_count": len(tasks) - len(pending),
            "worker_seconds_sum": float(
                math.fsum(item["performance"]["total_seconds"] for item in outputs)
            ),
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    validate_paired_k4_l8_payload(payload)
    return payload


def validate_paired_k4_l8_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != PAIRED_K4_L8_SCHEMA_VERSION:
        raise ValueError("Unsupported paired K4 L8 schema.")
    if payload.get("validation_method") != PAIRED_K4_L8_METHOD:
        raise ValueError("Unsupported paired K4 L8 method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")
    configuration = payload.get("configuration", {})
    if configuration.get("sequence_length") != 8:
        raise ValueError("Paired K4 validation must use L=8.")
    if configuration.get("maximum_cluster_length") != 4:
        raise ValueError("Paired K4 validation must retain clusters through K4.")
    strata = payload.get("holdout", {}).get("strata")
    if not isinstance(strata, Mapping) or len(strata) != 9:
        raise ValueError("Paired K4 L8 strata are incomplete.")
    expected = {(0,) * 8}
    expected.update(
        tuple(2 if position == rare else 0 for position in range(8))
        for rare in range(8)
    )
    if {_parse_pattern(key) for key in strata} != expected:
        raise ValueError("Paired K4 L8 order patterns are incomplete.")
    if set(payload.get("paired_residual_results", {})) != {
        "zero_order2_condition",
        "exactly_one_order2_condition",
    }:
        raise ValueError("Paired K4 L8 result scopes are incomplete.")
    fingerprint = payload.get("validation_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Paired K4 L8 fingerprint mismatch.")


def write_paired_k4_l8_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_paired_k4_l8_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
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
    temporary.replace(target)

"""Validation of additive compiled-cost models for randomized DF circuits.

This module deliberately keeps the validation target narrow.  It compares the
cost obtained by transpiling a complete circuit with the sum of costs obtained
by transpiling its constituent pieces, and separately checks convergence of the
Monte Carlo estimator against exact enumeration when the trajectory space is
small enough.
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
from .df_partial_s2_cost import (
    CompiledPartialS2CostEstimate,
    estimate_exact_compiled_partial_s2_cost,
    estimate_monte_carlo_compiled_partial_s2_cost,
)
from .rte import (
    CircuitCost,
    CompilerSettings,
    make_rte_config,
    require_integer_count,
    step_taylor_truncation_residual_bound,
)
from .rte_compiled_cost import (
    CompiledSequenceCostEstimate,
    TranspiledCircuitCostCache,
    estimate_compiled_occurrence_cost,
)


RANDOM_CIRCUIT_COST_VALIDATION_SCHEMA_VERSION = (
    "random_circuit_cost_validation_v1"
)
RANDOM_CIRCUIT_COST_VALIDATION_METHOD = (
    "paired_full_circuit_vs_separately_transpiled_additive_parts_v1"
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


def _normalized_sample_counts(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(
        sorted(
            require_integer_count(value, name="monte_carlo_sample_count", minimum=1)
            for value in values
        )
    )
    if not result:
        raise ValueError("monte_carlo_sample_counts must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError("monte_carlo_sample_counts must not contain duplicates.")
    return result


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


def _relative(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _paired_cost_payload(
    *,
    full: CircuitCost,
    additive: CircuitCost,
    difference: CircuitCost,
    full_standard_error: CircuitCost | None,
    additive_standard_error: CircuitCost | None,
    difference_standard_error: CircuitCost | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in _METRICS:
        full_value = float(getattr(full, metric))
        additive_value = float(getattr(additive, metric))
        difference_value = float(getattr(difference, metric))
        additive_error = additive_value - full_value
        metrics[metric] = {
            "full": full_value,
            "additive": additive_value,
            "full_minus_additive": difference_value,
            "additive_minus_full": additive_error,
            "signed_additive_relative_error": _relative(
                additive_error,
                full_value,
            ),
            "absolute_additive_relative_error": (
                None
                if full_value == 0.0
                else float(abs(additive_error / full_value))
            ),
            "full_standard_error": (
                None
                if full_standard_error is None
                else float(getattr(full_standard_error, metric))
            ),
            "additive_standard_error": (
                None
                if additive_standard_error is None
                else float(getattr(additive_standard_error, metric))
            ),
            "difference_standard_error": (
                None
                if difference_standard_error is None
                else float(getattr(difference_standard_error, metric))
            ),
        }
    return {"metrics": metrics}


def _partial_s2_payload(
    estimate: CompiledPartialS2CostEstimate,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload = _paired_cost_payload(
        full=estimate.expected_cost,
        additive=estimate.additive_expected_cost,
        difference=estimate.nonadditive_difference,
        full_standard_error=estimate.standard_error,
        additive_standard_error=estimate.additive_standard_error,
        difference_standard_error=estimate.difference_standard_error,
    )
    payload.update(
        {
            "estimate_kind": estimate.estimate_kind,
            "sample_count": estimate.sample_count,
            "enumerated_event_sequence_count": (
                estimate.enumerated_event_sequence_count
            ),
            "event_sequence_probability_sum": (
                estimate.event_sequence_probability_sum
            ),
            "single_event_space_size": estimate.single_event_space_size,
            "unique_full_step_circuit_count": (
                estimate.unique_full_step_circuit_count
            ),
            "unique_compiled_circuit_count": estimate.unique_compiled_circuit_count,
            "cache": {
                "hits": estimate.transpile_cache_hit_count,
                "misses": estimate.transpile_cache_miss_count,
                "bypasses": estimate.transpile_cache_bypass_count,
                "evictions": estimate.transpile_cache_eviction_count,
            },
            "elapsed_seconds": float(elapsed_seconds),
        }
    )
    return payload


def _occurrence_payload(
    estimate: CompiledSequenceCostEstimate,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload = _paired_cost_payload(
        full=estimate.sequence_expected_cost,
        additive=estimate.additive_expected_cost,
        difference=estimate.nonadditive_difference,
        full_standard_error=estimate.sequence_standard_error,
        additive_standard_error=estimate.additive_standard_error,
        difference_standard_error=estimate.difference_standard_error,
    )
    payload.update(
        {
            "estimate_kind": "monte_carlo_compiled_rte_occurrence_expectation",
            "sample_count": estimate.sample_count,
            "event_count_per_sample": estimate.event_count_per_sample,
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
            "elapsed_seconds": float(elapsed_seconds),
        }
    )
    return payload


def _convergence_payload(
    estimate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in _METRICS:
        current = estimate["metrics"][metric]
        exact = reference["metrics"][metric]
        full_delta = float(current["full"] - exact["full"])
        additive_delta = float(current["additive"] - exact["additive"])
        full_se = current["full_standard_error"]
        additive_se = current["additive_standard_error"]
        metrics[metric] = {
            "full_minus_exact": full_delta,
            "full_signed_relative_error": _relative(full_delta, exact["full"]),
            "full_absolute_z_score": (
                None if full_se in (None, 0.0) else float(abs(full_delta) / full_se)
            ),
            "additive_minus_exact": additive_delta,
            "additive_signed_relative_error": _relative(
                additive_delta,
                exact["additive"],
            ),
            "additive_absolute_z_score": (
                None
                if additive_se in (None, 0.0)
                else float(abs(additive_delta) / additive_se)
            ),
        }
    return {"metrics": metrics}


def _maximum_absolute_relative_error(result: Mapping[str, Any]) -> float | None:
    values = [
        metric["absolute_additive_relative_error"]
        for metric in result["metrics"].values()
        if metric["absolute_additive_relative_error"] is not None
    ]
    return None if not values else float(max(values))


def validate_random_circuit_cost_model(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    delta_time: float,
    rte_steps: int,
    finite_taylor_order: int,
    monte_carlo_sample_counts: Sequence[int],
    compiler: CompilerSettings,
    evaluation_scopes: Sequence[str] = ("partial_s2", "rte_occurrence"),
    seed: int = 20260822,
    coefficient_atol: float = 1e-12,
    maximum_exact_event_sequences: int = 1_000,
    maximum_samples: int = 10_000,
    cache_maximum_entries: int = 4_096,
    maximum_untranspiled_circuit_size: int = 100_000,
    maximum_planned_instruction_applications: int = 1_000_000_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one paired compiled-cost validation point.

    ``partial_s2`` compares the complete step with separately transpiled
    forward-half, RTE occurrence, and reverse-half pieces. ``rte_occurrence``
    compares a complete RTE event sequence with the sum of separately
    transpiled event circuits.  Both comparisons are paired trajectory by
    trajectory before averaging.
    """
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    rte_steps = require_integer_count(rte_steps, name="rte_steps", minimum=1)
    finite_taylor_order = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    if finite_taylor_order % 2:
        raise ValueError("finite_taylor_order must be even.")
    delta_time = float(delta_time)
    if not math.isfinite(delta_time) or delta_time <= 0.0:
        raise ValueError("delta_time must be finite and positive.")
    maximum_exact_event_sequences = require_integer_count(
        maximum_exact_event_sequences,
        name="maximum_exact_event_sequences",
    )
    maximum_samples = require_integer_count(
        maximum_samples,
        name="maximum_samples",
        minimum=1,
    )
    sample_counts = _normalized_sample_counts(monte_carlo_sample_counts)
    if sample_counts[-1] > maximum_samples:
        raise ValueError("A requested sample count exceeds maximum_samples.")
    scopes = tuple(dict.fromkeys(str(scope) for scope in evaluation_scopes))
    allowed_scopes = {"partial_s2", "rte_occurrence"}
    if not scopes or any(scope not in allowed_scopes for scope in scopes):
        raise ValueError(
            "evaluation_scopes must contain partial_s2 and/or rte_occurrence."
        )

    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=coefficient_atol,
    )
    if preparation.is_deterministic_only:
        raise ValueError("Random-circuit validation requires a non-empty RTE tail.")
    tau = preparation.exact_rte_lambda_r * delta_time / rte_steps
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=delta_time,
        rte_steps=rte_steps,
        truncation_tolerance=_explicit_cutoff_tolerance(
            tau,
            finite_taylor_order,
        ),
        finite_taylor_order=finite_taylor_order,
        seed=seed,
    )
    component_count = len(preparation.rte_preparation.symbolic_tail.components)
    single_event_space_size = sum(
        component_count ** (order + 1)
        for order, probability in zip(
            distribution.orders,
            distribution.order_probabilities,
            strict=True,
        )
        if probability > 0.0
    )
    event_sequence_space_size = single_event_space_size**rte_steps
    cache = TranspiledCircuitCostCache(maximum_entries=cache_maximum_entries)

    exact_payload: dict[str, Any] | None = None
    exact_skip_reason: str | None = None
    if (
        "partial_s2" in scopes
        and event_sequence_space_size <= maximum_exact_event_sequences
    ):
        exact_started = time.perf_counter()
        exact = estimate_exact_compiled_partial_s2_cost(
            preparation,
            delta_time,
            config,
            distribution,
            compiler,
            maximum_event_sequences=maximum_exact_event_sequences,
            maximum_untranspiled_circuit_size=(
                maximum_untranspiled_circuit_size
            ),
            maximum_planned_instruction_applications=(
                maximum_planned_instruction_applications
            ),
            cache=cache,
        )
        exact_payload = _partial_s2_payload(
            exact,
            elapsed_seconds=time.perf_counter() - exact_started,
        )
    elif "partial_s2" in scopes:
        exact_skip_reason = (
            f"event_sequence_space_size={event_sequence_space_size} exceeds "
            f"maximum_exact_event_sequences={maximum_exact_event_sequences}"
        )

    partial_monte_carlo: list[dict[str, Any]] = []
    occurrence_monte_carlo: list[dict[str, Any]] = []
    for sample_count in sample_counts:
        if "partial_s2" in scopes:
            point_started = time.perf_counter()
            partial = estimate_monte_carlo_compiled_partial_s2_cost(
                preparation,
                delta_time,
                config,
                distribution,
                compiler,
                sample_count=sample_count,
                seed=seed,
                maximum_samples=maximum_samples,
                maximum_untranspiled_circuit_size=(
                    maximum_untranspiled_circuit_size
                ),
                maximum_planned_instruction_applications=(
                    maximum_planned_instruction_applications
                ),
                cache=cache,
            )
            partial_point = _partial_s2_payload(
                partial,
                elapsed_seconds=time.perf_counter() - point_started,
            )
            if exact_payload is not None:
                partial_point["convergence_against_exact"] = _convergence_payload(
                    partial_point,
                    exact_payload,
                )
            partial_monte_carlo.append(partial_point)

        if "rte_occurrence" in scopes:
            point_started = time.perf_counter()
            occurrence = estimate_compiled_occurrence_cost(
                preparation.rte_preparation,
                config,
                distribution,
                compiler,
                sequence_sample_count=sample_count,
                seed=seed,
                maximum_samples=maximum_samples,
                maximum_rte_steps=max(16, rte_steps),
                maximum_untranspiled_circuit_size=(
                    maximum_untranspiled_circuit_size
                ),
                maximum_planned_instruction_applications=(
                    maximum_planned_instruction_applications
                ),
                cache=cache,
            )
            occurrence_monte_carlo.append(
                _occurrence_payload(
                    occurrence,
                    elapsed_seconds=time.perf_counter() - point_started,
                )
            )

    largest_partial = partial_monte_carlo[-1] if partial_monte_carlo else None
    largest_occurrence = (
        occurrence_monte_carlo[-1] if occurrence_monte_carlo else None
    )
    payload: dict[str, Any] = {
        "schema_version": RANDOM_CIRCUIT_COST_VALIDATION_SCHEMA_VERSION,
        "validation_method": RANDOM_CIRCUIT_COST_VALIDATION_METHOD,
        "final_cost_evaluation_performed": False,
        "acceptance_threshold_decided": False,
        "scope": {
            "purpose": (
                "measure nonadditivity of separately transpiled cost pieces and "
                "check Monte Carlo convergence where exact enumeration is feasible"
            ),
            "partial_s2_comparison": (
                "complete partial-S2 step versus separately transpiled forward half, "
                "RTE occurrence, and reverse half"
            ),
            "rte_occurrence_comparison": (
                "complete RTE event sequence versus the sum of separately "
                "transpiled event circuits"
            ),
            "paired_sampling": True,
            "monte_carlo_seed_prefix_reuse": True,
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
            "delta_time": delta_time,
            "rte_steps": rte_steps,
            "finite_taylor_order": finite_taylor_order,
            "seed": seed,
            "coefficient_atol": coefficient_atol,
            "lambda_r": preparation.exact_rte_lambda_r,
            "dimensionless_short_step_time": tau,
            "component_count": component_count,
            "single_event_space_size": single_event_space_size,
            "event_sequence_space_size": event_sequence_space_size,
            "monte_carlo_sample_counts": list(sample_counts),
            "evaluation_scopes": list(scopes),
            "maximum_exact_event_sequences": maximum_exact_event_sequences,
            "compiler": _compiler_payload(compiler),
        },
        "partial_s2": {
            "status": "executed" if "partial_s2" in scopes else "not_requested",
            "exact": exact_payload,
            "exact_skip_reason": exact_skip_reason,
            "monte_carlo": partial_monte_carlo,
        },
        "rte_occurrence": {
            "status": (
                "executed" if "rte_occurrence" in scopes else "not_requested"
            ),
            "exact": None,
            "exact_status": "not_implemented_for_compiled_sequence_cost",
            "monte_carlo": occurrence_monte_carlo,
        },
        "summary": {
            "exact_partial_s2_available": exact_payload is not None,
            "largest_monte_carlo_sample_count": sample_counts[-1],
            "exact_partial_s2_maximum_absolute_additive_relative_error": (
                None
                if exact_payload is None
                else _maximum_absolute_relative_error(exact_payload)
            ),
            "largest_sample_partial_s2_maximum_absolute_additive_relative_error": (
                None
                if largest_partial is None
                else _maximum_absolute_relative_error(largest_partial)
            ),
            "largest_sample_rte_occurrence_maximum_absolute_additive_relative_error": (
                None
                if largest_occurrence is None
                else _maximum_absolute_relative_error(largest_occurrence)
            ),
            "interpretation_status": (
                "pilot_measurement_only_no_model_acceptance_threshold_yet"
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


def validate_random_circuit_cost_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RANDOM_CIRCUIT_COST_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Unsupported random-circuit cost validation schema.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("validation_fingerprint must be a SHA-256 hex string.")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Random-circuit cost validation fingerprint mismatch.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")


def write_random_circuit_cost_validation(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_random_circuit_cost_payload(payload)
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

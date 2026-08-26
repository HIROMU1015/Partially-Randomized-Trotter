"""Sampled validation of compiled-cost invariance under RTE angle changes."""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .df_hamiltonian import DFHamiltonian
from .df_partial_randomized_pf import df_hamiltonian_hash, split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
from .df_rte_circuit import DFRTEEventSequenceCircuitRequest
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .rte import CompilerSettings, _make_event, make_rte_config, require_integer_count
from .rte_compiled_cost import TranspiledCircuitCostCache
from .rte_connected_cluster_cost_validation import (
    _METRICS,
    _ORDERS,
    _compiler_payload,
    _derived_seed,
    _explicit_cutoff_tolerance,
    _fingerprint,
    _pattern_key,
)


RTE_COST_ANGLE_INVARIANCE_SCHEMA_VERSION = "rte_cost_angle_invariance_v1"
RTE_COST_ANGLE_INVARIANCE_METHOD = "fixed_event_numeric_angle_metric_comparison_v1"


def validate_rte_cost_angle_invariance(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    short_step_times: Sequence[float],
    compiler: CompilerSettings,
    sample_count_per_pattern: int,
    seed: int,
    cluster_lengths: Sequence[int] = (1, 2, 3),
    cache_maximum_entries: int = 32_768,
    persistent_cache_path: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile identical event identities at several numeric RTE angles."""
    started = time.perf_counter()
    sample_count = require_integer_count(
        sample_count_per_pattern,
        name="sample_count_per_pattern",
        minimum=1,
    )
    lengths = tuple(
        sorted(
            {
                require_integer_count(value, name="cluster_length", minimum=1)
                for value in cluster_lengths
            }
        )
    )
    if not lengths or any(length > 3 for length in lengths):
        raise ValueError("Angle-invariance cluster lengths must be in 1, 2, 3.")
    times = tuple(float(value) for value in short_step_times)
    if len(times) < 2 or any(not math.isfinite(value) or value <= 0.0 for value in times):
        raise ValueError("At least two finite positive short-step times are required.")
    if len(set(times)) != len(times):
        raise ValueError("Short-step times must be distinct.")

    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian, partition, identity_policy="extract_identity_phase"
    )
    if preparation.is_deterministic_only:
        raise ValueError("Angle-invariance validation requires a non-empty tail.")
    components = preparation.rte_preparation.symbolic_tail.components
    probability_sum = math.fsum(component.probability for component in components)
    component_probabilities = np.asarray(
        [component.probability / probability_sum for component in components],
        dtype=float,
    )
    distributions = {}
    order_indices = {}
    for short_step_time in times:
        tau = preparation.exact_rte_lambda_r * short_step_time
        _config, distribution = make_rte_config(
            preparation.rte_preparation.symbolic_tail,
            evolution_time=short_step_time,
            rte_steps=1,
            truncation_tolerance=_explicit_cutoff_tolerance(tau),
            finite_taylor_order=2,
            seed=seed,
        )
        if distribution.orders != _ORDERS:
            raise RuntimeError("Angle-invariance validation requires orders (0, 2).")
        distributions[short_step_time] = distribution
        order_indices[short_step_time] = {
            order: distribution.orders.index(order) for order in distribution.orders
        }

    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    cache = TranspiledCircuitCostCache(
        maximum_entries=cache_maximum_entries,
        persistent_path=persistent_cache_path,
    )
    mismatches = []
    comparison_count = 0
    boundary_counts = {"same_basis": 0, "different_basis": 0}
    per_metric_maximum_difference = {metric: 0 for metric in _METRICS}
    for length in lengths:
        for pattern in itertools.product(_ORDERS, repeat=length):
            pattern_key = _pattern_key(pattern)
            rng = np.random.Generator(
                np.random.PCG64(
                    _derived_seed(
                        seed,
                        role="angle_invariance",
                        length=length,
                        pattern=pattern,
                    )
                )
            )
            for sample_index in range(sample_count):
                selected_indices = [
                    tuple(
                        int(value)
                        for value in rng.choice(
                            len(components),
                            size=order + 1,
                            p=component_probabilities,
                        )
                    )
                    for order in pattern
                ]
                baseline = None
                for short_step_time in times:
                    distribution = distributions[short_step_time]
                    events = tuple(
                        _make_event(
                            indices,
                            components,
                            distribution,
                            order_indices[short_step_time][order],
                        )
                        for order, indices in zip(pattern, selected_indices, strict=True)
                    )
                    request = DFRTEEventSequenceCircuitRequest(
                        events=events,
                        component_specs=preparation.rte_preparation.component_specs,
                        controlled=False,
                        ancilla_qubit=None,
                        cancel_adjacent_equal_bases=True,
                        tail_id=preparation.rte_preparation.symbolic_tail.tail_id,
                        tail_hash=preparation.rte_preparation.symbolic_tail.tail_hash,
                        occurrence_rte_steps=length,
                    )
                    built = builder.build_sequence(request)
                    cost, _key, _cached = cache.get_or_transpile(
                        built.circuit,
                        compiler,
                        circuit_fingerprint=built.circuit_fingerprint,
                    )
                    values = {metric: int(getattr(cost, metric)) for metric in _METRICS}
                    if baseline is None:
                        baseline = values
                    else:
                        comparison_count += 1
                        differences = {
                            metric: abs(values[metric] - baseline[metric])
                            for metric in _METRICS
                        }
                        for metric, difference in differences.items():
                            per_metric_maximum_difference[metric] = max(
                                per_metric_maximum_difference[metric], difference
                            )
                        if any(differences.values()):
                            mismatches.append(
                                {
                                    "cluster_length": length,
                                    "order_pattern": list(pattern),
                                    "sample_index": sample_index,
                                    "short_step_time": short_step_time,
                                    "baseline_short_step_time": times[0],
                                    "metric_differences": differences,
                                }
                            )
                if length >= 2:
                    baseline_distribution = distributions[times[0]]
                    baseline_events = tuple(
                        _make_event(
                            indices,
                            components,
                            baseline_distribution,
                            order_indices[times[0]][order],
                        )
                        for order, indices in zip(pattern, selected_indices, strict=True)
                    )
                    for left, right in zip(baseline_events[:-1], baseline_events[1:]):
                        left_endpoint = left.application_sequence[-1]
                        right_endpoint = right.application_sequence[0]
                        left_basis = left_endpoint.basis_hash or left_endpoint.basis_id
                        right_basis = right_endpoint.basis_hash or right_endpoint.basis_id
                        boundary_counts[
                            "same_basis" if left_basis == right_basis else "different_basis"
                        ] += 1

    payload: dict[str, Any] = {
        "schema_version": RTE_COST_ANGLE_INVARIANCE_SCHEMA_VERSION,
        "validation_method": RTE_COST_ANGLE_INVARIANCE_METHOD,
        "final_cost_evaluation_performed": False,
        "hamiltonian": {
            "hash": df_hamiltonian_hash(hamiltonian),
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": hamiltonian.n_blocks,
            "metadata": dict(hamiltonian.metadata),
            "preparation_hash": preparation.preparation_hash,
            "partition_hash": preparation.partition_hash,
        },
        "configuration": {
            "ld": int(ld),
            "short_step_times": list(times),
            "dimensionless_step_times": [
                preparation.exact_rte_lambda_r * value for value in times
            ],
            "finite_taylor_order": 2,
            "cluster_lengths": list(lengths),
            "sample_count_per_pattern": sample_count,
            "seed": int(seed),
            "compiler": _compiler_payload(compiler),
        },
        "coverage": {
            "trajectory_count": sample_count * sum(2**length for length in lengths),
            "metric_comparison_count": comparison_count,
            "boundary_counts": boundary_counts,
        },
        "summary": {
            "sampled_metric_invariance_passed": not mismatches,
            "mismatch_count": len(mismatches),
            "per_metric_maximum_absolute_difference": per_metric_maximum_difference,
            "structural_cache_reuse_enabled": False,
        },
        "mismatches": mismatches,
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "cache_hits": cache.hit_count,
            "cache_misses": cache.miss_count,
            "persistent_cache_hits": cache.persistent_hit_count,
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    validate_rte_cost_angle_invariance_payload(payload)
    return payload


def validate_rte_cost_angle_invariance_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RTE_COST_ANGLE_INVARIANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported RTE cost angle-invariance schema.")
    if payload.get("validation_method") != RTE_COST_ANGLE_INVARIANCE_METHOD:
        raise ValueError("Unsupported RTE cost angle-invariance method.")
    if payload.get("summary", {}).get("structural_cache_reuse_enabled") is not False:
        raise ValueError("Sampled angle validation must not enable structural cache reuse.")
    fingerprint = payload.get("validation_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("RTE cost angle-invariance fingerprint mismatch.")


def write_rte_cost_angle_invariance_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_rte_cost_angle_invariance_payload(payload)
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

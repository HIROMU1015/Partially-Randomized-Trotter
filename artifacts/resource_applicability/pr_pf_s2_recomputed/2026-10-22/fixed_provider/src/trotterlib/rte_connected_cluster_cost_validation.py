"""Operational connected-cluster estimator and held-out compiled-cost validation."""

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
from .df_partial_randomized_pf import df_hamiltonian_hash, split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
from .df_rte_circuit import DFRTEEventSequenceCircuitRequest
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .rte import (
    CompilerSettings,
    _make_event,
    make_rte_config,
    require_integer_count,
    step_taylor_truncation_residual_bound,
)
from .rte_compiled_cost import TranspiledCircuitCostCache


CONNECTED_CLUSTER_SCHEMA_VERSION = "rte_connected_cluster_cost_validation_v1"
CONNECTED_CLUSTER_METHOD = "pilot_allocated_connected_cluster_l4_l6_l8_v1"
CONNECTED_CLUSTER_CALIBRATION_SCHEMA_VERSION = (
    "rte_connected_cluster_cost_calibration_v2"
)
CONNECTED_CLUSTER_CALIBRATION_METHOD = (
    "offline_pilot_allocated_connected_cluster_k1_k3_v2"
)
CONNECTED_CLUSTER_TRANSFER_SCHEMA_VERSION = (
    "rte_connected_cluster_transfer_validation_v2"
)
CONNECTED_CLUSTER_TRANSFER_METHOD = "fixed_calibration_full_holdout_v2"
CONNECTED_CLUSTER_K4_EXTRAPOLATION_SCHEMA_VERSION = (
    "rte_connected_cluster_k4_extrapolation_diagnostic_v1"
)
CONNECTED_CLUSTER_K4_EXTRAPOLATION_METHOD = (
    "fit_l4_residual_as_k4_and_transfer_to_l6_v1"
)
CONNECTED_CLUSTER_K4_CALIBRATION_SCHEMA_VERSION = (
    "rte_connected_cluster_k4_calibration_v1"
)
CONNECTED_CLUSTER_K4_CALIBRATION_METHOD = (
    "independent_paired_four_event_coefficient_holdout_v1"
)
CONNECTED_CLUSTER_SUPPLEMENT_SCHEMA_VERSION = (
    "rte_connected_cluster_holdout_supplement_v1"
)
CONNECTED_CLUSTER_SUPPLEMENT_METHOD = "independent_holdout_precision_supplement_v1"
CONNECTED_CLUSTER_HAMILTONIAN_SNAPSHOT_SCHEMA_VERSION = (
    "connected_cluster_df_hamiltonian_snapshot_v1"
)
CONNECTED_CLUSTER_CHECKPOINT_SCHEMA_VERSION = "connected_cluster_checkpoint_v1"
CONNECTED_CLUSTER_TASK_IMPLEMENTATION_VERSION = (
    "connected_cluster_direct_pair_conditioning_and_chunk_merge_v2"
)
CONNECTED_CLUSTER_CHUNK_TASK_IMPLEMENTATION_VERSION = (
    "connected_cluster_incremental_sample_chunks_v3"
)
_ORDERS = (0, 2)
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
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def write_connected_cluster_hamiltonian_snapshot(
    hamiltonian: DFHamiltonian, path: str | Path
) -> None:
    """Persist the exact floating-point DF representation used by a validation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(
            CONNECTED_CLUSTER_HAMILTONIAN_SNAPSHOT_SCHEMA_VERSION
        ),
        hamiltonian_hash=np.asarray(df_hamiltonian_hash(hamiltonian)),
        constant=np.asarray(hamiltonian.constant, dtype=np.float64),
        one_body=np.asarray(hamiltonian.one_body, dtype=np.complex128),
        lambdas=np.asarray(hamiltonian.lambdas, dtype=np.float64),
        g_matrices=np.asarray(hamiltonian.g_matrices, dtype=np.complex128),
        metadata_json=np.asarray(_canonical_json(hamiltonian.metadata)),
    )
    temporary.replace(target)


def load_connected_cluster_hamiltonian_snapshot(path: str | Path) -> DFHamiltonian:
    """Load and integrity-check an exact connected-cluster DF snapshot."""
    with np.load(Path(path), allow_pickle=False) as data:
        if str(data["schema_version"][()]) != (
            CONNECTED_CLUSTER_HAMILTONIAN_SNAPSHOT_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported connected-cluster Hamiltonian snapshot.")
        matrices = np.asarray(data["g_matrices"], dtype=np.complex128)
        hamiltonian = DFHamiltonian(
            constant=float(data["constant"][()]),
            one_body=np.asarray(data["one_body"], dtype=np.complex128),
            lambdas=np.asarray(data["lambdas"], dtype=np.float64),
            g_matrices=tuple(matrices[index] for index in range(matrices.shape[0])),
            metadata=json.loads(str(data["metadata_json"][()])),
        )
        expected_hash = str(data["hamiltonian_hash"][()])
    if df_hamiltonian_hash(hamiltonian) != expected_hash:
        raise ValueError("Connected-cluster Hamiltonian snapshot hash mismatch.")
    return hamiltonian


def _pattern_key(pattern: Sequence[int]) -> str:
    return ",".join(str(value) for value in pattern)


def _parse_pattern(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _parameter_key(pattern: Sequence[int]) -> str:
    return f"k{len(pattern)}:{_pattern_key(pattern)}"


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


def _derived_chunk_seed(
    master_seed: int,
    *,
    role: str,
    length: int,
    pattern: Sequence[int],
    chunk_index: int,
) -> int:
    encoded = _canonical_json(
        {
            "master_seed": master_seed,
            "role": role,
            "length": length,
            "pattern": list(pattern),
            "chunk_index": chunk_index,
            "stream_scheme": CONNECTED_CLUSTER_CHUNK_TASK_IMPLEMENTATION_VERSION,
        }
    ).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _sample_chunks(
    sample_count: int, chunk_size: int
) -> tuple[tuple[int, int, int], ...]:
    total = require_integer_count(sample_count, name="sample_count", minimum=2)
    size = require_integer_count(chunk_size, name="sample_chunk_size", minimum=2)
    return tuple(
        (index, start, min(total, start + size))
        for index, start in enumerate(range(0, total, size))
    )


def _explicit_cutoff_tolerance(tau: float) -> float:
    residual = step_taylor_truncation_residual_bound(tau, 2)
    if not math.isfinite(residual):
        raise ValueError("The finite-Taylor residual overflowed.")
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


def _sample_statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 1:
        raise ValueError("At least one sample is required.")
    variance = 0.0 if array.size == 1 else float(np.var(array, ddof=1))
    return {
        "mean": float(np.mean(array)),
        "unbiased_sample_variance": variance,
        "standard_error": float(math.sqrt(variance / array.size)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _weighted_exact_statistics(
    values: Sequence[float], weights: Sequence[float]
) -> dict[str, float]:
    total = math.fsum(weights)
    normalized = [weight / total for weight in weights]
    mean = math.fsum(
        weight * value for weight, value in zip(normalized, values, strict=True)
    )
    variance = math.fsum(
        weight * (value - mean) ** 2
        for weight, value in zip(normalized, values, strict=True)
    )
    return {
        "mean": float(mean),
        "unbiased_sample_variance": float(variance),
        "standard_error": 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _merge_sample_statistics(
    first: Mapping[str, float],
    first_count: int,
    second: Mapping[str, float],
    second_count: int,
) -> dict[str, float]:
    """Merge two independent sample summaries without retaining raw samples."""
    if first_count < 1 or second_count < 1:
        raise ValueError("Merged sample summaries require non-empty inputs.")
    total = first_count + second_count
    first_mean = float(first["mean"])
    second_mean = float(second["mean"])
    mean = (first_count * first_mean + second_count * second_mean) / total
    corrected_sum = (
        (first_count - 1) * float(first["unbiased_sample_variance"])
        + (second_count - 1) * float(second["unbiased_sample_variance"])
        + first_count * (first_mean - mean) ** 2
        + second_count * (second_mean - mean) ** 2
    )
    variance = corrected_sum / (total - 1)
    return {
        "mean": float(mean),
        "unbiased_sample_variance": float(variance),
        "standard_error": float(math.sqrt(variance / total)),
        "minimum": float(min(first["minimum"], second["minimum"])),
        "maximum": float(max(first["maximum"], second["maximum"])),
    }


def _cost_vector(cost: Any) -> np.ndarray:
    return np.asarray([float(getattr(cost, metric)) for metric in _METRICS])


def _chunk_metadata(
    task: Mapping[str, Any], *, base_seed: int | None, rng_seed: int | None
) -> dict[str, Any]:
    if "chunk_index" not in task:
        return {}
    return {
        "sampling_stream_scheme": CONNECTED_CLUSTER_CHUNK_TASK_IMPLEMENTATION_VERSION,
        "base_seed": base_seed,
        "chunk_seed": rng_seed,
        "chunk_index": int(task["chunk_index"]),
        "sample_start": int(task["sample_start"]),
        "sample_stop": int(task["sample_stop"]),
    }


def _prepare_worker(task: Mapping[str, Any]):
    hamiltonian = task["hamiltonian"]
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
        evolution_time=short_step_time,
        rte_steps=1,
        truncation_tolerance=_explicit_cutoff_tolerance(tau),
        finite_taylor_order=2,
        seed=int(task["master_seed"]),
    )
    if distribution.orders != _ORDERS:
        raise RuntimeError("Connected K=2 validation requires orders (0, 2).")
    components = preparation.rte_preparation.symbolic_tail.components
    probability_sum = math.fsum(item.probability for item in components)
    component_probabilities = np.asarray(
        [item.probability / probability_sum for item in components],
        dtype=float,
    )
    order_indices = {
        order: distribution.orders.index(order) for order in distribution.orders
    }
    return (
        preparation,
        distribution,
        components,
        component_probabilities,
        order_indices,
    )


def _compile_calibration_length(task: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    length = int(task["cluster_length"])
    role = str(task["role"])
    compiler = task["compiler"]
    (
        preparation,
        distribution,
        components,
        component_probabilities,
        order_indices,
    ) = _prepare_worker(task)
    cache = TranspiledCircuitCostCache(
        maximum_entries=int(task["cache_maximum_entries"]),
        persistent_path=task.get("persistent_cache_path"),
    )
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    basis_probabilities: dict[str, float] = {}
    for component in components:
        basis_key = component.basis_hash or component.basis_id
        basis_probabilities[basis_key] = (
            basis_probabilities.get(basis_key, 0.0) + component.probability
        )
    component_indices_by_basis = {key: [] for key in basis_probabilities}
    for component_index, component in enumerate(components):
        basis_key = component.basis_hash or component.basis_id
        component_indices_by_basis[basis_key].append(component_index)
    conditional_component_probabilities = {
        key: np.asarray(
            [component_probabilities[index] for index in indices], dtype=float
        )
        / math.fsum(component_probabilities[index] for index in indices)
        for key, indices in component_indices_by_basis.items()
    }
    same_boundary_probability = math.fsum(
        probability**2 for probability in basis_probabilities.values()
    )
    different_boundary_probability = 1.0 - same_boundary_probability

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
            raise ValueError("A connected-cluster circuit exceeded the size limit.")
        cost, cache_key, _cached = cache.get_or_transpile(
            built.circuit,
            compiler,
            circuit_fingerprint=built.circuit_fingerprint,
        )
        return _cost_vector(cost), cache_key

    def connected_value(events: Sequence[Any]) -> tuple[np.ndarray, set[str]]:
        keys: set[str] = set()
        if length == 1:
            value, key = compile_window(events)
            keys.add(key)
            return value, keys
        if length == 2:
            full, key = compile_window(events)
            keys.add(key)
            first, key = compile_window(events[:1])
            keys.add(key)
            second, key = compile_window(events[1:])
            keys.add(key)
            return full - first - second, keys
        full, key = compile_window(events)
        keys.add(key)
        left, key = compile_window(events[:-1])
        keys.add(key)
        right, key = compile_window(events[1:])
        keys.add(key)
        middle, key = compile_window(events[1:-1])
        keys.add(key)
        return full - left - right + middle, keys

    strata: dict[str, Any] = {}
    sample_counts = task["sample_counts"]
    requested_patterns = task.get("patterns")
    patterns = (
        tuple(itertools.product(_ORDERS, repeat=length))
        if requested_patterns is None
        else tuple(tuple(int(value) for value in pattern) for pattern in requested_patterns)
    )
    for pattern in patterns:
        if len(pattern) != length or any(value not in _ORDERS for value in pattern):
            raise ValueError("Connected-cluster calibration pattern is invalid.")
        key = _pattern_key(pattern)
        sample_count = int(sample_counts[key])
        base_seed = _derived_seed(
            int(task["master_seed"]),
            role=role,
            length=length,
            pattern=pattern,
        )
        seed = int(task.get("chunk_seed", base_seed))
        rng = np.random.Generator(np.random.PCG64(seed))
        samples = [[] for _ in _METRICS]
        digest = hashlib.sha256()
        unique_circuits: set[str] = set()
        stratum_started = time.perf_counter()

        if (
            role == "production"
            and length == 1
            and pattern == (0,)
            and bool(task.get("exact_order0_single", False))
        ):
            exact_samples = [[] for _ in _METRICS]
            exact_weights = []
            for component_index, component in enumerate(components):
                event = _make_event(
                    [component_index],
                    components,
                    distribution,
                    order_indices[0],
                )
                value, keys = connected_value([event])
                unique_circuits.update(keys)
                exact_weights.append(component.probability)
                for index in range(len(_METRICS)):
                    exact_samples[index].append(value[index])
            strata[key] = {
                "parameter_key": _parameter_key(pattern),
                "order_pattern": list(pattern),
                "rare_order_count": 0,
                "estimate_kind": "exact_conditional_order0_enumeration",
                "sample_count": None,
                "enumerated_event_count": len(components),
                "seed": None,
                "metric_statistics": {
                    metric: _weighted_exact_statistics(
                        exact_samples[index], exact_weights
                    )
                    for index, metric in enumerate(_METRICS)
                },
                "event_stream_rolling_digest": None,
                "unique_compiled_circuit_count": len(unique_circuits),
                "elapsed_seconds": float(time.perf_counter() - stratum_started),
                **_chunk_metadata(task, base_seed=None, rng_seed=None),
            }
            continue

        if length == 2 and bool(task.get("pair_boundary_stratified", False)):
            class_samples = {
                "same_basis": [[] for _ in _METRICS],
                "different_basis": [[] for _ in _METRICS],
            }
            accepted = {"same_basis": 0, "different_basis": 0}
            probabilities = {
                "same_basis": same_boundary_probability,
                "different_basis": different_boundary_probability,
            }
            active_classes = tuple(
                key for key, probability in probabilities.items() if probability > 1e-15
            )
            if bool(task.get("direct_pair_conditioning", False)):
                basis_keys = tuple(sorted(basis_probabilities))
                same_weights = np.asarray(
                    [basis_probabilities[key] ** 2 for key in basis_keys],
                    dtype=float,
                )
                if same_weights.sum() > 0.0:
                    same_weights /= same_weights.sum()
                different_pairs = tuple(
                    (left, right)
                    for left in basis_keys
                    for right in basis_keys
                    if left != right
                )
                different_weights = np.asarray(
                    [
                        basis_probabilities[left] * basis_probabilities[right]
                        for left, right in different_pairs
                    ],
                    dtype=float,
                )
                if different_weights.size and different_weights.sum() > 0.0:
                    different_weights /= different_weights.sum()

                def constrained_event(order: int, basis: str, endpoint: str):
                    count = order + 1
                    indices = list(
                        rng.choice(
                            len(components),
                            size=count,
                            p=component_probabilities,
                        )
                    )
                    if endpoint == "last":
                        constrained_position = 0
                    elif endpoint == "first":
                        constrained_position = 0 if order == 0 else 1
                    else:
                        raise ValueError("Unknown conditioned event endpoint.")
                    local_indices = component_indices_by_basis[basis]
                    indices[constrained_position] = int(
                        rng.choice(
                            local_indices,
                            p=conditional_component_probabilities[basis],
                        )
                    )
                    return _make_event(
                        indices,
                        components,
                        distribution,
                        order_indices[order],
                    )

                for boundary_class in active_classes:
                    for _ in range(sample_count):
                        if boundary_class == "same_basis":
                            basis = basis_keys[int(rng.choice(len(basis_keys), p=same_weights))]
                            left_basis, right_basis = basis, basis
                        else:
                            pair_index = int(
                                rng.choice(len(different_pairs), p=different_weights)
                            )
                            left_basis, right_basis = different_pairs[pair_index]
                        events = [
                            constrained_event(pattern[0], left_basis, "last"),
                            constrained_event(pattern[1], right_basis, "first"),
                        ]
                        for event in events:
                            encoded = json.dumps(
                                event.to_dict(),
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                            digest.update(len(encoded).to_bytes(8, "big"))
                            digest.update(encoded)
                        value, keys = connected_value(events)
                        unique_circuits.update(keys)
                        accepted[boundary_class] += 1
                        for index in range(len(_METRICS)):
                            class_samples[boundary_class][index].append(value[index])
                attempted = sum(accepted.values())
            else:
                attempted = 0
                while any(accepted[key] < sample_count for key in active_classes):
                    attempted += 1
                    events = []
                    for order in pattern:
                        indices = rng.choice(
                            len(components),
                            size=order + 1,
                            p=component_probabilities,
                        )
                        events.append(
                            _make_event(
                                indices,
                                components,
                                distribution,
                                order_indices[order],
                            )
                        )
                    left_endpoint = events[0].application_sequence[-1]
                    right_endpoint = events[1].application_sequence[0]
                    left_basis = left_endpoint.basis_hash or left_endpoint.basis_id
                    right_basis = right_endpoint.basis_hash or right_endpoint.basis_id
                    boundary_class = (
                        "same_basis"
                        if left_basis == right_basis
                        else "different_basis"
                    )
                    if (
                        boundary_class not in active_classes
                        or accepted[boundary_class] >= sample_count
                    ):
                        continue
                    for event in events:
                        encoded = json.dumps(
                            event.to_dict(),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                        digest.update(len(encoded).to_bytes(8, "big"))
                        digest.update(encoded)
                    value, keys = connected_value(events)
                    unique_circuits.update(keys)
                    accepted[boundary_class] += 1
                    for index in range(len(_METRICS)):
                        class_samples[boundary_class][index].append(value[index])
            metric_statistics = {}
            class_payload = {}
            for boundary_class, values_by_metric in class_samples.items():
                class_payload[boundary_class] = {
                    "analytic_probability": probabilities[boundary_class],
                    "sample_count": accepted[boundary_class],
                    "metric_statistics": (
                        {
                            metric: _sample_statistics(values_by_metric[index])
                            for index, metric in enumerate(_METRICS)
                        }
                        if boundary_class in active_classes
                        else None
                    ),
                }
            for index, metric in enumerate(_METRICS):
                active_stats = [
                    (
                        probabilities[boundary_class],
                        class_payload[boundary_class]["metric_statistics"][metric],
                    )
                    for boundary_class in active_classes
                ]
                mean = math.fsum(
                    probability * statistics["mean"]
                    for probability, statistics in active_stats
                )
                effective_variance = math.fsum(
                    probability**2 * statistics["unbiased_sample_variance"]
                    for probability, statistics in active_stats
                )
                metric_statistics[metric] = {
                    "mean": float(mean),
                    "unbiased_sample_variance": float(effective_variance),
                    "standard_error": float(
                        math.sqrt(effective_variance / sample_count)
                    ),
                    "minimum": float(
                        min(statistics["minimum"] for _, statistics in active_stats)
                    ),
                    "maximum": float(
                        max(statistics["maximum"] for _, statistics in active_stats)
                    ),
                }
            strata[key] = {
                "parameter_key": _parameter_key(pattern),
                "order_pattern": list(pattern),
                "rare_order_count": sum(order == 2 for order in pattern),
                "estimate_kind": "analytic_same_different_basis_stratification",
                "sample_count_per_boundary_class": sample_count,
                "accepted_sample_count": sum(accepted.values()),
                "attempted_trajectory_count": attempted,
                "boundary_sampling_method": (
                    "direct_analytic_conditioning"
                    if bool(task.get("direct_pair_conditioning", False))
                    else "iid_rejection"
                ),
                "seed": base_seed,
                "boundary_classes": class_payload,
                "metric_statistics": metric_statistics,
                "event_stream_rolling_digest": digest.hexdigest(),
                "unique_compiled_circuit_count": len(unique_circuits),
                "elapsed_seconds": float(time.perf_counter() - stratum_started),
                **_chunk_metadata(task, base_seed=base_seed, rng_seed=seed),
            }
            continue

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
            value, keys = connected_value(events)
            unique_circuits.update(keys)
            for index in range(len(_METRICS)):
                samples[index].append(value[index])
        strata[key] = {
            "parameter_key": _parameter_key(pattern),
            "order_pattern": list(pattern),
            "rare_order_count": sum(order == 2 for order in pattern),
            "estimate_kind": "conditional_monte_carlo",
            "sample_count": sample_count,
            "seed": base_seed,
            "metric_statistics": {
                metric: _sample_statistics(samples[index])
                for index, metric in enumerate(_METRICS)
            },
            "event_stream_rolling_digest": digest.hexdigest(),
            "unique_compiled_circuit_count": len(unique_circuits),
            "elapsed_seconds": float(time.perf_counter() - stratum_started),
            **_chunk_metadata(task, base_seed=base_seed, rng_seed=seed),
        }
    return {
        "role": role,
        "cluster_length": length,
        "hamiltonian_hash": df_hamiltonian_hash(task["hamiltonian"]),
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


def _compile_full_holdout_length(task: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    length = int(task["sequence_length"])
    compiler = task["compiler"]
    (
        preparation,
        distribution,
        components,
        component_probabilities,
        order_indices,
    ) = _prepare_worker(task)
    cache = TranspiledCircuitCostCache(
        maximum_entries=int(task["cache_maximum_entries"]),
        persistent_path=task.get("persistent_cache_path"),
    )
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    requested_patterns = task.get("patterns")
    if requested_patterns is None:
        patterns = [(0,) * length]
        patterns.extend(
            tuple(2 if index == rare else 0 for index in range(length))
            for rare in range(length)
        )
    else:
        patterns = [
            tuple(int(value) for value in pattern) for pattern in requested_patterns
        ]
    strata: dict[str, Any] = {}
    for pattern in patterns:
        if len(pattern) != length or any(value not in _ORDERS for value in pattern):
            raise ValueError("Connected-cluster holdout pattern is invalid.")
        sample_count = (
            int(task["zero_sample_count"])
            if 2 not in pattern
            else int(task["single_rare_sample_count"])
        )
        base_seed = _derived_seed(
            int(task["master_seed"]),
            role="operational_holdout",
            length=length,
            pattern=pattern,
        )
        seed = int(task.get("chunk_seed", base_seed))
        rng = np.random.Generator(np.random.PCG64(seed))
        samples = [[] for _ in _METRICS]
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
                raise ValueError("A connected-cluster holdout exceeded the size limit.")
            cost, cache_key, _cached = cache.get_or_transpile(
                built.circuit,
                compiler,
                circuit_fingerprint=built.circuit_fingerprint,
            )
            unique_circuits.add(cache_key)
            values = _cost_vector(cost)
            for index in range(len(_METRICS)):
                samples[index].append(values[index])
        strata[_pattern_key(pattern)] = {
            "order_pattern": list(pattern),
            "rare_order_count": sum(order == 2 for order in pattern),
            "sample_count": sample_count,
            "seed": base_seed,
            "metric_statistics": {
                metric: _sample_statistics(samples[index])
                for index, metric in enumerate(_METRICS)
            },
            "event_stream_rolling_digest": digest.hexdigest(),
            "unique_compiled_circuit_count": len(unique_circuits),
            "elapsed_seconds": float(time.perf_counter() - stratum_started),
            **_chunk_metadata(task, base_seed=base_seed, rng_seed=seed),
        }
    return {
        "role": "holdout",
        "sequence_length": length,
        "hamiltonian_hash": df_hamiltonian_hash(task["hamiltonian"]),
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


def _merge_statistics_sequence(
    chunks: Sequence[tuple[int, Mapping[str, float]]],
) -> tuple[int, dict[str, float]]:
    if not chunks:
        raise ValueError("Cannot merge an empty statistics sequence.")
    count, statistics = chunks[0][0], dict(chunks[0][1])
    for next_count, next_statistics in chunks[1:]:
        statistics = _merge_sample_statistics(
            statistics, count, next_statistics, next_count
        )
        count += next_count
    return count, statistics


def _combined_chunk_digest(chunks: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        encoded = _canonical_json(
            {
                "chunk_index": chunk.get("chunk_index"),
                "sample_start": chunk.get("sample_start"),
                "sample_stop": chunk.get("sample_stop"),
                "chunk_seed": chunk.get("chunk_seed"),
                "event_stream_rolling_digest": chunk.get(
                    "event_stream_rolling_digest"
                ),
            }
        ).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _merge_stratum_chunks(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("Cannot merge an empty connected-cluster stratum.")
    if len(chunks) == 1 and "chunk_index" not in chunks[0]:
        return dict(chunks[0])
    ordered = sorted(chunks, key=lambda item: int(item.get("chunk_index", 0)))
    chunk_indices = [int(item["chunk_index"]) for item in ordered]
    if chunk_indices != list(range(len(chunk_indices))):
        raise RuntimeError("Connected-cluster sample chunks are not contiguous.")
    base = ordered[0]
    for item in ordered[1:]:
        for field in (
            "parameter_key",
            "order_pattern",
            "rare_order_count",
            "estimate_kind",
            "seed",
        ):
            if item.get(field) != base.get(field):
                raise RuntimeError(f"Connected-cluster chunk {field} differs.")
    result = {
        key: value
        for key, value in base.items()
        if key
        not in {
            "metric_statistics",
            "boundary_classes",
            "sample_count",
            "sample_count_per_boundary_class",
            "accepted_sample_count",
            "attempted_trajectory_count",
            "event_stream_rolling_digest",
            "unique_compiled_circuit_count",
            "elapsed_seconds",
            "chunk_seed",
            "chunk_index",
            "sample_start",
            "sample_stop",
        }
    }
    if "boundary_classes" in base:
        classes: dict[str, Any] = {}
        for boundary_class in base["boundary_classes"]:
            class_chunks = [
                item["boundary_classes"][boundary_class] for item in ordered
            ]
            probability = float(class_chunks[0]["analytic_probability"])
            if any(
                float(item["analytic_probability"]) != probability
                for item in class_chunks[1:]
            ):
                raise RuntimeError(
                    "Connected-cluster chunk boundary probabilities differ."
                )
            active = class_chunks[0]["metric_statistics"] is not None
            if any(
                (item["metric_statistics"] is not None) != active
                for item in class_chunks
            ):
                raise RuntimeError("Connected-cluster chunk boundary activity differs.")
            class_count = sum(int(item["sample_count"]) for item in class_chunks)
            classes[boundary_class] = {
                "analytic_probability": probability,
                "sample_count": class_count,
                "metric_statistics": (
                    {
                        metric: _merge_statistics_sequence(
                            [
                                (
                                    int(item["sample_count"]),
                                    item["metric_statistics"][metric],
                                )
                                for item in class_chunks
                            ]
                        )[1]
                        for metric in _METRICS
                    }
                    if active
                    else None
                ),
            }
        active_classes = tuple(
            key
            for key, value in classes.items()
            if value["metric_statistics"] is not None
        )
        per_class_counts = {classes[key]["sample_count"] for key in active_classes}
        if len(per_class_counts) != 1:
            raise RuntimeError("Connected-cluster chunk boundary counts differ.")
        sample_count = next(iter(per_class_counts))
        metric_statistics = {}
        for metric in _METRICS:
            active_stats = [
                (
                    classes[key]["analytic_probability"],
                    classes[key]["metric_statistics"][metric],
                )
                for key in active_classes
            ]
            mean = math.fsum(
                probability * statistics["mean"]
                for probability, statistics in active_stats
            )
            effective_variance = math.fsum(
                probability**2 * statistics["unbiased_sample_variance"]
                for probability, statistics in active_stats
            )
            metric_statistics[metric] = {
                "mean": float(mean),
                "unbiased_sample_variance": float(effective_variance),
                "standard_error": float(math.sqrt(effective_variance / sample_count)),
                "minimum": float(
                    min(statistics["minimum"] for _, statistics in active_stats)
                ),
                "maximum": float(
                    max(statistics["maximum"] for _, statistics in active_stats)
                ),
            }
        result.update(
            {
                "sample_count_per_boundary_class": sample_count,
                "accepted_sample_count": sum(
                    int(item["accepted_sample_count"]) for item in ordered
                ),
                "attempted_trajectory_count": sum(
                    int(item["attempted_trajectory_count"]) for item in ordered
                ),
                "boundary_classes": classes,
                "metric_statistics": metric_statistics,
            }
        )
    else:
        counts = [int(item["sample_count"]) for item in ordered]
        result["sample_count"] = sum(counts)
        result["metric_statistics"] = {
            metric: _merge_statistics_sequence(
                [
                    (count, item["metric_statistics"][metric])
                    for count, item in zip(counts, ordered, strict=True)
                ]
            )[1]
            for metric in _METRICS
        }
    result.update(
        {
            "sampling_stream_scheme": (
                CONNECTED_CLUSTER_CHUNK_TASK_IMPLEMENTATION_VERSION
            ),
            "chunk_count": len(ordered),
            "chunk_seeds": [int(item["chunk_seed"]) for item in ordered],
            "sample_ranges": [
                [int(item["sample_start"]), int(item["sample_stop"])]
                for item in ordered
            ],
            "event_stream_rolling_digest": _combined_chunk_digest(ordered),
            "unique_compiled_circuit_count": sum(
                int(item["unique_compiled_circuit_count"]) for item in ordered
            ),
            "unique_compiled_circuit_count_semantics": "sum_over_chunks_upper_bound",
            "elapsed_seconds": float(
                math.fsum(float(item["elapsed_seconds"]) for item in ordered)
            ),
            "maximum_chunk_seconds": float(
                max(float(item["elapsed_seconds"]) for item in ordered)
            ),
        }
    )
    return result


def _sequence_form(pattern: Sequence[int]) -> dict[str, float]:
    form: dict[str, float] = {}
    for window_length in (1, 2, 3):
        for start in range(len(pattern) - window_length + 1):
            key = _parameter_key(pattern[start : start + window_length])
            form[key] = form.get(key, 0.0) + 1.0
    return form


def _average_forms(forms: Sequence[Mapping[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    weight = 1.0 / len(forms)
    for form in forms:
        for key, coefficient in form.items():
            result[key] = result.get(key, 0.0) + weight * coefficient
    return result


def _required_forms(
    order_probabilities: Mapping[int, float],
    lengths: Sequence[int] = (4, 6, 8),
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for raw_length in lengths:
        length = require_integer_count(raw_length, name="event_count", minimum=1)
        zero_pattern = (0,) * length
        result[f"L{length}:zero"] = _sequence_form(zero_pattern)
        one_forms = [
            _sequence_form(
                tuple(2 if index == rare else 0 for index in range(length))
            )
            for rare in range(length)
        ]
        result[f"L{length}:one"] = _average_forms(one_forms)
        full: dict[str, float] = {}
        for window_length in (1, 2, 3):
            multiplicity = length - window_length + 1
            for pattern in itertools.product(_ORDERS, repeat=window_length):
                probability = math.prod(order_probabilities[value] for value in pattern)
                full[_parameter_key(pattern)] = multiplicity * probability
        result[f"L{length}:full"] = full
    return result


def _iid_sequence_form(
    event_count: int,
    order_probabilities: Mapping[int, float],
) -> dict[str, float]:
    """Return the expected local-window multiplicities for one IID event sequence."""
    length = require_integer_count(event_count, name="event_count", minimum=1)
    form: dict[str, float] = {}
    for window_length in range(1, min(3, length) + 1):
        multiplicity = length - window_length + 1
        for pattern in itertools.product(_ORDERS, repeat=window_length):
            probability = math.prod(order_probabilities[value] for value in pattern)
            form[_parameter_key(pattern)] = multiplicity * probability
    return form


def _statistics_lookup(
    outputs: Mapping[int, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        stratum["parameter_key"]: stratum
        for output in outputs.values()
        for stratum in output["strata"].values()
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


def _maximum_form_relative_standard_error(
    outputs: Mapping[int, Mapping[str, Any]],
    forms: Mapping[str, Mapping[str, float]],
    *,
    metric: str = "rz_count",
) -> float:
    lookup = _statistics_lookup(outputs)
    values = []
    for form in forms.values():
        mean, standard_error = _evaluate_form(form, lookup, metric)
        values.append(math.inf if mean == 0.0 else standard_error / abs(mean))
    return float(max(values, default=0.0))


def _pilot_cost_per_allocation_sample(stratum: Mapping[str, Any]) -> float:
    """Return a cache- and scheduler-independent relative work estimate."""
    applications = [int(order) + 1 for order in stratum["order_pattern"]]
    if len(applications) == 1:
        relative_work = applications[0]
    elif len(applications) == 2:
        # G(e1,e2), G(e1), G(e2).
        relative_work = 2 * math.fsum(applications)
    elif len(applications) == 3:
        # G(123), G(12), G(23), G(2).
        relative_work = (
            2 * applications[0]
            + 4 * applications[1]
            + 2 * applications[2]
        )
    else:
        raise ValueError("Unsupported connected-cluster allocation length.")
    if "boundary_classes" in stratum:
        # One allocation unit requests one sample in each active boundary class.
        active_classes = sum(
            value["metric_statistics"] is not None
            for value in stratum["boundary_classes"].values()
        )
        relative_work *= active_classes
    return float(relative_work)


def _allocate_production_samples(
    pilot: Mapping[int, Mapping[str, Any]],
    forms: Mapping[str, Mapping[str, float]],
    *,
    relative_standard_error_target: float,
    minimum_samples: int,
    maximum_samples: int,
    safety_factor: float,
    exact_parameter_keys: Sequence[str] = (),
    cost_aware: bool = False,
) -> tuple[dict[int, dict[str, int]], dict[str, Any]]:
    lookup = _statistics_lookup(pilot)
    exact_keys = set(exact_parameter_keys)
    required = {key: minimum_samples for key in lookup}
    raw_required = {key: minimum_samples for key in lookup}
    scope_plans: dict[str, Any] = {}
    for scope, form in forms.items():
        mean, _se = _evaluate_form(form, lookup, "rz_count")
        absolute_target = max(1.0, relative_standard_error_target * abs(mean))
        active = [key for key, coefficient in form.items() if coefficient != 0.0]
        sensitivities = {
            key: (
                0.0
                if key in exact_keys
                else abs(form[key])
                * math.sqrt(
                    lookup[key]["metric_statistics"]["rz_count"][
                        "unbiased_sample_variance"
                    ]
                )
            )
            for key in active
        }
        sample_costs = {
            key: _pilot_cost_per_allocation_sample(lookup[key]) for key in active
        }
        total_sensitivity = math.fsum(sensitivities.values())
        cost_weighted_total_sensitivity = math.fsum(
            sensitivities[key] * math.sqrt(sample_costs[key]) for key in active
        )
        scope_requirements: dict[str, int] = {}
        raw_scope_requirements: dict[str, int] = {}
        for key in active:
            if key in exact_keys:
                scope_requirements[key] = minimum_samples
                raw_scope_requirements[key] = minimum_samples
                continue
            coefficient = form[key]
            variance = lookup[key]["metric_statistics"]["rz_count"][
                "unbiased_sample_variance"
            ]
            if cost_aware:
                raw_estimate = (
                    safety_factor
                    * cost_weighted_total_sensitivity
                    * abs(coefficient)
                    * math.sqrt(variance)
                    / (math.sqrt(sample_costs[key]) * absolute_target**2)
                )
            else:
                raw_estimate = (
                    safety_factor
                    * total_sensitivity
                    * abs(coefficient)
                    * math.sqrt(variance)
                    / absolute_target**2
                )
            raw_count = max(minimum_samples, math.ceil(raw_estimate))
            count = min(maximum_samples, raw_count)
            required[key] = max(required[key], count)
            raw_required[key] = max(raw_required[key], raw_count)
            scope_requirements[key] = count
            raw_scope_requirements[key] = raw_count
        scope_plans[scope] = {
            "pilot_prediction": mean,
            "absolute_standard_error_target": absolute_target,
            "total_sensitivity": total_sensitivity,
            "cost_weighted_total_sensitivity": cost_weighted_total_sensitivity,
            "deterministic_relative_work_per_allocation_sample": sample_costs,
            "per_parameter_required_samples": scope_requirements,
            "unclamped_per_parameter_required_samples": raw_scope_requirements,
        }
    by_length: dict[int, dict[str, int]] = {1: {}, 2: {}, 3: {}}
    capped: list[str] = []
    for key, count in required.items():
        prefix, pattern_key = key.split(":", maxsplit=1)
        length = int(prefix[1:])
        by_length[length][pattern_key] = count
        if raw_required[key] > maximum_samples:
            capped.append(key)
    return by_length, {
        "allocation_rule": (
            "neyman_deterministic_relative_work_cost"
            if cost_aware
            else "neyman_equal_per_sample_cost_across_parameters"
        ),
        "pilot_metric": "rz_count",
        "relative_standard_error_target": relative_standard_error_target,
        "minimum_samples": minimum_samples,
        "maximum_samples": maximum_samples,
        "safety_factor": safety_factor,
        "exact_parameter_keys": sorted(exact_keys),
        "capped_parameter_keys": sorted(capped),
        "unclamped_production_sample_counts": {
            str(length): {
                pattern: raw_required[f"k{length}:{pattern}"]
                for pattern in counts
            }
            for length, counts in by_length.items()
        },
        "scope_plans": scope_plans,
        "production_sample_counts": {
            str(length): counts for length, counts in by_length.items()
        },
    }


def _aggregate_holdout(
    strata: Mapping[str, Any],
    patterns: Sequence[str],
    metric: str,
) -> tuple[float, float]:
    weight = 1.0 / len(patterns)
    mean = math.fsum(
        weight * strata[key]["metric_statistics"][metric]["mean"]
        for key in patterns
    )
    variance = math.fsum(
        (weight * strata[key]["metric_statistics"][metric]["standard_error"])
        ** 2
        for key in patterns
    )
    return float(mean), float(math.sqrt(variance))


def _comparison(
    prediction: float,
    prediction_se: float,
    actual: float,
    actual_se: float,
) -> dict[str, float | None]:
    difference = prediction - actual
    combined = math.hypot(prediction_se, actual_se)
    return {
        "prediction": prediction,
        "prediction_standard_error": prediction_se,
        "prediction_relative_95_half_width": (
            None if prediction == 0.0 else 1.96 * prediction_se / abs(prediction)
        ),
        "actual": actual,
        "actual_standard_error": actual_se,
        "prediction_minus_actual": difference,
        "combined_standard_error": combined,
        "absolute_relative_error": (
            None if actual == 0.0 else abs(difference) / abs(actual)
        ),
        "absolute_z_score": None if combined == 0.0 else abs(difference) / combined,
        "pointwise_normal_relative_95_upper_diagnostic": (
            None
            if actual == 0.0
            else (abs(difference) + 1.96 * combined) / abs(actual)
        ),
    }


def _checkpoint_task_fingerprint(task: Mapping[str, Any]) -> str:
    identity = {
        key: value
        for key, value in task.items()
        if key not in {"hamiltonian", "compiler"}
    }
    identity["hamiltonian_hash"] = df_hamiltonian_hash(task["hamiltonian"])
    identity["compiler"] = _compiler_payload(task["compiler"])
    identity["task_implementation_version"] = str(
        task.get(
            "task_implementation_version",
            CONNECTED_CLUSTER_TASK_IMPLEMENTATION_VERSION,
        )
    )
    return _fingerprint(identity)


def _checkpoint_path(
    directory: Path, *, stage: str, task: Mapping[str, Any]
) -> Path:
    explicit_key = task.get("task_key")
    if explicit_key is not None:
        key = str(explicit_key)
        safe_characters = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        )
        if not key or any(character not in safe_characters for character in key):
            raise ValueError("Connected-cluster checkpoint task_key is not filesystem-safe.")
        return directory / f"{stage}_{key}.json"
    if "cluster_length" in task:
        key = f"k{int(task['cluster_length'])}"
    else:
        key = f"L{int(task['sequence_length'])}"
    return directory / f"{stage}_{key}.json"


def _load_task_checkpoint(path: Path, task: Mapping[str, Any]) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if envelope.get("schema_version") != CONNECTED_CLUSTER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported connected-cluster checkpoint: {path}")
    expected_version = str(
        task.get(
            "task_implementation_version",
            CONNECTED_CLUSTER_TASK_IMPLEMENTATION_VERSION,
        )
    )
    if envelope.get("task_implementation_version") != expected_version:
        raise ValueError(f"Connected-cluster checkpoint task mismatch: {path}")
    expected = _checkpoint_task_fingerprint(task)
    if envelope.get("task_fingerprint") != expected:
        raise ValueError(f"Connected-cluster checkpoint task mismatch: {path}")
    output = envelope.get("output")
    if not isinstance(output, dict) or envelope.get("output_fingerprint") != _fingerprint(
        output
    ):
        raise ValueError(f"Connected-cluster checkpoint output mismatch: {path}")
    return output


def _write_task_checkpoint(
    path: Path, task: Mapping[str, Any], output: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": CONNECTED_CLUSTER_CHECKPOINT_SCHEMA_VERSION,
        "task_implementation_version": str(
            task.get(
                "task_implementation_version",
                CONNECTED_CLUSTER_TASK_IMPLEMENTATION_VERSION,
            )
        ),
        "task_fingerprint": _checkpoint_task_fingerprint(task),
        "output_fingerprint": _fingerprint(output),
        "output": output,
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


def _try_load_task_checkpoint(
    directory: str | Path | None,
    *,
    stage: str,
    task: Mapping[str, Any],
) -> dict[str, Any] | None:
    if directory is None:
        return None
    root = Path(directory)
    path = _checkpoint_path(root, stage=stage, task=task)
    if path.exists():
        try:
            return _load_task_checkpoint(path, task)
        except ValueError as error:
            if "task mismatch" not in str(error):
                raise
    fingerprint = _checkpoint_task_fingerprint(task)
    alternate = path.with_name(f"{path.stem}_{fingerprint[:12]}{path.suffix}")
    if alternate.exists():
        return _load_task_checkpoint(alternate, task)
    return None


def _run_tasks(
    tasks: Sequence[Mapping[str, Any]],
    workers: int,
    function,
    *,
    checkpoint_directory: str | Path | None = None,
    checkpoint_stage: str | None = None,
):
    if checkpoint_directory is None:
        if workers == 1:
            return [function(task) for task in tasks]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=context,
        ) as executor:
            futures = [executor.submit(function, task) for task in tasks]
            return [future.result() for future in as_completed(futures)]

    if checkpoint_stage is None:
        raise ValueError("checkpoint_stage is required with checkpoint_directory.")
    directory = Path(checkpoint_directory)
    completed: dict[str, dict[str, Any]] = {}
    missing: list[tuple[Mapping[str, Any], Path]] = []
    for task in tasks:
        fingerprint = _checkpoint_task_fingerprint(task)
        path = _checkpoint_path(directory, stage=checkpoint_stage, task=task)
        if path.exists():
            try:
                completed[fingerprint] = _load_task_checkpoint(path, task)
                continue
            except ValueError as error:
                if "task mismatch" not in str(error):
                    raise
                path = path.with_name(
                    f"{path.stem}_{fingerprint[:12]}{path.suffix}"
                )
                if path.exists():
                    completed[fingerprint] = _load_task_checkpoint(path, task)
                    continue
        missing.append((task, path))
    if workers == 1:
        for task, path in missing:
            output = function(task)
            _write_task_checkpoint(path, task, output)
            completed[_checkpoint_task_fingerprint(task)] = output
    elif missing:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(missing)),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(function, task): (task, path)
                for task, path in missing
            }
            for future in as_completed(futures):
                task, path = futures[future]
                output = future.result()
                _write_task_checkpoint(path, task, output)
                completed[_checkpoint_task_fingerprint(task)] = output
    return [completed[_checkpoint_task_fingerprint(task)] for task in tasks]


def _merge_stage_outputs(
    outputs: Sequence[Mapping[str, Any]],
    *,
    length_field: str,
) -> list[dict[str, Any]]:
    """Merge pattern-level worker outputs back into one payload per length."""
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for output in outputs:
        grouped.setdefault(int(output[length_field]), []).append(output)
    merged: list[dict[str, Any]] = []
    for length, items in sorted(grouped.items()):
        roles = {item.get("role") for item in items}
        hamiltonian_hashes = {item.get("hamiltonian_hash") for item in items}
        preparation_hashes = {item["preparation_hash"] for item in items}
        partition_hashes = {item["partition_hash"] for item in items}
        distributions = {_fingerprint(item["distribution"]) for item in items}
        if len(roles) != 1 or None in roles:
            raise RuntimeError("Chunked connected-cluster roles differ.")
        if len(hamiltonian_hashes) != 1 or None in hamiltonian_hashes:
            raise RuntimeError("Chunked connected-cluster Hamiltonians differ.")
        if len(preparation_hashes) != 1 or len(partition_hashes) != 1:
            raise RuntimeError("Chunked connected-cluster preparations differ.")
        if len(distributions) != 1:
            raise RuntimeError("Chunked connected-cluster distributions differ.")
        stratum_chunks: dict[str, list[Mapping[str, Any]]] = {}
        for item in items:
            for key, stratum in item["strata"].items():
                stratum_chunks.setdefault(key, []).append(stratum)
        strata = {
            key: _merge_stratum_chunks(chunks)
            for key, chunks in sorted(stratum_chunks.items())
        }
        performance = {
            "maximum_chunk_seconds": float(
                max(item["performance"]["total_seconds"] for item in items)
            ),
            "aggregate_worker_seconds": float(
                math.fsum(item["performance"]["total_seconds"] for item in items)
            ),
            "cache_hits": int(
                math.fsum(item["performance"]["cache_hits"] for item in items)
            ),
            "cache_misses": int(
                math.fsum(item["performance"]["cache_misses"] for item in items)
            ),
            "cache_evictions": int(
                math.fsum(item["performance"]["cache_evictions"] for item in items)
            ),
            "persistent_cache_hits": int(
                math.fsum(
                    item["performance"].get("persistent_cache_hits", 0)
                    for item in items
                )
            ),
            "persistent_cache_writes": int(
                math.fsum(
                    item["performance"].get("persistent_cache_writes", 0)
                    for item in items
                )
            ),
            "chunk_count": len(items),
        }
        merged.append(
            {
                "role": next(iter(roles)),
                length_field: length,
                "hamiltonian_hash": next(iter(hamiltonian_hashes)),
                "preparation_hash": next(iter(preparation_hashes)),
                "partition_hash": next(iter(partition_hashes)),
                "distribution": items[0]["distribution"],
                "strata": strata,
                "performance": performance,
            }
        )
    return merged


def _run_calibration_stage(
    common_task: Mapping[str, Any],
    *,
    role: str,
    sample_counts: Mapping[int, Mapping[str, int]],
    maximum_workers: int,
    checkpoint_directory: str | Path | None,
    chunk_patterns: bool,
    exact_order0_single: bool,
    sample_chunk_size: int | None,
) -> tuple[list[dict[str, Any]], float]:
    stage_started = time.perf_counter()
    tasks: list[dict[str, Any]] = []
    restored_outputs: list[dict[str, Any]] = []
    for length in (1, 2, 3):
        patterns = tuple(itertools.product(_ORDERS, repeat=length))
        if sample_chunk_size is not None:
            for pattern in patterns:
                pattern_key = _pattern_key(pattern)
                count = int(sample_counts[length][pattern_key])
                legacy_task = {
                    **common_task,
                    "role": role,
                    "cluster_length": length,
                    "patterns": (pattern,),
                    "task_key": f"k{length}_p{pattern_key.replace(',', '-')}",
                    "sample_counts": {pattern_key: count},
                    "pair_boundary_stratified": True,
                    "exact_order0_single": exact_order0_single,
                }
                restored = _try_load_task_checkpoint(
                    checkpoint_directory,
                    stage=role,
                    task=legacy_task,
                )
                if restored is not None:
                    restored_outputs.append(restored)
                    continue
                exact = (
                    role == "production"
                    and length == 1
                    and pattern == (0,)
                    and exact_order0_single
                )
                if exact:
                    tasks.append(legacy_task)
                    continue
                for chunk_index, sample_start, sample_stop in _sample_chunks(
                    count, sample_chunk_size
                ):
                    tasks.append(
                        {
                            **legacy_task,
                            "task_key": (
                                f"k{length}_p{pattern_key.replace(',', '-')}_"
                                f"c{chunk_index:06d}"
                            ),
                            "sample_counts": {pattern_key: sample_stop - sample_start},
                            "sample_start": sample_start,
                            "sample_stop": sample_stop,
                            "chunk_index": chunk_index,
                            "chunk_seed": _derived_chunk_seed(
                                int(common_task["master_seed"]),
                                role=role,
                                length=length,
                                pattern=pattern,
                                chunk_index=chunk_index,
                            ),
                            "task_implementation_version": (
                                CONNECTED_CLUSTER_CHUNK_TASK_IMPLEMENTATION_VERSION
                            ),
                        }
                    )
            continue
        if chunk_patterns:
            for pattern in patterns:
                pattern_key = _pattern_key(pattern)
                tasks.append(
                    {
                        **common_task,
                        "role": role,
                        "cluster_length": length,
                        "patterns": (pattern,),
                        "task_key": f"k{length}_p{pattern_key.replace(',', '-')}",
                        "sample_counts": {
                            pattern_key: int(sample_counts[length][pattern_key])
                        },
                        "pair_boundary_stratified": True,
                        "exact_order0_single": exact_order0_single,
                    }
                )
        else:
            tasks.append(
                {
                    **common_task,
                    "role": role,
                    "cluster_length": length,
                    "sample_counts": dict(sample_counts[length]),
                    "pair_boundary_stratified": True,
                    "exact_order0_single": exact_order0_single,
                }
            )
    outputs = list(restored_outputs)
    if tasks:
        outputs.extend(
            _run_tasks(
                tasks,
                maximum_workers,
                _compile_calibration_length,
                checkpoint_directory=checkpoint_directory,
                checkpoint_stage=role,
            )
        )
    merged = _merge_stage_outputs(outputs, length_field="cluster_length")
    return merged, float(time.perf_counter() - stage_started)


def calibrate_connected_cluster_cost_model(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    compiler: CompilerSettings,
    pilot_sample_count: int,
    minimum_production_sample_count: int,
    maximum_production_sample_count: int,
    prediction_relative_standard_error_target: float,
    allocation_safety_factor: float,
    seed: int,
    target_event_counts: Sequence[int] = (4, 6, 8),
    maximum_workers: int = 3,
    cache_maximum_entries: int = 32_768,
    persistent_cache_path: str | Path | None = None,
    maximum_circuit_size: int = 150_000,
    checkpoint_directory: str | Path | None = None,
    cost_aware_allocation: bool = True,
    chunk_patterns: bool = True,
    sample_chunk_size: int | None = None,
    adaptive_production_rounds: int = 0,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate local connected coefficients without compiling full holdouts."""
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    reference_rte_steps = require_integer_count(
        reference_rte_steps, name="reference_rte_steps", minimum=1
    )
    pilot_sample_count = require_integer_count(
        pilot_sample_count, name="pilot_sample_count", minimum=2
    )
    minimum_production_sample_count = require_integer_count(
        minimum_production_sample_count,
        name="minimum_production_sample_count",
        minimum=2,
    )
    maximum_production_sample_count = require_integer_count(
        maximum_production_sample_count,
        name="maximum_production_sample_count",
        minimum=minimum_production_sample_count,
    )
    maximum_workers = require_integer_count(
        maximum_workers, name="maximum_workers", minimum=1
    )
    if sample_chunk_size is not None:
        sample_chunk_size = require_integer_count(
            sample_chunk_size, name="sample_chunk_size", minimum=2
        )
    adaptive_production_rounds = require_integer_count(
        adaptive_production_rounds,
        name="adaptive_production_rounds",
        minimum=0,
    )
    for name, value in (
        (
            "prediction_relative_standard_error_target",
            prediction_relative_standard_error_target,
        ),
        ("allocation_safety_factor", allocation_safety_factor),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(reference_delta_time) or reference_delta_time <= 0.0:
        raise ValueError("reference_delta_time must be finite and positive.")
    normalized_target_lengths = tuple(
        sorted(
            {
                require_integer_count(value, name="target_event_count", minimum=1)
                for value in target_event_counts
            }
        )
    )
    if not normalized_target_lengths:
        raise ValueError("At least one target event count is required.")

    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian, partition, identity_policy="extract_identity_phase"
    )
    if preparation.is_deterministic_only:
        raise ValueError("Connected-cluster calibration requires a non-empty tail.")
    short_step_time = float(reference_delta_time) / reference_rte_steps
    tau = preparation.exact_rte_lambda_r * short_step_time
    _config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=short_step_time,
        rte_steps=1,
        truncation_tolerance=_explicit_cutoff_tolerance(tau),
        finite_taylor_order=2,
        seed=seed,
    )
    order_probabilities = dict(
        zip(distribution.orders, distribution.order_probabilities, strict=True)
    )
    forms = _required_forms(order_probabilities, normalized_target_lengths)
    common_task = {
        "hamiltonian": hamiltonian,
        "ld": ld,
        "short_step_time": short_step_time,
        "compiler": compiler,
        "master_seed": seed,
        "cache_maximum_entries": cache_maximum_entries,
        "persistent_cache_path": (
            None if persistent_cache_path is None else str(persistent_cache_path)
        ),
        "maximum_circuit_size": maximum_circuit_size,
        "direct_pair_conditioning": True,
    }
    pilot_counts = {
        length: {
            _pattern_key(pattern): pilot_sample_count
            for pattern in itertools.product(_ORDERS, repeat=length)
        }
        for length in (1, 2, 3)
    }
    pilot_outputs, pilot_stage_elapsed_seconds = _run_calibration_stage(
        common_task,
        role="pilot",
        sample_counts=pilot_counts,
        maximum_workers=maximum_workers,
        checkpoint_directory=checkpoint_directory,
        chunk_patterns=chunk_patterns,
        exact_order0_single=False,
        sample_chunk_size=sample_chunk_size,
    )
    pilot = {item["cluster_length"]: item for item in pilot_outputs}
    production_counts, allocation = _allocate_production_samples(
        pilot,
        forms,
        relative_standard_error_target=prediction_relative_standard_error_target,
        minimum_samples=minimum_production_sample_count,
        maximum_samples=maximum_production_sample_count,
        safety_factor=allocation_safety_factor,
        exact_parameter_keys=("k1:0",),
        cost_aware=cost_aware_allocation,
    )
    production_outputs, production_stage_elapsed_seconds = _run_calibration_stage(
        common_task,
        role="production",
        sample_counts=production_counts,
        maximum_workers=maximum_workers,
        checkpoint_directory=checkpoint_directory,
        chunk_patterns=chunk_patterns,
        exact_order0_single=True,
        sample_chunk_size=sample_chunk_size,
    )
    production = {item["cluster_length"]: item for item in production_outputs}
    adaptive_history = [
        {
            "round": 0,
            "maximum_realized_rz_relative_standard_error": (
                _maximum_form_relative_standard_error(production, forms)
            ),
            "production_sample_counts": {
                str(length): dict(counts)
                for length, counts in production_counts.items()
            },
        }
    ]
    adaptive_stop_reason = "disabled"
    for adaptive_round in range(1, adaptive_production_rounds + 1):
        realized = adaptive_history[-1][
            "maximum_realized_rz_relative_standard_error"
        ]
        if realized <= prediction_relative_standard_error_target:
            adaptive_stop_reason = "precision_target_met"
            break
        proposed_counts, proposed_allocation = _allocate_production_samples(
            production,
            forms,
            relative_standard_error_target=(
                prediction_relative_standard_error_target
            ),
            minimum_samples=minimum_production_sample_count,
            maximum_samples=maximum_production_sample_count,
            safety_factor=allocation_safety_factor,
            exact_parameter_keys=("k1:0",),
            cost_aware=cost_aware_allocation,
        )
        next_counts = {
            length: {
                pattern: max(
                    int(production_counts[length][pattern]),
                    int(proposed_counts[length][pattern]),
                )
                for pattern in production_counts[length]
            }
            for length in production_counts
        }
        if next_counts == production_counts:
            adaptive_stop_reason = (
                "maximum_sample_cap_reached"
                if proposed_allocation["capped_parameter_keys"]
                else "allocation_did_not_increase"
            )
            break
        production_counts = next_counts
        production_outputs, elapsed = _run_calibration_stage(
            common_task,
            role="production",
            sample_counts=production_counts,
            maximum_workers=maximum_workers,
            checkpoint_directory=checkpoint_directory,
            chunk_patterns=chunk_patterns,
            exact_order0_single=True,
            sample_chunk_size=sample_chunk_size,
        )
        production_stage_elapsed_seconds += elapsed
        production = {
            item["cluster_length"]: item for item in production_outputs
        }
        allocation = proposed_allocation
        adaptive_history.append(
            {
                "round": adaptive_round,
                "maximum_realized_rz_relative_standard_error": (
                    _maximum_form_relative_standard_error(production, forms)
                ),
                "production_sample_counts": {
                    str(length): dict(counts)
                    for length, counts in production_counts.items()
                },
            }
        )
    else:
        if adaptive_production_rounds > 0:
            adaptive_stop_reason = (
                "precision_target_met"
                if adaptive_history[-1][
                    "maximum_realized_rz_relative_standard_error"
                ]
                <= prediction_relative_standard_error_target
                else "adaptive_round_limit_reached"
            )
    allocation = {
        **allocation,
        "production_sample_counts": {
            str(length): dict(counts)
            for length, counts in production_counts.items()
        },
        "adaptive_history": adaptive_history,
        "adaptive_stop_reason": adaptive_stop_reason,
    }
    hashes = {preparation.preparation_hash}
    partition_hashes = {preparation.partition_hash}
    for item in (*pilot_outputs, *production_outputs):
        hashes.add(item["preparation_hash"])
        partition_hashes.add(item["partition_hash"])
    if len(hashes) != 1 or len(partition_hashes) != 1:
        raise RuntimeError("Connected-cluster calibration worker preparations differ.")

    configuration = {
        "ld": ld,
        "reference_delta_time": float(reference_delta_time),
        "reference_rte_steps": reference_rte_steps,
        "short_step_time": short_step_time,
        "dimensionless_step_time": tau,
        "finite_taylor_order": 2,
        "distribution": distribution.to_dict(),
        "target_event_counts": list(normalized_target_lengths),
        "pilot_sample_count_per_pattern": pilot_sample_count,
        "seed": seed,
        "compiler": _compiler_payload(compiler),
        "cost_aware_allocation": bool(cost_aware_allocation),
        "chunk_patterns": bool(chunk_patterns),
        "sample_chunk_size": sample_chunk_size,
        "sampling_stream_scheme": (
            None
            if sample_chunk_size is None
            else CONNECTED_CLUSTER_CHUNK_TASK_IMPLEMENTATION_VERSION
        ),
        "adaptive_production_rounds": adaptive_production_rounds,
        "checkpoint_directory": (
            None if checkpoint_directory is None else str(checkpoint_directory)
        ),
        "persistent_cache_path": (
            None if persistent_cache_path is None else str(persistent_cache_path)
        ),
    }
    condition_payload = {
        "hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
        "preparation_hash": preparation.preparation_hash,
        "partition_hash": preparation.partition_hash,
        "ld": ld,
        "short_step_time": short_step_time,
        "finite_taylor_order": 2,
        "compiler": _compiler_payload(compiler),
    }
    payload: dict[str, Any] = {
        "schema_version": CONNECTED_CLUSTER_CALIBRATION_SCHEMA_VERSION,
        "calibration_method": CONNECTED_CLUSTER_CALIBRATION_METHOD,
        "final_cost_evaluation_performed": False,
        "condition": condition_payload,
        "condition_fingerprint": _fingerprint(condition_payload),
        "hamiltonian": {
            "hash": df_hamiltonian_hash(hamiltonian),
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": hamiltonian.n_blocks,
            "metadata": dict(hamiltonian.metadata),
            "preparation_hash": preparation.preparation_hash,
            "partition_hash": preparation.partition_hash,
        },
        "configuration": configuration,
        "allocation": allocation,
        "pilot": {str(key): value for key, value in pilot.items()},
        "production": {str(key): value for key, value in production.items()},
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "maximum_workers": maximum_workers,
            "pilot_stage_elapsed_seconds": pilot_stage_elapsed_seconds,
            "production_stage_elapsed_seconds": production_stage_elapsed_seconds,
            "pilot": {
                f"k{item['cluster_length']}": item["performance"]
                for item in pilot_outputs
            },
            "production": {
                f"k{item['cluster_length']}": item["performance"]
                for item in production_outputs
            },
        },
        "provenance": dict(provenance or {}),
    }
    payload["calibration_fingerprint"] = _fingerprint(payload)
    validate_connected_cluster_calibration_payload(payload)
    return payload


def _expected_calibration_condition(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the scientific condition independently of its fingerprint."""
    hamiltonian = payload["hamiltonian"]
    configuration = payload["configuration"]
    return {
        "hamiltonian_hash": hamiltonian["hash"],
        "preparation_hash": hamiltonian["preparation_hash"],
        "partition_hash": hamiltonian["partition_hash"],
        "ld": int(configuration["ld"]),
        "short_step_time": float(configuration["short_step_time"]),
        "finite_taylor_order": int(configuration["finite_taylor_order"]),
        "compiler": configuration["compiler"],
    }


def _validate_stratum_chunk_semantics(
    stratum: Mapping[str, Any],
    *,
    master_seed: int,
    role: str,
    length: int,
    pattern: Sequence[int],
    sample_chunk_size: int | None,
) -> set[int]:
    scheme = stratum.get("sampling_stream_scheme")
    if scheme is None:
        return set()
    if scheme != CONNECTED_CLUSTER_CHUNK_TASK_IMPLEMENTATION_VERSION:
        raise ValueError("Connected-cluster sampling stream scheme mismatch.")
    if sample_chunk_size is None:
        raise ValueError("Connected-cluster chunk size is missing.")
    count_field = (
        "sample_count_per_boundary_class"
        if "sample_count_per_boundary_class" in stratum
        else "sample_count"
    )
    sample_count = int(stratum[count_field])
    expected_chunks = _sample_chunks(sample_count, sample_chunk_size)
    expected_ranges = [[start, stop] for _, start, stop in expected_chunks]
    if stratum.get("sample_ranges") != expected_ranges:
        raise ValueError("Connected-cluster sample chunk ranges mismatch.")
    expected_seeds = [
        _derived_chunk_seed(
            master_seed,
            role=role,
            length=length,
            pattern=pattern,
            chunk_index=chunk_index,
        )
        for chunk_index, _start, _stop in expected_chunks
    ]
    if stratum.get("chunk_seeds") != expected_seeds:
        raise ValueError("Connected-cluster sample chunk seeds mismatch.")
    if stratum.get("chunk_count") != len(expected_chunks):
        raise ValueError("Connected-cluster sample chunk count mismatch.")
    return set(expected_seeds)


def _validate_calibration_stage_semantics(
    payload: Mapping[str, Any], *, role: str
) -> set[int]:
    configuration = payload["configuration"]
    hamiltonian = payload["hamiltonian"]
    stage = payload[role]
    seeds: set[int] = set()
    chunk_seeds: set[int] = set()
    raw_chunk_size = configuration.get("sample_chunk_size")
    sample_chunk_size = None if raw_chunk_size is None else int(raw_chunk_size)
    for length in (1, 2, 3):
        item = stage[str(length)]
        if item.get("role") != role:
            raise ValueError(f"Connected-cluster {role} role mismatch.")
        if item.get("cluster_length") != length:
            raise ValueError(f"Connected-cluster {role} length mismatch.")
        if item.get("hamiltonian_hash") != hamiltonian["hash"]:
            raise ValueError(f"Connected-cluster {role} Hamiltonian hash mismatch.")
        if item.get("preparation_hash") != hamiltonian["preparation_hash"]:
            raise ValueError(f"Connected-cluster {role} preparation hash mismatch.")
        if item.get("partition_hash") != hamiltonian["partition_hash"]:
            raise ValueError(f"Connected-cluster {role} partition hash mismatch.")
        if item.get("distribution") != configuration["distribution"]:
            raise ValueError(f"Connected-cluster {role} distribution mismatch.")
        expected_patterns = {
            _pattern_key(pattern)
            for pattern in itertools.product(_ORDERS, repeat=length)
        }
        if set(item.get("strata", {})) != expected_patterns:
            raise ValueError(f"Connected-cluster {role} patterns are incomplete.")
        for pattern_key, stratum in item["strata"].items():
            pattern = _parse_pattern(pattern_key)
            if tuple(stratum.get("order_pattern", ())) != pattern:
                raise ValueError(f"Connected-cluster {role} order pattern mismatch.")
            if stratum.get("parameter_key") != _parameter_key(pattern):
                raise ValueError(f"Connected-cluster {role} parameter key mismatch.")
            if stratum.get("rare_order_count") != sum(
                value == 2 for value in pattern
            ):
                raise ValueError(f"Connected-cluster {role} rare-order count mismatch.")
            exact = (
                role == "production"
                and pattern == (0,)
                and stratum.get("estimate_kind")
                == "exact_conditional_order0_enumeration"
            )
            expected_seed = (
                None
                if exact
                else _derived_seed(
                    int(configuration["seed"]),
                    role=role,
                    length=length,
                    pattern=pattern,
                )
            )
            if stratum.get("seed") != expected_seed:
                raise ValueError(f"Connected-cluster {role} seed mismatch.")
            if expected_seed is not None:
                if expected_seed in seeds:
                    raise ValueError(f"Connected-cluster {role} seeds are not unique.")
                seeds.add(expected_seed)
            stratum_chunk_seeds = _validate_stratum_chunk_semantics(
                stratum,
                master_seed=int(configuration["seed"]),
                role=role,
                length=length,
                pattern=pattern,
                sample_chunk_size=sample_chunk_size,
            )
            if chunk_seeds & stratum_chunk_seeds:
                raise ValueError("Connected-cluster sample chunk seeds are not unique.")
            chunk_seeds.update(stratum_chunk_seeds)
    return seeds


def validate_connected_cluster_calibration_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CONNECTED_CLUSTER_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("Unsupported connected-cluster calibration schema.")
    if payload.get("calibration_method") != CONNECTED_CLUSTER_CALIBRATION_METHOD:
        raise ValueError("Unsupported connected-cluster calibration method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("A calibration cannot contain a final cost evaluation.")
    if set(payload.get("pilot", {})) != {"1", "2", "3"}:
        raise ValueError("Pilot connected-cluster lengths are incomplete.")
    if set(payload.get("production", {})) != {"1", "2", "3"}:
        raise ValueError("Production connected-cluster lengths are incomplete.")
    try:
        expected_condition = _expected_calibration_condition(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Connected-cluster calibration condition is incomplete."
        ) from error
    if payload.get("condition") != expected_condition:
        raise ValueError("Connected-cluster calibration condition mismatch.")
    if payload.get("condition_fingerprint") != _fingerprint(expected_condition):
        raise ValueError(
            "Connected-cluster calibration condition fingerprint mismatch."
        )
    preparation_hashes = {
        item["preparation_hash"]
        for role in ("pilot", "production")
        for item in payload[role].values()
    }
    preparation_hashes.add(payload.get("hamiltonian", {}).get("preparation_hash"))
    if len(preparation_hashes) != 1:
        raise ValueError("Connected-cluster calibration preparation hashes differ.")
    pilot_seeds = _validate_calibration_stage_semantics(payload, role="pilot")
    production_seeds = _validate_calibration_stage_semantics(
        payload, role="production"
    )
    if pilot_seeds & production_seeds:
        raise ValueError("Pilot and production connected-cluster seeds overlap.")
    fingerprint = payload.get("calibration_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("calibration_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Connected-cluster calibration fingerprint mismatch.")


def write_connected_cluster_calibration(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_connected_cluster_calibration_payload(payload)
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


def load_connected_cluster_calibration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_connected_cluster_calibration_payload(payload)
    return payload


def predict_connected_cluster_cost(
    calibration: Mapping[str, Any],
    *,
    event_count: int,
    order_pattern: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Predict cost without circuit construction or transpilation."""
    validate_connected_cluster_calibration_payload(calibration)
    length = require_integer_count(event_count, name="event_count", minimum=1)
    distribution = calibration["configuration"]["distribution"]
    order_probabilities = dict(
        zip(
            (int(value) for value in distribution["orders"]),
            (float(value) for value in distribution["order_probabilities"]),
            strict=True,
        )
    )
    if order_pattern is None:
        form = _iid_sequence_form(length, order_probabilities)
        prediction_kind = "iid_analytic_order_weighting"
        normalized_pattern = None
    else:
        normalized_pattern = tuple(int(value) for value in order_pattern)
        if len(normalized_pattern) != length:
            raise ValueError("order_pattern length must equal event_count.")
        if any(value not in _ORDERS for value in normalized_pattern):
            raise ValueError("order_pattern contains an unsupported Taylor order.")
        form = _sequence_form(normalized_pattern)
        prediction_kind = "conditioned_order_pattern"
    lookup = _statistics_lookup(
        {int(key): value for key, value in calibration["production"].items()}
    )
    metrics = {}
    for metric in _METRICS:
        mean, standard_error = _evaluate_form(form, lookup, metric)
        metrics[metric] = {
            "mean": mean,
            "calibration_standard_error": standard_error,
            "calibration_relative_95_half_width": (
                None if mean == 0.0 else 1.96 * standard_error / abs(mean)
            ),
            "empirical_model_discrepancy": None,
        }
    return {
        "prediction_kind": prediction_kind,
        "event_count": length,
        "order_pattern": (
            None if normalized_pattern is None else list(normalized_pattern)
        ),
        "condition_fingerprint": calibration["condition_fingerprint"],
        "calibration_fingerprint": calibration["calibration_fingerprint"],
        "prediction_form": dict(sorted(form.items())),
        "metrics": metrics,
        "requires_qiskit_transpile": False,
        "validation_status": "calibrated_not_transfer_validated_by_this_payload",
    }


def _run_holdout_stage(
    common_task: Mapping[str, Any],
    *,
    lengths: Sequence[int],
    zero_sample_count: int,
    single_rare_sample_count: int,
    maximum_workers: int,
    checkpoint_directory: str | Path | None,
    chunk_patterns: bool,
    sample_chunk_size: int | None,
) -> tuple[list[dict[str, Any]], float]:
    stage_started = time.perf_counter()
    tasks: list[dict[str, Any]] = []
    restored_outputs: list[dict[str, Any]] = []
    for raw_length in lengths:
        length = require_integer_count(raw_length, name="holdout_length", minimum=1)
        patterns = [(0,) * length]
        patterns.extend(
            tuple(2 if index == rare else 0 for index in range(length))
            for rare in range(length)
        )
        if sample_chunk_size is not None:
            for pattern in patterns:
                pattern_key = _pattern_key(pattern)
                count = (
                    zero_sample_count
                    if 2 not in pattern
                    else single_rare_sample_count
                )
                legacy_task = {
                    **common_task,
                    "sequence_length": length,
                    "patterns": (pattern,),
                    "task_key": f"L{length}_p{pattern_key.replace(',', '-')}",
                    "zero_sample_count": zero_sample_count,
                    "single_rare_sample_count": single_rare_sample_count,
                }
                restored = _try_load_task_checkpoint(
                    checkpoint_directory,
                    stage="holdout",
                    task=legacy_task,
                )
                if restored is not None:
                    restored_outputs.append(restored)
                    continue
                for chunk_index, sample_start, sample_stop in _sample_chunks(
                    count, sample_chunk_size
                ):
                    chunk_count = sample_stop - sample_start
                    tasks.append(
                        {
                            **legacy_task,
                            "task_key": (
                                f"L{length}_p{pattern_key.replace(',', '-')}_"
                                f"c{chunk_index:06d}"
                            ),
                            "zero_sample_count": chunk_count,
                            "single_rare_sample_count": chunk_count,
                            "sample_start": sample_start,
                            "sample_stop": sample_stop,
                            "chunk_index": chunk_index,
                            "chunk_seed": _derived_chunk_seed(
                                int(common_task["master_seed"]),
                                role="operational_holdout",
                                length=length,
                                pattern=pattern,
                                chunk_index=chunk_index,
                            ),
                            "task_implementation_version": (
                                CONNECTED_CLUSTER_CHUNK_TASK_IMPLEMENTATION_VERSION
                            ),
                        }
                    )
            continue
        if chunk_patterns:
            for pattern in patterns:
                pattern_key = _pattern_key(pattern)
                tasks.append(
                    {
                        **common_task,
                        "sequence_length": length,
                        "patterns": (pattern,),
                        "task_key": f"L{length}_p{pattern_key.replace(',', '-')}",
                        "zero_sample_count": zero_sample_count,
                        "single_rare_sample_count": single_rare_sample_count,
                    }
                )
        else:
            tasks.append(
                {
                    **common_task,
                    "sequence_length": length,
                    "zero_sample_count": zero_sample_count,
                    "single_rare_sample_count": single_rare_sample_count,
                }
            )
    outputs = list(restored_outputs)
    if tasks:
        outputs.extend(
            _run_tasks(
                tasks,
                maximum_workers,
                _compile_full_holdout_length,
                checkpoint_directory=checkpoint_directory,
                checkpoint_stage="holdout",
            )
        )
    merged = _merge_stage_outputs(outputs, length_field="sequence_length")
    return merged, float(time.perf_counter() - stage_started)


def validate_connected_cluster_calibration_holdout(
    calibration: Mapping[str, Any],
    hamiltonian: DFHamiltonian,
    *,
    compiler: CompilerSettings,
    holdout_lengths: Sequence[int] = (4, 6, 8),
    holdout_zero_sample_count: int,
    holdout_single_rare_sample_count: int,
    seed: int,
    maximum_workers: int = 3,
    cache_maximum_entries: int = 32_768,
    persistent_cache_path: str | Path | None = None,
    maximum_circuit_size: int = 150_000,
    checkpoint_directory: str | Path | None = None,
    chunk_patterns: bool = True,
    sample_chunk_size: int | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one fixed calibration on unused, fully compiled event sequences."""
    started = time.perf_counter()
    validate_connected_cluster_calibration_payload(calibration)
    if df_hamiltonian_hash(hamiltonian) != calibration["hamiltonian"]["hash"]:
        raise ValueError("Holdout Hamiltonian differs from the calibration snapshot.")
    if _compiler_payload(compiler) != calibration["configuration"]["compiler"]:
        raise ValueError("Holdout compiler differs from the calibration compiler.")
    zero_count = require_integer_count(
        holdout_zero_sample_count,
        name="holdout_zero_sample_count",
        minimum=2,
    )
    rare_count = require_integer_count(
        holdout_single_rare_sample_count,
        name="holdout_single_rare_sample_count",
        minimum=2,
    )
    workers = require_integer_count(maximum_workers, name="maximum_workers", minimum=1)
    if sample_chunk_size is not None:
        sample_chunk_size = require_integer_count(
            sample_chunk_size, name="sample_chunk_size", minimum=2
        )
    lengths = tuple(
        sorted(
            {
                require_integer_count(value, name="holdout_length", minimum=1)
                for value in holdout_lengths
            }
        )
    )
    if not lengths:
        raise ValueError("At least one holdout length is required.")
    configuration = calibration["configuration"]
    if int(seed) == int(configuration["seed"]):
        raise ValueError("Holdout seed must differ from the calibration seed.")
    common_task = {
        "hamiltonian": hamiltonian,
        "ld": int(configuration["ld"]),
        "short_step_time": float(configuration["short_step_time"]),
        "compiler": compiler,
        "master_seed": int(seed),
        "cache_maximum_entries": cache_maximum_entries,
        "persistent_cache_path": (
            None if persistent_cache_path is None else str(persistent_cache_path)
        ),
        "maximum_circuit_size": maximum_circuit_size,
    }
    outputs, holdout_stage_elapsed_seconds = _run_holdout_stage(
        common_task,
        lengths=lengths,
        zero_sample_count=zero_count,
        single_rare_sample_count=rare_count,
        maximum_workers=workers,
        checkpoint_directory=checkpoint_directory,
        chunk_patterns=chunk_patterns,
        sample_chunk_size=sample_chunk_size,
    )
    expected_preparation_hash = calibration["hamiltonian"]["preparation_hash"]
    expected_partition_hash = calibration["hamiltonian"]["partition_hash"]
    if any(
        item["preparation_hash"] != expected_preparation_hash
        or item["partition_hash"] != expected_partition_hash
        for item in outputs
    ):
        raise ValueError("Holdout preparation differs from the calibration.")
    holdout = {item["sequence_length"]: item for item in outputs}
    production_lookup = _statistics_lookup(
        {int(key): value for key, value in calibration["production"].items()}
    )
    comparisons: dict[str, Any] = {}
    primary: list[Mapping[str, Any]] = []
    all_metrics: list[Mapping[str, Any]] = []
    for length in lengths:
        strata = holdout[length]["strata"]
        zero_patterns = tuple(
            key for key, value in strata.items() if value["rare_order_count"] == 0
        )
        one_patterns = tuple(
            key for key, value in strata.items() if value["rare_order_count"] == 1
        )
        forms = {
            "zero_order2_condition": _sequence_form((0,) * length),
            "exactly_one_order2_condition": _average_forms(
                [_sequence_form(_parse_pattern(pattern)) for pattern in one_patterns]
            ),
        }
        comparisons[str(length)] = {}
        for scope, patterns in (
            ("zero_order2_condition", zero_patterns),
            ("exactly_one_order2_condition", one_patterns),
        ):
            metric_results = {}
            for metric in _METRICS:
                prediction, prediction_se = _evaluate_form(
                    forms[scope], production_lookup, metric
                )
                actual, actual_se = _aggregate_holdout(strata, patterns, metric)
                result = _comparison(prediction, prediction_se, actual, actual_se)
                metric_results[metric] = result
                all_metrics.append(result)
                if metric == "rz_count":
                    primary.append(result)
            comparisons[str(length)][scope] = {
                "patterns": list(patterns),
                "prediction_form": dict(sorted(forms[scope].items())),
                "metrics": metric_results,
            }

    def max_present(field: str, values: Sequence[Mapping[str, Any]]):
        present = [value[field] for value in values if value[field] is not None]
        return None if not present else float(max(present))

    summary = {
        "primary_maximum_absolute_relative_error": max_present(
            "absolute_relative_error", primary
        ),
        "primary_maximum_absolute_z_score": max_present(
            "absolute_z_score", primary
        ),
        "primary_maximum_pointwise_normal_95_upper_diagnostic": max_present(
            "pointwise_normal_relative_95_upper_diagnostic", primary
        ),
        "primary_maximum_prediction_relative_95_half_width": max_present(
            "prediction_relative_95_half_width", primary
        ),
        "primary_point_tolerance_passed": all(
            value["absolute_relative_error"] is not None
            and value["absolute_relative_error"] <= 0.05
            for value in primary
        ),
        "primary_prediction_precision_passed": all(
            value["prediction_relative_95_half_width"] is not None
            and value["prediction_relative_95_half_width"] <= 0.02
            for value in primary
        ),
        "all_metrics_maximum_absolute_relative_error": max_present(
            "absolute_relative_error", all_metrics
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": CONNECTED_CLUSTER_TRANSFER_SCHEMA_VERSION,
        "validation_method": CONNECTED_CLUSTER_TRANSFER_METHOD,
        "final_cost_evaluation_performed": False,
        "condition_fingerprint": calibration["condition_fingerprint"],
        "calibration_fingerprint": calibration["calibration_fingerprint"],
        "calibration_reference": {
            "calibration_fingerprint": calibration["calibration_fingerprint"],
            "condition": calibration["condition"],
            "condition_fingerprint": calibration["condition_fingerprint"],
            "hamiltonian_hash": calibration["hamiltonian"]["hash"],
            "preparation_hash": calibration["hamiltonian"]["preparation_hash"],
            "partition_hash": calibration["hamiltonian"]["partition_hash"],
            "calibration_seed": int(calibration["configuration"]["seed"]),
            "distribution_fingerprint": _fingerprint(
                calibration["configuration"]["distribution"]
            ),
            "compiler": calibration["configuration"]["compiler"],
        },
        "configuration": {
            "holdout_lengths": list(lengths),
            "holdout_zero_sample_count_per_length": zero_count,
            "holdout_single_rare_sample_count_per_position": rare_count,
            "seed": int(seed),
            "compiler": _compiler_payload(compiler),
            "chunk_patterns": bool(chunk_patterns),
            "sample_chunk_size": sample_chunk_size,
            "sampling_stream_scheme": (
                None
                if sample_chunk_size is None
                else CONNECTED_CLUSTER_CHUNK_TASK_IMPLEMENTATION_VERSION
            ),
            "checkpoint_directory": (
                None if checkpoint_directory is None else str(checkpoint_directory)
            ),
            "persistent_cache_path": (
                None if persistent_cache_path is None else str(persistent_cache_path)
            ),
        },
        "holdout": {str(key): value for key, value in holdout.items()},
        "holdout_comparisons": comparisons,
        "summary": summary,
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "maximum_workers": workers,
            "holdout_stage_elapsed_seconds": holdout_stage_elapsed_seconds,
            "holdout": {
                f"L{item['sequence_length']}": item["performance"]
                for item in outputs
            },
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    validate_connected_cluster_transfer_payload(payload)
    return payload


def validate_connected_cluster_transfer_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CONNECTED_CLUSTER_TRANSFER_SCHEMA_VERSION:
        raise ValueError("Unsupported connected-cluster transfer schema.")
    if payload.get("validation_method") != CONNECTED_CLUSTER_TRANSFER_METHOD:
        raise ValueError("Unsupported connected-cluster transfer method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("A transfer validation cannot be a final cost evaluation.")
    configured = {
        str(value) for value in payload.get("configuration", {}).get("holdout_lengths", [])
    }
    if set(payload.get("holdout", {})) != configured:
        raise ValueError("Connected-cluster transfer holdout lengths are incomplete.")
    reference = payload.get("calibration_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("Connected-cluster calibration reference is missing.")
    if reference.get("calibration_fingerprint") != payload.get(
        "calibration_fingerprint"
    ):
        raise ValueError("Connected-cluster calibration fingerprint linkage mismatch.")
    condition = reference.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError("Connected-cluster transfer condition is missing.")
    if reference.get("condition_fingerprint") != _fingerprint(condition):
        raise ValueError("Connected-cluster transfer condition fingerprint mismatch.")
    if payload.get("condition_fingerprint") != reference.get(
        "condition_fingerprint"
    ):
        raise ValueError("Connected-cluster transfer condition linkage mismatch.")
    for key in ("hamiltonian_hash", "preparation_hash", "partition_hash"):
        if reference.get(key) != condition.get(key):
            raise ValueError(f"Connected-cluster transfer {key} mismatch.")
    if reference.get("compiler") != condition.get("compiler"):
        raise ValueError("Connected-cluster transfer condition compiler mismatch.")
    configuration = payload.get("configuration", {})
    if configuration.get("compiler") != reference.get("compiler"):
        raise ValueError("Connected-cluster transfer compiler mismatch.")
    try:
        holdout_seed = int(configuration["seed"])
        calibration_seed = int(reference["calibration_seed"])
        zero_count = int(configuration["holdout_zero_sample_count_per_length"])
        rare_count = int(
            configuration["holdout_single_rare_sample_count_per_position"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Connected-cluster transfer sampling configuration is incomplete."
        ) from error
    if holdout_seed == calibration_seed:
        raise ValueError("Connected-cluster transfer seed overlaps calibration.")
    comparison_lengths = set(payload.get("holdout_comparisons", {}))
    if comparison_lengths != configured:
        raise ValueError("Connected-cluster transfer comparisons are incomplete.")
    seen_seeds: set[int] = set()
    seen_chunk_seeds: set[int] = set()
    raw_chunk_size = configuration.get("sample_chunk_size")
    sample_chunk_size = None if raw_chunk_size is None else int(raw_chunk_size)
    for length_key, item in payload["holdout"].items():
        length = int(length_key)
        if item.get("role") != "holdout" or item.get("sequence_length") != length:
            raise ValueError("Connected-cluster holdout role or length mismatch.")
        if item.get("hamiltonian_hash") != reference.get("hamiltonian_hash"):
            raise ValueError("Connected-cluster holdout Hamiltonian hash mismatch.")
        if item.get("preparation_hash") != reference.get("preparation_hash"):
            raise ValueError("Connected-cluster holdout preparation hash mismatch.")
        if item.get("partition_hash") != reference.get("partition_hash"):
            raise ValueError("Connected-cluster holdout partition hash mismatch.")
        if _fingerprint(item.get("distribution", {})) != reference.get(
            "distribution_fingerprint"
        ):
            raise ValueError("Connected-cluster holdout distribution mismatch.")
        expected_patterns = {(0,) * length}
        expected_patterns.update(
            tuple(2 if index == rare else 0 for index in range(length))
            for rare in range(length)
        )
        expected_keys = {_pattern_key(pattern) for pattern in expected_patterns}
        if set(item.get("strata", {})) != expected_keys:
            raise ValueError("Connected-cluster holdout patterns are incomplete.")
        for pattern_key, stratum in item["strata"].items():
            pattern = _parse_pattern(pattern_key)
            if tuple(stratum.get("order_pattern", ())) != pattern:
                raise ValueError("Connected-cluster holdout order pattern mismatch.")
            expected_count = zero_count if 2 not in pattern else rare_count
            if stratum.get("sample_count") != expected_count:
                raise ValueError("Connected-cluster holdout sample count mismatch.")
            expected_seed = _derived_seed(
                holdout_seed,
                role="operational_holdout",
                length=length,
                pattern=pattern,
            )
            if stratum.get("seed") != expected_seed:
                raise ValueError("Connected-cluster holdout seed mismatch.")
            if expected_seed in seen_seeds:
                raise ValueError("Connected-cluster holdout seeds are not unique.")
            seen_seeds.add(expected_seed)
            stratum_chunk_seeds = _validate_stratum_chunk_semantics(
                stratum,
                master_seed=holdout_seed,
                role="operational_holdout",
                length=length,
                pattern=pattern,
                sample_chunk_size=sample_chunk_size,
            )
            if seen_chunk_seeds & stratum_chunk_seeds:
                raise ValueError(
                    "Connected-cluster holdout chunk seeds are not unique."
                )
            seen_chunk_seeds.update(stratum_chunk_seeds)
        comparison = payload["holdout_comparisons"][length_key]
        if set(comparison) != {
            "zero_order2_condition",
            "exactly_one_order2_condition",
        }:
            raise ValueError("Connected-cluster holdout comparison scopes differ.")
    fingerprint = payload.get("validation_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Connected-cluster transfer fingerprint mismatch.")


def write_connected_cluster_transfer_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_connected_cluster_transfer_payload(payload)
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


def diagnose_connected_cluster_k4_extrapolation(
    transfer: Mapping[str, Any],
    *,
    fit_length: int = 4,
    test_length: int = 6,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the missing four-event term at L=4 and test it at a longer L.

    The aggregate uncertainty is only a diagnostic because the L=4 and longer-L
    predictions share calibration coefficients whose covariance is unavailable.
    """
    validate_connected_cluster_transfer_payload(transfer)
    fit_length = require_integer_count(fit_length, name="fit_length", minimum=4)
    test_length = require_integer_count(test_length, name="test_length", minimum=4)
    if fit_length != 4:
        raise ValueError("The K4 residual fit currently requires fit_length=4.")
    if test_length <= fit_length:
        raise ValueError("test_length must be greater than fit_length.")
    comparisons = transfer["holdout_comparisons"]
    fit_key = str(fit_length)
    test_key = str(test_length)
    if fit_key not in comparisons or test_key not in comparisons:
        raise ValueError("The transfer payload lacks the requested fit/test lengths.")

    zero_fit = comparisons[fit_key]["zero_order2_condition"]["metrics"]
    one_fit = comparisons[fit_key]["exactly_one_order2_condition"]["metrics"]
    zero_test = comparisons[test_key]["zero_order2_condition"]["metrics"]
    one_test = comparisons[test_key]["exactly_one_order2_condition"]["metrics"]
    zero_multiplicity = float(test_length - 3)
    one_zero_multiplicity = float(
        (test_length - 4) * (test_length - 3) / test_length
    )
    one_rare_multiplicity = float(4 * (test_length - 3) / test_length)

    fitted_coefficients: dict[str, Any] = {}
    adjusted: dict[str, Any] = {
        "zero_order2_condition": {"metrics": {}},
        "exactly_one_order2_condition": {"metrics": {}},
    }
    point_errors: list[float] = []
    for metric in _METRICS:
        zero_coefficient = float(zero_fit[metric]["actual"]) - float(
            zero_fit[metric]["prediction"]
        )
        one_coefficient = float(one_fit[metric]["actual"]) - float(
            one_fit[metric]["prediction"]
        )
        zero_coefficient_se = float(zero_fit[metric]["combined_standard_error"])
        one_coefficient_se = float(one_fit[metric]["combined_standard_error"])
        fitted_coefficients[metric] = {
            "k4_zero_order2_mean": zero_coefficient,
            "k4_zero_order2_naive_independence_standard_error": zero_coefficient_se,
            "k4_exactly_one_order2_averaged_mean": one_coefficient,
            "k4_exactly_one_order2_averaged_naive_independence_standard_error": (
                one_coefficient_se
            ),
        }

        cases = (
            (
                "zero_order2_condition",
                zero_test[metric],
                zero_multiplicity * zero_coefficient,
                (zero_multiplicity * zero_coefficient_se) ** 2,
            ),
            (
                "exactly_one_order2_condition",
                one_test[metric],
                one_zero_multiplicity * zero_coefficient
                + one_rare_multiplicity * one_coefficient,
                (one_zero_multiplicity * zero_coefficient_se) ** 2
                + (one_rare_multiplicity * one_coefficient_se) ** 2,
            ),
        )
        for condition, source, correction, correction_variance in cases:
            prediction = float(source["prediction"]) + correction
            actual = float(source["actual"])
            difference = prediction - actual
            relative_error = None if actual == 0.0 else abs(difference) / abs(actual)
            prediction_se = math.sqrt(
                float(source["prediction_standard_error"]) ** 2
                + correction_variance
            )
            combined_se = math.hypot(
                prediction_se, float(source["actual_standard_error"])
            )
            diagnostic = {
                "uncorrected_prediction": float(source["prediction"]),
                "k4_correction": correction,
                "adjusted_prediction": prediction,
                "actual": actual,
                "adjusted_prediction_minus_actual": difference,
                "adjusted_absolute_relative_error": relative_error,
                "naive_independence_prediction_standard_error": prediction_se,
                "naive_independence_absolute_z_score": (
                    None if combined_se == 0.0 else abs(difference) / combined_se
                ),
                "uncertainty_status": (
                    "diagnostic_only_shared_calibration_covariance_not_available"
                ),
            }
            adjusted[condition]["metrics"][metric] = diagnostic
            if relative_error is not None:
                point_errors.append(relative_error)

    maximum_error = max(point_errors, default=0.0)
    payload = {
        "schema_version": CONNECTED_CLUSTER_K4_EXTRAPOLATION_SCHEMA_VERSION,
        "diagnostic_method": CONNECTED_CLUSTER_K4_EXTRAPOLATION_METHOD,
        "final_cost_evaluation_performed": False,
        "source_validation_fingerprint": transfer["validation_fingerprint"],
        "source_calibration_fingerprint": transfer["calibration_fingerprint"],
        "condition_fingerprint": transfer["condition_fingerprint"],
        "fit_length": fit_length,
        "test_length": test_length,
        "window_multiplicities": {
            "zero_order2_condition_k4_zero": zero_multiplicity,
            "exactly_one_order2_condition_k4_zero": one_zero_multiplicity,
            "exactly_one_order2_condition_k4_single_rare_average": (
                one_rare_multiplicity
            ),
        },
        "fitted_k4_coefficients": fitted_coefficients,
        "test_comparisons": adjusted,
        "summary": {
            "maximum_adjusted_absolute_relative_error": maximum_error,
            "point_tolerance": 0.05,
            "point_tolerance_passed": maximum_error <= 0.05,
            "interpretation": (
                "k4_point_transfer_supported"
                if maximum_error <= 0.05
                else "k4_point_transfer_not_supported"
            ),
            "uncertainty_is_acceptance_criterion": False,
        },
        "provenance": dict(provenance or {}),
    }
    payload["diagnostic_fingerprint"] = _fingerprint(payload)
    return payload


def validate_connected_cluster_k4_extrapolation_payload(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != (
        CONNECTED_CLUSTER_K4_EXTRAPOLATION_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported connected-cluster K4 diagnostic schema.")
    if payload.get("diagnostic_method") != CONNECTED_CLUSTER_K4_EXTRAPOLATION_METHOD:
        raise ValueError("Unsupported connected-cluster K4 diagnostic method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("A K4 diagnostic cannot contain a final cost evaluation.")
    fingerprint = payload.get("diagnostic_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("diagnostic_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Connected-cluster K4 diagnostic fingerprint mismatch.")


def write_connected_cluster_k4_extrapolation_diagnostic(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_connected_cluster_k4_extrapolation_payload(payload)
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


def calibrate_and_validate_connected_cluster_k4(
    transfer: Mapping[str, Any],
    hamiltonian: DFHamiltonian,
    *,
    compiler: CompilerSettings,
    sample_count_per_pattern: int,
    seed: int,
    maximum_workers: int = 3,
    cache_maximum_entries: int = 32_768,
    persistent_cache_path: str | Path | None = None,
    maximum_circuit_size: int = 200_000,
    checkpoint_directory: str | Path | None = None,
    sample_chunk_size: int | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate paired four-event coefficients and test fixed full holdouts."""
    started = time.perf_counter()
    validate_connected_cluster_transfer_payload(transfer)
    sample_count_per_pattern = require_integer_count(
        sample_count_per_pattern,
        name="sample_count_per_pattern",
        minimum=2,
    )
    maximum_workers = require_integer_count(
        maximum_workers, name="maximum_workers", minimum=1
    )
    if sample_chunk_size is not None:
        sample_chunk_size = require_integer_count(
            sample_chunk_size, name="sample_chunk_size", minimum=2
        )
    condition = transfer["calibration_reference"]["condition"]
    if df_hamiltonian_hash(hamiltonian) != condition["hamiltonian_hash"]:
        raise ValueError("K4 calibration Hamiltonian does not match the transfer.")
    if _compiler_payload(compiler) != condition["compiler"]:
        raise ValueError("K4 calibration compiler does not match the transfer.")
    ld = int(condition["ld"])
    short_step_time = float(condition["short_step_time"])
    patterns = tuple(
        pattern
        for pattern in itertools.product(_ORDERS, repeat=4)
        if sum(order == 2 for order in pattern) <= 1
    )
    common_task = {
        "hamiltonian": hamiltonian,
        "ld": ld,
        "short_step_time": short_step_time,
        "compiler": compiler,
        "master_seed": seed,
        "cache_maximum_entries": cache_maximum_entries,
        "persistent_cache_path": (
            None if persistent_cache_path is None else str(persistent_cache_path)
        ),
        "maximum_circuit_size": maximum_circuit_size,
        "role": "k4_calibration",
        "cluster_length": 4,
    }
    tasks: list[dict[str, Any]] = []
    for pattern in patterns:
        pattern_key = _pattern_key(pattern)
        base = {
            **common_task,
            "patterns": (pattern,),
            "sample_counts": {pattern_key: sample_count_per_pattern},
            "task_key": f"k4_p{pattern_key.replace(',', '-')}",
            "task_implementation_version": (
                CONNECTED_CLUSTER_K4_CALIBRATION_METHOD
            ),
        }
        if sample_chunk_size is None:
            tasks.append(base)
            continue
        for chunk_index, sample_start, sample_stop in _sample_chunks(
            sample_count_per_pattern, sample_chunk_size
        ):
            tasks.append(
                {
                    **base,
                    "task_key": (
                        f"k4_p{pattern_key.replace(',', '-')}_c{chunk_index:06d}"
                    ),
                    "sample_counts": {pattern_key: sample_stop - sample_start},
                    "sample_start": sample_start,
                    "sample_stop": sample_stop,
                    "chunk_index": chunk_index,
                    "chunk_seed": _derived_chunk_seed(
                        seed,
                        role="k4_calibration",
                        length=4,
                        pattern=pattern,
                        chunk_index=chunk_index,
                    ),
                }
            )
    outputs = _run_tasks(
        tasks,
        maximum_workers,
        _compile_calibration_length,
        checkpoint_directory=checkpoint_directory,
        checkpoint_stage=(
            None if checkpoint_directory is None else "k4_calibration"
        ),
    )
    merged = _merge_stage_outputs(outputs, length_field="cluster_length")
    if len(merged) != 1 or merged[0]["cluster_length"] != 4:
        raise RuntimeError("K4 calibration output is incomplete.")
    calibration = merged[0]

    def form(order_pattern: Sequence[int]) -> dict[str, float]:
        result: dict[str, float] = {}
        for start in range(len(order_pattern) - 3):
            key = _pattern_key(order_pattern[start : start + 4])
            result[key] = result.get(key, 0.0) + 1.0
        return result

    def average(forms: Sequence[Mapping[str, float]]) -> dict[str, float]:
        result: dict[str, float] = {}
        weight = 1.0 / len(forms)
        for current in forms:
            for key, coefficient in current.items():
                result[key] = result.get(key, 0.0) + weight * coefficient
        return result

    comparisons: dict[str, Any] = {}
    primary: list[Mapping[str, Any]] = []
    all_metrics: list[Mapping[str, Any]] = []
    for raw_length, source_by_scope in transfer["holdout_comparisons"].items():
        length = int(raw_length)
        forms = {
            "zero_order2_condition": form((0,) * length),
            "exactly_one_order2_condition": average(
                [
                    form(tuple(2 if index == rare else 0 for index in range(length)))
                    for rare in range(length)
                ]
            ),
        }
        comparisons[raw_length] = {}
        for scope, current_form in forms.items():
            metric_payload = {}
            for metric in _METRICS:
                correction = math.fsum(
                    coefficient
                    * calibration["strata"][key]["metric_statistics"][metric]["mean"]
                    for key, coefficient in current_form.items()
                )
                correction_se = math.sqrt(
                    math.fsum(
                        (
                            coefficient
                            * calibration["strata"][key]["metric_statistics"][metric][
                                "standard_error"
                            ]
                        )
                        ** 2
                        for key, coefficient in current_form.items()
                    )
                )
                source = source_by_scope[scope]["metrics"][metric]
                result = _comparison(
                    float(source["prediction"]) + correction,
                    math.hypot(
                        float(source["prediction_standard_error"]), correction_se
                    ),
                    float(source["actual"]),
                    float(source["actual_standard_error"]),
                )
                result.update(
                    {
                        "uncorrected_prediction": float(source["prediction"]),
                        "k4_correction": float(correction),
                        "k4_correction_standard_error": float(correction_se),
                    }
                )
                metric_payload[metric] = result
                all_metrics.append(result)
                if metric == "rz_count":
                    primary.append(result)
            comparisons[raw_length][scope] = {
                "k4_form": dict(sorted(current_form.items())),
                "metrics": metric_payload,
            }

    def maximum(field: str, values: Sequence[Mapping[str, Any]]):
        present = [value[field] for value in values if value[field] is not None]
        return None if not present else float(max(present))

    payload: dict[str, Any] = {
        "schema_version": CONNECTED_CLUSTER_K4_CALIBRATION_SCHEMA_VERSION,
        "validation_method": CONNECTED_CLUSTER_K4_CALIBRATION_METHOD,
        "final_cost_evaluation_performed": False,
        "source_validation_fingerprint": transfer["validation_fingerprint"],
        "source_calibration_fingerprint": transfer["calibration_fingerprint"],
        "condition_fingerprint": transfer["condition_fingerprint"],
        "configuration": {
            "ld": ld,
            "short_step_time": short_step_time,
            "finite_taylor_order": 2,
            "sample_count_per_pattern": sample_count_per_pattern,
            "sampled_order_patterns": [list(pattern) for pattern in patterns],
            "seed": seed,
            "compiler": _compiler_payload(compiler),
            "sample_chunk_size": sample_chunk_size,
            "checkpoint_directory": (
                None if checkpoint_directory is None else str(checkpoint_directory)
            ),
            "persistent_cache_path": (
                None if persistent_cache_path is None else str(persistent_cache_path)
            ),
        },
        "k4_calibration": calibration,
        "holdout_comparisons": comparisons,
        "summary": {
            "primary_maximum_absolute_relative_error": maximum(
                "absolute_relative_error", primary
            ),
            "primary_maximum_absolute_z_score": maximum(
                "absolute_z_score", primary
            ),
            "primary_maximum_pointwise_normal_95_upper_diagnostic": maximum(
                "pointwise_normal_relative_95_upper_diagnostic", primary
            ),
            "all_metrics_maximum_absolute_relative_error": maximum(
                "absolute_relative_error", all_metrics
            ),
            "primary_point_tolerance_passed": all(
                value["absolute_relative_error"] is not None
                and value["absolute_relative_error"] <= 0.05
                for value in primary
            ),
        },
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "maximum_workers": maximum_workers,
            "calibration": calibration["performance"],
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    validate_connected_cluster_k4_calibration_payload(payload)
    return payload


def validate_connected_cluster_k4_calibration_payload(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != (
        CONNECTED_CLUSTER_K4_CALIBRATION_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported connected-cluster K4 calibration schema.")
    if payload.get("validation_method") != CONNECTED_CLUSTER_K4_CALIBRATION_METHOD:
        raise ValueError("Unsupported connected-cluster K4 calibration method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("A K4 calibration cannot contain a final cost evaluation.")
    calibration = payload.get("k4_calibration")
    if not isinstance(calibration, Mapping) or calibration.get("cluster_length") != 4:
        raise ValueError("K4 calibration strata are missing.")
    expected = {
        _pattern_key(pattern)
        for pattern in itertools.product(_ORDERS, repeat=4)
        if sum(order == 2 for order in pattern) <= 1
    }
    if set(calibration.get("strata", {})) != expected:
        raise ValueError("K4 calibration order patterns are incomplete.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("K4 calibration fingerprint is invalid.")
    without = dict(payload)
    without.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without):
        raise ValueError("K4 calibration fingerprint mismatch.")


def write_connected_cluster_k4_calibration(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_connected_cluster_k4_calibration_payload(payload)
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


def validate_operational_connected_cluster_cost_model(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    compiler: CompilerSettings,
    pilot_sample_count: int,
    minimum_production_sample_count: int,
    maximum_production_sample_count: int,
    prediction_relative_standard_error_target: float,
    allocation_safety_factor: float,
    holdout_zero_sample_count: int,
    holdout_single_rare_sample_count: int,
    seed: int,
    maximum_workers: int = 3,
    cache_maximum_entries: int = 32_768,
    persistent_cache_path: str | Path | None = None,
    maximum_circuit_size: int = 150_000,
    checkpoint_directory: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit paired connected coefficients and validate unused full circuits."""
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    reference_rte_steps = require_integer_count(
        reference_rte_steps, name="reference_rte_steps", minimum=1
    )
    pilot_sample_count = require_integer_count(
        pilot_sample_count, name="pilot_sample_count", minimum=2
    )
    minimum_production_sample_count = require_integer_count(
        minimum_production_sample_count,
        name="minimum_production_sample_count",
        minimum=2,
    )
    maximum_production_sample_count = require_integer_count(
        maximum_production_sample_count,
        name="maximum_production_sample_count",
        minimum=minimum_production_sample_count,
    )
    holdout_zero_sample_count = require_integer_count(
        holdout_zero_sample_count,
        name="holdout_zero_sample_count",
        minimum=2,
    )
    holdout_single_rare_sample_count = require_integer_count(
        holdout_single_rare_sample_count,
        name="holdout_single_rare_sample_count",
        minimum=2,
    )
    maximum_workers = require_integer_count(
        maximum_workers, name="maximum_workers", minimum=1
    )
    for name, value in (
        (
            "prediction_relative_standard_error_target",
            prediction_relative_standard_error_target,
        ),
        ("allocation_safety_factor", allocation_safety_factor),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(reference_delta_time) or reference_delta_time <= 0.0:
        raise ValueError("reference_delta_time must be finite and positive.")

    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian, partition, identity_policy="extract_identity_phase"
    )
    if preparation.is_deterministic_only:
        raise ValueError("Connected-cluster validation requires a non-empty tail.")
    short_step_time = float(reference_delta_time) / reference_rte_steps
    tau = preparation.exact_rte_lambda_r * short_step_time
    _config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=short_step_time,
        rte_steps=1,
        truncation_tolerance=_explicit_cutoff_tolerance(tau),
        finite_taylor_order=2,
        seed=seed,
    )
    order_probabilities = dict(
        zip(distribution.orders, distribution.order_probabilities, strict=True)
    )
    forms = _required_forms(order_probabilities)

    common_task = {
        "hamiltonian": hamiltonian,
        "ld": ld,
        "short_step_time": short_step_time,
        "compiler": compiler,
        "master_seed": seed,
        "cache_maximum_entries": cache_maximum_entries,
        "persistent_cache_path": (
            None if persistent_cache_path is None else str(persistent_cache_path)
        ),
        "maximum_circuit_size": maximum_circuit_size,
    }
    pilot_tasks = [
        {
            **common_task,
            "role": "pilot",
            "cluster_length": length,
            "sample_counts": {
                _pattern_key(pattern): pilot_sample_count
                for pattern in itertools.product(_ORDERS, repeat=length)
            },
            "pair_boundary_stratified": True,
            "exact_order0_single": False,
        }
        for length in (1, 2, 3)
    ]
    pilot_outputs = _run_tasks(
        pilot_tasks,
        maximum_workers,
        _compile_calibration_length,
        checkpoint_directory=checkpoint_directory,
        checkpoint_stage="pilot",
    )
    pilot = {item["cluster_length"]: item for item in pilot_outputs}
    production_counts, allocation = _allocate_production_samples(
        pilot,
        forms,
        relative_standard_error_target=prediction_relative_standard_error_target,
        minimum_samples=minimum_production_sample_count,
        maximum_samples=maximum_production_sample_count,
        safety_factor=allocation_safety_factor,
        exact_parameter_keys=("k1:0",),
    )
    production_tasks = [
        {
            **common_task,
            "role": "production",
            "cluster_length": length,
            "sample_counts": production_counts[length],
            "pair_boundary_stratified": True,
            "exact_order0_single": True,
        }
        for length in (1, 2, 3)
    ]
    production_outputs = _run_tasks(
        production_tasks,
        maximum_workers,
        _compile_calibration_length,
        checkpoint_directory=checkpoint_directory,
        checkpoint_stage="production",
    )
    production = {item["cluster_length"]: item for item in production_outputs}
    lookup = _statistics_lookup(production)

    holdout_tasks = [
        {
            **common_task,
            "sequence_length": length,
            "zero_sample_count": holdout_zero_sample_count,
            "single_rare_sample_count": holdout_single_rare_sample_count,
        }
        for length in (4, 6, 8)
    ]
    holdout_outputs = _run_tasks(
        holdout_tasks,
        maximum_workers,
        _compile_full_holdout_length,
        checkpoint_directory=checkpoint_directory,
        checkpoint_stage="holdout",
    )
    holdout = {item["sequence_length"]: item for item in holdout_outputs}

    hashes = {preparation.preparation_hash}
    partition_hashes = {preparation.partition_hash}
    for item in (*pilot_outputs, *production_outputs, *holdout_outputs):
        hashes.add(item["preparation_hash"])
        partition_hashes.add(item["partition_hash"])
    if len(hashes) != 1 or len(partition_hashes) != 1:
        raise RuntimeError("Operational validation worker preparations differ.")

    comparisons: dict[str, Any] = {}
    primary = []
    all_metrics = []
    for length in (4, 6, 8):
        strata = holdout[length]["strata"]
        zero_patterns = tuple(
            key for key, value in strata.items() if value["rare_order_count"] == 0
        )
        one_patterns = tuple(
            key for key, value in strata.items() if value["rare_order_count"] == 1
        )
        comparisons[str(length)] = {}
        for scope, patterns in (
            ("zero_order2_condition", zero_patterns),
            ("exactly_one_order2_condition", one_patterns),
        ):
            form = forms[f"L{length}:{'zero' if scope.startswith('zero') else 'one'}"]
            metric_results = {}
            for metric in _METRICS:
                prediction, prediction_se = _evaluate_form(form, lookup, metric)
                actual, actual_se = _aggregate_holdout(strata, patterns, metric)
                result = _comparison(
                    prediction, prediction_se, actual, actual_se
                )
                metric_results[metric] = result
                all_metrics.append(result)
                if metric == "rz_count":
                    primary.append(result)
            comparisons[str(length)][scope] = {
                "patterns": list(patterns),
                "prediction_form": dict(sorted(form.items())),
                "metrics": metric_results,
            }

    def max_present(field: str, values: Sequence[Mapping[str, Any]]):
        present = [value[field] for value in values if value[field] is not None]
        return None if not present else float(max(present))

    payload: dict[str, Any] = {
        "schema_version": CONNECTED_CLUSTER_SCHEMA_VERSION,
        "validation_method": CONNECTED_CLUSTER_METHOD,
        "final_cost_evaluation_performed": False,
        "acceptance_policy": {
            "primary_metric": "rz_count",
            "required_lengths": [4, 6, 8],
            "required_conditions": [
                "zero_order2_condition",
                "exactly_one_order2_condition",
            ],
            "relative_point_tolerance": 0.05,
            "pointwise_normal_relative_95_diagnostic_tolerance": 0.05,
            "prediction_relative_95_half_width_tolerance": 0.02,
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
            "distribution": distribution.to_dict(),
            "pilot_sample_count_per_pattern": pilot_sample_count,
            "holdout_zero_sample_count_per_length": holdout_zero_sample_count,
            "holdout_single_rare_sample_count_per_position": (
                holdout_single_rare_sample_count
            ),
            "seed": seed,
            "compiler": _compiler_payload(compiler),
            "checkpoint_directory": (
                None if checkpoint_directory is None else str(checkpoint_directory)
            ),
            "persistent_cache_path": (
                None if persistent_cache_path is None else str(persistent_cache_path)
            ),
        },
        "allocation": allocation,
        "pilot": {str(key): value for key, value in pilot.items()},
        "production": {str(key): value for key, value in production.items()},
        "holdout": {str(key): value for key, value in holdout.items()},
        "holdout_comparisons": comparisons,
        "summary": {
            "primary_maximum_absolute_relative_error": max_present(
                "absolute_relative_error", primary
            ),
            "primary_maximum_absolute_z_score": max_present(
                "absolute_z_score", primary
            ),
            "primary_maximum_pointwise_normal_95_upper_diagnostic": max_present(
                "pointwise_normal_relative_95_upper_diagnostic", primary
            ),
            "primary_maximum_prediction_relative_95_half_width": max_present(
                "prediction_relative_95_half_width", primary
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
            "primary_prediction_precision_passed": all(
                value["prediction_relative_95_half_width"] is not None
                and value["prediction_relative_95_half_width"] <= 0.02
                for value in primary
            ),
            "all_metrics_maximum_absolute_relative_error": max_present(
                "absolute_relative_error", all_metrics
            ),
            "all_metrics_maximum_pointwise_normal_95_upper_diagnostic": max_present(
                "pointwise_normal_relative_95_upper_diagnostic", all_metrics
            ),
            "allocation_cap_was_reached": bool(
                allocation["capped_parameter_keys"]
            ),
        },
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "maximum_workers": maximum_workers,
            "pilot_worker_seconds": {
                f"k{item['cluster_length']}": item["performance"]
                for item in pilot_outputs
            },
            "production_worker_seconds": {
                f"k{item['cluster_length']}": item["performance"]
                for item in production_outputs
            },
            "holdout_worker_seconds": {
                f"L{item['sequence_length']}": item["performance"]
                for item in holdout_outputs
            },
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def supplement_connected_cluster_holdout_precision(
    source_payload: Mapping[str, Any],
    hamiltonian: DFHamiltonian,
    *,
    compiler: CompilerSettings,
    additional_zero_sample_count: int,
    additional_single_rare_sample_count: int,
    seed: int,
    maximum_workers: int = 3,
    cache_maximum_entries: int = 32_768,
    maximum_circuit_size: int = 150_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add an independent holdout replication while keeping fitted coefficients fixed."""
    started = time.perf_counter()
    validate_connected_cluster_payload(source_payload)
    additional_zero_sample_count = require_integer_count(
        additional_zero_sample_count,
        name="additional_zero_sample_count",
        minimum=2,
    )
    additional_single_rare_sample_count = require_integer_count(
        additional_single_rare_sample_count,
        name="additional_single_rare_sample_count",
        minimum=2,
    )
    maximum_workers = require_integer_count(
        maximum_workers, name="maximum_workers", minimum=1
    )
    configuration = source_payload["configuration"]
    if int(seed) == int(configuration["seed"]):
        raise ValueError("Supplement seed must differ from the source validation seed.")
    if _compiler_payload(compiler) != configuration["compiler"]:
        raise ValueError("Supplement compiler settings differ from the source artifact.")

    ld = int(configuration["ld"])
    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian, partition, identity_policy="extract_identity_phase"
    )
    source_hamiltonian = source_payload["hamiltonian"]
    if hamiltonian.n_qubits != int(source_hamiltonian["n_qubits"]):
        raise ValueError("Supplement Hamiltonian qubit count differs from the source.")
    if hamiltonian.n_blocks != int(source_hamiltonian["df_rank"]):
        raise ValueError("Supplement DF rank differs from the source.")
    source_metadata = source_hamiltonian["metadata"]
    for key, source_value in source_metadata.items():
        rebuilt_value = hamiltonian.metadata.get(key)
        if isinstance(source_value, (int, float)) and not isinstance(source_value, bool):
            if rebuilt_value is None or not math.isclose(
                float(rebuilt_value),
                float(source_value),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Supplement Hamiltonian metadata differs at {key}."
                )
        elif rebuilt_value != source_value:
            raise ValueError(f"Supplement Hamiltonian metadata differs at {key}.")
    short_step_time = float(configuration["short_step_time"])
    tau = preparation.exact_rte_lambda_r * short_step_time
    _rebuilt_config, rebuilt_distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=short_step_time,
        rte_steps=1,
        truncation_tolerance=_explicit_cutoff_tolerance(tau),
        finite_taylor_order=2,
        seed=int(seed),
    )
    source_distribution = configuration["distribution"]
    if tuple(source_distribution["orders"]) != rebuilt_distribution.orders:
        raise ValueError("Supplement Taylor orders differ from the source.")
    distribution_checks = {
        "dimensionless_step_time": rebuilt_distribution.dimensionless_step_time,
        "exact_finite_distribution": rebuilt_distribution.exact_finite_distribution,
        "paper_upper_bound": rebuilt_distribution.paper_upper_bound,
        "truncation_residual_bound": rebuilt_distribution.truncation_residual_bound,
    }
    for key, rebuilt_value in distribution_checks.items():
        if not math.isclose(
            float(source_distribution[key]),
            float(rebuilt_value),
            rel_tol=1e-10,
            abs_tol=1e-14,
        ):
            raise ValueError(f"Supplement RTE distribution differs at {key}.")
    if not all(
        math.isclose(float(first), float(second), rel_tol=1e-10, abs_tol=1e-14)
        for first, second in zip(
            source_distribution["order_probabilities"],
            rebuilt_distribution.order_probabilities,
            strict=True,
        )
    ):
        raise ValueError("Supplement RTE order probabilities differ from the source.")
    rebuild_identity = {
        "status": (
            "exact_hash_match"
            if preparation.preparation_hash
            == source_hamiltonian["preparation_hash"]
            else "numerically_equivalent_rebuild_with_hash_mismatch"
        ),
        "source_preparation_hash": source_hamiltonian["preparation_hash"],
        "rebuilt_preparation_hash": preparation.preparation_hash,
        "source_partition_hash": source_hamiltonian["partition_hash"],
        "rebuilt_partition_hash": preparation.partition_hash,
        "metadata_relative_tolerance": 1e-10,
        "metadata_absolute_tolerance": 1e-12,
        "distribution_relative_tolerance": 1e-10,
        "distribution_absolute_tolerance": 1e-14,
        "independent_holdout_replication_check_required": True,
    }

    common_task = {
        "hamiltonian": hamiltonian,
        "ld": ld,
        "short_step_time": short_step_time,
        "compiler": compiler,
        "master_seed": int(seed),
        "cache_maximum_entries": cache_maximum_entries,
        "maximum_circuit_size": maximum_circuit_size,
    }
    tasks = [
        {
            **common_task,
            "sequence_length": length,
            "zero_sample_count": additional_zero_sample_count,
            "single_rare_sample_count": additional_single_rare_sample_count,
        }
        for length in (4, 6, 8)
    ]
    outputs = _run_tasks(
        tasks, maximum_workers, _compile_full_holdout_length
    )
    additional_holdout = {item["sequence_length"]: item for item in outputs}

    combined_holdout: dict[int, Any] = {}
    source_seeds: set[int] = set()
    additional_seeds: set[int] = set()
    for length in (4, 6, 8):
        source_stage = source_payload["holdout"][str(length)]
        added_stage = additional_holdout[length]
        if added_stage["preparation_hash"] != preparation.preparation_hash:
            raise RuntimeError("Supplement worker preparation hashes differ.")
        combined_strata = {}
        if set(source_stage["strata"]) != set(added_stage["strata"]):
            raise RuntimeError("Supplement holdout patterns differ from the source.")
        for key, source_stratum in source_stage["strata"].items():
            added_stratum = added_stage["strata"][key]
            first_count = int(source_stratum["sample_count"])
            second_count = int(added_stratum["sample_count"])
            source_seeds.add(int(source_stratum["seed"]))
            additional_seeds.add(int(added_stratum["seed"]))
            combined_strata[key] = {
                "order_pattern": list(source_stratum["order_pattern"]),
                "rare_order_count": int(source_stratum["rare_order_count"]),
                "sample_count": first_count + second_count,
                "source_sample_count": first_count,
                "additional_sample_count": second_count,
                "source_seed": int(source_stratum["seed"]),
                "additional_seed": int(added_stratum["seed"]),
                "metric_statistics": {
                    metric: _merge_sample_statistics(
                        source_stratum["metric_statistics"][metric],
                        first_count,
                        added_stratum["metric_statistics"][metric],
                        second_count,
                    )
                    for metric in _METRICS
                },
            }
        combined_holdout[length] = {
            "sequence_length": length,
            "preparation_hash": added_stage["preparation_hash"],
            "partition_hash": added_stage["partition_hash"],
            "distribution": added_stage["distribution"],
            "strata": combined_strata,
        }
    if source_seeds & additional_seeds:
        raise RuntimeError("Supplement holdout seeds overlap the source holdout seeds.")

    replication_comparisons: dict[str, Any] = {}
    replication_results = []
    for length in (4, 6, 8):
        source_strata = source_payload["holdout"][str(length)]["strata"]
        added_strata = additional_holdout[length]["strata"]
        replication_comparisons[str(length)] = {}
        for scope, rare_count in (
            ("zero_order2_condition", 0),
            ("exactly_one_order2_condition", 1),
        ):
            patterns = tuple(
                key
                for key, value in source_strata.items()
                if value["rare_order_count"] == rare_count
            )
            metric_results = {}
            for metric in _METRICS:
                source_mean, source_se = _aggregate_holdout(
                    source_strata, patterns, metric
                )
                added_mean, added_se = _aggregate_holdout(
                    added_strata, patterns, metric
                )
                result = _comparison(source_mean, source_se, added_mean, added_se)
                metric_results[metric] = result
                replication_results.append(result)
            replication_comparisons[str(length)][scope] = {
                "patterns": list(patterns),
                "metrics": metric_results,
            }

    distribution = configuration["distribution"]
    order_probabilities = dict(
        zip(distribution["orders"], distribution["order_probabilities"], strict=True)
    )
    forms = _required_forms(order_probabilities)
    lookup = _statistics_lookup(
        {int(key): value for key, value in source_payload["production"].items()}
    )
    comparisons: dict[str, Any] = {}
    primary = []
    all_metrics = []
    for length in (4, 6, 8):
        strata = combined_holdout[length]["strata"]
        zero_patterns = tuple(
            key for key, value in strata.items() if value["rare_order_count"] == 0
        )
        one_patterns = tuple(
            key for key, value in strata.items() if value["rare_order_count"] == 1
        )
        comparisons[str(length)] = {}
        for scope, patterns in (
            ("zero_order2_condition", zero_patterns),
            ("exactly_one_order2_condition", one_patterns),
        ):
            form_key = "zero" if scope.startswith("zero") else "one"
            form = forms[f"L{length}:{form_key}"]
            metric_results = {}
            for metric in _METRICS:
                prediction, prediction_se = _evaluate_form(form, lookup, metric)
                actual, actual_se = _aggregate_holdout(strata, patterns, metric)
                result = _comparison(prediction, prediction_se, actual, actual_se)
                metric_results[metric] = result
                all_metrics.append(result)
                if metric == "rz_count":
                    primary.append(result)
            comparisons[str(length)][scope] = {
                "patterns": list(patterns),
                "prediction_form": dict(sorted(form.items())),
                "metrics": metric_results,
            }

    def max_present(field: str, values: Sequence[Mapping[str, Any]]):
        present = [value[field] for value in values if value[field] is not None]
        return None if not present else float(max(present))

    summary = {
        "primary_maximum_absolute_relative_error": max_present(
            "absolute_relative_error", primary
        ),
        "primary_maximum_absolute_z_score": max_present(
            "absolute_z_score", primary
        ),
        "primary_maximum_pointwise_normal_95_upper_diagnostic": max_present(
            "pointwise_normal_relative_95_upper_diagnostic", primary
        ),
        "primary_maximum_prediction_relative_95_half_width": max_present(
            "prediction_relative_95_half_width", primary
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
        "primary_prediction_precision_passed": all(
            value["prediction_relative_95_half_width"] is not None
            and value["prediction_relative_95_half_width"] <= 0.02
            for value in primary
        ),
        "all_metrics_maximum_absolute_relative_error": max_present(
            "absolute_relative_error", all_metrics
        ),
        "all_metrics_maximum_pointwise_normal_95_upper_diagnostic": max_present(
            "pointwise_normal_relative_95_upper_diagnostic", all_metrics
        ),
        "replication_maximum_absolute_relative_difference": max_present(
            "absolute_relative_error", replication_results
        ),
        "replication_maximum_absolute_z_score": max_present(
            "absolute_z_score", replication_results
        ),
        "replication_consistency_z3_passed": all(
            value["absolute_z_score"] is not None
            and value["absolute_z_score"] <= 3.0
            for value in replication_results
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": CONNECTED_CLUSTER_SUPPLEMENT_SCHEMA_VERSION,
        "validation_method": CONNECTED_CLUSTER_SUPPLEMENT_METHOD,
        "final_cost_evaluation_performed": False,
        "source_validation_fingerprint": source_payload["validation_fingerprint"],
        "source_configuration": configuration,
        "source_hamiltonian": source_payload["hamiltonian"],
        "rebuild_identity": rebuild_identity,
        "configuration": {
            "additional_zero_sample_count_per_length": additional_zero_sample_count,
            "additional_single_rare_sample_count_per_position": (
                additional_single_rare_sample_count
            ),
            "seed": int(seed),
            "compiler": _compiler_payload(compiler),
        },
        "additional_holdout": {
            str(key): value for key, value in additional_holdout.items()
        },
        "combined_holdout": {
            str(key): value for key, value in combined_holdout.items()
        },
        "holdout_comparisons": comparisons,
        "independent_holdout_replication_comparisons": replication_comparisons,
        "summary": summary,
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "maximum_workers": maximum_workers,
            "additional_holdout_worker_seconds": {
                f"L{item['sequence_length']}": item["performance"]
                for item in outputs
            },
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_connected_cluster_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CONNECTED_CLUSTER_SCHEMA_VERSION:
        raise ValueError("Unsupported connected-cluster schema.")
    if payload.get("validation_method") != CONNECTED_CLUSTER_METHOD:
        raise ValueError("Unsupported connected-cluster method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")
    if set(payload.get("pilot", {})) != {"1", "2", "3"}:
        raise ValueError("Pilot connected-cluster lengths are incomplete.")
    if set(payload.get("production", {})) != {"1", "2", "3"}:
        raise ValueError("Production connected-cluster lengths are incomplete.")
    if set(payload.get("holdout", {})) != {"4", "6", "8"}:
        raise ValueError("Full-circuit holdout lengths are incomplete.")
    hashes = {
        item["preparation_hash"]
        for role in ("pilot", "production", "holdout")
        for item in payload[role].values()
    }
    hashes.add(payload.get("hamiltonian", {}).get("preparation_hash"))
    if len(hashes) != 1:
        raise ValueError("Connected-cluster preparation hashes differ.")
    seeds = [
        stratum["seed"]
        for role in ("pilot", "production", "holdout")
        for item in payload[role].values()
        for stratum in item["strata"].values()
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Connected-cluster seeds must be unique.")
    fingerprint = payload.get("validation_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Connected-cluster fingerprint mismatch.")


def write_connected_cluster_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_connected_cluster_payload(payload)
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


def validate_connected_cluster_supplement_payload(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != CONNECTED_CLUSTER_SUPPLEMENT_SCHEMA_VERSION:
        raise ValueError("Unsupported connected-cluster supplement schema.")
    if payload.get("validation_method") != CONNECTED_CLUSTER_SUPPLEMENT_METHOD:
        raise ValueError("Unsupported connected-cluster supplement method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")
    if set(payload.get("additional_holdout", {})) != {"4", "6", "8"}:
        raise ValueError("Supplement holdout lengths are incomplete.")
    if set(payload.get("combined_holdout", {})) != {"4", "6", "8"}:
        raise ValueError("Combined supplement lengths are incomplete.")
    source_seeds = set()
    additional_seeds = set()
    for stage in payload["combined_holdout"].values():
        for stratum in stage["strata"].values():
            source_seeds.add(int(stratum["source_seed"]))
            additional_seeds.add(int(stratum["additional_seed"]))
    if source_seeds & additional_seeds:
        raise ValueError("Supplement and source holdout seeds overlap.")
    fingerprint = payload.get("validation_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Connected-cluster supplement fingerprint mismatch.")


def write_connected_cluster_supplement_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_connected_cluster_supplement_payload(payload)
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

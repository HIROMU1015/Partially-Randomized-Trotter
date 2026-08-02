"""Level-5 repeated-step compiled expectations for short DF trajectories."""

from __future__ import annotations

import hashlib
import itertools
import math
import platform
import random
from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias

from .df_partial_s2 import (
    DFPartialS2Preparation,
    QiskitDFPartialS2CircuitBuilder,
    estimate_df_partial_s2_untranspiled_size_upper_bound,
)
from .df_partial_s2_cost import _request_for_events, _validate_rte_inputs
from .df_partial_s2_repeated import (
    DFPartialS2AttenuationMetadata,
    DFPartialS2RepeatedRequest,
    DFPartialS2TruncationMetadata,
    QiskitDFPartialS2RepeatedCircuitBuilder,
    RepeatedCircuitConstructionPolicy,
)
from .rte import (
    CircuitCost,
    CompilerSettings,
    PROBABILITY_ATOL,
    RTEConfig,
    RTEEvent,
    RTEFiniteDistribution,
    enumerate_rte_events,
    require_integer_count,
)
from .rte_compiled_cost import (
    CompiledMetricAccumulator,
    CompiledMetricStatistics,
    TranspiledCircuitCost,
    TranspiledCircuitCostCache,
    circuit_cost_from_metric_statistics,
    subtract_compiled_costs,
    sum_compiled_costs,
)


RepeatedPartialS2EstimateKind: TypeAlias = Literal[
    "exact_compiled_repeated_partial_s2_expectation",
    "monte_carlo_compiled_repeated_partial_s2_expectation",
]
RepeatedCostEvaluationMode: TypeAlias = Literal[
    "full_diagnostics",
    "selected_only",
]


@dataclass(frozen=True)
class CompiledRepeatedPartialS2CostEstimate:
    """Full repeated transpilation and matched additive diagnostics."""

    estimate_kind: RepeatedPartialS2EstimateKind
    evaluation_mode: RepeatedCostEvaluationMode
    expected_cost: CircuitCost
    standard_error: CircuitCost | None
    raw_concatenation_expected_cost: CircuitCost | None
    boundary_optimized_expected_cost: CircuitCost | None
    boundary_optimization_difference: CircuitCost | None
    matched_per_step_expected_cost: CircuitCost | None
    matched_per_step_standard_error: CircuitCost | None
    cross_step_nonadditive_difference: CircuitCost | None
    cross_step_difference_standard_error: CircuitCost | None
    primitive_additive_expected_cost: CircuitCost | None
    full_metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    raw_metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...] | None
    boundary_optimized_metric_statistics: tuple[
        tuple[str, CompiledMetricStatistics], ...
    ] | None
    boundary_difference_metric_statistics: tuple[
        tuple[str, CompiledMetricStatistics], ...
    ] | None
    matched_per_step_metric_statistics: tuple[
        tuple[str, CompiledMetricStatistics], ...
    ] | None
    cross_step_difference_metric_statistics: tuple[
        tuple[str, CompiledMetricStatistics], ...
    ] | None
    primitive_additive_metric_statistics: tuple[
        tuple[str, CompiledMetricStatistics], ...
    ] | None
    sample_count: int | None
    enumerated_trajectory_count: int | None
    trajectory_probability_sum: float | None
    normalized_trajectory_probability_sum: float | None
    probability_atol: float
    single_event_space_size: int
    single_step_event_sequence_count: int
    trajectory_space_size: int
    repetition_count: int
    rte_steps_per_repetition: int
    master_seed: int | None
    sampled_trajectory_seeds: tuple[int, ...] | None
    trajectory_seed_policy: str
    provenance_fingerprints: tuple[str, ...]
    circuit_semantics_fingerprints: tuple[str, ...]
    unique_trajectory_circuit_count: int
    unique_raw_trajectory_circuit_count: int | None
    unique_boundary_optimized_trajectory_circuit_count: int | None
    unique_compiled_circuit_count: int
    transpile_cache_hit_count: int
    transpile_cache_miss_count: int
    transpile_cache_bypass_count: int
    transpile_cache_eviction_count: int
    transpile_cache_maximum_entries: int
    processed_trajectory_count: int
    provenance_rolling_digest: str
    circuit_semantics_rolling_digest: str
    sampled_trajectory_seed_digest: str | None
    master_prng_type: str | None
    event_prng_type: str | None
    python_version: str
    numpy_version: str | None
    sampling_convention_version: str | None
    compiler: CompilerSettings
    controlled: bool
    ancilla_qubit: int | None
    boundary_optimization_policy: RepeatedCircuitConstructionPolicy
    maximum_trajectories: int | None
    maximum_untranspiled_circuit_size: int
    attenuation: DFPartialS2AttenuationMetadata
    truncation: DFPartialS2TruncationMetadata
    circuit_granularity: Literal["repeated_partial_s2_steps"] = (
        "repeated_partial_s2_steps"
    )
    fidelity_level: Literal[5] = 5
    statistics_policy: Literal["online_welford_v1"] = "online_welford_v1"


def _as_cost(
    statistics: tuple[tuple[str, CompiledMetricStatistics], ...],
    compiler: CompilerSettings,
    estimate_kind: str,
    *,
    standard_error: bool = False,
) -> CircuitCost | None:
    return circuit_cost_from_metric_statistics(
        statistics,
        compiler=compiler,
        estimate_kind=estimate_kind,
        use_standard_error=standard_error,
        fidelity_level=5,
    )


def _compile_repeated_samples(
    requests: Sequence[DFPartialS2RepeatedRequest],
    compiler: CompilerSettings,
    *,
    estimate_kind: RepeatedPartialS2EstimateKind,
    evaluation_mode: RepeatedCostEvaluationMode,
    weights: Sequence[float] | None,
    trajectory_probability_sum: float | None,
    master_seed: int | None,
    sampled_trajectory_seeds: tuple[int, ...] | None,
    single_event_space_size: int,
    single_step_event_sequence_count: int,
    trajectory_space_size: int,
    maximum_trajectories: int | None,
    maximum_untranspiled_circuit_size: int,
    cache: TranspiledCircuitCostCache | None,
    backend: Any | None,
) -> CompiledRepeatedPartialS2CostEstimate:
    if not requests:
        raise ValueError("At least one repeated trajectory request is required.")
    if evaluation_mode not in ("full_diagnostics", "selected_only"):
        raise ValueError("Unsupported repeated compiled-cost evaluation mode.")
    working_cache = cache if cache is not None else TranspiledCircuitCostCache()
    initial_misses = working_cache.miss_count
    initial_bypasses = working_cache.bypass_count
    initial_evictions = working_cache.eviction_count
    repeated_builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    step_builder = QiskitDFPartialS2CircuitBuilder()
    if weights is not None and len(weights) != len(requests):
        raise ValueError("Trajectory weights length must match requests.")
    weighted = weights is not None
    selected_accumulator = CompiledMetricAccumulator(weighted=weighted)
    raw_accumulator = (
        CompiledMetricAccumulator(weighted=weighted)
        if evaluation_mode == "full_diagnostics"
        else None
    )
    optimized_accumulator = (
        CompiledMetricAccumulator(weighted=weighted)
        if evaluation_mode == "full_diagnostics"
        else None
    )
    boundary_accumulator = (
        CompiledMetricAccumulator(weighted=weighted)
        if evaluation_mode == "full_diagnostics"
        else None
    )
    matched_accumulator = (
        CompiledMetricAccumulator(weighted=weighted)
        if evaluation_mode == "full_diagnostics"
        else None
    )
    cross_accumulator = (
        CompiledMetricAccumulator(weighted=weighted)
        if evaluation_mode == "full_diagnostics"
        else None
    )
    primitive_accumulator = (
        CompiledMetricAccumulator(weighted=weighted)
        if evaluation_mode == "full_diagnostics"
        else None
    )
    all_keys: set[str] = set()
    selected_circuit_fingerprints: set[str] = set()
    raw_circuit_fingerprints: set[str] = set()
    optimized_circuit_fingerprints: set[str] = set()
    ordered_provenance_fingerprints: list[str] = []
    ordered_semantics_fingerprints: list[str] = []
    provenance_digest = hashlib.sha256()
    semantics_digest = hashlib.sha256()
    cache_hits = 0
    first_selected_result = None

    def compile_circuit(circuit, fingerprint):
        nonlocal cache_hits
        cost, key, cached = working_cache.get_or_transpile(
            circuit,
            compiler,
            circuit_fingerprint=fingerprint,
            backend=backend,
        )
        all_keys.add(key)
        cache_hits += int(cached)
        return cost

    def require_size_limit(result) -> None:
        if result.untranspiled_circuit_size > maximum_untranspiled_circuit_size:
            raise ValueError(
                "Untranspiled repeated partial-S2 circuit exceeds the "
                "configured size limit."
            )

    for request_index, request in enumerate(requests):
        weight = None if weights is None else weights[request_index]
        planned_size = sum(
            estimate_df_partial_s2_untranspiled_size_upper_bound(step_request)
            for step_request in request.iter_step_requests()
        )
        if planned_size > maximum_untranspiled_circuit_size:
            raise ValueError(
                "Planned repeated partial-S2 circuit upper bound exceeds the "
                "configured size limit before circuit construction."
            )
        if evaluation_mode == "selected_only":
            selected_result = repeated_builder.build(
                request,
                construction_policy=request.construction_policy,
            )
            require_size_limit(selected_result)
            selected_cost = compile_circuit(
                selected_result.circuit,
                selected_result.circuit_semantics_fingerprint,
            )
        else:
            raw_result = repeated_builder.build(
                request,
                construction_policy="raw_concatenation",
            )
            optimized_result = repeated_builder.build(
                request,
                construction_policy="boundary_optimized",
            )
            require_size_limit(raw_result)
            require_size_limit(optimized_result)
            raw_cost = compile_circuit(
                raw_result.circuit,
                raw_result.circuit_semantics_fingerprint,
            )
            optimized_cost = compile_circuit(
                optimized_result.circuit,
                optimized_result.circuit_semantics_fingerprint,
            )
            raw_accumulator.update(raw_cost, weight=weight)
            optimized_accumulator.update(optimized_cost, weight=weight)
            raw_circuit_fingerprints.add(raw_result.circuit_semantics_fingerprint)
            optimized_circuit_fingerprints.add(
                optimized_result.circuit_semantics_fingerprint
            )
            boundary_accumulator.update(
                subtract_compiled_costs(optimized_cost, raw_cost),
                weight=weight,
            )
            if request.construction_policy == "raw_concatenation":
                selected_cost = raw_cost
                selected_result = raw_result
            else:
                selected_cost = optimized_cost
                selected_result = optimized_result

        selected_accumulator.update(selected_cost, weight=weight)
        selected_circuit_fingerprints.add(
            selected_result.circuit_semantics_fingerprint
        )
        ordered_provenance_fingerprints.append(
            selected_result.provenance_fingerprint
        )
        ordered_semantics_fingerprints.append(
            selected_result.circuit_semantics_fingerprint
        )
        provenance_digest.update(
            bytes.fromhex(selected_result.provenance_fingerprint)
        )
        semantics_digest.update(
            bytes.fromhex(selected_result.circuit_semantics_fingerprint)
        )
        if first_selected_result is None:
            first_selected_result = selected_result

        if evaluation_mode == "selected_only":
            continue

        per_step_costs: list[TranspiledCircuitCost] = []
        primitive_costs: list[TranspiledCircuitCost] = []
        for step_request in request.iter_step_requests():
            step_result = step_builder.build_step(step_request)
            if step_result.untranspiled_circuit_size > (
                maximum_untranspiled_circuit_size
            ):
                raise ValueError(
                    "Untranspiled partial-S2 step exceeds the configured size limit."
                )
            step_cost = compile_circuit(
                step_result.circuit,
                step_result.compiler_independent_fingerprint,
            )
            per_step_costs.append(step_cost)

            parts = step_builder.build_additive_circuits(step_request)
            for part, fingerprint in (
                (parts.forward_deterministic_half, parts.forward_fingerprint),
                (parts.rte_occurrence, parts.rte_fingerprint),
                (parts.reverse_deterministic_half, parts.reverse_fingerprint),
            ):
                if part.size() > maximum_untranspiled_circuit_size:
                    raise ValueError(
                        "Untranspiled primitive circuit exceeds the configured "
                        "size limit."
                    )
                part_cost = compile_circuit(part, fingerprint)
                primitive_costs.append(part_cost)

        matched = sum_compiled_costs(per_step_costs)
        primitive_additive = sum_compiled_costs(primitive_costs)
        matched_accumulator.update(matched, weight=weight)
        primitive_accumulator.update(primitive_additive, weight=weight)
        cross_accumulator.update(
            subtract_compiled_costs(selected_cost, matched),
            weight=weight,
        )

    if first_selected_result is None:
        raise RuntimeError("Repeated partial-S2 metadata could not be constructed.")
    selected_statistics = selected_accumulator.finalize()
    raw_statistics = (
        raw_accumulator.finalize()
        if raw_accumulator is not None
        else None
    )
    optimized_statistics = (
        optimized_accumulator.finalize()
        if optimized_accumulator is not None
        else None
    )
    boundary_statistics = (
        boundary_accumulator.finalize()
        if boundary_accumulator is not None
        else None
    )
    matched_statistics = (
        matched_accumulator.finalize()
        if matched_accumulator is not None
        else None
    )
    cross_statistics = (
        cross_accumulator.finalize()
        if cross_accumulator is not None
        else None
    )
    primitive_statistics = (
        primitive_accumulator.finalize()
        if primitive_accumulator is not None
        else None
    )
    expected = _as_cost(selected_statistics, compiler, estimate_kind)
    raw_expected = (
        _as_cost(
            raw_statistics,
            compiler,
            "compiled_repeated_partial_s2_raw_concatenation",
        )
        if raw_statistics is not None
        else None
    )
    optimized_expected = (
        _as_cost(
            optimized_statistics,
            compiler,
            "compiled_repeated_partial_s2_boundary_optimized",
        )
        if optimized_statistics is not None
        else None
    )
    boundary_difference = (
        _as_cost(
            boundary_statistics,
            compiler,
            "compiled_repeated_partial_s2_boundary_difference",
        )
        if boundary_statistics is not None
        else None
    )
    matched_expected = (
        _as_cost(
            matched_statistics,
            compiler,
            "compiled_repeated_partial_s2_matched_step_sum",
        )
        if matched_statistics is not None
        else None
    )
    cross_difference = (
        _as_cost(
            cross_statistics,
            compiler,
            "compiled_repeated_partial_s2_cross_step_difference",
        )
        if cross_statistics is not None
        else None
    )
    primitive_expected = (
        _as_cost(
            primitive_statistics,
            compiler,
            "compiled_repeated_partial_s2_primitive_additive_sum",
        )
        if primitive_statistics is not None
        else None
    )
    if expected is None:
        raise RuntimeError("Repeated compiled expectation could not be constructed.")
    if evaluation_mode == "full_diagnostics" and any(
        item is None
        for item in (
            raw_expected,
            optimized_expected,
            boundary_difference,
            matched_expected,
            cross_difference,
            primitive_expected,
        )
    ):
        raise RuntimeError("Repeated compiled diagnostics could not be constructed.")
    exact = estimate_kind == "exact_compiled_repeated_partial_s2_expectation"
    first_request = requests[0]
    return CompiledRepeatedPartialS2CostEstimate(
        estimate_kind=estimate_kind,
        evaluation_mode=evaluation_mode,
        expected_cost=expected,
        standard_error=_as_cost(
            selected_statistics,
            compiler,
            estimate_kind,
            standard_error=True,
        ),
        raw_concatenation_expected_cost=raw_expected,
        boundary_optimized_expected_cost=optimized_expected,
        boundary_optimization_difference=boundary_difference,
        matched_per_step_expected_cost=matched_expected,
        matched_per_step_standard_error=(
            _as_cost(
                matched_statistics,
                compiler,
                "compiled_repeated_partial_s2_matched_step_sum",
                standard_error=True,
            )
            if matched_statistics is not None
            else None
        ),
        cross_step_nonadditive_difference=cross_difference,
        cross_step_difference_standard_error=(
            _as_cost(
                cross_statistics,
                compiler,
                "compiled_repeated_partial_s2_cross_step_difference",
                standard_error=True,
            )
            if cross_statistics is not None
            else None
        ),
        primitive_additive_expected_cost=primitive_expected,
        full_metric_statistics=selected_statistics,
        raw_metric_statistics=raw_statistics,
        boundary_optimized_metric_statistics=optimized_statistics,
        boundary_difference_metric_statistics=boundary_statistics,
        matched_per_step_metric_statistics=matched_statistics,
        cross_step_difference_metric_statistics=cross_statistics,
        primitive_additive_metric_statistics=primitive_statistics,
        sample_count=None if exact else len(requests),
        enumerated_trajectory_count=len(requests) if exact else None,
        trajectory_probability_sum=trajectory_probability_sum,
        normalized_trajectory_probability_sum=(
            math.fsum(weights) if weights is not None else None
        ),
        probability_atol=PROBABILITY_ATOL,
        single_event_space_size=single_event_space_size,
        single_step_event_sequence_count=single_step_event_sequence_count,
        trajectory_space_size=trajectory_space_size,
        repetition_count=first_request.repetition_count,
        rte_steps_per_repetition=(
            0 if first_request.rte_config is None else first_request.rte_config.rte_steps
        ),
        master_seed=master_seed,
        sampled_trajectory_seeds=sampled_trajectory_seeds,
        trajectory_seed_policy=(
            "not_applicable_exact_step_major_enumeration"
            if exact
            else (
                "master Random(seed) draws trajectory seeds; each trajectory seed "
                "draws q independent step seeds (q=1 reuses the trajectory seed)"
            )
        ),
        provenance_fingerprints=tuple(ordered_provenance_fingerprints),
        circuit_semantics_fingerprints=tuple(ordered_semantics_fingerprints),
        unique_trajectory_circuit_count=len(selected_circuit_fingerprints),
        unique_raw_trajectory_circuit_count=(
            len(raw_circuit_fingerprints)
            if evaluation_mode == "full_diagnostics"
            else None
        ),
        unique_boundary_optimized_trajectory_circuit_count=(
            len(optimized_circuit_fingerprints)
            if evaluation_mode == "full_diagnostics"
            else None
        ),
        unique_compiled_circuit_count=len(all_keys),
        transpile_cache_hit_count=cache_hits,
        transpile_cache_miss_count=working_cache.miss_count - initial_misses,
        transpile_cache_bypass_count=(
            working_cache.bypass_count - initial_bypasses
        ),
        transpile_cache_eviction_count=(
            working_cache.eviction_count - initial_evictions
        ),
        transpile_cache_maximum_entries=working_cache.maximum_entries,
        processed_trajectory_count=len(requests),
        provenance_rolling_digest=provenance_digest.hexdigest(),
        circuit_semantics_rolling_digest=semantics_digest.hexdigest(),
        sampled_trajectory_seed_digest=(
            None
            if sampled_trajectory_seeds is None
            else hashlib.sha256(
                b"".join(
                    int(value).to_bytes(8, byteorder="big", signed=False)
                    for value in sampled_trajectory_seeds
                )
            ).hexdigest()
        ),
        master_prng_type=(
            None if exact else "python.random.Random(MT19937)"
        ),
        event_prng_type=(
            None
            if first_request.rte_config is None
            else first_request.rte_config.prng_type
        ),
        python_version=platform.python_version(),
        numpy_version=(
            None
            if first_request.rte_config is None
            else first_request.rte_config.numpy_version
        ),
        sampling_convention_version=(
            None
            if first_request.rte_config is None
            else first_request.rte_config.sampling_convention_version
        ),
        compiler=compiler,
        controlled=first_request.controlled,
        ancilla_qubit=first_request.ancilla_qubit,
        boundary_optimization_policy=first_request.construction_policy,
        maximum_trajectories=maximum_trajectories,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        attenuation=first_selected_result.attenuation,
        truncation=first_selected_result.truncation,
    )


def _request_from_step_event_sequences(
    preparation: DFPartialS2Preparation,
    *,
    step_time: float,
    repetition_count: int,
    config: RTEConfig | None,
    distribution: RTEFiniteDistribution | None,
    step_event_sequences: Sequence[tuple[RTEEvent, ...]],
    step_seeds: Sequence[int | None],
    master_seed: int | None,
    trajectory_seed: int | None,
    sampling_policy: str,
    controlled: bool,
    ancilla_qubit: int | None,
    cancel_adjacent_equal_bases: bool,
    construction_policy: RepeatedCircuitConstructionPolicy,
) -> DFPartialS2RepeatedRequest:
    step_requests = tuple(
        _request_for_events(
            preparation,
            step_time=step_time,
            config=config,
            distribution=distribution,
            events=events,
            seed=step_seeds[index],
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
        )
        for index, events in enumerate(step_event_sequences)
    )
    if len(step_requests) != repetition_count:
        raise ValueError("Trajectory step sequence count does not match repetition_count.")
    return DFPartialS2RepeatedRequest.from_step_requests(
        step_requests,
        master_seed=master_seed,
        trajectory_seed=trajectory_seed,
        sampling_policy=sampling_policy,
        construction_policy=construction_policy,
    )


def estimate_exact_compiled_repeated_partial_s2_cost(
    preparation: DFPartialS2Preparation,
    step_time: float,
    repetition_count: int,
    rte_config: RTEConfig | None,
    rte_distribution: RTEFiniteDistribution | None,
    compiler: CompilerSettings,
    *,
    controlled: bool = False,
    ancilla_qubit: int | None = None,
    cancel_adjacent_equal_bases: bool = True,
    construction_policy: RepeatedCircuitConstructionPolicy = (
        "boundary_optimized"
    ),
    evaluation_mode: RepeatedCostEvaluationMode = "full_diagnostics",
    maximum_trajectories: int = 10_000,
    maximum_untranspiled_circuit_size: int = 100_000,
    cache: TranspiledCircuitCostCache | None = None,
    backend: Any | None = None,
) -> CompiledRepeatedPartialS2CostEstimate:
    """Enumerate ``M**q`` trajectories only after a complete preflight count."""
    count = require_integer_count(
        repetition_count,
        name="repetition_count",
        minimum=1,
    )
    maximum_trajectories = require_integer_count(
        maximum_trajectories,
        name="maximum_trajectories",
        minimum=1,
    )
    maximum_untranspiled_circuit_size = require_integer_count(
        maximum_untranspiled_circuit_size,
        name="maximum_untranspiled_circuit_size",
        minimum=1,
    )
    single_event_count = _validate_rte_inputs(
        preparation,
        float(step_time),
        rte_config,
        rte_distribution,
    )
    rte_steps = 0 if rte_config is None else rte_config.rte_steps
    sequence_count = 1 if rte_config is None else single_event_count**rte_steps
    trajectory_count = sequence_count**count
    if trajectory_count > maximum_trajectories:
        raise ValueError(
            "Exact repeated partial-S2 trajectory space has "
            f"{trajectory_count} trajectories (M={sequence_count}, q={count}), "
            f"above maximum_trajectories={maximum_trajectories}."
        )

    if rte_config is None:
        event_sequences: Sequence[tuple[RTEEvent, ...]] = ((),)
    else:
        events = enumerate_rte_events(
            preparation.rte_preparation.symbolic_tail.components,
            rte_distribution,
            max_events=single_event_count,
        )
        event_sequences = tuple(itertools.product(events, repeat=rte_steps))
    trajectories = tuple(itertools.product(event_sequences, repeat=count))
    weights = tuple(
        math.prod(
            event.event_probability
            for step_events in trajectory
            for event in step_events
        )
        for trajectory in trajectories
    )
    probability_sum = math.fsum(weights)
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_ATOL,
    ):
        raise ValueError("Exact repeated trajectory probabilities must sum to one.")
    normalized_weights = tuple(weight / probability_sum for weight in weights)
    exact_step_seeds: tuple[int | None, ...] = (
        (None,) * count
        if rte_config is None
        else tuple(range(count))
    )
    requests = tuple(
        _request_from_step_event_sequences(
            preparation,
            step_time=step_time,
            repetition_count=count,
            config=rte_config,
            distribution=rte_distribution,
            step_event_sequences=trajectory,
            step_seeds=exact_step_seeds,
            master_seed=None,
            trajectory_seed=None,
            sampling_policy="exact_enumeration_step_major_v1",
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
            construction_policy=construction_policy,
        )
        for trajectory in trajectories
    )
    return _compile_repeated_samples(
        requests,
        compiler,
        estimate_kind="exact_compiled_repeated_partial_s2_expectation",
        evaluation_mode=evaluation_mode,
        weights=normalized_weights,
        trajectory_probability_sum=probability_sum,
        master_seed=None,
        sampled_trajectory_seeds=None,
        single_event_space_size=single_event_count,
        single_step_event_sequence_count=sequence_count,
        trajectory_space_size=trajectory_count,
        maximum_trajectories=maximum_trajectories,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        cache=cache,
        backend=backend,
    )


def estimate_monte_carlo_compiled_repeated_partial_s2_cost(
    preparation: DFPartialS2Preparation,
    step_time: float,
    repetition_count: int,
    rte_config: RTEConfig | None,
    rte_distribution: RTEFiniteDistribution | None,
    compiler: CompilerSettings,
    *,
    sample_count: int,
    seed: int,
    maximum_samples: int = 10_000,
    controlled: bool = False,
    ancilla_qubit: int | None = None,
    cancel_adjacent_equal_bases: bool = True,
    construction_policy: RepeatedCircuitConstructionPolicy = (
        "boundary_optimized"
    ),
    evaluation_mode: RepeatedCostEvaluationMode = "full_diagnostics",
    maximum_untranspiled_circuit_size: int = 100_000,
    cache: TranspiledCircuitCostCache | None = None,
    backend: Any | None = None,
) -> CompiledRepeatedPartialS2CostEstimate:
    """Directly sample full trajectories and use an unweighted sample mean."""
    count = require_integer_count(
        repetition_count,
        name="repetition_count",
        minimum=1,
    )
    sample_count = require_integer_count(sample_count, name="sample_count", minimum=1)
    maximum_samples = require_integer_count(
        maximum_samples,
        name="maximum_samples",
        minimum=1,
    )
    if sample_count > maximum_samples:
        raise ValueError(
            f"sample_count={sample_count} exceeds maximum_samples={maximum_samples}."
        )
    master_seed = require_integer_count(seed, name="seed")
    maximum_untranspiled_circuit_size = require_integer_count(
        maximum_untranspiled_circuit_size,
        name="maximum_untranspiled_circuit_size",
        minimum=1,
    )
    single_event_count = _validate_rte_inputs(
        preparation,
        float(step_time),
        rte_config,
        rte_distribution,
    )
    rte_steps = 0 if rte_config is None else rte_config.rte_steps
    sequence_count = 1 if rte_config is None else single_event_count**rte_steps
    trajectory_space_size = sequence_count**count
    master_rng = random.Random(master_seed)
    trajectory_seeds = tuple(
        master_rng.randrange(0, 2**63) for _ in range(sample_count)
    )
    requests: list[DFPartialS2RepeatedRequest] = []
    for trajectory_seed in trajectory_seeds:
        if rte_config is None:
            step_seeds: tuple[int | None, ...] = (None,) * count
            step_sequences: tuple[tuple[RTEEvent, ...], ...] = ((),) * count
        else:
            if count == 1:
                normalized_step_seeds = (trajectory_seed,)
            else:
                trajectory_rng = random.Random(trajectory_seed)
                normalized_step_seeds = tuple(
                    trajectory_rng.randrange(0, 2**63) for _ in range(count)
                )
            step_seeds = normalized_step_seeds
            step_sequences = tuple(
                preparation.rte_preparation.sample_events(
                    rte_distribution,
                    sample_count=rte_config.rte_steps,
                    seed=step_seed,
                )
                for step_seed in normalized_step_seeds
            )
        requests.append(
            _request_from_step_event_sequences(
                preparation,
                step_time=step_time,
                repetition_count=count,
                config=rte_config,
                distribution=rte_distribution,
                step_event_sequences=step_sequences,
                step_seeds=step_seeds,
                master_seed=master_seed,
                trajectory_seed=trajectory_seed,
                sampling_policy="monte_carlo_master_trajectory_step_v1",
                controlled=controlled,
                ancilla_qubit=ancilla_qubit,
                cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
                construction_policy=construction_policy,
            )
        )
    return _compile_repeated_samples(
        requests,
        compiler,
        estimate_kind="monte_carlo_compiled_repeated_partial_s2_expectation",
        evaluation_mode=evaluation_mode,
        weights=None,
        trajectory_probability_sum=None,
        master_seed=master_seed,
        sampled_trajectory_seeds=trajectory_seeds,
        single_event_space_size=single_event_count,
        single_step_event_sequence_count=sequence_count,
        trajectory_space_size=trajectory_space_size,
        maximum_trajectories=maximum_samples,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        cache=cache,
        backend=backend,
    )

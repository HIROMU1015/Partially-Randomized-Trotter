"""Compiled expectations for one complete DF partial-S2 circuit step."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .df_partial_s2 import (
    DFPartialS2Preparation,
    DFPartialS2StepRequest,
    QiskitDFPartialS2CircuitBuilder,
    estimate_df_partial_s2_untranspiled_size_upper_bound,
)
from .df_rte_circuit import DFRTEEventSequenceCircuitRequest
from .rte import (
    CircuitCost,
    CompilerSettings,
    PROBABILITY_ATOL,
    RTE_PARAMETER_ABS_TOL,
    RTE_PARAMETER_REL_TOL,
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


PartialS2EstimateKind = Literal[
    "exact_compiled_partial_s2_expectation",
    "monte_carlo_compiled_partial_s2_expectation",
]


@dataclass(frozen=True)
class CompiledPartialS2CostEstimate:
    """Level-5 full-step, additive, and matched nonadditive statistics."""

    estimate_kind: PartialS2EstimateKind
    expected_cost: CircuitCost
    standard_error: CircuitCost | None
    additive_expected_cost: CircuitCost
    additive_standard_error: CircuitCost | None
    nonadditive_difference: CircuitCost
    difference_standard_error: CircuitCost | None
    forward_half_expected_cost: CircuitCost
    rte_occurrence_expected_cost: CircuitCost
    reverse_half_expected_cost: CircuitCost
    full_metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    additive_metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    difference_metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    sample_count: int | None
    enumerated_event_sequence_count: int | None
    event_sequence_probability_sum: float | None
    single_event_space_size: int
    unique_full_step_circuit_count: int
    unique_compiled_circuit_count: int
    transpile_cache_hit_count: int
    transpile_cache_miss_count: int
    transpile_cache_bypass_count: int
    transpile_cache_eviction_count: int
    transpile_cache_maximum_entries: int
    compiler: CompilerSettings
    controlled: bool
    ancilla_qubit: int | None
    basis_reuse_policy: Literal["disabled", "raw_adjacent_equal_basis", "none"]
    seed: int | None
    maximum_event_sequences: int | None
    maximum_untranspiled_circuit_size: int
    statistics_policy: Literal["online_welford_v1"] = "online_welford_v1"


def _validate_rte_inputs(
    preparation: DFPartialS2Preparation,
    step_time: float,
    config: RTEConfig | None,
    distribution: RTEFiniteDistribution | None,
) -> int:
    if preparation.is_deterministic_only:
        if config is not None or distribution is not None:
            raise ValueError("Deterministic-only partial-S2 cost requires no RTE data.")
        return 1
    if config is None or distribution is None:
        raise ValueError("Randomized partial-S2 cost requires config and distribution.")
    tail = preparation.rte_preparation.symbolic_tail
    if config.tail_id != tail.tail_id or config.tail_hash != tail.tail_hash:
        raise ValueError("RTE config tail identity does not match preparation.")
    if not math.isclose(
        config.lambda_r,
        preparation.exact_rte_lambda_r,
        rel_tol=RTE_PARAMETER_REL_TOL,
        abs_tol=RTE_PARAMETER_ABS_TOL,
    ):
        raise ValueError("RTE config must use exact_rte_lambda_r.")
    if not math.isclose(
        config.evolution_time,
        step_time,
        rel_tol=0.0,
        abs_tol=1e-14,
    ):
        raise ValueError("RTE evolution_time must equal partial-S2 step_time.")
    if config.finite_taylor_order != distribution.finite_taylor_order:
        raise ValueError("RTE config and distribution Taylor cutoffs differ.")
    if not math.isclose(
        config.dimensionless_step_time,
        distribution.dimensionless_step_time,
        rel_tol=RTE_PARAMETER_REL_TOL,
        abs_tol=RTE_PARAMETER_ABS_TOL,
    ):
        raise ValueError("RTE config and distribution step times differ.")
    if not math.isclose(
        config.distribution_normalization,
        distribution.exact_finite_distribution,
        rel_tol=RTE_PARAMETER_REL_TOL,
        abs_tol=RTE_PARAMETER_ABS_TOL,
    ):
        raise ValueError("RTE config and distribution normalizations differ.")
    component_count = len(tail.components)
    return sum(
        component_count ** (order + 1) for order in distribution.orders
    )


def _request_for_events(
    preparation: DFPartialS2Preparation,
    *,
    step_time: float,
    config: RTEConfig | None,
    distribution: RTEFiniteDistribution | None,
    events: tuple[RTEEvent, ...],
    seed: int | None,
    controlled: bool,
    ancilla_qubit: int | None,
    cancel_adjacent_equal_bases: bool,
) -> DFPartialS2StepRequest:
    occurrence = None
    if config is not None:
        tail = preparation.rte_preparation.symbolic_tail
        occurrence = DFRTEEventSequenceCircuitRequest(
            events=events,
            component_specs=preparation.rte_preparation.component_specs,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
            tail_id=tail.tail_id,
            tail_hash=tail.tail_hash,
            occurrence_rte_steps=config.rte_steps,
        )
    return DFPartialS2StepRequest(
        preparation=preparation,
        step_time=float(step_time),
        rte_config=config,
        rte_distribution=distribution,
        rte_occurrence=occurrence,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        seed=seed,
    )


def _compile_request_samples(
    requests: Sequence[DFPartialS2StepRequest],
    compiler: CompilerSettings,
    *,
    estimate_kind: PartialS2EstimateKind,
    weights: Sequence[float] | None,
    event_sequence_probability_sum: float | None,
    controlled: bool,
    ancilla_qubit: int | None,
    seed: int | None,
    maximum_event_sequences: int | None,
    maximum_untranspiled_circuit_size: int,
    single_event_space_size: int,
    cache: TranspiledCircuitCostCache | None,
    backend: Any | None,
) -> CompiledPartialS2CostEstimate:
    working_cache = cache if cache is not None else TranspiledCircuitCostCache()
    initial_misses = working_cache.miss_count
    initial_bypasses = working_cache.bypass_count
    initial_evictions = working_cache.eviction_count
    builder = QiskitDFPartialS2CircuitBuilder()
    if weights is not None and len(weights) != len(requests):
        raise ValueError("Event-sequence weights length must match requests.")
    weighted = weights is not None
    full_accumulator = CompiledMetricAccumulator(weighted=weighted)
    additive_accumulator = CompiledMetricAccumulator(weighted=weighted)
    difference_accumulator = CompiledMetricAccumulator(weighted=weighted)
    forward_accumulator = CompiledMetricAccumulator(weighted=weighted)
    rte_accumulator = CompiledMetricAccumulator(weighted=weighted)
    reverse_accumulator = CompiledMetricAccumulator(weighted=weighted)
    all_keys: set[str] = set()
    full_keys: set[str] = set()
    cache_hits = 0

    for request_index, request in enumerate(requests):
        weight = None if weights is None else weights[request_index]
        planned_size = estimate_df_partial_s2_untranspiled_size_upper_bound(request)
        if planned_size > maximum_untranspiled_circuit_size:
            raise ValueError(
                "Planned partial-S2 circuit upper bound exceeds the configured "
                "size limit before circuit construction."
            )
        result = builder.build_step(request)
        if result.untranspiled_circuit_size > maximum_untranspiled_circuit_size:
            raise ValueError(
                "Untranspiled partial-S2 circuit exceeds the configured size limit."
            )
        full_cost, full_key, full_cached = working_cache.get_or_transpile(
            result.circuit,
            compiler,
            circuit_fingerprint=result.compiler_independent_fingerprint,
            backend=backend,
        )
        full_accumulator.update(full_cost, weight=weight)
        all_keys.add(full_key)
        full_keys.add(full_key)
        cache_hits += int(full_cached)

        parts = builder.build_additive_circuits(request)
        part_costs: list[TranspiledCircuitCost] = []
        for circuit, fingerprint in (
            (parts.forward_deterministic_half, parts.forward_fingerprint),
            (parts.rte_occurrence, parts.rte_fingerprint),
            (parts.reverse_deterministic_half, parts.reverse_fingerprint),
        ):
            cost, key, was_cached = working_cache.get_or_transpile(
                circuit,
                compiler,
                circuit_fingerprint=fingerprint,
                backend=backend,
            )
            part_costs.append(cost)
            all_keys.add(key)
            cache_hits += int(was_cached)
        forward_accumulator.update(part_costs[0], weight=weight)
        rte_accumulator.update(part_costs[1], weight=weight)
        reverse_accumulator.update(part_costs[2], weight=weight)
        additive = sum_compiled_costs(part_costs)
        additive_accumulator.update(additive, weight=weight)
        difference_accumulator.update(
            subtract_compiled_costs(full_cost, additive),
            weight=weight,
        )

    full_statistics = full_accumulator.finalize()
    additive_statistics = additive_accumulator.finalize()
    difference_statistics = difference_accumulator.finalize()
    forward_statistics = forward_accumulator.finalize()
    rte_statistics = rte_accumulator.finalize()
    reverse_statistics = reverse_accumulator.finalize()

    def make_cost(
        statistics: tuple[tuple[str, CompiledMetricStatistics], ...],
        kind: str,
        *,
        standard_error: bool = False,
    ) -> CircuitCost | None:
        return circuit_cost_from_metric_statistics(
            statistics,
            compiler=compiler,
            estimate_kind=kind,
            use_standard_error=standard_error,
            fidelity_level=5,
        )

    expected = make_cost(full_statistics, estimate_kind)
    additive_expected = make_cost(
        additive_statistics,
        "compiled_partial_s2_additive_expectation",
    )
    difference_expected = make_cost(
        difference_statistics,
        "compiled_partial_s2_nonadditive_difference",
    )
    forward_expected = make_cost(
        forward_statistics,
        "compiled_partial_s2_additive_expectation",
    )
    rte_expected = make_cost(
        rte_statistics,
        "compiled_partial_s2_additive_expectation",
    )
    reverse_expected = make_cost(
        reverse_statistics,
        "compiled_partial_s2_additive_expectation",
    )
    if any(
        item is None
        for item in (
            expected,
            additive_expected,
            difference_expected,
            forward_expected,
            rte_expected,
            reverse_expected,
        )
    ):
        raise RuntimeError("Compiled partial-S2 expectation could not be constructed.")
    is_exact = estimate_kind == "exact_compiled_partial_s2_expectation"
    first = requests[0]
    if first.rte_occurrence is None:
        reuse_policy: Literal[
            "disabled", "raw_adjacent_equal_basis", "none"
        ] = "none"
    elif first.rte_occurrence.cancel_adjacent_equal_bases:
        reuse_policy = "raw_adjacent_equal_basis"
    else:
        reuse_policy = "disabled"
    return CompiledPartialS2CostEstimate(
        estimate_kind=estimate_kind,
        expected_cost=expected,
        standard_error=make_cost(
            full_statistics,
            estimate_kind,
            standard_error=True,
        ),
        additive_expected_cost=additive_expected,
        additive_standard_error=make_cost(
            additive_statistics,
            "compiled_partial_s2_additive_expectation",
            standard_error=True,
        ),
        nonadditive_difference=difference_expected,
        difference_standard_error=make_cost(
            difference_statistics,
            "compiled_partial_s2_nonadditive_difference",
            standard_error=True,
        ),
        forward_half_expected_cost=forward_expected,
        rte_occurrence_expected_cost=rte_expected,
        reverse_half_expected_cost=reverse_expected,
        full_metric_statistics=full_statistics,
        additive_metric_statistics=additive_statistics,
        difference_metric_statistics=difference_statistics,
        sample_count=None if is_exact else len(requests),
        enumerated_event_sequence_count=len(requests) if is_exact else None,
        event_sequence_probability_sum=event_sequence_probability_sum,
        single_event_space_size=single_event_space_size,
        unique_full_step_circuit_count=len(full_keys),
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
        compiler=compiler,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        basis_reuse_policy=reuse_policy,
        seed=None if is_exact else seed,
        maximum_event_sequences=maximum_event_sequences,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
    )


def estimate_exact_compiled_partial_s2_cost(
    preparation: DFPartialS2Preparation,
    step_time: float,
    rte_config: RTEConfig | None,
    rte_distribution: RTEFiniteDistribution | None,
    compiler: CompilerSettings,
    *,
    controlled: bool = False,
    ancilla_qubit: int | None = None,
    cancel_adjacent_equal_bases: bool = True,
    maximum_event_sequences: int = 10_000,
    maximum_untranspiled_circuit_size: int = 100_000,
    cache: TranspiledCircuitCostCache | None = None,
    backend: Any | None = None,
) -> CompiledPartialS2CostEstimate:
    """Exactly enumerate and compile every finite event sequence for one step."""
    maximum_event_sequences = require_integer_count(
        maximum_event_sequences,
        name="maximum_event_sequences",
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
    if sequence_count > maximum_event_sequences:
        raise ValueError(
            "Exact partial-S2 event sequence space has "
            f"{sequence_count} sequences, above "
            f"maximum_event_sequences={maximum_event_sequences}."
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
    weights = tuple(
        math.prod(event.event_probability for event in sequence)
        for sequence in event_sequences
    )
    probability_sum = math.fsum(weights)
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_ATOL,
    ):
        raise ValueError("Exact partial-S2 event sequence probabilities must sum to one.")
    normalized_weights = tuple(weight / probability_sum for weight in weights)
    requests = tuple(
        _request_for_events(
            preparation,
            step_time=step_time,
            config=rte_config,
            distribution=rte_distribution,
            events=tuple(sequence),
            seed=0 if rte_config is not None else None,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
        )
        for sequence in event_sequences
    )
    return _compile_request_samples(
        requests,
        compiler,
        estimate_kind="exact_compiled_partial_s2_expectation",
        weights=normalized_weights,
        event_sequence_probability_sum=probability_sum,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        seed=None,
        maximum_event_sequences=maximum_event_sequences,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        single_event_space_size=single_event_count,
        cache=cache,
        backend=backend,
    )


def estimate_monte_carlo_compiled_partial_s2_cost(
    preparation: DFPartialS2Preparation,
    step_time: float,
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
    maximum_untranspiled_circuit_size: int = 100_000,
    cache: TranspiledCircuitCostCache | None = None,
    backend: Any | None = None,
) -> CompiledPartialS2CostEstimate:
    """Compile unweighted classically sampled complete partial-S2 steps."""
    sample_count = require_integer_count(
        sample_count,
        name="sample_count",
        minimum=1,
    )
    maximum_samples = require_integer_count(
        maximum_samples,
        name="maximum_samples",
        minimum=1,
    )
    if sample_count > maximum_samples:
        raise ValueError(
            f"sample_count={sample_count} exceeds maximum_samples={maximum_samples}."
        )
    seed = require_integer_count(seed, name="seed")
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
    rng = random.Random(seed)
    requests: list[DFPartialS2StepRequest] = []
    for _ in range(sample_count):
        if rte_config is None:
            events: tuple[RTEEvent, ...] = ()
            occurrence_seed = None
        else:
            occurrence_seed = rng.randrange(0, 2**63)
            events = preparation.rte_preparation.sample_events(
                rte_distribution,
                sample_count=rte_config.rte_steps,
                seed=occurrence_seed,
            )
        requests.append(
            _request_for_events(
                preparation,
                step_time=step_time,
                config=rte_config,
                distribution=rte_distribution,
                events=events,
                seed=occurrence_seed,
                controlled=controlled,
                ancilla_qubit=ancilla_qubit,
                cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
            )
        )
    return _compile_request_samples(
        requests,
        compiler,
        estimate_kind="monte_carlo_compiled_partial_s2_expectation",
        weights=None,
        event_sequence_probability_sum=None,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        seed=seed,
        maximum_event_sequences=maximum_samples,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        single_event_space_size=single_event_count,
        cache=cache,
        backend=backend,
    )

"""Compiled costs of complete short-round RPE Hadamard interrogations.

The estimators in this module build one controlled repeated partial-S2
trajectory, wrap that same trajectory on both Hadamard axes, and transpile the
two complete measurement-bearing interrogation circuits independently.  They
do not prepare a system state, execute a backend, or count quantum shots.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np

from .df_partial_s2 import DFPartialS2Preparation
from .df_partial_s2_repeated import (
    QiskitDFPartialS2RepeatedCircuitBuilder,
    RepeatedCircuitConstructionPolicy,
)
from .df_partial_s2_repeated_cost import (
    DFPartialS2RepeatedTrajectoryStream,
    make_exact_df_partial_s2_repeated_trajectory_stream,
    make_monte_carlo_df_partial_s2_repeated_trajectory_stream,
    plan_compiled_repeated_partial_s2_workload,
)
from .rpe_hadamard_interrogation import (
    RPE_HADAMARD_INTERROGATION_SCOPE,
    QiskitRPEHadamardInterrogationBuilder,
    RPEHadamardAxis,
    RPEHadamardInterrogationRequest,
    round_index_for_short_rpe_repetition_count,
)
from .rpe_resource_accounting import (
    RPE_COST_METRICS,
    RPERoundCompiledCost,
    RPERoundCostRequest,
)
from .rte import (
    CircuitCost,
    CompilerSettings,
    PROBABILITY_ATOL,
    RTE_FINITE_DISTRIBUTION_SCHEMA_VERSION,
    RTE_PRNG_TYPE,
    RTE_SAMPLING_CONVENTION_VERSION,
    RTEConfig,
    RTEFiniteDistribution,
    require_integer_count,
)
from .rte_compiled_cost import (
    CompiledCostWorkloadBudget,
    CompiledCostWorkloadPlan,
    CompiledMetricAccumulator,
    CompiledMetricStatistics,
    TranspiledCircuitCost,
    TranspiledCircuitCostCache,
    canonical_backend_fingerprint_or_none,
    circuit_cost_from_metric_statistics,
    compiler_settings_hash,
    plan_compiled_cost_workload,
    require_actual_workload_within_plan,
    require_compiled_cost_workload_within_budget,
)


DFRPEHadamardCostEvaluationMethod: TypeAlias = Literal["exact", "monte_carlo"]
DFRPEHadamardEstimateKind: TypeAlias = Literal[
    "exact_compiled_rpe_hadamard_interrogation_expectation",
    "monte_carlo_compiled_rpe_hadamard_interrogation_expectation",
]

DF_RPE_HADAMARD_COMPILED_COST_PROVIDER_VERSION = (
    "df_rpe_hadamard_compiled_cost_provider_v1"
)
DF_RPE_HADAMARD_COMPILED_COST_EVALUATION_SCHEMA_VERSION = (
    "df_rpe_hadamard_compiled_cost_evaluation_v1"
)
DF_RPE_HADAMARD_MEASUREMENT_POLICY = "include_ancilla_z_measurement"


@dataclass(frozen=True)
class DFRPEHadamardCompiledMetricValues:
    """The six resource-accounting metrics for one transpiled wrapper."""

    rz_count: float
    rz_depth: float
    cx_count: float
    cx_depth: float
    total_depth: float
    circuit_size: float


@dataclass(frozen=True)
class DFRPEHadamardTrajectoryCostRecord:
    """Retained paired-axis costs and provenance for one trajectory."""

    trajectory_index: int
    probability: float | None
    trajectory_seed: int | None
    evolution_provenance_fingerprint: str
    evolution_circuit_semantics_fingerprint: str
    cosine_wrapper_fingerprint: str
    sine_wrapper_fingerprint: str
    cosine_wrapper_circuit_semantics_fingerprint: str
    sine_wrapper_circuit_semantics_fingerprint: str
    cosine_actual_circuit_fingerprint: str | None
    sine_actual_circuit_fingerprint: str | None
    cosine_cost: DFRPEHadamardCompiledMetricValues
    sine_cost: DFRPEHadamardCompiledMetricValues
    constant_phase: float
    extracted_identity_phase: float
    rte_relative_phase: float


@dataclass(frozen=True)
class DFRPEHadamardAxisCompiledCostEstimate:
    """Expected compiled cost and circuit-semantics audit for one axis."""

    axis: RPEHadamardAxis
    expected_cost: CircuitCost
    standard_error: CircuitCost | None
    metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    compiled_cost_evaluation_fingerprint: str | None
    wrapper_circuit_semantics_fingerprints: tuple[str, ...]
    wrapper_circuit_semantics_multiset_digest: str
    evaluation_input_digest: str
    unique_wrapper_circuit_count: int


@dataclass(frozen=True)
class DFRPEHadamardCompiledCostEstimate:
    """Paired cosine/sine compiled expectation over one trajectory set."""

    estimate_kind: DFRPEHadamardEstimateKind
    evaluation_method: DFRPEHadamardCostEvaluationMethod
    cosine: DFRPEHadamardAxisCompiledCostEstimate
    sine: DFRPEHadamardAxisCompiledCostEstimate
    round_index: int
    repetition_count: int
    q_m: int
    step_time: float
    delta_time: float
    total_evolution_time: float
    t_m: float
    sample_count: int | None
    enumerated_trajectory_count: int | None
    trajectory_probability_sum: float | None
    normalized_trajectory_probability_sum: float | None
    trajectory_space_size: int
    processed_trajectory_count: int
    retained_trajectory_records: tuple[DFRPEHadamardTrajectoryCostRecord, ...]
    maximum_retained_trajectory_records: int
    trajectory_records_truncated: bool
    trajectory_provenance_digest: str
    evolution_circuit_semantics_multiset_digest: str
    sampled_trajectory_seeds: tuple[int, ...] | None
    sampled_trajectory_seed_digest: str | None
    master_seed: int | None
    shared_axis_trajectory_set: bool
    compiler: CompilerSettings
    backend_fingerprint: str | None
    backend_context_canonical: bool
    construction_policy: RepeatedCircuitConstructionPolicy
    circuit_scope: Literal[
        "single_hadamard_interrogation_without_state_preparation"
    ]
    measurement_policy: Literal["include_ancilla_z_measurement"]
    measurement_included: bool
    state_preparation_included: bool
    backend_execution_included: bool
    quantum_shots_executed: int
    classical_samples_are_quantum_shots: bool
    fresh_iid_trajectory_per_hadamard_shot_verified: bool
    complete: bool
    incomplete_reason: str | None
    compiled_cost_evaluation_fingerprint: str | None
    unique_compiled_circuit_count: int
    transpile_cache_hit_count: int
    transpile_cache_miss_count: int
    transpile_cache_bypass_count: int
    transpile_cache_eviction_count: int
    transpile_cache_maximum_entries: int
    planned_build_requests: int
    actual_build_requests: int
    planned_transpile_requests: int
    actual_cache_requests: int
    planned_instruction_applications: int
    actual_built_instruction_total: int
    workload_budget: CompiledCostWorkloadBudget
    workload_policy_version: str
    cost_metrics: tuple[str, ...] = RPE_COST_METRICS
    statistics_policy: Literal["online_welford_v1"] = "online_welford_v1"


def _metric_values(cost: TranspiledCircuitCost) -> DFRPEHadamardCompiledMetricValues:
    return DFRPEHadamardCompiledMetricValues(
        rz_count=float(cost.rz_count),
        rz_depth=float(cost.rz_depth),
        cx_count=float(cost.cx_count),
        cx_depth=float(cost.cx_depth),
        total_depth=float(cost.total_depth),
        circuit_size=float(cost.circuit_size),
    )


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest_entries(entries: list[str], *, ordered: bool) -> str:
    values = entries if ordered else sorted(entries)
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cost_from_statistics(
    statistics: tuple[tuple[str, CompiledMetricStatistics], ...],
    *,
    compiler: CompilerSettings,
    estimate_kind: DFRPEHadamardEstimateKind,
    standard_error: bool,
) -> CircuitCost | None:
    return circuit_cost_from_metric_statistics(
        statistics,
        compiler=compiler,
        estimate_kind=estimate_kind,
        use_standard_error=standard_error,
        fidelity_level=5,
    )


def plan_compiled_rpe_hadamard_workload(
    preparation: DFPartialS2Preparation,
    repetition_count: int,
    rte_config: RTEConfig | None,
    rte_distribution: RTEFiniteDistribution | None,
    *,
    trajectory_count: int,
) -> CompiledCostWorkloadPlan:
    """Plan two complete measurement-bearing wrappers per trajectory."""
    count = require_integer_count(
        repetition_count,
        name="repetition_count",
        minimum=1,
    )
    round_index_for_short_rpe_repetition_count(count)
    trajectories = require_integer_count(
        trajectory_count,
        name="trajectory_count",
        minimum=1,
    )
    one_evolution = plan_compiled_repeated_partial_s2_workload(
        preparation,
        count,
        rte_config,
        rte_distribution,
        trajectory_count=1,
        controlled=True,
        evaluation_mode="selected_only",
    )
    evolution_instruction_bound = (
        one_evolution.planned_untranspiled_instruction_applications
    )
    # cosine: H + evolution + H + measure; sine adds S-dagger.
    paired_wrapper_instruction_bound = 2 * evolution_instruction_bound + 7
    return plan_compiled_cost_workload(
        work_item_count=trajectories,
        circuits_per_work_item=2,
        instruction_applications_per_work_item=(
            paired_wrapper_instruction_bound
        ),
    )


def _axis_evaluation_fingerprint(
    *,
    axis: RPEHadamardAxis,
    stream: DFPartialS2RepeatedTrajectoryStream,
    round_index: int,
    step_time: float,
    wrapper_semantics_digest: str,
    evaluation_input_digest: str,
    sampled_seed_digest: str | None,
    compiler: CompilerSettings,
    backend_fingerprint: str | None,
    complete: bool,
) -> str | None:
    if backend_fingerprint is None:
        return None
    return _sha256_json(
        {
            "schema_version": (
                DF_RPE_HADAMARD_COMPILED_COST_EVALUATION_SCHEMA_VERSION
            ),
            "circuit_scope": RPE_HADAMARD_INTERROGATION_SCOPE,
            "axis": axis,
            "measurement_policy": DF_RPE_HADAMARD_MEASUREMENT_POLICY,
            "round_index": round_index,
            "q_m": stream.repetition_count,
            "delta_time": float(step_time).hex(),
            "t_m": float(stream.repetition_count * step_time).hex(),
            "wrapper_circuit_semantics_multiset_digest": (
                wrapper_semantics_digest
            ),
            "evaluation_input_digest": evaluation_input_digest,
            "compiler_settings_hash": compiler_settings_hash(compiler),
            "backend_fingerprint": backend_fingerprint,
            "cost_metrics": RPE_COST_METRICS,
            "evaluation_method": stream.evaluation_method,
            "trajectory_or_sample_count": stream.expected_record_count,
            "master_seed": stream.master_seed,
            "sampled_trajectory_seed_digest": sampled_seed_digest,
            "rte_prng_type": RTE_PRNG_TYPE,
            "rte_sampling_convention_version": (
                RTE_SAMPLING_CONVENTION_VERSION
            ),
            "complete": complete,
        }
    )


def _compile_hadamard_trajectory_stream_with_plan(
    stream: DFPartialS2RepeatedTrajectoryStream,
    step_time: float,
    compiler: CompilerSettings,
    *,
    construction_policy: RepeatedCircuitConstructionPolicy,
    workload_plan: CompiledCostWorkloadPlan,
    maximum_untranspiled_circuit_size: int,
    maximum_retained_trajectory_records: int,
    maximum_build_requests: int,
    maximum_transpile_requests: int,
    maximum_planned_instruction_applications: int,
    cache: TranspiledCircuitCostCache | None,
    backend: Any | None,
) -> DFRPEHadamardCompiledCostEstimate:
    """Compile one paired wrapper for every item in a preflighted stream."""
    if not isinstance(compiler, CompilerSettings):
        raise TypeError("compiler must be a CompilerSettings instance.")
    size_limit = require_integer_count(
        maximum_untranspiled_circuit_size,
        name="maximum_untranspiled_circuit_size",
        minimum=1,
    )
    retention_limit = require_integer_count(
        maximum_retained_trajectory_records,
        name="maximum_retained_trajectory_records",
    )
    workload_budget = CompiledCostWorkloadBudget(
        maximum_build_requests=maximum_build_requests,
        maximum_transpile_requests=maximum_transpile_requests,
        maximum_planned_instruction_applications=(
            maximum_planned_instruction_applications
        ),
    )
    require_compiled_cost_workload_within_budget(workload_plan, workload_budget)
    per_trajectory_instruction_bound = (
        workload_plan.planned_untranspiled_instruction_applications
        // stream.expected_record_count
    )
    evolution_instruction_bound = (per_trajectory_instruction_bound - 7) // 2
    if evolution_instruction_bound + 4 > size_limit:
        raise ValueError(
            "Planned RPE Hadamard wrapper upper bound exceeds the configured "
            "size limit before circuit construction."
        )
    weighted = stream.evaluation_method == "exact"
    estimate_kind: DFRPEHadamardEstimateKind = (
        "exact_compiled_rpe_hadamard_interrogation_expectation"
        if weighted
        else "monte_carlo_compiled_rpe_hadamard_interrogation_expectation"
    )
    round_index = round_index_for_short_rpe_repetition_count(
        stream.repetition_count
    )
    working_cache = cache if cache is not None else TranspiledCircuitCostCache()
    initial_misses = working_cache.miss_count
    initial_bypasses = working_cache.bypass_count
    initial_evictions = working_cache.eviction_count
    repeated_builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    wrapper_builder = QiskitRPEHadamardInterrogationBuilder()
    compiled_rows: list[
        tuple[
            str,
            float | None,
            DFRPEHadamardCompiledMetricValues,
            DFRPEHadamardCompiledMetricValues,
        ]
    ] = []
    retained_records: list[DFRPEHadamardTrajectoryCostRecord] = []
    cosine_semantics_entries: list[str] = []
    sine_semantics_entries: list[str] = []
    cosine_evaluation_entries: list[str] = []
    sine_evaluation_entries: list[str] = []
    evolution_semantics_entries: list[str] = []
    provenance_entries: list[str] = []
    sampled_seeds: list[int] = []
    cosine_semantics_retained: list[str] = []
    sine_semantics_retained: list[str] = []
    cosine_semantics_unique: set[str] = set()
    sine_semantics_unique: set[str] = set()
    cache_keys: set[str] = set()
    cache_hits = 0
    processed = 0
    probability_sum = 0.0
    actual_build_requests = 0
    actual_cache_requests = 0
    actual_built_instruction_total = 0

    for index, (request, weight) in enumerate(stream.records):
        if weighted:
            if weight is None:
                raise ValueError("Exact trajectories require probability weights.")
            normalized_weight = float(weight)
            if not math.isfinite(normalized_weight) or normalized_weight < 0.0:
                raise ValueError("Exact trajectory weights must be non-negative.")
            probability_sum += normalized_weight
        else:
            if weight is not None:
                raise ValueError("Monte Carlo trajectories must be unweighted.")
            normalized_weight = None
            if request.trajectory_seed is None:
                raise ValueError("Monte Carlo trajectories require trajectory seeds.")
            sampled_seeds.append(request.trajectory_seed)

        evolution = repeated_builder.build(
            request,
            construction_policy=construction_policy,
        )
        cosine = wrapper_builder.build(
            RPEHadamardInterrogationRequest(
                evolution=evolution,
                axis="cosine",
                include_measurement=True,
            )
        )
        sine = wrapper_builder.build(
            RPEHadamardInterrogationRequest(
                evolution=evolution,
                axis="sine",
                include_measurement=True,
            )
        )
        if (
            cosine.wrapped_trajectory_fingerprint
            != sine.wrapped_trajectory_fingerprint
        ):
            raise RuntimeError("Cosine and sine did not share one trajectory.")
        if not cosine.include_measurement or not sine.include_measurement:
            raise RuntimeError("Compiled-cost wrappers must include measurement.")
        if cosine.circuit.num_clbits != 1 or sine.circuit.num_clbits != 1:
            raise RuntimeError("Each compiled-cost wrapper requires one clbit.")
        if cosine.state_preparation_included or sine.state_preparation_included:
            raise RuntimeError("State preparation must not enter wrapper costs.")
        if cosine.additional_control_applied or sine.additional_control_applied:
            raise RuntimeError("The wrapper must not control evolution twice.")
        if cosine.circuit.size() > size_limit or sine.circuit.size() > size_limit:
            raise ValueError(
                "Untranspiled RPE Hadamard wrapper exceeds the configured size limit."
            )

        axis_costs: list[TranspiledCircuitCost] = []
        for wrapper in (cosine, sine):
            actual_build_requests += 1
            actual_cache_requests += 1
            actual_built_instruction_total += int(wrapper.circuit.size())
            cost, key, cached = working_cache.get_or_transpile(
                wrapper.circuit,
                compiler,
                circuit_fingerprint=wrapper.compiler_independent_fingerprint,
                backend=backend,
            )
            axis_costs.append(cost)
            cache_keys.add(key)
            cache_hits += int(cached)
        cosine_cost, sine_cost = axis_costs
        cosine_values = _metric_values(cosine_cost)
        sine_values = _metric_values(sine_cost)
        probability_key = (
            None if normalized_weight is None else normalized_weight.hex()
        )
        row_key = _sha256_json(
            {
                "probability": probability_key,
                "evolution_semantics": evolution.circuit_semantics_fingerprint,
                "cosine_semantics": (
                    cosine.wrapper_circuit_semantics_fingerprint
                ),
                "sine_semantics": sine.wrapper_circuit_semantics_fingerprint,
            }
        )
        compiled_rows.append(
            (row_key, normalized_weight, cosine_values, sine_values)
        )
        cosine_entry = _sha256_json(
            {
                "probability": probability_key,
                "wrapper_semantics": (
                    cosine.wrapper_circuit_semantics_fingerprint
                ),
            }
        )
        sine_entry = _sha256_json(
            {
                "probability": probability_key,
                "wrapper_semantics": sine.wrapper_circuit_semantics_fingerprint,
            }
        )
        cosine_semantics_entries.append(
            cosine.wrapper_circuit_semantics_fingerprint
        )
        sine_semantics_entries.append(
            sine.wrapper_circuit_semantics_fingerprint
        )
        cosine_evaluation_entries.append(cosine_entry)
        sine_evaluation_entries.append(sine_entry)
        evolution_semantics_entries.append(
            evolution.circuit_semantics_fingerprint
        )
        provenance_entries.append(
            _sha256_json(
                {
                    "trajectory_seed": request.trajectory_seed,
                    "evolution_provenance": evolution.provenance_fingerprint,
                    "cosine_wrapper": cosine.wrapper_fingerprint,
                    "sine_wrapper": sine.wrapper_fingerprint,
                }
            )
        )
        cosine_semantics_unique.add(
            cosine.wrapper_circuit_semantics_fingerprint
        )
        sine_semantics_unique.add(sine.wrapper_circuit_semantics_fingerprint)
        if index < retention_limit:
            cosine_semantics_retained.append(
                cosine.wrapper_circuit_semantics_fingerprint
            )
            sine_semantics_retained.append(
                sine.wrapper_circuit_semantics_fingerprint
            )
            retained_records.append(
                DFRPEHadamardTrajectoryCostRecord(
                    trajectory_index=index,
                    probability=normalized_weight,
                    trajectory_seed=request.trajectory_seed,
                    evolution_provenance_fingerprint=(
                        evolution.provenance_fingerprint
                    ),
                    evolution_circuit_semantics_fingerprint=(
                        evolution.circuit_semantics_fingerprint
                    ),
                    cosine_wrapper_fingerprint=cosine.wrapper_fingerprint,
                    sine_wrapper_fingerprint=sine.wrapper_fingerprint,
                    cosine_wrapper_circuit_semantics_fingerprint=(
                        cosine.wrapper_circuit_semantics_fingerprint
                    ),
                    sine_wrapper_circuit_semantics_fingerprint=(
                        sine.wrapper_circuit_semantics_fingerprint
                    ),
                    cosine_actual_circuit_fingerprint=(
                        cosine_cost.actual_circuit_fingerprint
                    ),
                    sine_actual_circuit_fingerprint=(
                        sine_cost.actual_circuit_fingerprint
                    ),
                    cosine_cost=cosine_values,
                    sine_cost=sine_values,
                    constant_phase=cosine.constant_phase,
                    extracted_identity_phase=(
                        cosine.extracted_identity_phase
                    ),
                    rte_relative_phase=cosine.rte_relative_phase,
                )
            )
        processed += 1

    if processed != stream.expected_record_count:
        raise RuntimeError("Trajectory iterator count differs from its preflight.")
    if weighted and not math.isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_ATOL,
    ):
        raise ValueError("Exact trajectory probabilities must sum to one.")
    require_actual_workload_within_plan(
        workload_plan,
        actual_build_requests=actual_build_requests,
        actual_cache_requests=actual_cache_requests,
        actual_built_instruction_total=actual_built_instruction_total,
    )

    # Exact weighted statistics are accumulated in a canonical semantic order,
    # so changing only enumeration order cannot change floating-point results.
    ordered_rows = (
        sorted(compiled_rows, key=lambda item: item[0])
        if weighted
        else compiled_rows
    )
    cosine_accumulator = CompiledMetricAccumulator(weighted=weighted)
    sine_accumulator = CompiledMetricAccumulator(weighted=weighted)
    for _key, weight, cosine_values, sine_values in ordered_rows:
        cosine_accumulator.update(cosine_values, weight=weight)
        sine_accumulator.update(sine_values, weight=weight)
    cosine_statistics = cosine_accumulator.finalize()
    sine_statistics = sine_accumulator.finalize()
    cosine_expected = _cost_from_statistics(
        cosine_statistics,
        compiler=compiler,
        estimate_kind=estimate_kind,
        standard_error=False,
    )
    sine_expected = _cost_from_statistics(
        sine_statistics,
        compiler=compiler,
        estimate_kind=estimate_kind,
        standard_error=False,
    )
    cosine_standard_error = _cost_from_statistics(
        cosine_statistics,
        compiler=compiler,
        estimate_kind=estimate_kind,
        standard_error=True,
    )
    sine_standard_error = _cost_from_statistics(
        sine_statistics,
        compiler=compiler,
        estimate_kind=estimate_kind,
        standard_error=True,
    )
    if cosine_expected is None or sine_expected is None:
        raise RuntimeError("Hadamard compiled expectation could not be constructed.")

    cosine_semantics_digest = _digest_entries(
        cosine_semantics_entries,
        ordered=False,
    )
    sine_semantics_digest = _digest_entries(
        sine_semantics_entries,
        ordered=False,
    )
    cosine_evaluation_input_digest = _digest_entries(
        cosine_evaluation_entries,
        ordered=False,
    )
    sine_evaluation_input_digest = _digest_entries(
        sine_evaluation_entries,
        ordered=False,
    )
    sampled_seed_digest = (
        None
        if weighted
        else _digest_entries([str(seed) for seed in sampled_seeds], ordered=True)
    )
    backend_fingerprint = canonical_backend_fingerprint_or_none(backend)
    backend_context_canonical = backend_fingerprint is not None
    cosine_evaluation_fingerprint = _axis_evaluation_fingerprint(
        axis="cosine",
        stream=stream,
        round_index=round_index,
        step_time=step_time,
        wrapper_semantics_digest=cosine_semantics_digest,
        evaluation_input_digest=cosine_evaluation_input_digest,
        sampled_seed_digest=sampled_seed_digest,
        compiler=compiler,
        backend_fingerprint=backend_fingerprint,
        complete=True,
    )
    sine_evaluation_fingerprint = _axis_evaluation_fingerprint(
        axis="sine",
        stream=stream,
        round_index=round_index,
        step_time=step_time,
        wrapper_semantics_digest=sine_semantics_digest,
        evaluation_input_digest=sine_evaluation_input_digest,
        sampled_seed_digest=sampled_seed_digest,
        compiler=compiler,
        backend_fingerprint=backend_fingerprint,
        complete=True,
    )
    paired_fingerprint = (
        None
        if cosine_evaluation_fingerprint is None
        or sine_evaluation_fingerprint is None
        else _sha256_json(
            {
                "schema_version": (
                    DF_RPE_HADAMARD_COMPILED_COST_EVALUATION_SCHEMA_VERSION
                ),
                "cosine": cosine_evaluation_fingerprint,
                "sine": sine_evaluation_fingerprint,
                "complete": True,
            }
        )
    )
    return DFRPEHadamardCompiledCostEstimate(
        estimate_kind=estimate_kind,
        evaluation_method=stream.evaluation_method,
        cosine=DFRPEHadamardAxisCompiledCostEstimate(
            axis="cosine",
            expected_cost=cosine_expected,
            standard_error=cosine_standard_error,
            metric_statistics=cosine_statistics,
            compiled_cost_evaluation_fingerprint=(
                cosine_evaluation_fingerprint
            ),
            wrapper_circuit_semantics_fingerprints=tuple(
                cosine_semantics_retained
            ),
            wrapper_circuit_semantics_multiset_digest=(
                cosine_semantics_digest
            ),
            evaluation_input_digest=cosine_evaluation_input_digest,
            unique_wrapper_circuit_count=len(cosine_semantics_unique),
        ),
        sine=DFRPEHadamardAxisCompiledCostEstimate(
            axis="sine",
            expected_cost=sine_expected,
            standard_error=sine_standard_error,
            metric_statistics=sine_statistics,
            compiled_cost_evaluation_fingerprint=sine_evaluation_fingerprint,
            wrapper_circuit_semantics_fingerprints=tuple(
                sine_semantics_retained
            ),
            wrapper_circuit_semantics_multiset_digest=sine_semantics_digest,
            evaluation_input_digest=sine_evaluation_input_digest,
            unique_wrapper_circuit_count=len(sine_semantics_unique),
        ),
        round_index=round_index,
        repetition_count=stream.repetition_count,
        q_m=stream.repetition_count,
        step_time=float(step_time),
        delta_time=float(step_time),
        total_evolution_time=float(stream.repetition_count * step_time),
        t_m=float(stream.repetition_count * step_time),
        sample_count=None if weighted else stream.expected_record_count,
        enumerated_trajectory_count=(
            stream.expected_record_count if weighted else None
        ),
        trajectory_probability_sum=probability_sum if weighted else None,
        normalized_trajectory_probability_sum=1.0 if weighted else None,
        trajectory_space_size=stream.trajectory_space_size,
        processed_trajectory_count=processed,
        retained_trajectory_records=tuple(retained_records),
        maximum_retained_trajectory_records=retention_limit,
        trajectory_records_truncated=processed > len(retained_records),
        trajectory_provenance_digest=_digest_entries(
            provenance_entries,
            ordered=not weighted,
        ),
        evolution_circuit_semantics_multiset_digest=_digest_entries(
            evolution_semantics_entries,
            ordered=False,
        ),
        sampled_trajectory_seeds=(
            None if weighted else tuple(sampled_seeds[:retention_limit])
        ),
        sampled_trajectory_seed_digest=sampled_seed_digest,
        master_seed=stream.master_seed,
        shared_axis_trajectory_set=True,
        compiler=compiler,
        backend_fingerprint=backend_fingerprint,
        backend_context_canonical=backend_context_canonical,
        construction_policy=construction_policy,
        circuit_scope=RPE_HADAMARD_INTERROGATION_SCOPE,
        measurement_policy=DF_RPE_HADAMARD_MEASUREMENT_POLICY,
        measurement_included=True,
        state_preparation_included=False,
        backend_execution_included=False,
        quantum_shots_executed=0,
        classical_samples_are_quantum_shots=False,
        fresh_iid_trajectory_per_hadamard_shot_verified=False,
        complete=True,
        incomplete_reason=None,
        compiled_cost_evaluation_fingerprint=paired_fingerprint,
        unique_compiled_circuit_count=len(cache_keys),
        transpile_cache_hit_count=cache_hits,
        transpile_cache_miss_count=working_cache.miss_count - initial_misses,
        transpile_cache_bypass_count=(
            working_cache.bypass_count - initial_bypasses
        ),
        transpile_cache_eviction_count=(
            working_cache.eviction_count - initial_evictions
        ),
        transpile_cache_maximum_entries=working_cache.maximum_entries,
        planned_build_requests=workload_plan.circuit_build_request_count,
        actual_build_requests=actual_build_requests,
        planned_transpile_requests=workload_plan.transpile_cache_request_count,
        actual_cache_requests=actual_cache_requests,
        planned_instruction_applications=(
            workload_plan.planned_untranspiled_instruction_applications
        ),
        actual_built_instruction_total=actual_built_instruction_total,
        workload_budget=workload_budget,
        workload_policy_version=workload_plan.workload_policy_version,
    )


def estimate_exact_compiled_rpe_hadamard_cost(
    preparation: DFPartialS2Preparation,
    step_time: float,
    repetition_count: int,
    rte_config: RTEConfig | None,
    rte_distribution: RTEFiniteDistribution | None,
    compiler: CompilerSettings,
    *,
    construction_policy: RepeatedCircuitConstructionPolicy = (
        "boundary_optimized"
    ),
    maximum_trajectories: int = 10_000,
    maximum_untranspiled_circuit_size: int = 100_000,
    maximum_retained_trajectory_records: int = 1_024,
    maximum_build_requests: int = 1_000_000,
    maximum_transpile_requests: int = 1_000_000,
    maximum_planned_instruction_applications: int = 100_000_000,
    cache: TranspiledCircuitCostCache | None = None,
    backend: Any | None = None,
) -> DFRPEHadamardCompiledCostEstimate:
    """Exactly weight complete X/Y wrapper costs over all trajectories."""
    stream = make_exact_df_partial_s2_repeated_trajectory_stream(
        preparation,
        step_time,
        repetition_count,
        rte_config,
        rte_distribution,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        construction_policy=construction_policy,
        maximum_trajectories=maximum_trajectories,
    )
    plan = plan_compiled_rpe_hadamard_workload(
        preparation,
        stream.repetition_count,
        rte_config,
        rte_distribution,
        trajectory_count=stream.expected_record_count,
    )
    return _compile_hadamard_trajectory_stream_with_plan(
        stream,
        step_time,
        compiler,
        construction_policy=construction_policy,
        workload_plan=plan,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        maximum_retained_trajectory_records=(
            maximum_retained_trajectory_records
        ),
        maximum_build_requests=maximum_build_requests,
        maximum_transpile_requests=maximum_transpile_requests,
        maximum_planned_instruction_applications=(
            maximum_planned_instruction_applications
        ),
        cache=cache,
        backend=backend,
    )


def estimate_monte_carlo_compiled_rpe_hadamard_cost(
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
    construction_policy: RepeatedCircuitConstructionPolicy = (
        "boundary_optimized"
    ),
    maximum_untranspiled_circuit_size: int = 100_000,
    maximum_retained_trajectory_records: int = 1_024,
    maximum_build_requests: int = 1_000_000,
    maximum_transpile_requests: int = 1_000_000,
    maximum_planned_instruction_applications: int = 100_000_000,
    cache: TranspiledCircuitCostCache | None = None,
    backend: Any | None = None,
) -> DFRPEHadamardCompiledCostEstimate:
    """Estimate complete X/Y wrapper costs from shared sampled trajectories."""
    stream = make_monte_carlo_df_partial_s2_repeated_trajectory_stream(
        preparation,
        step_time,
        repetition_count,
        rte_config,
        rte_distribution,
        sample_count=sample_count,
        seed=seed,
        maximum_samples=maximum_samples,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        construction_policy=construction_policy,
    )
    plan = plan_compiled_rpe_hadamard_workload(
        preparation,
        stream.repetition_count,
        rte_config,
        rte_distribution,
        trajectory_count=stream.expected_record_count,
    )
    return _compile_hadamard_trajectory_stream_with_plan(
        stream,
        step_time,
        compiler,
        construction_policy=construction_policy,
        workload_plan=plan,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        maximum_retained_trajectory_records=(
            maximum_retained_trajectory_records
        ),
        maximum_build_requests=maximum_build_requests,
        maximum_transpile_requests=maximum_transpile_requests,
        maximum_planned_instruction_applications=(
            maximum_planned_instruction_applications
        ),
        cache=cache,
        backend=backend,
    )


@dataclass(frozen=True)
class DFRPEHadamardCompiledCostProvider:
    """Adapt complete short Hadamard wrapper costs to RPE accounting."""

    compiler: CompilerSettings
    evaluation_method: DFRPEHadamardCostEvaluationMethod = "exact"
    sample_count: int | None = None
    seed: int | None = None
    construction_policy: RepeatedCircuitConstructionPolicy = (
        "boundary_optimized"
    )
    maximum_repetition_count: int = 4
    maximum_trajectories: int = 10_000
    maximum_samples: int = 10_000
    maximum_untranspiled_circuit_size: int = 100_000
    maximum_retained_trajectory_records: int = 1_024
    maximum_build_requests: int = 1_000_000
    maximum_transpile_requests: int = 1_000_000
    maximum_planned_instruction_applications: int = 100_000_000
    cache: TranspiledCircuitCostCache | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    backend: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.compiler, CompilerSettings):
            raise TypeError("compiler must be a CompilerSettings instance.")
        if self.evaluation_method not in ("exact", "monte_carlo"):
            raise ValueError("evaluation_method must be 'exact' or 'monte_carlo'.")
        if self.construction_policy not in (
            "raw_concatenation",
            "boundary_optimized",
        ):
            raise ValueError("Unsupported repeated circuit construction policy.")
        for name, minimum in (
            ("maximum_repetition_count", 1),
            ("maximum_trajectories", 1),
            ("maximum_samples", 1),
            ("maximum_untranspiled_circuit_size", 1),
            ("maximum_retained_trajectory_records", 0),
            ("maximum_build_requests", 1),
            ("maximum_transpile_requests", 1),
            ("maximum_planned_instruction_applications", 1),
        ):
            object.__setattr__(
                self,
                name,
                require_integer_count(getattr(self, name), name=name, minimum=minimum),
            )
        if self.maximum_repetition_count > 4:
            raise ValueError("maximum_repetition_count must not exceed 4.")
        if self.evaluation_method == "exact":
            if self.sample_count is not None or self.seed is not None:
                raise ValueError(
                    "Exact compiled-cost evaluation does not accept sample_count "
                    "or seed."
                )
        else:
            if self.sample_count is None or self.seed is None:
                raise ValueError(
                    "Monte Carlo evaluation requires sample_count and seed."
                )
            object.__setattr__(
                self,
                "sample_count",
                require_integer_count(
                    self.sample_count,
                    name="sample_count",
                    minimum=1,
                ),
            )
            object.__setattr__(
                self,
                "seed",
                require_integer_count(self.seed, name="seed"),
            )
            if self.sample_count > self.maximum_samples:
                raise ValueError(
                    f"sample_count={self.sample_count} exceeds "
                    f"maximum_samples={self.maximum_samples}."
                )

    def __call__(self, request: RPERoundCostRequest) -> RPERoundCompiledCost:
        return self.evaluate(request)

    def _validate_round_request(
        self,
        request: RPERoundCostRequest,
    ) -> tuple[DFPartialS2Preparation, float, int, bool]:
        if not isinstance(request, RPERoundCostRequest):
            raise TypeError("request must be an RPERoundCostRequest instance.")
        preparation = request.preparation
        if not isinstance(preparation, DFPartialS2Preparation):
            raise TypeError("request.preparation must be a DFPartialS2Preparation.")
        repetition_count = require_integer_count(
            request.specification.q_m,
            name="specification.q_m",
            minimum=1,
        )
        round_index_for_short_rpe_repetition_count(repetition_count)
        if repetition_count > self.maximum_repetition_count:
            raise ValueError(
                f"q_m={repetition_count} exceeds maximum_repetition_count="
                f"{self.maximum_repetition_count}."
            )
        step_time = float(request.specification.delta_time)
        if not math.isfinite(step_time) or step_time <= 0.0:
            raise ValueError("specification.delta_time must be finite and positive.")
        deterministic_only = preparation.is_deterministic_only
        rte_steps = require_integer_count(
            request.rte_steps_per_occurrence,
            name="rte_steps_per_occurrence",
            minimum=0 if deterministic_only else 1,
        )
        finite_order = require_integer_count(
            request.finite_taylor_order,
            name="finite_taylor_order",
        )
        if finite_order % 2:
            raise ValueError("finite_taylor_order must be non-negative and even.")
        config = request.rte_config
        distribution = request.rte_distribution
        if deterministic_only:
            if rte_steps != 0 or finite_order != 0:
                raise ValueError(
                    "A deterministic tail requires zero RTE steps and cutoff."
                )
            if config is not None or distribution is not None:
                raise ValueError("A deterministic tail requires no RTE inputs.")
        else:
            if config is None or distribution is None:
                raise ValueError("A randomized tail requires RTE inputs.")
            if config.rte_steps != rte_steps:
                raise ValueError("RTE config steps do not match the request.")
            if config.finite_taylor_order != finite_order:
                raise ValueError("RTE config cutoff does not match the request.")
            if not math.isclose(
                config.evolution_time,
                step_time,
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise ValueError("RTE evolution_time must equal delta_time.")
        return preparation, step_time, repetition_count, deterministic_only

    def estimate_round(
        self,
        request: RPERoundCostRequest,
    ) -> DFRPEHadamardCompiledCostEstimate:
        preparation, step_time, repetition_count, deterministic_only = (
            self._validate_round_request(request)
        )
        common = {
            "construction_policy": self.construction_policy,
            "maximum_untranspiled_circuit_size": (
                self.maximum_untranspiled_circuit_size
            ),
            "maximum_retained_trajectory_records": (
                self.maximum_retained_trajectory_records
            ),
            "maximum_build_requests": self.maximum_build_requests,
            "maximum_transpile_requests": self.maximum_transpile_requests,
            "maximum_planned_instruction_applications": (
                self.maximum_planned_instruction_applications
            ),
            "cache": self.cache,
            "backend": self.backend,
        }
        if deterministic_only or self.evaluation_method == "exact":
            return estimate_exact_compiled_rpe_hadamard_cost(
                preparation,
                step_time,
                repetition_count,
                request.rte_config,
                request.rte_distribution,
                self.compiler,
                maximum_trajectories=self.maximum_trajectories,
                **common,
            )
        if self.sample_count is None or self.seed is None:  # pragma: no cover
            raise RuntimeError("Monte Carlo provider is not configured.")
        return estimate_monte_carlo_compiled_rpe_hadamard_cost(
            preparation,
            step_time,
            repetition_count,
            request.rte_config,
            request.rte_distribution,
            self.compiler,
            sample_count=self.sample_count,
            seed=self.seed,
            maximum_samples=self.maximum_samples,
            **common,
        )

    def _cost_model_fingerprint(self) -> tuple[str | None, dict[str, Any]]:
        backend_fingerprint = canonical_backend_fingerprint_or_none(self.backend)
        payload = {
            "provider_version": DF_RPE_HADAMARD_COMPILED_COST_PROVIDER_VERSION,
            "compiler_settings_hash": compiler_settings_hash(self.compiler),
            "backend_fingerprint": backend_fingerprint,
            "evaluation_method": self.evaluation_method,
            "sample_count": self.sample_count,
            "seed": self.seed,
            "construction_policy": self.construction_policy,
            "circuit_scope": RPE_HADAMARD_INTERROGATION_SCOPE,
            "measurement_policy": DF_RPE_HADAMARD_MEASUREMENT_POLICY,
            "fidelity_level": 5,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "rte_prng_type": RTE_PRNG_TYPE,
            "rte_sampling_convention_version": RTE_SAMPLING_CONVENTION_VERSION,
            "rte_finite_distribution_schema_version": (
                RTE_FINITE_DISTRIBUTION_SCHEMA_VERSION
            ),
            "fingerprint_policy": "df_rpe_hadamard_cost_model_v1",
        }
        fingerprint = (
            _sha256_json(payload) if backend_fingerprint is not None else None
        )
        return fingerprint, payload

    def evaluate(self, request: RPERoundCostRequest) -> RPERoundCompiledCost:
        estimate = self.estimate_round(request)
        cost_model_fingerprint, payload = self._cost_model_fingerprint()
        deterministic_only = request.preparation.is_deterministic_only
        return RPERoundCompiledCost(
            cosine_expected_cost=estimate.cosine.expected_cost,
            sine_expected_cost=estimate.sine.expected_cost,
            cosine_standard_error=estimate.cosine.standard_error,
            sine_standard_error=estimate.sine.standard_error,
            evaluation_method=estimate.evaluation_method,
            classical_sample_count=estimate.sample_count,
            circuit_cost_scope=RPE_HADAMARD_INTERROGATION_SCOPE,
            cost_model_fingerprint=cost_model_fingerprint,
            metadata=(
                (
                    "provider_version",
                    DF_RPE_HADAMARD_COMPILED_COST_PROVIDER_VERSION,
                ),
                ("cost_model_fingerprint_policy", "df_rpe_hadamard_cost_model_v1"),
                ("compiler_settings_hash", payload["compiler_settings_hash"]),
                ("backend_fingerprint", estimate.backend_fingerprint),
                (
                    "backend_context_canonical",
                    estimate.backend_context_canonical,
                ),
                ("requested_evaluation_method", self.evaluation_method),
                ("actual_evaluation_method", estimate.evaluation_method),
                ("estimate_kind", estimate.estimate_kind),
                ("construction_policy", estimate.construction_policy),
                ("round_index", estimate.round_index),
                ("q_m", estimate.q_m),
                ("delta_time", estimate.delta_time),
                ("t_m", estimate.t_m),
                ("trajectory_space_size", estimate.trajectory_space_size),
                (
                    "processed_trajectory_count",
                    estimate.processed_trajectory_count,
                ),
                ("configured_classical_sample_count", self.sample_count),
                ("classical_sample_count", estimate.sample_count),
                (
                    "classical_samples_are_quantum_shots",
                    estimate.classical_samples_are_quantum_shots,
                ),
                ("shared_axis_trajectory_set", estimate.shared_axis_trajectory_set),
                ("measurement_policy", estimate.measurement_policy),
                ("measurements_included", estimate.measurement_included),
                ("hadamard_test_included", True),
                (
                    "state_preparation_included",
                    estimate.state_preparation_included,
                ),
                ("backend_execution_included", False),
                ("quantum_shots_executed", 0),
                (
                    "fresh_iid_trajectory_per_hadamard_shot_verified",
                    False,
                ),
                ("ordinary_control_semantics", "diag(I,U)"),
                ("complete", estimate.complete),
                (
                    "compiled_cost_evaluation_fingerprint",
                    estimate.compiled_cost_evaluation_fingerprint,
                ),
                (
                    "cosine_compiled_cost_evaluation_fingerprint",
                    estimate.cosine.compiled_cost_evaluation_fingerprint,
                ),
                (
                    "sine_compiled_cost_evaluation_fingerprint",
                    estimate.sine.compiled_cost_evaluation_fingerprint,
                ),
                ("deterministic_only", deterministic_only),
                ("deterministic_exact_short_circuit", deterministic_only),
                ("rte_prng_type", payload["rte_prng_type"]),
                (
                    "rte_sampling_convention_version",
                    payload["rte_sampling_convention_version"],
                ),
            ),
        )

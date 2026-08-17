"""Reproducible medium-q compiled-cost benchmark datasets for RPE wrappers.

This is a validation-only path.  It directly transpiles complete, measured
Hadamard-interrogation wrappers without state preparation or backend execution.
It deliberately does not fit a proxy or feed one into resource accounting.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias

from .df_partial_s2 import DFPartialS2Preparation
from .df_partial_s2_repeated import RepeatedCircuitConstructionPolicy
from .df_partial_s2_repeated_cost import (
    make_exact_df_partial_s2_repeated_trajectory_stream,
    make_monte_carlo_df_partial_s2_repeated_trajectory_stream,
    plan_compiled_repeated_partial_s2_workload,
)
from .df_rpe_hadamard_compiled_cost import (
    DF_RPE_HADAMARD_MEASUREMENT_POLICY,
    DFRPEHadamardCompiledCostEstimate,
    DFRPEHadamardCompiledMetricValues,
    _compile_hadamard_trajectory_stream_with_plan,
)
from .rpe_hadamard_interrogation import (
    RPE_HADAMARD_INTERROGATION_SCOPE,
    QiskitRPEHadamardInterrogationBuilder,
    RPEHadamardAxis,
    RPEHadamardInterrogationRequest,
    RPEHadamardInterrogationResult,
)
from .rpe_resource_accounting import RPE_COST_METRICS
from .rte import (
    CompilerSettings,
    RTEConfig,
    RTEFiniteDistribution,
    require_integer_count,
)
from .rte_compiled_cost import (
    CompiledCostWorkloadPlan,
    CompiledMetricStatistics,
    TranspiledCircuitCostCache,
    canonical_backend_fingerprint_or_none,
    compiler_settings_hash,
    plan_compiled_cost_workload,
)


BenchmarkPartition: TypeAlias = Literal["calibration", "holdout"]
BenchmarkPointStatus: TypeAlias = Literal["complete", "failed"]
BenchmarkEvaluationMethod: TypeAlias = Literal["exact", "monte_carlo"]

RPE_HADAMARD_COMPILED_COST_BENCHMARK_SCHEMA_VERSION = (
    "rpe_hadamard_compiled_cost_benchmark_dataset_v1"
)
RPE_HADAMARD_COMPILED_COST_BENCHMARK_POINT_SCHEMA_VERSION = (
    "rpe_hadamard_compiled_cost_benchmark_point_v1"
)
RPE_HADAMARD_BENCHMARK_PATH = "medium_q_validation_benchmark"
RPE_HADAMARD_CONTROL_CONVENTION = "ordinary_controlled_diag_I_U_m"


def _canonical_value(value: Any) -> Any:
    """Normalize values for stable hashes while retaining exact float bits."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Benchmark fingerprints require finite floats.")
        return {"float_hex": value.hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    return value


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def round_index_for_benchmark_repetition_count(
    repetition_count: int,
    *,
    maximum_repetition_count: int,
) -> int:
    """Validate a bounded positive power of two and return its RPE index."""
    count = require_integer_count(
        repetition_count,
        name="repetition_count",
        minimum=1,
    )
    maximum = require_integer_count(
        maximum_repetition_count,
        name="maximum_repetition_count",
        minimum=1,
    )
    if count > maximum:
        raise ValueError(
            f"repetition_count={count} exceeds maximum_repetition_count={maximum}."
        )
    if count & (count - 1):
        raise ValueError("Benchmark repetition_count must be a positive power of two.")
    return count.bit_length() - 1


class QiskitRPEHadamardBenchmarkCircuitBuilder:
    """Validation-only bounded-power-of-two wrapper construction path."""

    def __init__(self, *, maximum_repetition_count: int) -> None:
        self.maximum_repetition_count = require_integer_count(
            maximum_repetition_count,
            name="maximum_repetition_count",
            minimum=1,
        )
        self._shared_builder = QiskitRPEHadamardInterrogationBuilder()

    def build(
        self,
        request: RPEHadamardInterrogationRequest,
    ) -> RPEHadamardInterrogationResult:
        if not isinstance(request, RPEHadamardInterrogationRequest):
            raise TypeError("request must be an RPEHadamardInterrogationRequest.")
        round_index = round_index_for_benchmark_repetition_count(
            request.evolution.repetition_count,
            maximum_repetition_count=self.maximum_repetition_count,
        )
        ancilla, total_time = self._shared_builder._validate_evolution_common(
            request.evolution
        )
        return self._shared_builder._build_validated(
            request,
            round_index=round_index,
            ancilla=ancilla,
            total_time=total_time,
        )


@dataclass(frozen=True)
class RPEHadamardBenchmarkTrajectoryRecord:
    """One retained trajectory contribution for one benchmark axis."""

    trajectory_index: int
    probability: float | None
    trajectory_seed: int | None
    step_seeds: tuple[int | None, ...]
    evolution_provenance_fingerprint: str
    evolution_circuit_semantics_fingerprint: str
    wrapper_provenance_fingerprint: str
    wrapper_circuit_semantics_fingerprint: str
    actual_circuit_fingerprint: str | None
    cost: DFRPEHadamardCompiledMetricValues
    constant_phase: float
    extracted_identity_phase: float
    rte_relative_phase: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "step_seeds": list(self.step_seeds),
            "cost": asdict(self.cost),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardBenchmarkTrajectoryRecord":
        values = dict(payload)
        values["step_seeds"] = tuple(values["step_seeds"])
        values["cost"] = DFRPEHadamardCompiledMetricValues(**values["cost"])
        return cls(**values)


def _statistics_to_dict(
    statistics: tuple[tuple[str, CompiledMetricStatistics], ...],
) -> dict[str, dict[str, float | None]]:
    return {name: asdict(value) for name, value in statistics}


def _statistics_from_dict(
    payload: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, CompiledMetricStatistics], ...]:
    if not payload:
        return ()
    return tuple(
        (name, CompiledMetricStatistics(**payload[name]))
        for name in RPE_COST_METRICS
    )


@dataclass(frozen=True)
class RPEHadamardCompiledCostBenchmarkPoint:
    """One axis expectation or an explicit failed benchmark point."""

    partition: BenchmarkPartition
    round_index: int
    repetition_count: int
    q_m: int
    delta_time: float
    t_m: float
    rte_steps_per_occurrence: int
    finite_taylor_order: int
    tail_kind: Literal["deterministic", "randomized"]
    axis: RPEHadamardAxis
    evaluation_method: BenchmarkEvaluationMethod
    sample_count: int | None
    enumerated_trajectory_count: int | None
    trajectory_space_size: int | None
    master_seed: int | None
    sampled_trajectory_seeds: tuple[int, ...] | None
    step_seed_hierarchy: tuple[tuple[int | None, ...], ...]
    trajectory_probability_sum: float | None
    trajectory_provenance_digest: str | None
    sampling_provenance_fingerprint: str
    evolution_circuit_semantics_digest: str | None
    wrapper_circuit_semantics_digest: str | None
    actual_circuit_fingerprint_digest: str | None
    retained_trajectory_records: tuple[RPEHadamardBenchmarkTrajectoryRecord, ...]
    trajectory_records_truncated: bool
    metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    compiler_settings: CompilerSettings
    compiler_settings_fingerprint: str
    backend_fingerprint: str | None
    compiler_context_fingerprint: str
    evaluation_configuration_fingerprint: str
    construction_policy: RepeatedCircuitConstructionPolicy
    measurement_included: bool
    state_preparation_included: bool
    backend_execution_included: bool
    quantum_shots_executed: int
    wrapped_evolution_already_controlled: bool
    additional_control_applied: bool
    fresh_iid_trajectory_per_hadamard_shot_verified: bool
    benchmark_validation_path: bool
    benchmark_path: str
    circuit_scope: str
    status: BenchmarkPointStatus
    failure_reason: str | None
    planned_build_requests: int | None
    actual_build_requests: int | None
    planned_transpile_requests: int | None
    actual_transpile_requests: int | None
    planned_instruction_applications: int | None
    actual_built_instruction_total: int | None
    point_fingerprint: str = ""
    schema_version: str = RPE_HADAMARD_COMPILED_COST_BENCHMARK_POINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.partition not in ("calibration", "holdout"):
            raise ValueError("Unsupported benchmark partition.")
        if self.axis not in ("cosine", "sine"):
            raise ValueError("Unsupported benchmark axis.")
        if self.evaluation_method not in ("exact", "monte_carlo"):
            raise ValueError("Unsupported benchmark evaluation method.")
        if self.repetition_count != self.q_m or self.q_m < 1:
            raise ValueError(
                "repetition_count and q_m must be the same positive value."
            )
        if self.q_m & (self.q_m - 1):
            raise ValueError("A benchmark point q_m must be a power of two.")
        if self.round_index != self.q_m.bit_length() - 1:
            raise ValueError("round_index must equal log2(q_m).")
        if not math.isfinite(self.delta_time) or self.delta_time <= 0.0:
            raise ValueError("Benchmark delta_time must be finite and positive.")
        if self.t_m != float(self.q_m * self.delta_time):
            raise ValueError("Benchmark t_m must equal q_m*delta_time.")
        if self.benchmark_path != RPE_HADAMARD_BENCHMARK_PATH:
            raise ValueError("Unsupported benchmark validation path.")
        if self.circuit_scope != RPE_HADAMARD_INTERROGATION_SCOPE:
            raise ValueError("Unsupported benchmark circuit scope.")
        if self.metric_statistics:
            statistics_by_name = dict(self.metric_statistics)
            if set(statistics_by_name) != set(RPE_COST_METRICS):
                raise ValueError(
                    "Complete metric statistics must contain all cost metrics."
                )
            object.__setattr__(
                self,
                "metric_statistics",
                tuple((name, statistics_by_name[name]) for name in RPE_COST_METRICS),
            )
        if self.status == "complete" and self.failure_reason is not None:
            raise ValueError("A complete benchmark point cannot have a failure reason.")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("A failed benchmark point requires a failure reason.")
        if self.status == "complete" and not self.metric_statistics:
            raise ValueError("A complete benchmark point requires metric statistics.")
        if self.status == "failed" and self.metric_statistics:
            raise ValueError(
                "A failed benchmark point cannot contain metric statistics."
            )
        if self.status == "complete" and (
            not self.measurement_included
            or self.state_preparation_included
            or self.backend_execution_included
            or self.quantum_shots_executed != 0
            or not self.wrapped_evolution_already_controlled
            or self.additional_control_applied
        ):
            raise ValueError("Complete benchmark point scope flags are inconsistent.")
        expected = _sha256_json(self._fingerprint_payload())
        if self.point_fingerprint and self.point_fingerprint != expected:
            raise ValueError("Benchmark point fingerprint does not match its content.")
        object.__setattr__(self, "point_fingerprint", expected)

    @property
    def metric_means(self) -> dict[str, float]:
        return {name: value.mean for name, value in self.metric_statistics}

    def _metric_mean(self, name: str) -> float | None:
        return self.metric_means.get(name)

    @property
    def rz_count(self) -> float | None:
        return self._metric_mean("rz_count")

    @property
    def rz_depth(self) -> float | None:
        return self._metric_mean("rz_depth")

    @property
    def cx_count(self) -> float | None:
        return self._metric_mean("cx_count")

    @property
    def cx_depth(self) -> float | None:
        return self._metric_mean("cx_depth")

    @property
    def total_depth(self) -> float | None:
        return self._metric_mean("total_depth")

    @property
    def circuit_size(self) -> float | None:
        return self._metric_mean("circuit_size")

    def _payload(self, *, include_fingerprint: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "partition": self.partition,
            "round_index": self.round_index,
            "m": self.round_index,
            "repetition_count": self.repetition_count,
            "q_m": self.q_m,
            "delta_time": self.delta_time,
            "t_m": self.t_m,
            "r_m": self.rte_steps_per_occurrence,
            "K_m": self.finite_taylor_order,
            "tail_kind": self.tail_kind,
            "axis": self.axis,
            "evaluation_method": self.evaluation_method,
            "sample_count": self.sample_count,
            "enumerated_trajectory_count": self.enumerated_trajectory_count,
            "trajectory_space_size": self.trajectory_space_size,
            "master_seed": self.master_seed,
            "sampled_trajectory_seeds": (
                None
                if self.sampled_trajectory_seeds is None
                else list(self.sampled_trajectory_seeds)
            ),
            "step_seed_hierarchy": [list(seeds) for seeds in self.step_seed_hierarchy],
            "trajectory_probability_sum": self.trajectory_probability_sum,
            "trajectory_provenance_digest": self.trajectory_provenance_digest,
            "sampling_provenance_fingerprint": self.sampling_provenance_fingerprint,
            "evolution_circuit_semantics_digest": (
                self.evolution_circuit_semantics_digest
            ),
            "wrapper_circuit_semantics_digest": (
                self.wrapper_circuit_semantics_digest
            ),
            "actual_circuit_fingerprint_digest": (
                self.actual_circuit_fingerprint_digest
            ),
            "retained_trajectory_records": [
                record.to_dict() for record in self.retained_trajectory_records
            ],
            "trajectory_records_truncated": self.trajectory_records_truncated,
            "cost": self.metric_means,
            "rz_count": self.rz_count,
            "rz_depth": self.rz_depth,
            "cx_count": self.cx_count,
            "cx_depth": self.cx_depth,
            "total_depth": self.total_depth,
            "circuit_size": self.circuit_size,
            "metric_statistics": _statistics_to_dict(self.metric_statistics),
            "transpile_configuration": asdict(self.compiler_settings),
            "compiler_settings_fingerprint": self.compiler_settings_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "compiler_context_fingerprint": self.compiler_context_fingerprint,
            "evaluation_configuration_fingerprint": (
                self.evaluation_configuration_fingerprint
            ),
            "construction_policy": self.construction_policy,
            "measurement_included": self.measurement_included,
            "state_preparation_included": self.state_preparation_included,
            "backend_execution_included": self.backend_execution_included,
            "quantum_shots_executed": self.quantum_shots_executed,
            "wrapped_evolution_already_controlled": (
                self.wrapped_evolution_already_controlled
            ),
            "additional_control_applied": self.additional_control_applied,
            "fresh_iid_trajectory_per_hadamard_shot_verified": (
                self.fresh_iid_trajectory_per_hadamard_shot_verified
            ),
            "benchmark_validation_path": self.benchmark_validation_path,
            "benchmark_path": self.benchmark_path,
            "circuit_scope": self.circuit_scope,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "workload": {
                "planned_build_requests": self.planned_build_requests,
                "actual_build_requests": self.actual_build_requests,
                "planned_transpile_requests": self.planned_transpile_requests,
                "actual_transpile_requests": self.actual_transpile_requests,
                "planned_instruction_applications": (
                    self.planned_instruction_applications
                ),
                "actual_built_instruction_total": (
                    self.actual_built_instruction_total
                ),
            },
        }
        if include_fingerprint:
            payload["point_fingerprint"] = self.point_fingerprint
        return payload

    def _fingerprint_payload(self) -> dict[str, Any]:
        return self._payload(include_fingerprint=False)

    def to_dict(self) -> dict[str, Any]:
        return self._payload(include_fingerprint=True)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardCompiledCostBenchmarkPoint":
        compiler_payload = dict(payload["transpile_configuration"])
        compiler_payload["basis_gates"] = tuple(compiler_payload["basis_gates"])
        if compiler_payload["coupling_map"] is not None:
            compiler_payload["coupling_map"] = tuple(
                tuple(edge) for edge in compiler_payload["coupling_map"]
            )
        workload = payload["workload"]
        metric_statistics = _statistics_from_dict(payload["metric_statistics"])
        metric_means = {name: value.mean for name, value in metric_statistics}
        if payload["cost"] != metric_means or any(
            payload[name] != metric_means.get(name) for name in RPE_COST_METRICS
        ):
            raise ValueError("Serialized benchmark cost fields are inconsistent.")
        return cls(
            partition=payload["partition"],
            round_index=int(payload["round_index"]),
            repetition_count=int(payload["repetition_count"]),
            q_m=int(payload["q_m"]),
            delta_time=float(payload["delta_time"]),
            t_m=float(payload["t_m"]),
            rte_steps_per_occurrence=int(payload["r_m"]),
            finite_taylor_order=int(payload["K_m"]),
            tail_kind=payload["tail_kind"],
            axis=payload["axis"],
            evaluation_method=payload["evaluation_method"],
            sample_count=payload["sample_count"],
            enumerated_trajectory_count=payload["enumerated_trajectory_count"],
            trajectory_space_size=payload["trajectory_space_size"],
            master_seed=payload["master_seed"],
            sampled_trajectory_seeds=(
                None
                if payload["sampled_trajectory_seeds"] is None
                else tuple(payload["sampled_trajectory_seeds"])
            ),
            step_seed_hierarchy=tuple(
                tuple(seeds) for seeds in payload["step_seed_hierarchy"]
            ),
            trajectory_probability_sum=payload["trajectory_probability_sum"],
            trajectory_provenance_digest=payload["trajectory_provenance_digest"],
            sampling_provenance_fingerprint=(
                payload["sampling_provenance_fingerprint"]
            ),
            evolution_circuit_semantics_digest=(
                payload["evolution_circuit_semantics_digest"]
            ),
            wrapper_circuit_semantics_digest=(
                payload["wrapper_circuit_semantics_digest"]
            ),
            actual_circuit_fingerprint_digest=(
                payload["actual_circuit_fingerprint_digest"]
            ),
            retained_trajectory_records=tuple(
                RPEHadamardBenchmarkTrajectoryRecord.from_dict(record)
                for record in payload["retained_trajectory_records"]
            ),
            trajectory_records_truncated=bool(payload["trajectory_records_truncated"]),
            metric_statistics=metric_statistics,
            compiler_settings=CompilerSettings(**compiler_payload),
            compiler_settings_fingerprint=payload["compiler_settings_fingerprint"],
            backend_fingerprint=payload["backend_fingerprint"],
            compiler_context_fingerprint=payload["compiler_context_fingerprint"],
            evaluation_configuration_fingerprint=(
                payload["evaluation_configuration_fingerprint"]
            ),
            construction_policy=payload["construction_policy"],
            measurement_included=bool(payload["measurement_included"]),
            state_preparation_included=bool(payload["state_preparation_included"]),
            backend_execution_included=bool(payload["backend_execution_included"]),
            quantum_shots_executed=int(payload["quantum_shots_executed"]),
            wrapped_evolution_already_controlled=bool(
                payload["wrapped_evolution_already_controlled"]
            ),
            additional_control_applied=bool(payload["additional_control_applied"]),
            fresh_iid_trajectory_per_hadamard_shot_verified=bool(
                payload["fresh_iid_trajectory_per_hadamard_shot_verified"]
            ),
            benchmark_validation_path=bool(payload["benchmark_validation_path"]),
            benchmark_path=payload["benchmark_path"],
            circuit_scope=payload["circuit_scope"],
            status=payload["status"],
            failure_reason=payload["failure_reason"],
            planned_build_requests=workload["planned_build_requests"],
            actual_build_requests=workload["actual_build_requests"],
            planned_transpile_requests=workload["planned_transpile_requests"],
            actual_transpile_requests=workload["actual_transpile_requests"],
            planned_instruction_applications=(
                workload["planned_instruction_applications"]
            ),
            actual_built_instruction_total=workload["actual_built_instruction_total"],
            point_fingerprint=payload["point_fingerprint"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class RPEHadamardCompiledCostBenchmarkRequest:
    """Configuration for calibration/holdout reference-cost generation."""

    preparation: DFPartialS2Preparation
    delta_time: float
    calibration_repetition_counts: tuple[int, ...]
    holdout_repetition_counts: tuple[int, ...]
    rte_steps_per_occurrence: int
    finite_taylor_order: int
    rte_config: RTEConfig | None
    rte_distribution: RTEFiniteDistribution | None
    compiler: CompilerSettings
    evaluation_method: BenchmarkEvaluationMethod = "exact"
    sample_count: int | None = None
    seed: int | None = None
    generation_id: str = "manual"
    maximum_repetition_count: int = 64
    maximum_trajectories: int = 10_000
    maximum_samples: int = 10_000
    maximum_untranspiled_circuit_size: int = 100_000
    maximum_retained_trajectory_records: int = 1_024
    maximum_build_requests: int = 1_000_000
    maximum_transpile_requests: int = 1_000_000
    maximum_planned_instruction_applications: int = 100_000_000
    construction_policy: RepeatedCircuitConstructionPolicy = "boundary_optimized"
    cache: TranspiledCircuitCostCache | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    backend: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.preparation, DFPartialS2Preparation):
            raise TypeError("preparation must be a DFPartialS2Preparation.")
        if not isinstance(self.compiler, CompilerSettings):
            raise TypeError("compiler must be a CompilerSettings instance.")
        delta = float(self.delta_time)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta_time must be finite and positive.")
        object.__setattr__(self, "delta_time", delta)
        if not isinstance(self.generation_id, str) or not self.generation_id.strip():
            raise ValueError("generation_id must not be empty.")
        object.__setattr__(self, "generation_id", self.generation_id.strip())
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
        calibration = self._normalize_partition(
            self.calibration_repetition_counts,
            name="calibration_repetition_counts",
        )
        holdout = self._normalize_partition(
            self.holdout_repetition_counts,
            name="holdout_repetition_counts",
        )
        if set(calibration).intersection(holdout):
            raise ValueError(
                "Calibration and holdout repetition counts must be disjoint."
            )
        if not calibration and not holdout:
            raise ValueError("At least one benchmark repetition count is required.")
        object.__setattr__(self, "calibration_repetition_counts", calibration)
        object.__setattr__(self, "holdout_repetition_counts", holdout)
        rte_steps = require_integer_count(
            self.rte_steps_per_occurrence,
            name="rte_steps_per_occurrence",
        )
        finite_order = require_integer_count(
            self.finite_taylor_order,
            name="finite_taylor_order",
        )
        if finite_order % 2:
            raise ValueError("finite_taylor_order must be non-negative and even.")
        object.__setattr__(self, "rte_steps_per_occurrence", rte_steps)
        object.__setattr__(self, "finite_taylor_order", finite_order)
        if self.preparation.is_deterministic_only:
            if rte_steps or finite_order or self.rte_config or self.rte_distribution:
                raise ValueError(
                    "A deterministic tail requires r_m=K_m=0 and no RTE inputs."
                )
            if self.evaluation_method != "exact":
                raise ValueError(
                    "A deterministic tail benchmark must use exact evaluation."
                )
        else:
            if self.rte_config is None or self.rte_distribution is None:
                raise ValueError(
                    "A randomized tail requires RTE config and distribution."
                )
            if self.rte_config.rte_steps != rte_steps:
                raise ValueError(
                    "rte_steps_per_occurrence must match rte_config.rte_steps."
                )
            if self.rte_config.finite_taylor_order != finite_order:
                raise ValueError("finite_taylor_order must match rte_config.")
        if self.evaluation_method == "exact":
            if self.sample_count is not None or self.seed is not None:
                raise ValueError(
                    "Exact benchmark evaluation does not accept sample_count or seed."
                )
        else:
            if self.sample_count is None or self.seed is None:
                raise ValueError(
                    "Monte Carlo benchmark evaluation requires sample_count and seed."
                )
            samples = require_integer_count(
                self.sample_count,
                name="sample_count",
                minimum=1,
            )
            master_seed = require_integer_count(self.seed, name="seed")
            if samples > self.maximum_samples:
                raise ValueError(
                    f"sample_count={samples} exceeds "
                    f"maximum_samples={self.maximum_samples}."
                )
            object.__setattr__(self, "sample_count", samples)
            object.__setattr__(self, "seed", master_seed)

    def _normalize_partition(
        self,
        values: tuple[int, ...],
        *,
        name: str,
    ) -> tuple[int, ...]:
        if not isinstance(values, tuple):
            raise TypeError(f"{name} must be a tuple.")
        normalized = tuple(sorted(set(values)))
        if len(normalized) != len(values):
            raise ValueError(f"{name} must not contain duplicates.")
        for count in normalized:
            round_index_for_benchmark_repetition_count(
                count,
                maximum_repetition_count=self.maximum_repetition_count,
            )
        return normalized


@dataclass(frozen=True)
class RPEHadamardCompiledCostBenchmarkDataset:
    """Versioned, fingerprinted calibration/holdout benchmark dataset."""

    generation_id: str
    calibration_repetition_counts: tuple[int, ...]
    holdout_repetition_counts: tuple[int, ...]
    preparation_fingerprint: str
    hamiltonian_fingerprint: str
    partition_fingerprint: str
    ld: int
    num_system_qubits: int
    delta_time: float
    rte_steps_per_occurrence: int
    finite_taylor_order: int
    product_formula_order: int
    control_convention: str
    construction_policy: RepeatedCircuitConstructionPolicy
    compiler_settings: CompilerSettings
    compiler_settings_fingerprint: str
    backend_fingerprint: str | None
    requested_evaluation_method: BenchmarkEvaluationMethod
    sample_count: int | None
    master_seed: int | None
    calibration_mc_seeds: tuple[tuple[int, int], ...]
    holdout_mc_seeds: tuple[tuple[int, int], ...]
    workload_limits: tuple[tuple[str, int], ...]
    records: tuple[RPEHadamardCompiledCostBenchmarkPoint, ...]
    proxy_fit_performed: bool = False
    holdout_used_for_proxy_fit: bool = False
    dataset_fingerprint: str = ""
    schema_version: str = RPE_HADAMARD_COMPILED_COST_BENCHMARK_SCHEMA_VERSION
    cost_metrics: tuple[str, ...] = RPE_COST_METRICS
    measurement_policy: str = DF_RPE_HADAMARD_MEASUREMENT_POLICY
    circuit_scope: str = RPE_HADAMARD_INTERROGATION_SCOPE
    benchmark_path: str = RPE_HADAMARD_BENCHMARK_PATH

    def __post_init__(self) -> None:
        if self.proxy_fit_performed or self.holdout_used_for_proxy_fit:
            raise ValueError("Benchmark dataset generation must not fit a proxy.")
        if self.benchmark_path != RPE_HADAMARD_BENCHMARK_PATH:
            raise ValueError("Unsupported benchmark validation path.")
        if self.circuit_scope != RPE_HADAMARD_INTERROGATION_SCOPE:
            raise ValueError("Unsupported benchmark circuit scope.")
        if self.control_convention != RPE_HADAMARD_CONTROL_CONVENTION:
            raise ValueError("Unsupported controlled-evolution convention.")
        if set(self.calibration_repetition_counts).intersection(
            self.holdout_repetition_counts
        ):
            raise ValueError("Calibration and holdout repetitions must be disjoint.")
        object.__setattr__(
            self,
            "records",
            tuple(sorted(self.records, key=lambda record: record.point_fingerprint)),
        )
        expected_keys = {
            (partition, count, axis)
            for partition, counts in (
                ("calibration", self.calibration_repetition_counts),
                ("holdout", self.holdout_repetition_counts),
            )
            for count in counts
            for axis in ("cosine", "sine")
        }
        actual_keys = {
            (record.partition, record.q_m, record.axis) for record in self.records
        }
        if len(actual_keys) != len(self.records) or actual_keys != expected_keys:
            raise ValueError(
                "Dataset records must cover every requested partition/q_m/axis once."
            )
        calibration_seeds = {seed for _q, seed in self.calibration_mc_seeds}
        holdout_seeds = {seed for _q, seed in self.holdout_mc_seeds}
        if calibration_seeds.intersection(holdout_seeds):
            raise ValueError("Calibration and holdout MC seed series must be disjoint.")
        expected = _sha256_json(self._fingerprint_payload())
        if self.dataset_fingerprint and self.dataset_fingerprint != expected:
            raise ValueError("Dataset fingerprint does not match its content.")
        object.__setattr__(self, "dataset_fingerprint", expected)

    @property
    def requested_point_count(self) -> int:
        return 2 * (
            len(self.calibration_repetition_counts)
            + len(self.holdout_repetition_counts)
        )

    @property
    def completed_point_count(self) -> int:
        return sum(record.status == "complete" for record in self.records)

    @property
    def failed_point_count(self) -> int:
        return sum(record.status == "failed" for record in self.records)

    @property
    def complete(self) -> bool:
        return (
            len(self.records) == self.requested_point_count
            and self.completed_point_count == self.requested_point_count
        )

    @property
    def incomplete_reasons(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    record.failure_reason
                    for record in self.records
                    if record.failure_reason is not None
                }
            )
        )

    def _payload(self, *, include_fingerprint: bool) -> dict[str, Any]:
        canonical_records = sorted(
            (record.to_dict() for record in self.records),
            key=lambda item: item["point_fingerprint"],
        )
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "calibration_repetition_counts": list(
                self.calibration_repetition_counts
            ),
            "holdout_repetition_counts": list(self.holdout_repetition_counts),
            "preparation_fingerprint": self.preparation_fingerprint,
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "partition_fingerprint": self.partition_fingerprint,
            "ld": self.ld,
            "num_system_qubits": self.num_system_qubits,
            "delta_time": self.delta_time,
            "rte_steps_per_occurrence": self.rte_steps_per_occurrence,
            "finite_taylor_order": self.finite_taylor_order,
            "product_formula_order": self.product_formula_order,
            "control_convention": self.control_convention,
            "construction_policy": self.construction_policy,
            "compiler_settings": asdict(self.compiler_settings),
            "compiler_settings_fingerprint": self.compiler_settings_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "cost_metrics": list(self.cost_metrics),
            "requested_evaluation_method": self.requested_evaluation_method,
            "sample_count": self.sample_count,
            "master_seed": self.master_seed,
            "calibration_mc_seeds": [list(item) for item in self.calibration_mc_seeds],
            "holdout_mc_seeds": [list(item) for item in self.holdout_mc_seeds],
            "measurement_policy": self.measurement_policy,
            "circuit_scope": self.circuit_scope,
            "benchmark_path": self.benchmark_path,
            "workload_limits": dict(self.workload_limits),
            "requested_point_count": self.requested_point_count,
            "completed_point_count": self.completed_point_count,
            "failed_point_count": self.failed_point_count,
            "complete": self.complete,
            "incomplete_reasons": list(self.incomplete_reasons),
            "proxy_fit_performed": self.proxy_fit_performed,
            "holdout_used_for_proxy_fit": self.holdout_used_for_proxy_fit,
            "records": canonical_records,
        }
        if include_fingerprint:
            payload["dataset_fingerprint"] = self.dataset_fingerprint
        return payload

    def _fingerprint_payload(self) -> dict[str, Any]:
        return self._payload(include_fingerprint=False)

    def to_dict(self) -> dict[str, Any]:
        return self._payload(include_fingerprint=True)

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "RPEHadamardCompiledCostBenchmarkDataset":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardCompiledCostBenchmarkDataset":
        if payload.get("schema_version") != (
            RPE_HADAMARD_COMPILED_COST_BENCHMARK_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported RPE Hadamard benchmark dataset schema.")
        compiler_payload = dict(payload["compiler_settings"])
        compiler_payload["basis_gates"] = tuple(compiler_payload["basis_gates"])
        if compiler_payload["coupling_map"] is not None:
            compiler_payload["coupling_map"] = tuple(
                tuple(edge) for edge in compiler_payload["coupling_map"]
            )
        dataset = cls(
            generation_id=payload["generation_id"],
            calibration_repetition_counts=tuple(
                payload["calibration_repetition_counts"]
            ),
            holdout_repetition_counts=tuple(payload["holdout_repetition_counts"]),
            preparation_fingerprint=payload["preparation_fingerprint"],
            hamiltonian_fingerprint=payload["hamiltonian_fingerprint"],
            partition_fingerprint=payload["partition_fingerprint"],
            ld=int(payload["ld"]),
            num_system_qubits=int(payload["num_system_qubits"]),
            delta_time=float(payload["delta_time"]),
            rte_steps_per_occurrence=int(payload["rte_steps_per_occurrence"]),
            finite_taylor_order=int(payload["finite_taylor_order"]),
            product_formula_order=int(payload["product_formula_order"]),
            control_convention=payload["control_convention"],
            construction_policy=payload["construction_policy"],
            compiler_settings=CompilerSettings(**compiler_payload),
            compiler_settings_fingerprint=payload["compiler_settings_fingerprint"],
            backend_fingerprint=payload["backend_fingerprint"],
            requested_evaluation_method=payload["requested_evaluation_method"],
            sample_count=payload["sample_count"],
            master_seed=payload["master_seed"],
            calibration_mc_seeds=tuple(
                tuple(item) for item in payload["calibration_mc_seeds"]
            ),
            holdout_mc_seeds=tuple(tuple(item) for item in payload["holdout_mc_seeds"]),
            workload_limits=tuple(sorted(payload["workload_limits"].items())),
            records=tuple(
                RPEHadamardCompiledCostBenchmarkPoint.from_dict(record)
                for record in payload["records"]
            ),
            proxy_fit_performed=bool(payload["proxy_fit_performed"]),
            holdout_used_for_proxy_fit=bool(payload["holdout_used_for_proxy_fit"]),
            dataset_fingerprint=payload["dataset_fingerprint"],
            schema_version=payload["schema_version"],
            cost_metrics=tuple(payload["cost_metrics"]),
            measurement_policy=payload["measurement_policy"],
            circuit_scope=payload["circuit_scope"],
            benchmark_path=payload["benchmark_path"],
        )
        if dataset.requested_point_count != payload["requested_point_count"]:
            raise ValueError("Serialized requested point count is inconsistent.")
        if dataset.completed_point_count != payload["completed_point_count"]:
            raise ValueError("Serialized completed point count is inconsistent.")
        if dataset.failed_point_count != payload["failed_point_count"]:
            raise ValueError("Serialized failed point count is inconsistent.")
        if dataset.complete != payload["complete"]:
            raise ValueError("Serialized dataset completeness is inconsistent.")
        return dataset


@dataclass(frozen=True)
class RPEHadamardCompiledCostBenchmarkResult:
    """Generation result; estimates are retained for direct validation only."""

    dataset: RPEHadamardCompiledCostBenchmarkDataset
    estimates: tuple[DFRPEHadamardCompiledCostEstimate, ...] = field(
        repr=False,
        compare=False,
    )


def plan_compiled_rpe_hadamard_benchmark_workload(
    preparation: DFPartialS2Preparation,
    repetition_count: int,
    rte_config: RTEConfig | None,
    rte_distribution: RTEFiniteDistribution | None,
    *,
    trajectory_count: int,
    maximum_repetition_count: int,
) -> CompiledCostWorkloadPlan:
    """Plan two full measured wrappers for one bounded medium-q point."""
    count = require_integer_count(repetition_count, name="repetition_count", minimum=1)
    round_index_for_benchmark_repetition_count(
        count,
        maximum_repetition_count=maximum_repetition_count,
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
    paired_wrapper_instruction_bound = (
        2 * one_evolution.planned_untranspiled_instruction_applications + 7
    )
    return plan_compiled_cost_workload(
        work_item_count=trajectory_count,
        circuits_per_work_item=2,
        instruction_applications_per_work_item=paired_wrapper_instruction_bound,
    )


def _derived_point_seed(
    master_seed: int,
    partition: BenchmarkPartition,
    repetition_count: int,
) -> int:
    digest = hashlib.sha256(
        (
            "rpe_hadamard_benchmark_partition_seed_v1|"
            f"{master_seed}|{partition}|{repetition_count}"
        ).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _estimate_point(
    request: RPEHadamardCompiledCostBenchmarkRequest,
    *,
    repetition_count: int,
    point_seed: int | None,
) -> DFRPEHadamardCompiledCostEstimate:
    round_index = round_index_for_benchmark_repetition_count(
        repetition_count,
        maximum_repetition_count=request.maximum_repetition_count,
    )
    if request.evaluation_method == "exact":
        stream = make_exact_df_partial_s2_repeated_trajectory_stream(
            request.preparation,
            request.delta_time,
            repetition_count,
            request.rte_config,
            request.rte_distribution,
            controlled=True,
            ancilla_qubit=request.preparation.num_system_qubits,
            construction_policy=request.construction_policy,
            maximum_trajectories=request.maximum_trajectories,
        )
    else:
        if request.sample_count is None or point_seed is None:
            raise RuntimeError("Validated Monte Carlo request lost its sample inputs.")
        stream = make_monte_carlo_df_partial_s2_repeated_trajectory_stream(
            request.preparation,
            request.delta_time,
            repetition_count,
            request.rte_config,
            request.rte_distribution,
            sample_count=request.sample_count,
            seed=point_seed,
            maximum_samples=request.maximum_samples,
            controlled=True,
            ancilla_qubit=request.preparation.num_system_qubits,
            construction_policy=request.construction_policy,
        )
    workload_plan = plan_compiled_rpe_hadamard_benchmark_workload(
        request.preparation,
        repetition_count,
        request.rte_config,
        request.rte_distribution,
        trajectory_count=stream.expected_record_count,
        maximum_repetition_count=request.maximum_repetition_count,
    )
    return _compile_hadamard_trajectory_stream_with_plan(
        stream,
        request.delta_time,
        request.compiler,
        construction_policy=request.construction_policy,
        workload_plan=workload_plan,
        maximum_untranspiled_circuit_size=(
            request.maximum_untranspiled_circuit_size
        ),
        maximum_retained_trajectory_records=(
            request.maximum_retained_trajectory_records
        ),
        maximum_build_requests=request.maximum_build_requests,
        maximum_transpile_requests=request.maximum_transpile_requests,
        maximum_planned_instruction_applications=(
            request.maximum_planned_instruction_applications
        ),
        cache=request.cache,
        backend=request.backend,
        validated_round_index=round_index,
        validated_wrapper_builder=QiskitRPEHadamardBenchmarkCircuitBuilder(
            maximum_repetition_count=request.maximum_repetition_count
        ),
    )


def _compiler_context(
    request: RPEHadamardCompiledCostBenchmarkRequest,
) -> tuple[str, str | None, str]:
    compiler_fingerprint = compiler_settings_hash(request.compiler)
    backend_fingerprint = canonical_backend_fingerprint_or_none(request.backend)
    context = _sha256_json(
        {
            "compiler_settings_fingerprint": compiler_fingerprint,
            "backend_fingerprint": backend_fingerprint,
        }
    )
    return compiler_fingerprint, backend_fingerprint, context


def _trajectory_record(
    record: Any,
    axis: RPEHadamardAxis,
) -> RPEHadamardBenchmarkTrajectoryRecord:
    if axis == "cosine":
        wrapper_provenance = record.cosine_wrapper_fingerprint
        wrapper_semantics = record.cosine_wrapper_circuit_semantics_fingerprint
        actual = record.cosine_actual_circuit_fingerprint
        cost = record.cosine_cost
    else:
        wrapper_provenance = record.sine_wrapper_fingerprint
        wrapper_semantics = record.sine_wrapper_circuit_semantics_fingerprint
        actual = record.sine_actual_circuit_fingerprint
        cost = record.sine_cost
    return RPEHadamardBenchmarkTrajectoryRecord(
        trajectory_index=record.trajectory_index,
        probability=record.probability,
        trajectory_seed=record.trajectory_seed,
        step_seeds=record.step_seeds,
        evolution_provenance_fingerprint=record.evolution_provenance_fingerprint,
        evolution_circuit_semantics_fingerprint=(
            record.evolution_circuit_semantics_fingerprint
        ),
        wrapper_provenance_fingerprint=wrapper_provenance,
        wrapper_circuit_semantics_fingerprint=wrapper_semantics,
        actual_circuit_fingerprint=actual,
        cost=cost,
        constant_phase=record.constant_phase,
        extracted_identity_phase=record.extracted_identity_phase,
        rte_relative_phase=record.rte_relative_phase,
    )


def _complete_point(
    request: RPEHadamardCompiledCostBenchmarkRequest,
    *,
    partition: BenchmarkPartition,
    estimate: DFRPEHadamardCompiledCostEstimate,
    axis: RPEHadamardAxis,
) -> RPEHadamardCompiledCostBenchmarkPoint:
    axis_estimate = estimate.cosine if axis == "cosine" else estimate.sine
    retained = tuple(
        _trajectory_record(record, axis)
        for record in estimate.retained_trajectory_records
    )
    compiler_fingerprint, backend_fingerprint, compiler_context = _compiler_context(
        request
    )
    sampling_provenance = _sha256_json(
        {
            "evaluation_method": estimate.evaluation_method,
            "master_seed": estimate.master_seed,
            "sampled_trajectory_seed_digest": (
                estimate.sampled_trajectory_seed_digest
            ),
            "trajectory_provenance_digest": estimate.trajectory_provenance_digest,
            "trajectory_probability_sum": estimate.trajectory_probability_sum,
            "processed_trajectory_count": estimate.processed_trajectory_count,
        }
    )
    evaluation_configuration = _sha256_json(
        {
            "partition": partition,
            "axis": axis,
            "q_m": estimate.q_m,
            "r_m": request.rte_steps_per_occurrence,
            "K_m": request.finite_taylor_order,
            "evaluation_method": estimate.evaluation_method,
            "sample_count": estimate.sample_count,
            "construction_policy": request.construction_policy,
            "compiler_context": compiler_context,
            "measurement_policy": DF_RPE_HADAMARD_MEASUREMENT_POLICY,
            "circuit_scope": RPE_HADAMARD_INTERROGATION_SCOPE,
        }
    )
    return RPEHadamardCompiledCostBenchmarkPoint(
        partition=partition,
        round_index=estimate.round_index,
        repetition_count=estimate.repetition_count,
        q_m=estimate.q_m,
        delta_time=estimate.delta_time,
        t_m=estimate.t_m,
        rte_steps_per_occurrence=request.rte_steps_per_occurrence,
        finite_taylor_order=request.finite_taylor_order,
        tail_kind=(
            "deterministic"
            if request.preparation.is_deterministic_only
            else "randomized"
        ),
        axis=axis,
        evaluation_method=estimate.evaluation_method,
        sample_count=estimate.sample_count,
        enumerated_trajectory_count=estimate.enumerated_trajectory_count,
        trajectory_space_size=estimate.trajectory_space_size,
        master_seed=estimate.master_seed,
        sampled_trajectory_seeds=estimate.sampled_trajectory_seeds,
        step_seed_hierarchy=tuple(record.step_seeds for record in retained),
        trajectory_probability_sum=estimate.trajectory_probability_sum,
        trajectory_provenance_digest=estimate.trajectory_provenance_digest,
        sampling_provenance_fingerprint=sampling_provenance,
        evolution_circuit_semantics_digest=(
            estimate.evolution_circuit_semantics_multiset_digest
        ),
        wrapper_circuit_semantics_digest=(
            axis_estimate.wrapper_circuit_semantics_multiset_digest
        ),
        actual_circuit_fingerprint_digest=(
            axis_estimate.actual_circuit_fingerprint_multiset_digest
        ),
        retained_trajectory_records=retained,
        trajectory_records_truncated=estimate.trajectory_records_truncated,
        metric_statistics=axis_estimate.metric_statistics,
        compiler_settings=request.compiler,
        compiler_settings_fingerprint=compiler_fingerprint,
        backend_fingerprint=backend_fingerprint,
        compiler_context_fingerprint=compiler_context,
        evaluation_configuration_fingerprint=evaluation_configuration,
        construction_policy=request.construction_policy,
        measurement_included=estimate.measurement_included,
        state_preparation_included=estimate.state_preparation_included,
        backend_execution_included=estimate.backend_execution_included,
        quantum_shots_executed=estimate.quantum_shots_executed,
        wrapped_evolution_already_controlled=True,
        additional_control_applied=False,
        fresh_iid_trajectory_per_hadamard_shot_verified=(
            estimate.fresh_iid_trajectory_per_hadamard_shot_verified
        ),
        benchmark_validation_path=True,
        benchmark_path=RPE_HADAMARD_BENCHMARK_PATH,
        circuit_scope=estimate.circuit_scope,
        status="complete",
        failure_reason=None,
        planned_build_requests=estimate.planned_build_requests,
        actual_build_requests=estimate.actual_build_requests,
        planned_transpile_requests=estimate.planned_transpile_requests,
        actual_transpile_requests=estimate.actual_cache_requests,
        planned_instruction_applications=estimate.planned_instruction_applications,
        actual_built_instruction_total=estimate.actual_built_instruction_total,
    )


def _failed_point(
    request: RPEHadamardCompiledCostBenchmarkRequest,
    *,
    partition: BenchmarkPartition,
    repetition_count: int,
    point_seed: int | None,
    axis: RPEHadamardAxis,
    failure_reason: str,
) -> RPEHadamardCompiledCostBenchmarkPoint:
    round_index = repetition_count.bit_length() - 1
    compiler_fingerprint, backend_fingerprint, compiler_context = _compiler_context(
        request
    )
    sampling_provenance = _sha256_json(
        {
            "evaluation_method": request.evaluation_method,
            "master_seed": point_seed,
            "failure_reason": failure_reason,
        }
    )
    evaluation_configuration = _sha256_json(
        {
            "partition": partition,
            "axis": axis,
            "q_m": repetition_count,
            "r_m": request.rte_steps_per_occurrence,
            "K_m": request.finite_taylor_order,
            "evaluation_method": request.evaluation_method,
            "sample_count": request.sample_count,
            "construction_policy": request.construction_policy,
            "compiler_context": compiler_context,
            "measurement_policy": DF_RPE_HADAMARD_MEASUREMENT_POLICY,
            "circuit_scope": RPE_HADAMARD_INTERROGATION_SCOPE,
        }
    )
    return RPEHadamardCompiledCostBenchmarkPoint(
        partition=partition,
        round_index=round_index,
        repetition_count=repetition_count,
        q_m=repetition_count,
        delta_time=request.delta_time,
        t_m=float(repetition_count * request.delta_time),
        rte_steps_per_occurrence=request.rte_steps_per_occurrence,
        finite_taylor_order=request.finite_taylor_order,
        tail_kind=(
            "deterministic"
            if request.preparation.is_deterministic_only
            else "randomized"
        ),
        axis=axis,
        evaluation_method=request.evaluation_method,
        sample_count=(
            request.sample_count
            if request.evaluation_method == "monte_carlo"
            else None
        ),
        enumerated_trajectory_count=None,
        trajectory_space_size=None,
        master_seed=point_seed,
        sampled_trajectory_seeds=None,
        step_seed_hierarchy=(),
        trajectory_probability_sum=None,
        trajectory_provenance_digest=None,
        sampling_provenance_fingerprint=sampling_provenance,
        evolution_circuit_semantics_digest=None,
        wrapper_circuit_semantics_digest=None,
        actual_circuit_fingerprint_digest=None,
        retained_trajectory_records=(),
        trajectory_records_truncated=False,
        metric_statistics=(),
        compiler_settings=request.compiler,
        compiler_settings_fingerprint=compiler_fingerprint,
        backend_fingerprint=backend_fingerprint,
        compiler_context_fingerprint=compiler_context,
        evaluation_configuration_fingerprint=evaluation_configuration,
        construction_policy=request.construction_policy,
        measurement_included=True,
        state_preparation_included=False,
        backend_execution_included=False,
        quantum_shots_executed=0,
        wrapped_evolution_already_controlled=True,
        additional_control_applied=False,
        fresh_iid_trajectory_per_hadamard_shot_verified=False,
        benchmark_validation_path=True,
        benchmark_path=RPE_HADAMARD_BENCHMARK_PATH,
        circuit_scope=RPE_HADAMARD_INTERROGATION_SCOPE,
        status="failed",
        failure_reason=failure_reason,
        planned_build_requests=None,
        actual_build_requests=None,
        planned_transpile_requests=None,
        actual_transpile_requests=None,
        planned_instruction_applications=None,
        actual_built_instruction_total=None,
    )


def generate_rpe_hadamard_compiled_cost_benchmark_dataset(
    request: RPEHadamardCompiledCostBenchmarkRequest,
) -> RPEHadamardCompiledCostBenchmarkResult:
    """Generate a complete or explicitly partial calibration/holdout dataset."""
    if not isinstance(request, RPEHadamardCompiledCostBenchmarkRequest):
        raise TypeError("request must be an RPEHadamardCompiledCostBenchmarkRequest.")
    partitions: tuple[tuple[BenchmarkPartition, tuple[int, ...]], ...] = (
        ("calibration", request.calibration_repetition_counts),
        ("holdout", request.holdout_repetition_counts),
    )
    seed_map: dict[tuple[BenchmarkPartition, int], int] = {}
    if request.evaluation_method == "monte_carlo":
        if request.seed is None:
            raise RuntimeError("Validated Monte Carlo request lost its master seed.")
        for partition, counts in partitions:
            for count in counts:
                seed_map[(partition, count)] = _derived_point_seed(
                    request.seed,
                    partition,
                    count,
                )
        if len(set(seed_map.values())) != len(seed_map):
            raise RuntimeError("Derived calibration/holdout seed series collided.")

    records: list[RPEHadamardCompiledCostBenchmarkPoint] = []
    estimates: list[DFRPEHadamardCompiledCostEstimate] = []
    for partition, counts in partitions:
        for count in counts:
            point_seed = seed_map.get((partition, count))
            try:
                estimate = _estimate_point(
                    request,
                    repetition_count=count,
                    point_seed=point_seed,
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                for axis in ("cosine", "sine"):
                    records.append(
                        _failed_point(
                            request,
                            partition=partition,
                            repetition_count=count,
                            point_seed=point_seed,
                            axis=axis,
                            failure_reason=reason,
                        )
                    )
                continue
            estimates.append(estimate)
            for axis in ("cosine", "sine"):
                records.append(
                    _complete_point(
                        request,
                        partition=partition,
                        estimate=estimate,
                        axis=axis,
                    )
                )

    compiler_fingerprint, backend_fingerprint, _context = _compiler_context(request)
    workload_limits = tuple(
        sorted(
            {
                "maximum_repetition_count": request.maximum_repetition_count,
                "maximum_trajectories": request.maximum_trajectories,
                "maximum_samples": request.maximum_samples,
                "maximum_untranspiled_circuit_size": (
                    request.maximum_untranspiled_circuit_size
                ),
                "maximum_retained_trajectory_records": (
                    request.maximum_retained_trajectory_records
                ),
                "maximum_build_requests": request.maximum_build_requests,
                "maximum_transpile_requests": request.maximum_transpile_requests,
                "maximum_planned_instruction_applications": (
                    request.maximum_planned_instruction_applications
                ),
            }.items()
        )
    )
    dataset = RPEHadamardCompiledCostBenchmarkDataset(
        generation_id=request.generation_id,
        calibration_repetition_counts=request.calibration_repetition_counts,
        holdout_repetition_counts=request.holdout_repetition_counts,
        preparation_fingerprint=request.preparation.preparation_hash,
        hamiltonian_fingerprint=request.preparation.hamiltonian_hash,
        partition_fingerprint=request.preparation.partition_hash,
        ld=request.preparation.ld,
        num_system_qubits=request.preparation.num_system_qubits,
        delta_time=request.delta_time,
        rte_steps_per_occurrence=request.rte_steps_per_occurrence,
        finite_taylor_order=request.finite_taylor_order,
        product_formula_order=2,
        control_convention=RPE_HADAMARD_CONTROL_CONVENTION,
        construction_policy=request.construction_policy,
        compiler_settings=request.compiler,
        compiler_settings_fingerprint=compiler_fingerprint,
        backend_fingerprint=backend_fingerprint,
        requested_evaluation_method=request.evaluation_method,
        sample_count=request.sample_count,
        master_seed=request.seed,
        calibration_mc_seeds=tuple(
            (count, seed_map[("calibration", count)])
            for count in request.calibration_repetition_counts
            if ("calibration", count) in seed_map
        ),
        holdout_mc_seeds=tuple(
            (count, seed_map[("holdout", count)])
            for count in request.holdout_repetition_counts
            if ("holdout", count) in seed_map
        ),
        workload_limits=workload_limits,
        records=tuple(records),
    )
    return RPEHadamardCompiledCostBenchmarkResult(
        dataset=dataset,
        estimates=tuple(estimates),
    )

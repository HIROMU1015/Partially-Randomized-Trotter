"""Short repeated DF partial-S2 trajectories and circuit construction.

This module deliberately stops at short, explicitly constructed trajectories.
It does not build Hadamard tests, perform quantum sampling, or assign RPE shots.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseGate

from .df_partial_s2 import (
    DFPartialS2CircuitResult,
    DFPartialS2Preparation,
    DFPartialS2StepRequest,
    QiskitDFPartialS2CircuitBuilder,
)
from .df_rte_circuit import DFRTEEventSequenceCircuitRequest
from .rte import (
    RTEConfig,
    RTEEvent,
    RTEFiniteDistribution,
    finite_rte_attenuation,
    require_integer_count,
    rte_occurrence_truncation_from_config,
)


RepeatedCircuitConstructionPolicy: TypeAlias = Literal[
    "raw_concatenation",
    "boundary_optimized",
]
BoundaryOptimizationPolicy: TypeAlias = Literal[
    "none",
    "fuse_equal_boundary_block_and_aggregate_step_phases",
]
EventOrderPolicy: TypeAlias = Literal["step_major_circuit_append_order"]
MatrixProductConvention: TypeAlias = Literal["U_(q-1)...U_1_U_0"]
REPEATED_STEP_TIME_ATOL = 1e-14
DEFAULT_SAMPLING_POLICY = "explicit_step_major_seed_hierarchy_v1"


def repetition_count_for_rpe_round(round_index: int) -> int:
    """Return the future RPE mapping ``q = 2**m`` without building a circuit."""
    normalized = require_integer_count(round_index, name="round_index")
    return 1 << normalized


def _event_fingerprint(event: RTEEvent) -> str:
    encoded = json.dumps(
        event.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DFPartialS2TrajectoryStep:
    """One ordered RTE occurrence inside a repeated partial-S2 trajectory."""

    step_index: int
    step_seed: int | None
    ordered_event_fingerprints: tuple[str, ...]
    ordered_taylor_orders: tuple[int, ...]
    ordered_selected_component_ids: tuple[tuple[str, ...], ...]
    occurrence_fingerprint: str | None


@dataclass(frozen=True)
class DFPartialS2AttenuationMetadata:
    """Finite-distribution attenuation metadata; never a gate-cost term."""

    per_step_attenuation: float
    repetition_count: int
    total_log_attenuation: float
    total_attenuation: float | None
    underflowed: bool
    saturated: bool
    rte_config: RTEConfig | None
    tail_hash: str


@dataclass(frozen=True)
class DFPartialS2TruncationMetadata:
    """Finite Taylor error only; product-formula bias is intentionally absent."""

    finite_taylor_order: int | None
    rte_steps_per_occurrence: int
    step_truncation_residual_bound: float
    occurrence_truncation_residual_bound: float
    partial_s2_randomized_residual_bound: float
    repetition_count: int
    repeated_partial_s2_residual_bound: float
    configured_step_error_budget: float | None
    allocator_origin: Literal[
        "deterministic_only",
        "RTEConfig.truncation_tolerance_and_finite_taylor_order",
    ]
    error_kind: Literal["finite_RTE_Taylor_truncation"] = (
        "finite_RTE_Taylor_truncation"
    )
    product_formula_bias_included: bool = False


@dataclass(frozen=True)
class DFPartialS2RepeatedRequest:
    """A complete, ordered request for ``q`` partial-S2 repetitions."""

    preparation: DFPartialS2Preparation
    step_time: float
    repetition_count: int
    rte_config: RTEConfig | None
    rte_distribution: RTEFiniteDistribution | None
    rte_occurrences: tuple[DFRTEEventSequenceCircuitRequest, ...] = ()
    controlled: bool = False
    ancilla_qubit: int | None = None
    step_seeds: tuple[int | None, ...] = ()
    master_seed: int | None = None
    trajectory_seed: int | None = None
    sampling_policy: str = DEFAULT_SAMPLING_POLICY
    construction_policy: RepeatedCircuitConstructionPolicy = "raw_concatenation"
    event_order_policy: EventOrderPolicy = "step_major_circuit_append_order"
    matrix_product_convention: MatrixProductConvention = "U_(q-1)...U_1_U_0"

    def __post_init__(self) -> None:
        repetition_count = require_integer_count(
            self.repetition_count,
            name="repetition_count",
            minimum=1,
        )
        object.__setattr__(self, "repetition_count", repetition_count)
        if not math.isfinite(self.step_time):
            raise ValueError("step_time must be finite.")
        if self.construction_policy not in (
            "raw_concatenation",
            "boundary_optimized",
        ):
            raise ValueError("Unsupported repeated circuit construction policy.")
        if self.event_order_policy != "step_major_circuit_append_order":
            raise ValueError("Unsupported trajectory event-order policy.")
        if self.matrix_product_convention != "U_(q-1)...U_1_U_0":
            raise ValueError("Unsupported repeated-step matrix-product convention.")
        if self.master_seed is not None:
            object.__setattr__(
                self,
                "master_seed",
                require_integer_count(self.master_seed, name="master_seed"),
            )
        if self.trajectory_seed is not None:
            object.__setattr__(
                self,
                "trajectory_seed",
                require_integer_count(self.trajectory_seed, name="trajectory_seed"),
            )
        if not isinstance(self.sampling_policy, str) or not self.sampling_policy:
            raise ValueError("sampling_policy must be a non-empty string.")

        if self.preparation.is_deterministic_only:
            if self.rte_config is not None or self.rte_distribution is not None:
                raise ValueError("Deterministic-only repetition requires no RTE data.")
            if self.rte_occurrences:
                raise ValueError(
                    "Deterministic-only repetition must not contain RTE occurrences."
                )
            seeds = self.step_seeds or (None,) * repetition_count
            if len(seeds) != repetition_count or any(
                seed is not None for seed in seeds
            ):
                raise ValueError(
                    "Deterministic-only step_seeds must contain one None per step."
                )
            object.__setattr__(self, "step_seeds", tuple(seeds))
        else:
            if self.rte_config is None or self.rte_distribution is None:
                raise ValueError("Randomized repetition requires RTE config/distribution.")
            if len(self.rte_occurrences) != repetition_count:
                raise ValueError(
                    "Randomized repetition requires one RTE occurrence per step."
                )
            if len(self.step_seeds) != repetition_count:
                raise ValueError("step_seeds must contain one seed per repetition.")
            normalized_seeds: list[int] = []
            for seed in self.step_seeds:
                if seed is None:
                    raise ValueError("Randomized repetition step seeds must not be None.")
                normalized_seeds.append(
                    require_integer_count(seed, name="step_seed")
                )
            object.__setattr__(self, "step_seeds", tuple(normalized_seeds))

        # Reuse the one-step validator for every occurrence and control condition.
        tuple(self.iter_step_requests())

    def iter_step_requests(self) -> Sequence[DFPartialS2StepRequest]:
        if self.preparation.is_deterministic_only:
            occurrences: tuple[DFRTEEventSequenceCircuitRequest | None, ...] = (
                (None,) * self.repetition_count
            )
        else:
            occurrences = self.rte_occurrences
        return tuple(
            DFPartialS2StepRequest(
                preparation=self.preparation,
                step_time=float(self.step_time),
                rte_config=self.rte_config,
                rte_distribution=self.rte_distribution,
                rte_occurrence=occurrence,
                controlled=self.controlled,
                ancilla_qubit=self.ancilla_qubit,
                seed=self.step_seeds[index],
            )
            for index, occurrence in enumerate(occurrences)
        )

    @classmethod
    def from_step_requests(
        cls,
        requests: Sequence[DFPartialS2StepRequest],
        *,
        master_seed: int | None = None,
        trajectory_seed: int | None = None,
        sampling_policy: str = DEFAULT_SAMPLING_POLICY,
        construction_policy: RepeatedCircuitConstructionPolicy = (
            "raw_concatenation"
        ),
    ) -> "DFPartialS2RepeatedRequest":
        if not requests:
            raise ValueError("At least one partial-S2 step request is required.")
        first = requests[0]
        for request in requests[1:]:
            if request.preparation.preparation_hash != (
                first.preparation.preparation_hash
            ):
                raise ValueError("Every repeated step must use the same preparation.")
            if not math.isclose(
                request.step_time,
                first.step_time,
                rel_tol=0.0,
                abs_tol=REPEATED_STEP_TIME_ATOL,
            ):
                raise ValueError("Every repeated step must use the same step_time.")
            if request.rte_config != first.rte_config:
                raise ValueError("Every repeated step must use the same RTE config.")
            if request.rte_distribution != first.rte_distribution:
                raise ValueError(
                    "Every repeated step must use the same finite distribution."
                )
            if request.controlled != first.controlled or (
                request.ancilla_qubit != first.ancilla_qubit
            ):
                raise ValueError(
                    "Every repeated step must use the same control/ancilla condition."
                )
        return cls(
            preparation=first.preparation,
            step_time=first.step_time,
            repetition_count=len(requests),
            rte_config=first.rte_config,
            rte_distribution=first.rte_distribution,
            rte_occurrences=tuple(
                request.rte_occurrence
                for request in requests
                if request.rte_occurrence is not None
            ),
            controlled=first.controlled,
            ancilla_qubit=first.ancilla_qubit,
            step_seeds=tuple(request.seed for request in requests),
            master_seed=master_seed,
            trajectory_seed=trajectory_seed,
            sampling_policy=sampling_policy,
            construction_policy=construction_policy,
        )


def make_df_partial_s2_repeated_request(
    preparation: DFPartialS2Preparation,
    *,
    step_time: float,
    repetition_count: int,
    rte_config: RTEConfig | None,
    rte_distribution: RTEFiniteDistribution | None,
    seed: int = 0,
    controlled: bool = False,
    ancilla_qubit: int | None = None,
    cancel_adjacent_equal_bases: bool = True,
    construction_policy: RepeatedCircuitConstructionPolicy = "raw_concatenation",
) -> DFPartialS2RepeatedRequest:
    """Sample independent occurrences using a reproducible two-level seed rule."""
    count = require_integer_count(
        repetition_count,
        name="repetition_count",
        minimum=1,
    )
    master_seed = require_integer_count(seed, name="seed")
    if preparation.is_deterministic_only:
        return DFPartialS2RepeatedRequest(
            preparation=preparation,
            step_time=float(step_time),
            repetition_count=count,
            rte_config=rte_config,
            rte_distribution=rte_distribution,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            step_seeds=(None,) * count,
            master_seed=master_seed,
            sampling_policy="direct_master_to_step_seeds_v1",
            construction_policy=construction_policy,
        )
    if rte_config is None or rte_distribution is None:
        raise ValueError("Randomized repetition requires RTE config/distribution.")
    if count == 1:
        step_seeds = (master_seed,)
    else:
        rng = random.Random(master_seed)
        step_seeds = tuple(rng.randrange(0, 2**63) for _ in range(count))
    occurrences = tuple(
        preparation.rte_preparation.sample_occurrence_request(
            rte_config,
            rte_distribution,
            seed=step_seed,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
        )
        for step_seed in step_seeds
    )
    return DFPartialS2RepeatedRequest(
        preparation=preparation,
        step_time=float(step_time),
        repetition_count=count,
        rte_config=rte_config,
        rte_distribution=rte_distribution,
        rte_occurrences=occurrences,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        step_seeds=step_seeds,
        master_seed=master_seed,
        sampling_policy="direct_master_to_step_seeds_v1",
        construction_policy=construction_policy,
    )


@dataclass(frozen=True)
class DFPartialS2RepeatedCircuitResult:
    """Constructed short trajectory plus audit and numerical metadata."""

    circuit: QuantumCircuit
    preparation_hash: str
    hamiltonian_hash: str
    partition_hash: str
    step_time: float
    repetition_count: int
    controlled: bool
    ancilla_qubit: int | None
    master_seed: int | None
    trajectory_seed: int | None
    sampling_policy: str
    trajectory: tuple[DFPartialS2TrajectoryStep, ...]
    trajectory_fingerprint: str
    provenance_fingerprint: str
    circuit_semantics_fingerprint: str
    total_random_event_count: int
    rte_steps_per_repetition: int
    event_order_policy: EventOrderPolicy
    matrix_product_convention: MatrixProductConvention
    construction_policy: RepeatedCircuitConstructionPolicy
    boundary_optimization_policy: BoundaryOptimizationPolicy
    fused_boundary_count: int
    constant_phase: float
    extracted_identity_phase: float
    rte_relative_phase: float
    attenuation: DFPartialS2AttenuationMetadata
    truncation: DFPartialS2TruncationMetadata
    step_results: tuple[DFPartialS2CircuitResult, ...]
    compiler_independent_fingerprint: str
    untranspiled_circuit_size: int
    untranspiled_circuit_depth: int
    circuit_qubit_count: int
    circuit_granularity: Literal["repeated_partial_s2_steps"] = (
        "repeated_partial_s2_steps"
    )
    fidelity_level: Literal[5] = 5


class QiskitDFPartialS2RepeatedCircuitBuilder:
    """Build raw or locally boundary-optimized repeated partial-S2 circuits."""

    def __init__(self) -> None:
        self._step_builder = QiskitDFPartialS2CircuitBuilder()

    @staticmethod
    def _compose(circuit: QuantumCircuit, part: QuantumCircuit) -> None:
        circuit.compose(
            part,
            qubits=tuple(range(circuit.num_qubits)),
            inplace=True,
        )

    @staticmethod
    def _attenuation(
        request: DFPartialS2RepeatedRequest,
    ) -> DFPartialS2AttenuationMetadata:
        config = request.rte_config
        if config is None:
            per_step = 1.0
            total_log = 0.0
        else:
            per_step = finite_rte_attenuation(config)
            total_log = -float(
                request.repetition_count
                * config.rte_steps
                * math.log(config.distribution_normalization)
            )
        try:
            total = float(math.exp(total_log))
        except OverflowError:
            total = math.inf
        underflowed = total == 0.0
        saturated = not math.isfinite(total)
        return DFPartialS2AttenuationMetadata(
            per_step_attenuation=float(per_step),
            repetition_count=request.repetition_count,
            total_log_attenuation=total_log,
            total_attenuation=None if underflowed or saturated else total,
            underflowed=underflowed,
            saturated=saturated,
            rte_config=config,
            tail_hash=request.preparation.tail_extraction.tail_hash,
        )

    @staticmethod
    def _truncation(
        request: DFPartialS2RepeatedRequest,
    ) -> DFPartialS2TruncationMetadata:
        config = request.rte_config
        if config is None:
            return DFPartialS2TruncationMetadata(
                finite_taylor_order=None,
                rte_steps_per_occurrence=0,
                step_truncation_residual_bound=0.0,
                occurrence_truncation_residual_bound=0.0,
                partial_s2_randomized_residual_bound=0.0,
                repetition_count=request.repetition_count,
                repeated_partial_s2_residual_bound=0.0,
                configured_step_error_budget=None,
                allocator_origin="deterministic_only",
            )
        record = rte_occurrence_truncation_from_config(
            "df-partial-s2-randomized-tail",
            config,
            round_occurrence_count=request.repetition_count,
        )
        return DFPartialS2TruncationMetadata(
            finite_taylor_order=config.finite_taylor_order,
            rte_steps_per_occurrence=config.rte_steps,
            step_truncation_residual_bound=(
                record.step_truncation_residual_bound
            ),
            occurrence_truncation_residual_bound=(
                record.occurrence_truncation_residual_bound
            ),
            partial_s2_randomized_residual_bound=(
                record.occurrence_truncation_residual_bound
            ),
            repetition_count=request.repetition_count,
            repeated_partial_s2_residual_bound=(
                record.round_contribution_residual_bound
            ),
            configured_step_error_budget=config.truncation_tolerance,
            allocator_origin=(
                "RTEConfig.truncation_tolerance_and_finite_taylor_order"
            ),
        )

    @staticmethod
    def _trajectory(
        request: DFPartialS2RepeatedRequest,
        step_results: Sequence[DFPartialS2CircuitResult],
    ) -> tuple[tuple[DFPartialS2TrajectoryStep, ...], str]:
        steps: list[DFPartialS2TrajectoryStep] = []
        for index, (step_request, result) in enumerate(
            zip(request.iter_step_requests(), step_results, strict=True)
        ):
            events = (
                ()
                if step_request.rte_occurrence is None
                else step_request.rte_occurrence.events
            )
            steps.append(
                DFPartialS2TrajectoryStep(
                    step_index=index,
                    step_seed=step_request.seed,
                    ordered_event_fingerprints=tuple(
                        _event_fingerprint(event) for event in events
                    ),
                    ordered_taylor_orders=tuple(
                        event.taylor_order for event in events
                    ),
                    ordered_selected_component_ids=tuple(
                        event.selected_component_ids for event in events
                    ),
                    occurrence_fingerprint=result.rte_sequence_fingerprint,
                )
            )
        payload = {
            "tail_hash": request.preparation.tail_extraction.tail_hash,
            "master_seed": request.master_seed,
            "trajectory_seed": request.trajectory_seed,
            "sampling_policy": request.sampling_policy,
            "event_order_policy": request.event_order_policy,
            "matrix_product_convention": request.matrix_product_convention,
            "steps": [
                {
                    "step_index": step.step_index,
                    "step_seed": step.step_seed,
                    "ordered_event_fingerprints": (
                        step.ordered_event_fingerprints
                    ),
                    "occurrence_fingerprint": step.occurrence_fingerprint,
                }
                for step in steps
            ],
            "provenance_fingerprint_policy": "df_partial_s2_provenance_v2",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return tuple(steps), hashlib.sha256(encoded).hexdigest()

    def _build_raw(
        self,
        step_requests: Sequence[DFPartialS2StepRequest],
        step_results: Sequence[DFPartialS2CircuitResult],
    ) -> QuantumCircuit:
        circuit = self._step_builder._new_circuit(step_requests[0])
        for result in step_results:
            self._compose(circuit, result.circuit)
        return circuit

    def _append_aggregated_phases(
        self,
        circuit: QuantumCircuit,
        request: DFPartialS2RepeatedRequest,
    ) -> None:
        total_phase = -request.step_time * request.repetition_count * (
            request.preparation.constant_coefficient
            + request.preparation.extracted_identity_coefficient
        )
        if abs(total_phase) <= 1e-15:
            return
        if request.controlled:
            circuit.append(PhaseGate(total_phase), [request.ancilla_qubit])
        else:
            circuit.global_phase += total_phase

    def _build_boundary_optimized(
        self,
        request: DFPartialS2RepeatedRequest,
        step_requests: Sequence[DFPartialS2StepRequest],
    ) -> QuantumCircuit:
        circuit = self._step_builder._new_circuit(step_requests[0])
        self._append_aggregated_phases(circuit, request)
        blocks = request.preparation.deterministic_blocks
        last_step = request.repetition_count - 1
        for index, step_request in enumerate(step_requests):
            forward_blocks = blocks if index == 0 else blocks[1:]
            for block in forward_blocks:
                self._step_builder._append_block(circuit, block, step_request)

            parts = self._step_builder.build_additive_circuits(step_request)
            self._compose(circuit, parts.rte_occurrence)

            if index == last_step:
                for block in reversed(blocks):
                    self._step_builder._append_block(circuit, block, step_request)
            elif blocks:
                for block in reversed(blocks[1:]):
                    self._step_builder._append_block(circuit, block, step_request)
                # The current reverse half of block 0 and the next forward half
                # are adjacent and share exactly the same recorded basis.
                self._step_builder._append_block(
                    circuit,
                    blocks[0],
                    step_request,
                    evolution_time=request.step_time,
                )
        return circuit

    def build(
        self,
        request: DFPartialS2RepeatedRequest,
        *,
        construction_policy: RepeatedCircuitConstructionPolicy | None = None,
    ) -> DFPartialS2RepeatedCircuitResult:
        """Build the complete short trajectory under the requested policy."""
        policy = construction_policy or request.construction_policy
        if policy not in ("raw_concatenation", "boundary_optimized"):
            raise ValueError("Unsupported repeated circuit construction policy.")
        step_requests = tuple(request.iter_step_requests())
        step_results = tuple(
            self._step_builder.build_step(step_request)
            for step_request in step_requests
        )
        if policy == "raw_concatenation":
            circuit = self._build_raw(step_requests, step_results)
            boundary_policy: BoundaryOptimizationPolicy = "none"
            fused_boundaries = 0
        else:
            circuit = self._build_boundary_optimized(request, step_requests)
            boundary_policy = (
                "fuse_equal_boundary_block_and_aggregate_step_phases"
            )
            fused_boundaries = (
                request.repetition_count - 1
                if request.preparation.deterministic_blocks
                else 0
            )

        trajectory, provenance_fingerprint = self._trajectory(
            request,
            step_results,
        )
        basis_reuse_conditions = tuple(
            (
                "none"
                if step_request.rte_occurrence is None
                else "raw_adjacent_equal_basis"
                if step_request.rte_occurrence.cancel_adjacent_equal_bases
                else "disabled"
            )
            for step_request in step_requests
        )
        fingerprint_payload = {
            "preparation_hash": request.preparation.preparation_hash,
            "hamiltonian_hash": request.preparation.hamiltonian_hash,
            "partition_hash": request.preparation.partition_hash,
            "tail_hash": request.preparation.tail_extraction.tail_hash,
            "step_time": request.step_time,
            "repetition_count": request.repetition_count,
            "rte_steps_per_repetition": (
                0 if request.rte_config is None else request.rte_config.rte_steps
            ),
            "ordered_trajectory_events": tuple(
                {
                    "step_index": step.step_index,
                    "ordered_event_fingerprints": step.ordered_event_fingerprints,
                    "occurrence_fingerprint": step.occurrence_fingerprint,
                }
                for step in trajectory
            ),
            "step_circuit_fingerprints": tuple(
                result.compiler_independent_fingerprint for result in step_results
            ),
            "controlled": request.controlled,
            "ancilla_qubit": request.ancilla_qubit,
            "construction_policy": policy,
            "boundary_optimization_policy": boundary_policy,
            "identity_policy": request.preparation.identity_policy,
            "basis_reuse_conditions": basis_reuse_conditions,
            "event_order_policy": request.event_order_policy,
            "matrix_product_convention": request.matrix_product_convention,
            "fingerprint_policy": "df_partial_s2_circuit_semantics_v2",
        }
        encoded = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        semantics_fingerprint = hashlib.sha256(encoded).hexdigest()
        return DFPartialS2RepeatedCircuitResult(
            circuit=circuit,
            preparation_hash=request.preparation.preparation_hash,
            hamiltonian_hash=request.preparation.hamiltonian_hash,
            partition_hash=request.preparation.partition_hash,
            step_time=request.step_time,
            repetition_count=request.repetition_count,
            controlled=request.controlled,
            ancilla_qubit=request.ancilla_qubit,
            master_seed=request.master_seed,
            trajectory_seed=request.trajectory_seed,
            sampling_policy=request.sampling_policy,
            trajectory=trajectory,
            trajectory_fingerprint=provenance_fingerprint,
            provenance_fingerprint=provenance_fingerprint,
            circuit_semantics_fingerprint=semantics_fingerprint,
            total_random_event_count=sum(
                result.randomized_event_count for result in step_results
            ),
            rte_steps_per_repetition=(
                0 if request.rte_config is None else request.rte_config.rte_steps
            ),
            event_order_policy=request.event_order_policy,
            matrix_product_convention=request.matrix_product_convention,
            construction_policy=policy,
            boundary_optimization_policy=boundary_policy,
            fused_boundary_count=fused_boundaries,
            constant_phase=float(
                -request.step_time
                * request.repetition_count
                * request.preparation.constant_coefficient
            ),
            extracted_identity_phase=float(
                -request.step_time
                * request.repetition_count
                * request.preparation.extracted_identity_coefficient
            ),
            rte_relative_phase=float(
                math.fsum(result.rte_relative_phase for result in step_results)
            ),
            attenuation=self._attenuation(request),
            truncation=self._truncation(request),
            step_results=step_results,
            compiler_independent_fingerprint=semantics_fingerprint,
            untranspiled_circuit_size=int(circuit.size()),
            untranspiled_circuit_depth=int(circuit.depth() or 0),
            circuit_qubit_count=circuit.num_qubits,
        )

"""Dense-free requests, results, and Protocol for DF RTE event circuits.

The concrete Qiskit implementation lives in :mod:`trotterlib.df_rte_qiskit`.
The boundary here prevents builders from silently reordering RTE events or
controlling DF basis changes when only the diagonal evolution needs control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol, TypeAlias

from .df_rte_tail import DFBasisDefinition, DFBasisRegistry, SymbolicRTETail
from .rte import (
    BasisChangeOperation,
    DeterministicOnlyRTETailError,
    RTE_PARAMETER_ABS_TOL,
    RTE_PARAMETER_REL_TOL,
    RTEConfig,
    RTEEvent,
    RTEFiniteDistribution,
    iter_sample_rte_events,
    require_integer_count,
    sample_rte_events,
)


@dataclass(frozen=True)
class DFRTEComponentCircuitSpec:
    """Metadata for one non-identity DF-conjugated Z/ZZ involution."""

    component_id: str
    coefficient_abs: float
    coefficient_sign: int
    df_fragment_id: str
    basis_id: str
    diagonal_pauli_support: tuple[int, ...]
    basis_change_operations: tuple[BasisChangeOperation, ...]
    num_system_qubits: int
    basis_hash: str | None = None
    representation: Literal["established_df_basis_conjugation"] = (
        "established_df_basis_conjugation"
    )
    normalized_operator_kind: Literal["hermitian_involution"] = (
        "hermitian_involution"
    )

    def __post_init__(self) -> None:
        if self.coefficient_abs <= 0.0:
            raise ValueError("coefficient_abs must be positive.")
        if self.coefficient_sign not in (-1, 1):
            raise ValueError("coefficient_sign must be -1 or +1.")
        if len(self.diagonal_pauli_support) not in (1, 2):
            raise ValueError("Non-identity DF components require Z or ZZ support.")


@dataclass(frozen=True)
class DFRTEIdentityCircuitSpec:
    """Identity component with explicit global/controlled phase semantics."""

    component_id: str
    coefficient_abs: float
    coefficient_sign: int
    num_system_qubits: int
    df_fragment_id: str | None = None
    basis_id: str | None = None
    basis_hash: str | None = None
    diagonal_pauli_support: tuple[()] = ()
    uncontrolled_action: Literal["global_phase"] = "global_phase"
    controlled_action: Literal["relative_ancilla_phase"] = "relative_ancilla_phase"

    def __post_init__(self) -> None:
        if self.coefficient_abs <= 0.0:
            raise ValueError("coefficient_abs must be positive.")
        if self.coefficient_sign not in (-1, 1):
            raise ValueError("coefficient_sign must be -1 or +1.")


DFRTECircuitSpec: TypeAlias = DFRTEComponentCircuitSpec | DFRTEIdentityCircuitSpec


@dataclass(frozen=True)
class DFRTEEventPreparation:
    """Dense-free bundle needed to sample and validate DF RTE event requests."""

    symbolic_tail: SymbolicRTETail
    component_specs: tuple[DFRTECircuitSpec, ...]
    basis_registry: DFBasisRegistry

    def __post_init__(self) -> None:
        component_ids = tuple(
            component.component_id for component in self.symbolic_tail.components
        )
        spec_ids = tuple(spec.component_id for spec in self.component_specs)
        if spec_ids != component_ids:
            raise ValueError(
                "Circuit specs must preserve the symbolic component ordering."
            )
        definitions = {
            metadata.basis_id: metadata
            for metadata in self.symbolic_tail.basis_definitions
        }
        for basis_id in self.symbolic_tail.referenced_basis_ids:
            registered = self.basis_registry.definition(basis_id)
            if registered.metadata != definitions[basis_id]:
                raise ValueError(
                    f"Registry definition for {basis_id!r} does not match the tail."
                )
        for component, spec in zip(
            self.symbolic_tail.components,
            self.component_specs,
            strict=True,
        ):
            if spec.coefficient_abs != component.coefficient_abs:
                raise ValueError("Circuit spec coefficient does not match the tail.")
            if spec.coefficient_sign != component.coefficient_sign:
                raise ValueError("Circuit spec sign does not match the tail.")
            if spec.basis_id != component.basis_id:
                raise ValueError("Circuit spec basis ID does not match the tail.")
            if spec.basis_hash != component.basis_hash:
                raise ValueError("Circuit spec basis hash does not match the tail.")
            if spec.diagonal_pauli_support != component.diagonal_pauli_support:
                raise ValueError("Circuit spec support does not match the tail.")
            if isinstance(spec, DFRTEIdentityCircuitSpec) != component.is_identity:
                raise ValueError("Circuit spec identity type does not match the tail.")
            if component.basis_id is None:
                raise ValueError("DF symbolic component is missing a basis ID.")
            definition = self.basis_registry.definition(component.basis_id)
            if definition.basis_hash != component.basis_hash:
                raise ValueError("Component basis hash does not match the registry.")
            if definition.metadata.operations != component.basis_change_operations:
                raise ValueError(
                    "Component basis operations do not match the registry."
                )

    def resolve_event_basis_definitions(
        self,
        event: RTEEvent,
    ) -> tuple[DFBasisDefinition, ...]:
        """Resolve and fingerprint-check each event application in its order."""
        resolved: list[DFBasisDefinition] = []
        for application in event.application_sequence:
            if application.basis_id is None:
                raise ValueError("DF event application is missing a basis ID.")
            definition = self.basis_registry.definition(application.basis_id)
            if definition.basis_hash != application.basis_hash:
                raise ValueError("Event basis hash does not match the registry.")
            if definition.metadata.operations != application.basis_change_operations:
                raise ValueError("Event basis operations do not match the registry.")
            resolved.append(definition)
        return tuple(resolved)

    def sample_events(
        self,
        distribution: RTEFiniteDistribution,
        *,
        sample_count: int,
        seed: int,
    ) -> tuple[RTEEvent, ...]:
        """Classically sample events from the symbolic normalized components."""
        if self.symbolic_tail.is_deterministic_only:
            raise DeterministicOnlyRTETailError(
                "The tail has no randomized components to sample."
            )
        return sample_rte_events(
            self.symbolic_tail.components,
            distribution,
            sample_count=sample_count,
            seed=seed,
        )

    def iter_sample_events(
        self,
        distribution: RTEFiniteDistribution,
        *,
        sample_count: int,
        seed: int,
    ) -> Iterator[RTEEvent]:
        """Yield sampled events once while preserving the public tuple API."""
        if self.symbolic_tail.is_deterministic_only:
            raise DeterministicOnlyRTETailError(
                "The tail has no randomized components to sample."
            )
        return iter_sample_rte_events(
            self.symbolic_tail.components,
            distribution,
            sample_count=sample_count,
            seed=seed,
        )

    def request_for_event(
        self,
        event: RTEEvent,
        *,
        controlled: bool = False,
        ancilla_qubit: int | None = None,
        cancel_adjacent_equal_bases: bool = True,
    ) -> DFRTEEventCircuitRequest:
        """Validate registry resolution and create a builder request."""
        self.resolve_event_basis_definitions(event)
        return DFRTEEventCircuitRequest(
            event=event,
            component_specs=self.component_specs,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
            tail_id=self.symbolic_tail.tail_id,
            tail_hash=self.symbolic_tail.tail_hash,
        )

    def sample_requests(
        self,
        distribution: RTEFiniteDistribution,
        *,
        sample_count: int,
        seed: int,
        controlled: bool = False,
        ancilla_qubit: int | None = None,
        cancel_adjacent_equal_bases: bool = True,
    ) -> tuple[DFRTEEventCircuitRequest, ...]:
        """Sample events and convert each to a validated circuit request."""
        return tuple(
            self.request_for_event(
                event,
                controlled=controlled,
                ancilla_qubit=ancilla_qubit,
                cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
            )
            for event in self.sample_events(
                distribution,
                sample_count=sample_count,
                seed=seed,
            )
        )

    def sample_occurrence_request(
        self,
        config: RTEConfig,
        distribution: RTEFiniteDistribution,
        *,
        seed: int,
        controlled: bool = False,
        ancilla_qubit: int | None = None,
        cancel_adjacent_equal_bases: bool = True,
    ) -> DFRTEEventSequenceCircuitRequest:
        """Sample exactly one configured RTE occurrence of ``rte_steps`` events."""
        if config.tail_id != self.symbolic_tail.tail_id:
            raise ValueError("RTE config tail ID does not match the symbolic tail.")
        if config.tail_hash != self.symbolic_tail.tail_hash:
            raise ValueError("RTE config tail hash does not match the symbolic tail.")
        if not math.isclose(
            config.lambda_r,
            self.symbolic_tail.lambda_r,
            rel_tol=RTE_PARAMETER_REL_TOL,
            abs_tol=RTE_PARAMETER_ABS_TOL,
        ):
            raise ValueError("RTE config lambda_r does not match the symbolic tail.")
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
        events = self.sample_events(
            distribution,
            sample_count=config.rte_steps,
            seed=seed,
        )
        if len(events) != config.rte_steps:
            raise RuntimeError("Sampled occurrence event count does not match rte_steps.")
        for event in events:
            self.resolve_event_basis_definitions(event)
        return DFRTEEventSequenceCircuitRequest(
            events=events,
            component_specs=self.component_specs,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
            tail_id=self.symbolic_tail.tail_id,
            tail_hash=self.symbolic_tail.tail_hash,
            occurrence_rte_steps=config.rte_steps,
        )


def _validate_component_specs(
    events: tuple[RTEEvent, ...],
    component_specs: tuple[DFRTECircuitSpec, ...],
) -> None:
    identifiers = [spec.component_id for spec in component_specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("component circuit spec IDs must be unique.")
    specs = {spec.component_id: spec for spec in component_specs}
    selected = {
        application.component_id
        for event in events
        for application in event.application_sequence
    }
    missing = selected - set(specs)
    if missing:
        raise ValueError(f"Missing component circuit specs: {sorted(missing)}")
    for event in events:
        for application in event.application_sequence:
            spec = specs[application.component_id]
            if spec.coefficient_sign != application.coefficient_sign:
                raise ValueError("Event/spec coefficient sign mismatch.")
            if spec.coefficient_abs != application.coefficient_abs:
                raise ValueError("Event/spec absolute coefficient mismatch.")
            if isinstance(spec, DFRTEIdentityCircuitSpec) != application.is_identity:
                raise ValueError("Event/spec identity classification mismatch.")
            if spec.basis_id != application.basis_id:
                raise ValueError("Event/spec basis ID mismatch.")
            if spec.basis_hash != application.basis_hash:
                raise ValueError("Event/spec basis hash mismatch.")
            if spec.diagonal_pauli_support != application.diagonal_pauli_support:
                raise ValueError("Event/spec diagonal Pauli support mismatch.")


@dataclass(frozen=True)
class DFRTEEventCircuitRequest:
    event: RTEEvent
    component_specs: tuple[DFRTECircuitSpec, ...]
    controlled: bool = False
    ancilla_qubit: int | None = None
    preserve_event_order: bool = True
    cancel_adjacent_equal_bases: bool = True
    control_diagonal_only: bool = True
    identity_as_relative_ancilla_phase: bool = True
    tail_id: str | None = None
    tail_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.preserve_event_order:
            raise ValueError("Baseline RTE forbids event reordering.")
        if self.controlled and self.ancilla_qubit is None:
            raise ValueError("A controlled event requires ancilla_qubit.")
        if self.ancilla_qubit is not None:
            object.__setattr__(
                self,
                "ancilla_qubit",
                require_integer_count(self.ancilla_qubit, name="ancilla_qubit"),
            )
        if (self.tail_id is None) != (self.tail_hash is None):
            raise ValueError("tail_id and tail_hash must be supplied together.")
        _validate_component_specs((self.event,), self.component_specs)


@dataclass(frozen=True)
class DFRTEEventSequenceCircuitRequest:
    """Ordered events for one tail evolution, including step-boundary reuse."""

    events: tuple[RTEEvent, ...]
    component_specs: tuple[DFRTECircuitSpec, ...]
    controlled: bool = False
    ancilla_qubit: int | None = None
    preserve_event_order: bool = True
    cancel_adjacent_equal_bases: bool = True
    control_diagonal_only: bool = True
    identity_as_relative_ancilla_phase: bool = True
    tail_id: str | None = None
    tail_hash: str | None = None
    occurrence_rte_steps: int | None = None

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("An RTE event sequence must not be empty.")
        if not self.preserve_event_order:
            raise ValueError("Baseline RTE forbids event reordering.")
        if self.controlled and self.ancilla_qubit is None:
            raise ValueError("A controlled event sequence requires ancilla_qubit.")
        if self.ancilla_qubit is not None:
            object.__setattr__(
                self,
                "ancilla_qubit",
                require_integer_count(self.ancilla_qubit, name="ancilla_qubit"),
            )
        if (self.tail_id is None) != (self.tail_hash is None):
            raise ValueError("tail_id and tail_hash must be supplied together.")
        if self.occurrence_rte_steps is not None:
            steps = require_integer_count(
                self.occurrence_rte_steps,
                name="occurrence_rte_steps",
                minimum=1,
            )
            object.__setattr__(self, "occurrence_rte_steps", steps)
            if len(self.events) != steps:
                raise ValueError(
                    "Occurrence sequence event count must equal occurrence_rte_steps."
                )
        _validate_component_specs(self.events, self.component_specs)


@dataclass(frozen=True)
class DFRTEEventCircuitResult:
    circuit: Any
    preserved_component_order: tuple[str, ...]
    cancelled_basis_change_pairs: int
    relative_ancilla_phase: float
    basis_switch_count: int
    controlled: bool = False
    event_count: int = 1
    application_count: int = 0
    naive_basis_change_count: int = 0
    emitted_basis_change_count: int = 0
    accumulated_global_phase: float = 0.0
    basis_reuse_policy: Literal["disabled", "raw_adjacent_equal_basis"] = (
        "disabled"
    )
    circuit_qubit_count: int = 0
    circuit_fingerprint: str = ""

    def __post_init__(self) -> None:
        for name, minimum in (
            ("cancelled_basis_change_pairs", 0),
            ("basis_switch_count", 0),
            ("event_count", 1),
            ("application_count", 0),
            ("naive_basis_change_count", 0),
            ("emitted_basis_change_count", 0),
            ("circuit_qubit_count", 0),
        ):
            object.__setattr__(
                self,
                name,
                require_integer_count(getattr(self, name), name=name, minimum=minimum),
            )


class DFRTEEventCircuitBuilder(Protocol):
    """Interface implemented by concrete DF RTE event circuit builders."""

    def build_event(
        self, request: DFRTEEventCircuitRequest
    ) -> DFRTEEventCircuitResult: ...

    def build_sequence(
        self, request: DFRTEEventSequenceCircuitRequest
    ) -> DFRTEEventCircuitResult: ...

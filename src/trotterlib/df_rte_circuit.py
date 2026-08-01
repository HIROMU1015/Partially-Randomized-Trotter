"""Types and Protocol for a future DF RTE event circuit implementation.

No circuit builder is implemented here.  The protocol below prevents a
future builder from silently reordering RTE events or controlling DF basis
changes when only the diagonal evolution needs control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

from .rte import BasisChangeOperation, RTEEvent


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
    diagonal_pauli_support: tuple[()] = ()
    uncontrolled_action: Literal["global_phase"] = "global_phase"
    controlled_action: Literal["relative_ancilla_phase"] = "relative_ancilla_phase"

    def __post_init__(self) -> None:
        if self.coefficient_abs <= 0.0:
            raise ValueError("coefficient_abs must be positive.")
        if self.coefficient_sign not in (-1, 1):
            raise ValueError("coefficient_sign must be -1 or +1.")


DFRTECircuitSpec: TypeAlias = DFRTEComponentCircuitSpec | DFRTEIdentityCircuitSpec


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

    def __post_init__(self) -> None:
        if not self.preserve_event_order:
            raise ValueError("Baseline RTE forbids event reordering.")
        if self.controlled and self.ancilla_qubit is None:
            raise ValueError("A controlled event requires ancilla_qubit.")
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

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("An RTE event sequence must not be empty.")
        if not self.preserve_event_order:
            raise ValueError("Baseline RTE forbids event reordering.")
        if self.controlled and self.ancilla_qubit is None:
            raise ValueError("A controlled event sequence requires ancilla_qubit.")
        _validate_component_specs(self.events, self.component_specs)


@dataclass(frozen=True)
class DFRTEEventCircuitResult:
    circuit: Any
    preserved_component_order: tuple[str, ...]
    cancelled_basis_change_pairs: int
    relative_ancilla_phase: float
    basis_switch_count: int


class DFRTEEventCircuitBuilder(Protocol):
    """Proposed API implemented only after the finite RTE math milestone."""

    def build_event(
        self, request: DFRTEEventCircuitRequest
    ) -> DFRTEEventCircuitResult: ...

    def build_sequence(
        self, request: DFRTEEventSequenceCircuitRequest
    ) -> DFRTEEventCircuitResult: ...

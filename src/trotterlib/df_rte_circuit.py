"""Typed boundary for the next-stage DF RTE event circuit implementation.

No circuit is built in the current milestone.  The protocol below prevents a
future builder from silently reordering RTE events or controlling DF basis
changes when only the diagonal evolution needs control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .rte import RTEEvent


@dataclass(frozen=True)
class DFRTEComponentCircuitSpec:
    """Metadata for ``U_l^dagger P_m U_l`` from one DF diagonal fragment."""

    component_id: str
    df_fragment_id: str
    basis_id: str
    diagonal_involution_id: str
    num_system_qubits: int
    representation: Literal["U_dagger_D_U"] = "U_dagger_D_U"
    normalized_operator_kind: Literal["hermitian_involution"] = (
        "hermitian_involution"
    )


@dataclass(frozen=True)
class DFRTEEventCircuitRequest:
    event: RTEEvent
    component_specs: tuple[DFRTEComponentCircuitSpec, ...]
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
        available = {spec.component_id for spec in self.component_specs}
        missing = set(self.event.selected_component_ids) - available
        if missing:
            raise ValueError(f"Missing component circuit specs: {sorted(missing)}")


@dataclass(frozen=True)
class DFRTEEventSequenceCircuitRequest:
    """Ordered events for one tail evolution, including step-boundary reuse."""

    events: tuple[RTEEvent, ...]
    component_specs: tuple[DFRTEComponentCircuitSpec, ...]
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
        selected = {
            component_id
            for event in self.events
            for component_id in event.selected_component_ids
        }
        available = {spec.component_id for spec in self.component_specs}
        missing = selected - available
        if missing:
            raise ValueError(f"Missing component circuit specs: {sorted(missing)}")


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

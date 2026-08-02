"""Concrete Qiskit builder for dense-free DF RTE event requests."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseGate, RZGate, RZZGate, ZGate

from .df_rte_circuit import (
    DFRTECircuitSpec,
    DFRTEEventCircuitRequest,
    DFRTEEventCircuitResult,
    DFRTEEventSequenceCircuitRequest,
)
from .df_rte_tail import DFBasisDefinition, DFBasisRegistry
from .rte import RTEEvent, RTEEventApplication


_PHASE_VALIDATION_ATOL = 1e-12


def estimate_df_rte_untranspiled_size_upper_bound(
    request: DFRTEEventCircuitRequest | DFRTEEventSequenceCircuitRequest,
) -> int:
    """Bound instruction count without allocating a Qiskit circuit.

    The bound deliberately assumes that every non-identity application emits
    both basis changes.  Adjacent-basis reuse can therefore only make the
    subsequently built circuit smaller.  It is used as the pre-build OOM guard;
    callers must retain their post-build size check as a consistency guard.
    """
    events = (
        (request.event,)
        if isinstance(request, DFRTEEventCircuitRequest)
        else request.events
    )
    size = 0
    for event in events:
        _validate_event_application_order(event)
        for application in event.application_sequence:
            if application.is_identity:
                continue
            size += 2 * len(application.basis_change_operations)
            support = application.diagonal_pauli_support
            if support is None or len(support) not in (1, 2):
                raise ValueError(
                    "Non-identity DF applications require Z or ZZ support."
                )
            size += len(support) if application.role == "product" else 1
    # Every event contributes a phase in the worst case, but the builder
    # combines them into at most one ancilla-relative phase gate.
    if request.controlled and events:
        size += 1
    return size


def _event_phase_angle(phase: complex) -> float:
    value = complex(phase)
    if abs(value - 1.0) <= _PHASE_VALIDATION_ATOL:
        return 0.0
    if abs(value + 1.0) <= _PHASE_VALIDATION_ATOL:
        return math.pi
    raise ValueError("RTE event phase must be exactly +1 or -1 for paired Taylor events.")


def _validate_event_application_order(event: RTEEvent) -> None:
    applications = event.application_sequence
    if not applications or applications[-1].role != "rotation":
        raise ValueError("An RTE event must end with exactly one rotation application.")
    if any(application.role != "product" for application in applications[:-1]):
        raise ValueError("Only the final RTE event application may be a rotation.")
    if tuple(application.application_index for application in applications) != tuple(
        range(len(applications))
    ):
        raise ValueError("RTE event application indices must be contiguous.")
    component_ids = tuple(application.component_id for application in applications)
    if component_ids != event.selected_component_ids:
        raise ValueError("RTE event application order differs from selected components.")
    if applications[-1].component_id != event.rotation_component_id:
        raise ValueError("RTE event rotation component metadata is inconsistent.")
    if component_ids[:-1] != event.product_component_ids:
        raise ValueError("RTE event product component metadata is inconsistent.")


def _circuit_fingerprint(
    events: Sequence[RTEEvent],
    *,
    tail_id: str | None,
    tail_hash: str | None,
    controlled: bool,
    ancilla_qubit: int | None,
    basis_reuse_policy: str,
) -> str:
    event_payloads = []
    for event in events:
        event_payload = event.to_dict()
        for field_name in (
            "event_probability",
            "event_coefficient",
            "event_normalization",
        ):
            event_payload.pop(field_name, None)
        event_payloads.append(event_payload)
    payload = {
        "tail_id": tail_id,
        "tail_hash": tail_hash,
        "events": event_payloads,
        "controlled": bool(controlled),
        "ancilla_qubit": ancilla_qubit,
        "basis_reuse_policy": basis_reuse_policy,
        "fingerprint_policy": "df_rte_event_circuit_v2",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class QiskitDFRTEEventCircuitBuilder:
    """Build Qiskit circuits while preserving event and component order."""

    def __init__(self, *, basis_registry: DFBasisRegistry) -> None:
        self._basis_registry = basis_registry

    @property
    def basis_registry(self) -> DFBasisRegistry:
        return self._basis_registry

    def build_event(
        self,
        request: DFRTEEventCircuitRequest,
    ) -> DFRTEEventCircuitResult:
        return self._build(
            events=(request.event,),
            component_specs=request.component_specs,
            controlled=request.controlled,
            ancilla_qubit=request.ancilla_qubit,
            cancel_adjacent_equal_bases=request.cancel_adjacent_equal_bases,
            control_diagonal_only=request.control_diagonal_only,
            identity_as_relative_ancilla_phase=(
                request.identity_as_relative_ancilla_phase
            ),
            tail_id=request.tail_id,
            tail_hash=request.tail_hash,
        )

    def build_sequence(
        self,
        request: DFRTEEventSequenceCircuitRequest,
    ) -> DFRTEEventCircuitResult:
        return self._build(
            events=request.events,
            component_specs=request.component_specs,
            controlled=request.controlled,
            ancilla_qubit=request.ancilla_qubit,
            cancel_adjacent_equal_bases=request.cancel_adjacent_equal_bases,
            control_diagonal_only=request.control_diagonal_only,
            identity_as_relative_ancilla_phase=(
                request.identity_as_relative_ancilla_phase
            ),
            tail_id=request.tail_id,
            tail_hash=request.tail_hash,
        )

    def _build(
        self,
        *,
        events: tuple[RTEEvent, ...],
        component_specs: tuple[DFRTECircuitSpec, ...],
        controlled: bool,
        ancilla_qubit: int | None,
        cancel_adjacent_equal_bases: bool,
        control_diagonal_only: bool,
        identity_as_relative_ancilla_phase: bool,
        tail_id: str | None,
        tail_hash: str | None,
    ) -> DFRTEEventCircuitResult:
        if not events:
            raise ValueError("At least one RTE event is required.")
        if controlled and not control_diagonal_only:
            raise ValueError("DF controlled events must not control basis changes.")
        if controlled and not identity_as_relative_ancilla_phase:
            raise ValueError("Controlled identity phases must remain ancilla-relative.")
        num_system_qubits = self._num_system_qubits(component_specs)
        if controlled:
            if ancilla_qubit is None:
                raise ValueError("A controlled event requires an ancilla qubit.")
            if ancilla_qubit < num_system_qubits:
                raise ValueError("Ancilla qubit must not overlap the system register.")
            num_circuit_qubits = ancilla_qubit + 1
        else:
            num_circuit_qubits = num_system_qubits
        circuit = QuantumCircuit(num_circuit_qubits)
        flattened: list[tuple[RTEEvent, RTEEventApplication]] = []
        accumulated_phase = 0.0
        for event in events:
            _validate_event_application_order(event)
            accumulated_phase += _event_phase_angle(event.phase)
            flattened.extend((event, application) for application in event.application_sequence)

        reuse_enabled = bool(cancel_adjacent_equal_bases)
        reuse_policy = (
            "raw_adjacent_equal_basis" if reuse_enabled else "disabled"
        )
        nonidentity_count = sum(
            not application.is_identity for _event, application in flattened
        )
        active_definition: DFBasisDefinition | None = None
        active_key: tuple[str, str] | None = None
        last_group_key: tuple[str, str] | None = None
        emitted_basis_changes = 0
        cancelled_pairs = 0
        basis_switches = 0

        for event, application in flattened:
            if application.is_identity:
                if active_definition is not None:
                    self._append_basis(circuit, active_definition, inverse=False)
                    emitted_basis_changes += 1
                    active_definition = None
                    active_key = None
                accumulated_phase += self._identity_application_phase(
                    event,
                    application,
                )
                continue

            definition = self._resolve_basis(application, num_system_qubits)
            key = (definition.basis_id, definition.basis_hash)
            can_reuse = reuse_enabled and active_key == key
            if can_reuse:
                cancelled_pairs += 1
            else:
                if active_definition is not None:
                    self._append_basis(circuit, active_definition, inverse=False)
                    emitted_basis_changes += 1
                if last_group_key is not None and last_group_key != key:
                    basis_switches += 1
                self._append_basis(circuit, definition, inverse=True)
                emitted_basis_changes += 1
                active_definition = definition
                active_key = key
                last_group_key = key

            accumulated_phase += self._append_central_application(
                circuit,
                event,
                application,
                controlled=controlled,
                ancilla_qubit=ancilla_qubit,
            )
            if not reuse_enabled:
                self._append_basis(circuit, definition, inverse=False)
                emitted_basis_changes += 1
                active_definition = None
                active_key = None

        if active_definition is not None:
            self._append_basis(circuit, active_definition, inverse=False)
            emitted_basis_changes += 1

        if controlled:
            if accumulated_phase != 0.0:
                circuit.append(PhaseGate(accumulated_phase), [ancilla_qubit])
            global_phase = 0.0
            relative_phase = accumulated_phase
        else:
            circuit.global_phase += accumulated_phase
            global_phase = accumulated_phase
            relative_phase = 0.0

        fingerprint = _circuit_fingerprint(
            events,
            tail_id=tail_id,
            tail_hash=tail_hash,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            basis_reuse_policy=reuse_policy,
        )
        return DFRTEEventCircuitResult(
            circuit=circuit,
            preserved_component_order=tuple(
                application.component_id for _event, application in flattened
            ),
            cancelled_basis_change_pairs=cancelled_pairs,
            relative_ancilla_phase=float(relative_phase),
            basis_switch_count=basis_switches,
            controlled=controlled,
            event_count=len(events),
            application_count=len(flattened),
            naive_basis_change_count=2 * nonidentity_count,
            emitted_basis_change_count=emitted_basis_changes,
            accumulated_global_phase=float(global_phase),
            basis_reuse_policy=reuse_policy,
            circuit_qubit_count=circuit.num_qubits,
            circuit_fingerprint=fingerprint,
        )

    @staticmethod
    def _num_system_qubits(
        component_specs: tuple[DFRTECircuitSpec, ...],
    ) -> int:
        if not component_specs:
            raise ValueError("DF event circuit specs must not be empty.")
        sizes = {spec.num_system_qubits for spec in component_specs}
        if len(sizes) != 1:
            raise ValueError("All DF event circuit specs must use one system size.")
        num_system_qubits = next(iter(sizes))
        if num_system_qubits <= 0:
            raise ValueError("num_system_qubits must be positive.")
        return num_system_qubits

    def _resolve_basis(
        self,
        application: RTEEventApplication,
        num_system_qubits: int,
    ) -> DFBasisDefinition:
        if application.basis_id is None or application.basis_hash is None:
            raise ValueError("Non-identity DF applications require basis ID and hash.")
        definition = self._basis_registry.definition(application.basis_id)
        if definition.basis_hash != application.basis_hash:
            raise ValueError("Event basis hash does not match the builder registry.")
        if definition.num_system_qubits != num_system_qubits:
            raise ValueError("Event basis system size does not match circuit specs.")
        if definition.metadata.operations != application.basis_change_operations:
            raise ValueError("Event basis operations do not match the builder registry.")
        return definition

    @staticmethod
    def _append_basis(
        circuit: QuantumCircuit,
        definition: DFBasisDefinition,
        *,
        inverse: bool,
    ) -> None:
        operations = list(definition.runtime_operations)
        if inverse:
            operations.reverse()
        for gate, qubits in operations:
            circuit.append(gate.inverse() if inverse else gate, list(qubits))

    @staticmethod
    def _identity_application_phase(
        event: RTEEvent,
        application: RTEEventApplication,
    ) -> float:
        if application.role == "product":
            return math.pi if application.coefficient_sign == -1 else 0.0
        if application.role == "rotation":
            return -event.unsigned_rotation_angle
        raise ValueError(f"Unsupported RTE application role: {application.role}")

    @staticmethod
    def _append_central_application(
        circuit: QuantumCircuit,
        event: RTEEvent,
        application: RTEEventApplication,
        *,
        controlled: bool,
        ancilla_qubit: int | None,
    ) -> float:
        support = application.diagonal_pauli_support
        if support is None or len(support) not in (1, 2):
            raise ValueError("Non-identity DF applications require Z or ZZ support.")
        if application.role == "product":
            if controlled:
                for qubit in support:
                    circuit.cz(ancilla_qubit, qubit)
            else:
                for qubit in support:
                    circuit.append(ZGate(), [qubit])
            return math.pi if application.coefficient_sign == -1 else 0.0
        if application.role != "rotation":
            raise ValueError(f"Unsupported RTE application role: {application.role}")
        angle = event.unsigned_rotation_angle
        if len(support) == 1:
            gate = RZGate(2.0 * angle)
        else:
            gate = RZZGate(2.0 * angle)
        if controlled:
            circuit.append(gate.control(1), [ancilla_qubit, *support])
        else:
            circuit.append(gate, list(support))
        return 0.0

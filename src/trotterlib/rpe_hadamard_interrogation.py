"""Single-interrogation X/Y wrappers for short controlled DF partial-S2 circuits.

This module wraps one already-controlled repeated partial-S2 trajectory.  It
does not prepare the system state, execute shots, reconstruct an RPE phase, or
build a full multi-round RPE circuit.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Clbit

from .df_partial_s2_repeated import DFPartialS2RepeatedCircuitResult
from .rte import require_integer_count


RPEHadamardAxis: TypeAlias = Literal["cosine", "sine"]
RPEHadamardSignalComponent: TypeAlias = Literal["real", "imaginary"]
RPEHadamardInterrogationScope: TypeAlias = Literal[
    "single_hadamard_interrogation_without_state_preparation"
]

RPE_HADAMARD_INTERROGATION_SCHEMA_VERSION = (
    "rpe_hadamard_interrogation_wrapper_v1"
)
RPE_HADAMARD_CIRCUIT_SEMANTICS_SCHEMA_VERSION = (
    "rpe_hadamard_interrogation_circuit_semantics_v1"
)
RPE_HADAMARD_INTERROGATION_SCOPE: RPEHadamardInterrogationScope = (
    "single_hadamard_interrogation_without_state_preparation"
)
RPE_HADAMARD_BIT_VALUE_MAPPING = ((0, 1), (1, -1))
RPE_HADAMARD_ALLOWED_REPETITION_COUNTS = (1, 2, 4)


def round_index_for_short_rpe_repetition_count(repetition_count: int) -> int:
    """Map the directly constructible ``q_m`` values 1, 2, and 4 to ``m``."""
    count = require_integer_count(
        repetition_count,
        name="repetition_count",
        minimum=1,
    )
    if count not in RPE_HADAMARD_ALLOWED_REPETITION_COUNTS:
        raise ValueError(
            "A short RPE Hadamard interrogation requires "
            "repetition_count in (1, 2, 4)."
        )
    return count.bit_length() - 1


@dataclass(frozen=True)
class RPEHadamardInterrogationRequest:
    """One axis wrapper request for an existing controlled trajectory."""

    evolution: DFPartialS2RepeatedCircuitResult
    axis: RPEHadamardAxis
    include_measurement: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.evolution, DFPartialS2RepeatedCircuitResult):
            raise TypeError(
                "evolution must be a DFPartialS2RepeatedCircuitResult."
            )
        if self.axis not in ("cosine", "sine"):
            raise ValueError("axis must be 'cosine' or 'sine'.")
        if not isinstance(self.include_measurement, bool):
            raise TypeError("include_measurement must be boolean.")


@dataclass(frozen=True)
class RPEHadamardInterrogationResult:
    """One constructed interrogation circuit and its audit metadata."""

    circuit: QuantumCircuit
    round_index: int
    repetition_count: int
    q_m: int
    step_time: float
    delta_time: float
    total_evolution_time: float
    t_m: float
    axis: RPEHadamardAxis
    ancilla_qubit: int
    measurement_clbit: Clbit | None
    measurement_clbit_index: int | None
    measurement_register_name: str | None
    bit_value_mapping: tuple[tuple[int, int], ...]
    signal_component: RPEHadamardSignalComponent
    signal_definition: str
    estimator_definition: str
    wrapped_trajectory_fingerprint: str
    wrapped_provenance_fingerprint: str
    wrapped_circuit_semantics_fingerprint: str
    wrapper_fingerprint: str
    wrapper_circuit_semantics_fingerprint: str
    compiler_independent_fingerprint: str
    include_measurement: bool
    state_preparation_included: bool
    backend_execution_included: bool
    quantum_shots_executed: int
    wrapped_evolution_already_controlled: bool
    additional_control_applied: bool
    fresh_iid_trajectory_per_hadamard_shot_verified: bool
    constant_phase: float
    extracted_identity_phase: float
    rte_relative_phase: float
    circuit_scope: RPEHadamardInterrogationScope
    wrapper_schema_version: str
    circuit_granularity: Literal["single_rpe_hadamard_interrogation"] = (
        "single_rpe_hadamard_interrogation"
    )


class QiskitRPEHadamardInterrogationBuilder:
    """Wrap an already-controlled short trajectory without re-controlling it."""

    @staticmethod
    def _validate_evolution(
        evolution: DFPartialS2RepeatedCircuitResult,
    ) -> tuple[int, int, float]:
        if not evolution.controlled:
            raise ValueError("The wrapped repeated evolution must be controlled.")
        if evolution.ancilla_qubit is None:
            raise ValueError("The wrapped repeated evolution requires an ancilla.")
        round_index = round_index_for_short_rpe_repetition_count(
            evolution.repetition_count
        )
        circuit = evolution.circuit
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("evolution.circuit must be a QuantumCircuit.")
        if evolution.circuit_qubit_count != circuit.num_qubits:
            raise ValueError(
                "Recorded circuit_qubit_count does not match the evolution circuit."
            )
        ancilla = require_integer_count(
            evolution.ancilla_qubit,
            name="ancilla_qubit",
        )
        if circuit.num_qubits < 2 or ancilla != circuit.num_qubits - 1:
            raise ValueError(
                "The ancilla must be immediately after the system register."
            )
        if circuit.num_clbits != 0 or any(
            instruction.operation.name == "measure"
            for instruction in circuit.data
        ):
            raise ValueError(
                "The wrapped evolution must not contain classical bits or "
                "measurements."
            )
        if len(evolution.step_results) != evolution.repetition_count:
            raise ValueError(
                "The repeated evolution must contain one step result per repetition."
            )
        if any(
            not step.controlled
            or step.ancilla_qubit != ancilla
            or step.circuit_qubit_count != circuit.num_qubits
            for step in evolution.step_results
        ):
            raise ValueError(
                "Repeated-step control metadata is inconsistent with the wrapper."
            )
        if not evolution.trajectory_fingerprint:
            raise ValueError("The wrapped trajectory fingerprint must not be empty.")
        if not evolution.circuit_semantics_fingerprint:
            raise ValueError(
                "The wrapped circuit-semantics fingerprint must not be empty."
            )
        delta_time = float(evolution.step_time)
        if not math.isfinite(delta_time):
            raise ValueError("The wrapped evolution step_time must be finite.")
        total_time = float(evolution.repetition_count * delta_time)
        if not math.isfinite(total_time):
            raise ValueError("Derived t_m=q_m*delta_time must be finite.")
        return round_index, ancilla, total_time

    @staticmethod
    def _new_wrapper_circuit(evolution: QuantumCircuit) -> QuantumCircuit:
        return QuantumCircuit(
            *evolution.qregs,
            name="rpe_hadamard_interrogation",
        )

    @staticmethod
    def _wrapper_fingerprint(
        *,
        request: RPEHadamardInterrogationRequest,
        round_index: int,
        ancilla_qubit: int,
        total_time: float,
        signal_component: RPEHadamardSignalComponent,
    ) -> str:
        evolution = request.evolution
        payload = {
            "wrapper_schema_version": RPE_HADAMARD_INTERROGATION_SCHEMA_VERSION,
            "axis": request.axis,
            "include_measurement": request.include_measurement,
            "bit_value_mapping": RPE_HADAMARD_BIT_VALUE_MAPPING,
            "signal_definition": "Z_m=<psi|U_m|psi>",
            "signal_component": signal_component,
            "estimator_definition": (
                "E[(-1)^b]=Re(Z_m)"
                if request.axis == "cosine"
                else "E[(-1)^b]=Im(Z_m)"
            ),
            "sine_axis_gate_order": (
                None if request.axis == "cosine" else "Sdg_then_H"
            ),
            "ancilla_qubit": ancilla_qubit,
            "round_index": round_index,
            "q_m": evolution.repetition_count,
            "delta_time": float(evolution.step_time).hex(),
            "t_m": total_time.hex(),
            "wrapped_trajectory_fingerprint": (
                evolution.trajectory_fingerprint
            ),
            "wrapped_provenance_fingerprint": (
                evolution.provenance_fingerprint
            ),
            "wrapped_circuit_semantics_fingerprint": (
                evolution.circuit_semantics_fingerprint
            ),
            "circuit_scope": RPE_HADAMARD_INTERROGATION_SCOPE,
            "state_preparation_included": False,
            "additional_control_applied": False,
            "fresh_iid_trajectory_per_hadamard_shot_verified": False,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _wrapper_circuit_semantics_fingerprint(
        *,
        request: RPEHadamardInterrogationRequest,
        round_index: int,
        ancilla_qubit: int,
        total_time: float,
    ) -> str:
        evolution = request.evolution
        payload = {
            "circuit_semantics_schema_version": (
                RPE_HADAMARD_CIRCUIT_SEMANTICS_SCHEMA_VERSION
            ),
            "wrapper_schema_version": RPE_HADAMARD_INTERROGATION_SCHEMA_VERSION,
            "wrapped_circuit_semantics_fingerprint": (
                evolution.circuit_semantics_fingerprint
            ),
            "axis": request.axis,
            "include_measurement": request.include_measurement,
            "wrapper_gate_sequence": (
                "H_wrapped_evolution_H"
                if request.axis == "cosine"
                else "H_wrapped_evolution_Sdg_H"
            ),
            "ancilla_qubit": ancilla_qubit,
            "round_index": round_index,
            "q_m": evolution.repetition_count,
            "delta_time": float(evolution.step_time).hex(),
            "t_m": total_time.hex(),
            "circuit_scope": RPE_HADAMARD_INTERROGATION_SCOPE,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def build(
        self,
        request: RPEHadamardInterrogationRequest,
    ) -> RPEHadamardInterrogationResult:
        """Construct one X/cosine or Y/sine Hadamard interrogation."""
        if not isinstance(request, RPEHadamardInterrogationRequest):
            raise TypeError(
                "request must be an RPEHadamardInterrogationRequest."
            )
        evolution = request.evolution
        round_index, ancilla, total_time = self._validate_evolution(evolution)
        signal_component: RPEHadamardSignalComponent = (
            "real" if request.axis == "cosine" else "imaginary"
        )

        circuit = self._new_wrapper_circuit(evolution.circuit)
        circuit.h(ancilla)
        circuit.compose(
            evolution.circuit,
            qubits=tuple(range(evolution.circuit.num_qubits)),
            inplace=True,
        )
        if request.axis == "sine":
            circuit.sdg(ancilla)
        circuit.h(ancilla)

        measurement_clbit: Clbit | None = None
        measurement_clbit_index: int | None = None
        measurement_register_name: str | None = None
        if request.include_measurement:
            measurement_register = ClassicalRegister(1, "rpe_measure")
            circuit.add_register(measurement_register)
            measurement_clbit = measurement_register[0]
            measurement_clbit_index = circuit.find_bit(measurement_clbit).index
            measurement_register_name = measurement_register.name
            circuit.measure(ancilla, measurement_clbit)

        wrapper_fingerprint = self._wrapper_fingerprint(
            request=request,
            round_index=round_index,
            ancilla_qubit=ancilla,
            total_time=total_time,
            signal_component=signal_component,
        )
        wrapper_circuit_semantics_fingerprint = (
            self._wrapper_circuit_semantics_fingerprint(
                request=request,
                round_index=round_index,
                ancilla_qubit=ancilla,
                total_time=total_time,
            )
        )
        estimator_definition = (
            "E[(-1)^b]=Re(Z_m)"
            if request.axis == "cosine"
            else "E[(-1)^b]=Im(Z_m)"
        )
        return RPEHadamardInterrogationResult(
            circuit=circuit,
            round_index=round_index,
            repetition_count=evolution.repetition_count,
            q_m=evolution.repetition_count,
            step_time=float(evolution.step_time),
            delta_time=float(evolution.step_time),
            total_evolution_time=total_time,
            t_m=total_time,
            axis=request.axis,
            ancilla_qubit=ancilla,
            measurement_clbit=measurement_clbit,
            measurement_clbit_index=measurement_clbit_index,
            measurement_register_name=measurement_register_name,
            bit_value_mapping=RPE_HADAMARD_BIT_VALUE_MAPPING,
            signal_component=signal_component,
            signal_definition="Z_m=<psi|U_m|psi>",
            estimator_definition=estimator_definition,
            wrapped_trajectory_fingerprint=evolution.trajectory_fingerprint,
            wrapped_provenance_fingerprint=evolution.provenance_fingerprint,
            wrapped_circuit_semantics_fingerprint=(
                evolution.circuit_semantics_fingerprint
            ),
            wrapper_fingerprint=wrapper_fingerprint,
            wrapper_circuit_semantics_fingerprint=(
                wrapper_circuit_semantics_fingerprint
            ),
            compiler_independent_fingerprint=(
                wrapper_circuit_semantics_fingerprint
            ),
            include_measurement=request.include_measurement,
            state_preparation_included=False,
            backend_execution_included=False,
            quantum_shots_executed=0,
            wrapped_evolution_already_controlled=True,
            additional_control_applied=False,
            fresh_iid_trajectory_per_hadamard_shot_verified=False,
            constant_phase=evolution.constant_phase,
            extracted_identity_phase=evolution.extracted_identity_phase,
            rte_relative_phase=evolution.rte_relative_phase,
            circuit_scope=RPE_HADAMARD_INTERROGATION_SCOPE,
            wrapper_schema_version=RPE_HADAMARD_INTERROGATION_SCHEMA_VERSION,
        )

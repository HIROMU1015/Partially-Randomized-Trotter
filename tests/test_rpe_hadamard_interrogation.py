from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, Statevector

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.df_partial_s2_repeated import (
    QiskitDFPartialS2RepeatedCircuitBuilder,
    make_df_partial_s2_repeated_request,
)
from trotterlib.rpe_hadamard_interrogation import (
    RPE_HADAMARD_BIT_VALUE_MAPPING,
    RPE_HADAMARD_INTERROGATION_SCOPE,
    QiskitRPEHadamardInterrogationBuilder,
    RPEHadamardInterrogationRequest,
    round_index_for_short_rpe_repetition_count,
)
from trotterlib.rte import make_rte_config


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.13,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.4, -0.3]),
        g_matrices=(
            np.asarray([[1.0]], dtype=np.complex128),
            np.asarray([[0.7]], dtype=np.complex128),
        ),
        metadata={"name": "rpe-hadamard-interrogation"},
    )


def _controlled_evolution(
    *,
    deterministic_only: bool,
    repetition_count: int = 1,
    seed: int = 7,
    step_time: float = 0.11,
):
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(
            hamiltonian,
            hamiltonian.n_blocks if deterministic_only else 1,
        ),
        identity_policy="extract_identity_phase",
    )
    if deterministic_only:
        config = None
        distribution = None
    else:
        config, distribution = make_rte_config(
            preparation.rte_preparation.symbolic_tail,
            evolution_time=step_time,
            rte_steps=1,
            truncation_tolerance=1.0,
            finite_taylor_order=2,
        )
    request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=step_time,
        repetition_count=repetition_count,
        rte_config=config,
        rte_distribution=distribution,
        seed=seed,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        construction_policy="boundary_optimized",
    )
    evolution = QiskitDFPartialS2RepeatedCircuitBuilder().build(request)
    return preparation, evolution


def _build(evolution, axis: str, *, include_measurement: bool = False):
    return QiskitRPEHadamardInterrogationBuilder().build(
        RPEHadamardInterrogationRequest(
            evolution=evolution,
            axis=axis,
            include_measurement=include_measurement,
        )
    )


def _direct_signal(evolution, system_state: np.ndarray) -> complex:
    controlled = np.asarray(Operator(evolution.circuit).data)
    dimension = system_state.size
    unitary = controlled[dimension:, dimension:]
    return complex(np.vdot(system_state, unitary @ system_state))


def _wrapped_z_expectation(circuit: QuantumCircuit, system_state: np.ndarray) -> float:
    initial = np.concatenate(
        (system_state, np.zeros_like(system_state)),
    )
    state = Statevector(initial).evolve(circuit)
    observable = Pauli("Z" + "I" * int(math.log2(system_state.size)))
    return float(np.real(state.expectation_value(observable)))


@pytest.mark.parametrize(
    ("repetition_count", "expected_round"),
    ((1, 0), (2, 1), (4, 2)),
)
def test_short_repetition_count_maps_to_round_and_time(
    repetition_count: int,
    expected_round: int,
) -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=True,
        repetition_count=repetition_count,
    )
    result = _build(evolution, "cosine")

    assert round_index_for_short_rpe_repetition_count(repetition_count) == (
        expected_round
    )
    assert result.round_index == expected_round
    assert result.q_m == repetition_count
    assert result.repetition_count == repetition_count
    assert result.delta_time == pytest.approx(0.11)
    assert result.t_m == pytest.approx(repetition_count * 0.11)
    assert result.total_evolution_time == result.t_m


@pytest.mark.parametrize("invalid_count", (3, 8))
def test_non_short_or_non_power_of_two_repetition_count_is_rejected(
    invalid_count: int,
) -> None:
    with pytest.raises(ValueError, match=r"in \(1, 2, 4\)"):
        round_index_for_short_rpe_repetition_count(invalid_count)


@pytest.mark.parametrize("deterministic_only", (True, False))
@pytest.mark.parametrize(
    ("axis", "component"),
    (("cosine", "real"), ("sine", "imaginary")),
)
def test_wrapper_expectation_matches_wrapped_controlled_signal(
    deterministic_only: bool,
    axis: str,
    component: str,
) -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=deterministic_only,
        repetition_count=2,
    )
    system_state = np.asarray(
        [math.sqrt(0.3), 1j * math.sqrt(0.7)],
        dtype=np.complex128,
    )[: 1 << (evolution.circuit.num_qubits - 1)]
    system_state /= np.linalg.norm(system_state)
    signal = _direct_signal(evolution, system_state)
    result = _build(evolution, axis)
    actual = _wrapped_z_expectation(result.circuit, system_state)
    expected = signal.real if axis == "cosine" else signal.imag

    assert actual == pytest.approx(expected, abs=1e-12)
    assert result.signal_component == component
    assert result.estimator_definition == (
        "E[(-1)^b]=Re(Z_m)"
        if axis == "cosine"
        else "E[(-1)^b]=Im(Z_m)"
    )


def test_known_negative_sine_sign_matches_exp_minus_i_energy_time() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=True,
        repetition_count=1,
    )
    theta = 0.41
    known_phase = QuantumCircuit(evolution.circuit.num_qubits)
    known_phase.p(-theta, evolution.ancilla_qubit)
    known_evolution = replace(
        evolution,
        circuit=known_phase,
        circuit_semantics_fingerprint="known-exp-minus-i-theta",
        compiler_independent_fingerprint="known-exp-minus-i-theta",
        untranspiled_circuit_size=int(known_phase.size()),
        untranspiled_circuit_depth=int(known_phase.depth() or 0),
    )
    result = _build(known_evolution, "sine")
    system_state = np.asarray([1.0, 0.0], dtype=np.complex128)[
        : 1 << (known_phase.num_qubits - 1)
    ]

    assert _wrapped_z_expectation(result.circuit, system_state) == pytest.approx(
        -math.sin(theta),
        abs=1e-12,
    )
    assert result.estimator_definition == "E[(-1)^b]=Im(Z_m)"


def test_bit_mapping_and_sine_gate_order_are_explicit() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=False,
        repetition_count=1,
    )
    result = _build(evolution, "sine")
    operation_names = tuple(
        instruction.operation.name for instruction in result.circuit.data
    )
    evolution_size = len(evolution.circuit.data)

    assert result.bit_value_mapping == ((0, 1), (1, -1))
    assert result.bit_value_mapping == RPE_HADAMARD_BIT_VALUE_MAPPING
    assert operation_names[0] == "h"
    assert operation_names[1 + evolution_size :] == ("sdg", "h")


def test_cosine_and_sine_wrap_exactly_the_same_trajectory() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=False,
        repetition_count=4,
        seed=91,
    )
    cosine = _build(evolution, "cosine")
    sine = _build(evolution, "sine")

    assert cosine.wrapped_trajectory_fingerprint == (
        sine.wrapped_trajectory_fingerprint
    )
    assert cosine.wrapped_provenance_fingerprint == (
        sine.wrapped_provenance_fingerprint
    )
    assert cosine.wrapped_circuit_semantics_fingerprint == (
        sine.wrapped_circuit_semantics_fingerprint
    )
    assert cosine.wrapper_fingerprint != sine.wrapper_fingerprint
    assert cosine.wrapper_circuit_semantics_fingerprint != (
        sine.wrapper_circuit_semantics_fingerprint
    )


def test_input_evolution_is_not_modified_and_no_control_is_added() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=False,
        repetition_count=2,
    )
    original = evolution.circuit.copy()
    result = _build(evolution, "sine")

    assert evolution.circuit == original
    assert result.wrapped_evolution_already_controlled is True
    assert result.additional_control_applied is False
    assert result.circuit.size() == evolution.circuit.size() + 3
    wrapped_names = tuple(
        instruction.operation.name
        for instruction in result.circuit.data[1 : 1 + len(evolution.circuit.data)]
    )
    original_names = tuple(
        instruction.operation.name for instruction in evolution.circuit.data
    )
    assert wrapped_names == original_names


def test_measurement_is_optional_and_uses_one_new_classical_bit() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=True,
        repetition_count=1,
    )
    without_measurement = _build(
        evolution,
        "cosine",
        include_measurement=False,
    )
    with_measurement = _build(
        evolution,
        "cosine",
        include_measurement=True,
    )

    Operator(without_measurement.circuit)
    assert without_measurement.circuit.num_clbits == 0
    assert without_measurement.measurement_clbit is None
    assert without_measurement.measurement_clbit_index is None
    assert with_measurement.circuit.num_clbits == 1
    assert with_measurement.measurement_clbit is not None
    assert with_measurement.measurement_clbit_index == 0
    assert with_measurement.measurement_register_name == "rpe_measure"
    assert with_measurement.circuit.data[-1].operation.name == "measure"
    assert without_measurement.wrapper_fingerprint != (
        with_measurement.wrapper_fingerprint
    )
    assert without_measurement.wrapper_circuit_semantics_fingerprint != (
        with_measurement.wrapper_circuit_semantics_fingerprint
    )


def test_wrapper_fingerprint_is_reproducible_and_binds_trajectory() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=False,
        repetition_count=2,
        seed=12,
    )
    first = _build(evolution, "cosine")
    second = _build(evolution, "cosine")
    changed_trajectory = _build(
        replace(
            evolution,
            trajectory_fingerprint="different-trajectory",
            provenance_fingerprint="different-trajectory",
        ),
        "cosine",
    )

    assert first.wrapper_fingerprint == second.wrapper_fingerprint
    assert first.wrapper_circuit_semantics_fingerprint == (
        second.wrapper_circuit_semantics_fingerprint
    )
    assert first.compiler_independent_fingerprint == (
        first.wrapper_circuit_semantics_fingerprint
    )
    assert first.wrapper_fingerprint != changed_trajectory.wrapper_fingerprint
    assert first.wrapper_circuit_semantics_fingerprint == (
        changed_trajectory.wrapper_circuit_semantics_fingerprint
    )
    assert len(first.wrapper_fingerprint) == 64


def test_deterministic_seed_changes_only_audit_fingerprint() -> None:
    _preparation, first_evolution = _controlled_evolution(
        deterministic_only=True,
        repetition_count=2,
        seed=12,
    )
    _preparation, second_evolution = _controlled_evolution(
        deterministic_only=True,
        repetition_count=2,
        seed=13,
    )
    first = _build(first_evolution, "cosine")
    second = _build(second_evolution, "cosine")

    assert first_evolution.circuit == second_evolution.circuit
    assert first.wrapped_trajectory_fingerprint != (
        second.wrapped_trajectory_fingerprint
    )
    assert first.wrapped_circuit_semantics_fingerprint == (
        second.wrapped_circuit_semantics_fingerprint
    )
    assert first.wrapper_fingerprint != second.wrapper_fingerprint
    assert first.wrapper_circuit_semantics_fingerprint == (
        second.wrapper_circuit_semantics_fingerprint
    )
    assert first.compiler_independent_fingerprint == (
        second.compiler_independent_fingerprint
    )


def test_wrapper_semantics_fingerprint_binds_wrapped_semantics() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=True,
        repetition_count=1,
    )
    original = _build(evolution, "cosine")
    changed = _build(
        replace(
            evolution,
            circuit_semantics_fingerprint="different-circuit-semantics",
            compiler_independent_fingerprint="different-circuit-semantics",
        ),
        "cosine",
    )

    assert original.wrapper_circuit_semantics_fingerprint != (
        changed.wrapper_circuit_semantics_fingerprint
    )
    assert original.compiler_independent_fingerprint != (
        changed.compiler_independent_fingerprint
    )


def test_control_and_short_round_input_guards() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=True,
        repetition_count=1,
    )
    builder = QiskitRPEHadamardInterrogationBuilder()

    with pytest.raises(ValueError, match="must be controlled"):
        builder.build(
            RPEHadamardInterrogationRequest(
                replace(evolution, controlled=False, ancilla_qubit=None),
                "cosine",
                False,
            )
        )
    with pytest.raises(ValueError, match="immediately after"):
        builder.build(
            RPEHadamardInterrogationRequest(
                replace(evolution, ancilla_qubit=0),
                "cosine",
                False,
            )
        )
    with pytest.raises(ValueError, match=r"in \(1, 2, 4\)"):
        builder.build(
            RPEHadamardInterrogationRequest(
                replace(evolution, repetition_count=8),
                "cosine",
                False,
            )
        )


def test_evolution_with_classical_data_or_measurement_is_rejected() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=True,
        repetition_count=1,
    )
    measured = evolution.circuit.copy()
    classical = ClassicalRegister(1, "unexpected")
    measured.add_register(classical)
    measured.measure(evolution.ancilla_qubit, classical[0])

    with pytest.raises(ValueError, match="classical bits or measurements"):
        _build(replace(evolution, circuit=measured), "cosine")


def test_request_rejects_invalid_axis_and_non_result_input() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=True,
        repetition_count=1,
    )
    with pytest.raises(ValueError, match="axis"):
        RPEHadamardInterrogationRequest(evolution, "invalid", False)
    with pytest.raises(TypeError, match="DFPartialS2RepeatedCircuitResult"):
        RPEHadamardInterrogationRequest(evolution.circuit, "cosine", False)


def test_relative_phases_and_input_global_phase_are_preserved() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=False,
        repetition_count=2,
    )
    assert evolution.constant_phase != 0.0
    assert evolution.extracted_identity_phase != 0.0
    phased_circuit = evolution.circuit.copy()
    phased_circuit.global_phase += 0.23
    phased = replace(
        evolution,
        circuit=phased_circuit,
        circuit_semantics_fingerprint="same-evolution-with-global-phase",
    )
    result = _build(phased, "cosine")

    assert float(result.circuit.global_phase) == pytest.approx(
        float(phased_circuit.global_phase)
    )
    assert result.constant_phase == evolution.constant_phase
    assert result.extracted_identity_phase == evolution.extracted_identity_phase
    assert result.rte_relative_phase == evolution.rte_relative_phase
    system_state = np.asarray([1.0, 0.0], dtype=np.complex128)[
        : 1 << (evolution.circuit.num_qubits - 1)
    ]
    assert _wrapped_z_expectation(result.circuit, system_state) == pytest.approx(
        _direct_signal(evolution, system_state).real,
        abs=1e-12,
    )


def test_scope_and_fresh_iid_claims_remain_limited() -> None:
    _preparation, evolution = _controlled_evolution(
        deterministic_only=False,
        repetition_count=1,
    )
    result = _build(evolution, "cosine", include_measurement=True)

    assert result.circuit_scope == RPE_HADAMARD_INTERROGATION_SCOPE
    assert result.circuit_scope != "full_rpe_circuit"
    assert result.state_preparation_included is False
    assert result.backend_execution_included is False
    assert result.quantum_shots_executed == 0
    assert result.fresh_iid_trajectory_per_hadamard_shot_verified is False

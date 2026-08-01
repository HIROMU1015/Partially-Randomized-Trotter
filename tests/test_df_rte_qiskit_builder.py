from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RZGate
from qiskit.quantum_info import Operator

import trotterlib.df_rte_tail as df_tail
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.df_rte_tail import (
    extract_df_diagonal_tail,
    extraction_to_normalized_rte_tail,
    prepare_df_rte_event_inputs,
)
from trotterlib.df_trotter.model import DFBlock
from trotterlib.rte import enumerate_rte_events, event_unitary, make_rte_config


def _block(
    eta: tuple[float, ...] = (1.0, 0.7),
    *,
    angle: float = 0.23,
) -> DFBlock:
    return DFBlock(
        U_ops=((RYGate(angle), (0,)),),
        eta=np.asarray(eta),
        lam=0.7,
    )


def _small_case(*, identity_policy: str = "faithful_identity_in_tail"):
    extraction = extract_df_diagonal_tail(
        "builder-small",
        (_block(),),
        identity_policy=identity_policy,
    )
    preparation = prepare_df_rte_event_inputs(extraction)
    config, distribution = make_rte_config(
        preparation.symbolic_tail,
        evolution_time=0.2,
        rte_steps=2,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    events = enumerate_rte_events(
        preparation.symbolic_tail.components,
        distribution,
        max_events=1_000,
    )
    dense_tail = extraction_to_normalized_rte_tail(extraction)
    operators = dict(
        zip(
            (item.component_id for item in dense_tail.components),
            dense_tail.operators,
            strict=True,
        )
    )
    return preparation, config, distribution, events, operators


def _controlled_reference(unitary: np.ndarray) -> np.ndarray:
    identity = np.eye(unitary.shape[0], dtype=np.complex128)
    zero = np.zeros_like(identity)
    return np.block([[identity, zero], [zero, unitary]])


def test_every_small_event_matches_dense_reference_with_all_signs_and_phases() -> None:
    preparation, _config, _distribution, events, operators = _small_case()
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )

    for event in events:
        built = builder.build_event(preparation.request_for_event(event))
        actual = np.asarray(Operator(built.circuit).data)
        expected = event_unitary(event, operators)
        assert np.allclose(actual, expected, atol=1e-12)
        assert built.preserved_component_order == event.selected_component_ids

    assert {event.taylor_order for event in events} == {0, 2}
    assert any(event.phase == -1 for event in events)
    supports = {
        application.diagonal_pauli_support
        for event in events
        for application in event.application_sequence
    }
    assert supports >= {
        (),
        (0,),
        (0, 1),
    }
    signs = {
        application.coefficient_sign
        for event in events
        for application in event.application_sequence
    }
    assert signs == {
        -1,
        1,
    }


@pytest.mark.parametrize(
    "selector",
    (
        lambda event: event.taylor_order == 0
        and event.application_sequence[-1].is_identity,
        lambda event: event.taylor_order == 2
        and any(item.is_identity for item in event.application_sequence[:-1]),
        lambda event: event.taylor_order == 2
        and event.application_sequence[-1].diagonal_pauli_support == (0, 1),
    ),
)
def test_controlled_event_is_exact_block_diagonal(selector) -> None:
    preparation, _config, _distribution, events, operators = _small_case()
    event = next(event for event in events if selector(event))
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    built = builder.build_event(
        preparation.request_for_event(
            event,
            controlled=True,
            ancilla_qubit=2,
        )
    )

    expected = _controlled_reference(event_unitary(event, operators))
    assert np.allclose(Operator(built.circuit).data, expected, atol=1e-12)
    assert built.accumulated_global_phase == 0.0
    assert built.relative_ancilla_phase != 0.0 or event.phase == 1


def test_sequence_order_and_cross_event_basis_reuse_preserve_unitary() -> None:
    preparation, config, distribution, _events, operators = _small_case(
        identity_policy="extract_identity_phase"
    )
    request = preparation.sample_occurrence_request(
        config,
        distribution,
        seed=9,
        cancel_adjacent_equal_bases=True,
    )
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    reused = builder.build_sequence(request)
    unreused = builder.build_sequence(
        replace(request, cancel_adjacent_equal_bases=False)
    )

    expected = np.eye(4, dtype=np.complex128)
    for event in request.events:
        expected = event_unitary(event, operators) @ expected
    assert np.allclose(Operator(reused.circuit).data, expected, atol=1e-12)
    assert np.allclose(Operator(unreused.circuit).data, expected, atol=1e-12)
    assert reused.event_count == config.rte_steps
    assert reused.emitted_basis_change_count < unreused.emitted_basis_change_count
    assert reused.cancelled_basis_change_pairs > 0
    assert (
        reused.naive_basis_change_count - reused.emitted_basis_change_count
        == 2 * reused.cancelled_basis_change_pairs
    )
    assert reused.application_count == sum(
        len(event.application_sequence) for event in request.events
    )
    assert reused.preserved_component_order == tuple(
        application.component_id
        for event in request.events
        for application in event.application_sequence
    )

    controlled_request = replace(request, controlled=True, ancilla_qubit=2)
    controlled = builder.build_sequence(controlled_request)
    assert np.allclose(
        Operator(controlled.circuit).data,
        _controlled_reference(expected),
        atol=1e-12,
    )


def test_different_fragment_bases_are_not_cancelled() -> None:
    extraction = extract_df_diagonal_tail(
        "two-fragments",
        (_block(angle=0.13), _block(angle=-0.31)),
        identity_policy="extract_identity_phase",
    )
    preparation = prepare_df_rte_event_inputs(extraction)
    _config, distribution = make_rte_config(
        preparation.symbolic_tail,
        evolution_time=0.05,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    events = enumerate_rte_events(
        preparation.symbolic_tail.components,
        distribution,
        max_events=10_000,
    )
    event = next(
        event
        for event in events
        if event.taylor_order == 2
        and len({item.basis_id for item in event.application_sequence}) > 1
        and all(not item.is_identity for item in event.application_sequence)
    )
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    built = builder.build_event(preparation.request_for_event(event))

    raw_equal_neighbors = sum(
        left.basis_id == right.basis_id and left.basis_hash == right.basis_hash
        for left, right in zip(
            event.application_sequence,
            event.application_sequence[1:],
        )
    )
    assert built.cancelled_basis_change_pairs == raw_equal_neighbors
    assert built.basis_switch_count >= 1

    order_zero = [item for item in events if item.taylor_order == 0]
    left = order_zero[0]
    right = next(
        item
        for item in order_zero
        if item.application_sequence[0].basis_id
        != left.application_sequence[0].basis_id
    )
    sequence = builder.build_sequence(
        DFRTEEventSequenceCircuitRequest(
            events=(left, right),
            component_specs=preparation.component_specs,
            tail_id=preparation.symbolic_tail.tail_id,
            tail_hash=preparation.symbolic_tail.tail_hash,
            occurrence_rte_steps=2,
        )
    )
    assert sequence.cancelled_basis_change_pairs == 0
    assert sequence.basis_switch_count == 1


def test_basis_hash_mismatch_and_invalid_control_register_are_rejected() -> None:
    preparation, _config, _distribution, events, _operators = _small_case()
    event = next(
        event for event in events if not event.application_sequence[0].is_identity
    )
    application = replace(event.application_sequence[0], basis_hash="wrong")
    bad_event = replace(
        event,
        application_sequence=(application, *event.application_sequence[1:]),
    )
    with pytest.raises(ValueError, match="basis hash"):
        preparation.request_for_event(bad_event)
    with pytest.raises(ValueError, match="ancilla_qubit"):
        preparation.request_for_event(event, controlled=True)

    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    with pytest.raises(ValueError, match="overlap"):
        builder.build_event(
            preparation.request_for_event(
                event,
                controlled=True,
                ancilla_qubit=0,
            )
        )

    with pytest.raises(ValueError, match="phase"):
        builder.build_event(
            preparation.request_for_event(replace(event, phase=1j))
        )


def test_controlled_circuit_controls_only_the_central_action() -> None:
    preparation, _config, _distribution, events, _operators = _small_case(
        identity_policy="extract_identity_phase"
    )
    event = next(event for event in events if event.taylor_order == 2)
    built = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    ).build_event(
        preparation.request_for_event(event, controlled=True, ancilla_qubit=2)
    )
    names = [item.operation.name for item in built.circuit.data]

    assert "ry" in names
    assert "cry" not in names
    assert "cz" in names
    assert "crz" in names
    assert all(
        built.circuit.find_bit(qubit).index != 2
        for item in built.circuit.data
        if item.operation.name == "ry"
        for qubit in item.qubits
    )


@pytest.mark.parametrize("num_qubits", (20, 26))
def test_high_qubit_occurrence_circuit_build_is_dense_free(
    num_qubits: int,
    monkeypatch,
) -> None:
    original_operator = df_tail.Operator

    def local_operator_only(value):
        if isinstance(value, QuantumCircuit):
            raise AssertionError("many-body Operator(circuit).data was requested")
        return original_operator(value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a dense DF-tail helper was called")

    monkeypatch.setattr(df_tail, "Operator", local_operator_only)
    for name in (
        "basis_change_unitary",
        "diagonal_pauli_matrix",
        "component_dense_operator",
        "dense_extracted_df_tail",
        "dense_df_block_hamiltonian",
        "extraction_to_normalized_rte_tail",
    ):
        monkeypatch.setattr(df_tail, name, forbidden)

    extraction = extract_df_diagonal_tail(
        "high-qubit-builder",
        (_block(tuple(np.linspace(0.2, 1.1, num_qubits))),),
        identity_policy="extract_identity_phase",
    )
    preparation = prepare_df_rte_event_inputs(extraction)
    config, distribution = make_rte_config(
        preparation.symbolic_tail,
        evolution_time=0.001,
        rte_steps=2,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    request = preparation.sample_occurrence_request(
        config,
        distribution,
        seed=14,
        controlled=True,
        ancilla_qubit=num_qubits,
    )
    built = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    ).build_sequence(request)

    assert built.circuit_qubit_count == num_qubits + 1
    assert built.event_count == 2
    assert built.circuit.size() > 0
    assert built.circuit_fingerprint

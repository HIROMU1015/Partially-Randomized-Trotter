from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.library import RYGate, RZGate

import trotterlib.df_rte_tail as df_tail
from trotterlib.df_rte_tail import (
    BasisFingerprintError,
    DFBasisRegistry,
    extract_df_diagonal_tail,
    extraction_to_normalized_rte_tail,
    extraction_to_symbolic_rte_tail,
    prepare_df_rte_event_inputs,
)
from trotterlib.df_trotter.model import DFBlock
from trotterlib.rte import (
    DeterministicOnlyRTETailError,
    finite_rte_distribution,
    make_rte_config,
    sample_rte_events,
)


def _block(num_qubits: int, angle: float = 0.17) -> DFBlock:
    return DFBlock(
        U_ops=((RYGate(angle), (0,)),),
        eta=np.linspace(0.2, 1.1, num_qubits),
        lam=0.37,
    )


@pytest.mark.parametrize("num_qubits", (20, 26))
def test_high_qubit_symbolic_tail_to_event_request_is_dense_free(
    num_qubits,
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

    extraction = extract_df_diagonal_tail("integrated", (_block(num_qubits),))
    preparation = prepare_df_rte_event_inputs(extraction)
    config, distribution = make_rte_config(
        preparation.symbolic_tail,
        evolution_time=0.01,
        rte_steps=np.int64(2),
        truncation_tolerance=1e-8,
    )
    requests = preparation.sample_requests(
        distribution,
        sample_count=np.int64(3),
        seed=41,
        controlled=True,
        ancilla_qubit=num_qubits,
    )
    repeated = preparation.sample_requests(
        distribution,
        sample_count=3,
        seed=41,
        controlled=True,
        ancilla_qubit=num_qubits,
    )

    assert config.tail_hash == extraction.tail_hash
    assert len(requests) == 3
    assert [item.event.selected_component_ids for item in requests] == [
        item.event.selected_component_ids for item in repeated
    ]
    for request in requests:
        definitions = preparation.resolve_event_basis_definitions(request.event)
        assert len(definitions) == len(request.event.application_sequence)
        assert all(definition.runtime_operations for definition in definitions)
        assert request.ancilla_qubit == num_qubits

    event = requests[0].event
    bad_application = replace(event.application_sequence[0], basis_hash="wrong")
    bad_event = replace(
        event,
        application_sequence=(bad_application, *event.application_sequence[1:]),
    )
    with pytest.raises(ValueError, match="basis hash"):
        preparation.request_for_event(bad_event)


def test_symbolic_and_dense_small_system_paths_agree() -> None:
    extraction = extract_df_diagonal_tail("small-reference", (_block(3),))
    symbolic = extraction_to_symbolic_rte_tail(extraction)
    dense = extraction_to_normalized_rte_tail(extraction)
    preparation = prepare_df_rte_event_inputs(extraction)

    assert symbolic.tail_hash == dense.tail_hash == extraction.tail_hash
    assert symbolic.lambda_r == dense.lambda_r
    assert [component.component_id for component in symbolic.components] == [
        component.component_id for component in dense.components
    ]
    for symbolic_component, dense_component, extracted_component in zip(
        symbolic.components,
        dense.components,
        extraction.components,
        strict=True,
    ):
        assert symbolic_component.coefficient_abs == extracted_component.coefficient_abs
        assert symbolic_component.coefficient_sign == dense_component.coefficient_sign
        assert symbolic_component.probability == dense_component.probability
        assert symbolic_component.is_identity == dense_component.is_identity
        assert (
            symbolic_component.diagonal_pauli_support
            == dense_component.diagonal_pauli_support
        )

    _, distribution = make_rte_config(
        symbolic,
        evolution_time=0.1,
        rte_steps=2,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    symbolic_events = preparation.sample_events(
        distribution,
        sample_count=6,
        seed=7,
    )
    dense_events = sample_rte_events(
        dense.components,
        distribution,
        sample_count=6,
        seed=7,
    )

    assert [event.to_dict() for event in symbolic_events] == [
        event.to_dict() for event in dense_events
    ]
    assert [event.unsigned_rotation_angle for event in symbolic_events] == [
        event.unsigned_rotation_angle for event in dense_events
    ]


def test_zero_randomized_tail_is_explicit() -> None:
    block = DFBlock(
        U_ops=((RYGate(0.2), (0,)),),
        eta=np.asarray([1.0, 1.0]),
        lam=1.0,
    )
    identity_only = extract_df_diagonal_tail(
        "identity-only",
        (block,),
        identity_policy="faithful_identity_in_tail",
        coefficient_atol=1.1,
    )
    faithful_symbolic = extraction_to_symbolic_rte_tail(identity_only)
    assert not faithful_symbolic.is_deterministic_only
    assert len(faithful_symbolic.components) == 1
    assert faithful_symbolic.components[0].is_identity

    extracted_identity = extract_df_diagonal_tail(
        "identity-only",
        (block,),
        identity_policy="extract_identity_phase",
        coefficient_atol=1.1,
    )
    deterministic = extraction_to_symbolic_rte_tail(extracted_identity)
    preparation = prepare_df_rte_event_inputs(extracted_identity)
    assert deterministic.is_deterministic_only
    assert deterministic.components == ()
    assert deterministic.deterministic_identity_coefficient == 1.5
    assert preparation.component_specs == ()

    with pytest.raises(DeterministicOnlyRTETailError):
        make_rte_config(
            deterministic,
            evolution_time=0.1,
            rte_steps=1,
            truncation_tolerance=1e-6,
        )
    with pytest.raises(DeterministicOnlyRTETailError):
        preparation.sample_events(
            finite_rte_distribution(0.1, 2),
            sample_count=1,
            seed=0,
        )
    with pytest.raises(DeterministicOnlyRTETailError):
        extraction_to_normalized_rte_tail(extracted_identity)

    threshold_empty = extract_df_diagonal_tail(
        "threshold-empty",
        (block,),
        identity_policy="faithful_identity_in_tail",
        coefficient_atol=1.6,
    )
    threshold_symbolic = extraction_to_symbolic_rte_tail(threshold_empty)
    assert threshold_symbolic.is_deterministic_only
    assert threshold_symbolic.extraction_metadata.threshold_dropped_component_count == 4


def test_basis_fingerprint_is_canonical_and_rejects_opaque_operations() -> None:
    registry = DFBasisRegistry()
    first = registry.register(
        ((RYGate(0.2), (0,)), (RZGate(0.3), (1,))),
        num_system_qubits=4,
    )
    same = registry.register(
        ((RYGate(0.2), (0,)), (RZGate(0.3), (1,))),
        num_system_qubits=4,
    )
    reordered = registry.register(
        ((RZGate(0.3), (1,)), (RYGate(0.2), (0,))),
        num_system_qubits=4,
    )
    different_parameter = registry.register(
        ((RYGate(0.21), (0,)), (RZGate(0.3), (1,))),
        num_system_qubits=4,
    )
    different_support = registry.register(
        ((RYGate(0.2), (1,)), (RZGate(0.3), (0,))),
        num_system_qubits=4,
    )

    assert first.basis_hash == same.basis_hash
    assert first.basis_hash != reordered.basis_hash
    assert first.basis_hash != different_parameter.basis_hash
    assert first.basis_hash != different_support.basis_hash

    opaque_a = Instruction("opaque", 1, 0, [])
    opaque_b = Instruction("opaque", 1, 0, [])
    for opaque in (opaque_a, opaque_b):
        with pytest.raises(BasisFingerprintError, match="stable local matrix"):
            registry.register(
                ((opaque, (0,)),),
                num_system_qubits=4,
            )

    wide = Instruction("wide-opaque", 5, 0, [])
    with pytest.raises(BasisFingerprintError, match="wider than four"):
        registry.register(
            ((wide, (0, 1, 2, 3, 4)),),
            num_system_qubits=5,
        )

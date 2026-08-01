from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

import trotterlib.df_rte_tail as df_tail
from trotterlib.df_rte_tail import (
    DFBasisRegistry,
    component_dense_operator,
    dense_df_block_hamiltonian,
    dense_extracted_df_tail,
    diagonal_pauli_matrix,
    extract_df_diagonal_tail,
    extraction_to_normalized_rte_tail,
)
from trotterlib.df_trotter.model import DFBlock


def _synthetic_block(num_qubits: int, angle: float = 0.17) -> DFBlock:
    return DFBlock(
        U_ops=((RYGate(angle), (0,)),),
        eta=np.linspace(0.2, 1.1, num_qubits),
        lam=0.37,
    )


@pytest.mark.parametrize("num_qubits", (20, 26))
def test_large_symbolic_extraction_never_calls_many_body_dense_path(
    num_qubits,
    monkeypatch,
) -> None:
    original_operator = df_tail.Operator

    def local_gate_operator_only(value):
        if isinstance(value, QuantumCircuit):
            raise AssertionError("many-body Operator(circuit).data was requested")
        return original_operator(value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a dense DF-tail reference function was called")

    monkeypatch.setattr(df_tail, "Operator", local_gate_operator_only)
    monkeypatch.setattr(df_tail, "basis_change_unitary", forbidden)
    monkeypatch.setattr(df_tail, "diagonal_pauli_matrix", forbidden)

    block = _synthetic_block(num_qubits)
    first = extract_df_diagonal_tail("large-symbolic", (block,))
    second = extract_df_diagonal_tail("large-symbolic", (block,))
    expected_count = 1 + num_qubits + math.comb(num_qubits, 2)

    assert len(first.components) == expected_count
    assert first.extraction_metadata.randomized_component_count == expected_count
    np.testing.assert_allclose(
        first.rte_lambda_r,
        sum(abs(component.coefficient) for component in first.components),
    )
    assert first.tail_hash == second.tail_hash
    assert [component.component_id for component in first.components] == [
        component.component_id for component in second.components
    ]
    supports = [component.diagonal_pauli_support for component in first.components]
    assert supports == sorted(supports, key=lambda support: (len(support), support))
    assert first.basis_definitions == second.basis_definitions


def test_dense_references_refuse_large_system_before_allocation(monkeypatch) -> None:
    extraction = extract_df_diagonal_tail(
        "guarded",
        (_synthetic_block(20),),
    )

    def allocation_forbidden(*_args, **_kwargs):
        raise AssertionError("dense allocation was reached before the qubit guard")

    monkeypatch.setattr(df_tail, "QuantumCircuit", allocation_forbidden)
    monkeypatch.setattr(df_tail.np, "eye", allocation_forbidden)
    monkeypatch.setattr(df_tail.np, "empty", allocation_forbidden)
    monkeypatch.setattr(df_tail.np, "ones", allocation_forbidden)
    component = extraction.components[0]

    with pytest.raises(ValueError, match="small-system dense reference"):
        extraction.basis_unitary(component.basis_id)
    with pytest.raises(ValueError, match="small-system dense reference"):
        diagonal_pauli_matrix(20, ())
    with pytest.raises(ValueError, match="small-system dense reference"):
        component_dense_operator(extraction, component)
    with pytest.raises(ValueError, match="small-system dense reference"):
        dense_extracted_df_tail(extraction)
    with pytest.raises(ValueError, match="small-system dense reference"):
        dense_df_block_hamiltonian(_synthetic_block(20))
    with pytest.raises(ValueError, match="small-system dense reference"):
        extraction_to_normalized_rte_tail(extraction)


def test_basis_registry_hash_sharing_and_conflict_detection() -> None:
    registry = DFBasisRegistry()
    first = registry.register(
        ((RYGate(0.2), (0,)),),
        num_system_qubits=6,
    )
    same = registry.register(
        ((RYGate(0.2), (0,)),),
        num_system_qubits=6,
    )
    different = registry.register(
        ((RYGate(0.3), (0,)),),
        num_system_qubits=6,
    )

    assert first.basis_hash == same.basis_hash
    assert first.basis_id == same.basis_id
    assert first.basis_hash != different.basis_hash

    shared = extract_df_diagonal_tail(
        "shared-basis",
        (_synthetic_block(6), _synthetic_block(6)),
        fragment_ids=("fragment-a", "fragment-b"),
    )
    assert len({component.basis_id for component in shared.components}) == 1
    assert len({component.basis_hash for component in shared.components}) == 1
    assert any(
        component.component_id.startswith("fragment-a:")
        for component in shared.components
    )
    assert any(
        component.component_id.startswith("fragment-b:")
        for component in shared.components
    )

    explicit = DFBasisRegistry()
    explicit.register(
        ((RYGate(0.2), (0,)),),
        num_system_qubits=6,
        basis_id="shared-id",
    )
    with pytest.raises(ValueError, match="different operation sequence"):
        explicit.register(
            ((RYGate(0.3), (0,)),),
            num_system_qubits=6,
            basis_id="shared-id",
        )


def test_identity_and_threshold_metadata_are_separate_and_consistent() -> None:
    block = DFBlock(
        U_ops=((RYGate(0.11), (0,)),),
        eta=np.asarray([1.0, 1.0]),
        lam=1.0,
    )
    faithful = extract_df_diagonal_tail(
        "metadata",
        (block,),
        identity_policy="faithful_identity_in_tail",
        coefficient_atol=0.75,
    )
    extracted = extract_df_diagonal_tail(
        "metadata",
        (block,),
        identity_policy="extract_identity_phase",
        coefficient_atol=0.75,
    )
    different_threshold = extract_df_diagonal_tail(
        "metadata",
        (block,),
        identity_policy="extract_identity_phase",
        coefficient_atol=0.25,
    )
    different_basis = extract_df_diagonal_tail(
        "metadata",
        (
            DFBlock(
                U_ops=((RYGate(0.12), (0,)),),
                eta=block.eta,
                lam=block.lam,
            ),
        ),
        identity_policy="extract_identity_phase",
        coefficient_atol=0.75,
    )

    faithful_meta = faithful.extraction_metadata
    extracted_meta = extracted.extraction_metadata
    assert faithful_meta.threshold_input_component_count == 4
    assert faithful_meta.threshold_retained_component_count == 3
    assert faithful_meta.threshold_dropped_component_count == 1
    assert faithful_meta.retained_identity_component_count == 1
    assert faithful_meta.extracted_identity_component_count == 0
    assert faithful_meta.randomized_component_count == 3
    assert extracted_meta.extracted_identity_component_count == 1
    assert extracted_meta.randomized_component_count == 2
    assert extracted_meta.threshold_dropped_component_count == 1
    assert extracted_meta.threshold_dropped_coefficient_l1 == 0.5
    assert extracted_meta.threshold_operator_error_bound == 0.5
    assert extracted_meta.extracted_identity_coefficient == 1.5
    assert len(extracted.components) == extracted_meta.randomized_component_count
    np.testing.assert_allclose(
        extracted.rte_lambda_r,
        sum(abs(component.coefficient) for component in extracted.components),
    )
    assert extracted.normalization_metadata.input_component_count == len(
        extracted.components
    )
    assert extracted.normalization_metadata.retained_component_count == len(
        extracted.components
    )
    assert extracted.normalization_metadata.dropped_component_count == 0
    normalized = extraction_to_normalized_rte_tail(extracted)
    assert normalized.normalization_metadata == extracted.normalization_metadata
    assert faithful.tail_hash != extracted.tail_hash
    assert extracted.tail_hash != different_threshold.tail_hash
    assert extracted.tail_hash != different_basis.tail_hash

from __future__ import annotations

import numpy as np
from openfermion.chem import MolecularData as OpenFermionMolecularData
from qiskit.circuit.library import RYGate

from trotterlib.df_hamiltonian import (
    build_df_h_d_from_molecule,
    clear_df_integral_session_cache,
)
from trotterlib.df_partial_randomized_pf import (
    df_hamiltonian_to_model,
    split_df_hamiltonian_by_ld,
)
from trotterlib.df_rte_circuit import (
    DFRTEEventCircuitRequest,
    DFRTEIdentityCircuitSpec,
)
from trotterlib.df_rte_tail import (
    controlled_identity_evolution_operator,
    dense_df_block_hamiltonian,
    dense_extracted_df_tail,
    exact_df_diagonal_coefficients,
    extract_df_diagonal_tail,
    extraction_component_circuit_specs,
    extraction_to_normalized_rte_tail,
    uncontrolled_identity_evolution_operator,
)
from trotterlib.df_trotter.model import DFBlock
from trotterlib.df_trotter.ops import build_df_blocks
from trotterlib.rte import enumerate_rte_events, finite_rte_distribution


def test_exact_diagonal_i_z_zz_expansion_and_identity_policies() -> None:
    first = DFBlock(
        U_ops=((RYGate(0.31), (0,)),),
        eta=np.asarray([0.8, -0.3]),
        lam=-0.4,
    )
    second = DFBlock(
        U_ops=((RYGate(-0.27), (1,)),),
        eta=np.asarray([0.2, 0.7]),
        lam=0.25,
    )
    faithful = extract_df_diagonal_tail(
        "two-fragment-tail",
        (first, second),
        fragment_ids=("df-a", "df-b"),
        basis_ids=("basis-a", "basis-b"),
        identity_policy="faithful_identity_in_tail",
    )
    extracted = extract_df_diagonal_tail(
        "two-fragment-tail",
        (first, second),
        fragment_ids=("df-a", "df-b"),
        basis_ids=("basis-a", "basis-b"),
        identity_policy="extract_identity_phase",
    )
    exact = dense_df_block_hamiltonian(first) + dense_df_block_hamiltonian(second)

    np.testing.assert_allclose(dense_extracted_df_tail(faithful), exact, atol=1e-13)
    np.testing.assert_allclose(dense_extracted_df_tail(extracted), exact, atol=1e-13)
    assert sum(component.is_identity for component in faithful.components) == 2
    assert not any(component.is_identity for component in extracted.components)
    assert extracted.deterministic_identity_coefficient == faithful.identity_coefficient
    assert faithful.rte_lambda_r > extracted.rte_lambda_r
    # Equal Z0 supports from distinct fragment bases are never aggregated.
    z0_components = [
        component
        for component in faithful.components
        if component.diagonal_pauli_support == (0,)
    ]
    assert {component.basis_id for component in z0_components} == {"basis-a", "basis-b"}
    assert [component.component_id for component in faithful.components] == sorted(
        (component.component_id for component in faithful.components),
        key=lambda identifier: (
            identifier.split(":", maxsplit=1)[0],
            len(identifier.split(":", maxsplit=1)[1]),
            identifier,
        ),
    )
    normalized = extraction_to_normalized_rte_tail(extracted)
    identity = np.eye(exact.shape[0], dtype=np.complex128)
    np.testing.assert_allclose(
        normalized.dense_hamiltonian
        + extracted.deterministic_identity_coefficient * identity,
        exact,
        atol=1e-13,
    )
    assert normalized.tail_hash == extracted.tail_hash
    faithful_tail = extraction_to_normalized_rte_tail(faithful)
    event = enumerate_rte_events(
        faithful_tail.components,
        finite_rte_distribution(0.1, 0),
    )[0]
    specs = extraction_component_circuit_specs(faithful)
    request = DFRTEEventCircuitRequest(
        event=event,
        component_specs=specs,
        controlled=True,
        ancilla_qubit=2,
    )
    assert request.event.application_sequence[-1].component_id == event.rotation_component_id
    assert request.event.application_sequence[-1].basis_hash is not None
    assert any(isinstance(spec, DFRTEIdentityCircuitSpec) for spec in specs)


def test_direct_number_basis_coefficients_reconstruct_fragment() -> None:
    eta = np.asarray([0.4, -0.6, 0.2])
    lam = -0.35
    coefficients = dict(exact_df_diagonal_coefficients(eta, lam))
    for basis_state in range(1 << len(eta)):
        reconstructed = 0.0
        for support, coefficient in coefficients.items():
            eigenvalue = (-1) ** sum(
                (basis_state >> qubit) & 1 for qubit in support
            )
            reconstructed += coefficient * eigenvalue
        occupation_sum = sum(
            eta[qubit] * ((basis_state >> qubit) & 1)
            for qubit in range(len(eta))
        )
        np.testing.assert_allclose(reconstructed, lam * occupation_sum**2, atol=1e-14)


def test_identity_global_phase_becomes_controlled_relative_phase() -> None:
    coefficient = -0.37
    evolution_time = 0.23
    system_dimension = 4
    uncontrolled = uncontrolled_identity_evolution_operator(
        coefficient, evolution_time, system_dimension
    )
    controlled = controlled_identity_evolution_operator(
        coefficient, evolution_time, system_dimension
    )
    phase = np.exp(-1j * evolution_time * coefficient)

    np.testing.assert_allclose(uncontrolled, phase * np.eye(system_dimension))
    np.testing.assert_allclose(
        controlled[:system_dimension, :system_dimension], np.eye(system_dimension)
    )
    np.testing.assert_allclose(
        controlled[system_dimension:, system_dimension:],
        phase * np.eye(system_dimension),
    )
    assert not np.allclose(controlled, phase * np.eye(2 * system_dimension))


def test_h2_df_tail_reconstructs_selected_conjugated_fragments(
    tmp_path,
    monkeypatch,
    request,
) -> None:
    clear_df_integral_session_cache()
    request.addfinalizer(clear_df_integral_session_cache)

    def temporary_molecular_data(*args, **kwargs):
        kwargs["data_directory"] = str(tmp_path)
        return OpenFermionMolecularData(*args, **kwargs)

    monkeypatch.setattr(
        "trotterlib.df_hamiltonian.MolecularData",
        temporary_molecular_data,
    )
    hamiltonian, _sector = build_df_h_d_from_molecule(2)
    partition = split_df_hamiltonian_by_ld(hamiltonian, 1)
    selected = hamiltonian.select_blocks(partition.randomized_block_indices)
    blocks = build_df_blocks(df_hamiltonian_to_model(selected))
    faithful = extract_df_diagonal_tail(
        "H2-selected-DF-tail",
        blocks,
        fragment_ids=tuple(
            f"df-fragment-{index}" for index in partition.randomized_block_indices
        ),
        identity_policy="faithful_identity_in_tail",
        ranking_proxy_lambda_r=partition.lambda_r,
    )
    extracted = extract_df_diagonal_tail(
        "H2-selected-DF-tail",
        blocks,
        fragment_ids=tuple(
            f"df-fragment-{index}" for index in partition.randomized_block_indices
        ),
        identity_policy="extract_identity_phase",
        ranking_proxy_lambda_r=partition.lambda_r,
    )
    exact = sum(
        (dense_df_block_hamiltonian(block) for block in blocks),
        np.zeros((1 << hamiltonian.n_qubits,) * 2, dtype=np.complex128),
    )

    np.testing.assert_allclose(dense_extracted_df_tail(faithful), exact, atol=2e-12)
    np.testing.assert_allclose(dense_extracted_df_tail(extracted), exact, atol=2e-12)
    assert faithful.ranking_proxy_lambda_r == partition.lambda_r
    assert not np.isclose(faithful.rte_lambda_r, partition.lambda_r)
    assert faithful.tail_hash != extracted.tail_hash
    assert faithful.normalization_metadata.coefficient_atol == 0.0

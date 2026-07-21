import numpy as np
from openfermion import InteractionOperator, get_sparse_operator
from openfermion.chem import MolecularData as OpenFermionMolecularData

from trotterlib.df_hamiltonian import (
    DFHamiltonian,
    PhysicalSector,
    _NUMBA_AVAILABLE,
    _h_chain_integrals_session_cached,
    build_df_h_d_from_molecule,
    clear_df_integral_session_cache,
    df_hamiltonian_from_integrals,
    df_linear_operator,
    expand_sector_state,
    solve_df_ground_state,
)
from trotterlib.config import (
    get_df_rank_selection_for_molecule,
    resolve_df_rank_for_molecule,
)


def test_one_body_number_operator_in_number_sector():
    hamiltonian = DFHamiltonian(
        constant=0.0,
        one_body=np.diag([1.0, 2.0]).astype(np.complex128),
        lambdas=np.zeros(0, dtype=float),
        g_matrices=tuple(),
        metadata={},
    )
    sector = PhysicalSector.number_sector(n_qubits=2, n_electrons=1)
    linop, _counter = df_linear_operator(hamiltonian, sector)

    dense = np.column_stack(
        [linop @ np.eye(sector.dimension, dtype=np.complex128)[:, i] for i in range(sector.dimension)]
    )

    assert np.allclose(np.sort(np.linalg.eigvalsh(dense)), [1.0, 2.0])


def test_df_square_block_is_applied_matrix_free():
    g = np.diag([1.0, -1.0]).astype(np.complex128)
    hamiltonian = DFHamiltonian(
        constant=0.25,
        one_body=np.zeros((2, 2), dtype=np.complex128),
        lambdas=np.array([0.5]),
        g_matrices=(g,),
        metadata={},
    )
    sector = PhysicalSector.number_sector(n_qubits=2, n_electrons=1)
    linop, _counter = df_linear_operator(hamiltonian, sector)
    dense = np.column_stack(
        [linop @ np.eye(sector.dimension, dtype=np.complex128)[:, i] for i in range(sector.dimension)]
    )

    assert np.allclose(dense, 0.75 * np.eye(sector.dimension))


def test_df_truncation_value_is_not_added_to_hamiltonian_constant(monkeypatch):
    truncation_value = 0.125

    def fake_low_rank_two_body_decomposition(_two_body, **_kwargs):
        return (
            np.asarray([0.5]),
            np.asarray([np.diag([1.0, -1.0])], dtype=np.complex128),
            np.zeros((2, 2), dtype=np.complex128),
            truncation_value,
        )

    monkeypatch.setattr(
        "trotterlib.df_hamiltonian.low_rank_two_body_decomposition",
        fake_low_rank_two_body_decomposition,
    )

    hamiltonian = df_hamiltonian_from_integrals(
        constant=0.75,
        one_body=np.zeros((2, 2), dtype=np.complex128),
        two_body=np.zeros((2, 2, 2, 2), dtype=np.complex128),
        df_rank=1,
    )

    assert hamiltonian.constant == 0.75
    assert hamiltonian.metadata["df_truncation_value"] == truncation_value


def test_h3_selected_rank_ground_state_matches_interaction_operator(
    tmp_path,
    monkeypatch,
    request,
):
    clear_df_integral_session_cache()
    request.addfinalizer(clear_df_integral_session_cache)

    def temporary_molecular_data(*args, **kwargs):
        kwargs["data_directory"] = str(tmp_path)
        return OpenFermionMolecularData(*args, **kwargs)

    monkeypatch.setattr(
        "trotterlib.df_hamiltonian.MolecularData",
        temporary_molecular_data,
    )
    integrals = _h_chain_integrals_session_cached(3, distance=1.0, basis="sto-3g")
    hamiltonian, sector = build_df_h_d_from_molecule(3)

    result = solve_df_ground_state(
        hamiltonian,
        sector,
        matrix_free_backend="python",
        tol=1e-12,
    )
    exact_operator = InteractionOperator(
        integrals["constant"],
        integrals["one_body"],
        integrals["two_body"],
    )
    exact_sparse = get_sparse_operator(exact_operator, n_qubits=hamiltonian.n_qubits)
    exact_sector = exact_sparse[sector.basis_indices, :][:, sector.basis_indices].toarray()
    exact_energies, exact_states = np.linalg.eigh(exact_sector)
    exact_ground_state = exact_states[:, 0]
    overlap = abs(np.vdot(exact_ground_state, result.sector_state_vector)) ** 2

    assert hamiltonian.metadata["df_rank_actual"] == 5
    assert hamiltonian.metadata["df_truncation_value"] > 1e-4
    assert np.isclose(hamiltonian.constant, integrals["constant"], atol=1e-12)
    assert result.residual_norm < 1e-9
    assert abs(result.energy - exact_energies[0]) < 1e-5
    assert overlap > 1.0 - 1e-8


def test_solve_df_ground_state_returns_residual_and_full_state():
    hamiltonian = DFHamiltonian(
        constant=0.0,
        one_body=np.diag([3.0, 1.0]).astype(np.complex128),
        lambdas=np.zeros(0, dtype=float),
        g_matrices=tuple(),
        metadata={},
    )
    sector = PhysicalSector.number_sector(n_qubits=2, n_electrons=1)

    result = solve_df_ground_state(hamiltonian, sector, tol=1e-12)

    assert result.converged
    assert np.isclose(result.energy, 1.0)
    assert result.residual_norm < 1e-8
    assert result.state_vector.shape == (4,)
    assert np.isclose(np.linalg.norm(result.state_vector), 1.0)

    reconstructed = expand_sector_state(result.sector_state_vector, sector)
    assert np.allclose(result.state_vector, reconstructed)


def test_numba_backend_matches_python_backend_when_available():
    if not _NUMBA_AVAILABLE:
        return
    hamiltonian = DFHamiltonian(
        constant=0.25,
        one_body=np.array([[1.0, 0.2], [0.2, 2.0]], dtype=np.complex128),
        lambdas=np.array([0.5]),
        g_matrices=(np.array([[0.7, 0.1], [0.1, -0.3]], dtype=np.complex128),),
        metadata={},
    )
    sector = PhysicalSector.number_sector(n_qubits=2, n_electrons=1)
    vec = np.array([0.6, -0.8], dtype=np.complex128)
    py_op, _ = df_linear_operator(hamiltonian, sector, backend="python")
    nb_op, _ = df_linear_operator(
        hamiltonian,
        sector,
        backend="numba",
        num_threads=2,
    )

    assert np.allclose(py_op @ vec, nb_op @ vec)


def test_configured_df_rank_defaults_match_reference_project():
    selection = get_df_rank_selection_for_molecule(10)

    assert selection is not None
    assert selection["selected_rank"] == 25
    assert selection["full_rank"] == 100
    assert resolve_df_rank_for_molecule(10) == 25
    assert resolve_df_rank_for_molecule(10, df_rank=7) == 7

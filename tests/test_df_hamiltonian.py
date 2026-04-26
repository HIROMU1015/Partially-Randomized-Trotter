import numpy as np

from trotterlib.df_hamiltonian import (
    DFHamiltonian,
    PhysicalSector,
    _NUMBA_AVAILABLE,
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

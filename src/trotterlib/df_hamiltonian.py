from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from openfermion import low_rank_two_body_decomposition
from openfermion.chem import MolecularData
from openfermion.linalg import jw_number_indices, jw_sz_indices
from openfermionpyscf import run_pyscf
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh, lobpcg

from .chemistry_hamiltonian import _NUMBA_AVAILABLE, _resolve_numba_threads, geo
from .config import (
    DEFAULT_BASIS,
    DEFAULT_DISTANCE,
    get_df_rank_selection_for_molecule,
    resolve_df_rank_for_molecule,
)

_FUSED_NUMBA_WORKSPACE_LIMIT_BYTES = 2 * 1024**3
_FUSED_NUMBA_MIN_BLOCK_DIM_WORK = 10_000_000
_FUSED_NUMBA_AUTO_BLOCK_CHUNK_SIZE = 12

if _NUMBA_AVAILABLE:
    from numba import njit, prange
else:  # pragma: no cover - exercised only without numba installed
    njit = None
    prange = range


if _NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _apply_one_body_csr_numba_into(
        coeff_matrix: np.ndarray,
        state: np.ndarray,
        target_indptr: np.ndarray,
        source_positions: np.ndarray,
        p_indices: np.ndarray,
        q_indices: np.ndarray,
        signs: np.ndarray,
        out: np.ndarray,
    ) -> np.ndarray:
        for target_pos in prange(out.shape[0]):
            acc = 0.0 + 0.0j
            for entry in range(target_indptr[target_pos], target_indptr[target_pos + 1]):
                acc += (
                    coeff_matrix[p_indices[entry], q_indices[entry]]
                    * signs[entry]
                    * state[source_positions[entry]]
                )
            out[target_pos] = acc
        return out

    @njit(cache=True)
    def _df_matvec_csr_numba_streaming(
        state: np.ndarray,
        constant: float,
        one_body: np.ndarray,
        lambdas: np.ndarray,
        g_matrices: np.ndarray,
        target_indptr: np.ndarray,
        source_positions: np.ndarray,
        p_indices: np.ndarray,
        q_indices: np.ndarray,
        signs: np.ndarray,
    ) -> np.ndarray:
        out = np.empty_like(state)
        tmp = np.empty_like(state)
        tmp2 = np.empty_like(state)
        _apply_one_body_csr_numba_into(
            one_body,
            state,
            target_indptr,
            source_positions,
            p_indices,
            q_indices,
            signs,
            tmp,
        )
        out[:] = constant * state + tmp
        for block_index in range(lambdas.shape[0]):
            _apply_one_body_csr_numba_into(
                g_matrices[block_index],
                state,
                target_indptr,
                source_positions,
                p_indices,
                q_indices,
                signs,
                tmp,
            )
            _apply_one_body_csr_numba_into(
                g_matrices[block_index],
                tmp,
                target_indptr,
                source_positions,
                p_indices,
                q_indices,
                signs,
                tmp2,
            )
            out += lambdas[block_index] * tmp2
        return out

    @njit(cache=True, parallel=True)
    def _df_first_pass_csr_numba_into(
        state: np.ndarray,
        one_body: np.ndarray,
        g_matrices: np.ndarray,
        target_indptr: np.ndarray,
        source_positions: np.ndarray,
        p_indices: np.ndarray,
        q_indices: np.ndarray,
        signs: np.ndarray,
        one_body_out: np.ndarray,
        block_out: np.ndarray,
    ) -> None:
        for target_pos in prange(one_body_out.shape[0]):
            one_acc = 0.0 + 0.0j
            for entry in range(target_indptr[target_pos], target_indptr[target_pos + 1]):
                one_acc += (
                    one_body[p_indices[entry], q_indices[entry]]
                    * signs[entry]
                    * state[source_positions[entry]]
                )
            one_body_out[target_pos] = one_acc

            for block_index in range(g_matrices.shape[0]):
                block_acc = 0.0 + 0.0j
                for entry in range(target_indptr[target_pos], target_indptr[target_pos + 1]):
                    block_acc += (
                        g_matrices[block_index, p_indices[entry], q_indices[entry]]
                        * signs[entry]
                        * state[source_positions[entry]]
                    )
                block_out[block_index, target_pos] = block_acc

    @njit(cache=True, parallel=True)
    def _df_second_pass_csr_numba_into(
        state: np.ndarray,
        constant: float,
        lambdas: np.ndarray,
        g_matrices: np.ndarray,
        target_indptr: np.ndarray,
        source_positions: np.ndarray,
        p_indices: np.ndarray,
        q_indices: np.ndarray,
        signs: np.ndarray,
        one_body_out: np.ndarray,
        block_first_pass: np.ndarray,
        out: np.ndarray,
    ) -> None:
        for target_pos in prange(out.shape[0]):
            acc = constant * state[target_pos] + one_body_out[target_pos]
            for block_index in range(g_matrices.shape[0]):
                block_acc = 0.0 + 0.0j
                for entry in range(target_indptr[target_pos], target_indptr[target_pos + 1]):
                    block_acc += (
                        g_matrices[block_index, p_indices[entry], q_indices[entry]]
                        * signs[entry]
                        * block_first_pass[block_index, source_positions[entry]]
                    )
                acc += lambdas[block_index] * block_acc
            out[target_pos] = acc

    @njit(cache=True)
    def _df_matvec_csr_numba_fused(
        state: np.ndarray,
        constant: float,
        one_body: np.ndarray,
        lambdas: np.ndarray,
        g_matrices: np.ndarray,
        target_indptr: np.ndarray,
        source_positions: np.ndarray,
        p_indices: np.ndarray,
        q_indices: np.ndarray,
        signs: np.ndarray,
        one_body_workspace: np.ndarray,
        block_workspace: np.ndarray,
        out_workspace: np.ndarray,
    ) -> np.ndarray:
        _df_first_pass_csr_numba_into(
            state,
            one_body,
            g_matrices,
            target_indptr,
            source_positions,
            p_indices,
            q_indices,
            signs,
            one_body_workspace,
            block_workspace,
        )
        _df_second_pass_csr_numba_into(
            state,
            constant,
            lambdas,
            g_matrices,
            target_indptr,
            source_positions,
            p_indices,
            q_indices,
            signs,
            one_body_workspace,
            block_workspace,
            out_workspace,
        )
        return out_workspace.copy()

    @njit(cache=True, parallel=True)
    def _df_chunk_first_pass_csr_numba_into(
        state: np.ndarray,
        g_matrices: np.ndarray,
        block_start: int,
        block_stop: int,
        target_indptr: np.ndarray,
        source_positions: np.ndarray,
        p_indices: np.ndarray,
        q_indices: np.ndarray,
        signs: np.ndarray,
        block_out: np.ndarray,
    ) -> None:
        for target_pos in prange(block_out.shape[1]):
            for local_block in range(block_stop - block_start):
                block_index = block_start + local_block
                block_acc = 0.0 + 0.0j
                for entry in range(target_indptr[target_pos], target_indptr[target_pos + 1]):
                    block_acc += (
                        g_matrices[block_index, p_indices[entry], q_indices[entry]]
                        * signs[entry]
                        * state[source_positions[entry]]
                    )
                block_out[local_block, target_pos] = block_acc

    @njit(cache=True, parallel=True)
    def _df_chunk_second_pass_csr_numba_accumulate(
        lambdas: np.ndarray,
        g_matrices: np.ndarray,
        block_start: int,
        block_stop: int,
        target_indptr: np.ndarray,
        source_positions: np.ndarray,
        p_indices: np.ndarray,
        q_indices: np.ndarray,
        signs: np.ndarray,
        block_first_pass: np.ndarray,
        out: np.ndarray,
    ) -> None:
        for target_pos in prange(out.shape[0]):
            acc = 0.0 + 0.0j
            for local_block in range(block_stop - block_start):
                block_index = block_start + local_block
                block_acc = 0.0 + 0.0j
                for entry in range(target_indptr[target_pos], target_indptr[target_pos + 1]):
                    block_acc += (
                        g_matrices[block_index, p_indices[entry], q_indices[entry]]
                        * signs[entry]
                        * block_first_pass[local_block, source_positions[entry]]
                    )
                acc += lambdas[block_index] * block_acc
            out[target_pos] += acc

    @njit(cache=True)
    def _df_matvec_csr_numba_chunked(
        state: np.ndarray,
        constant: float,
        one_body: np.ndarray,
        lambdas: np.ndarray,
        g_matrices: np.ndarray,
        target_indptr: np.ndarray,
        source_positions: np.ndarray,
        p_indices: np.ndarray,
        q_indices: np.ndarray,
        signs: np.ndarray,
        block_chunk_size: int,
        one_body_workspace: np.ndarray,
        block_workspace: np.ndarray,
        out_workspace: np.ndarray,
    ) -> np.ndarray:
        _apply_one_body_csr_numba_into(
            one_body,
            state,
            target_indptr,
            source_positions,
            p_indices,
            q_indices,
            signs,
            one_body_workspace,
        )
        out_workspace[:] = constant * state + one_body_workspace
        for block_start in range(0, lambdas.shape[0], block_chunk_size):
            block_stop = min(lambdas.shape[0], block_start + block_chunk_size)
            _df_chunk_first_pass_csr_numba_into(
                state,
                g_matrices,
                block_start,
                block_stop,
                target_indptr,
                source_positions,
                p_indices,
                q_indices,
                signs,
                block_workspace,
            )
            _df_chunk_second_pass_csr_numba_accumulate(
                lambdas,
                g_matrices,
                block_start,
                block_stop,
                target_indptr,
                source_positions,
                p_indices,
                q_indices,
                signs,
                block_workspace,
                out_workspace,
            )
        return out_workspace.copy()


@dataclass(frozen=True)
class DFHamiltonian:
    """
    Double-factorized Hamiltonian in spin-orbital form.

    The represented operator is
        constant
        + sum_pq one_body[p, q] a_p^dagger a_q
        + sum_l lambdas[l] A_l^2,
    where A_l = sum_pq g_matrices[l][p, q] a_p^dagger a_q.
    """

    constant: float
    one_body: np.ndarray
    lambdas: np.ndarray
    g_matrices: tuple[np.ndarray, ...]
    metadata: dict[str, Any]

    @property
    def n_qubits(self) -> int:
        return int(self.one_body.shape[0])

    @property
    def n_blocks(self) -> int:
        return int(len(self.g_matrices))

    def select_blocks(self, block_indices: Sequence[int]) -> "DFHamiltonian":
        indices = tuple(int(idx) for idx in block_indices)
        if any(idx < 0 or idx >= self.n_blocks for idx in indices):
            raise ValueError("block_indices contains an out-of-range DF block index.")
        return DFHamiltonian(
            constant=self.constant,
            one_body=self.one_body,
            lambdas=np.asarray([self.lambdas[idx] for idx in indices], dtype=np.float64),
            g_matrices=tuple(self.g_matrices[idx] for idx in indices),
            metadata={**self.metadata, "selected_block_indices": indices},
        )

    def hermitized(self, *, tol: float = 1e-10) -> "DFHamiltonian":
        one_body = _as_hermitian(self.one_body, name="one_body", tol=tol)
        g_matrices = tuple(
            _as_hermitian(g_matrix, name=f"g_matrices[{idx}]", tol=tol)
            for idx, g_matrix in enumerate(self.g_matrices)
        )
        lambdas = np.real_if_close(self.lambdas, tol=1000).astype(np.float64)
        return DFHamiltonian(
            constant=float(np.real_if_close(self.constant)),
            one_body=one_body,
            lambdas=lambdas,
            g_matrices=g_matrices,
            metadata={**self.metadata, "hermitized": True},
        )


@dataclass(frozen=True)
class PhysicalSector:
    """Basis restriction used by the matrix-free DF solver."""

    n_qubits: int
    basis_indices: np.ndarray
    n_electrons: int | None = None
    nelec_alpha: int | None = None
    nelec_beta: int | None = None
    sz_value: float | None = None

    @property
    def dimension(self) -> int:
        return int(self.basis_indices.size)

    @classmethod
    def number_sector(cls, *, n_qubits: int, n_electrons: int) -> "PhysicalSector":
        indices = np.asarray(jw_number_indices(int(n_electrons), int(n_qubits)), dtype=np.int64)
        indices.sort()
        return cls(
            n_qubits=int(n_qubits),
            basis_indices=indices,
            n_electrons=int(n_electrons),
        )

    @classmethod
    def spin_sector(
        cls,
        *,
        n_qubits: int,
        nelec_alpha: int,
        nelec_beta: int,
    ) -> "PhysicalSector":
        n_electrons = int(nelec_alpha) + int(nelec_beta)
        sz_value = 0.5 * float(int(nelec_alpha) - int(nelec_beta))
        indices = np.asarray(
            jw_sz_indices(sz_value, int(n_qubits), n_electrons=n_electrons),
            dtype=np.int64,
        )
        indices.sort()
        return cls(
            n_qubits=int(n_qubits),
            basis_indices=indices,
            n_electrons=n_electrons,
            nelec_alpha=int(nelec_alpha),
            nelec_beta=int(nelec_beta),
            sz_value=sz_value,
        )

    @classmethod
    def from_h_chain(
        cls,
        molecule_type: int,
        *,
        charge: int | None = None,
        multiplicity: int | None = None,
    ) -> "PhysicalSector":
        geometry, default_multiplicity, default_charge = geo(int(molecule_type))
        del geometry
        charge = default_charge if charge is None else int(charge)
        multiplicity = default_multiplicity if multiplicity is None else int(multiplicity)
        n_electrons = int(molecule_type) - charge
        spin_diff = multiplicity - 1
        if (n_electrons + spin_diff) % 2 != 0:
            raise ValueError("Incompatible H-chain electron count and multiplicity.")
        nelec_alpha = (n_electrons + spin_diff) // 2
        nelec_beta = n_electrons - nelec_alpha
        return cls.spin_sector(
            n_qubits=2 * int(molecule_type),
            nelec_alpha=nelec_alpha,
            nelec_beta=nelec_beta,
        )


@dataclass(frozen=True)
class DFGroundStateResult:
    energy: float
    state_vector: np.ndarray
    sector_state_vector: np.ndarray
    sector: PhysicalSector
    converged: bool
    residual_norm: float
    matvec_count: int
    elapsed_s: float
    solver: str
    message: str


@dataclass(frozen=True)
class _ExcitationTable:
    target_indptr: np.ndarray
    source_positions: np.ndarray
    target_positions: np.ndarray
    p_indices: np.ndarray
    q_indices: np.ndarray
    signs: np.ndarray


def build_df_h_d_from_molecule(
    molecule_type: int,
    *,
    distance: float | None = None,
    basis: str = DEFAULT_BASIS,
    df_rank: int | None = None,
    df_tol: float | None = None,
    block_indices: Sequence[int] | None = None,
) -> tuple[DFHamiltonian, PhysicalSector]:
    """
    Build a DF-based deterministic H_D for an H-chain molecule.

    `df_rank` and `block_indices` are the current deterministic-subset knobs.
    Future partial-randomized policies can choose a different subset before
    calling the solver without changing the solver itself.
    """
    if distance is None:
        distance = DEFAULT_DISTANCE
    geometry, multiplicity, charge = geo(int(molecule_type), distance)
    molecule = MolecularData(geometry, basis, multiplicity, charge, f"df_d{int(distance * 100)}")
    molecule = run_pyscf(molecule, run_scf=1, run_fci=0)
    interaction = molecule.get_molecular_hamiltonian()
    resolved_df_rank = resolve_df_rank_for_molecule(int(molecule_type), df_rank)
    rank_selection = (
        get_df_rank_selection_for_molecule(int(molecule_type))
        if df_rank is None
        else None
    )
    hamiltonian = df_hamiltonian_from_integrals(
        constant=float(interaction.constant),
        one_body=np.asarray(interaction.one_body_tensor),
        two_body=np.asarray(interaction.two_body_tensor),
        df_rank=resolved_df_rank,
        df_tol=df_tol,
        metadata={
            "molecule_type": int(molecule_type),
            "distance": float(distance),
            "basis": basis,
            "multiplicity": int(multiplicity),
            "charge": int(charge),
            "hf_energy": float(molecule.hf_energy),
            "df_rank_source": (
                "config"
                if df_rank is None and resolved_df_rank is not None
                else "explicit"
                if df_rank is not None
                else "openfermion_default"
            ),
            "df_rank_config_selection": rank_selection,
        },
    )
    if block_indices is not None:
        hamiltonian = hamiltonian.select_blocks(block_indices)
    sector = PhysicalSector.from_h_chain(
        int(molecule_type),
        charge=int(charge),
        multiplicity=int(multiplicity),
    )
    return hamiltonian, sector


def df_hamiltonian_from_integrals(
    *,
    constant: float,
    one_body: np.ndarray,
    two_body: np.ndarray,
    df_rank: int | None = None,
    df_tol: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> DFHamiltonian:
    """Create a DFHamiltonian from spin-orbital InteractionOperator tensors."""
    one_body = np.asarray(one_body, dtype=np.complex128)
    two_body = np.asarray(two_body, dtype=np.complex128)
    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("one_body must be a square spin-orbital tensor.")
    n_qubits = int(one_body.shape[0])
    if two_body.shape != (n_qubits, n_qubits, n_qubits, n_qubits):
        raise ValueError("two_body must have shape (N, N, N, N).")

    kwargs = _low_rank_kwargs(df_rank=df_rank, df_tol=df_tol)
    eigenvalues, one_body_squares, one_body_correction, constant_correction = (
        low_rank_two_body_decomposition(two_body, **kwargs)
    )
    g_stack = np.asarray(one_body_squares, dtype=np.complex128)
    if g_stack.ndim != 3 or g_stack.shape[1:] != (n_qubits, n_qubits):
        raise ValueError("low_rank_two_body_decomposition returned invalid G matrices.")

    hamiltonian = DFHamiltonian(
        constant=float(np.real_if_close(constant + constant_correction)),
        one_body=one_body + np.asarray(one_body_correction, dtype=np.complex128),
        lambdas=np.asarray(eigenvalues, dtype=np.float64).reshape(-1),
        g_matrices=tuple(g_stack[idx] for idx in range(g_stack.shape[0])),
        metadata={
            **(metadata or {}),
            "df_rank_requested": df_rank,
            "df_tol_requested": df_tol,
            "df_rank_actual": int(g_stack.shape[0]),
        },
    )
    return hamiltonian.hermitized()


def solve_df_ground_state(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    matrix_free_backend: str = "auto",
    matrix_free_threads: int | None = None,
    matrix_free_block_chunk_size: int | None = None,
    solver: str = "eigsh",
    use_preconditioner: bool = True,
    lobpcg_block_size: int = 4,
    tol: float = 1e-10,
    maxiter: int | None = None,
    ncv: int | None = None,
    v0: np.ndarray | None = None,
    expand_state: bool = True,
) -> DFGroundStateResult:
    """Solve the DF H_D ground state in a fixed physical sector."""
    if solver not in ("eigsh", "lobpcg"):
        raise ValueError("solver must be 'eigsh' or 'lobpcg'.")
    if lobpcg_block_size <= 0:
        raise ValueError("lobpcg_block_size must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if hamiltonian.n_qubits != sector.n_qubits:
        raise ValueError("Hamiltonian and sector n_qubits mismatch.")
    if sector.dimension == 0:
        raise ValueError("Physical sector is empty.")

    linop, matvec_counter = df_linear_operator(
        hamiltonian,
        sector,
        backend=matrix_free_backend,
        num_threads=matrix_free_threads,
        block_chunk_size=matrix_free_block_chunk_size,
    )
    initial = _normalize_initial_vector(v0, sector)
    if initial is None:
        initial = _hartree_fock_initial_vector(sector)

    started = time.perf_counter()
    converged = True
    message = "converged"
    if sector.dimension <= 3:
        dense = _linear_operator_to_dense(linop)
        evals, evecs = np.linalg.eigh(dense)
        idx = int(np.argmin(evals.real))
        energy = float(np.real_if_close(evals[idx]))
        sector_state = np.asarray(evecs[:, idx], dtype=np.complex128)
    elif solver == "eigsh":
        try:
            vals, vecs = eigsh(
                linop,
                k=1,
                which="SA",
                tol=float(tol),
                maxiter=maxiter,
                ncv=ncv,
                v0=initial,
                return_eigenvectors=True,
            )
            idx = int(np.argmin(vals.real))
            energy = float(np.real_if_close(vals[idx]))
            sector_state = np.asarray(vecs[:, idx], dtype=np.complex128)
        except ArpackNoConvergence as exc:
            converged = False
            message = str(exc)
            if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
                raise
            idx = int(np.argmin(exc.eigenvalues.real))
            energy = float(np.real_if_close(exc.eigenvalues[idx]))
            sector_state = np.asarray(exc.eigenvectors[:, idx], dtype=np.complex128)
    else:
        block_size = min(int(lobpcg_block_size), max(1, sector.dimension - 1))
        x0 = _lobpcg_initial_block(initial, sector.dimension, block_size)
        preconditioner = (
            df_diagonal_preconditioner(hamiltonian, sector)
            if use_preconditioner
            else None
        )
        vals, vecs, residual_history = lobpcg(
            linop,
            x0,
            M=preconditioner,
            largest=False,
            tol=float(tol),
            maxiter=maxiter,
            retResidualNormsHistory=True,
        )
        idx = int(np.argmin(vals.real))
        energy = float(np.real_if_close(vals[idx]))
        sector_state = np.asarray(vecs[:, idx], dtype=np.complex128)
        final_residual = (
            float(np.ravel(residual_history[-1])[idx])
            if len(residual_history) > 0
            else float("nan")
        )
        message = f"lobpcg final residual estimate={final_residual:.3e}"

    elapsed_s = time.perf_counter() - started
    norm = np.linalg.norm(sector_state)
    if norm == 0:
        raise RuntimeError("Ground-state solver returned a zero vector.")
    sector_state = sector_state / norm
    residual = linop @ sector_state - energy * sector_state
    residual_norm = float(np.linalg.norm(residual))
    full_state = (
        expand_sector_state(sector_state, sector)
        if expand_state
        else np.empty(0, dtype=np.complex128)
    )
    return DFGroundStateResult(
        energy=energy,
        state_vector=full_state,
        sector_state_vector=sector_state,
        sector=sector,
        converged=converged and residual_norm <= max(100.0 * tol, 1e-8),
        residual_norm=residual_norm,
        matvec_count=matvec_counter["count"],
        elapsed_s=elapsed_s,
        solver=solver,
        message=message,
    )


def df_linear_operator(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    backend: str = "auto",
    num_threads: int | None = None,
    block_chunk_size: int | None = None,
) -> tuple[LinearOperator, dict[str, int]]:
    if backend not in ("auto", "numba", "python"):
        raise ValueError("backend must be 'auto', 'numba', or 'python'.")
    if backend == "numba" and not _NUMBA_AVAILABLE:
        raise RuntimeError("backend='numba' requires numba to be installed.")
    if num_threads is not None and num_threads < 0:
        raise ValueError("num_threads must be non-negative.")
    if backend in ("auto", "numba") and _NUMBA_AVAILABLE:
        _resolve_numba_threads(num_threads)
    elif num_threads is not None:
        raise RuntimeError("num_threads requires the numba backend.")
    if block_chunk_size is not None and block_chunk_size <= 0:
        raise ValueError("block_chunk_size must be positive when provided.")

    table = _build_excitation_table(sector)
    counter = {"count": 0}
    use_numba = _NUMBA_AVAILABLE and backend in ("auto", "numba")
    g_stack = (
        np.stack(hamiltonian.g_matrices).astype(np.complex128, copy=False)
        if hamiltonian.g_matrices
        else np.empty((0, hamiltonian.n_qubits, hamiltonian.n_qubits), dtype=np.complex128)
    )
    use_fused_numba = False
    use_chunked_numba = False
    resolved_block_chunk_size = hamiltonian.n_blocks
    one_body_workspace = np.empty(0, dtype=np.complex128)
    block_workspace = np.empty((0, 0), dtype=np.complex128)
    out_workspace = np.empty(0, dtype=np.complex128)
    if use_numba:
        if block_chunk_size is not None:
            resolved_block_chunk_size = min(int(block_chunk_size), hamiltonian.n_blocks)
        else:
            resolved_block_chunk_size = min(
                _FUSED_NUMBA_AUTO_BLOCK_CHUNK_SIZE,
                hamiltonian.n_blocks,
            )
        workspace_block_count = resolved_block_chunk_size
        fused_workspace_bytes = (
            (2 * sector.dimension) + (workspace_block_count * sector.dimension)
        ) * np.dtype(np.complex128).itemsize
        can_use_fused_family = (
            hamiltonian.n_blocks > 0
            and hamiltonian.n_blocks * sector.dimension >= _FUSED_NUMBA_MIN_BLOCK_DIM_WORK
            and fused_workspace_bytes <= _FUSED_NUMBA_WORKSPACE_LIMIT_BYTES
        )
        use_chunked_numba = (
            can_use_fused_family
            and resolved_block_chunk_size < hamiltonian.n_blocks
        )
        use_fused_numba = (
            can_use_fused_family
            and resolved_block_chunk_size >= hamiltonian.n_blocks
        )
        if use_fused_numba or use_chunked_numba:
            one_body_workspace = np.empty(sector.dimension, dtype=np.complex128)
            block_workspace = np.empty(
                (workspace_block_count, sector.dimension),
                dtype=np.complex128,
            )
            out_workspace = np.empty(sector.dimension, dtype=np.complex128)

    def _matvec(vec: np.ndarray) -> np.ndarray:
        counter["count"] += 1
        state = np.asarray(vec, dtype=np.complex128).reshape(-1)
        if state.shape[0] != sector.dimension:
            raise ValueError(
                f"Sector vector dimension mismatch: got {state.shape[0]}, "
                f"expected {sector.dimension}."
            )
        if use_numba:
            if use_chunked_numba:
                return _df_matvec_csr_numba_chunked(
                    state,
                    float(hamiltonian.constant),
                    np.asarray(hamiltonian.one_body, dtype=np.complex128),
                    np.asarray(hamiltonian.lambdas, dtype=np.float64),
                    g_stack,
                    table.target_indptr,
                    table.source_positions,
                    table.p_indices,
                    table.q_indices,
                    table.signs,
                    int(resolved_block_chunk_size),
                    one_body_workspace,
                    block_workspace,
                    out_workspace,
                )
            if use_fused_numba:
                return _df_matvec_csr_numba_fused(
                    state,
                    float(hamiltonian.constant),
                    np.asarray(hamiltonian.one_body, dtype=np.complex128),
                    np.asarray(hamiltonian.lambdas, dtype=np.float64),
                    g_stack,
                    table.target_indptr,
                    table.source_positions,
                    table.p_indices,
                    table.q_indices,
                    table.signs,
                    one_body_workspace,
                    block_workspace,
                    out_workspace,
                )
            return _df_matvec_csr_numba_streaming(
                state,
                float(hamiltonian.constant),
                np.asarray(hamiltonian.one_body, dtype=np.complex128),
                np.asarray(hamiltonian.lambdas, dtype=np.float64),
                g_stack,
                table.target_indptr,
                table.source_positions,
                table.p_indices,
                table.q_indices,
                table.signs,
            )
        return _df_matvec_python(hamiltonian, state, table, sector.dimension)

    return (
        LinearOperator(
            shape=(sector.dimension, sector.dimension),
            matvec=_matvec,
            rmatvec=_matvec,
            dtype=np.complex128,
        ),
        counter,
    )


def df_diagonal(hamiltonian: DFHamiltonian, sector: PhysicalSector) -> np.ndarray:
    """Return the determinant-basis diagonal of the DF Hamiltonian in `sector`."""
    occupations = _sector_occupations(sector)
    diag = np.full(sector.dimension, hamiltonian.constant, dtype=np.float64)
    one_body_diag = np.real(np.diag(hamiltonian.one_body)).astype(np.float64)
    diag += occupations @ one_body_diag

    for lam, g_matrix in zip(hamiltonian.lambdas, hamiltonian.g_matrices):
        g_diag = np.real(np.diag(g_matrix)).astype(np.float64)
        trace_occ = occupations @ g_diag
        abs_g2 = np.abs(g_matrix) ** 2
        total_to_occ = occupations @ np.sum(abs_g2, axis=0).real
        occ_abs = (occupations @ abs_g2.real) * occupations
        transition = total_to_occ - np.sum(occ_abs, axis=1)
        diag += float(lam) * (trace_occ * trace_occ + transition)
    return diag


def df_diagonal_preconditioner(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    floor: float = 1e-8,
) -> LinearOperator:
    """Build a positive diagonal inverse preconditioner for LOBPCG."""
    diagonal = df_diagonal(hamiltonian, sector)
    scale = np.abs(diagonal - float(np.min(diagonal)))
    scale = np.maximum(scale, float(floor))
    inv_scale = 1.0 / scale

    def _matvec(vec: np.ndarray) -> np.ndarray:
        state = np.asarray(vec, dtype=np.complex128)
        if state.ndim == 1:
            return inv_scale * state
        return inv_scale[:, None] * state

    return LinearOperator(
        shape=(sector.dimension, sector.dimension),
        matvec=_matvec,
        matmat=_matvec,
        dtype=np.complex128,
    )


def expand_sector_state(sector_state: np.ndarray, sector: PhysicalSector) -> np.ndarray:
    full_state = np.zeros(1 << sector.n_qubits, dtype=np.complex128)
    full_state[np.asarray(sector.basis_indices, dtype=np.int64)] = np.asarray(
        sector_state,
        dtype=np.complex128,
    ).reshape(-1)
    return full_state


def _sector_occupations(sector: PhysicalSector) -> np.ndarray:
    basis = np.asarray(sector.basis_indices, dtype=np.int64)
    occupations = np.empty((sector.dimension, sector.n_qubits), dtype=np.float64)
    for mode in range(sector.n_qubits):
        bit = 1 << (sector.n_qubits - 1 - mode)
        occupations[:, mode] = ((basis & bit) != 0).astype(np.float64)
    return occupations


def _df_matvec_python(
    hamiltonian: DFHamiltonian,
    state: np.ndarray,
    table: _ExcitationTable,
    dimension: int,
) -> np.ndarray:
    out = hamiltonian.constant * state
    out = out + _apply_one_body(hamiltonian.one_body, state, table, dimension)
    for lam, g_matrix in zip(hamiltonian.lambdas, hamiltonian.g_matrices):
        tmp = _apply_one_body(g_matrix, state, table, dimension)
        out = out + float(lam) * _apply_one_body(g_matrix, tmp, table, dimension)
    return out


def _apply_one_body(
    coeff_matrix: np.ndarray,
    state: np.ndarray,
    table: _ExcitationTable,
    dimension: int,
) -> np.ndarray:
    out = np.empty(dimension, dtype=np.complex128)
    for target_pos in range(dimension):
        start = int(table.target_indptr[target_pos])
        stop = int(table.target_indptr[target_pos + 1])
        entries = slice(start, stop)
        coeffs = coeff_matrix[table.p_indices[entries], table.q_indices[entries]]
        weighted = coeffs * table.signs[entries] * state[table.source_positions[entries]]
        out[target_pos] = np.sum(weighted, dtype=np.complex128)
    return out


def _linear_operator_to_dense(linear_operator: LinearOperator) -> np.ndarray:
    dim = int(linear_operator.shape[0])
    basis = np.eye(dim, dtype=np.complex128)
    return np.column_stack([linear_operator @ basis[:, idx] for idx in range(dim)])


def _build_excitation_table(sector: PhysicalSector) -> _ExcitationTable:
    basis = np.asarray(sector.basis_indices, dtype=np.int64)
    index_to_pos = {int(index): pos for pos, index in enumerate(basis)}
    before_masks = _before_mode_masks(sector.n_qubits)
    src_positions: list[int] = []
    target_positions: list[int] = []
    p_indices: list[int] = []
    q_indices: list[int] = []
    signs: list[int] = []

    for src_pos, basis_index_raw in enumerate(basis):
        basis_index = int(basis_index_raw)
        for q in range(sector.n_qubits):
            q_bit = 1 << (sector.n_qubits - 1 - q)
            if (basis_index & q_bit) == 0:
                continue
            removed = basis_index ^ q_bit
            sign_annihilate = -1 if (basis_index & before_masks[q]).bit_count() % 2 else 1
            for p in range(sector.n_qubits):
                p_bit = 1 << (sector.n_qubits - 1 - p)
                if (removed & p_bit) != 0:
                    continue
                target_index = removed | p_bit
                target_pos = index_to_pos.get(target_index)
                if target_pos is None:
                    continue
                sign_create = -1 if (removed & before_masks[p]).bit_count() % 2 else 1
                src_positions.append(src_pos)
                target_positions.append(target_pos)
                p_indices.append(p)
                q_indices.append(q)
                signs.append(sign_annihilate * sign_create)

    target_array = np.asarray(target_positions, dtype=np.int64)
    order = np.argsort(target_array, kind="stable")
    sorted_targets = target_array[order]
    counts = np.bincount(sorted_targets, minlength=sector.dimension)
    target_indptr = np.empty(sector.dimension + 1, dtype=np.int64)
    target_indptr[0] = 0
    target_indptr[1:] = np.cumsum(counts, dtype=np.int64)

    return _ExcitationTable(
        target_indptr=target_indptr,
        source_positions=np.asarray(src_positions, dtype=np.int64)[order],
        target_positions=sorted_targets,
        p_indices=np.asarray(p_indices, dtype=np.int64)[order],
        q_indices=np.asarray(q_indices, dtype=np.int64)[order],
        signs=np.asarray(signs, dtype=np.float64)[order],
    )


def _before_mode_masks(n_qubits: int) -> list[int]:
    masks = []
    for mode in range(int(n_qubits)):
        mask = 0
        for earlier in range(mode):
            mask |= 1 << (int(n_qubits) - 1 - earlier)
        masks.append(mask)
    return masks


def _hartree_fock_initial_vector(sector: PhysicalSector) -> np.ndarray | None:
    if sector.nelec_alpha is None or sector.nelec_beta is None:
        return None
    occupied = [2 * idx for idx in range(sector.nelec_alpha)]
    occupied.extend(2 * idx + 1 for idx in range(sector.nelec_beta))
    hf_index = sum(1 << (sector.n_qubits - 1 - mode) for mode in occupied)
    positions = np.where(sector.basis_indices == hf_index)[0]
    if positions.size != 1:
        return None
    vec = np.zeros(sector.dimension, dtype=np.complex128)
    vec[int(positions[0])] = 1.0
    return vec


def _normalize_initial_vector(
    v0: np.ndarray | None,
    sector: PhysicalSector,
) -> np.ndarray | None:
    if v0 is None:
        return None
    vec = np.asarray(v0, dtype=np.complex128).reshape(-1)
    if vec.shape[0] == sector.dimension:
        pass
    elif vec.shape[0] == (1 << sector.n_qubits):
        vec = vec[np.asarray(sector.basis_indices, dtype=np.int64)]
    else:
        raise ValueError(
            f"Initial vector length must be sector dim {sector.dimension} or "
            f"full dim {1 << sector.n_qubits}."
        )
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Initial vector must be nonzero.")
    return vec / norm


def _lobpcg_initial_block(
    initial: np.ndarray | None,
    dimension: int,
    block_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(12345)
    x0 = rng.normal(size=(dimension, block_size)).astype(np.complex128)
    if initial is not None:
        x0[:, 0] = np.asarray(initial, dtype=np.complex128).reshape(-1)
    q, _r = np.linalg.qr(x0, mode="reduced")
    return np.asarray(q[:, :block_size], dtype=np.complex128)


def _low_rank_kwargs(*, df_rank: int | None, df_tol: float | None) -> dict[str, Any]:
    params = inspect.signature(low_rank_two_body_decomposition).parameters
    kwargs: dict[str, Any] = {}
    if df_rank is not None:
        if df_rank <= 0:
            raise ValueError("df_rank must be positive.")
        if "final_rank" in params:
            kwargs["final_rank"] = int(df_rank)
        elif "max_rank" in params:
            kwargs["max_rank"] = int(df_rank)
        else:
            kwargs["rank"] = int(df_rank)
    if df_tol is not None:
        if df_tol <= 0:
            raise ValueError("df_tol must be positive.")
        if "truncation_threshold" in params:
            kwargs["truncation_threshold"] = float(df_tol)
        else:
            kwargs["tol"] = float(df_tol)
    if "spin_basis" in params:
        kwargs["spin_basis"] = True
    return kwargs


def _as_hermitian(matrix: np.ndarray, *, name: str, tol: float) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square.")
    skew_norm = float(np.linalg.norm(arr - arr.conj().T))
    if skew_norm > tol:
        # The low-rank routine can return tiny numerical asymmetries.  Keep the
        # model Hermitian because eigsh assumes a Hermitian LinearOperator.
        pass
    return 0.5 * (arr + arr.conj().T)

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Any

import numpy as np
from openfermion import count_qubits
from openfermion.chem import MolecularData
from openfermion.linalg import get_sparse_operator
from openfermion.ops import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from scipy.sparse.linalg import LinearOperator, eigsh

from .config import DEFAULT_BASIS, DEFAULT_DISTANCE

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    get_num_threads = None
    njit = None
    prange = range
    set_num_threads = None
    _NUMBA_AVAILABLE = False


_REAL_MATRIX_TOL = 1e-14


def available_cpu_count() -> int:
    """Return the number of CPUs available to this process."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_numba_threads(num_threads: Optional[int]) -> int:
    """Choose a numba thread count, respecting explicit requests and runtime caps."""
    requested = available_cpu_count() if num_threads is None else int(num_threads)
    if requested <= 0:
        requested = available_cpu_count()
    if not _NUMBA_AVAILABLE:
        return requested
    try:
        set_num_threads(requested)
        return int(get_num_threads())
    except ValueError:
        # NUMBA_NUM_THREADS can cap the runtime maximum. Use the current maximum
        # mask instead of failing when auto-detected CPUs exceed that cap.
        current = int(get_num_threads())
        set_num_threads(current)
        return current


if _NUMBA_AVAILABLE:

    @njit(cache=True, inline="always")
    def _parity64(value: np.int64) -> np.int64:
        value ^= value >> 32
        value ^= value >> 16
        value ^= value >> 8
        value ^= value >> 4
        value &= 0xF
        return (0x6996 >> value) & 1


    @njit(cache=True, parallel=True)
    def _pauli_sum_matvec_real_numba(
        state: np.ndarray,
        identity_coeff: float,
        coeffs: np.ndarray,
        flip_masks: np.ndarray,
        phase_masks: np.ndarray,
    ) -> np.ndarray:
        out = np.empty_like(state)
        n_terms = coeffs.shape[0]
        for index in prange(state.shape[0]):
            acc = identity_coeff * state[index]
            for term_index in range(n_terms):
                source = index ^ flip_masks[term_index]
                value = coeffs[term_index] * state[source]
                if _parity64(source & phase_masks[term_index]) != 0:
                    value = -value
                acc += value
            out[index] = acc
        return out


    @njit(cache=True, parallel=True)
    def _pauli_sum_matvec_complex_numba(
        state: np.ndarray,
        identity_coeff: complex,
        coeffs: np.ndarray,
        flip_masks: np.ndarray,
        phase_masks: np.ndarray,
    ) -> np.ndarray:
        out = np.empty_like(state)
        n_terms = coeffs.shape[0]
        for index in prange(state.shape[0]):
            acc = identity_coeff * state[index]
            for term_index in range(n_terms):
                source = index ^ flip_masks[term_index]
                value = coeffs[term_index] * state[source]
                if _parity64(source & phase_masks[term_index]) != 0:
                    value = -value
                acc += value
            out[index] = acc
        return out


def call_geometry(
    Hchain: int,
    distance: float,
) -> Tuple[List[Tuple[str, Tuple[float, float, float]]], int, int]:
    """H-chain の構造と多重度・電荷を返す。"""
    # multiplicity と charge の規則
    if Hchain % 2 == 0:
        multiplicity = 1
        charge = 0
    else:
        multiplicity = 3
        charge = +1

    # 原子の直線配置を作成
    shift = (Hchain - 1) / 2.0

    geometry = [("H", (0.0, 0.0, distance * (i - shift))) for i in range(Hchain)]

    return geometry, multiplicity, charge


def geo(
    mol_type: int,
    distance: Optional[float] = None,
) -> Tuple[List[Tuple[str, Tuple[float, float, float]]], int, int]:
    """分子タイプに対応する H-chain 情報を返す。"""
    # distance 未指定時はデフォルト値を使用
    if distance is None:
        distance = DEFAULT_DISTANCE
    return call_geometry(mol_type, distance)

def ham_list_maker(hamiltonian: QubitOperator) -> List[QubitOperator]:
    """OpenFermion ハミルトニアンを項ごとのリストに変換する。"""
    # QubitOperator を項単位に展開
    return [term for term in hamiltonian]


def _compiled_qubit_operator_terms(
    qubit_operator: QubitOperator,
) -> Tuple[complex, Tuple[Tuple[complex, Tuple[Tuple[int, str], ...]], ...]]:
    """Split QubitOperator into identity coefficient and per-term Pauli actions."""
    identity_coeff = 0.0 + 0.0j
    compiled_terms: list[Tuple[complex, Tuple[Tuple[int, str], ...]]] = []
    for pauli_term, coeff in qubit_operator.terms.items():
        coeff_complex = complex(coeff)
        if pauli_term == ():
            identity_coeff += coeff_complex
            continue
        compiled_terms.append(
            (
                coeff_complex,
                tuple(sorted(((int(qubit), str(pauli)) for qubit, pauli in pauli_term))),
            )
        )
    return identity_coeff, tuple(compiled_terms)


def _compile_pauli_bitmasks(
    qubit_operator: QubitOperator,
    *,
    n_qubits: int,
) -> Tuple[complex, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Compile Pauli strings into bit masks for fast matrix-free application.

    OpenFermion's JW convention maps qubit q to bit position n_qubits - 1 - q.
    For output basis index i, the corresponding source index is i ^ flip_mask.
    The phase is coeff * (1j)**num_y times (-1)**popcount(source & phase_mask).
    """
    identity_coeff = 0.0 + 0.0j
    coeffs: list[complex] = []
    flip_masks: list[int] = []
    phase_masks: list[int] = []
    max_mask = (1 << n_qubits) - 1

    for pauli_term, coeff in qubit_operator.terms.items():
        coeff_complex = complex(coeff)
        if pauli_term == ():
            identity_coeff += coeff_complex
            continue

        flip_mask = 0
        phase_mask = 0
        y_count = 0
        for qubit_raw, pauli_raw in pauli_term:
            qubit = int(qubit_raw)
            pauli = str(pauli_raw)
            if qubit < 0 or qubit >= n_qubits:
                raise ValueError(
                    f"Pauli term acts on qubit {qubit}, but n_qubits={n_qubits}."
                )
            bit = 1 << (n_qubits - 1 - qubit)
            if pauli == "X":
                flip_mask |= bit
            elif pauli == "Y":
                flip_mask |= bit
                phase_mask |= bit
                y_count += 1
            elif pauli == "Z":
                phase_mask |= bit
            else:
                raise ValueError(f"Unsupported Pauli operator: {pauli}")

        if flip_mask > max_mask or phase_mask > max_mask:
            raise ValueError("Compiled Pauli mask exceeds the statevector dimension.")
        coeffs.append(coeff_complex * ((1j) ** y_count))
        flip_masks.append(flip_mask)
        phase_masks.append(phase_mask)

    coeff_array = np.asarray(coeffs, dtype=np.complex128)
    flip_array = np.asarray(flip_masks, dtype=np.int64)
    phase_array = np.asarray(phase_masks, dtype=np.int64)
    is_real = (
        abs(identity_coeff.imag) <= _REAL_MATRIX_TOL
        and (
            coeff_array.size == 0
            or float(np.max(np.abs(coeff_array.imag))) <= _REAL_MATRIX_TOL
        )
    )
    return identity_coeff, coeff_array, flip_array, phase_array, is_real


def _apply_single_pauli_axis(
    state_tensor: np.ndarray,
    axis: int,
    pauli: str,
) -> np.ndarray:
    """
    Apply a single-qubit Pauli action along one tensor axis.

    The tensor axes follow OpenFermion's JW basis convention used in the DF
    project: qubit 0 is the most-significant bit of the computational-basis
    index, i.e. basis index = sum_q occ_q * 2**(n_qubits - 1 - q).
    """
    moved = np.moveaxis(state_tensor, axis, 0)
    if pauli == "X":
        transformed = np.stack((moved[1], moved[0]), axis=0)
    elif pauli == "Y":
        transformed = np.stack((-1j * moved[1], 1j * moved[0]), axis=0)
    elif pauli == "Z":
        transformed = np.stack((moved[0], -moved[1]), axis=0)
    else:
        raise ValueError(f"Unsupported Pauli operator: {pauli}")
    return np.moveaxis(transformed, 0, axis)


def matrix_free_qubit_operator(
    qubit_operator: QubitOperator,
    *,
    n_qubits: Optional[int] = None,
    backend: str = "auto",
    num_threads: Optional[int] = None,
) -> LinearOperator:
    """
    Build a matrix-free LinearOperator for a QubitOperator.

    This avoids materializing the sparse matrix and instead applies each Pauli
    term directly to the statevector tensor with OpenFermion-consistent index
    ordering.
    """
    if backend not in ("auto", "numba", "python"):
        raise ValueError("backend must be 'auto', 'numba', or 'python'.")
    if backend == "numba" and not _NUMBA_AVAILABLE:
        raise RuntimeError("matrix_free backend='numba' requires numba to be installed.")
    if num_threads is not None and num_threads < 0:
        raise ValueError("num_threads must be non-negative.")
    if backend in ("auto", "numba"):
        if not _NUMBA_AVAILABLE:
            if num_threads is not None:
                raise RuntimeError("num_threads requires the numba matrix-free backend.")
        else:
            _resolve_numba_threads(num_threads)

    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    n_qubits = int(n_qubits)
    if n_qubits < 0:
        raise ValueError("n_qubits must be non-negative.")
    dimension = 1 << n_qubits
    use_numba = _NUMBA_AVAILABLE and backend in ("auto", "numba")
    if use_numba:
        identity_coeff, coeffs, flip_masks, phase_masks, is_real = _compile_pauli_bitmasks(
            qubit_operator,
            n_qubits=n_qubits,
        )

        if is_real:
            identity_real = float(identity_coeff.real)
            coeffs_real = coeffs.real.astype(np.float64, copy=False)

            def _matvec(vec: np.ndarray) -> np.ndarray:
                state = np.asarray(vec, dtype=np.float64).reshape(-1)
                if state.shape[0] != dimension:
                    raise ValueError(
                        "Statevector dimension mismatch: "
                        f"got {state.shape[0]}, expected {dimension}."
                    )
                return _pauli_sum_matvec_real_numba(
                    state,
                    identity_real,
                    coeffs_real,
                    flip_masks,
                    phase_masks,
                )

            dtype = np.float64
        else:

            def _matvec(vec: np.ndarray) -> np.ndarray:
                state = np.asarray(vec, dtype=np.complex128).reshape(-1)
                if state.shape[0] != dimension:
                    raise ValueError(
                        "Statevector dimension mismatch: "
                        f"got {state.shape[0]}, expected {dimension}."
                    )
                return _pauli_sum_matvec_complex_numba(
                    state,
                    identity_coeff,
                    coeffs,
                    flip_masks,
                    phase_masks,
                )

            dtype = np.complex128

        return LinearOperator(
            shape=(dimension, dimension),
            matvec=_matvec,
            rmatvec=_matvec,
            dtype=dtype,
        )

    identity_coeff, compiled_terms = _compiled_qubit_operator_terms(qubit_operator)
    tensor_shape = (2,) * n_qubits

    def _matvec(vec: np.ndarray) -> np.ndarray:
        state = np.asarray(vec, dtype=np.complex128).reshape(-1)
        if state.shape[0] != dimension:
            raise ValueError(
                f"Statevector dimension mismatch: got {state.shape[0]}, expected {dimension}."
            )
        out = np.zeros(dimension, dtype=np.complex128)
        if identity_coeff != 0.0:
            out += identity_coeff * state
        if not compiled_terms:
            return out
        if n_qubits == 0:
            return out
        state_tensor = state.reshape(tensor_shape)
        for coeff, pauli_actions in compiled_terms:
            term_tensor = state_tensor
            for qubit, pauli in pauli_actions:
                if qubit < 0 or qubit >= n_qubits:
                    raise ValueError(
                        f"Pauli term acts on qubit {qubit}, but n_qubits={n_qubits}."
                    )
                term_tensor = _apply_single_pauli_axis(term_tensor, qubit, pauli)
            out += coeff * np.asarray(term_tensor, dtype=np.complex128).reshape(-1)
        return out

    return LinearOperator(
        shape=(dimension, dimension),
        matvec=_matvec,
        rmatvec=_matvec,
        dtype=np.complex128,
    )


def _linear_operator_to_dense(linear_operator: LinearOperator) -> np.ndarray:
    """Materialize a tiny LinearOperator for dense fallback / validation."""
    dim = int(linear_operator.shape[0])
    basis = np.eye(dim, dtype=np.complex128)
    return np.column_stack([linear_operator @ basis[:, idx] for idx in range(dim)])


def ham_ground_energy(
    jw_hamiltonian: QubitOperator,
    *,
    n_qubits: Optional[int] = None,
    method: str = "matrix_free",
    matrix_free_backend: str = "auto",
    matrix_free_threads: Optional[int] = None,
    return_max_eig: bool = True,
    solver_tol: float = 1e-10,
    solver_maxiter: Optional[int] = None,
    solver_ncv: Optional[int] = None,
    v0: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, Optional[float]]:
    """
    Return the ground-state energy, eigenvector, and optionally the maximum eigenvalue.

    The default path is matrix-free so the sparse matrix is not materialized.
    `matrix_free_backend="auto"` uses the multithreaded numba bitmask kernel when
    available, while `method="sparse"` remains as a reference / validation fallback.
    """
    if n_qubits is None:
        n_qubits = count_qubits(jw_hamiltonian)
    n_qubits = int(n_qubits)
    dimension = 1 << n_qubits
    if dimension <= 0:
        raise ValueError("Hamiltonian must act on at least a one-dimensional space.")
    if solver_tol <= 0.0:
        raise ValueError("solver_tol must be positive.")
    if solver_ncv is not None and solver_ncv <= 2:
        raise ValueError("solver_ncv must be greater than 2 for k=1 eigsh.")

    if method == "matrix_free":
        linear_operator = matrix_free_qubit_operator(
            jw_hamiltonian,
            n_qubits=n_qubits,
            backend=matrix_free_backend,
            num_threads=matrix_free_threads,
        )
    elif method == "sparse":
        linear_operator = get_sparse_operator(jw_hamiltonian, n_qubits=n_qubits)
    else:
        raise ValueError("method must be either 'matrix_free' or 'sparse'.")

    if v0 is not None:
        v0_dtype = np.float64 if linear_operator.dtype == np.float64 else np.complex128
        v0 = np.asarray(v0, dtype=v0_dtype).reshape(-1)
        if v0.shape[0] != dimension:
            raise ValueError(
                f"Initial vector dimension mismatch: got {v0.shape[0]}, expected {dimension}."
            )

    if dimension <= 4:
        dense_matrix = (
            linear_operator.toarray()
            if hasattr(linear_operator, "toarray")
            else _linear_operator_to_dense(linear_operator)
        )
        evals, evecs = np.linalg.eigh(np.asarray(dense_matrix, dtype=np.complex128))
        min_index = int(np.argmin(evals.real))
        energy = float(np.real_if_close(evals[min_index]))
        state_vec = np.asarray(evecs[:, min_index : min_index + 1], dtype=np.complex128)
        max_eig = float(np.real_if_close(np.max(evals.real))) if return_max_eig else None
        return energy, state_vec, max_eig

    vals, vecs = eigsh(
        linear_operator,
        k=1,
        return_eigenvectors=True,
        which="SA",
        tol=float(solver_tol),
        maxiter=solver_maxiter,
        ncv=solver_ncv,
        v0=v0,
    )
    energy = float(np.real_if_close(vals[0]))
    state_vec = np.asarray(vecs[:, :1], dtype=np.complex128)
    max_eig = None
    if return_max_eig:
        max_vals = eigsh(
            linear_operator,
            k=1,
            return_eigenvectors=False,
            which="LA",
            tol=float(solver_tol),
            maxiter=solver_maxiter,
            ncv=solver_ncv,
        )
        max_eig = float(np.real_if_close(max_vals[0]))
    return energy, state_vec, max_eig


def jw_hamiltonian_maker(
    mol_type: int,
    distance: Optional[float] = None,
) -> Tuple[QubitOperator, float, str, int]:
    """
    JW 変換ハミルトニアンを構築して返す。

    Returns:
        jw_hamiltonian, HF_energy, ham_name, num_qubits
    """
    # 分子情報を組み立てて PySCF で計算
    basis = DEFAULT_BASIS
    if distance is None:
        distance = DEFAULT_DISTANCE
    geometry, multiplicity, charge = geo(mol_type, distance)
    name_distance = int(distance * 100)
    description = f"distance_{name_distance}_charge_{charge}"  # 保存先のファイル名
    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_pyscf(molecule, run_scf=1, run_fci=0)
    hf_energy = molecule.hf_energy
    # フェルミオン演算子を JW に変換
    jw_hamiltonian = jordan_wigner(
        get_fermion_operator(molecule.get_molecular_hamiltonian())
    )
    num_qubits = count_qubits(jw_hamiltonian)
    file_path = molecule.filename
    ham_name = os.path.splitext(os.path.basename(file_path))[0]
    print(ham_name)
    return jw_hamiltonian, hf_energy, ham_name, num_qubits


def min_hamiltonian_grouper(
    hamiltonian: QubitOperator, ham_name: str
) -> Tuple[List[QubitOperator], str]:
    """可換な項でグルーピングし、グループごとの QubitOperator のリストを返す。"""

    def are_commuting(op1: QubitOperator, op2: QubitOperator) -> bool:
        # 単一項同士の可換性を判定
        if len(op1.terms) != 1 or len(op2.terms) != 1:
            raise ValueError("Only single-term QubitOperators are supported.")
        term1 = list(op1.terms.keys())[0]
        term2 = list(op2.terms.keys())[0]
        n_anticommute = 0
        qubits = set(index for index, _ in term1).union(index for index, _ in term2)
        for q in qubits:
            op1_pauli = dict(term1).get(q, "I")
            op2_pauli = dict(term2).get(q, "I")
            if op1_pauli == "I" or op2_pauli == "I":
                continue
            if op1_pauli != op2_pauli:
                n_anticommute += 1
        return n_anticommute % 2 == 0

    def group_commuting_terms(qubit_hamiltonian: QubitOperator) -> List[QubitOperator]:
        # 貪欲に可換クリークへ分割
        terms = [
            QubitOperator(term, coeff)
            for term, coeff in qubit_hamiltonian.terms.items()
        ]
        groups: List[List[QubitOperator]] = []
        for term in terms:
            for group in groups:
                if all(are_commuting(term, other) for other in group):
                    group.append(term)
                    break
            else:
                groups.append([term])
        return [sum(group, QubitOperator()) for group in groups]

    # グループごとに QubitOperator を合算
    grouped_ops = group_commuting_terms(hamiltonian)
    return grouped_ops, f"{ham_name}_grouping"

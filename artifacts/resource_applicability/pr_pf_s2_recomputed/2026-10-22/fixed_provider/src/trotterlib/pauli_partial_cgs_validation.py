from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion.ops import QubitOperator
from scipy.linalg import eigh, schur
from scipy.optimize import minimize_scalar

from .analysis_utils import loglog_average_coeff, loglog_fit
from .config import PARTIAL_RANDOMIZED_Q_MAX, PARTIAL_RANDOMIZED_Q_MIN
from .df_hamiltonian import PhysicalSector
from .partial_randomized_pf import (
    SortedPauliHamiltonian,
    deterministic_step_cost,
    optimize_error_budget_and_kappa,
    randomized_prefactor_B,
    split_hamiltonian_by_ld,
)


@dataclass(frozen=True)
class FullHTargetState:
    """A target eigenstate of the full Hamiltonian in its physical sector."""

    energy: float
    state: np.ndarray
    residual_norm: float
    sector: PhysicalSector
    global_ground_energy: float
    sector_ground_multiplicity: int
    sector_excitation_gap: float | None
    full_space_target_energy_multiplicity: int
    full_space_target_energy_gap: float | None
    full_space_target_eigenspace: np.ndarray
    target_eigenspace_projection_residual: float
    max_full_spectral_distance_from_target: float


@dataclass(frozen=True)
class PartialS2CgsFit:
    """State-specific and eigenbranch Cgs fits for one Pauli prefix."""

    ld: int
    lambda_r: float
    delta_values: tuple[float, ...]
    qpe_rmse_biases: tuple[float, ...]
    ground_space_mean_rmse_biases: tuple[float, ...]
    ground_space_worst_rmse_biases: tuple[float, ...]
    qpe_mean_biases: tuple[float, ...]
    qpe_energy_stds: tuple[float, ...]
    signal_phase_biases: tuple[float, ...]
    branch_energy_biases: tuple[float, ...]
    branch_overlaps: tuple[float, ...]
    target_cluster_ground_biases: tuple[float, ...]
    target_cluster_max_abs_biases: tuple[float, ...]
    target_cluster_projector_overlaps: tuple[float, ...]
    target_cluster_ranks: tuple[int, ...]
    target_cluster_ground_state_weights: tuple[float, ...]
    unitary_defects: tuple[float, ...]
    phase_branch_cut_clearances: tuple[float, ...]
    c_gs_qpe_rmse: float
    c_gs_ground_space_mean_rmse: float
    c_gs_ground_space_worst_rmse: float
    c_gs_qpe_mean: float
    c_gs_signal: float
    c_gs_branch: float
    c_gs_target_cluster_ground: float
    c_gs_target_cluster_max_abs: float
    qpe_rmse_fit_slope: float | None
    ground_space_mean_rmse_fit_slope: float | None
    ground_space_worst_rmse_fit_slope: float | None
    qpe_mean_fit_slope: float | None
    signal_fit_slope: float | None
    branch_fit_slope: float | None
    target_cluster_ground_fit_slope: float | None
    target_cluster_max_abs_fit_slope: float | None
    min_branch_overlap: float
    min_target_cluster_projector_overlap: float
    min_target_cluster_principal_overlap: float
    min_target_cluster_ground_state_weight: float
    min_phase_branch_cut_clearance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _dense_operator(operator: QubitOperator, num_qubits: int) -> np.ndarray:
    return np.asarray(
        get_sparse_operator(operator, n_qubits=int(num_qubits)).toarray(),
        dtype=np.complex128,
    )


def dense_full_hamiltonian(sorted_hamiltonian: SortedPauliHamiltonian) -> np.ndarray:
    """Materialize the exact JW Hamiltonian represented by ``sorted_hamiltonian``."""

    dim = 1 << int(sorted_hamiltonian.num_qubits)
    matrix = float(sorted_hamiltonian.identity_coeff) * np.eye(dim, dtype=np.complex128)
    for term in sorted_hamiltonian.sorted_terms:
        matrix += _dense_operator(term.operator, sorted_hamiltonian.num_qubits)
    return 0.5 * (matrix + matrix.conj().T)


def full_h_target_state(
    sorted_hamiltonian: SortedPauliHamiltonian,
    *,
    physical_sector: PhysicalSector | None = None,
    degeneracy_tolerance: float = 1e-9,
) -> FullHTargetState:
    """Solve the full-H target state, using the H-chain physical sector by default."""

    full_h = dense_full_hamiltonian(sorted_hamiltonian)
    sector = physical_sector or PhysicalSector.from_h_chain(
        sorted_hamiltonian.molecule_type
    )
    if int(sector.n_qubits) != int(sorted_hamiltonian.num_qubits):
        raise ValueError("physical sector and Hamiltonian have different qubit counts")

    indices = np.asarray(sector.basis_indices, dtype=np.int64)
    sector_h = full_h[np.ix_(indices, indices)]
    sector_evals, sector_evecs = eigh(sector_h)
    energy = float(np.real_if_close(sector_evals[0]))
    state = np.zeros(full_h.shape[0], dtype=np.complex128)
    state[indices] = np.asarray(sector_evecs[:, 0], dtype=np.complex128)
    state /= np.linalg.norm(state)
    residual_norm = float(np.linalg.norm(full_h @ state - energy * state))
    full_evals, full_evecs = eigh(full_h)
    full_evals = np.asarray(full_evals, dtype=float)
    tolerance = float(degeneracy_tolerance)
    sector_offsets = np.asarray(sector_evals, dtype=float) - energy
    sector_ground_multiplicity = int(np.count_nonzero(np.abs(sector_offsets) <= tolerance))
    sector_excited_offsets = sector_offsets[sector_offsets > tolerance]
    full_offsets = np.abs(full_evals - energy)
    full_space_target_energy_multiplicity = int(
        np.count_nonzero(full_offsets <= tolerance)
    )
    full_cluster_mask = full_offsets <= tolerance
    full_target_eigenspace = np.asarray(
        full_evecs[:, full_cluster_mask], dtype=np.complex128
    )
    target_projection = full_target_eigenspace @ (
        full_target_eigenspace.conj().T @ state
    )
    full_noncluster_offsets = full_offsets[full_offsets > tolerance]
    return FullHTargetState(
        energy=energy,
        state=state,
        residual_norm=residual_norm,
        sector=sector,
        global_ground_energy=float(np.real_if_close(full_evals[0])),
        sector_ground_multiplicity=sector_ground_multiplicity,
        sector_excitation_gap=(
            None
            if sector_excited_offsets.size == 0
            else float(np.min(sector_excited_offsets))
        ),
        full_space_target_energy_multiplicity=full_space_target_energy_multiplicity,
        full_space_target_energy_gap=(
            None
            if full_noncluster_offsets.size == 0
            else float(np.min(full_noncluster_offsets))
        ),
        full_space_target_eigenspace=full_target_eigenspace,
        target_eigenspace_projection_residual=float(
            np.linalg.norm(target_projection - state)
        ),
        max_full_spectral_distance_from_target=float(np.max(full_offsets)),
    )


def dense_ranked_pauli_terms(
    sorted_hamiltonian: SortedPauliHamiltonian,
) -> tuple[np.ndarray, ...]:
    return tuple(
        _dense_operator(term.operator, sorted_hamiltonian.num_qubits)
        for term in sorted_hamiltonian.sorted_terms
    )


def _pauli_term_exponential(
    term_matrix: np.ndarray,
    coefficient: float,
    time: float,
) -> np.ndarray:
    """Exponentiate hP analytically, where ``term_matrix == h * P``."""

    coefficient = float(coefficient)
    if coefficient == 0.0:
        return np.eye(term_matrix.shape[0], dtype=np.complex128)
    pauli = np.asarray(term_matrix, dtype=np.complex128) / coefficient
    angle = float(time) * coefficient
    return (
        math.cos(angle) * np.eye(term_matrix.shape[0], dtype=np.complex128)
        - 1j * math.sin(angle) * pauli
    )


def partial_s2_unitary(
    sorted_hamiltonian: SortedPauliHamiltonian,
    dense_terms: Sequence[np.ndarray],
    *,
    ld: int,
    delta: float,
) -> np.ndarray:
    """Build S2(H_1,...,H_LD,H_R), treating H_R as one exact block."""

    ld = int(ld)
    delta = float(delta)
    if ld < 0 or ld > sorted_hamiltonian.num_terms:
        raise ValueError("ld is outside the Pauli-prefix range")
    if len(dense_terms) != sorted_hamiltonian.num_terms:
        raise ValueError("dense_terms does not match the sorted Hamiltonian")

    dim = 1 << int(sorted_hamiltonian.num_qubits)
    identity = np.eye(dim, dtype=np.complex128)
    h_r = float(sorted_hamiltonian.identity_coeff) * identity
    for matrix in dense_terms[ld:]:
        h_r = h_r + np.asarray(matrix, dtype=np.complex128)
    h_r = 0.5 * (h_r + h_r.conj().T)
    hr_evals, hr_evecs = eigh(h_r)
    return _partial_s2_unitary_from_tail_eigendecomposition(
        sorted_hamiltonian,
        dense_terms,
        ld=ld,
        delta=delta,
        hr_evals=hr_evals,
        hr_evecs=hr_evecs,
    )


def _partial_s2_unitary_from_tail_eigendecomposition(
    sorted_hamiltonian: SortedPauliHamiltonian,
    dense_terms: Sequence[np.ndarray],
    *,
    ld: int,
    delta: float,
    hr_evals: np.ndarray,
    hr_evecs: np.ndarray,
) -> np.ndarray:
    dim = 1 << int(sorted_hamiltonian.num_qubits)
    identity = np.eye(dim, dtype=np.complex128)
    hr_exp = (hr_evecs * np.exp(-1j * delta * hr_evals)) @ hr_evecs.conj().T

    unitary = identity
    half_steps: list[np.ndarray] = []
    for term, matrix in zip(sorted_hamiltonian.sorted_terms[:ld], dense_terms[:ld]):
        step = _pauli_term_exponential(matrix, term.coeff, 0.5 * delta)
        half_steps.append(step)
        unitary = step @ unitary
    unitary = hr_exp @ unitary
    for step in reversed(half_steps):
        unitary = step @ unitary
    return unitary


def _unwrap_energy_from_phase(phase: float, delta: float, reference: float) -> float:
    raw = -float(phase) / float(delta)
    period = 2.0 * math.pi / float(delta)
    return float(raw + period * round((float(reference) - raw) / period))


def _fixed_order_fit(
    delta_values: Sequence[float],
    biases: Sequence[float],
    *,
    order: int,
    noise_floor: float = 1e-14,
) -> tuple[float, float | None]:
    delta_array = np.asarray(delta_values, dtype=float)
    bias_array = np.asarray(biases, dtype=float)
    mask = (delta_array > 0.0) & (bias_array > float(noise_floor))
    if not np.any(mask):
        return 0.0, None
    coeff = float(
        loglog_average_coeff(
            delta_array[mask],
            bias_array[mask],
            float(order),
            mask_nonpositive=True,
        )
    )
    slope: float | None = None
    if int(np.count_nonzero(mask)) >= 2:
        slope = float(
            loglog_fit(
                delta_array[mask], bias_array[mask], mask_nonpositive=True
            ).slope
        )
    return coeff, slope


def fit_partial_s2_cgs(
    sorted_hamiltonian: SortedPauliHamiltonian,
    dense_terms: Sequence[np.ndarray],
    target: FullHTargetState,
    *,
    ld: int,
    delta_values: Sequence[float],
    noise_floor: float = 1e-14,
) -> PartialS2CgsFit:
    """Fit the full-H state-specific Cgs for one deterministic prefix."""

    partition = split_hamiltonian_by_ld(sorted_hamiltonian, int(ld))
    deltas = tuple(float(value) for value in delta_values)
    if not deltas or any(value <= 0.0 for value in deltas):
        raise ValueError("delta_values must contain positive values")
    if (
        max(deltas) * float(target.max_full_spectral_distance_from_target)
        >= math.pi
    ):
        raise ValueError(
            "delta grid is not phase-alias safe around the target energy: "
            "require max(delta) * max_j|E_j-E_target| < pi"
        )

    qpe_rmse_biases: list[float] = []
    ground_space_mean_rmse_biases: list[float] = []
    ground_space_worst_rmse_biases: list[float] = []
    qpe_mean_biases: list[float] = []
    qpe_energy_stds: list[float] = []
    signal_biases: list[float] = []
    branch_biases: list[float] = []
    branch_overlaps: list[float] = []
    target_cluster_ground_biases: list[float] = []
    target_cluster_max_abs_biases: list[float] = []
    target_cluster_projector_overlaps: list[float] = []
    target_cluster_principal_overlaps: list[float] = []
    target_cluster_ranks: list[int] = []
    target_cluster_ground_state_weights: list[float] = []
    unitary_defects: list[float] = []
    phase_branch_cut_clearances: list[float] = []
    identity = np.eye(target.state.size, dtype=np.complex128)
    h_r = float(sorted_hamiltonian.identity_coeff) * identity
    for matrix in dense_terms[int(ld) :]:
        h_r = h_r + np.asarray(matrix, dtype=np.complex128)
    h_r = 0.5 * (h_r + h_r.conj().T)
    hr_evals, hr_evecs = eigh(h_r)

    for delta in deltas:
        unitary = _partial_s2_unitary_from_tail_eigendecomposition(
            sorted_hamiltonian,
            dense_terms,
            ld=int(ld),
            delta=delta,
            hr_evals=hr_evals,
            hr_evecs=hr_evecs,
        )
        unitary_defects.append(
            float(np.linalg.norm(unitary.conj().T @ unitary - identity, ord=2))
        )

        amplitude = complex(np.vdot(target.state, unitary @ target.state))
        signal_energy = _unwrap_energy_from_phase(
            np.angle(amplitude), delta, target.energy
        )
        signal_biases.append(abs(signal_energy - target.energy))

        triangular, vectors = schur(unitary, output="complex")
        eigenvalues = np.diag(triangular)
        centered_phases = np.angle(
            np.exp(1j * delta * float(target.energy)) * eigenvalues
        )
        phase_branch_cut_clearance = float(
            math.pi - np.max(np.abs(centered_phases))
        )
        if phase_branch_cut_clearance <= 1e-8:
            raise ValueError(
                "S2 spectrum reaches the target-centered logarithm branch cut; "
                "reduce delta_values"
            )
        phase_branch_cut_clearances.append(phase_branch_cut_clearance)
        weights = np.abs(vectors.conj().T @ target.state) ** 2
        weights = np.asarray(weights, dtype=float)
        weights /= float(np.sum(weights))
        branch_energies = np.asarray(
            [
                _unwrap_energy_from_phase(np.angle(value), delta, target.energy)
                for value in eigenvalues
            ],
            dtype=float,
        )
        energy_shifts = branch_energies - float(target.energy)
        mean_shift = float(np.sum(weights * energy_shifts))
        qpe_rmse_biases.append(
            float(math.sqrt(max(0.0, np.sum(weights * (energy_shifts**2)))))
        )
        qpe_mean_biases.append(abs(mean_shift))
        qpe_energy_stds.append(
            float(
                math.sqrt(
                    max(0.0, np.sum(weights * ((energy_shifts - mean_shift) ** 2)))
                )
            )
        )

        ground_amplitudes = (
            vectors.conj().T @ target.full_space_target_eigenspace
        )
        ground_second_moment = ground_amplitudes.conj().T @ (
            (energy_shifts**2)[:, np.newaxis] * ground_amplitudes
        )
        ground_second_moment = 0.5 * (
            ground_second_moment + ground_second_moment.conj().T
        )
        ground_second_moment_evals = np.asarray(
            np.linalg.eigvalsh(ground_second_moment), dtype=float
        )
        ground_space_mean_rmse_biases.append(
            float(
                math.sqrt(
                    max(
                        0.0,
                        float(np.trace(ground_second_moment).real)
                        / target.full_space_target_energy_multiplicity,
                    )
                )
            )
        )
        ground_space_worst_rmse_biases.append(
            float(math.sqrt(max(0.0, float(ground_second_moment_evals[-1]))))
        )

        target_gap = target.full_space_target_energy_gap
        if target_gap is None:
            cluster_indices = np.argsort(
                -np.sum(np.abs(ground_amplitudes) ** 2, axis=1)
            )[: target.full_space_target_energy_multiplicity]
        else:
            cluster_indices = np.flatnonzero(
                np.abs(energy_shifts) < 0.5 * float(target_gap)
            )
        target_cluster_ranks.append(int(cluster_indices.size))
        full_cluster_weights = np.sum(np.abs(ground_amplitudes) ** 2, axis=1)
        target_cluster_projector_overlaps.append(
            float(
                np.sum(full_cluster_weights[cluster_indices])
                / target.full_space_target_energy_multiplicity
            )
        )
        if cluster_indices.size == 0:
            target_cluster_principal_overlaps.append(0.0)
        else:
            cluster_ground_amplitudes = ground_amplitudes[cluster_indices, :]
            principal_overlap_matrix = (
                cluster_ground_amplitudes.conj().T @ cluster_ground_amplitudes
            )
            principal_overlap_matrix = 0.5 * (
                principal_overlap_matrix + principal_overlap_matrix.conj().T
            )
            target_cluster_principal_overlaps.append(
                float(
                    np.clip(
                        np.min(np.linalg.eigvalsh(principal_overlap_matrix)),
                        0.0,
                        1.0,
                    )
                )
            )
        if cluster_indices.size == 0:
            target_cluster_ground_biases.append(float("nan"))
            target_cluster_max_abs_biases.append(float("nan"))
            target_cluster_ground_state_weights.append(0.0)
        else:
            cluster_shifts = energy_shifts[cluster_indices]
            ground_shift = float(np.min(cluster_shifts))
            target_cluster_ground_biases.append(abs(ground_shift))
            target_cluster_max_abs_biases.append(
                float(np.max(np.abs(cluster_shifts)))
            )
            ground_mask = np.abs(cluster_shifts - ground_shift) <= 1e-10
            target_cluster_ground_state_weights.append(
                float(np.sum(weights[cluster_indices[ground_mask]]))
            )

        branch_index = int(np.argmax(weights))
        branch_energy = float(branch_energies[branch_index])
        branch_biases.append(abs(branch_energy - target.energy))
        branch_overlaps.append(float(np.real_if_close(weights[branch_index])))

    c_qpe_rmse, slope_qpe_rmse = _fixed_order_fit(
        deltas, qpe_rmse_biases, order=2, noise_floor=noise_floor
    )
    c_ground_mean_rmse, slope_ground_mean_rmse = _fixed_order_fit(
        deltas, ground_space_mean_rmse_biases, order=2, noise_floor=noise_floor
    )
    c_ground_worst_rmse, slope_ground_worst_rmse = _fixed_order_fit(
        deltas, ground_space_worst_rmse_biases, order=2, noise_floor=noise_floor
    )
    c_qpe_mean, slope_qpe_mean = _fixed_order_fit(
        deltas, qpe_mean_biases, order=2, noise_floor=noise_floor
    )
    c_signal, slope_signal = _fixed_order_fit(
        deltas, signal_biases, order=2, noise_floor=noise_floor
    )
    c_branch, slope_branch = _fixed_order_fit(
        deltas, branch_biases, order=2, noise_floor=noise_floor
    )
    c_cluster_ground, slope_cluster_ground = _fixed_order_fit(
        deltas, target_cluster_ground_biases, order=2, noise_floor=noise_floor
    )
    c_cluster_max_abs, slope_cluster_max_abs = _fixed_order_fit(
        deltas, target_cluster_max_abs_biases, order=2, noise_floor=noise_floor
    )
    return PartialS2CgsFit(
        ld=int(ld),
        lambda_r=float(partition.lambda_r),
        delta_values=deltas,
        qpe_rmse_biases=tuple(qpe_rmse_biases),
        ground_space_mean_rmse_biases=tuple(ground_space_mean_rmse_biases),
        ground_space_worst_rmse_biases=tuple(ground_space_worst_rmse_biases),
        qpe_mean_biases=tuple(qpe_mean_biases),
        qpe_energy_stds=tuple(qpe_energy_stds),
        signal_phase_biases=tuple(signal_biases),
        branch_energy_biases=tuple(branch_biases),
        branch_overlaps=tuple(branch_overlaps),
        target_cluster_ground_biases=tuple(target_cluster_ground_biases),
        target_cluster_max_abs_biases=tuple(target_cluster_max_abs_biases),
        target_cluster_projector_overlaps=tuple(
            target_cluster_projector_overlaps
        ),
        target_cluster_ranks=tuple(target_cluster_ranks),
        target_cluster_ground_state_weights=tuple(
            target_cluster_ground_state_weights
        ),
        unitary_defects=tuple(unitary_defects),
        phase_branch_cut_clearances=tuple(phase_branch_cut_clearances),
        c_gs_qpe_rmse=float(c_qpe_rmse),
        c_gs_ground_space_mean_rmse=float(c_ground_mean_rmse),
        c_gs_ground_space_worst_rmse=float(c_ground_worst_rmse),
        c_gs_qpe_mean=float(c_qpe_mean),
        c_gs_signal=float(c_signal),
        c_gs_branch=float(c_branch),
        c_gs_target_cluster_ground=float(c_cluster_ground),
        c_gs_target_cluster_max_abs=float(c_cluster_max_abs),
        qpe_rmse_fit_slope=slope_qpe_rmse,
        ground_space_mean_rmse_fit_slope=slope_ground_mean_rmse,
        ground_space_worst_rmse_fit_slope=slope_ground_worst_rmse,
        qpe_mean_fit_slope=slope_qpe_mean,
        signal_fit_slope=slope_signal,
        branch_fit_slope=slope_branch,
        target_cluster_ground_fit_slope=slope_cluster_ground,
        target_cluster_max_abs_fit_slope=slope_cluster_max_abs,
        min_branch_overlap=float(min(branch_overlaps)),
        min_target_cluster_projector_overlap=float(
            min(target_cluster_projector_overlaps)
        ),
        min_target_cluster_principal_overlap=float(
            min(target_cluster_principal_overlaps)
        ),
        min_target_cluster_ground_state_weight=float(
            min(target_cluster_ground_state_weights)
        ),
        min_phase_branch_cut_clearance=float(min(phase_branch_cut_clearances)),
    )


def partial_s2_deterministic_step_cost(ld: int, total_terms: int) -> int:
    """Count deterministic Pauli exponentials in S2(..., H_R)."""

    ld = int(ld)
    total_terms = int(total_terms)
    if ld <= 0:
        return 0
    if ld < total_terms:
        return 2 * ld
    return 2 * ld - 1


def optimized_cost_for_cgs(
    *,
    ld: int,
    total_terms: int,
    lambda_r: float,
    c_gs: float,
    epsilon_total: float,
    step_cost_rule: str = "partial_s2",
    randomized_method: str = "qdrift",
    g_rand: float = 1.0,
    error_budget_rule: str = "quadrature",
    max_delta: float | None = None,
) -> dict[str, object]:
    if step_cost_rule == "partial_s2":
        step_cost = partial_s2_deterministic_step_cost(ld, total_terms)
    elif step_cost_rule == "repo_hd_only":
        step_cost = deterministic_step_cost(int(ld), "2nd")
    else:
        raise ValueError("unsupported step_cost_rule")
    if max_delta is None:
        budget = optimize_error_budget_and_kappa(
            epsilon_total=float(epsilon_total),
            order=2,
            deterministic_step_cost_value=int(step_cost),
            c_gs=float(c_gs),
            lambda_r=float(lambda_r),
            randomized_method=randomized_method,
            g_rand=float(g_rand),
            kappa_mode="optimize",
            error_budget_rule=error_budget_rule,
        )
        implied_delta = (
            None
            if step_cost <= 0 or c_gs <= 0.0
            else math.sqrt(float(budget.eps_trot) / float(c_gs))
        )
        return {
            "ld": int(ld),
            "deterministic_step_cost": int(step_cost),
            "lambda_r": float(lambda_r),
            "c_gs": float(c_gs),
            "q_opt": float(budget.q_ratio),
            "eps_qpe": float(budget.eps_qpe),
            "eps_trot": float(budget.eps_trot),
            "kappa_opt": None if budget.kappa is None else float(budget.kappa),
            "g_det": float(budget.g_det),
            "g_rand": float(budget.g_rand),
            "g_total": float(budget.g_total),
            "implied_delta_uncapped": implied_delta,
            "delta_used": implied_delta,
            "max_delta": None,
            "delta_cap_active": False,
            "boundary_hit_q": bool(budget.boundary_hit_q),
            "boundary_hit_kappa": bool(budget.boundary_hit_kappa),
        }

    delta_cap = float(max_delta)
    if delta_cap <= 0.0:
        raise ValueError("max_delta must be positive")
    if error_budget_rule != "quadrature":
        raise ValueError("delta-capped validation currently requires quadrature budget")
    kappa = 2.0 if float(lambda_r) > 0.0 else None
    b_value = (
        randomized_prefactor_B(
            kappa,
            randomized_method=randomized_method,
            g_rand=float(g_rand),
        )
        if kappa is not None
        else 0.0
    )

    def evaluate(q_ratio: float) -> dict[str, float | bool | None]:
        eps_qpe = float(epsilon_total) * float(q_ratio)
        eps_trot = float(epsilon_total) * math.sqrt(
            max(0.0, 1.0 - float(q_ratio) ** 2)
        )
        implied = (
            math.inf
            if float(c_gs) <= 0.0
            else math.sqrt(eps_trot / float(c_gs))
        )
        delta_used = min(implied, delta_cap)
        g_det_value = 0.0
        if step_cost > 0:
            g_det_value = (
                math.inf
                if eps_qpe <= 0.0 or delta_used <= 0.0
                else float(step_cost) / (eps_qpe * delta_used)
            )
        g_rand_value = 0.0
        if float(lambda_r) > 0.0:
            g_rand_value = (
                math.inf
                if eps_qpe <= 0.0
                else b_value * float(lambda_r) ** 2 / eps_qpe**2
            )
        return {
            "q_ratio": float(q_ratio),
            "eps_qpe": eps_qpe,
            "eps_trot": eps_trot,
            "implied_delta_uncapped": None if math.isinf(implied) else implied,
            "delta_used": delta_used,
            "delta_cap_active": bool(implied > delta_cap),
            "g_det": g_det_value,
            "g_rand": g_rand_value,
            "g_total": g_det_value + g_rand_value,
        }

    result = minimize_scalar(
        lambda q_ratio: float(evaluate(float(q_ratio))["g_total"]),
        method="bounded",
        bounds=(PARTIAL_RANDOMIZED_Q_MIN, PARTIAL_RANDOMIZED_Q_MAX),
        options={"xatol": 1e-12},
    )
    q_opt = float(result.x)
    capped = evaluate(q_opt)
    return {
        "ld": int(ld),
        "deterministic_step_cost": int(step_cost),
        "lambda_r": float(lambda_r),
        "c_gs": float(c_gs),
        "q_opt": float(capped["q_ratio"]),
        "eps_qpe": float(capped["eps_qpe"]),
        "eps_trot": float(capped["eps_trot"]),
        "kappa_opt": kappa,
        "g_det": float(capped["g_det"]),
        "g_rand": float(capped["g_rand"]),
        "g_total": float(capped["g_total"]),
        "implied_delta_uncapped": capped["implied_delta_uncapped"],
        "delta_used": float(capped["delta_used"]),
        "max_delta": delta_cap,
        "delta_cap_active": bool(capped["delta_cap_active"]),
        "boundary_hit_q": bool(
            q_opt <= PARTIAL_RANDOMIZED_Q_MIN + 1e-6
            or q_opt >= PARTIAL_RANDOMIZED_Q_MAX - 1e-6
        ),
        "boundary_hit_kappa": False,
    }

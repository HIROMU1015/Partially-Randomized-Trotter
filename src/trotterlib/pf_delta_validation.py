"""Holdout validation of the empirical DF Product Formula error surrogate."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import expm, schur

from .df_hamiltonian import DFHamiltonian, PhysicalSector
from .df_partial_randomized_pf import (
    fit_df_cgs_with_perturbation,
    split_df_hamiltonian_by_ld,
)
from .df_partial_s2 import (
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from .df_trotter.circuit import simulate_statevector
from .finite_rte_signal_validation import (
    _array_fingerprint,
    _circuit_operator_in_openfermion_sector,
    _complex_payload,
    _explicit_cutoff_tolerance,
    _normalized_symbolic_tail_in_sector,
    _qiskit_to_openfermion_sector_permutation,
    dense_df_operator_in_sector,
)
from .rte import require_integer_count


PF_DELTA_VALIDATION_SCHEMA_VERSION = "pf_delta_validation_v5"
PF_DELTA_VALIDATION_METHOD = (
    "cpu_qiskit_exact_tail_partial_s2_paper_d6_single_phase_holdout_v5"
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def paper_d6_perturbative_energy_bias(
    relative_survival_amplitude: complex,
    energy: float,
    delta_time: float,
    *,
    minimum_sine_abs: float = 0.1,
) -> dict[str, Any]:
    """Evaluate paper Eq. (D6) and reject a small sine denominator.

    ``relative_survival_amplitude`` is
    ``exp(i E delta) <psi_0|U_PF(delta)|psi_0>``.  No small-delta fallback is
    substituted when the published denominator is ill-conditioned.
    """
    relative = complex(relative_survival_amplitude)
    energy_value = float(energy)
    delta_value = float(delta_time)
    threshold = float(minimum_sine_abs)
    if not math.isfinite(relative.real) or not math.isfinite(relative.imag):
        raise ValueError("relative_survival_amplitude must be finite.")
    if not math.isfinite(energy_value):
        raise ValueError("energy must be finite.")
    if not math.isfinite(delta_value) or delta_value <= 0.0:
        raise ValueError("delta_time must be finite and positive.")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("minimum_sine_abs must lie strictly in (0, 1).")

    phase = energy_value * delta_value
    sine = float(math.sin(phase))
    sine_abs = float(abs(sine))
    denominator = float(delta_value * sine)
    well_conditioned = bool(sine_abs >= threshold)
    numerator = float(np.real(np.exp(-1j * phase) * (relative - 1.0)))
    signed_bias = None if not well_conditioned else float(numerator / denominator)
    return {
        "definition": "paper Eq. (D6) full-H-ground-state perturbative bias",
        "numerator_real_delta_overlap": numerator,
        "denominator_delta_sine": denominator,
        "sine_denominator_abs": sine_abs,
        "minimum_sine_abs": threshold,
        "well_conditioned": well_conditioned,
        "signed_energy_bias": signed_bias,
        "absolute_energy_bias": (
            None if signed_bias is None else float(abs(signed_bias))
        ),
    }


def _float_grid(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError(f"{name} must not be empty.")
    if any(not math.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError(f"{name} must contain finite positive values.")
    if len(set(grid)) != len(grid):
        raise ValueError(f"{name} must not contain duplicates.")
    return tuple(sorted(grid))


def _q_grid(values: Sequence[int]) -> tuple[int, ...]:
    grid = tuple(
        sorted(
            require_integer_count(value, name="q_values", minimum=1)
            for value in values
        )
    )
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("q_values must be non-empty and unique.")
    if any(value & (value - 1) for value in grid):
        raise ValueError("q_values must contain positive powers of two.")
    return grid


def _spectral_operator(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    evolution_time: float,
) -> np.ndarray:
    phases = np.exp(-1j * float(evolution_time) * eigenvalues)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def _signal(operator: np.ndarray, state: np.ndarray) -> complex:
    return complex(np.vdot(state, operator @ state))


def _phase_distance(left: complex, right: complex) -> float | None:
    if abs(left) == 0.0 or abs(right) == 0.0:
        return None
    return float(abs(np.angle(complex(left) * np.conj(complex(right)))))


def _openfermion_sector_state_to_qiskit_full(
    state: np.ndarray,
    sector: PhysicalSector,
    permutation: np.ndarray,
) -> np.ndarray:
    """Embed an OpenFermion-ordered sector state in Qiskit's bit ordering."""
    sector_state = np.asarray(state, dtype=np.complex128).reshape(-1)
    if sector_state.size != sector.dimension:
        raise ValueError("Sector state dimension does not match the physical sector.")
    qiskit_sector = np.empty_like(sector_state)
    qiskit_sector[np.asarray(permutation, dtype=np.int64)] = sector_state
    full = np.zeros(1 << sector.n_qubits, dtype=np.complex128)
    full[np.asarray(sector.basis_indices, dtype=np.int64)] = qiskit_sector
    return full


def _qiskit_full_state_to_openfermion_sector(
    state: np.ndarray,
    sector: PhysicalSector,
    permutation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Restrict a Qiskit state to the sector and restore OpenFermion ordering."""
    full = np.asarray(state, dtype=np.complex128).reshape(-1)
    expected_dimension = 1 << sector.n_qubits
    if full.size != expected_dimension:
        raise ValueError("Qiskit state dimension does not match the physical sector.")
    inside = np.asarray(sector.basis_indices, dtype=np.int64)
    outside_mask = np.ones(expected_dimension, dtype=bool)
    outside_mask[inside] = False
    leakage_norm = float(np.linalg.norm(full[outside_mask]))
    return (
        np.asarray(full[inside][np.asarray(permutation, dtype=np.int64)]),
        leakage_norm,
    )


def _simulate_qiskit_exact_tail_partial_s2(
    *,
    initial_sector_state: np.ndarray,
    forward_circuit: Any,
    exact_tail_sector_operator: np.ndarray,
    reverse_circuit: Any,
    sector: PhysicalSector,
    permutation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run both PF halves in Qiskit CPU and apply the exact tail in between."""
    initial_full = _openfermion_sector_state_to_qiskit_full(
        initial_sector_state,
        sector,
        permutation,
    )
    after_forward_full = simulate_statevector(forward_circuit, initial_full)
    after_forward_sector, forward_leakage = (
        _qiskit_full_state_to_openfermion_sector(
            after_forward_full,
            sector,
            permutation,
        )
    )
    after_tail_sector = (
        np.asarray(exact_tail_sector_operator, dtype=np.complex128)
        @ after_forward_sector
    )
    after_tail_full = _openfermion_sector_state_to_qiskit_full(
        after_tail_sector,
        sector,
        permutation,
    )
    final_full = simulate_statevector(reverse_circuit, after_tail_full)
    final_sector, reverse_leakage = _qiskit_full_state_to_openfermion_sector(
        final_full,
        sector,
        permutation,
    )
    return final_sector, max(forward_leakage, reverse_leakage)


def _phase_and_perturbative_energy_bias(
    *,
    initial_state: np.ndarray,
    evolved_state: np.ndarray,
    ground_energy: float,
    delta_time: float,
    paper_d6_minimum_sine_abs: float = 1e-6,
) -> dict[str, Any]:
    """Compare exact survival phase with its first-order state-difference proxy."""
    state = np.asarray(initial_state, dtype=np.complex128).reshape(-1)
    evolved = np.asarray(evolved_state, dtype=np.complex128).reshape(-1)
    exact_phase = complex(np.exp(-1j * float(ground_energy) * float(delta_time)))
    overlap = complex(np.vdot(state, evolved))
    relative_overlap = complex(np.exp(1j * ground_energy * delta_time) * overlap)
    signed_phase_bias = float(-np.angle(relative_overlap) / delta_time)
    delta_state = (evolved - exact_phase * state) / (-1j * delta_time)
    signed_perturbative_bias = float(
        np.real(
            np.exp(1j * ground_energy * delta_time)
            * np.vdot(state, delta_state)
        )
    )
    paper_d6 = paper_d6_perturbative_energy_bias(
        relative_overlap,
        ground_energy,
        delta_time,
        minimum_sine_abs=paper_d6_minimum_sine_abs,
    )
    return {
        "relative_survival_amplitude": _complex_payload(relative_overlap),
        "relative_survival_radius": float(abs(relative_overlap)),
        "signed_phase_energy_bias": signed_phase_bias,
        "absolute_phase_energy_bias": float(abs(signed_phase_bias)),
        "signed_linearized_perturbative_energy_bias": signed_perturbative_bias,
        "absolute_linearized_perturbative_energy_bias": float(
            abs(signed_perturbative_bias)
        ),
        "paper_d6_perturbative_energy_bias": paper_d6,
    }


def _qpe_spectral_energy_distribution(
    *,
    unitary: np.ndarray,
    state: np.ndarray,
    target_energy: float,
    delta_time: float,
    cluster_energy_tolerance: float,
    numerical_atol: float,
) -> dict[str, Any]:
    """Resolve the target-state QPE distribution on the target-centered branch."""
    matrix = np.asarray(unitary, dtype=np.complex128)
    psi = np.asarray(state, dtype=np.complex128).reshape(-1)
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    unitary_defect = float(
        np.linalg.norm(matrix.conj().T @ matrix - identity, ord=2)
    )
    triangular, vectors = schur(matrix, output="complex")
    eigenvalues = np.asarray(np.diag(triangular), dtype=np.complex128)
    centered_phases = np.angle(
        np.exp(1j * float(target_energy) * float(delta_time)) * eigenvalues
    )
    energy_shifts = np.asarray(-centered_phases / float(delta_time), dtype=float)
    branch_cut_clearance = float(math.pi - np.max(np.abs(centered_phases)))
    raw_weights = np.asarray(np.abs(vectors.conj().T @ psi) ** 2, dtype=float)
    raw_weight_sum = float(np.sum(raw_weights))
    if not math.isfinite(raw_weight_sum) or raw_weight_sum <= 0.0:
        raise ValueError("The QPE spectral weights have invalid normalization.")
    weights = raw_weights / raw_weight_sum

    signed_mean_bias = float(np.sum(weights * energy_shifts))
    qpe_rmse = float(
        math.sqrt(max(0.0, float(np.sum(weights * energy_shifts**2))))
    )
    qpe_energy_std = float(
        math.sqrt(
            max(
                0.0,
                float(np.sum(weights * (energy_shifts - signed_mean_bias) ** 2)),
            )
        )
    )
    moment_identity_error = float(
        abs(qpe_rmse**2 - (signed_mean_bias**2 + qpe_energy_std**2))
    )
    relative_survival_from_spectrum = complex(
        np.sum(weights * np.exp(-1j * delta_time * energy_shifts))
    )
    relative_survival_direct = complex(
        np.exp(1j * target_energy * delta_time) * np.vdot(psi, matrix @ psi)
    )
    survival_reconstruction_error = float(
        abs(relative_survival_from_spectrum - relative_survival_direct)
    )

    clusters: list[dict[str, Any]] = []
    current_indices: list[int] = []
    for index in np.argsort(energy_shifts):
        integer_index = int(index)
        if current_indices and abs(
            float(energy_shifts[integer_index])
            - float(energy_shifts[current_indices[0]])
        ) > float(cluster_energy_tolerance):
            cluster_weights = weights[current_indices]
            cluster_shifts = energy_shifts[current_indices]
            cluster_weight = float(np.sum(cluster_weights))
            clusters.append(
                {
                    "weight": cluster_weight,
                    "signed_energy_shift": float(
                        np.sum(cluster_weights * cluster_shifts) / cluster_weight
                    ),
                    "minimum_signed_energy_shift": float(np.min(cluster_shifts)),
                    "maximum_signed_energy_shift": float(np.max(cluster_shifts)),
                    "schur_multiplicity": len(current_indices),
                }
            )
            current_indices = []
        current_indices.append(integer_index)
    if current_indices:
        cluster_weights = weights[current_indices]
        cluster_shifts = energy_shifts[current_indices]
        cluster_weight = float(np.sum(cluster_weights))
        clusters.append(
            {
                "weight": cluster_weight,
                "signed_energy_shift": float(
                    np.sum(cluster_weights * cluster_shifts) / cluster_weight
                ),
                "minimum_signed_energy_shift": float(np.min(cluster_shifts)),
                "maximum_signed_energy_shift": float(np.max(cluster_shifts)),
                "schur_multiplicity": len(current_indices),
            }
        )
    clusters.sort(
        key=lambda cluster: (
            -float(cluster["weight"]),
            float(cluster["signed_energy_shift"]),
        )
    )
    for rank, cluster in enumerate(clusters, start=1):
        cluster["weight_rank"] = rank
    dominant_cluster = clusters[0]
    weight_normalization_error = float(abs(np.sum(weights) - 1.0))
    numerical_consistency_pass = bool(
        unitary_defect <= numerical_atol
        and weight_normalization_error <= numerical_atol
        and survival_reconstruction_error <= 10.0 * numerical_atol
        and moment_identity_error <= 10.0 * numerical_atol
        and branch_cut_clearance > numerical_atol
    )
    return {
        "definition": (
            "target-centered Schur eigenphase distribution of one direct-tail "
            "partial-S2 step"
        ),
        "unitary_defect_spectral_norm": unitary_defect,
        "schur_offdiagonal_frobenius_norm": float(
            np.linalg.norm(np.triu(triangular, k=1), ord="fro")
        ),
        "maximum_eigenvalue_modulus_error": float(
            np.max(np.abs(np.abs(eigenvalues) - 1.0))
        ),
        "target_centered_phase_branch_cut_clearance": branch_cut_clearance,
        "raw_weight_sum": raw_weight_sum,
        "weight_normalization_error": weight_normalization_error,
        "signed_qpe_mean_energy_bias": signed_mean_bias,
        "absolute_qpe_mean_energy_bias": float(abs(signed_mean_bias)),
        "qpe_energy_rmse": qpe_rmse,
        "qpe_energy_standard_deviation": qpe_energy_std,
        "qpe_second_moment_identity_error": moment_identity_error,
        "relative_survival_amplitude_from_spectrum": _complex_payload(
            relative_survival_from_spectrum
        ),
        "relative_survival_reconstruction_error": survival_reconstruction_error,
        "phase_cluster_energy_tolerance": float(cluster_energy_tolerance),
        "phase_cluster_count": len(clusters),
        "dominant_phase_cluster_weight": float(dominant_cluster["weight"]),
        "dominant_phase_cluster_signed_energy_bias": float(
            dominant_cluster["signed_energy_shift"]
        ),
        "dominant_phase_cluster_absolute_energy_bias": float(
            abs(float(dominant_cluster["signed_energy_shift"]))
        ),
        "non_dominant_phase_cluster_weight": float(
            max(0.0, 1.0 - float(dominant_cluster["weight"]))
        ),
        "effective_phase_cluster_count": float(
            1.0 / sum(float(cluster["weight"]) ** 2 for cluster in clusters)
        ),
        "phase_clusters": clusters,
        "numerical_consistency_pass": numerical_consistency_pass,
    }


def _fit_fixed_second_order(
    delta_values: Sequence[float],
    biases: Sequence[float],
    *,
    numerical_atol: float,
) -> tuple[float | None, float | None]:
    return _fit_fixed_power(
        delta_values,
        biases,
        fixed_power=2.0,
        numerical_atol=numerical_atol,
    )


def _fit_fixed_power(
    delta_values: Sequence[float],
    values: Sequence[float],
    *,
    fixed_power: float,
    numerical_atol: float,
) -> tuple[float | None, float | None]:
    delta_array = np.asarray(delta_values, dtype=float)
    value_array = np.asarray(values, dtype=float)
    mask = (delta_array > 0.0) & (value_array > numerical_atol)
    if not np.any(mask):
        return None, None
    fixed_coeff = float(
        np.exp(
            np.mean(
                np.log(value_array[mask])
                - float(fixed_power) * np.log(delta_array[mask])
            )
        )
    )
    slope = (
        None
        if np.count_nonzero(mask) < 2
        else float(
            np.polyfit(
                np.log(delta_array[mask]),
                np.log(value_array[mask]),
                1,
            )[0]
        )
    )
    return fixed_coeff, slope


def _state_comparison(
    *,
    label: str,
    state: np.ndarray,
    exact_operator: np.ndarray,
    pf_operator: np.ndarray,
    operator_error: float,
    delta_time: float,
    q_value: int,
    surrogate_coefficient: float,
    beta_pf_budget: float,
    relative_tolerance: float,
    numerical_atol: float,
) -> dict[str, Any]:
    exact_signal = _signal(exact_operator, state)
    pf_signal = _signal(pf_operator, state)
    signal_error = float(abs(pf_signal - exact_signal))
    phase_error = _phase_distance(pf_signal, exact_signal)
    exact_radius = float(abs(exact_signal))
    phase_bound = (
        float(math.asin(operator_error / exact_radius))
        if 0.0 <= operator_error < exact_radius and exact_radius > 0.0
        else None
    )
    predicted_energy_bias = float(surrogate_coefficient * delta_time**2)
    predicted_phase_error = float(
        q_value * delta_time * predicted_energy_bias
    )
    actual_energy_bias = (
        None if phase_error is None else float(phase_error / (q_value * delta_time))
    )
    prediction_relative_error = (
        None
        if actual_energy_bias is None or actual_energy_bias <= numerical_atol
        else float(
            abs(predicted_energy_bias - actual_energy_bias) / actual_energy_bias
        )
    )
    actual_over_prediction = (
        None
        if actual_energy_bias is None or predicted_energy_bias == 0.0
        else float(actual_energy_bias / predicted_energy_bias)
    )
    return {
        "state_label": label,
        "exact_signal": _complex_payload(exact_signal),
        "pf_signal": _complex_payload(pf_signal),
        "exact_signal_radius": exact_radius,
        "pf_signal_radius": float(abs(pf_signal)),
        "pf_signal_error": signal_error,
        "pf_signal_within_operator_error": bool(
            signal_error <= operator_error + numerical_atol
        ),
        "actual_pf_phase_error": phase_error,
        "numerical_operator_phase_bound": phase_bound,
        "operator_phase_bound_applicable": phase_bound is not None,
        "operator_phase_bound_pass": bool(
            phase_bound is not None
            and phase_error is not None
            and phase_error <= phase_bound + numerical_atol
        ),
        "actual_energy_phase_bias": actual_energy_bias,
        "surrogate_predicted_energy_bias": predicted_energy_bias,
        "surrogate_predicted_phase_error": predicted_phase_error,
        "surrogate_prediction_relative_error": prediction_relative_error,
        "actual_over_surrogate_prediction": actual_over_prediction,
        "surrogate_point_prediction_within_tolerance": bool(
            prediction_relative_error is not None
            and prediction_relative_error <= relative_tolerance + numerical_atol
        ),
        "surrogate_underpredicts_actual": bool(
            actual_energy_bias is not None
            and predicted_energy_bias + numerical_atol < actual_energy_bias
        ),
        "actual_pf_phase_within_provisional_budget": bool(
            phase_error is not None
            and phase_error <= beta_pf_budget + numerical_atol
        ),
    }


def validate_pf_delta_grid(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    ld: int = 3,
    surrogate_calibration_times: Sequence[float] = (0.01, 0.02, 0.04, 0.08),
    validation_delta_times: Sequence[float] = (
        0.0125,
        0.025,
        0.05,
        0.1,
        0.2,
        0.4,
    ),
    q_values: Sequence[int] = (1, 2, 4),
    beta_pf_budget: float = 0.08,
    surrogate_relative_tolerance: float = 0.25,
    scaling_slope_interval: tuple[float, float] = (1.5, 2.5),
    coefficient_atol: float = 1e-12,
    seed: int = 20260818,
    matrix_free_backend: str = "python",
    numerical_atol: float = 1e-10,
    perturbative_relative_tolerance: float = 1e-4,
    paper_d6_minimum_sine_abs: float = 1e-6,
    qpe_cluster_energy_tolerance: float = 1e-8,
    minimum_dominant_qpe_cluster_weight: float = 0.9995,
    maximum_single_phase_contamination: float = 8e-4,
    dominant_branch_phase_relative_tolerance: float = 0.02,
    qpe_mean_phase_relative_tolerance: float = 0.01,
    qpe_rmse_phase_relative_tolerance: float = 0.25,
    maximum_dense_reference_qubits: int = 8,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare exact-sector partial-S2 errors with an unused-time surrogate grid."""
    started = time.perf_counter()
    if hamiltonian.n_qubits != sector.n_qubits:
        raise ValueError("Hamiltonian and sector n_qubits differ.")
    ld_value = require_integer_count(ld, name="ld")
    maximum_dense_reference_qubits = require_integer_count(
        maximum_dense_reference_qubits,
        name="maximum_dense_reference_qubits",
        minimum=1,
    )
    if ld_value >= hamiltonian.n_blocks:
        raise ValueError("Validation requires a non-empty randomized tail.")
    calibration = _float_grid(
        surrogate_calibration_times,
        name="surrogate_calibration_times",
    )
    validation = _float_grid(
        validation_delta_times,
        name="validation_delta_times",
    )
    if set(calibration).intersection(validation):
        raise ValueError("Surrogate calibration and validation times must be disjoint.")
    q_grid = _q_grid(q_values)
    for name, value in (
        ("beta_pf_budget", beta_pf_budget),
        ("surrogate_relative_tolerance", surrogate_relative_tolerance),
        ("numerical_atol", numerical_atol),
        ("perturbative_relative_tolerance", perturbative_relative_tolerance),
        ("paper_d6_minimum_sine_abs", paper_d6_minimum_sine_abs),
        ("qpe_cluster_energy_tolerance", qpe_cluster_energy_tolerance),
        ("maximum_single_phase_contamination", maximum_single_phase_contamination),
        (
            "dominant_branch_phase_relative_tolerance",
            dominant_branch_phase_relative_tolerance,
        ),
        ("qpe_mean_phase_relative_tolerance", qpe_mean_phase_relative_tolerance),
        ("qpe_rmse_phase_relative_tolerance", qpe_rmse_phase_relative_tolerance),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if paper_d6_minimum_sine_abs >= 1.0:
        raise ValueError("paper_d6_minimum_sine_abs must be smaller than 1.")
    if not (
        math.isfinite(float(minimum_dominant_qpe_cluster_weight))
        and 0.0 < minimum_dominant_qpe_cluster_weight <= 1.0
    ):
        raise ValueError(
            "minimum_dominant_qpe_cluster_weight must be finite and in (0, 1]."
        )
    if not math.isfinite(float(coefficient_atol)) or coefficient_atol < 0.0:
        raise ValueError("coefficient_atol must be finite and non-negative.")
    slope_min, slope_max = (float(value) for value in scaling_slope_interval)
    if not (
        math.isfinite(slope_min)
        and math.isfinite(slope_max)
        and 0.0 < slope_min < slope_max
    ):
        raise ValueError("scaling_slope_interval must be finite and increasing.")

    dense_started = time.perf_counter()
    dense_hamiltonian = dense_df_operator_in_sector(
        hamiltonian,
        sector,
        matrix_free_backend=matrix_free_backend,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
    spectral_width = float(eigenvalues[-1] - eigenvalues[0])
    physical_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    dense_elapsed = time.perf_counter() - dense_started

    preparation_started = time.perf_counter()
    partition = split_df_hamiltonian_by_ld(hamiltonian, ld_value)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=coefficient_atol,
    )
    permutation = _qiskit_to_openfermion_sector_permutation(sector)
    if preparation.is_deterministic_only:
        normalized_tail = np.zeros(
            (sector.dimension, sector.dimension), dtype=np.complex128
        )
        normalized_tail_leakage = 0.0
    else:
        normalized_tail_result = _normalized_symbolic_tail_in_sector(
            preparation,
            sector,
            permutation,
            max_dense_qubits=maximum_dense_reference_qubits,
        )
        normalized_tail_leakage = normalized_tail_result.leakage_frobenius_norm
        if normalized_tail_leakage > numerical_atol:
            raise ValueError("The retained symbolic RTE tail leaks outside the sector.")
        normalized_tail = normalized_tail_result.matrix
    preparation_elapsed = time.perf_counter() - preparation_started

    surrogate_started = time.perf_counter()
    surrogate = fit_df_cgs_with_perturbation(
        hamiltonian,
        sector,
        partition,
        "2nd",
        t_values=calibration,
        evolution_backend="cpu",
        matrix_free_backend=matrix_free_backend,
        parallel_times=False,
        use_ground_state_cache=False,
        require_usable_estimate=False,
    )
    surrogate_elapsed = time.perf_counter() - surrogate_started

    grid_started = time.perf_counter()
    points: list[dict[str, Any]] = []
    builder = QiskitDFPartialS2CircuitBuilder()
    for delta_time in validation:
        reference_request = make_df_partial_s2_step_request(
            preparation,
            step_time=delta_time,
            rte_steps=1,
            truncation_tolerance=_explicit_cutoff_tolerance(
                preparation.exact_rte_lambda_r * delta_time,
                0,
            ),
            finite_taylor_order=0,
            seed=seed,
        )
        parts = builder.build_additive_circuits(reference_request)
        forward = _circuit_operator_in_openfermion_sector(
            parts.forward_deterministic_half,
            sector,
            permutation,
        )
        reverse = _circuit_operator_in_openfermion_sector(
            parts.reverse_deterministic_half,
            sector,
            permutation,
        )
        maximum_leakage = max(
            forward.leakage_frobenius_norm,
            reverse.leakage_frobenius_norm,
        )
        if maximum_leakage > numerical_atol:
            raise ValueError("Deterministic partial-S2 halves leak outside the sector.")
        exact_tail = expm(
            -1j
            * delta_time
            * preparation.exact_rte_lambda_r
            * normalized_tail
        )
        pf_step = reverse.matrix @ exact_tail @ forward.matrix
        qiskit_evolved_state, qiskit_state_leakage = (
            _simulate_qiskit_exact_tail_partial_s2(
                initial_sector_state=physical_state,
                forward_circuit=parts.forward_deterministic_half,
                exact_tail_sector_operator=exact_tail,
                reverse_circuit=parts.reverse_deterministic_half,
                sector=sector,
                permutation=permutation,
            )
        )
        matrix_evolved_state = pf_step @ physical_state
        qiskit_estimators = _phase_and_perturbative_energy_bias(
            initial_state=physical_state,
            evolved_state=qiskit_evolved_state,
            ground_energy=float(eigenvalues[0]),
            delta_time=delta_time,
            paper_d6_minimum_sine_abs=paper_d6_minimum_sine_abs,
        )
        matrix_estimators = _phase_and_perturbative_energy_bias(
            initial_state=physical_state,
            evolved_state=matrix_evolved_state,
            ground_energy=float(eigenvalues[0]),
            delta_time=delta_time,
            paper_d6_minimum_sine_abs=paper_d6_minimum_sine_abs,
        )
        qpe_spectrum = _qpe_spectral_energy_distribution(
            unitary=pf_step,
            state=physical_state,
            target_energy=float(eigenvalues[0]),
            delta_time=delta_time,
            cluster_energy_tolerance=qpe_cluster_energy_tolerance,
            numerical_atol=numerical_atol,
        )
        paper_d6 = qiskit_estimators["paper_d6_perturbative_energy_bias"]
        paper_d6_bias = paper_d6["absolute_energy_bias"]
        dominant_bias = float(
            qpe_spectrum["dominant_phase_cluster_absolute_energy_bias"]
        )
        paper_d6_vs_dominant_relative_difference = (
            None
            if paper_d6_bias is None
            else float(
                abs(float(paper_d6_bias) - dominant_bias)
                / max(dominant_bias, numerical_atol)
            )
        )
        signed_phase_bias = float(qiskit_estimators["signed_phase_energy_bias"])
        signed_perturbative_bias = float(
            qiskit_estimators["signed_linearized_perturbative_energy_bias"]
        )
        phase_bias = float(qiskit_estimators["absolute_phase_energy_bias"])
        perturbative_bias = float(
            qiskit_estimators["absolute_linearized_perturbative_energy_bias"]
        )
        perturbative_relative_difference = float(
            abs(signed_perturbative_bias - signed_phase_bias)
            / max(abs(signed_phase_bias), numerical_atol)
        )
        signed_qpe_mean_bias = float(qpe_spectrum["signed_qpe_mean_energy_bias"])
        qpe_rmse = float(qpe_spectrum["qpe_energy_rmse"])
        phase_vs_qpe_mean_relative_difference = float(
            abs(signed_phase_bias - signed_qpe_mean_bias)
            / max(
                abs(signed_phase_bias),
                abs(signed_qpe_mean_bias),
                numerical_atol,
            )
        )
        phase_vs_qpe_rmse_relative_difference = float(
            abs(abs(signed_phase_bias) - qpe_rmse)
            / max(qpe_rmse, numerical_atol)
        )
        qpe_spectrum["phase_vs_qpe_mean_relative_difference"] = (
            phase_vs_qpe_mean_relative_difference
        )
        qpe_spectrum["phase_vs_qpe_rmse_relative_difference"] = (
            phase_vs_qpe_rmse_relative_difference
        )
        qpe_spectrum["phase_matches_qpe_mean_within_tolerance"] = bool(
            phase_vs_qpe_mean_relative_difference
            <= qpe_mean_phase_relative_tolerance
        )
        qpe_spectrum["phase_is_qpe_rmse_proxy_within_tolerance"] = bool(
            phase_vs_qpe_rmse_relative_difference
            <= qpe_rmse_phase_relative_tolerance
        )
        qpe_spectrum["dominant_phase_cluster_weight_pass"] = bool(
            float(qpe_spectrum["dominant_phase_cluster_weight"])
            >= minimum_dominant_qpe_cluster_weight - numerical_atol
        )
        non_dominant_weight = float(
            qpe_spectrum["non_dominant_phase_cluster_weight"]
        )
        spectral_phase_contamination_bound = (
            math.pi
            if non_dominant_weight >= 0.5
            else float(
                math.asin(
                    min(
                        1.0,
                        non_dominant_weight / (1.0 - non_dominant_weight),
                    )
                )
            )
        )
        qpe_spectrum["single_phase_signal_radius_lower_bound"] = float(
            max(0.0, 1.0 - 2.0 * non_dominant_weight)
        )
        qpe_spectrum["single_phase_contamination_bound"] = (
            spectral_phase_contamination_bound
        )
        qpe_spectrum["single_phase_contamination_bound_pass"] = bool(
            spectral_phase_contamination_bound
            <= maximum_single_phase_contamination + numerical_atol
        )
        qiskit_matrix_state_error = float(
            np.linalg.norm(qiskit_evolved_state - matrix_evolved_state)
        )
        qiskit_matrix_phase_bias_error = float(
            abs(
                float(qiskit_estimators["signed_phase_energy_bias"])
                - float(matrix_estimators["signed_phase_energy_bias"])
            )
        )
        pf_eigenvalues, pf_eigenvectors = np.linalg.eig(pf_step)
        overlaps = np.abs(pf_eigenvectors.conj().T @ physical_state)
        effective_index = int(np.argmax(overlaps))
        effective_state = np.asarray(
            pf_eigenvectors[:, effective_index],
            dtype=np.complex128,
        )
        effective_state /= np.linalg.norm(effective_state)
        states = {
            "physical_df_ground_state": physical_state,
            "effective_partial_s2_eigenstate_control": effective_state,
        }
        q_results: list[dict[str, Any]] = []
        for q_value in q_grid:
            exact_operator = _spectral_operator(
                eigenvalues,
                eigenvectors,
                q_value * delta_time,
            )
            pf_operator = np.linalg.matrix_power(pf_step, q_value)
            operator_error = float(
                np.linalg.norm(pf_operator - exact_operator, ord=2)
            )
            state_results = [
                _state_comparison(
                    label=label,
                    state=state,
                    exact_operator=exact_operator,
                    pf_operator=pf_operator,
                    operator_error=operator_error,
                    delta_time=delta_time,
                    q_value=q_value,
                    surrogate_coefficient=surrogate.coeff,
                    beta_pf_budget=beta_pf_budget,
                    relative_tolerance=surrogate_relative_tolerance,
                    numerical_atol=numerical_atol,
                )
                for label, state in states.items()
            ]
            relative_physical_signal = complex(
                np.exp(1j * float(eigenvalues[0]) * q_value * delta_time)
                * _signal(pf_operator, physical_state)
            )
            dominant_branch_bias = float(
                qpe_spectrum["dominant_phase_cluster_signed_energy_bias"]
            )
            dominant_branch_signal = complex(
                np.exp(-1j * q_value * delta_time * dominant_branch_bias)
            )
            signal_phase_contamination = float(
                abs(
                    np.angle(
                        relative_physical_signal
                        * np.conj(dominant_branch_signal)
                    )
                )
            )
            signed_signal_energy_bias = float(
                -np.angle(relative_physical_signal) / (q_value * delta_time)
            )
            signal_vs_dominant_relative_difference = float(
                abs(signed_signal_energy_bias - dominant_branch_bias)
                / max(abs(dominant_branch_bias), numerical_atol)
            )
            q_results.append(
                {
                    "q_m": q_value,
                    "t_m": float(q_value * delta_time),
                    "pf_operator_error_spectral_norm": operator_error,
                    "single_dominant_phase_diagnostic": {
                        "relative_physical_pf_signal": _complex_payload(
                            relative_physical_signal
                        ),
                        "relative_physical_pf_signal_radius": float(
                            abs(relative_physical_signal)
                        ),
                        "dominant_branch_relative_signal": _complex_payload(
                            dominant_branch_signal
                        ),
                        "dominant_branch_signed_energy_bias": (
                            dominant_branch_bias
                        ),
                        "signed_signal_energy_bias": signed_signal_energy_bias,
                        "signal_phase_contamination": signal_phase_contamination,
                        "signal_phase_contamination_within_analytic_bound": bool(
                            signal_phase_contamination
                            <= spectral_phase_contamination_bound + numerical_atol
                        ),
                        "signal_vs_dominant_branch_relative_difference": (
                            signal_vs_dominant_relative_difference
                        ),
                        "signal_matches_dominant_branch_within_tolerance": bool(
                            signal_vs_dominant_relative_difference
                            <= dominant_branch_phase_relative_tolerance
                            + numerical_atol
                        ),
                    },
                    "state_results": state_results,
                }
            )
        points.append(
            {
                "delta_time": delta_time,
                "alias_phase_span": float(delta_time * spectral_width),
                "alias_phase_span_le_pi_over_two": bool(
                    delta_time * spectral_width <= math.pi / 2.0 + numerical_atol
                ),
                "deterministic_half_maximum_sector_leakage_frobenius_norm": (
                    maximum_leakage
                ),
                "effective_state_physical_overlap": float(
                    overlaps[effective_index]
                ),
                "effective_state_fingerprint": _array_fingerprint(effective_state),
                "qpe_spectral_energy_distribution": qpe_spectrum,
                "cpu_qiskit_direct_tail_validation": {
                    "execution_model": (
                        "Qiskit Statevector deterministic halves with an exact "
                        "sector expm action for exp(-i H_R delta)"
                    ),
                    "qiskit_statevector_backend": "qiskit.quantum_info.Statevector",
                    "exact_tail_backend": "scipy.linalg.expm_sector_matrix",
                    "maximum_sector_leakage_norm": qiskit_state_leakage,
                    "statevector_l2_error_vs_sector_matrix": (
                        qiskit_matrix_state_error
                    ),
                    "signed_matrix_phase_energy_bias": float(
                        matrix_estimators["signed_phase_energy_bias"]
                    ),
                    "signed_qiskit_phase_energy_bias": float(
                        qiskit_estimators["signed_phase_energy_bias"]
                    ),
                    "absolute_qiskit_phase_energy_bias": phase_bias,
                    "signed_linearized_perturbative_energy_bias": float(
                        qiskit_estimators[
                            "signed_linearized_perturbative_energy_bias"
                        ]
                    ),
                    "absolute_linearized_perturbative_energy_bias": (
                        perturbative_bias
                    ),
                    "paper_d6_perturbative_energy_bias": paper_d6,
                    "paper_d6_vs_dominant_eigenphase_relative_difference": (
                        paper_d6_vs_dominant_relative_difference
                    ),
                    "phase_bias_absolute_error_vs_sector_matrix": (
                        qiskit_matrix_phase_bias_error
                    ),
                    "perturbative_vs_phase_relative_difference": (
                        perturbative_relative_difference
                    ),
                    "relative_survival_amplitude": qiskit_estimators[
                        "relative_survival_amplitude"
                    ],
                    "relative_survival_radius": qiskit_estimators[
                        "relative_survival_radius"
                    ],
                    "qiskit_matrix_consistency_pass": bool(
                        qiskit_state_leakage <= numerical_atol
                        and qiskit_matrix_state_error <= numerical_atol
                        and qiskit_matrix_phase_bias_error <= numerical_atol
                    ),
                    "linearized_perturbation_consistency_pass": bool(
                        perturbative_relative_difference
                        <= perturbative_relative_tolerance
                    ),
                },
                "q_results": q_results,
            }
        )
    grid_elapsed = time.perf_counter() - grid_started

    state_results = [
        state_result
        for point in points
        for q_result in point["q_results"]
        for state_result in q_result["state_results"]
    ]
    physical_results = [
        state_result
        for state_result in state_results
        if state_result["state_label"] == "physical_df_ground_state"
    ]
    physical_q1 = [
        next(
            result
            for result in q_result["state_results"]
            if result["state_label"] == "physical_df_ground_state"
        )
        for point in points
        for q_result in point["q_results"]
        if q_result["q_m"] == 1
    ]
    scaling_values = [
        result["actual_energy_phase_bias"] for result in physical_q1
    ]
    qiskit_validations = [
        point["cpu_qiskit_direct_tail_validation"] for point in points
    ]
    qiskit_phase_biases = [
        result["absolute_qiskit_phase_energy_bias"]
        for result in qiskit_validations
    ]
    perturbative_biases = [
        result["absolute_linearized_perturbative_energy_bias"]
        for result in qiskit_validations
    ]
    paper_d6_conditioned_indices = [
        index
        for index, result in enumerate(qiskit_validations)
        if result["paper_d6_perturbative_energy_bias"]["well_conditioned"]
    ]
    paper_d6_deltas = [validation[index] for index in paper_d6_conditioned_indices]
    paper_d6_biases = [
        float(
            qiskit_validations[index]["paper_d6_perturbative_energy_bias"][
                "absolute_energy_bias"
            ]
        )
        for index in paper_d6_conditioned_indices
    ]
    qpe_spectra = [
        point["qpe_spectral_energy_distribution"] for point in points
    ]
    qpe_mean_biases = [
        result["absolute_qpe_mean_energy_bias"] for result in qpe_spectra
    ]
    qpe_rmse_biases = [result["qpe_energy_rmse"] for result in qpe_spectra]
    qpe_energy_stds = [
        result["qpe_energy_standard_deviation"] for result in qpe_spectra
    ]
    dominant_cluster_biases = [
        result["dominant_phase_cluster_absolute_energy_bias"]
        for result in qpe_spectra
    ]
    paper_d6_reference_dominant_biases = [
        dominant_cluster_biases[index]
        for index in paper_d6_conditioned_indices
    ]
    non_dominant_cluster_weights = [
        result["non_dominant_phase_cluster_weight"]
        for result in qpe_spectra
    ]
    single_phase_q_diagnostics = [
        q_result["single_dominant_phase_diagnostic"]
        for point in points
        for q_result in point["q_results"]
    ]
    qiskit_phase_coefficient, qiskit_phase_slope = _fit_fixed_second_order(
        validation,
        qiskit_phase_biases,
        numerical_atol=numerical_atol,
    )
    perturbative_coefficient, perturbative_slope = _fit_fixed_second_order(
        validation,
        perturbative_biases,
        numerical_atol=numerical_atol,
    )
    paper_d6_coefficient, paper_d6_slope = _fit_fixed_second_order(
        paper_d6_deltas,
        paper_d6_biases,
        numerical_atol=numerical_atol,
    )
    (
        paper_d6_reference_dominant_coefficient,
        paper_d6_reference_dominant_slope,
    ) = _fit_fixed_second_order(
        paper_d6_deltas,
        paper_d6_reference_dominant_biases,
        numerical_atol=numerical_atol,
    )
    qpe_mean_coefficient, qpe_mean_slope = _fit_fixed_second_order(
        validation,
        qpe_mean_biases,
        numerical_atol=numerical_atol,
    )
    qpe_rmse_coefficient, qpe_rmse_slope = _fit_fixed_second_order(
        validation,
        qpe_rmse_biases,
        numerical_atol=numerical_atol,
    )
    qpe_std_coefficient, qpe_std_slope = _fit_fixed_second_order(
        validation,
        qpe_energy_stds,
        numerical_atol=numerical_atol,
    )
    dominant_cluster_coefficient, dominant_cluster_slope = _fit_fixed_second_order(
        validation,
        dominant_cluster_biases,
        numerical_atol=numerical_atol,
    )
    (
        non_dominant_weight_coefficient,
        non_dominant_weight_slope,
    ) = _fit_fixed_power(
        validation,
        non_dominant_cluster_weights,
        fixed_power=4.0,
        numerical_atol=np.finfo(float).eps,
    )
    scaling_slope = (
        None
        if any(value is None or value <= numerical_atol for value in scaling_values)
        else float(
            np.polyfit(
                np.log(np.asarray(validation, dtype=float)),
                np.log(np.asarray(scaling_values, dtype=float)),
                1,
            )[0]
        )
    )
    prediction_errors = [
        result["surrogate_prediction_relative_error"]
        for result in physical_results
        if result["surrogate_prediction_relative_error"] is not None
    ]
    actual_over_predictions = [
        result["actual_over_surrogate_prediction"]
        for result in physical_results
        if result["actual_over_surrogate_prediction"] is not None
    ]
    all_signal_bounds = all(
        result["pf_signal_within_operator_error"] for result in state_results
    )
    all_phase_bounds_applicable = all(
        result["operator_phase_bound_applicable"] for result in state_results
    )
    all_phase_bounds_pass = all(
        result["operator_phase_bound_pass"]
        for result in state_results
        if result["operator_phase_bound_applicable"]
    )
    all_pf_budgets = all(
        result["actual_pf_phase_within_provisional_budget"]
        for result in state_results
    )
    maximum_prediction_error = (
        None if not prediction_errors else float(max(prediction_errors))
    )
    scaling_pass = bool(
        scaling_slope is not None and slope_min <= scaling_slope <= slope_max
    )
    prediction_pass = bool(
        prediction_errors
        and maximum_prediction_error is not None
        and maximum_prediction_error <= surrogate_relative_tolerance
    )
    surrogate_usable = bool(surrogate.metadata.get("screening_usable", False))
    qiskit_perturbation_pass = bool(
        all(result["qiskit_matrix_consistency_pass"] for result in qiskit_validations)
        and all(
            result["linearized_perturbation_consistency_pass"]
            for result in qiskit_validations
        )
    )
    qpe_rmse_scaling_pass = bool(
        qpe_rmse_slope is not None and slope_min <= qpe_rmse_slope <= slope_max
    )
    qpe_spectral_diagnostics_pass = bool(
        qpe_rmse_scaling_pass
        and all(result["numerical_consistency_pass"] for result in qpe_spectra)
        and all(
            result["phase_matches_qpe_mean_within_tolerance"]
            for result in qpe_spectra
        )
        and all(
            result["phase_is_qpe_rmse_proxy_within_tolerance"]
            for result in qpe_spectra
        )
    )
    shift_invariant_dominant_coefficient_relative_difference = (
        None
        if perturbative_coefficient is None
        or dominant_cluster_coefficient is None
        else float(
            abs(perturbative_coefficient - dominant_cluster_coefficient)
            / max(abs(dominant_cluster_coefficient), numerical_atol)
        )
    )
    paper_d6_dominant_coefficient_relative_difference = (
        None
        if paper_d6_coefficient is None
        or paper_d6_reference_dominant_coefficient is None
        else float(
            abs(paper_d6_coefficient - paper_d6_reference_dominant_coefficient)
            / max(abs(paper_d6_reference_dominant_coefficient), numerical_atol)
        )
    )
    paper_d6_estimator_pass = bool(
        len(paper_d6_conditioned_indices) >= 2
        and all(
            qiskit_validations[index][
                "paper_d6_vs_dominant_eigenphase_relative_difference"
            ]
            is not None
            and qiskit_validations[index][
                "paper_d6_vs_dominant_eigenphase_relative_difference"
            ]
            <= dominant_branch_phase_relative_tolerance + numerical_atol
            for index in paper_d6_conditioned_indices
        )
    )
    scalable_primary_coefficient_estimator_pass = bool(
        paper_d6_estimator_pass
        and paper_d6_dominant_coefficient_relative_difference is not None
        and paper_d6_dominant_coefficient_relative_difference
        <= dominant_branch_phase_relative_tolerance + numerical_atol
    )
    single_dominant_phase_approximation_pass = bool(
        all(result["numerical_consistency_pass"] for result in qpe_spectra)
        and all(
            result["dominant_phase_cluster_weight_pass"]
            for result in qpe_spectra
        )
        and all(
            result["single_phase_contamination_bound_pass"]
            for result in qpe_spectra
        )
        and all(
            result["signal_phase_contamination_within_analytic_bound"]
            for result in single_phase_q_diagnostics
        )
        and all(
            result["signal_matches_dominant_branch_within_tolerance"]
            for result in single_phase_q_diagnostics
        )
    )
    single_dominant_phase_cost_model_pass = bool(
        single_dominant_phase_approximation_pass
        and scalable_primary_coefficient_estimator_pass
    )
    overall_pass = bool(
        surrogate_usable
        and scaling_pass
        and prediction_pass
        and all_signal_bounds
        and all_phase_bounds_applicable
        and all_phase_bounds_pass
        and all_pf_budgets
        and qiskit_perturbation_pass
        and single_dominant_phase_cost_model_pass
    )

    payload: dict[str, Any] = {
        "schema_version": PF_DELTA_VALIDATION_SCHEMA_VERSION,
        "validation_method": PF_DELTA_VALIDATION_METHOD,
        "scope": "partial_s2_product_formula_error_and_empirical_surrogate_only",
        "final_cost_evaluation_performed": False,
        "provenance": dict(provenance or {}),
        "request": {
            "ld": ld_value,
            "product_formula": "2nd",
            "surrogate_calibration_times": list(calibration),
            "validation_delta_times": list(validation),
            "calibration_validation_times_disjoint": True,
            "q_values": list(q_grid),
            "beta_pf_budget": float(beta_pf_budget),
            "surrogate_relative_tolerance": float(
                surrogate_relative_tolerance
            ),
            "scaling_slope_interval": [slope_min, slope_max],
            "coefficient_atol": float(coefficient_atol),
            "seed": int(seed),
            "matrix_free_backend": matrix_free_backend,
            "numerical_atol": float(numerical_atol),
            "perturbative_relative_tolerance": float(
                perturbative_relative_tolerance
            ),
            "paper_d6_minimum_sine_abs": float(
                paper_d6_minimum_sine_abs
            ),
            "qpe_cluster_energy_tolerance": float(
                qpe_cluster_energy_tolerance
            ),
            "minimum_dominant_qpe_cluster_weight": float(
                minimum_dominant_qpe_cluster_weight
            ),
            "maximum_single_phase_contamination": float(
                maximum_single_phase_contamination
            ),
            "dominant_branch_phase_relative_tolerance": float(
                dominant_branch_phase_relative_tolerance
            ),
            "qpe_mean_phase_relative_tolerance": float(
                qpe_mean_phase_relative_tolerance
            ),
            "qpe_rmse_phase_relative_tolerance": float(
                qpe_rmse_phase_relative_tolerance
            ),
            "maximum_dense_reference_qubits": maximum_dense_reference_qubits,
        },
        "hamiltonian": {
            "metadata": dict(hamiltonian.metadata),
            "n_qubits": hamiltonian.n_qubits,
            "n_df_blocks": hamiltonian.n_blocks,
            "hamiltonian_hash": preparation.hamiltonian_hash,
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
            "sector_dimension": sector.dimension,
            "sector_n_electrons": sector.n_electrons,
            "ground_energy": float(eigenvalues[0]),
            "spectral_width": spectral_width,
            "df_truncation_value": hamiltonian.metadata.get(
                "df_truncation_value"
            ),
        },
        "partial_s2": {
            "randomized_block_indices": list(
                preparation.randomized_block_indices
            ),
            "deterministic_block_indices": list(
                preparation.deterministic_fragment_indices
            ),
            "exact_rte_lambda_r": float(preparation.exact_rte_lambda_r),
            "threshold_operator_error_bound": float(
                preparation.threshold_operator_error_bound
            ),
            "normalized_tail_sector_leakage_frobenius_norm": (
                normalized_tail_leakage
            ),
        },
        "surrogate": {
            "estimate_kind": surrogate.estimate_kind,
            "is_rigorous_bound": surrogate.is_rigorous_bound,
            "estimator_status": surrogate.estimator_status,
            "screening_usable": surrogate_usable,
            "coefficient": float(surrogate.coeff),
            "fixed_order_coefficient": float(
                surrogate.fit_coeff_fixed_order
            ),
            "free_fit_slope": surrogate.fit_slope,
            "free_fit_coefficient": surrogate.fit_coeff,
            "calibration_energy_biases": list(
                surrogate.perturbation_errors
            ),
            "signed_calibration_energy_biases": list(
                surrogate.signed_phase_biases
            ),
            "relative_overlap_magnitudes": list(
                surrogate.relative_overlap_magnitudes
            ),
            "fit_window_coefficients": surrogate.metadata.get(
                "fit_window_coefficients"
            ),
            "fit_window_relative_spread": surrogate.metadata.get(
                "fit_window_relative_spread"
            ),
            "ground_state_energy": surrogate.metadata.get(
                "ground_state_energy"
            ),
            "ground_state_residual_norm": surrogate.metadata.get(
                "ground_state_residual_norm"
            ),
            "empirical_only_statement": (
                "The fitted coefficient is a screening point predictor, not a "
                "rigorous Product Formula error upper bound."
            ),
        },
        "states": {
            "physical_df_ground_state": {
                "definition": "lowest eigenvector in the full DF sector",
                "state_fingerprint": _array_fingerprint(physical_state),
            },
            "effective_partial_s2_eigenstate_control": {
                "definition": (
                    "per-delta PF-step eigenvector with maximum physical-state overlap"
                ),
            },
        },
        "points": points,
        "summary": {
            "validation_delta_count": len(validation),
            "q_point_count": len(validation) * len(q_grid),
            "state_evaluation_count": len(state_results),
            "all_alias_phase_spans_le_pi_over_two": all(
                point["alias_phase_span_le_pi_over_two"] for point in points
            ),
            "all_signal_errors_within_numerical_operator_error": (
                all_signal_bounds
            ),
            "all_operator_phase_bounds_applicable": all_phase_bounds_applicable,
            "all_operator_phase_bounds_pass": all_phase_bounds_pass,
            "all_actual_pf_phases_within_provisional_budget": all_pf_budgets,
            "physical_q1_energy_bias_scaling_slope": scaling_slope,
            "cpu_qiskit_phase_bias_fixed_second_order_coefficient": (
                qiskit_phase_coefficient
            ),
            "cpu_qiskit_phase_bias_free_fit_slope": qiskit_phase_slope,
            "cpu_linearized_perturbative_fixed_second_order_coefficient": (
                perturbative_coefficient
            ),
            "cpu_linearized_perturbative_free_fit_slope": perturbative_slope,
            "cpu_paper_d6_fixed_second_order_coefficient": (
                paper_d6_coefficient
            ),
            "cpu_paper_d6_free_fit_slope": paper_d6_slope,
            "paper_d6_conditioned_delta_count": len(
                paper_d6_conditioned_indices
            ),
            "paper_d6_ill_conditioned_delta_count": (
                len(validation) - len(paper_d6_conditioned_indices)
            ),
            "paper_d6_reference_dominant_fixed_second_order_coefficient": (
                paper_d6_reference_dominant_coefficient
            ),
            "paper_d6_reference_dominant_free_fit_slope": (
                paper_d6_reference_dominant_slope
            ),
            "qpe_mean_energy_bias_fixed_second_order_coefficient": (
                qpe_mean_coefficient
            ),
            "qpe_mean_energy_bias_free_fit_slope": qpe_mean_slope,
            "qpe_energy_rmse_fixed_second_order_coefficient": (
                qpe_rmse_coefficient
            ),
            "qpe_energy_rmse_free_fit_slope": qpe_rmse_slope,
            "qpe_energy_standard_deviation_fixed_second_order_coefficient": (
                qpe_std_coefficient
            ),
            "qpe_energy_standard_deviation_free_fit_slope": qpe_std_slope,
            "dominant_phase_cluster_bias_fixed_second_order_coefficient": (
                dominant_cluster_coefficient
            ),
            "dominant_phase_cluster_bias_free_fit_slope": (
                dominant_cluster_slope
            ),
            "recommended_pf_coefficient_kind": (
                "dominant_eigenphase_branch_energy_bias"
            ),
            "recommended_pf_fixed_second_order_coefficient": (
                dominant_cluster_coefficient
            ),
            "scalable_pf_coefficient_estimator_kind": (
                "paper_eq_d6_full_h_ground_state_perturbation"
            ),
            "scalable_pf_fixed_second_order_coefficient": (
                paper_d6_coefficient
            ),
            "shift_invariant_coefficient_policy": "diagnostic_only",
            "qpe_rmse_coefficient_policy": (
                "diagnostic_only_not_primary_cost_input"
            ),
            "qpe_rpe_statistical_error_policy": (
                "separate_from_product_formula_coefficient"
            ),
            "non_dominant_phase_cluster_weight_fixed_fourth_order_coefficient": (
                non_dominant_weight_coefficient
            ),
            "non_dominant_phase_cluster_weight_free_fit_slope": (
                non_dominant_weight_slope
            ),
            "minimum_dominant_phase_cluster_weight": float(
                min(
                    result["dominant_phase_cluster_weight"]
                    for result in qpe_spectra
                )
            ),
            "maximum_non_dominant_phase_cluster_weight": float(
                max(
                    result["non_dominant_phase_cluster_weight"]
                    for result in qpe_spectra
                )
            ),
            "minimum_single_phase_signal_radius_lower_bound": float(
                min(
                    result["single_phase_signal_radius_lower_bound"]
                    for result in qpe_spectra
                )
            ),
            "maximum_single_phase_contamination_bound": float(
                max(
                    result["single_phase_contamination_bound"]
                    for result in qpe_spectra
                )
            ),
            "minimum_observed_physical_pf_signal_radius": float(
                min(
                    result["relative_physical_pf_signal_radius"]
                    for result in single_phase_q_diagnostics
                )
            ),
            "maximum_observed_signal_phase_contamination": float(
                max(
                    result["signal_phase_contamination"]
                    for result in single_phase_q_diagnostics
                )
            ),
            "maximum_signal_vs_dominant_branch_relative_difference": float(
                max(
                    result[
                        "signal_vs_dominant_branch_relative_difference"
                    ]
                    for result in single_phase_q_diagnostics
                )
            ),
            "shift_invariant_vs_dominant_branch_coefficient_relative_difference": (
                shift_invariant_dominant_coefficient_relative_difference
            ),
            "paper_d6_vs_dominant_branch_coefficient_relative_difference": (
                paper_d6_dominant_coefficient_relative_difference
            ),
            "maximum_paper_d6_vs_dominant_eigenphase_relative_difference": (
                None
                if not paper_d6_conditioned_indices
                else float(
                    max(
                        qiskit_validations[index][
                            "paper_d6_vs_dominant_eigenphase_relative_difference"
                        ]
                        for index in paper_d6_conditioned_indices
                    )
                )
            ),
            "maximum_phase_vs_qpe_mean_relative_difference": float(
                max(
                    result["phase_vs_qpe_mean_relative_difference"]
                    for result in qpe_spectra
                )
            ),
            "maximum_phase_vs_qpe_rmse_relative_difference": float(
                max(
                    result["phase_vs_qpe_rmse_relative_difference"]
                    for result in qpe_spectra
                )
            ),
            "qpe_rmse_to_phase_fixed_coefficient_ratio": (
                None
                if qpe_rmse_coefficient is None
                or qiskit_phase_coefficient is None
                or qiskit_phase_coefficient == 0.0
                else float(qpe_rmse_coefficient / qiskit_phase_coefficient)
            ),
            "all_qpe_spectral_numerical_consistency_pass": all(
                result["numerical_consistency_pass"] for result in qpe_spectra
            ),
            "all_dominant_phase_cluster_weight_pass": all(
                result["dominant_phase_cluster_weight_pass"]
                for result in qpe_spectra
            ),
            "all_phase_matches_qpe_mean_within_tolerance": all(
                result["phase_matches_qpe_mean_within_tolerance"]
                for result in qpe_spectra
            ),
            "all_phase_is_qpe_rmse_proxy_within_tolerance": all(
                result["phase_is_qpe_rmse_proxy_within_tolerance"]
                for result in qpe_spectra
            ),
            "qpe_rmse_scaling_acceptance_pass": qpe_rmse_scaling_pass,
            "qpe_spectral_diagnostics_pass": qpe_spectral_diagnostics_pass,
            "all_single_phase_contamination_bounds_pass": all(
                result["single_phase_contamination_bound_pass"]
                for result in qpe_spectra
            ),
            "all_signal_phase_contamination_within_analytic_bound": all(
                result["signal_phase_contamination_within_analytic_bound"]
                for result in single_phase_q_diagnostics
            ),
            "all_signals_match_dominant_branch_within_tolerance": all(
                result["signal_matches_dominant_branch_within_tolerance"]
                for result in single_phase_q_diagnostics
            ),
            "single_dominant_phase_approximation_validation_pass": (
                single_dominant_phase_approximation_pass
            ),
            "scalable_primary_coefficient_estimator_validation_pass": (
                scalable_primary_coefficient_estimator_pass
            ),
            "single_dominant_phase_cost_model_validation_pass": (
                single_dominant_phase_cost_model_pass
            ),
            "qpe_spectral_energy_model_validation_pass": (
                single_dominant_phase_cost_model_pass
            ),
            "maximum_qiskit_statevector_l2_error_vs_sector_matrix": float(
                max(
                    result["statevector_l2_error_vs_sector_matrix"]
                    for result in qiskit_validations
                )
            ),
            "maximum_qiskit_phase_bias_absolute_error_vs_sector_matrix": float(
                max(
                    result["phase_bias_absolute_error_vs_sector_matrix"]
                    for result in qiskit_validations
                )
            ),
            "maximum_linearized_perturbative_vs_phase_relative_difference": float(
                max(
                    result["perturbative_vs_phase_relative_difference"]
                    for result in qiskit_validations
                )
            ),
            "all_cpu_qiskit_matrix_consistency_pass": all(
                result["qiskit_matrix_consistency_pass"]
                for result in qiskit_validations
            ),
            "all_linearized_perturbation_consistency_pass": all(
                result["linearized_perturbation_consistency_pass"]
                for result in qiskit_validations
            ),
            "cpu_qiskit_perturbation_validation_pass": qiskit_perturbation_pass,
            "paper_d6_estimator_validation_pass": paper_d6_estimator_pass,
            "scaling_slope_acceptance_pass": scaling_pass,
            "maximum_physical_surrogate_prediction_relative_error": (
                maximum_prediction_error
            ),
            "surrogate_point_prediction_acceptance_pass": prediction_pass,
            "physical_surrogate_underprediction_count": sum(
                result["surrogate_underpredicts_actual"]
                for result in physical_results
            ),
            "physical_surrogate_comparison_count": len(physical_results),
            "maximum_physical_actual_over_surrogate_prediction": (
                None
                if not actual_over_predictions
                else float(max(actual_over_predictions))
            ),
            "candidate_1p2_margin_covers_physical_holdout": bool(
                actual_over_predictions
                and max(actual_over_predictions) <= 1.2 + numerical_atol
            ),
            "surrogate_validated_as_rigorous_upper_bound": False,
            "empirical_screening_validation_pass": overall_pass,
            "overall_pass": overall_pass,
        },
        "performance": {
            "dense_hamiltonian_and_spectrum_seconds": float(dense_elapsed),
            "partial_s2_preparation_seconds": float(preparation_elapsed),
            "surrogate_fit_seconds": float(surrogate_elapsed),
            "holdout_grid_seconds": float(grid_elapsed),
            "total_seconds": float(time.perf_counter() - started),
            "acceleration": (
                "one Hamiltonian eigensystem and one symbolic tail are reused; "
                "q repetitions use matrix powers"
            ),
        },
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_pf_delta_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != PF_DELTA_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Unsupported PF delta validation schema.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("validation_fingerprint must be a SHA-256 hex string.")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("PF delta validation fingerprint mismatch.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")
    surrogate = payload.get("surrogate")
    if (
        not isinstance(surrogate, Mapping)
        or surrogate.get("is_rigorous_bound") is not False
    ):
        raise ValueError("The PF surrogate must remain explicitly empirical.")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("PF delta validation summary is missing.")
    for key in (
        "all_cpu_qiskit_matrix_consistency_pass",
        "all_linearized_perturbation_consistency_pass",
        "cpu_qiskit_perturbation_validation_pass",
        "all_qpe_spectral_numerical_consistency_pass",
        "qpe_spectral_energy_model_validation_pass",
        "single_dominant_phase_approximation_validation_pass",
        "scalable_primary_coefficient_estimator_validation_pass",
        "single_dominant_phase_cost_model_validation_pass",
    ):
        if not isinstance(summary.get(key), bool):
            raise ValueError(f"PF delta validation summary is missing {key}.")
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("PF delta validation points are missing.")
    if any(
        not isinstance(point, Mapping)
        or not isinstance(point.get("cpu_qiskit_direct_tail_validation"), Mapping)
        or not isinstance(point.get("qpe_spectral_energy_distribution"), Mapping)
        or not all(
            isinstance(q_result, Mapping)
            and isinstance(
                q_result.get("single_dominant_phase_diagnostic"), Mapping
            )
            for q_result in point.get("q_results", [])
        )
        for point in points
    ):
        raise ValueError(
            "Every PF delta point must include CPU Qiskit and QPE spectral validation."
        )


def write_pf_delta_validation(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_pf_delta_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

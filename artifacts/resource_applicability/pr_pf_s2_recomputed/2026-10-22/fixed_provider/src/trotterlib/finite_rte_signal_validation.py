"""Small-system validation of finite-RTE signal and attenuation models.

The validation path is intentionally separate from compiled-cost estimation.  It
uses guarded full-system matrices to translate small Qiskit references, then
performs the comparisons in a caller-supplied physical sector.  Independent RTE
trajectories are averaged analytically with the finite Taylor operator, so the
cost does not grow with the number of possible trajectory strings.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from .df_hamiltonian import DFHamiltonian, PhysicalSector, df_linear_operator
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import (
    DFPartialS2Preparation,
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from .df_rte_tail import basis_change_unitary
from .rte import (
    compose_truncation_residual_bounds,
    finite_rte_attenuation,
    finite_rte_corrected_operator,
    make_rte_config,
    require_integer_count,
    step_taylor_truncation_residual_bound,
)


FINITE_RTE_SIGNAL_VALIDATION_SCHEMA_VERSION = (
    "finite_rte_signal_validation_v1"
)
FINITE_RTE_SIGNAL_VALIDATION_METHOD = (
    "sector_restricted_finite_taylor_operator_moments_v1"
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


def _complex_payload(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def _array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.complex128))
    return hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest()


def _normalized_unique_counts(
    values: Sequence[int],
    *,
    name: str,
    minimum: int,
    even: bool = False,
) -> tuple[int, ...]:
    normalized = tuple(
        require_integer_count(value, name=name, minimum=minimum) for value in values
    )
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    if even and any(value % 2 for value in normalized):
        raise ValueError(f"{name} must contain only even values.")
    return tuple(sorted(normalized))


def dense_df_operator_in_sector(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    matrix_free_backend: str = "python",
) -> np.ndarray:
    """Materialize a guarded small-sector DF operator once for validation."""
    if hamiltonian.n_qubits != sector.n_qubits:
        raise ValueError("Hamiltonian and sector n_qubits differ.")
    operator, _counter = df_linear_operator(
        hamiltonian,
        sector,
        backend=matrix_free_backend,
    )
    identity = np.eye(sector.dimension, dtype=np.complex128)
    return np.column_stack(
        [operator @ identity[:, column] for column in range(sector.dimension)]
    )


def _bit_reverse(value: int, width: int) -> int:
    return int(f"{int(value):0{int(width)}b}"[::-1], 2)


def _qiskit_to_openfermion_sector_permutation(
    sector: PhysicalSector,
) -> np.ndarray:
    positions = {
        int(basis_index): position
        for position, basis_index in enumerate(sector.basis_indices)
    }
    reversed_positions: list[int] = []
    for basis_index in sector.basis_indices:
        reversed_index = _bit_reverse(int(basis_index), sector.n_qubits)
        if reversed_index not in positions:
            raise ValueError(
                "The selected sector is not invariant under the Qiskit/OpenFermion "
                "bit-order conversion. Use a complete particle-number sector."
            )
        reversed_positions.append(positions[reversed_index])
    return np.asarray(reversed_positions, dtype=np.int64)


@dataclass(frozen=True)
class _SectorCircuitOperator:
    matrix: np.ndarray
    leakage_frobenius_norm: float


def _matrix_in_openfermion_sector(
    full: np.ndarray,
    sector: PhysicalSector,
    permutation: np.ndarray,
) -> _SectorCircuitOperator:
    full = np.asarray(full, dtype=np.complex128)
    expected_dimension = 1 << sector.n_qubits
    if full.shape != (expected_dimension, expected_dimension):
        raise ValueError("Full operator dimension does not match the sector.")
    inside = np.asarray(sector.basis_indices, dtype=np.int64)
    outside_mask = np.ones(expected_dimension, dtype=bool)
    outside_mask[inside] = False
    outside = np.flatnonzero(outside_mask)
    leakage = float(np.linalg.norm(full[np.ix_(outside, inside)], ord="fro"))
    qiskit_sector = full[np.ix_(inside, inside)]
    openfermion_sector = qiskit_sector[np.ix_(permutation, permutation)]
    return _SectorCircuitOperator(openfermion_sector, leakage)


def _circuit_operator_in_openfermion_sector(
    circuit: Any,
    sector: PhysicalSector,
    permutation: np.ndarray,
) -> _SectorCircuitOperator:
    return _matrix_in_openfermion_sector(
        np.asarray(Operator(circuit).data, dtype=np.complex128),
        sector,
        permutation,
    )


def _normalized_symbolic_tail_in_sector(
    preparation: DFPartialS2Preparation,
    sector: PhysicalSector,
    permutation: np.ndarray,
    *,
    max_dense_qubits: int = 8,
) -> _SectorCircuitOperator:
    """Materialize the retained symbolic tail, grouping equal DF bases.

    Combining all diagonal I/Z/ZZ components in one basis before conjugation
    needs one dense multiplication per DF basis, rather than one per component.
    This also faithfully excludes components removed by ``coefficient_atol``.
    """
    extraction = preparation.tail_extraction
    if extraction.rte_lambda_r <= 0.0:
        raise ValueError("Validation requires a non-empty randomized tail.")
    full_dimension = 1 << sector.n_qubits
    basis_states = np.arange(full_dimension, dtype=np.uint64)
    diagonal_by_basis: dict[str, np.ndarray] = {}
    for component in extraction.components:
        diagonal = diagonal_by_basis.setdefault(
            component.basis_id,
            np.zeros(full_dimension, dtype=np.float64),
        )
        signs = np.ones(full_dimension, dtype=np.float64)
        parity = np.zeros(full_dimension, dtype=np.uint64)
        for qubit in component.diagonal_pauli_support:
            parity ^= (basis_states >> int(qubit)) & 1
        signs[parity.astype(bool)] = -1.0
        diagonal += float(component.coefficient) * signs

    full = np.zeros((full_dimension, full_dimension), dtype=np.complex128)
    for basis_id, diagonal in diagonal_by_basis.items():
        basis = basis_change_unitary(
            extraction.basis_definition(basis_id),
            max_dense_qubits=max_dense_qubits,
        )
        full += (basis * diagonal[np.newaxis, :]) @ basis.conj().T
    full /= preparation.exact_rte_lambda_r
    return _matrix_in_openfermion_sector(full, sector, permutation)


def _phase_distance(left: complex, right: complex) -> float | None:
    if abs(left) == 0.0 or abs(right) == 0.0:
        return None
    return float(abs(np.angle(complex(left) * np.conj(complex(right)))))


def _hoeffding_shots(
    *,
    attenuation: float,
    reference_radius: float,
    signal_error_bound: float,
    beta_stat_budget: float,
    alpha_axis: float,
) -> int | None:
    radius_lower = attenuation * (reference_radius - signal_error_bound)
    if radius_lower <= 0.0:
        return None
    coordinate = radius_lower * math.sin(beta_stat_budget) / math.sqrt(2.0)
    if coordinate <= 0.0:
        return None
    return int(math.ceil(2.0 / (coordinate * coordinate) * math.log(2.0 / alpha_axis)))


def _explicit_cutoff_tolerance(
    dimensionless_step_time: float,
    finite_taylor_order: int,
) -> float:
    """Return a positive tolerance that admits one explicitly requested cutoff."""
    residual = step_taylor_truncation_residual_bound(
        dimensionless_step_time,
        finite_taylor_order,
    )
    if not math.isfinite(residual):
        raise ValueError("The explicit finite Taylor residual overflowed.")
    if residual == 0.0:
        return float(np.nextafter(0.0, 1.0))
    return float(math.nextafter(residual, math.inf))


def _spectral_operator(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    evolution_time: float,
) -> np.ndarray:
    phases = np.exp(-1j * float(evolution_time) * eigenvalues)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def _signal(operator: np.ndarray, state: np.ndarray) -> complex:
    return complex(np.vdot(state, operator @ state))


def _state_result(
    *,
    label: str,
    state: np.ndarray,
    exact_operator: np.ndarray,
    pf_operator: np.ndarray,
    corrected_operator: np.ndarray,
    attenuated_operator: np.ndarray,
    attenuation: float,
    signal_error_bound: float,
    beta_pf_budget: float,
    beta_rte_budget: float,
    beta_stat_budget: float,
    alpha_axis: float,
    numerical_atol: float,
) -> dict[str, Any]:
    exact_signal = _signal(exact_operator, state)
    pf_signal = _signal(pf_operator, state)
    corrected_signal = _signal(corrected_operator, state)
    attenuated_signal = _signal(attenuated_operator, state)
    actual_error = float(abs(corrected_signal - pf_signal))
    reference_radius = float(abs(pf_signal))
    corrected_radius = float(abs(corrected_signal))
    observed_radius = float(abs(attenuated_signal))
    model_radius = float(attenuation * reference_radius)
    conservative_radius = float(
        attenuation * max(0.0, reference_radius - signal_error_bound)
    )
    phase_error = _phase_distance(corrected_signal, pf_signal)
    pf_phase_error = _phase_distance(pf_signal, exact_signal)
    phase_bound = (
        float(math.asin(signal_error_bound / reference_radius))
        if 0.0 <= signal_error_bound < reference_radius and reference_radius > 0.0
        else None
    )
    unit_shots = _hoeffding_shots(
        attenuation=attenuation,
        reference_radius=1.0,
        signal_error_bound=signal_error_bound,
        beta_stat_budget=beta_stat_budget,
        alpha_axis=alpha_axis,
    )
    actual_radius_shots = _hoeffding_shots(
        attenuation=attenuation,
        reference_radius=reference_radius,
        signal_error_bound=signal_error_bound,
        beta_stat_budget=beta_stat_budget,
        alpha_axis=alpha_axis,
    )
    shot_relative_difference = (
        None
        if unit_shots is None or actual_radius_shots is None or unit_shots == 0
        else float(abs(actual_radius_shots - unit_shots) / unit_shots)
    )
    return {
        "state_label": label,
        "exact_signal": _complex_payload(exact_signal),
        "pf_signal": _complex_payload(pf_signal),
        "corrected_finite_rte_signal": _complex_payload(corrected_signal),
        "attenuated_event_mean_signal": _complex_payload(attenuated_signal),
        "pf_vs_exact_signal_error": float(abs(pf_signal - exact_signal)),
        "pf_vs_exact_phase_error": pf_phase_error,
        "pf_phase_error_within_provisional_budget": bool(
            pf_phase_error is None
            or pf_phase_error <= beta_pf_budget + numerical_atol
        ),
        "finite_rte_signal_error": actual_error,
        "finite_rte_signal_bound_pass": bool(
            actual_error <= signal_error_bound + numerical_atol
        ),
        "reference_signal_radius": reference_radius,
        "reference_radius_deviation_from_one": float(abs(reference_radius - 1.0)),
        "corrected_finite_rte_radius": corrected_radius,
        "observed_attenuated_radius": observed_radius,
        "simplified_radius_model": model_radius,
        "conservative_radius_lower_bound": conservative_radius,
        "conservative_radius_bound_pass": bool(
            observed_radius + numerical_atol >= conservative_radius
        ),
        "radius_model_absolute_error": float(abs(observed_radius - model_radius)),
        "finite_rte_phase_error": phase_error,
        "finite_rte_phase_error_bound": phase_bound,
        "finite_rte_phase_bound_applicable": phase_bound is not None,
        "finite_rte_phase_bound_pass": bool(
            phase_bound is not None
            and phase_error is not None
            and phase_error <= phase_bound + numerical_atol
        ),
        "finite_rte_phase_bound_within_provisional_budget": bool(
            phase_bound is not None
            and phase_bound <= beta_rte_budget + numerical_atol
        ),
        "provisional_shots_unit_radius_per_axis": unit_shots,
        "provisional_shots_reference_radius_per_axis": actual_radius_shots,
        "provisional_shot_relative_difference": shot_relative_difference,
    }


def validate_finite_rte_signals(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    ld: int,
    delta_time: float | None = None,
    maximum_delta_time: float = 0.1,
    q_values: Sequence[int] = (1, 2, 4),
    rte_step_values: Sequence[int] = (1, 2, 4, 8),
    finite_taylor_orders: Sequence[int] = (0, 2, 4, 6),
    beta_rpe: float = 0.4,
    beta_pf_budget: float = 0.08,
    beta_rte_budget: float = 0.08,
    beta_stat_budget: float = 0.24,
    alpha_total: float = 0.05,
    seed: int = 20260818,
    coefficient_atol: float = 1e-12,
    matrix_free_backend: str = "python",
    numerical_atol: float = 1e-10,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate finite-RTE bounds and signal-radius models on a small sector.

    The returned payload is versioned and fingerprinted.  It contains no final
    compiled-cost claim and does not optimize any search variable.
    """
    started = time.perf_counter()
    if hamiltonian.n_qubits != sector.n_qubits:
        raise ValueError("Hamiltonian and sector n_qubits differ.")
    ld_value = require_integer_count(ld, name="ld")
    if ld_value >= hamiltonian.n_blocks:
        raise ValueError("Validation requires a non-empty randomized tail.")
    q_grid = _normalized_unique_counts(q_values, name="q_values", minimum=1)
    if any(value & (value - 1) for value in q_grid):
        raise ValueError("q_values must contain positive powers of two.")
    r_grid = _normalized_unique_counts(
        rte_step_values,
        name="rte_step_values",
        minimum=1,
    )
    k_grid = _normalized_unique_counts(
        finite_taylor_orders,
        name="finite_taylor_orders",
        minimum=0,
        even=True,
    )
    for name, value in (
        ("beta_rpe", beta_rpe),
        ("beta_pf_budget", beta_pf_budget),
        ("beta_rte_budget", beta_rte_budget),
        ("beta_stat_budget", beta_stat_budget),
        ("alpha_total", alpha_total),
        ("numerical_atol", numerical_atol),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(float(coefficient_atol)) or coefficient_atol < 0.0:
        raise ValueError("coefficient_atol must be finite and non-negative.")
    if beta_pf_budget + beta_rte_budget + beta_stat_budget > beta_rpe + 1e-15:
        raise ValueError("Provisional phase budgets exceed beta_rpe.")
    if not 0.0 < alpha_total < 1.0:
        raise ValueError("alpha_total must lie strictly in (0, 1).")
    if maximum_delta_time <= 0.0 or not math.isfinite(maximum_delta_time):
        raise ValueError("maximum_delta_time must be finite and positive.")

    dense_started = time.perf_counter()
    dense_hamiltonian = dense_df_operator_in_sector(
        hamiltonian,
        sector,
        matrix_free_backend=matrix_free_backend,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
    spectral_width = float(eigenvalues[-1] - eigenvalues[0])
    if delta_time is None:
        selected_delta = float(
            min(
                maximum_delta_time,
                math.inf if spectral_width == 0.0 else math.pi / (2.0 * spectral_width),
            )
        )
        delta_selection = "min(maximum_delta_time,pi/(2*spectral_width))"
    else:
        selected_delta = float(delta_time)
        delta_selection = "explicit"
    if not math.isfinite(selected_delta) or selected_delta <= 0.0:
        raise ValueError("delta_time must be finite and positive.")
    dense_elapsed = time.perf_counter() - dense_started

    preparation_started = time.perf_counter()
    partition = split_df_hamiltonian_by_ld(hamiltonian, ld_value)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=coefficient_atol,
    )
    identity = np.eye(sector.dimension, dtype=np.complex128)
    permutation = _qiskit_to_openfermion_sector_permutation(sector)
    normalized_tail_result = _normalized_symbolic_tail_in_sector(
        preparation,
        sector,
        permutation,
    )
    normalized_tail = normalized_tail_result.matrix
    if normalized_tail_result.leakage_frobenius_norm > numerical_atol:
        raise ValueError("The retained symbolic RTE tail leaks outside the sector.")

    reference_request = make_df_partial_s2_step_request(
        preparation,
        step_time=selected_delta,
        rte_steps=1,
        truncation_tolerance=_explicit_cutoff_tolerance(
            preparation.exact_rte_lambda_r * selected_delta,
            0,
        ),
        finite_taylor_order=0,
        seed=seed,
    )
    parts = QiskitDFPartialS2CircuitBuilder().build_additive_circuits(
        reference_request
    )
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
    if max(forward.leakage_frobenius_norm, reverse.leakage_frobenius_norm) > (
        numerical_atol
    ):
        raise ValueError("Deterministic partial-S2 halves leak outside the sector.")
    exact_tail_operator = expm(
        -1j * selected_delta * preparation.exact_rte_lambda_r * normalized_tail
    )
    pf_step = reverse.matrix @ exact_tail_operator @ forward.matrix
    preparation_elapsed = time.perf_counter() - preparation_started

    physical_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    pf_eigenvalues, pf_eigenvectors = np.linalg.eig(pf_step)
    overlaps = np.abs(pf_eigenvectors.conj().T @ physical_state)
    effective_index = int(np.argmax(overlaps))
    effective_state = np.asarray(
        pf_eigenvectors[:, effective_index],
        dtype=np.complex128,
    )
    effective_state /= np.linalg.norm(effective_state)
    effective_eigen_residual = float(
        np.linalg.norm(
            pf_step @ effective_state
            - pf_eigenvalues[effective_index] * effective_state
        )
    )
    states = {
        "physical_df_ground_state": physical_state,
        "effective_partial_s2_eigenstate_control": effective_state,
    }

    max_round_index = max(int(round(math.log2(q_value))) for q_value in q_grid)
    alpha_axis = float(alpha_total / (2 * (max_round_index + 1)))
    exact_by_q: dict[int, np.ndarray] = {}
    pf_by_q: dict[int, np.ndarray] = {}
    q_metadata: list[dict[str, Any]] = []
    for q_value in q_grid:
        exact_operator = _spectral_operator(
            eigenvalues,
            eigenvectors,
            q_value * selected_delta,
        )
        pf_operator = np.linalg.matrix_power(pf_step, q_value)
        exact_by_q[q_value] = exact_operator
        pf_by_q[q_value] = pf_operator
        q_metadata.append(
            {
                "q_m": q_value,
                "round_index": int(round(math.log2(q_value))),
                "t_m": float(q_value * selected_delta),
                "pf_operator_error_spectral_norm": float(
                    np.linalg.norm(pf_operator - exact_operator, ord=2)
                ),
                "state_reference_radii": {
                    label: float(abs(_signal(pf_operator, state)))
                    for label, state in states.items()
                },
            }
        )

    grid_started = time.perf_counter()
    points: list[dict[str, Any]] = []
    component_count = len(preparation.rte_preparation.symbolic_tail.components)
    for q_value in q_grid:
        pf_operator = pf_by_q[q_value]
        exact_operator = exact_by_q[q_value]
        for rte_steps in r_grid:
            for cutoff in k_grid:
                config, distribution = make_rte_config(
                    preparation.rte_preparation.symbolic_tail,
                    evolution_time=selected_delta,
                    rte_steps=rte_steps,
                    truncation_tolerance=_explicit_cutoff_tolerance(
                        preparation.exact_rte_lambda_r
                        * selected_delta
                        / rte_steps,
                        cutoff,
                    ),
                    finite_taylor_order=cutoff,
                    seed=seed,
                )
                corrected_tail = finite_rte_corrected_operator(
                    normalized_tail,
                    config,
                )
                corrected_step = reverse.matrix @ corrected_tail @ forward.matrix
                corrected_round = np.linalg.matrix_power(corrected_step, q_value)
                attenuation = finite_rte_attenuation(
                    config,
                    tail_evolutions=q_value,
                )
                attenuated_round = attenuation * corrected_round
                epsilon_z = compose_truncation_residual_bounds(
                    (
                        (
                            distribution.step_truncation_residual_bound,
                            rte_steps,
                            q_value,
                        ),
                    )
                )
                operator_error = float(
                    np.linalg.norm(corrected_round - pf_operator, ord=2)
                )
                event_count = sum(
                    component_count ** (order + 1)
                    for order in distribution.orders
                )
                log10_trajectory_count = float(
                    rte_steps * q_value * math.log10(event_count)
                )
                state_results = [
                    _state_result(
                        label=label,
                        state=state,
                        exact_operator=exact_operator,
                        pf_operator=pf_operator,
                        corrected_operator=corrected_round,
                        attenuated_operator=attenuated_round,
                        attenuation=attenuation,
                        signal_error_bound=epsilon_z,
                        beta_pf_budget=beta_pf_budget,
                        beta_rte_budget=beta_rte_budget,
                        beta_stat_budget=beta_stat_budget,
                        alpha_axis=alpha_axis,
                        numerical_atol=numerical_atol,
                    )
                    for label, state in states.items()
                ]
                points.append(
                    {
                        "q_m": q_value,
                        "round_index": int(round(math.log2(q_value))),
                        "r_m": rte_steps,
                        "K_m": cutoff,
                        "tau_m": float(config.dimensionless_step_time),
                        "finite_distribution_normalization": float(
                            distribution.exact_finite_distribution
                        ),
                        "attenuation": float(attenuation),
                        "step_truncation_residual_bound": float(
                            distribution.step_truncation_residual_bound
                        ),
                        "round_signal_error_bound": float(epsilon_z),
                        "corrected_operator_error_spectral_norm": operator_error,
                        "operator_error_bound_pass": bool(
                            operator_error <= epsilon_z + numerical_atol
                        ),
                        "attenuation_scaling_error_frobenius_norm": float(
                            np.linalg.norm(
                                attenuated_round - attenuation * corrected_round,
                                ord="fro",
                            )
                        ),
                        "one_short_step_event_count": int(event_count),
                        "explicit_round_trajectory_count_log10": (
                            log10_trajectory_count
                        ),
                        "state_results": state_results,
                    }
                )
    grid_elapsed = time.perf_counter() - grid_started

    all_operator_bounds = all(point["operator_error_bound_pass"] for point in points)
    all_signal_bounds = all(
        state_result["finite_rte_signal_bound_pass"]
        for point in points
        for state_result in point["state_results"]
    )
    all_radius_bounds = all(
        state_result["conservative_radius_bound_pass"]
        for point in points
        for state_result in point["state_results"]
    )
    applicable_phase_results = [
        state_result
        for point in points
        for state_result in point["state_results"]
        if state_result["finite_rte_phase_bound_applicable"]
    ]
    all_applicable_phase_bounds = all(
        state_result["finite_rte_phase_bound_pass"]
        for state_result in applicable_phase_results
    )
    all_pf_budgets = all(
        state_result["pf_phase_error_within_provisional_budget"]
        for point in points
        for state_result in point["state_results"]
    )
    all_rte_budgets = all(
        state_result["finite_rte_phase_bound_within_provisional_budget"]
        for point in points
        for state_result in point["state_results"]
    )
    physical_results = [
        state_result
        for point in points
        for state_result in point["state_results"]
        if state_result["state_label"] == "physical_df_ground_state"
    ]
    maximum_physical_radius_deviation = max(
        result["reference_radius_deviation_from_one"] for result in physical_results
    )
    shot_differences = [
        result["provisional_shot_relative_difference"]
        for result in physical_results
        if result["provisional_shot_relative_difference"] is not None
    ]
    maximum_shot_relative_difference = max(shot_differences, default=0.0)
    general_radius_recommended = bool(
        maximum_physical_radius_deviation > 0.01
        or maximum_shot_relative_difference > 0.05
    )
    provisional_phase_feasible_points = sum(
        all(
            state_result["pf_phase_error_within_provisional_budget"]
            and state_result[
                "finite_rte_phase_bound_within_provisional_budget"
            ]
            for state_result in point["state_results"]
        )
        for point in points
    )

    payload: dict[str, Any] = {
        "schema_version": FINITE_RTE_SIGNAL_VALIDATION_SCHEMA_VERSION,
        "validation_method": FINITE_RTE_SIGNAL_VALIDATION_METHOD,
        "scope": "finite_rte_signal_error_attenuation_and_radius_only",
        "final_cost_evaluation_performed": False,
        "provenance": dict(provenance or {}),
        "request": {
            "ld": ld_value,
            "delta_time": selected_delta,
            "delta_time_selection": delta_selection,
            "maximum_delta_time": float(maximum_delta_time),
            "q_values": list(q_grid),
            "rte_step_values": list(r_grid),
            "finite_taylor_orders": list(k_grid),
            "identity_policy": "extract_identity_phase",
            "coefficient_atol": float(coefficient_atol),
            "matrix_free_backend": matrix_free_backend,
            "seed": int(seed),
            "numerical_atol": float(numerical_atol),
            "provisional_allocation": {
                "beta_rpe": float(beta_rpe),
                "beta_pf_budget": float(beta_pf_budget),
                "beta_rte_budget": float(beta_rte_budget),
                "beta_stat_budget": float(beta_stat_budget),
                "alpha_total": float(alpha_total),
                "alpha_axis_for_validated_round_set": alpha_axis,
            },
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
            "sector_nelec_alpha": sector.nelec_alpha,
            "sector_nelec_beta": sector.nelec_beta,
            "ground_energy": float(eigenvalues[0]),
            "spectral_width": spectral_width,
            "alias_phase_span": float(selected_delta * spectral_width),
            "alias_phase_span_le_pi_over_two": bool(
                selected_delta * spectral_width <= math.pi / 2.0 + numerical_atol
            ),
            "df_truncation_value": hamiltonian.metadata.get(
                "df_truncation_value"
            ),
        },
        "partial_s2": {
            "randomized_block_indices": list(preparation.randomized_block_indices),
            "deterministic_block_indices": list(
                preparation.deterministic_fragment_indices
            ),
            "randomized_component_count": component_count,
            "ranking_proxy_lambda_r": float(preparation.ranking_proxy_lambda_r),
            "exact_rte_lambda_r": float(preparation.exact_rte_lambda_r),
            "extracted_identity_coefficient": float(
                preparation.extracted_identity_coefficient
            ),
            "threshold_operator_error_bound": float(
                preparation.threshold_operator_error_bound
            ),
            "threshold_dropped_component_count": (
                preparation.threshold_dropped_component_count
            ),
            "threshold_dropped_coefficient_l1": float(
                preparation.threshold_dropped_coefficient_l1
            ),
            "normalized_tail_sector_leakage_frobenius_norm": (
                normalized_tail_result.leakage_frobenius_norm
            ),
            "forward_sector_leakage_frobenius_norm": (
                forward.leakage_frobenius_norm
            ),
            "reverse_sector_leakage_frobenius_norm": (
                reverse.leakage_frobenius_norm
            ),
            "forward_unitarity_error_frobenius_norm": float(
                np.linalg.norm(forward.matrix.conj().T @ forward.matrix - identity)
            ),
            "reverse_unitarity_error_frobenius_norm": float(
                np.linalg.norm(reverse.matrix.conj().T @ reverse.matrix - identity)
            ),
        },
        "states": {
            "physical_df_ground_state": {
                "definition": "lowest eigenvector in the selected DF sector",
                "state_fingerprint": _array_fingerprint(physical_state),
            },
            "effective_partial_s2_eigenstate_control": {
                "definition": "PF-step eigenvector with maximum physical-state overlap",
                "state_fingerprint": _array_fingerprint(effective_state),
                "selected_eigenvalue": _complex_payload(
                    pf_eigenvalues[effective_index]
                ),
                "physical_state_overlap": float(overlaps[effective_index]),
                "eigen_residual_norm": effective_eigen_residual,
            },
        },
        "q_metadata": q_metadata,
        "points": points,
        "summary": {
            "point_count": len(points),
            "state_evaluation_count": 2 * len(points),
            "all_operator_error_bounds_pass": all_operator_bounds,
            "all_signal_error_bounds_pass": all_signal_bounds,
            "all_conservative_radius_bounds_pass": all_radius_bounds,
            "phase_bound_applicable_state_count": len(applicable_phase_results),
            "phase_bound_inapplicable_state_count": (
                2 * len(points) - len(applicable_phase_results)
            ),
            "all_phase_error_bounds_applicable": bool(
                len(applicable_phase_results) == 2 * len(points)
            ),
            "all_applicable_phase_error_bounds_pass": (
                all_applicable_phase_bounds
            ),
            "all_pf_phase_errors_within_provisional_budget": all_pf_budgets,
            "all_rte_phase_bounds_within_provisional_budget": all_rte_budgets,
            "provisional_phase_feasible_point_count": (
                provisional_phase_feasible_points
            ),
            "overall_pass": bool(
                all_operator_bounds
                and all_signal_bounds
                and all_radius_bounds
                and all_applicable_phase_bounds
            ),
            "maximum_physical_reference_radius_deviation_from_one": float(
                maximum_physical_radius_deviation
            ),
            "maximum_provisional_shot_relative_difference": float(
                maximum_shot_relative_difference
            ),
            "general_radius_implementation_recommended": (
                general_radius_recommended
            ),
            "general_radius_decision_rule": (
                "recommend when physical PF radius deviates from one by more than "
                "1% or provisional per-axis shots differ by more than 5%"
            ),
            "maximum_explicit_trajectory_count_log10_avoided": float(
                max(
                    point["explicit_round_trajectory_count_log10"]
                    for point in points
                )
            ),
        },
        "performance": {
            "dense_hamiltonian_and_spectrum_seconds": float(dense_elapsed),
            "partial_s2_preparation_seconds": float(preparation_elapsed),
            "validation_grid_seconds": float(grid_elapsed),
            "total_seconds": float(time.perf_counter() - started),
            "acceleration": (
                "independent trajectory strings replaced by sector-restricted "
                "finite-Taylor operator moments and matrix powers"
            ),
        },
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_finite_rte_signal_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FINITE_RTE_SIGNAL_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Unsupported finite-RTE signal validation schema.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("validation_fingerprint must be a SHA-256 hex string.")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Finite-RTE signal validation fingerprint mismatch.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")


def write_finite_rte_signal_validation(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_finite_rte_signal_payload(payload)
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

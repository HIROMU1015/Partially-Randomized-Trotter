"""System-size validation of the operational partial-S2 PF coefficient."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse.linalg import expm_multiply

from .config import PERTURBATION_FIT_STARTS, PERTURBATION_FIT_STEP
from .df_hamiltonian import (
    DFHamiltonian,
    PhysicalSector,
    df_diagonal,
    df_linear_operator,
    solve_df_ground_state,
)
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import (
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from .df_trotter.circuit import simulate_statevector
from .finite_rte_signal_validation import (
    _explicit_cutoff_tolerance,
    _qiskit_to_openfermion_sector_permutation,
)
from .pf_delta_validation import (
    _openfermion_sector_state_to_qiskit_full,
    _phase_and_perturbative_energy_bias,
    _qiskit_full_state_to_openfermion_sector,
    paper_d6_perturbative_energy_bias,
    validate_pf_delta_payload,
)
from .rte import require_integer_count


PF_C_SYSTEM_SIZE_SCHEMA_VERSION = "pf_c_system_size_validation_v2"
PF_C_SYSTEM_SIZE_METHOD = (
    "configured_qiskit_delta_window_exact_eigenphase_paper_d6_envelope_v2"
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


def configured_qiskit_delta_times(
    molecule_type: int,
    *,
    point_count: int | None = None,
    step: float = PERTURBATION_FIT_STEP,
) -> tuple[float, ...]:
    """Return the established lower-order Qiskit execution window for H_n."""
    molecule = require_integer_count(
        molecule_type,
        name="molecule_type",
        minimum=1,
    )
    if molecule not in PERTURBATION_FIT_STARTS:
        raise KeyError(f"No configured Qiskit delta window for H{molecule}.")
    if not math.isfinite(float(step)) or step <= 0.0:
        raise ValueError("step must be finite and positive.")
    count = (
        3
        if point_count is None and molecule >= 12
        else 4
        if point_count is None
        else require_integer_count(point_count, name="point_count", minimum=1)
    )
    start = float(PERTURBATION_FIT_STARTS[molecule][0])
    return tuple(round(start + float(step) * index, 10) for index in range(count))


def legacy_perturbation_conditioning(
    energy: float,
    delta_time: float,
    *,
    minimum_denominator_abs: float = 0.1,
) -> dict[str, Any]:
    """Diagnose the legacy-cosine and paper-Eq.-D6 sine denominators."""
    energy_value = float(energy)
    delta_value = float(delta_time)
    threshold = float(minimum_denominator_abs)
    if not math.isfinite(energy_value):
        raise ValueError("energy must be finite.")
    if not math.isfinite(delta_value) or delta_value <= 0.0:
        raise ValueError("delta_time must be finite and positive.")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("minimum_denominator_abs must lie strictly in (0, 1).")
    phase = energy_value * delta_value
    cosine_abs = float(abs(math.cos(phase)))
    sine_abs = float(abs(math.sin(phase)))
    return {
        "energy_times_delta": phase,
        "legacy_cosine_denominator_abs": cosine_abs,
        "legacy_sine_denominator_abs": sine_abs,
        "minimum_denominator_abs": threshold,
        "legacy_cosine_formula_well_conditioned": bool(
            cosine_abs >= threshold
        ),
        "legacy_sine_formula_well_conditioned": bool(sine_abs >= threshold),
        "paper_d6_formula_well_conditioned": bool(sine_abs >= threshold),
        "paper_d6_formula_uses_sine_denominator": True,
        "shift_invariant_formula_uses_trigonometric_denominator": False,
    }


def _random_tail_hamiltonian(
    hamiltonian: DFHamiltonian,
    randomized_indices: Sequence[int],
) -> DFHamiltonian:
    indices = tuple(int(index) for index in randomized_indices)
    return DFHamiltonian(
        constant=0.0,
        one_body=np.zeros_like(hamiltonian.one_body),
        lambdas=np.asarray(
            [hamiltonian.lambdas[index] for index in indices],
            dtype=np.float64,
        ),
        g_matrices=tuple(hamiltonian.g_matrices[index] for index in indices),
        metadata={
            **hamiltonian.metadata,
            "operator_role": "randomized_tail_only",
            "selected_block_indices": indices,
        },
    )


def validate_state_action_coefficient(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    molecule_type: int,
    ld: int,
    delta_times: Sequence[float] | None = None,
    matrix_free_backend: str = "auto",
    numerical_atol: float = 1e-9,
    perturbative_relative_tolerance: float = 1e-4,
    paper_d6_relative_tolerance: float = 0.02,
    paper_d6_minimum_sine_abs: float = 0.1,
    exact_reference_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate C by state action without building or diagonalizing the PF unitary."""
    molecule = require_integer_count(
        molecule_type,
        name="molecule_type",
        minimum=1,
    )
    ld_value = require_integer_count(ld, name="ld")
    deltas = (
        configured_qiskit_delta_times(molecule)
        if delta_times is None
        else tuple(float(value) for value in delta_times)
    )
    if not deltas or any(not math.isfinite(value) or value <= 0.0 for value in deltas):
        raise ValueError("delta_times must contain finite positive values.")
    if hamiltonian.n_qubits != sector.n_qubits:
        raise ValueError("Hamiltonian and sector n_qubits differ.")
    if ld_value >= hamiltonian.n_blocks:
        raise ValueError("State-action validation requires a non-empty tail.")

    ground = solve_df_ground_state(
        hamiltonian,
        sector,
        matrix_free_backend=matrix_free_backend,
        expand_state=False,
    )
    physical_state = np.asarray(ground.sector_state_vector, dtype=np.complex128)
    partition = split_df_hamiltonian_by_ld(hamiltonian, ld_value)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="faithful_identity_in_tail",
        coefficient_atol=0.0,
    )
    tail_hamiltonian = _random_tail_hamiltonian(
        hamiltonian,
        partition.randomized_block_indices,
    )
    tail_operator, tail_matvec_counter = df_linear_operator(
        tail_hamiltonian,
        sector,
        backend=matrix_free_backend,
    )
    tail_trace = complex(np.sum(df_diagonal(tail_hamiltonian, sector)))
    permutation = _qiskit_to_openfermion_sector_permutation(sector)
    initial_qiskit = _openfermion_sector_state_to_qiskit_full(
        physical_state,
        sector,
        permutation,
    )
    builder = QiskitDFPartialS2CircuitBuilder()
    reference_by_delta: dict[float, Mapping[str, Any]] = {}
    if exact_reference_payload is not None:
        validate_pf_delta_payload(exact_reference_payload)
        reference_by_delta = {
            float(point["delta_time"]): point
            for point in exact_reference_payload["points"]
        }

    point_records: list[dict[str, Any]] = []
    for delta in deltas:
        request = make_df_partial_s2_step_request(
            preparation,
            step_time=delta,
            rte_steps=1,
            truncation_tolerance=_explicit_cutoff_tolerance(
                preparation.exact_rte_lambda_r * delta,
                0,
            ),
            finite_taylor_order=0,
            seed=20260818,
        )
        parts = builder.build_additive_circuits(request)
        after_forward_qiskit = simulate_statevector(
            parts.forward_deterministic_half,
            initial_qiskit,
        )
        after_forward_sector, forward_leakage = (
            _qiskit_full_state_to_openfermion_sector(
                after_forward_qiskit,
                sector,
                permutation,
            )
        )
        after_tail_sector = expm_multiply(
            (-1j * delta) * tail_operator,
            after_forward_sector,
            traceA=(-1j * delta) * tail_trace,
        )
        after_tail_qiskit = _openfermion_sector_state_to_qiskit_full(
            np.asarray(after_tail_sector, dtype=np.complex128),
            sector,
            permutation,
        )
        final_qiskit = simulate_statevector(
            parts.reverse_deterministic_half,
            after_tail_qiskit,
        )
        final_sector, reverse_leakage = _qiskit_full_state_to_openfermion_sector(
            final_qiskit,
            sector,
            permutation,
        )
        estimators = _phase_and_perturbative_energy_bias(
            initial_state=physical_state,
            evolved_state=final_sector,
            ground_energy=float(ground.energy),
            delta_time=delta,
        )
        phase_bias = float(estimators["absolute_phase_energy_bias"])
        perturbative_bias = float(
            estimators["absolute_linearized_perturbative_energy_bias"]
        )
        relative_payload = estimators["relative_survival_amplitude"]
        relative_survival = complex(
            float(relative_payload["real"]),
            float(relative_payload["imag"]),
        )
        paper_d6 = paper_d6_perturbative_energy_bias(
            relative_survival,
            float(ground.energy),
            delta,
            minimum_sine_abs=paper_d6_minimum_sine_abs,
        )
        paper_d6_bias = paper_d6["absolute_energy_bias"]
        if paper_d6_bias is None:
            paper_d6_vs_phase_relative_difference = None
        else:
            paper_d6_vs_phase_relative_difference = float(
                abs(float(paper_d6_bias) - phase_bias)
                / max(phase_bias, numerical_atol)
            )
        relative_difference = float(
            abs(
                float(estimators["signed_phase_energy_bias"])
                - float(estimators["signed_linearized_perturbative_energy_bias"])
            )
            / max(phase_bias, numerical_atol)
        )
        reference_phase_difference = None
        reference_perturbative_difference = None
        reference_paper_d6_difference = None
        if delta in reference_by_delta:
            reference = reference_by_delta[delta][
                "cpu_qiskit_direct_tail_validation"
            ]
            reference_phase = float(reference["signed_qiskit_phase_energy_bias"])
            reference_perturbative = float(
                reference["signed_linearized_perturbative_energy_bias"]
            )
            reference_phase_difference = float(
                abs(float(estimators["signed_phase_energy_bias"]) - reference_phase)
            )
            reference_perturbative_difference = float(
                abs(
                    float(
                        estimators["signed_linearized_perturbative_energy_bias"]
                    )
                    - reference_perturbative
                )
            )
            reference_relative_payload = reference["relative_survival_amplitude"]
            reference_paper_d6 = paper_d6_perturbative_energy_bias(
                complex(
                    float(reference_relative_payload["real"]),
                    float(reference_relative_payload["imag"]),
                ),
                float(ground.energy),
                delta,
                minimum_sine_abs=paper_d6_minimum_sine_abs,
            )
            reference_paper_d6_bias = reference_paper_d6["absolute_energy_bias"]
            if paper_d6_bias is not None and reference_paper_d6_bias is not None:
                reference_paper_d6_difference = float(
                    abs(float(paper_d6_bias) - float(reference_paper_d6_bias))
                )
        point_records.append(
            {
                "delta_time": delta,
                "absolute_phase_energy_bias": phase_bias,
                "absolute_shift_invariant_perturbative_energy_bias": (
                    perturbative_bias
                ),
                "phase_point_coefficient": phase_bias / delta**2,
                "shift_invariant_perturbative_point_coefficient": (
                    perturbative_bias / delta**2
                ),
                "paper_d6_perturbative_energy_bias": paper_d6,
                "paper_d6_point_coefficient": (
                    None
                    if paper_d6_bias is None
                    else float(paper_d6_bias) / delta**2
                ),
                "paper_d6_vs_phase_relative_difference": (
                    paper_d6_vs_phase_relative_difference
                ),
                "perturbative_vs_phase_relative_difference": relative_difference,
                "relative_survival_radius": float(
                    estimators["relative_survival_radius"]
                ),
                "maximum_sector_leakage_norm": float(
                    max(forward_leakage, reverse_leakage)
                ),
                "absolute_phase_bias_difference_vs_dense_reference": (
                    reference_phase_difference
                ),
                "absolute_perturbative_bias_difference_vs_dense_reference": (
                    reference_perturbative_difference
                ),
                "absolute_paper_d6_bias_difference_vs_dense_reference": (
                    reference_paper_d6_difference
                ),
                "legacy_conditioning": legacy_perturbation_conditioning(
                    float(ground.energy),
                    delta,
                    minimum_denominator_abs=paper_d6_minimum_sine_abs,
                ),
            }
        )

    phase_coefficients = [
        float(point["phase_point_coefficient"]) for point in point_records
    ]
    perturbative_coefficients = [
        float(point["shift_invariant_perturbative_point_coefficient"])
        for point in point_records
    ]
    paper_d6_coefficients = [
        float(point["paper_d6_point_coefficient"])
        for point in point_records
        if point["paper_d6_point_coefficient"] is not None
    ]
    paper_d6_envelope = (
        None
        if len(paper_d6_coefficients) != len(point_records)
        else float(max(paper_d6_coefficients))
    )
    paper_d6_phase_differences = [
        float(point["paper_d6_vs_phase_relative_difference"])
        for point in point_records
        if point["paper_d6_vs_phase_relative_difference"] is not None
    ]
    reference_differences = [
        float(value)
        for point in point_records
        for value in (
            point["absolute_phase_bias_difference_vs_dense_reference"],
            point["absolute_perturbative_bias_difference_vs_dense_reference"],
            point["absolute_paper_d6_bias_difference_vs_dense_reference"],
        )
        if value is not None
    ]
    validation_pass = bool(
        ground.converged
        and all(
            point["maximum_sector_leakage_norm"] <= numerical_atol
            and point["perturbative_vs_phase_relative_difference"]
            <= perturbative_relative_tolerance
            and point["paper_d6_perturbative_energy_bias"]["well_conditioned"]
            and point["paper_d6_vs_phase_relative_difference"] is not None
            and point["paper_d6_vs_phase_relative_difference"]
            <= paper_d6_relative_tolerance
            for point in point_records
        )
        and len(paper_d6_coefficients) == len(point_records)
        and (
            not reference_differences
            or max(reference_differences) <= 10.0 * numerical_atol
        )
    )
    return {
        "molecule_type": molecule,
        "distance": float(hamiltonian.metadata["distance"]),
        "basis": str(hamiltonian.metadata["basis"]),
        "n_qubits": int(hamiltonian.n_qubits),
        "n_electrons": sector.n_electrons,
        "sector_dimension": int(sector.dimension),
        "df_rank_actual": int(hamiltonian.n_blocks),
        "ld": ld_value,
        "ld_policy": "floor(df_rank_actual/2)",
        "delta_times": list(deltas),
        "ground_energy": float(ground.energy),
        "ground_state_converged": bool(ground.converged),
        "ground_state_residual_norm": float(ground.residual_norm),
        "execution_model": (
            "Qiskit deterministic halves plus matrix-free sector "
            "scipy.sparse.linalg.expm_multiply for exact H_R action"
        ),
        "builds_or_diagonalizes_pf_unitary": False,
        "point_records": point_records,
        "phase_window_envelope_coefficient": float(max(phase_coefficients)),
        "shift_invariant_perturbative_window_envelope_coefficient": float(
            max(perturbative_coefficients)
        ),
        "paper_d6_window_envelope_coefficient": paper_d6_envelope,
        "operational_state_action_coefficient": paper_d6_envelope,
        "operational_state_action_coefficient_kind": "paper_eq_d6_perturbation",
        "maximum_perturbative_vs_phase_relative_difference": float(
            max(
                point["perturbative_vs_phase_relative_difference"]
                for point in point_records
            )
        ),
        "maximum_paper_d6_vs_phase_relative_difference": (
            None
            if not paper_d6_phase_differences
            else float(max(paper_d6_phase_differences))
        ),
        "paper_d6_ill_conditioned_point_count": sum(
            not point["paper_d6_perturbative_energy_bias"]["well_conditioned"]
            for point in point_records
        ),
        "maximum_absolute_difference_vs_dense_reference": (
            None if not reference_differences else float(max(reference_differences))
        ),
        "tail_matvec_count": int(tail_matvec_counter["count"]),
        "state_action_validation_pass": validation_pass,
    }


def summarize_size_result(
    pf_payload: Mapping[str, Any],
    *,
    molecule_type: int,
    core_artifact_path: str,
    paper_d6_minimum_sine_abs: float = 0.1,
    paper_d6_relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Reduce one full PF validation to the operational-C size-sweep record."""
    validate_pf_delta_payload(pf_payload)
    molecule = require_integer_count(
        molecule_type,
        name="molecule_type",
        minimum=1,
    )
    request = pf_payload["request"]
    hamiltonian = pf_payload["hamiltonian"]
    summary = pf_payload["summary"]
    energy = float(hamiltonian["ground_energy"])
    expected_deltas = configured_qiskit_delta_times(
        molecule,
        point_count=len(request["validation_delta_times"]),
    )
    actual_deltas = tuple(float(value) for value in request["validation_delta_times"])
    configured_delta_grid_match = actual_deltas == expected_deltas

    point_records: list[dict[str, Any]] = []
    exact_coefficients: list[float] = []
    perturbative_coefficients: list[float] = []
    paper_d6_coefficients: list[float] = []
    for point in pf_payload["points"]:
        delta = float(point["delta_time"])
        qpe = point["qpe_spectral_energy_distribution"]
        qiskit = point["cpu_qiskit_direct_tail_validation"]
        exact_bias = float(qpe["dominant_phase_cluster_absolute_energy_bias"])
        perturbative_bias = float(
            qiskit["absolute_linearized_perturbative_energy_bias"]
        )
        exact_coefficient = exact_bias / delta**2
        perturbative_coefficient = perturbative_bias / delta**2
        relative_payload = qiskit["relative_survival_amplitude"]
        paper_d6 = paper_d6_perturbative_energy_bias(
            complex(
                float(relative_payload["real"]),
                float(relative_payload["imag"]),
            ),
            energy,
            delta,
            minimum_sine_abs=paper_d6_minimum_sine_abs,
        )
        paper_d6_bias = paper_d6["absolute_energy_bias"]
        paper_d6_coefficient = (
            None
            if paper_d6_bias is None
            else float(paper_d6_bias) / delta**2
        )
        paper_d6_vs_exact_relative_difference = (
            None
            if paper_d6_coefficient is None
            else float(
                abs(paper_d6_coefficient - exact_coefficient)
                / max(abs(exact_coefficient), 1e-300)
            )
        )
        exact_coefficients.append(exact_coefficient)
        perturbative_coefficients.append(perturbative_coefficient)
        if paper_d6_coefficient is not None:
            paper_d6_coefficients.append(paper_d6_coefficient)
        point_records.append(
            {
                "delta_time": delta,
                "dominant_eigenphase_point_coefficient": exact_coefficient,
                "shift_invariant_perturbative_point_coefficient": (
                    perturbative_coefficient
                ),
                "paper_d6_perturbative_energy_bias": paper_d6,
                "paper_d6_point_coefficient": paper_d6_coefficient,
                "paper_d6_vs_dominant_eigenphase_relative_difference": (
                    paper_d6_vs_exact_relative_difference
                ),
                "perturbative_vs_phase_relative_difference": float(
                    qiskit["perturbative_vs_phase_relative_difference"]
                ),
                "relative_survival_radius": float(
                    qiskit["relative_survival_radius"]
                ),
                "legacy_conditioning": legacy_perturbation_conditioning(
                    energy,
                    delta,
                    minimum_denominator_abs=paper_d6_minimum_sine_abs,
                ),
            }
        )

    exact_envelope = float(max(exact_coefficients))
    perturbative_envelope = float(max(perturbative_coefficients))
    paper_d6_envelope = (
        None
        if len(paper_d6_coefficients) != len(point_records)
        else float(max(paper_d6_coefficients))
    )
    operational_envelope = (
        None
        if paper_d6_envelope is None
        else float(max(exact_envelope, paper_d6_envelope))
    )
    shift_invariant_envelope_relative_difference = float(
        abs(perturbative_envelope - exact_envelope)
        / max(abs(exact_envelope), 1e-300)
    )
    paper_d6_envelope_relative_difference = (
        None
        if paper_d6_envelope is None
        else float(
            abs(paper_d6_envelope - exact_envelope)
            / max(abs(exact_envelope), 1e-300)
        )
    )
    corrected_estimator_pass = bool(
        summary["cpu_qiskit_perturbation_validation_pass"]
    )
    paper_d6_estimator_pass = bool(
        len(paper_d6_coefficients) == len(point_records)
        and all(
            point["paper_d6_perturbative_energy_bias"]["well_conditioned"]
            and point[
                "paper_d6_vs_dominant_eigenphase_relative_difference"
            ]
            is not None
            and point[
                "paper_d6_vs_dominant_eigenphase_relative_difference"
            ]
            <= paper_d6_relative_tolerance
            for point in point_records
        )
    )
    return {
        "molecule_type": molecule,
        "distance": float(hamiltonian["metadata"]["distance"]),
        "basis": str(hamiltonian["metadata"]["basis"]),
        "n_qubits": int(hamiltonian["n_qubits"]),
        "n_electrons": int(hamiltonian["sector_n_electrons"]),
        "sector_dimension": int(hamiltonian["sector_dimension"]),
        "df_rank_actual": int(hamiltonian["n_df_blocks"]),
        "df_rank_source": hamiltonian["metadata"].get("df_rank_source"),
        "ld": int(request["ld"]),
        "ld_policy": "floor(df_rank_actual/2)",
        "ground_energy": energy,
        "configured_qiskit_delta_times": list(expected_deltas),
        "evaluated_delta_times": list(actual_deltas),
        "configured_delta_grid_match": configured_delta_grid_match,
        "delta_window_source": (
            "project PERTURBATION_FIT_STARTS lower-order window inherited "
            "from the Evaluation workflow"
        ),
        "estimator_definition": (
            "paper Eq. (D6) full-H-ground-state perturbative estimator with "
            "explicit sine-conditioning rejection"
        ),
        "shift_invariant_estimator_policy": "diagnostic_only",
        "point_records": point_records,
        "dominant_eigenphase_window_envelope_coefficient": exact_envelope,
        "shift_invariant_perturbative_window_envelope_coefficient": (
            perturbative_envelope
        ),
        "paper_d6_window_envelope_coefficient": paper_d6_envelope,
        "operational_validation_window_envelope_coefficient": operational_envelope,
        "operational_coefficient_kind": (
            "maximum_of_dominant_eigenphase_and_paper_eq_d6"
        ),
        "shift_invariant_vs_exact_envelope_relative_difference": (
            shift_invariant_envelope_relative_difference
        ),
        "paper_d6_vs_exact_envelope_relative_difference": (
            paper_d6_envelope_relative_difference
        ),
        "minimum_legacy_cosine_denominator_abs": float(
            min(
                point["legacy_conditioning"][
                    "legacy_cosine_denominator_abs"
                ]
                for point in point_records
            )
        ),
        "minimum_legacy_sine_denominator_abs": float(
            min(
                point["legacy_conditioning"]["legacy_sine_denominator_abs"]
                for point in point_records
            )
        ),
        "minimum_paper_d6_sine_denominator_abs": float(
            min(
                point["paper_d6_perturbative_energy_bias"][
                    "sine_denominator_abs"
                ]
                for point in point_records
            )
        ),
        "legacy_cosine_ill_conditioned_point_count": sum(
            not point["legacy_conditioning"][
                "legacy_cosine_formula_well_conditioned"
            ]
            for point in point_records
        ),
        "legacy_sine_ill_conditioned_point_count": sum(
            not point["legacy_conditioning"][
                "legacy_sine_formula_well_conditioned"
            ]
            for point in point_records
        ),
        "paper_d6_ill_conditioned_point_count": sum(
            not point["paper_d6_perturbative_energy_bias"]["well_conditioned"]
            for point in point_records
        ),
        "maximum_corrected_perturbative_vs_phase_relative_difference": float(
            summary[
                "maximum_linearized_perturbative_vs_phase_relative_difference"
            ]
        ),
        "single_dominant_phase_validation_pass": bool(
            summary["single_dominant_phase_approximation_validation_pass"]
        ),
        "corrected_perturbative_estimator_validation_pass": corrected_estimator_pass,
        "paper_d6_estimator_validation_pass": paper_d6_estimator_pass,
        "paper_d6_relative_tolerance": float(paper_d6_relative_tolerance),
        "operational_coefficient_usable": bool(
            configured_delta_grid_match
            and paper_d6_estimator_pass
            and summary["single_dominant_phase_approximation_validation_pass"]
        ),
        "core_pf_artifact_path": str(core_artifact_path),
        "core_pf_validation_fingerprint": pf_payload["validation_fingerprint"],
    }


def make_system_size_payload(
    size_results: Sequence[Mapping[str, Any]],
    *,
    request: Mapping[str, Any],
    state_action_results: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    results = [dict(result) for result in size_results]
    if not results:
        raise ValueError("size_results must not be empty.")
    molecule_sizes = [int(result["molecule_type"]) for result in results]
    if len(set(molecule_sizes)) != len(molecule_sizes):
        raise ValueError("size_results must contain unique molecule sizes.")
    results.sort(key=lambda result: int(result["molecule_type"]))
    state_results = [dict(result) for result in state_action_results]
    state_results.sort(key=lambda result: int(result["molecule_type"]))
    payload: dict[str, Any] = {
        "schema_version": PF_C_SYSTEM_SIZE_SCHEMA_VERSION,
        "validation_method": PF_C_SYSTEM_SIZE_METHOD,
        "scope": "operational_pf_coefficient_system_size_validation_only",
        "final_cost_evaluation_performed": False,
        "request": dict(request),
        "provenance": dict(provenance or {}),
        "size_results": results,
        "state_action_results": state_results,
        "summary": {
            "validated_molecule_sizes": [
                int(result["molecule_type"]) for result in results
            ],
            "validated_size_count": len(results),
            "minimum_n_qubits": min(int(result["n_qubits"]) for result in results),
            "maximum_n_qubits": max(int(result["n_qubits"]) for result in results),
            "all_configured_delta_grids_match": all(
                result["configured_delta_grid_match"] for result in results
            ),
            "all_single_dominant_phase_validation_pass": all(
                result["single_dominant_phase_validation_pass"]
                for result in results
            ),
            "all_corrected_perturbative_estimator_validation_pass": all(
                result["corrected_perturbative_estimator_validation_pass"]
                for result in results
            ),
            "all_paper_d6_estimator_validation_pass": all(
                result["paper_d6_estimator_validation_pass"]
                for result in results
            ),
            "all_operational_coefficients_usable": all(
                result["operational_coefficient_usable"] for result in results
            ),
            "maximum_paper_d6_vs_exact_envelope_relative_difference": float(
                max(
                    result[
                        "paper_d6_vs_exact_envelope_relative_difference"
                    ]
                    for result in results
                )
            ),
            "legacy_cosine_ill_conditioned_point_count": sum(
                int(result["legacy_cosine_ill_conditioned_point_count"])
                for result in results
            ),
            "legacy_sine_ill_conditioned_point_count": sum(
                int(result["legacy_sine_ill_conditioned_point_count"])
                for result in results
            ),
            "paper_d6_ill_conditioned_point_count": sum(
                int(result["paper_d6_ill_conditioned_point_count"])
                for result in results
            ),
            "h12_direct_validation_performed": any(
                int(result["molecule_type"]) == 12 for result in results
            ),
            "state_action_validated_molecule_sizes": [
                int(result["molecule_type"]) for result in state_results
            ],
            "maximum_state_action_n_qubits": (
                None
                if not state_results
                else max(int(result["n_qubits"]) for result in state_results)
            ),
            "all_state_action_validation_pass": bool(
                state_results
                and all(
                    result["state_action_validation_pass"]
                    for result in state_results
                )
            ),
        },
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_system_size_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != PF_C_SYSTEM_SIZE_SCHEMA_VERSION:
        raise ValueError("Unsupported PF C system-size validation schema.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("validation_fingerprint must be a SHA-256 hex string.")
    unsigned = dict(payload)
    unsigned.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(unsigned):
        raise ValueError("PF C system-size validation fingerprint mismatch.")
    results = payload.get("size_results")
    if not isinstance(results, list) or not results:
        raise ValueError("PF C system-size results are missing.")
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("Every system-size result must be a mapping.")
        for key in (
            "configured_delta_grid_match",
            "single_dominant_phase_validation_pass",
            "corrected_perturbative_estimator_validation_pass",
            "paper_d6_estimator_validation_pass",
            "operational_coefficient_usable",
        ):
            if not isinstance(result.get(key), bool):
                raise ValueError(f"System-size result is missing boolean {key}.")


def write_system_size_validation(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_system_size_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

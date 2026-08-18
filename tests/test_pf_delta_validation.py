from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.pf_delta_validation import (
    _qpe_spectral_energy_distribution,
    validate_pf_delta_grid,
    validate_pf_delta_payload,
    write_pf_delta_validation,
)


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.13,
        one_body=np.asarray(
            [[0.2, 0.03], [0.03, -0.1]],
            dtype=np.complex128,
        ),
        lambdas=np.asarray([0.2, -0.3]),
        g_matrices=(
            np.asarray([[1.0, 0.0], [0.0, 0.4]], dtype=np.complex128),
            np.asarray([[0.2, 0.0], [0.0, 2.0]], dtype=np.complex128),
        ),
        metadata={"name": "pf-delta-validation-toy"},
    )


def test_qpe_spectral_distribution_matches_known_branches() -> None:
    delta = 0.2
    target_energy = -1.1
    energy_shifts = np.asarray([0.03, 0.03, 0.8])
    weights = np.asarray([0.81, 0.09, 0.10])
    unitary = np.diag(np.exp(-1j * delta * (target_energy + energy_shifts)))
    state = np.sqrt(weights).astype(np.complex128)

    spectrum = _qpe_spectral_energy_distribution(
        unitary=unitary,
        state=state,
        delta_time=delta,
        target_energy=target_energy,
        cluster_energy_tolerance=1e-10,
        numerical_atol=1e-12,
    )

    expected_mean = float(np.dot(weights, energy_shifts))
    expected_rmse = float(np.sqrt(np.dot(weights, energy_shifts**2)))
    assert spectrum["signed_qpe_mean_energy_bias"] == pytest.approx(expected_mean)
    assert spectrum["qpe_energy_rmse"] == pytest.approx(expected_rmse)
    assert spectrum["dominant_phase_cluster_weight"] == pytest.approx(0.9)
    assert spectrum["dominant_phase_cluster_signed_energy_bias"] == pytest.approx(
        0.03
    )
    assert spectrum["non_dominant_phase_cluster_weight"] == pytest.approx(0.1)
    assert spectrum["effective_phase_cluster_count"] == pytest.approx(
        1.0 / (0.9**2 + 0.1**2)
    )
    multiplicities = sorted(
        cluster["schur_multiplicity"] for cluster in spectrum["phase_clusters"]
    )
    assert multiplicities == [1, 2]
    assert spectrum["numerical_consistency_pass"]


def test_pf_delta_validation_is_disjoint_and_tamper_evident(tmp_path) -> None:
    payload = validate_pf_delta_grid(
        _hamiltonian(),
        PhysicalSector.number_sector(n_qubits=2, n_electrons=1),
        ld=1,
        surrogate_calibration_times=(0.01, 0.02, 0.04, 0.08),
        validation_delta_times=(0.015, 0.03),
        q_values=(1, 2),
        surrogate_relative_tolerance=10.0,
        scaling_slope_interval=(0.1, 4.0),
        coefficient_atol=0.05,
        provenance={"test": True},
    )

    assert payload["request"]["calibration_validation_times_disjoint"]
    assert payload["final_cost_evaluation_performed"] is False
    assert payload["surrogate"]["is_rigorous_bound"] is False
    assert payload["summary"]["q_point_count"] == 4
    assert payload["summary"][
        "all_signal_errors_within_numerical_operator_error"
    ]
    assert payload["summary"]["all_cpu_qiskit_matrix_consistency_pass"]
    assert payload["summary"]["all_linearized_perturbation_consistency_pass"]
    assert payload["summary"]["cpu_qiskit_perturbation_validation_pass"]
    assert payload["summary"]["all_qpe_spectral_numerical_consistency_pass"]
    assert payload["summary"]["recommended_pf_coefficient_kind"] == (
        "dominant_eigenphase_branch_energy_bias"
    )
    assert payload["summary"]["recommended_pf_fixed_second_order_coefficient"] > 0.0
    assert payload["summary"]["scalable_pf_coefficient_estimator_kind"] == (
        "linearized_full_h_ground_state_perturbation"
    )
    assert payload["summary"]["qpe_rmse_coefficient_policy"] == (
        "diagnostic_only_not_primary_cost_input"
    )
    assert payload["summary"]["single_dominant_phase_approximation_validation_pass"]
    assert payload["summary"]["scalable_primary_coefficient_estimator_validation_pass"]
    assert payload["summary"]["single_dominant_phase_cost_model_validation_pass"]
    assert payload["summary"]["qpe_energy_rmse_fixed_second_order_coefficient"] > 0.0
    assert payload["summary"][
        "non_dominant_phase_cluster_weight_fixed_fourth_order_coefficient"
    ] > 0.0
    assert payload["summary"][
        "non_dominant_phase_cluster_weight_free_fit_slope"
    ] > 0.0
    assert payload["summary"][
        "maximum_physical_actual_over_surrogate_prediction"
    ] > 0.0
    assert all(
        point["cpu_qiskit_direct_tail_validation"][
            "qiskit_matrix_consistency_pass"
        ]
        for point in payload["points"]
    )
    for point in payload["points"]:
        spectrum = point["qpe_spectral_energy_distribution"]
        assert spectrum["numerical_consistency_pass"]
        assert sum(
            cluster["weight"] for cluster in spectrum["phase_clusters"]
        ) == pytest.approx(1.0, abs=1e-12)
        assert spectrum["qpe_energy_rmse"] ** 2 == pytest.approx(
            spectrum["signed_qpe_mean_energy_bias"] ** 2
            + spectrum["qpe_energy_standard_deviation"] ** 2,
            abs=1e-12,
        )
        assert spectrum["relative_survival_reconstruction_error"] < 1e-10
        assert spectrum["single_phase_contamination_bound_pass"]
        for q_result in point["q_results"]:
            diagnostic = q_result["single_dominant_phase_diagnostic"]
            assert diagnostic[
                "signal_phase_contamination_within_analytic_bound"
            ]
            assert diagnostic["signal_matches_dominant_branch_within_tolerance"]
    validate_pf_delta_payload(payload)

    path = tmp_path / "pf_delta.json"
    write_pf_delta_validation(payload, path)
    assert path.exists()

    tampered = deepcopy(payload)
    tampered["surrogate"]["coefficient"] += 1.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_pf_delta_payload(tampered)


def test_pf_delta_validation_rejects_overlapping_fit_and_holdout_times() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_pf_delta_grid(
            _hamiltonian(),
            PhysicalSector.number_sector(n_qubits=2, n_electrons=1),
            ld=1,
            surrogate_calibration_times=(0.01, 0.02),
            validation_delta_times=(0.02, 0.03),
            q_values=(1,),
        )

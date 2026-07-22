from __future__ import annotations

import numpy as np
from openfermion.ops import QubitOperator
from scipy.linalg import eigh, expm

from trotterlib.df_hamiltonian import PhysicalSector
from trotterlib.pauli_partial_cgs_validation import (
    FullHTargetState,
    dense_full_hamiltonian,
    dense_ranked_pauli_terms,
    fit_partial_s2_cgs,
    full_h_target_state,
    optimized_cost_for_cgs,
    partial_s2_deterministic_step_cost,
    partial_s2_unitary,
)
from trotterlib.partial_randomized_pf import (
    RankedPauliTerm,
    SortedPauliHamiltonian,
    build_sorted_pauli_hamiltonian,
)


def _one_qubit_noncommuting_hamiltonian() -> SortedPauliHamiltonian:
    raw = [(((0, "X"),), 0.8), (((0, "Z"),), 0.3)]
    terms = tuple(
        RankedPauliTerm(
            rank=index + 1,
            original_index=index,
            pauli_term=pauli,
            coeff=coeff,
            abs_coeff=abs(coeff),
            operator=QubitOperator(pauli, coeff),
        )
        for index, (pauli, coeff) in enumerate(raw)
    )
    return SortedPauliHamiltonian(
        molecule_type=1,
        distance=1.0,
        ham_name="toy_xz",
        num_qubits=1,
        identity_coeff=0.0,
        sorted_terms=terms,
    )


def _with_identity_shift(
    hamiltonian: SortedPauliHamiltonian, shift: float
) -> SortedPauliHamiltonian:
    return SortedPauliHamiltonian(
        molecule_type=hamiltonian.molecule_type,
        distance=hamiltonian.distance,
        ham_name=f"{hamiltonian.ham_name}_shifted",
        num_qubits=hamiltonian.num_qubits,
        identity_coeff=hamiltonian.identity_coeff + float(shift),
        sorted_terms=hamiltonian.sorted_terms,
    )


def _one_qubit_three_term_hamiltonian() -> SortedPauliHamiltonian:
    raw = [(((0, "X"),), 0.8), (((0, "Z"),), 0.3), (((0, "Y"),), -0.2)]
    terms = tuple(
        RankedPauliTerm(
            rank=index + 1,
            original_index=index,
            pauli_term=pauli,
            coeff=coeff,
            abs_coeff=abs(coeff),
            operator=QubitOperator(pauli, coeff),
        )
        for index, (pauli, coeff) in enumerate(raw)
    )
    return SortedPauliHamiltonian(
        molecule_type=1,
        distance=1.0,
        ham_name="toy_xzy",
        num_qubits=1,
        identity_coeff=0.11,
        sorted_terms=terms,
    )
def _target(sorted_hamiltonian: SortedPauliHamiltonian) -> FullHTargetState:
    matrix = dense_full_hamiltonian(sorted_hamiltonian)
    values, vectors = eigh(matrix)
    state = np.asarray(vectors[:, 0], dtype=np.complex128)
    energy = float(values[0])
    sector = PhysicalSector(
        n_qubits=1,
        basis_indices=np.asarray([0, 1], dtype=np.int64),
    )
    return FullHTargetState(
        energy=energy,
        state=state,
        residual_norm=float(np.linalg.norm(matrix @ state - energy * state)),
        sector=sector,
        global_ground_energy=energy,
        sector_ground_multiplicity=1,
        sector_excitation_gap=None,
        full_space_target_energy_multiplicity=1,
        full_space_target_energy_gap=None,
        full_space_target_eigenspace=state[:, np.newaxis],
        target_eigenspace_projection_residual=0.0,
        max_full_spectral_distance_from_target=float(values[-1] - values[0]),
    )


def test_ld_zero_is_exact_full_h_evolution() -> None:
    hamiltonian = _one_qubit_noncommuting_hamiltonian()
    dense_terms = dense_ranked_pauli_terms(hamiltonian)
    delta = 0.17
    actual = partial_s2_unitary(hamiltonian, dense_terms, ld=0, delta=delta)
    expected = expm(-1j * delta * dense_full_hamiltonian(hamiltonian))
    assert np.linalg.norm(actual - expected, ord=2) < 1e-12


def test_full_h_qpe_rmse_cgs_has_second_order_scaling() -> None:
    hamiltonian = _one_qubit_noncommuting_hamiltonian()
    dense_terms = dense_ranked_pauli_terms(hamiltonian)
    fit = fit_partial_s2_cgs(
        hamiltonian,
        dense_terms,
        _target(hamiltonian),
        ld=1,
        delta_values=(0.02, 0.03, 0.05, 0.08, 0.12),
        noise_floor=1e-14,
    )
    assert fit.c_gs_qpe_rmse > 0.0
    np.testing.assert_allclose(
        fit.c_gs_ground_space_worst_rmse,
        fit.c_gs_qpe_rmse,
        rtol=1e-10,
        atol=1e-14,
    )
    assert fit.qpe_rmse_fit_slope is not None
    assert abs(fit.qpe_rmse_fit_slope - 2.0) < 0.05
    assert fit.min_phase_branch_cut_clearance > 0.0
    assert fit.min_target_cluster_principal_overlap > 0.99
    np.testing.assert_allclose(
        np.square(fit.qpe_rmse_biases),
        np.square(fit.qpe_mean_biases) + np.square(fit.qpe_energy_stds),
        rtol=1e-12,
        atol=1e-15,
    )
    assert max(fit.unitary_defects) < 1e-12


def test_partial_s2_deterministic_cost_keeps_tail_between_halves() -> None:
    assert partial_s2_deterministic_step_cost(0, 5) == 0
    assert partial_s2_deterministic_step_cost(2, 5) == 4
    assert partial_s2_deterministic_step_cost(5, 5) == 9


def test_full_prefix_matches_manual_s2_and_is_time_reversible() -> None:
    hamiltonian = _one_qubit_noncommuting_hamiltonian()
    dense_terms = dense_ranked_pauli_terms(hamiltonian)
    delta = 0.17
    actual = partial_s2_unitary(hamiltonian, dense_terms, ld=2, delta=delta)
    expected = (
        expm(-0.5j * delta * dense_terms[0])
        @ expm(-1.0j * delta * dense_terms[1])
        @ expm(-0.5j * delta * dense_terms[0])
    )
    reverse = partial_s2_unitary(hamiltonian, dense_terms, ld=2, delta=-delta)
    assert np.linalg.norm(actual - expected, ord=2) < 1e-12
    assert np.linalg.norm(reverse - actual.conj().T, ord=2) < 1e-12


def test_identity_shift_does_not_change_fitted_cgs() -> None:
    hamiltonian = _one_qubit_noncommuting_hamiltonian()
    shifted = _with_identity_shift(hamiltonian, 0.7)
    deltas = (0.02, 0.03, 0.05, 0.08, 0.12)
    fit = fit_partial_s2_cgs(
        hamiltonian,
        dense_ranked_pauli_terms(hamiltonian),
        _target(hamiltonian),
        ld=1,
        delta_values=deltas,
        noise_floor=1e-14,
    )
    shifted_fit = fit_partial_s2_cgs(
        shifted,
        dense_ranked_pauli_terms(shifted),
        _target(shifted),
        ld=1,
        delta_values=deltas,
        noise_floor=1e-14,
    )
    np.testing.assert_allclose(
        [shifted_fit.c_gs_qpe_rmse, shifted_fit.c_gs_branch],
        [fit.c_gs_qpe_rmse, fit.c_gs_branch],
        rtol=1e-8,
        atol=1e-12,
    )


def test_nontrivial_exact_tail_is_between_prefix_halves() -> None:
    hamiltonian = _one_qubit_three_term_hamiltonian()
    dense_terms = dense_ranked_pauli_terms(hamiltonian)
    delta = 0.17
    identity = np.eye(2, dtype=np.complex128)
    exact_tail = hamiltonian.identity_coeff * identity + dense_terms[2]
    expected = (
        expm(-0.5j * delta * dense_terms[0])
        @ expm(-0.5j * delta * dense_terms[1])
        @ expm(-1.0j * delta * exact_tail)
        @ expm(-0.5j * delta * dense_terms[1])
        @ expm(-0.5j * delta * dense_terms[0])
    )
    actual = partial_s2_unitary(hamiltonian, dense_terms, ld=2, delta=delta)
    assert np.linalg.norm(actual - expected, ord=2) < 1e-12


def test_h3_target_is_sector_unique_but_full_space_degenerate() -> None:
    target = full_h_target_state(build_sorted_pauli_hamiltonian(3))
    assert target.sector_ground_multiplicity == 1
    assert target.full_space_target_energy_multiplicity == 3
    assert target.full_space_target_energy_gap is not None
    assert target.full_space_target_energy_gap > 0.05
    assert target.target_eigenspace_projection_residual < 1e-12


def test_cost_delta_cap_matches_uncapped_when_inactive_and_caps_when_needed() -> None:
    common = {
        "ld": 5,
        "total_terms": 5,
        "lambda_r": 0.0,
        "c_gs": 0.1,
        "epsilon_total": 1e-4,
    }
    uncapped = optimized_cost_for_cgs(**common)
    inactive = optimized_cost_for_cgs(**common, max_delta=1.0)
    np.testing.assert_allclose(inactive["g_total"], uncapped["g_total"], rtol=1e-10)
    assert inactive["delta_cap_active"] is False

    capped = optimized_cost_for_cgs(**common, max_delta=0.01)
    assert capped["delta_cap_active"] is True
    np.testing.assert_allclose(
        capped["g_det"],
        capped["deterministic_step_cost"]
        / (capped["eps_qpe"] * capped["delta_used"]),
        rtol=1e-12,
    )

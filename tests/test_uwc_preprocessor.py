from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from openfermion.ops import QubitOperator

import trotterlib.partial_randomized_pf as prpf
from trotterlib.partial_randomized_pf import (
    _cgs_cache_key_payload,
    _json_hash,
    _sorted_hamiltonian_hash,
    _sorted_pauli_hamiltonian_from_qubit_operator,
    analyze_partial_randomized_pf,
    save_partial_randomized_result,
)
from trotterlib.uwc import (
    UWCConfig,
    canonical_qubit_operator_terms,
    lambda_r_from_qubit_operator,
    preprocess_qubit_hamiltonian,
)


def _toy_qubit_hamiltonian() -> QubitOperator:
    return (
        QubitOperator((), 0.125)
        + QubitOperator(((0, "Z"),), 0.7)
        + QubitOperator(((1, "Z"),), -0.4)
        + QubitOperator(((0, "X"), (1, "X")), 0.2)
    )


def _install_toy_hamiltonian(monkeypatch) -> None:
    def fake_jw_hamiltonian_maker(molecule_type: int, distance: float = 1.0):
        del molecule_type, distance
        return _toy_qubit_hamiltonian(), 0.0, "toy_h2", 2

    monkeypatch.setattr(prpf, "jw_hamiltonian_maker", fake_jw_hamiltonian_maker)
    monkeypatch.setattr(
        prpf,
        "default_perturbation_t_values",
        lambda molecule_type, pf_label, **_kwargs: (0.05, 0.06),
    )
    monkeypatch.setattr(
        prpf,
        "load_cgs_json_cache",
        lambda: prpf._default_cgs_cache_document(),
    )
    monkeypatch.setattr(
        prpf,
        "save_cgs_json_cache",
        lambda cache_document, cache_path=prpf.PARTIAL_RANDOMIZED_CGS_CACHE_PATH: Path(
            cache_path
        ),
    )


def test_uwc_none_returns_identical_hamiltonian_terms() -> None:
    hamiltonian = _toy_qubit_hamiltonian()

    result = preprocess_qubit_hamiltonian(
        hamiltonian,
        UWCConfig(enabled=False),
        n_qubits=2,
        target_ld=1,
    )

    assert canonical_qubit_operator_terms(result.hamiltonian) == canonical_qubit_operator_terms(
        hamiltonian
    )
    assert result.metadata["preprocessor"] == "none"
    assert result.metadata["uwc_method"] == "none"


def test_uwc_none_keeps_pr_pf_candidate_results_unchanged(monkeypatch) -> None:
    _install_toy_hamiltonian(monkeypatch)

    baseline = analyze_partial_randomized_pf(
        2,
        epsilon_total=1e-2,
        pf_labels=("2nd",),
        ld_values=(0, 1),
        kappa_mode="fixed",
        kappa_value=2.0,
        matrix_free_backend="python",
    )
    explicit_none = analyze_partial_randomized_pf(
        2,
        epsilon_total=1e-2,
        uwc_config=UWCConfig(
            enabled=False,
            method="simple_shift",
            parameters={"shift": 0.5},
        ),
        pf_labels=("2nd",),
        ld_values=(0, 1),
        kappa_mode="fixed",
        kappa_value=2.0,
        matrix_free_backend="python",
    )

    assert baseline.best == explicit_none.best
    assert baseline.candidates == explicit_none.candidates
    assert explicit_none.preprocessor == "none"


def test_simple_shift_changes_uwc_hamiltonian_hash() -> None:
    hamiltonian = _toy_qubit_hamiltonian()
    original = preprocess_qubit_hamiltonian(hamiltonian, n_qubits=2, target_ld=1)
    shifted = preprocess_qubit_hamiltonian(
        hamiltonian,
        UWCConfig(
            enabled=True,
            method="simple_shift",
            target_ld=1,
            parameters={"shift": 0.25, "qubit": 0},
        ),
        n_qubits=2,
    )

    assert shifted.metadata["preprocessor"] == "uwc"
    assert (
        shifted.metadata["uwc_hamiltonian_hash"]
        != original.metadata["uwc_hamiltonian_hash"]
    )
    assert canonical_qubit_operator_terms(shifted.hamiltonian) != canonical_qubit_operator_terms(
        hamiltonian
    )


def test_cgs_cache_key_is_separated_for_uwc_metadata() -> None:
    sorted_hamiltonian = _sorted_pauli_hamiltonian_from_qubit_operator(
        _toy_qubit_hamiltonian(),
        molecule_type=2,
        distance=1.0,
        ham_name="toy_h2",
        num_qubits=2,
    )
    sorted_hash = _sorted_hamiltonian_hash(sorted_hamiltonian)
    base_payload = _cgs_cache_key_payload(
        sorted_hamiltonian=sorted_hamiltonian,
        sorted_hamiltonian_hash=sorted_hash,
        pf_label="2nd",
        ld=1,
        t_values=(0.05, 0.06),
        ground_state_tol=1e-10,
        ground_state_ncv=None,
    )
    uwc_payload = _cgs_cache_key_payload(
        sorted_hamiltonian=sorted_hamiltonian,
        sorted_hamiltonian_hash=sorted_hash,
        pf_label="2nd",
        ld=1,
        t_values=(0.05, 0.06),
        ground_state_tol=1e-10,
        ground_state_ncv=None,
        preprocessor_cache_metadata={
            "preprocessor": "uwc",
            "uwc_method": "simple_shift",
            "uwc_target_ld": 1,
            "uwc_parameters": {"shift": 0.25},
            "uwc_hamiltonian_hash": "different",
        },
    )

    assert _json_hash(base_payload) != _json_hash(uwc_payload)


def test_lambda_r_at_target_ld_uses_descending_abs_coefficients() -> None:
    hamiltonian = (
        QubitOperator(((0, "Z"),), -3.0)
        + QubitOperator(((1, "Z"),), 2.0)
        + QubitOperator(((0, "X"),), 1.0)
        + QubitOperator((), 100.0)
    )

    assert np.isclose(lambda_r_from_qubit_operator(hamiltonian, 1), 3.0)
    assert np.isclose(lambda_r_from_qubit_operator(hamiltonian, 2), 1.0)


def test_bliss_number_shift_preserves_target_sector_ground_energy() -> None:
    hamiltonian = _toy_qubit_hamiltonian()
    result = preprocess_qubit_hamiltonian(
        hamiltonian,
        UWCConfig(
            enabled=True,
            method="bliss",
            target_ld=1,
            parameters={"target_particle_number": 1, "theta": 0.6},
            sector_energy_check="error",
            sector_energy_tolerance=1e-10,
        ),
        n_qubits=2,
    )

    check = result.metadata["sector_preservation_check"]
    assert check["ground_energy_checked"]
    assert check["max_abs_shift_on_sector"] <= 1e-10
    assert check["ground_energy_difference"] <= 1e-10


def test_uwc_metadata_is_saved_and_pipeline_runs_end_to_end(monkeypatch, tmp_path) -> None:
    _install_toy_hamiltonian(monkeypatch)
    result = analyze_partial_randomized_pf(
        2,
        epsilon_total=1e-2,
        uwc_config=UWCConfig(
            enabled=True,
            method="simple_shift",
            target_ld=1,
            parameters={"shift": 0.1, "qubit": 0},
        ),
        pf_labels=("2nd",),
        ld_values=(1, 2),
        kappa_mode="fixed",
        kappa_value=2.0,
        matrix_free_backend="python",
    )

    assert result.preprocessor == "uwc"
    assert result.uwc_method == "simple_shift"
    assert result.uwc_lambda_r_at_target_ld <= result.uwc_l1_norm
    assert np.isfinite(result.best.g_total)

    output_path = tmp_path / "uwc_result.json"
    save_partial_randomized_result(result, output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["preprocessor"] == "uwc"
    assert data["uwc_method"] == "simple_shift"
    assert data["uwc_target_ld"] == 1
    assert data["uwc_parameters"] == {"shift": 0.1, "qubit": 0}
    assert "uwc_hamiltonian_hash" in data
    assert "original_lambda_r_at_target_ld" in data
    assert "uwc_lambda_r_at_target_ld" in data

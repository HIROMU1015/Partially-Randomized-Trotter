from __future__ import annotations

import json

import numpy as np
import pytest
from openfermion.ops import QubitOperator

import trotterlib.grouped_uwc_comparison as guc
from trotterlib.grouped_uwc_comparison import (
    compare_grouped_uwc_pf_qpe,
    grouped_step_pauli_rotations,
    grouped_step_rz_layers,
    save_grouped_uwc_comparison,
)
from trotterlib.uwc import UWCConfig
from trotterlib.grouped_uwc_theta_sweep import (
    run_grouped_uwc_theta_sweep,
    save_grouped_uwc_theta_sweep,
)


def _toy_grouped_hamiltonian() -> QubitOperator:
    return (
        QubitOperator((), 0.05)
        + QubitOperator(((0, "Z"),), 0.7)
        + QubitOperator(((0, "X"),), 0.2)
    )


def _install_toy_grouped_system(monkeypatch) -> tuple[int, int]:
    hamiltonian = _toy_grouped_hamiltonian()
    grouped_ops, _name = guc.min_hamiltonian_grouper(hamiltonian, "toy")
    cliques = tuple((op,) for op in grouped_ops)
    step_pauli = grouped_step_pauli_rotations(cliques, "2nd")
    step_rz = grouped_step_rz_layers(cliques, "2nd")

    def fake_jw_hamiltonian_maker(molecule_type: int, distance: float = 1.0):
        del molecule_type, distance
        return hamiltonian, 0.0, "toy_h2", 2

    monkeypatch.setattr(guc, "jw_hamiltonian_maker", fake_jw_hamiltonian_maker)
    monkeypatch.setitem(guc.DECOMPO_NUM, "H2", {"2nd": step_pauli})
    monkeypatch.setitem(guc.PF_RZ_LAYER, "H2", {"2nd": step_rz})
    return step_pauli, step_rz


def test_grouped_baseline_uses_grouped_tables_not_original_ungrouped(monkeypatch) -> None:
    step_pauli, step_rz = _install_toy_grouped_system(monkeypatch)

    result = compare_grouped_uwc_pf_qpe(
        2,
        pf_label="2nd",
        uwc_config=UWCConfig(enabled=False),
        target_error=1e-2,
        t_values=(0.05, 0.06, 0.07),
        baseline_alpha_source="fit",
        use_reference_rz_layers=False,
    )

    baseline = result.rows[0]
    assert baseline.method == "grouped_baseline"
    assert baseline.step_pauli_rotations == step_pauli
    assert baseline.step_rz_layers == step_rz
    assert baseline.cost_ratio_vs_grouped_baseline == 1.0
    assert baseline.grouping_rule == "min_hamiltonian_grouper"


def test_uwc_grouped_recomputes_grouping_step_cost_and_ratios(monkeypatch) -> None:
    _install_toy_grouped_system(monkeypatch)

    result = compare_grouped_uwc_pf_qpe(
        2,
        pf_label="2nd",
        uwc_config=UWCConfig(
            enabled=True,
            method="simple_shift",
            parameters={"shift": 0.3, "qubit": 1},
        ),
        target_error=1e-2,
        cost_metric="pauli_rotations",
        t_values=(0.05, 0.06, 0.07),
        baseline_alpha_source="fit",
        reuse_original_ground_state_for_uwc=False,
        use_reference_rz_layers=False,
    )

    baseline, uwc = result.rows
    assert uwc.method == "uwc_grouped"
    assert uwc.uwc_method == "simple_shift"
    assert uwc.step_pauli_rotations > baseline.step_pauli_rotations
    assert uwc.step_cost_ratio_vs_grouped_baseline > 1.0
    assert np.isfinite(uwc.total_pauli_rotations)
    assert np.isfinite(uwc.total_rz_layers)
    assert np.isfinite(uwc.total_t_depth)
    assert uwc.metadata is not None
    assert uwc.metadata["uwc_step_pauli_source"] == "regrouped_uwc_hamiltonian"


def test_grouped_uwc_comparison_json_contains_requested_fields(monkeypatch, tmp_path) -> None:
    _install_toy_grouped_system(monkeypatch)
    result = compare_grouped_uwc_pf_qpe(
        2,
        pf_label="2nd",
        uwc_config=UWCConfig(
            enabled=True,
            method="simple_shift",
            parameters={"shift": 0.1, "qubit": 1},
        ),
        target_error=1e-2,
        t_values=(0.05, 0.06, 0.07),
        baseline_alpha_source="fit",
        reuse_original_ground_state_for_uwc=False,
        use_reference_rz_layers=False,
    )

    output_path = tmp_path / "grouped_uwc.json"
    save_grouped_uwc_comparison(result, output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    row = data["rows"][1]

    for key in (
        "molecule",
        "method",
        "uwc_method",
        "uwc_objective",
        "pf_label",
        "order",
        "grouping_rule",
        "num_groups",
        "num_pauli_terms",
        "alpha",
        "qpe_iteration_factor",
        "step_pauli_rotations",
        "total_pauli_rotations",
        "step_rz_layers",
        "total_rz_layers",
        "total_t_depth",
        "cost_ratio_vs_grouped_baseline",
        "alpha_ratio_vs_grouped_baseline",
        "step_cost_ratio_vs_grouped_baseline",
    ):
        assert key in row


def test_grouped_alpha_fit_can_use_gpu_runner(monkeypatch) -> None:
    _install_toy_grouped_system(monkeypatch)

    def fake_simulate_statevector_gpu(qc, psi0, *, gpu_ids, **_kwargs):
        from qiskit.quantum_info import Statevector

        evolved = Statevector(np.asarray(psi0, dtype=np.complex128)).evolve(qc)
        return np.asarray(evolved.data, dtype=np.complex128), {
            "backend": "fake_gpu",
            "gpu_ids": tuple(gpu_ids),
        }

    monkeypatch.setattr(guc, "simulate_statevector_gpu", fake_simulate_statevector_gpu)

    result = compare_grouped_uwc_pf_qpe(
        2,
        pf_label="2nd",
        uwc_config=UWCConfig(
            enabled=True,
            method="simple_shift",
            parameters={"shift": 0.1, "qubit": 1},
        ),
        target_error=1e-2,
        t_values=(0.05, 0.06, 0.07),
        baseline_alpha_source="fit",
        reuse_original_ground_state_for_uwc=False,
        use_reference_rz_layers=False,
        alpha_backend="gpu",
        alpha_gpu_ids=("0", "1"),
        alpha_parallel_processes=1,
    )

    baseline, uwc = result.rows
    assert baseline.metadata is not None
    assert baseline.metadata["alpha_fit_backend"] == "gpu"
    assert baseline.metadata["alpha_gpu_ids"] == ("0", "1")
    assert uwc.metadata is not None
    assert uwc.metadata["alpha_fit_backend"] == "gpu"
    assert uwc.metadata["alpha_gpu_ids"] == ("0", "1")
    assert np.isfinite(uwc.alpha)


def test_bliss_target_particle_number_defaults_to_grouped_reference_state(monkeypatch) -> None:
    hamiltonian = QubitOperator((), -1.0) + QubitOperator(((0, "Z"),), 0.2)
    cliques = ((QubitOperator(((0, "Z"),), 0.2),),)
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    data = guc.GroupedHamiltonianData(
        molecule_type=2,
        molecule="H2",
        basis="sto-3g",
        distance=1.0,
        ham_name="toy",
        num_qubits=2,
        jw_hamiltonian=hamiltonian,
        cliques=cliques,
        ground_energy=-0.8,
        state_vector=state,
        grouping_rule="toy_grouping",
        identity_coeff=-1.0,
    )

    monkeypatch.setattr(guc, "build_grouped_hamiltonian_data", lambda *_args, **_kwargs: data)
    monkeypatch.setitem(guc.DECOMPO_NUM, "H2", {"2nd": 1})
    monkeypatch.setitem(guc.PF_RZ_LAYER, "H2", {"2nd": 1})
    monkeypatch.setattr(
        guc,
        "fit_grouped_trotter_alpha",
        lambda *_args, **_kwargs: guc.GroupedAlphaFitResult(
            alpha=0.01,
            times=(0.05, 0.06, 0.07),
            errors=(2.5e-5, 3.6e-5, 4.9e-5),
            backend="cpu",
            requested_backend="cpu",
            gpu_ids=("0",),
            parallel_processes=1,
            chunk_splits=1,
            optimization_level=0,
            profiles=tuple(),
        ),
    )

    result = compare_grouped_uwc_pf_qpe(
        2,
        pf_label="2nd",
        uwc_config=UWCConfig(
            enabled=True,
            method="bliss",
            parameters={"theta": 0.1, "power": "quadratic"},
        ),
        target_error=1e-2,
        t_values=(0.05, 0.06, 0.07),
        baseline_alpha_source="fit",
        reuse_original_ground_state_for_uwc=True,
        use_reference_rz_layers=False,
    )

    uwc = result.rows[1]
    assert uwc.metadata is not None
    uwc_parameters = uwc.metadata["uwc_metadata"]["uwc_parameters"]
    assert uwc_parameters["target_particle_number"] == 0
    assert uwc_parameters["target_particle_number_source"] == "grouped_reference_state"
    check = uwc.metadata["uwc_metadata"]["sector_preservation_check"]
    assert check["ground_energy_difference"] < 1e-12



def _install_theta_sweep_sector_system(monkeypatch) -> None:
    hamiltonian = QubitOperator(((0, "X"),), -1.0)
    cliques = ((QubitOperator(((0, "X"),), -1.0),),)
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    data = guc.GroupedHamiltonianData(
        molecule_type=2,
        molecule="H2",
        basis="sto-3g",
        distance=1.0,
        ham_name="theta_toy",
        num_qubits=2,
        jw_hamiltonian=hamiltonian,
        cliques=cliques,
        ground_energy=0.0,
        state_vector=state,
        grouping_rule="theta_toy_grouping",
        identity_coeff=0.0,
    )

    def fake_fit(cliques, **kwargs):
        del kwargs
        term_count = sum(
            1
            for clique in cliques
            for operator in clique
            for term, coeff in operator.terms.items()
            if term != () and abs(complex(coeff)) > 1e-12
        )
        alpha = 0.01 * max(1, term_count)
        return guc.GroupedAlphaFitResult(
            alpha=alpha,
            times=(0.05, 0.06, 0.07),
            errors=tuple(alpha * value * value for value in (0.05, 0.06, 0.07)),
            backend="cpu",
            requested_backend="cpu",
            gpu_ids=("0",),
            parallel_processes=1,
            chunk_splits=1,
            optimization_level=0,
            profiles=tuple(),
        )

    monkeypatch.setattr(guc, "build_grouped_hamiltonian_data", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(guc, "fit_grouped_trotter_alpha", fake_fit)
    monkeypatch.setitem(guc.DECOMPO_NUM, "H2", {"2nd": 1})
    monkeypatch.setitem(guc.PF_RZ_LAYER, "H2", {"2nd": 1})


def test_theta_sweep_fit_baseline_theta_zero_sanity_and_save(monkeypatch, tmp_path) -> None:
    _install_theta_sweep_sector_system(monkeypatch)

    result = run_grouped_uwc_theta_sweep(
        (2,),
        pf_labels=("2nd",),
        theta_values=(0.0, 0.1),
        power="quadratic",
        target_error=1e-2,
        cost_metric="pauli_rotations",
        baseline_alpha_source="fit",
        alpha_backend="cpu",
        use_reference_rz_layers=False,
    )

    assert len(result.rows) == 4
    uwc_rows = [row for row in result.rows if row["method"] == "uwc_grouped"]
    assert [row["theta"] for row in uwc_rows] == [0.0, 0.1]

    theta_zero = uwc_rows[0]
    assert theta_zero["alpha_ratio_vs_grouped_baseline"] == 1.0
    assert theta_zero["step_cost_ratio_vs_grouped_baseline"] == 1.0
    assert theta_zero["cost_ratio_vs_grouped_baseline"] == 1.0
    assert result.summary[0]["theta_zero_sanity_passed"] is True

    nonzero = uwc_rows[1]
    assert nonzero["target_particle_number"] == 0
    assert nonzero["target_particle_number_source"] == "grouped_reference_state"
    assert nonzero["sector_preservation_check"]["checked"] is True
    assert nonzero["power"] == "quadratic"
    assert nonzero["theta"] == 0.1
    assert nonzero["uwc_step_pauli_rotations"] > nonzero["baseline_step_pauli_rotations"]
    assert result.summary[0]["any_step_cost_changed"] is True

    expected_cost_ratio = nonzero["step_cost_ratio_vs_grouped_baseline"] * np.sqrt(
        nonzero["alpha_ratio_vs_grouped_baseline"]
    )
    assert np.isclose(nonzero["cost_ratio_vs_grouped_baseline"], expected_cost_ratio)

    json_path, csv_path = save_grouped_uwc_theta_sweep(result, tmp_path / "theta.json")
    assert json_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 4
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 5


def test_theta_sweep_sector_error_propagates(monkeypatch) -> None:
    _install_theta_sweep_sector_system(monkeypatch)

    with pytest.raises(ValueError):
        run_grouped_uwc_theta_sweep(
            (2,),
            pf_labels=("2nd",),
            theta_values=(0.1,),
            power="quadratic",
            target_error=1e-2,
            cost_metric="pauli_rotations",
            baseline_alpha_source="fit",
            alpha_backend="cpu",
            use_reference_rz_layers=False,
            uwc_target_particle_number=3,
            uwc_sector_energy_check="error",
        )

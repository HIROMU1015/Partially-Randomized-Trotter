from __future__ import annotations

import numpy as np

from trotterlib.df_hamiltonian import (
    DFGroundStateResult,
    DFHamiltonian,
    PhysicalSector,
    _h_chain_integrals_session_cached,
    clear_df_integral_session_cache,
)
from trotterlib.df_partial_randomized_pf import (
    _analytic_d_only_rz_cost,
    _build_d_only_cost_circuit,
    _df_ground_state_cache_key_payload,
    _df_ground_state_result_from_npz,
    _df_hamiltonian_hash,
    _json_hash,
    _rz_depth_from_circuit,
    _save_df_ground_state_npz,
    fit_df_cgs_with_perturbation,
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)
from trotterlib.df_screening_cost import optimize_df_screening_cost
from trotterlib.df_trotter.model import Block, DFBlock, OneBodyGaussianBlock
from trotterlib.partial_randomized_pf import (
    default_perturbation_t_values,
    total_cost_given_q_kappa,
)


def _toy_df_hamiltonian() -> DFHamiltonian:
    one_body = np.diag([0.2, -0.1]).astype(np.complex128)
    g0 = np.diag([1.0, 0.0]).astype(np.complex128)
    g1 = np.diag([0.0, 2.0]).astype(np.complex128)
    return DFHamiltonian(
        constant=0.05,
        one_body=one_body,
        lambdas=np.asarray([0.5, -0.25], dtype=np.float64),
        g_matrices=(g0, g1),
        metadata={
            "molecule_type": 2,
            "distance": 1.0,
            "basis": "toy",
            "df_rank_requested": 2,
            "df_tol_requested": None,
            "df_rank_actual": 2,
        },
    )


def test_rank_and_split_df_fragments_use_consistent_weight_rule() -> None:
    hamiltonian = _toy_df_hamiltonian()
    ranked = rank_df_fragments(hamiltonian)

    assert [fragment.original_index for fragment in ranked] == [1, 0]
    assert [fragment.rank for fragment in ranked] == [0, 1]

    partition = split_df_hamiltonian_by_ld(
        hamiltonian,
        1,
        ranked_fragments=ranked,
    )
    assert partition.deterministic_block_indices == (1,)
    assert partition.randomized_block_indices == (0,)
    assert partition.lambda_r == ranked[1].weight
    assert partition.weight_rule == "lambda_frobenius_squared"


def test_df_hamiltonian_hash_depends_on_matrix_entries_not_just_norms() -> None:
    hamiltonian = _toy_df_hamiltonian()
    altered = DFHamiltonian(
        constant=hamiltonian.constant,
        one_body=hamiltonian.one_body,
        lambdas=hamiltonian.lambdas,
        g_matrices=(
            np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
            hamiltonian.g_matrices[1],
        ),
        metadata=hamiltonian.metadata,
    )

    assert np.linalg.norm(altered.g_matrices[0]) == np.linalg.norm(
        hamiltonian.g_matrices[0]
    )
    assert _df_hamiltonian_hash(
        altered,
        weight_rule="ground_state",
    ) != _df_hamiltonian_hash(
        hamiltonian,
        weight_rule="ground_state",
    )


def test_df_ground_state_npz_validates_cache_payload(tmp_path) -> None:
    hamiltonian = _toy_df_hamiltonian()
    sector = PhysicalSector(n_qubits=2, basis_indices=np.arange(4, dtype=np.int64))
    payload = _df_ground_state_cache_key_payload(
        hamiltonian=hamiltonian,
        sector=sector,
        matrix_free_backend="auto",
        matrix_free_threads=None,
        matrix_free_block_chunk_size=None,
        ground_state_ncv=None,
        ground_state_tol=1e-10,
    )
    cache_key = _json_hash(payload)
    path = tmp_path / f"{cache_key}.npz"
    result = DFGroundStateResult(
        energy=-1.0,
        state_vector=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        sector_state_vector=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        sector=sector,
        converged=True,
        residual_norm=0.0,
        matvec_count=1,
        elapsed_s=0.0,
        solver="eigsh",
        message="converged",
    )

    _save_df_ground_state_npz(
        path,
        result,
        cache_key=cache_key,
        cache_payload=payload,
    )

    assert (
        _df_ground_state_result_from_npz(
            path,
            sector,
            expected_cache_key=cache_key,
            expected_cache_payload=payload,
        )
        is not None
    )
    mismatched_payload = {**payload, "ground_state_tol": 1e-8}
    assert (
        _df_ground_state_result_from_npz(
            path,
            sector,
            expected_cache_key=cache_key,
            expected_cache_payload=mismatched_payload,
        )
        is None
    )


def test_df_integral_session_cache_reuses_first_integrals(monkeypatch) -> None:
    clear_df_integral_session_cache()
    calls = {"count": 0}

    class FakeInteraction:
        def __init__(self, value: float) -> None:
            self.constant = value
            self.one_body_tensor = np.eye(4, dtype=np.complex128) * value
            self.two_body_tensor = np.zeros((4, 4, 4, 4), dtype=np.complex128)

    class FakeMolecule:
        def __init__(self, value: float) -> None:
            self.hf_energy = value
            self._interaction = FakeInteraction(value)

        def get_molecular_hamiltonian(self) -> FakeInteraction:
            return self._interaction

    def fake_run_pyscf(*_args, **_kwargs):
        calls["count"] += 1
        return FakeMolecule(float(calls["count"]))

    monkeypatch.setattr("trotterlib.df_hamiltonian.run_pyscf", fake_run_pyscf)

    first = _h_chain_integrals_session_cached(2, distance=1.0, basis="toy")
    second = _h_chain_integrals_session_cached(2, distance=1.0, basis="toy")
    assert calls["count"] == 1
    assert np.array_equal(first["one_body"], second["one_body"])

    clear_df_integral_session_cache()
    third = _h_chain_integrals_session_cached(2, distance=1.0, basis="toy")
    assert calls["count"] == 2
    assert not np.array_equal(first["one_body"], third["one_body"])


def test_df_cgs_fit_cpu_path_returns_df_surrogate_metadata() -> None:
    hamiltonian = _toy_df_hamiltonian()
    sector = PhysicalSector(n_qubits=2, basis_indices=np.arange(4, dtype=np.int64))
    partition = split_df_hamiltonian_by_ld(hamiltonian, 1)

    result = fit_df_cgs_with_perturbation(
        hamiltonian,
        sector,
        partition,
        "2nd",
        t_values=(0.05, 0.06),
        evolution_backend="cpu",
    )

    assert result.representation_type == "df"
    assert result.ld == 1
    assert result.lambda_r == partition.lambda_r
    assert result.evolution_backend == "cpu"
    assert "deterministic surrogate" in result.metadata["surrogate_note"]
    assert len(result.perturbation_errors) == 2


def test_default_perturbation_time_point_count_by_molecule_size() -> None:
    assert len(default_perturbation_t_values(11, "2nd")) == 4
    assert len(default_perturbation_t_values(12, "2nd")) == 3
    assert default_perturbation_t_values(12, "8th(Morales)") == (
        0.62,
        0.622,
        0.624,
    )


def test_error_budget_rule_switches_trotter_remainder() -> None:
    linear = total_cost_given_q_kappa(
        epsilon_total=1e-4,
        q_ratio=0.25,
        order=2,
        deterministic_step_cost_value=100,
        c_gs=1e-3,
        lambda_r=0.0,
        kappa=None,
        error_budget_rule="linear",
    )
    quadrature = total_cost_given_q_kappa(
        epsilon_total=1e-4,
        q_ratio=0.25,
        order=2,
        deterministic_step_cost_value=100,
        c_gs=1e-3,
        lambda_r=0.0,
        kappa=None,
        error_budget_rule="quadrature",
    )

    assert np.isclose(linear.eps_qpe, 2.5e-5)
    assert np.isclose(linear.eps_trot, 7.5e-5)
    assert np.isclose(quadrature.eps_qpe, 2.5e-5)
    assert np.isclose(quadrature.eps_trot, 1e-4 * np.sqrt(1.0 - 0.25**2))


def test_analytic_d_only_cost_matches_transpiled_reference() -> None:
    blocks = (
        Block.from_one_body_gaussian(
            OneBodyGaussianBlock(
                U_ops=(),
                eps=np.asarray([0.2, 0.0, -0.3], dtype=np.float64),
            )
        ),
        Block.from_df(
            DFBlock(
                U_ops=(),
                eta=np.asarray([1.0, -0.5, 0.25], dtype=np.float64),
                lam=0.7,
            )
        ),
    )

    for pf_label in ("2nd", "4th", "8th(Morales)"):
        analytic = _analytic_d_only_rz_cost(
            blocks,
            time=1.0,
            num_qubits=3,
            pf_label=pf_label,
        )
        reference_qc = _build_d_only_cost_circuit(
            blocks,
            time=1.0,
            num_qubits=3,
            pf_label=pf_label,
        )
        reference = _rz_depth_from_circuit(reference_qc)

        assert analytic["rz_count"] == reference["rz_count"]
        assert analytic["rz_depth"] == reference["rz_depth"]


def test_optimize_df_screening_cost_uses_anchor_cgs_table(tmp_path) -> None:
    cgs_table = tmp_path / "df_cgs_cost_table.json"
    cgs_table.write_text(
        """
        {
          "schema_version": 1,
          "table_type": "df_cgs_cost_input",
          "entries": [
            {
              "molecule_type": 3,
              "pf_label": "2nd",
              "order": 2,
              "ld": 2,
              "ld_anchor": 2,
              "is_screening_anchor": true,
              "source_kind": "screening_anchor",
              "df_rank_actual": 5,
              "lambda_r": 0.1,
              "c_gs_d": 0.002,
              "total_ref_rz_depth": 1023
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    result = optimize_df_screening_cost(
        cgs_table_path=cgs_table,
        epsilon_total=1e-3,
        molecule_min=3,
        molecule_max=3,
        pf_labels=("2nd",),
        kappa_mode="fixed",
        kappa_value=2.0,
    )

    assert result["model"] == "df_reduced_screening_cost_minimization_v1"
    assert result["best_overall"] is not None
    assert result["best_overall"]["molecule_type"] == 3
    assert result["best_overall"]["pf_label"] == "2nd"
    assert len(result["candidates"]) == 6

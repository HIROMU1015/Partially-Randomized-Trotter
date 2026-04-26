from __future__ import annotations

import numpy as np

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.df_partial_randomized_pf import (
    _analytic_d_only_rz_cost,
    _build_d_only_cost_circuit,
    _rz_depth_from_circuit,
    fit_df_cgs_with_perturbation,
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)
from trotterlib.df_screening_cost import optimize_df_screening_cost
from trotterlib.df_trotter.model import Block, DFBlock, OneBodyGaussianBlock
from trotterlib.partial_randomized_pf import default_perturbation_t_values


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

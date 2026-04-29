from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import (
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR,
    PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    PARTIAL_RANDOMIZED_KAPPA_MAX,
    PARTIAL_RANDOMIZED_KAPPA_MIN,
    pf_order,
)
from .df_hamiltonian import DFHamiltonian, build_df_h_d_from_molecule
from .df_partial_randomized_pf import (
    _DF_COST_BASIS_GATES,
    _analytic_d_only_rz_cost,
    _u_ops_rz_cost,
    df_hamiltonian_to_model,
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)
from .df_trotter.model import Block, DFModel
from .df_trotter.ops import (
    build_df_blocks_givens,
    build_one_body_gaussian_block_givens,
)
from .partial_randomized_pf import (
    optimize_error_budget_and_kappa,
    randomized_prefactor_b0,
)
from .pf_decomposition import iter_pf_steps
from .product_formula import _get_w_list


DEFAULT_DF_CGS_COST_TABLE = PARTIAL_RANDOMIZED_ARTIFACTS_DIR / "df_cgs_cost_table.json"
DEFAULT_DF_SCREENING_COST_OUTPUT = (
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR
    / "screening_results"
    / "df_screening_cost_minimization_eps_1.000e-04.json"
)
DF_SCREENING_COST_MODEL = "df_reduced_screening_cost_minimization_v1"


def load_df_anchor_cgs_table(path: str | Path) -> dict[int, dict[str, dict[str, Any]]]:
    """Load screening-anchor Cgs entries grouped by molecule_type and PF label."""
    table_path = Path(path)
    document = json.loads(table_path.read_text(encoding="utf-8"))
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for raw in document.get("entries", []):
        if not isinstance(raw, dict) or not raw.get("is_screening_anchor"):
            continue
        molecule_type = int(raw["molecule_type"])
        pf_label = str(raw["pf_label"])
        grouped.setdefault(molecule_type, {})[pf_label] = raw
    return grouped


def build_rank_ordered_df_cost_blocks(
    molecule_type: int,
) -> tuple[DFHamiltonian, list[Block]]:
    """
    Build one-body plus rank-ordered DF blocks used by the screening cost model.

    The returned block list is ordered as [one_body, ranked fragment 0, ...] when
    a nonzero one-body correction exists.
    """
    hamiltonian, _sector = build_df_h_d_from_molecule(int(molecule_type))
    ranked = rank_df_fragments(hamiltonian)
    full_model = df_hamiltonian_to_model(hamiltonian)
    blocks: list[Block] = []
    if np.linalg.norm(full_model.one_body_correction) > 1e-14:
        blocks.append(
            Block.from_one_body_gaussian(
                build_one_body_gaussian_block_givens(
                    full_model.one_body_correction,
                    sort="descending_abs",
                )
            )
        )
    for fragment in ranked:
        idx = int(fragment.original_index)
        one_block_model = DFModel(
            lambdas=np.asarray([hamiltonian.lambdas[idx]]),
            G_list=[np.asarray(hamiltonian.g_matrices[idx])],
            one_body_correction=np.zeros_like(hamiltonian.one_body),
            constant_correction=0.0,
            N=hamiltonian.n_qubits,
        )
        df_blocks = build_df_blocks_givens(one_block_model, sort="descending_abs")
        if len(df_blocks) != 1:
            raise RuntimeError("Expected one DF block for one ranked fragment.")
        blocks.append(Block.from_df(df_blocks[0]))
    return hamiltonian, blocks


def df_screening_costs_for_all_ld(
    *,
    hamiltonian: DFHamiltonian,
    blocks: Sequence[Block],
    pf_label: str,
    decompose_reps: int = 8,
    optimization_level: int = 0,
) -> dict[int, dict[str, int]]:
    """Return total_ref_rz_depth/count estimates for every LD in the reduced model."""
    has_one_body = bool(blocks and blocks[0].kind == "one_body_gaussian")
    u_block_costs = [
        _u_ops_rz_cost(
            block.payload.U_ops,
            hamiltonian.n_qubits,
            basis_gates=_DF_COST_BASIS_GATES,
            decompose_reps=decompose_reps,
            optimization_level=optimization_level,
        )
        for block in blocks
    ]
    costs_by_ld: dict[int, dict[str, int]] = {}
    for ld in range(0, hamiltonian.n_blocks + 1):
        stop = ld + 1 if has_one_body else ld
        active_blocks = list(blocks[:stop])
        u_count = 0
        u_depth = 0
        for term_idx, _weight in iter_pf_steps(len(active_blocks), _get_w_list(pf_label)):
            cost = u_block_costs[int(term_idx)]
            u_count += 2 * int(cost["rz_count"])
            u_depth += 2 * int(cost["rz_depth"])
        d_cost = _analytic_d_only_rz_cost(
            active_blocks,
            time=1.0,
            num_qubits=hamiltonian.n_qubits,
            pf_label=pf_label,
        )
        d_count = int(d_cost["rz_count"])
        d_depth = int(d_cost["rz_depth"])
        costs_by_ld[ld] = {
            "u_ref_rz_count": int(u_count),
            "u_ref_rz_depth": int(u_depth),
            "d_ref_rz_count": d_count,
            "d_ref_rz_depth": d_depth,
            "total_ref_rz_count": int(u_count + d_count),
            "total_ref_rz_depth": int(u_depth + d_depth),
        }
    return costs_by_ld


def optimize_df_screening_cost(
    *,
    cgs_table_path: str | Path = DEFAULT_DF_CGS_COST_TABLE,
    epsilon_total: float = 1e-4,
    molecule_min: int | None = None,
    molecule_max: int | None = None,
    pf_labels: Sequence[str] | None = None,
    kappa_mode: str = "optimize",
    kappa_value: float = PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    kappa_min: float = PARTIAL_RANDOMIZED_KAPPA_MIN,
    kappa_max: float = PARTIAL_RANDOMIZED_KAPPA_MAX,
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    error_budget_rule: str = "quadrature",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Optimize the DF reduced screening model using anchor Cgs values."""
    anchor_by_molecule = load_df_anchor_cgs_table(cgs_table_path)
    pf_filter = None if pf_labels is None else {str(label) for label in pf_labels}

    molecule_types = sorted(anchor_by_molecule)
    if molecule_min is not None:
        molecule_types = [value for value in molecule_types if value >= int(molecule_min)]
    if molecule_max is not None:
        molecule_types = [value for value in molecule_types if value <= int(molecule_max)]

    all_candidates: list[dict[str, Any]] = []
    best_by_molecule: list[dict[str, Any]] = []
    b0 = randomized_prefactor_b0(randomized_method, g_rand)

    for molecule_type in molecule_types:
        hamiltonian, blocks = build_rank_ordered_df_cost_blocks(molecule_type)
        ranked = rank_df_fragments(hamiltonian)
        molecule_candidates: list[dict[str, Any]] = []
        pf_entries = anchor_by_molecule[molecule_type]
        for pf_label in sorted(pf_entries, key=lambda label: pf_order(label)):
            if pf_filter is not None and pf_label not in pf_filter:
                continue
            anchor = pf_entries[pf_label]
            costs_by_ld = df_screening_costs_for_all_ld(
                hamiltonian=hamiltonian,
                blocks=blocks,
                pf_label=pf_label,
            )
            for ld in range(0, hamiltonian.n_blocks + 1):
                partition = split_df_hamiltonian_by_ld(
                    hamiltonian,
                    ld,
                    ranked_fragments=ranked,
                )
                step_cost = int(costs_by_ld[ld]["total_ref_rz_depth"])
                budget = optimize_error_budget_and_kappa(
                    epsilon_total=float(epsilon_total),
                    order=pf_order(pf_label),
                    deterministic_step_cost_value=step_cost,
                    c_gs=float(anchor["c_gs_d"]),
                    lambda_r=float(partition.lambda_r),
                    randomized_method=randomized_method,
                    g_rand=float(g_rand),
                    kappa_mode=kappa_mode,
                    kappa_value=float(kappa_value),
                    kappa_min=float(kappa_min),
                    kappa_max=float(kappa_max),
                    error_budget_rule=error_budget_rule,
                )
                candidate = {
                    "molecule": f"H{molecule_type}",
                    "molecule_type": int(molecule_type),
                    "df_rank_actual": int(hamiltonian.n_blocks),
                    "pf_label": pf_label,
                    "order": pf_order(pf_label),
                    "ld": int(ld),
                    "ld_anchor": int(anchor["ld_anchor"]),
                    "cgs_source_kind": str(anchor.get("source_kind", "screening_anchor")),
                    "c_gs_d_screen": float(anchor["c_gs_d"]),
                    "lambda_r": float(partition.lambda_r),
                    **costs_by_ld[ld],
                    "q_opt": budget.q_ratio,
                    "eps_qpe_opt": budget.eps_qpe,
                    "eps_trot_opt": budget.eps_trot,
                    "kappa_opt": budget.kappa,
                    "b_opt": budget.b_value,
                    "error_budget_rule": str(error_budget_rule),
                    "boundary_hit_q": budget.boundary_hit_q,
                    "boundary_hit_kappa": budget.boundary_hit_kappa,
                    "g_det": budget.g_det,
                    "g_rand": budget.g_rand,
                    "g_total": budget.g_total,
                }
                all_candidates.append(candidate)
                molecule_candidates.append(candidate)
        if molecule_candidates:
            best = min(molecule_candidates, key=lambda item: float(item["g_total"]))
            best_by_molecule.append(best)
            if progress_callback is not None:
                progress_callback(best)

    best_overall = (
        min(all_candidates, key=lambda item: float(item["g_total"]))
        if all_candidates
        else None
    )
    return {
        "schema_version": 1,
        "model": DF_SCREENING_COST_MODEL,
        "epsilon_total": float(epsilon_total),
        "cgs_rule": "C_gs,D(p,L_D)=C_gs,D(p,L_anchor=floor(rank/2))",
        "cost_rule": "total_ref_rz_depth(p,L_D) with analytic D-only RZ-depth",
        "randomized_method": randomized_method,
        "g_rand_input": float(g_rand),
        "b0": float(b0),
        "kappa_mode": kappa_mode,
        "kappa_value": float(kappa_value),
        "kappa_min": float(kappa_min),
        "kappa_max": float(kappa_max),
        "error_budget_rule": str(error_budget_rule),
        "cgs_table": str(Path(cgs_table_path)),
        "best_overall": best_overall,
        "best_by_molecule": best_by_molecule,
        "candidates": all_candidates,
    }


def save_df_screening_cost_result(result: dict[str, Any], path: str | Path) -> None:
    """Save an optimize_df_screening_cost result document as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

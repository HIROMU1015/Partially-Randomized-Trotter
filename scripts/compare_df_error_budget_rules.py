from __future__ import annotations

from compare_df_error_budget_rules_from_summaries import main as _summary_main

if __name__ == "__main__":
    raise SystemExit(_summary_main())

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.config import (  # noqa: E402
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR,
    PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    PARTIAL_RANDOMIZED_KAPPA_MAX,
    PARTIAL_RANDOMIZED_KAPPA_MIN,
    pf_order,
)
from trotterlib.df_cost_plotting import (  # noqa: E402
    DEFAULT_ANCHOR_REFINEMENT_DIR,
    load_anchor_cgs_rows,
)
from trotterlib.df_partial_randomized_pf import (  # noqa: E402
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)
from trotterlib.df_screening_cost import (  # noqa: E402
    build_rank_ordered_df_cost_blocks,
    df_screening_costs_for_all_ld,
)
from trotterlib.partial_randomized_pf import (  # noqa: E402
    optimize_error_budget_and_kappa,
    randomized_prefactor_b0,
)


DEFAULT_OUTPUT = (
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR
    / "screening_results"
    / "df_error_budget_rule_comparison_anchor_eps_1.000e-04.json"
)
DEFAULT_PF_LABELS = ("2nd", "4th", "4th(new_2)", "8th(Morales)")
ERROR_BUDGET_RULES = ("quadrature", "linear")


def _filter_anchor_rows(
    rows: Sequence[dict[str, Any]],
    *,
    molecule_min: int | None,
    molecule_max: int | None,
    pf_labels: Sequence[str] | None,
) -> list[dict[str, Any]]:
    pf_filter = None if pf_labels is None else {str(label) for label in pf_labels}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        try:
            molecule_type = int(row["molecule_type"])
            pf_label = str(row["pf_label"])
        except (KeyError, TypeError, ValueError):
            continue
        if molecule_min is not None and molecule_type < int(molecule_min):
            continue
        if molecule_max is not None and molecule_type > int(molecule_max):
            continue
        if pf_filter is not None and pf_label not in pf_filter:
            continue
        filtered.append(row)
    return filtered


def _dedupe_anchor_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["molecule_type"]), str(row["pf_label"]))
        by_key[key] = dict(row)
    return [by_key[key] for key in sorted(by_key, key=lambda item: (item[0], pf_order(item[1])))]


def _candidate_for_ld(
    *,
    row: dict[str, Any],
    hamiltonian: Any,
    ranked: Sequence[Any],
    step_costs: dict[str, int],
    ld: int,
    epsilon_total: float,
    kappa_mode: str,
    kappa_value: float,
    kappa_min: float,
    kappa_max: float,
    randomized_method: str,
    g_rand: float,
    error_budget_rule: str,
) -> dict[str, Any]:
    pf_label = str(row["pf_label"])
    partition = split_df_hamiltonian_by_ld(
        hamiltonian,
        int(ld),
        ranked_fragments=ranked,
    )
    budget = optimize_error_budget_and_kappa(
        epsilon_total=float(epsilon_total),
        order=pf_order(pf_label),
        deterministic_step_cost_value=int(step_costs["total_ref_rz_depth"]),
        c_gs=float(row["c_gs_d"]),
        lambda_r=float(partition.lambda_r),
        randomized_method=randomized_method,
        g_rand=float(g_rand),
        kappa_mode=kappa_mode,
        kappa_value=float(kappa_value),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),
        error_budget_rule=error_budget_rule,
    )
    molecule_type = int(row["molecule_type"])
    return {
        "molecule": f"H{molecule_type}",
        "molecule_type": molecule_type,
        "df_rank_actual": int(hamiltonian.n_blocks),
        "pf_label": pf_label,
        "order": pf_order(pf_label),
        "ld": int(ld),
        "ld_anchor": int(row["ld_anchor"]),
        "anchor_source": str(row.get("_artifact_source")),
        "c_gs_d_screen": float(row["c_gs_d"]),
        "lambda_r": float(partition.lambda_r),
        **step_costs,
        "q_opt": budget.q_ratio,
        "eps_qpe_opt": budget.eps_qpe,
        "eps_trot_opt": budget.eps_trot,
        "kappa_opt": budget.kappa,
        "b_opt": budget.b_value,
        "boundary_hit_q": budget.boundary_hit_q,
        "boundary_hit_kappa": budget.boundary_hit_kappa,
        "error_budget_rule": error_budget_rule,
        "g_det": budget.g_det,
        "g_rand": budget.g_rand,
        "g_total": budget.g_total,
    }


def compare_error_budget_rules(
    *,
    anchor_dir: str | Path = DEFAULT_ANCHOR_REFINEMENT_DIR,
    anchor_paths: Sequence[str | Path] | None = None,
    molecule_min: int | None = None,
    molecule_max: int | None = None,
    pf_labels: Sequence[str] | None = DEFAULT_PF_LABELS,
    epsilon_total: float = 1e-4,
    kappa_mode: str = "optimize",
    kappa_value: float = PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    kappa_min: float = PARTIAL_RANDOMIZED_KAPPA_MIN,
    kappa_max: float = PARTIAL_RANDOMIZED_KAPPA_MAX,
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
) -> dict[str, Any]:
    anchor_rows = load_anchor_cgs_rows(anchor_dir, paths=anchor_paths)
    anchor_rows = _filter_anchor_rows(
        anchor_rows,
        molecule_min=molecule_min,
        molecule_max=molecule_max,
        pf_labels=pf_labels,
    )
    anchor_rows = _dedupe_anchor_rows(anchor_rows)

    rows_by_molecule: dict[int, list[dict[str, Any]]] = {}
    for row in anchor_rows:
        rows_by_molecule.setdefault(int(row["molecule_type"]), []).append(row)

    comparisons: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for molecule_type in sorted(rows_by_molecule):
        hamiltonian, blocks = build_rank_ordered_df_cost_blocks(molecule_type)
        ranked = rank_df_fragments(hamiltonian)
        for row in sorted(rows_by_molecule[molecule_type], key=lambda item: pf_order(str(item["pf_label"]))):
            pf_label = str(row["pf_label"])
            costs_by_ld = df_screening_costs_for_all_ld(
                hamiltonian=hamiltonian,
                blocks=blocks,
                pf_label=pf_label,
            )
            candidates_by_rule: dict[str, list[dict[str, Any]]] = {}
            best_by_rule: dict[str, dict[str, Any]] = {}
            for error_budget_rule in ERROR_BUDGET_RULES:
                candidates = [
                    _candidate_for_ld(
                        row=row,
                        hamiltonian=hamiltonian,
                        ranked=ranked,
                        step_costs=costs_by_ld[ld],
                        ld=ld,
                        epsilon_total=epsilon_total,
                        kappa_mode=kappa_mode,
                        kappa_value=kappa_value,
                        kappa_min=kappa_min,
                        kappa_max=kappa_max,
                        randomized_method=randomized_method,
                        g_rand=g_rand,
                        error_budget_rule=error_budget_rule,
                    )
                    for ld in range(0, int(hamiltonian.n_blocks) + 1)
                ]
                candidates_by_rule[error_budget_rule] = candidates
                best_by_rule[error_budget_rule] = min(
                    candidates,
                    key=lambda item: float(item["g_total"]),
                )
                all_candidates.extend(candidates)

            quadrature_best = best_by_rule["quadrature"]
            linear_best = best_by_rule["linear"]
            comparisons.append(
                {
                    "molecule": f"H{molecule_type}",
                    "molecule_type": molecule_type,
                    "pf_label": pf_label,
                    "order": pf_order(pf_label),
                    "df_rank_actual": int(hamiltonian.n_blocks),
                    "ld_anchor": int(row["ld_anchor"]),
                    "c_gs_d_screen": float(row["c_gs_d"]),
                    "anchor_source": str(row.get("_artifact_source")),
                    "quadrature_best": quadrature_best,
                    "linear_best": linear_best,
                    "ld_changed": int(quadrature_best["ld"]) != int(linear_best["ld"]),
                    "delta_ld_linear_minus_quadrature": (
                        int(linear_best["ld"]) - int(quadrature_best["ld"])
                    ),
                }
            )

    changed = [item for item in comparisons if item["ld_changed"]]
    return {
        "schema_version": 1,
        "model": "df_anchor_screening_error_budget_rule_comparison_v1",
        "epsilon_total": float(epsilon_total),
        "rules": {
            "quadrature": "eps_qpe^2 + eps_trot^2 = epsilon_total^2",
            "linear": "eps_qpe + eps_trot = epsilon_total",
        },
        "q_ratio_definition": "q_ratio = eps_qpe / epsilon_total",
        "cgs_rule": "Use existing screening-anchor Cgs rows without recomputing Cgs.",
        "anchor_dir": str(Path(anchor_dir)),
        "anchor_paths": None if anchor_paths is None else [str(path) for path in anchor_paths],
        "molecule_min": molecule_min,
        "molecule_max": molecule_max,
        "pf_labels": None if pf_labels is None else [str(label) for label in pf_labels],
        "randomized_method": randomized_method,
        "g_rand_input": float(g_rand),
        "b0": randomized_prefactor_b0(randomized_method, g_rand),
        "kappa_mode": kappa_mode,
        "kappa_value": float(kappa_value),
        "kappa_min": float(kappa_min),
        "kappa_max": float(kappa_max),
        "num_anchor_rows": len(anchor_rows),
        "num_comparisons": len(comparisons),
        "num_ld_changed": len(changed),
        "changed": changed,
        "comparisons": comparisons,
        "candidates": all_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DF screening best LD under quadrature and linear error-budget rules "
            "using existing anchor Cgs rows."
        )
    )
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_REFINEMENT_DIR)
    parser.add_argument("--anchor-path", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--molecule-min", type=int)
    parser.add_argument("--molecule-max", type=int)
    parser.add_argument("--pf-label", action="append", dest="pf_labels")
    parser.add_argument("--epsilon-total", type=float, default=1e-4)
    parser.add_argument("--kappa-mode", default="optimize")
    parser.add_argument("--kappa-value", type=float, default=PARTIAL_RANDOMIZED_DEFAULT_KAPPA)
    parser.add_argument("--kappa-min", type=float, default=PARTIAL_RANDOMIZED_KAPPA_MIN)
    parser.add_argument("--kappa-max", type=float, default=PARTIAL_RANDOMIZED_KAPPA_MAX)
    parser.add_argument("--randomized-method", default=PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD)
    parser.add_argument("--g-rand", type=float, default=PARTIAL_RANDOMIZED_DEFAULT_G_RAND)
    args = parser.parse_args()

    result = compare_error_budget_rules(
        anchor_dir=args.anchor_dir,
        anchor_paths=args.anchor_path,
        molecule_min=args.molecule_min,
        molecule_max=args.molecule_max,
        pf_labels=tuple(args.pf_labels) if args.pf_labels else DEFAULT_PF_LABELS,
        epsilon_total=float(args.epsilon_total),
        kappa_mode=str(args.kappa_mode),
        kappa_value=float(args.kappa_value),
        kappa_min=float(args.kappa_min),
        kappa_max=float(args.kappa_max),
        randomized_method=str(args.randomized_method),
        g_rand=float(args.g_rand),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"({result['num_ld_changed']}/{result['num_comparisons']} LD changes)"
    )
    for item in result["changed"]:
        print(
            f"H{item['molecule_type']} {item['pf_label']}: "
            f"{item['quadrature_best']['ld']} -> {item['linear_best']['ld']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

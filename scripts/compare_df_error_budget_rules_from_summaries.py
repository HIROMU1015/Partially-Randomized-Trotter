from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


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
    load_anchor_refinement_summaries,
)
from trotterlib.partial_randomized_pf import (  # noqa: E402
    optimize_error_budget_and_kappa,
    randomized_prefactor_b0,
)


DEFAULT_OUTPUT = (
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR
    / "screening_results"
    / "df_error_budget_rule_comparison_from_summaries_eps_1.000e-04.json"
)
DEFAULT_PF_LABELS = ("2nd", "4th", "4th(new_2)", "8th(Morales)")
ERROR_BUDGET_RULES = ("quadrature", "linear")


def _passes_filter(
    row: Mapping[str, Any],
    *,
    molecule_min: int | None,
    molecule_max: int | None,
    pf_labels: Sequence[str] | None,
) -> bool:
    molecule_type = int(row["molecule_type"])
    pf_label = str(row["pf_label"])
    if molecule_min is not None and molecule_type < int(molecule_min):
        return False
    if molecule_max is not None and molecule_type > int(molecule_max):
        return False
    if pf_labels is not None and pf_label not in {str(label) for label in pf_labels}:
        return False
    return True


def _recompute_candidate(
    candidate: Mapping[str, Any],
    *,
    error_budget_rule: str,
    epsilon_total: float,
    kappa_mode: str,
    kappa_value: float,
    kappa_min: float,
    kappa_max: float,
    randomized_method: str,
    g_rand: float,
) -> dict[str, Any]:
    pf_label = str(candidate["pf_label"])
    budget = optimize_error_budget_and_kappa(
        epsilon_total=float(epsilon_total),
        order=int(candidate.get("order", pf_order(pf_label))),
        deterministic_step_cost_value=int(candidate["total_ref_rz_depth"]),
        c_gs=float(candidate["c_gs_d_screen"]),
        lambda_r=float(candidate["lambda_r"]),
        randomized_method=randomized_method,
        g_rand=float(g_rand),
        kappa_mode=kappa_mode,
        kappa_value=float(kappa_value),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),
        error_budget_rule=error_budget_rule,
    )
    return {
        "molecule": str(candidate["molecule"]),
        "molecule_type": int(candidate["molecule_type"]),
        "df_rank_actual": int(candidate["df_rank_actual"]),
        "pf_label": pf_label,
        "order": int(candidate.get("order", pf_order(pf_label))),
        "ld": int(candidate["ld"]),
        "ld_anchor": int(candidate["ld_anchor"]),
        "c_gs_d_screen": float(candidate["c_gs_d_screen"]),
        "lambda_r": float(candidate["lambda_r"]),
        "total_ref_rz_count": int(candidate["total_ref_rz_count"]),
        "total_ref_rz_depth": int(candidate["total_ref_rz_depth"]),
        "u_ref_rz_count": int(candidate["u_ref_rz_count"]),
        "u_ref_rz_depth": int(candidate["u_ref_rz_depth"]),
        "d_ref_rz_count": int(candidate["d_ref_rz_count"]),
        "d_ref_rz_depth": int(candidate["d_ref_rz_depth"]),
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


def compare_error_budget_rules_from_summaries(
    *,
    summary_dir: str | Path = DEFAULT_ANCHOR_REFINEMENT_DIR,
    summary_paths: Sequence[str | Path] | None = None,
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
    summaries = load_anchor_refinement_summaries(summary_dir, paths=summary_paths)
    summaries = [
        summary
        for summary in summaries
        if _passes_filter(
            summary,
            molecule_min=molecule_min,
            molecule_max=molecule_max,
            pf_labels=pf_labels,
        )
    ]

    comparisons: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for summary in sorted(
        summaries,
        key=lambda item: (int(item["molecule_type"]), pf_order(str(item["pf_label"]))),
    ):
        raw_candidates = [
            item
            for item in summary.get("screening_candidates", [])
            if isinstance(item, Mapping)
        ]
        if not raw_candidates:
            skipped.append(
                {
                    "molecule_type": int(summary["molecule_type"]),
                    "pf_label": str(summary["pf_label"]),
                    "reason": "missing screening_candidates",
                    "source": str(summary.get("_artifact_source")),
                }
            )
            continue

        best_by_rule: dict[str, dict[str, Any]] = {}
        for error_budget_rule in ERROR_BUDGET_RULES:
            candidates = [
                _recompute_candidate(
                    candidate,
                    error_budget_rule=error_budget_rule,
                    epsilon_total=epsilon_total,
                    kappa_mode=kappa_mode,
                    kappa_value=kappa_value,
                    kappa_min=kappa_min,
                    kappa_max=kappa_max,
                    randomized_method=randomized_method,
                    g_rand=g_rand,
                )
                for candidate in raw_candidates
            ]
            best_by_rule[error_budget_rule] = min(
                candidates,
                key=lambda item: float(item["g_total"]),
            )
            all_candidates.extend(candidates)

        quadrature_best = best_by_rule["quadrature"]
        linear_best = best_by_rule["linear"]
        comparisons.append(
            {
                "molecule": str(summary["molecule"]),
                "molecule_type": int(summary["molecule_type"]),
                "pf_label": str(summary["pf_label"]),
                "order": int(summary["order"]),
                "df_rank_actual": int(summary["df_rank_actual"]),
                "ld_anchor": int(summary["ld_anchor"]),
                "anchor_c_gs_d": float(summary["anchor_c_gs_d"]),
                "source": str(summary.get("_artifact_source")),
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
        "model": "df_anchor_summary_error_budget_rule_comparison_v1",
        "epsilon_total": float(epsilon_total),
        "rules": {
            "quadrature": "eps_qpe^2 + eps_trot^2 = epsilon_total^2",
            "linear": "eps_qpe + eps_trot = epsilon_total",
        },
        "q_ratio_definition": "q_ratio = eps_qpe / epsilon_total",
        "source_rule": (
            "Reuse existing anchor-refinement screening_candidates; do not recompute Cgs "
            "or DF step costs."
        ),
        "summary_dir": str(Path(summary_dir)),
        "summary_paths": None if summary_paths is None else [str(path) for path in summary_paths],
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
        "num_summaries": len(summaries),
        "num_comparisons": len(comparisons),
        "num_ld_changed": len(changed),
        "changed": changed,
        "skipped": skipped,
        "comparisons": comparisons,
        "candidates": all_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DF screening best LD under quadrature and linear error-budget rules "
            "using existing anchor-refinement summaries."
        )
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_ANCHOR_REFINEMENT_DIR)
    parser.add_argument("--summary-path", type=Path, action="append")
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

    result = compare_error_budget_rules_from_summaries(
        summary_dir=args.summary_dir,
        summary_paths=args.summary_path,
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

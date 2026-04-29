from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.config import (  # noqa: E402
    PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    PARTIAL_RANDOMIZED_KAPPA_MAX,
    PARTIAL_RANDOMIZED_KAPPA_MIN,
)
from trotterlib.df_cost_plotting import (  # noqa: E402
    DEFAULT_ANCHOR_REFINEMENT_DIR,
    DEFAULT_DF_REFERENCE_ARTIFACT_DIR,
    DEFAULT_DF_REFERENCE_RZ_LAYER_DIR,
    DEFAULT_DF_REFERENCE_RZ_LAYER_KEY,
    DEFAULT_MAX_LD_DIR,
    DEFAULT_OPTIMIZED_COST_PLOT_DIR,
    DEFAULT_PF_LABELS,
    build_cost_ratio_records,
    build_grouping_deterministic_cost_records,
    build_optimized_cost_records,
    build_reference_max_ld_sanity_records,
    collect_optimized_cost_comparisons,
    load_df_cost_plot_inputs,
    plot_cost_vs_ld_by_system_pf,
    plot_optimized_cost_by_pf,
    plot_ratio_summary,
    write_records_csv,
    write_records_json,
)


def _parse_csv(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    labels = tuple(item.strip() for item in raw.split(",") if item.strip())
    return labels or None


def _write_plots_for_rule(
    *,
    args: argparse.Namespace,
    inputs: dict[str, list[dict]],
    pf_labels: tuple[str, ...] | None,
    error_budget_rule: str,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = collect_optimized_cost_comparisons(
        inputs["summaries"],
        inputs["max_ld_rows"],
        molecule_min=args.molecule_min,
        molecule_max=args.molecule_max,
        pf_labels=pf_labels,
        epsilon_total=float(args.epsilon_total),
        kappa_mode=str(args.kappa_mode),
        kappa_value=float(args.kappa_value),
        kappa_min=float(args.kappa_min),
        kappa_max=float(args.kappa_max),
        randomized_method=str(args.randomized_method),
        g_rand=float(args.g_rand),
        error_budget_rule=error_budget_rule,
        df_cost_model=str(args.df_cost_model),
        reference_randomized_cost_mode=str(args.reference_randomized_cost_mode),
        df_reference_artifact_dir=args.df_reference_artifact_dir,
        df_reference_rz_layer_dir=args.df_reference_rz_layer_dir,
        df_reference_rz_layer_key=str(args.df_reference_rz_layer_key),
    )
    records = build_optimized_cost_records(comparisons)
    grouping_records: list[dict] = []
    if args.include_grouping_deterministic and error_budget_rule == "linear":
        grouping_records = build_grouping_deterministic_cost_records(
            molecule_min=args.molecule_min,
            molecule_max=args.molecule_max,
            pf_labels=pf_labels,
            epsilon_total=float(args.epsilon_total),
            grouping_cost_unit=str(args.grouping_cost_unit),
            grouping_qpe_beta=float(args.grouping_qpe_beta),
            use_original=bool(args.grouping_use_original),
        )
        records.extend(grouping_records)
    ratio_records = build_cost_ratio_records(records)
    sanity_records = build_reference_max_ld_sanity_records(records)

    write_records_json(output_dir / "optimized_cost_plot_records.json", records)
    write_records_csv(output_dir / "optimized_cost_plot_records.csv", records)
    write_records_json(output_dir / "optimized_cost_ratio_records.json", ratio_records)
    write_records_csv(output_dir / "optimized_cost_ratio_records.csv", ratio_records)
    if sanity_records:
        write_records_json(
            output_dir / "reference_max_ld_sanity_records.json",
            sanity_records,
        )
        write_records_csv(
            output_dir / "reference_max_ld_sanity_records.csv",
            sanity_records,
        )

    per_system_paths: list[Path] = []
    if not args.no_per_system_plots:
        per_system_paths = plot_cost_vs_ld_by_system_pf(
            records,
            output_dir / "per_system",
            image_format=str(args.image_format),
            dpi=int(args.dpi),
        )
    ratio_paths = plot_ratio_summary(
        ratio_records,
        output_dir,
        image_format=str(args.image_format),
        dpi=int(args.dpi),
        write_per_pf=True,
    )
    pf_comparison_path = plot_optimized_cost_by_pf(
        records,
        output_dir,
        cost_kind=str(args.pf_comparison_cost_kind),
        include_grouping_deterministic=bool(args.include_grouping_deterministic),
        image_format=str(args.image_format),
        dpi=int(args.dpi),
    )
    return {
        "error_budget_rule": error_budget_rule,
        "output_dir": str(output_dir),
        "comparisons": len(comparisons),
        "plot_records": len(records),
        "grouping_records": len(grouping_records),
        "ratio_records": len(ratio_records),
        "reference_max_ld_sanity_records": len(sanity_records),
        "per_system_plots": len(per_system_paths),
        "ratio_plots": [str(path) for path in ratio_paths],
        "pf_comparison_plot": (
            None if pf_comparison_path is None else str(pf_comparison_path)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot DF partially randomized Trotter LD-optimized costs."
    )
    parser.add_argument(
        "--anchor-refinement-dir",
        type=Path,
        default=DEFAULT_ANCHOR_REFINEMENT_DIR,
        help="Directory containing *.summary/refinement/anchor JSON files.",
    )
    parser.add_argument(
        "--max-ld-dir",
        type=Path,
        default=DEFAULT_MAX_LD_DIR,
        help="Directory containing max-LD Cgs JSON files.",
    )
    parser.add_argument("--molecule-min", type=int, default=3)
    parser.add_argument("--molecule-max", type=int, default=13)
    parser.add_argument(
        "--pf-labels",
        default=",".join(DEFAULT_PF_LABELS),
        help="Comma-separated PF labels.",
    )
    parser.add_argument("--epsilon-total", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OPTIMIZED_COST_PLOT_DIR)
    parser.add_argument("--image-format", default="png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--kappa-mode", choices=("fixed", "optimize"), default="optimize")
    parser.add_argument("--kappa-value", type=float, default=PARTIAL_RANDOMIZED_DEFAULT_KAPPA)
    parser.add_argument("--kappa-min", type=float, default=PARTIAL_RANDOMIZED_KAPPA_MIN)
    parser.add_argument("--kappa-max", type=float, default=PARTIAL_RANDOMIZED_KAPPA_MAX)
    parser.add_argument(
        "--randomized-method",
        choices=("qdrift", "rte"),
        default=PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    )
    parser.add_argument("--g-rand", type=float, default=PARTIAL_RANDOMIZED_DEFAULT_G_RAND)
    parser.add_argument(
        "--error-budget-rule",
        choices=("quadrature", "linear", "both"),
        default="quadrature",
        help="Error-budget rule used when recomputing plot costs.",
    )
    parser.add_argument(
        "--pf-comparison-cost-kind",
        choices=("actual_best", "screening"),
        default="actual_best",
        help="Cost kind used for the PF-vs-PF optimized cost summary.",
    )
    parser.add_argument(
        "--df-cost-model",
        choices=(
            "qiskit_decomposed_rz_depth",
            "df_reference_rz_layers",
            "reference_rz_layers",
        ),
        default="qiskit_decomposed_rz_depth",
        help=(
            "DF deterministic step-cost model. df_reference_rz_layers rescales "
            "stored LD costs so max-LD matches the DF reference artifact "
            "RZ-layer metric. reference_rz_layers is a deprecated alias."
        ),
    )
    parser.add_argument(
        "--df-reference-artifact-dir",
        type=Path,
        default=DEFAULT_DF_REFERENCE_ARTIFACT_DIR,
        help="Reference DF trotter_expo_coeff_df directory used by df_reference_rz_layers.",
    )
    parser.add_argument(
        "--df-reference-rz-layer-dir",
        type=Path,
        default=DEFAULT_DF_REFERENCE_RZ_LAYER_DIR,
        help="Fallback Reference DF df_rz_layer directory used by df_reference_rz_layers.",
    )
    parser.add_argument(
        "--df-reference-rz-layer-key",
        default=DEFAULT_DF_REFERENCE_RZ_LAYER_KEY,
        help="RZ-layer key to prefer inside DF reference artifacts.",
    )
    parser.add_argument(
        "--reference-randomized-cost-mode",
        choices=("input", "mean_fragment", "tail_mean_fragment", "tail_total"),
        default="tail_mean_fragment",
        help=(
            "Randomized sample cost used when --df-cost-model "
            "df_reference_rz_layers. input keeps --g-rand unchanged."
        ),
    )
    parser.add_argument(
        "--include-grouping-deterministic",
        action="store_true",
        help=(
            "Overlay deterministic-only grouping costs on the linear PF "
            "comparison plot."
        ),
    )
    parser.add_argument(
        "--grouping-cost-unit",
        choices=("pauli_rotations", "rz_layers"),
        default="rz_layers",
        help=(
            "Per-step cost table for grouping deterministic costs. "
            "rz_layers matches the partially randomized DF plots."
        ),
    )
    parser.add_argument(
        "--grouping-qpe-beta",
        type=float,
        default=1.0,
        help=(
            "QPE iteration prefactor for grouping deterministic costs. "
            "Use 1.2 to reproduce the older grouping convention."
        ),
    )
    parser.add_argument(
        "--grouping-use-original",
        action="store_true",
        help="Load grouping coefficients from trotter_expo_coeff_gr_original.",
    )
    parser.add_argument(
        "--no-per-system-plots",
        action="store_true",
        help="Only write aggregate tables and ratio plots.",
    )
    args = parser.parse_args()

    pf_labels = _parse_csv(args.pf_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_df_cost_plot_inputs(
        anchor_refinement_dir=args.anchor_refinement_dir,
        max_ld_dir=args.max_ld_dir,
    )
    rules = (
        ("quadrature", "linear")
        if args.error_budget_rule == "both"
        else (str(args.error_budget_rule),)
    )
    runs = []
    for rule in rules:
        output_dir = args.output_dir / rule if args.error_budget_rule == "both" else args.output_dir
        runs.append(
            _write_plots_for_rule(
                args=args,
                inputs=inputs,
                pf_labels=pf_labels,
                error_budget_rule=rule,
                output_dir=output_dir,
            )
        )

    summary = {
        "output_dir": str(args.output_dir),
        "df_cost_model": str(args.df_cost_model),
        "reference_randomized_cost_mode": str(args.reference_randomized_cost_mode),
        "df_reference_artifact_dir": str(args.df_reference_artifact_dir),
        "df_reference_rz_layer_dir": str(args.df_reference_rz_layer_dir),
        "df_reference_rz_layer_key": str(args.df_reference_rz_layer_key),
        "summary_rows": len(inputs["summaries"]),
        "anchor_rows": len(inputs["anchor_rows"]),
        "refinement_rows": len(inputs["refinement_rows"]),
        "max_ld_rows": len(inputs["max_ld_rows"]),
        "runs": runs,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

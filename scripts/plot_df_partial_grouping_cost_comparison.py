from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402

from trotterlib.config import PARTIAL_RANDOMIZED_ARTIFACTS_DIR, pf_order  # noqa: E402
from trotterlib.grouped_uwc_comparison import load_grouped_alpha_artifact  # noqa: E402


DEFAULT_LINEAR_DF_RECORDS = (
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR
    / "plots"
    / "optimized_costs_linear"
    / "optimized_cost_plot_records.json"
)
DEFAULT_GROUPING_STEP_CACHE = (
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR
    / "grouping_step_costs"
    / "grouping_step_costs.json"
)
DEFAULT_OUTPUT_DIR = (
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR
    / "plots"
    / "df_partial_grouping_cost_comparison"
)
DEFAULT_PF_LABELS = ("2nd", "4th", "4th(new_2)", "8th(Morales)")


METHOD_LABELS = {
    "df_partial_randomized": "DF partial randomized",
    "df_full_deterministic": "DF full deterministic",
    "grouping_deterministic": "grouping deterministic",
}
METHOD_STYLES = {
    "df_partial_randomized": {"marker": "o", "linestyle": "-", "linewidth": 2.3},
    "df_full_deterministic": {"marker": "s", "linestyle": "--", "linewidth": 2.0},
    "grouping_deterministic": {"marker": "^", "linestyle": ":", "linewidth": 2.3},
}


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_molecules(raw: str | None, *, default_min: int, default_max: int) -> tuple[int, ...]:
    if not raw:
        return tuple(range(int(default_min), int(default_max) + 1))
    values: list[int] = []
    for item in _parse_csv(raw):
        token = item.removeprefix("H")
        if "-" in token:
            lo_raw, hi_raw = token.split("-", 1)
            lo = int(lo_raw.removeprefix("H"))
            hi = int(hi_raw.removeprefix("H"))
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(token))
    return tuple(dict.fromkeys(values))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(str(key), None)
    return list(seen)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, default=_json_default)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, Mapping)]
    if isinstance(data, Mapping):
        rows = data.get("rows") or data.get("records")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    raise ValueError(f"Could not find rows in {path}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _best_by_molecule(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        g_total = _safe_float(row.get("g_total"))
        if g_total is None:
            continue
        molecule_type = int(row["molecule_type"])
        if molecule_type not in best or g_total < float(best[molecule_type]["g_total"]):
            best[molecule_type] = dict(row)
    return [best[key] for key in sorted(best)]


def _normalize_df_record(row: Mapping[str, Any], *, method: str) -> dict[str, Any]:
    step_cost = row.get("deterministic_step_cost_value", row.get("total_ref_rz_depth"))
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "molecule": str(row.get("molecule", f"H{int(row['molecule_type'])}")),
        "molecule_type": int(row["molecule_type"]),
        "pf_label": str(row["pf_label"]),
        "order": int(row.get("order", pf_order(str(row["pf_label"])))),
        "ld": int(row.get("ld", 0)),
        "df_rank_actual": row.get("df_rank_actual"),
        "step_cost": None if step_cost is None else float(step_cost),
        "step_cost_unit": "df_native_rz_depth_per_pf_step",
        "g_total": float(row["g_total"]),
        "g_det": row.get("g_det"),
        "g_rand": row.get("g_rand"),
        "g_rand_fraction": (
            None
            if _safe_float(row.get("g_rand")) is None
            else float(row.get("g_rand", 0.0)) / float(row["g_total"])
        ),
        "q_opt": row.get("q_opt"),
        "kappa_opt": row.get("kappa_opt"),
        "c_gs": row.get("c_gs_d"),
        "lambda_r": row.get("lambda_r"),
        "cost_kind": row.get("cost_kind"),
        "source_kind": row.get("source_kind"),
        "error_budget_rule": row.get("error_budget_rule"),
        "_artifact_source": row.get("_artifact_source"),
    }


def _df_method_rows(
    df_records: Sequence[Mapping[str, Any]],
    *,
    molecules: set[int],
    pf_labels: set[str],
    partial_cost_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    partial_candidates = [
        row
        for row in df_records
        if int(row.get("molecule_type", -1)) in molecules
        and str(row.get("pf_label")) in pf_labels
        and str(row.get("cost_kind")) == partial_cost_kind
    ]
    full_candidates = [
        row
        for row in df_records
        if int(row.get("molecule_type", -1)) in molecules
        and str(row.get("pf_label")) in pf_labels
        and str(row.get("cost_kind")) == "max_ld"
    ]
    partial = [
        _normalize_df_record(row, method="df_partial_randomized")
        for row in _best_by_molecule(partial_candidates)
    ]
    full = [
        _normalize_df_record(row, method="df_full_deterministic")
        for row in _best_by_molecule(full_candidates)
    ]
    return partial, full


def _linear_grouping_total_cost(
    *,
    step_cost: float,
    c_gs: float,
    order: int,
    epsilon_total: float,
    qpe_beta: float,
) -> dict[str, float]:
    q_opt = float(order) / float(order + 1)
    eps_qpe = float(epsilon_total) * q_opt
    eps_trot = float(epsilon_total) * (1.0 - q_opt)
    deterministic_scale = float(step_cost) * (float(c_gs) ** (1.0 / float(order)))
    g_total = float(qpe_beta) * deterministic_scale / (
        eps_qpe * (eps_trot ** (1.0 / float(order)))
    )
    return {
        "q_opt": q_opt,
        "eps_qpe_opt": eps_qpe,
        "eps_trot_opt": eps_trot,
        "g_total": float(g_total),
    }


def _grouping_rows_from_cache(
    grouping_cache_rows: Sequence[Mapping[str, Any]],
    *,
    molecules: set[int],
    pf_labels: set[str],
    epsilon_total: float,
    qpe_beta: float,
    use_original: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in grouping_cache_rows:
        molecule_type = int(row.get("molecule_type", -1))
        pf_label = str(row.get("pf_label"))
        if molecule_type not in molecules or pf_label not in pf_labels:
            continue
        step_cost = _safe_float(row.get("step_rz_layers"))
        if step_cost is None or step_cost <= 0.0:
            continue
        try:
            coeff, artifact_source = load_grouped_alpha_artifact(
                molecule_type,
                pf_label,
                use_original=use_original,
            )
        except Exception as exc:  # noqa: BLE001 - record missing coefficients for review.
            rows.append(
                {
                    "method": "grouping_deterministic",
                    "method_label": METHOD_LABELS["grouping_deterministic"],
                    "molecule": f"H{molecule_type}",
                    "molecule_type": molecule_type,
                    "pf_label": pf_label,
                    "order": int(row.get("order", pf_order(pf_label))),
                    "step_cost": step_cost,
                    "step_cost_unit": "grouping_rz_layers_per_pf_step",
                    "g_total": None,
                    "missing_reason": f"{type(exc).__name__}: {exc}",
                    "cost_kind": "grouping_deterministic",
                }
            )
            continue
        order = int(row.get("order", pf_order(pf_label)))
        budget = _linear_grouping_total_cost(
            step_cost=step_cost,
            c_gs=float(coeff),
            order=order,
            epsilon_total=epsilon_total,
            qpe_beta=qpe_beta,
        )
        rows.append(
            {
                "method": "grouping_deterministic",
                "method_label": METHOD_LABELS["grouping_deterministic"],
                "molecule": f"H{molecule_type}",
                "molecule_type": molecule_type,
                "pf_label": pf_label,
                "order": order,
                "ld": None,
                "step_cost": step_cost,
                "step_cost_unit": "grouping_rz_layers_per_pf_step",
                "step_pauli_rotations": row.get("step_pauli_rotations"),
                "g_total": budget["g_total"],
                "g_det": budget["g_total"],
                "g_rand": 0.0,
                "g_rand_fraction": 0.0,
                "q_opt": budget["q_opt"],
                "kappa_opt": None,
                "eps_qpe_opt": budget["eps_qpe_opt"],
                "eps_trot_opt": budget["eps_trot_opt"],
                "c_gs": float(coeff),
                "lambda_r": 0.0,
                "qpe_beta": float(qpe_beta),
                "cost_kind": "grouping_deterministic",
                "source_kind": "grouping_full_deterministic",
                "error_budget_rule": "linear",
                "grouping_step_cost_cache_source": row.get("cost_definition"),
                "_artifact_source": artifact_source,
            }
        )
    return rows


def _plot_best_costs(
    best_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    epsilon_total: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    for method in METHOD_LABELS:
        series = sorted(
            [row for row in best_rows if row.get("method") == method],
            key=lambda row: int(row["molecule_type"]),
        )
        if not series:
            continue
        style = METHOD_STYLES[method]
        xs = [int(row["molecule_type"]) for row in series]
        ys = [float(row["g_total"]) for row in series]
        labels = [str(row["pf_label"]) for row in series]
        ax.plot(xs, ys, label=METHOD_LABELS[method], **style)
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(
                label.replace("(Morales)", ""),
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                alpha=0.8,
            )
    ax.set_yscale("log")
    ax.set_xlabel("H-chain length")
    ax.set_ylabel("G_total (RZ-depth-equivalent, model-specific units)")
    ax.set_title(f"Best total cost comparison, linear error budget, epsilon={epsilon_total:g}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _best_by_method_molecule(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    out: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        g_total = _safe_float(row.get("g_total"))
        if g_total is None:
            continue
        key = (str(row["method"]), int(row["molecule_type"]))
        if key not in out or g_total < float(out[key]["g_total"]):
            out[key] = row
    return out


def _ratio_rows(best_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best = _best_by_method_molecule(best_rows)
    molecules = sorted({molecule for _method, molecule in best})
    rows: list[dict[str, Any]] = []
    for molecule_type in molecules:
        grouping = best.get(("grouping_deterministic", molecule_type))
        df_full = best.get(("df_full_deterministic", molecule_type))
        partial = best.get(("df_partial_randomized", molecule_type))
        grouping_total = _safe_float(None if grouping is None else grouping.get("g_total"))
        df_full_total = _safe_float(None if df_full is None else df_full.get("g_total"))
        partial_total = _safe_float(None if partial is None else partial.get("g_total"))
        if grouping_total and df_full_total:
            rows.append(
                {
                    "molecule": f"H{molecule_type}",
                    "molecule_type": molecule_type,
                    "ratio_kind": "df_full_over_grouping",
                    "numerator_method": "df_full_deterministic",
                    "denominator_method": "grouping_deterministic",
                    "g_total_ratio": df_full_total / grouping_total,
                    "g_total_num": df_full_total,
                    "g_total_den": grouping_total,
                    "pf_num": df_full.get("pf_label"),
                    "pf_den": grouping.get("pf_label"),
                }
            )
        if grouping_total and partial_total:
            rows.append(
                {
                    "molecule": f"H{molecule_type}",
                    "molecule_type": molecule_type,
                    "ratio_kind": "df_partial_over_grouping",
                    "numerator_method": "df_partial_randomized",
                    "denominator_method": "grouping_deterministic",
                    "g_total_ratio": partial_total / grouping_total,
                    "g_total_num": partial_total,
                    "g_total_den": grouping_total,
                    "pf_num": partial.get("pf_label"),
                    "pf_den": grouping.get("pf_label"),
                }
            )
        if df_full_total and partial_total:
            rows.append(
                {
                    "molecule": f"H{molecule_type}",
                    "molecule_type": molecule_type,
                    "ratio_kind": "df_partial_over_df_full",
                    "numerator_method": "df_partial_randomized",
                    "denominator_method": "df_full_deterministic",
                    "g_total_ratio": partial_total / df_full_total,
                    "g_total_num": partial_total,
                    "g_total_den": df_full_total,
                    "pf_num": partial.get("pf_label"),
                    "pf_den": df_full.get("pf_label"),
                }
            )
    return rows


def _plot_ratios(
    ratio_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    labels = {
        "df_full_over_grouping": "DF full / grouping",
        "df_partial_over_grouping": "DF partial / grouping",
        "df_partial_over_df_full": "DF partial / DF full",
    }
    styles = {
        "df_full_over_grouping": {"marker": "s", "linestyle": "--"},
        "df_partial_over_grouping": {"marker": "o", "linestyle": "-"},
        "df_partial_over_df_full": {"marker": "D", "linestyle": ":"},
    }
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for ratio_kind, label in labels.items():
        series = sorted(
            [row for row in ratio_rows if row["ratio_kind"] == ratio_kind],
            key=lambda row: int(row["molecule_type"]),
        )
        if not series:
            continue
        ax.plot(
            [int(row["molecule_type"]) for row in series],
            [float(row["g_total_ratio"]) for row in series],
            label=label,
            linewidth=2.0,
            **styles[ratio_kind],
        )
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("H-chain length")
    ax.set_ylabel("G_total ratio")
    ax.set_title("Best-cost ratios across DF partial, DF full, and grouping")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_step_costs(
    best_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for method in METHOD_LABELS:
        series = sorted(
            [
                row
                for row in best_rows
                if row.get("method") == method and _safe_float(row.get("step_cost")) is not None
            ],
            key=lambda row: int(row["molecule_type"]),
        )
        if not series:
            continue
        style = METHOD_STYLES[method]
        ax.plot(
            [int(row["molecule_type"]) for row in series],
            [float(row["step_cost"]) for row in series],
            label=METHOD_LABELS[method],
            **style,
        )
    ax.set_yscale("log")
    ax.set_xlabel("H-chain length")
    ax.set_ylabel("per-step cost used by each model")
    ax.set_title("Best-record per-step costs (DF-native vs grouping units)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot best-cost comparison between DF full deterministic, DF partial "
            "randomized, and grouping deterministic records."
        )
    )
    parser.add_argument("--df-records", type=Path, default=DEFAULT_LINEAR_DF_RECORDS)
    parser.add_argument("--grouping-step-cache", type=Path, default=DEFAULT_GROUPING_STEP_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--molecules", default="H3-H13")
    parser.add_argument("--pf-labels", default=",".join(DEFAULT_PF_LABELS))
    parser.add_argument(
        "--partial-cost-kind",
        choices=("actual_best", "actual_or_max_ld_cgs_optimized", "screening"),
        default="actual_best",
    )
    parser.add_argument("--epsilon-total", type=float, default=1e-4)
    parser.add_argument("--grouping-qpe-beta", type=float, default=1.0)
    parser.add_argument("--grouping-use-original", action="store_true")
    parser.add_argument("--image-format", default="png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    molecules = set(_parse_molecules(args.molecules, default_min=3, default_max=13))
    pf_labels = set(_parse_csv(args.pf_labels) or DEFAULT_PF_LABELS)

    df_records = _load_json_rows(args.df_records)
    grouping_cache_rows = _load_json_rows(args.grouping_step_cache)
    partial_rows, full_rows = _df_method_rows(
        df_records,
        molecules=molecules,
        pf_labels=pf_labels,
        partial_cost_kind=str(args.partial_cost_kind),
    )
    grouping_all_rows = _grouping_rows_from_cache(
        grouping_cache_rows,
        molecules=molecules,
        pf_labels=pf_labels,
        epsilon_total=float(args.epsilon_total),
        qpe_beta=float(args.grouping_qpe_beta),
        use_original=bool(args.grouping_use_original),
    )
    grouping_best_rows = _best_by_molecule(
        row for row in grouping_all_rows if _safe_float(row.get("g_total")) is not None
    )

    all_method_rows = partial_rows + full_rows + grouping_all_rows
    best_rows = partial_rows + full_rows + grouping_best_rows
    ratio_rows = _ratio_rows(best_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "cost_comparison_all_records.json", all_method_rows)
    _write_csv(output_dir / "cost_comparison_all_records.csv", all_method_rows)
    _write_json(output_dir / "cost_comparison_best_records.json", best_rows)
    _write_csv(output_dir / "cost_comparison_best_records.csv", best_rows)
    _write_json(output_dir / "cost_comparison_ratio_records.json", ratio_rows)
    _write_csv(output_dir / "cost_comparison_ratio_records.csv", ratio_rows)

    image_format = str(args.image_format).lstrip(".")
    cost_plot = output_dir / f"best_total_cost_comparison.{image_format}"
    ratio_plot = output_dir / f"best_cost_ratios.{image_format}"
    step_plot = output_dir / f"best_step_cost_comparison.{image_format}"
    _plot_best_costs(
        best_rows,
        cost_plot,
        epsilon_total=float(args.epsilon_total),
        dpi=int(args.dpi),
    )
    _plot_ratios(ratio_rows, ratio_plot, dpi=int(args.dpi))
    _plot_step_costs(best_rows, step_plot, dpi=int(args.dpi))

    summary = {
        "schema_version": 1,
        "df_records": str(args.df_records),
        "grouping_step_cache": str(args.grouping_step_cache),
        "output_dir": str(output_dir),
        "molecules": [f"H{molecule}" for molecule in sorted(molecules)],
        "pf_labels": sorted(pf_labels, key=lambda label: (pf_order(label), label)),
        "epsilon_total": float(args.epsilon_total),
        "error_budget_rule": "linear",
        "grouping_qpe_beta": float(args.grouping_qpe_beta),
        "partial_cost_kind": str(args.partial_cost_kind),
        "num_partial_best_rows": len(partial_rows),
        "num_df_full_best_rows": len(full_rows),
        "num_grouping_candidate_rows": len(grouping_all_rows),
        "num_grouping_best_rows": len(grouping_best_rows),
        "num_ratio_rows": len(ratio_rows),
        "plots": {
            "best_total_cost_comparison": str(cost_plot),
            "best_cost_ratios": str(ratio_plot),
            "best_step_cost_comparison": str(step_plot),
        },
    }
    _write_json(output_dir / "cost_comparison_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

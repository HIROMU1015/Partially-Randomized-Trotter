from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import (
    COLOR_MAP,
    DECOMPO_NUM,
    MARKER_MAP,
    P_DIR,
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR,
    PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    PARTIAL_RANDOMIZED_KAPPA_MAX,
    PARTIAL_RANDOMIZED_KAPPA_MIN,
    PF_RZ_LAYER,
    pf_order,
    pickle_dir,
)
from .io_cache import load_data
from .partial_randomized_pf import optimize_error_budget_and_kappa


DEFAULT_ANCHOR_REFINEMENT_DIR = (
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR / "df_cgs" / "anchor_refinement"
)
DEFAULT_MAX_LD_DIR = PARTIAL_RANDOMIZED_ARTIFACTS_DIR / "df_cgs" / "max_ld"
DEFAULT_OPTIMIZED_COST_PLOT_DIR = (
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR / "plots" / "optimized_costs"
)
DEFAULT_DF_REFERENCE_PROJECT_DIR = (
    Path.home() / "myproject" / "DF_project" / "Evaluation_numGate_highorder_DF"
)
DEFAULT_DF_REFERENCE_ARTIFACT_DIR = (
    DEFAULT_DF_REFERENCE_PROJECT_DIR / "artifacts" / "trotter_expo_coeff_df"
)
DEFAULT_DF_REFERENCE_RZ_LAYER_DIR = (
    DEFAULT_DF_REFERENCE_PROJECT_DIR / "artifacts" / "df_rz_layer"
)
DEFAULT_DF_REFERENCE_RZ_LAYER_KEY = "total_nonclifford_z_coloring_depth"
DEFAULT_PF_LABELS: tuple[str, ...] = (
    "2nd",
    "4th",
    "4th(new_2)",
    "8th(Morales)",
)
DF_COST_MODELS: tuple[str, ...] = (
    "qiskit_decomposed_rz_depth",
    "df_reference_rz_layers",
    "reference_rz_layers",
)
REFERENCE_RANDOMIZED_COST_MODES: tuple[str, ...] = (
    "input",
    "mean_fragment",
    "tail_mean_fragment",
    "tail_total",
)

COST_KIND_ORDER: dict[str, int] = {
    "screening": 0,
    "actual_window": 1,
    "actual_best": 2,
    "max_ld_cgs_optimized": 3,
    "max_ld": 4,
    "grouping_deterministic": 5,
}

_GROUPING_ORIGINAL_PF_LABEL_MAP: dict[str, str] = {
    "2nd": "w2",
    "4th": "w3",
    "4th(new_2)": "wmy4",
    "8th(Morales)": "w8",
    "10th(Morales)": "w1016",
    "8th(Yoshida)": "wyoshida",
}


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON list, or a dict containing an ``entries`` list."""
    json_path = Path(path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return [dict(item) for item in data["entries"] if isinstance(item, dict)]
    return []


def preferred_final_or_partial_paths(paths: Iterable[str | Path]) -> list[Path]:
    """
    Return one path per logical artifact, preferring ``*.json`` over
    ``*.partial.json`` when both exist.
    """
    best_by_name: dict[str, Path] = {}
    for raw_path in sorted(Path(path) for path in paths):
        path = Path(raw_path)
        if not path.exists():
            continue
        logical_name = _logical_artifact_name(path)
        current = best_by_name.get(logical_name)
        if current is None:
            best_by_name[logical_name] = path
        elif _source_priority(path) >= _source_priority(current):
            best_by_name[logical_name] = path
    return [best_by_name[name] for name in sorted(best_by_name)]


def load_anchor_refinement_summaries(
    directory: str | Path = DEFAULT_ANCHOR_REFINEMENT_DIR,
    *,
    paths: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Load and deduplicate anchor-refinement summary rows."""
    selected = (
        preferred_final_or_partial_paths(paths)
        if paths is not None
        else _preferred_paths_by_suffix(Path(directory), ".summary")
    )
    rows = _load_rows_with_source(selected)
    return _deduplicate_rows(rows, _summary_key)


def load_anchor_cgs_rows(
    directory: str | Path = DEFAULT_ANCHOR_REFINEMENT_DIR,
    *,
    paths: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Load and deduplicate anchor Cgs rows."""
    selected = (
        preferred_final_or_partial_paths(paths)
        if paths is not None
        else _preferred_paths_by_suffix(Path(directory), ".anchor")
    )
    return _deduplicate_rows(_load_rows_with_source(selected), _cgs_key)


def load_refinement_cgs_rows(
    directory: str | Path = DEFAULT_ANCHOR_REFINEMENT_DIR,
    *,
    paths: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Load and deduplicate explicit-LD refinement Cgs rows."""
    selected = (
        preferred_final_or_partial_paths(paths)
        if paths is not None
        else _preferred_paths_by_suffix(Path(directory), ".refinement")
    )
    return _deduplicate_rows(_load_rows_with_source(selected), _cgs_key)


def load_max_ld_cgs_rows(
    directory: str | Path = DEFAULT_MAX_LD_DIR,
    *,
    paths: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Load and deduplicate max-LD Cgs rows."""
    if paths is None:
        selected = preferred_final_or_partial_paths(
            path
            for path in Path(directory).glob("*.json")
            if ".cache." not in path.name and ".targets." not in path.name
        )
    else:
        selected = preferred_final_or_partial_paths(paths)
    return _deduplicate_rows(_load_rows_with_source(selected), _cgs_key)


def load_df_cost_plot_inputs(
    *,
    anchor_refinement_dir: str | Path = DEFAULT_ANCHOR_REFINEMENT_DIR,
    max_ld_dir: str | Path = DEFAULT_MAX_LD_DIR,
) -> dict[str, list[dict[str, Any]]]:
    """Load summary, anchor, refinement, and max-LD JSON inputs for plotting."""
    return {
        "summaries": load_anchor_refinement_summaries(anchor_refinement_dir),
        "anchor_rows": load_anchor_cgs_rows(anchor_refinement_dir),
        "refinement_rows": load_refinement_cgs_rows(anchor_refinement_dir),
        "max_ld_rows": load_max_ld_cgs_rows(max_ld_dir),
    }


def cost_for_cgs_row(
    row: Mapping[str, Any],
    *,
    epsilon_total: float = 1e-4,
    kappa_mode: str = "optimize",
    kappa_value: float = PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    kappa_min: float = PARTIAL_RANDOMIZED_KAPPA_MIN,
    kappa_max: float = PARTIAL_RANDOMIZED_KAPPA_MAX,
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    error_budget_rule: str = "quadrature",
    df_cost_model: str = "qiskit_decomposed_rz_depth",
    reference_context: Mapping[tuple[int, str], Mapping[str, Any]] | None = None,
    reference_randomized_cost_mode: str = "input",
) -> dict[str, Any]:
    """Recompute cost fields for one explicit Cgs row."""
    order = int(row.get("order", pf_order(str(row["pf_label"]))))
    cost_inputs = _df_cost_inputs_for_row(
        row,
        df_cost_model=df_cost_model,
        reference_context=reference_context,
        reference_randomized_cost_mode=reference_randomized_cost_mode,
        g_rand_input=g_rand,
    )
    budget = optimize_error_budget_and_kappa(
        epsilon_total=float(epsilon_total),
        order=order,
        deterministic_step_cost_value=int(cost_inputs["deterministic_step_cost_value"]),
        c_gs=float(row["c_gs_d"]),
        lambda_r=float(row["lambda_r"]),
        randomized_method=str(randomized_method),
        g_rand=float(cost_inputs["g_rand_input"]),
        kappa_mode=str(kappa_mode),
        kappa_value=float(kappa_value),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),
        error_budget_rule=error_budget_rule,
    )
    return {
        "molecule": _molecule_label(row),
        "molecule_type": int(row["molecule_type"]),
        "pf_label": str(row["pf_label"]),
        "order": order,
        "ld": int(row["ld"]),
        "lambda_r": float(row["lambda_r"]),
        "c_gs_d": float(row["c_gs_d"]),
        **cost_inputs,
        "q_opt": budget.q_ratio,
        "eps_qpe_opt": budget.eps_qpe,
        "eps_trot_opt": budget.eps_trot,
        "kappa_opt": budget.kappa,
        "b_opt": budget.b_value,
        "error_budget_rule": str(error_budget_rule),
        "g_det": budget.g_det,
        "g_rand": budget.g_rand,
        "g_total": budget.g_total,
        "boundary_hit_q": budget.boundary_hit_q,
        "boundary_hit_kappa": budget.boundary_hit_kappa,
        "source_kind": str(row.get("source_kind", "unknown")),
        "df_rank_actual": _optional_int(row.get("df_rank_actual")),
        "fit_slope": _optional_float(row.get("fit_slope")),
    }


def collect_optimized_cost_comparisons(
    summaries: Sequence[Mapping[str, Any]],
    max_ld_rows: Sequence[Mapping[str, Any]],
    *,
    molecule_min: int | None = None,
    molecule_max: int | None = None,
    pf_labels: Sequence[str] | None = None,
    epsilon_total: float = 1e-4,
    kappa_mode: str = "optimize",
    kappa_value: float = PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    kappa_min: float = PARTIAL_RANDOMIZED_KAPPA_MIN,
    kappa_max: float = PARTIAL_RANDOMIZED_KAPPA_MAX,
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    error_budget_rule: str = "quadrature",
    df_cost_model: str = "qiskit_decomposed_rz_depth",
    reference_randomized_cost_mode: str = "input",
    df_reference_artifact_dir: str | Path | None = DEFAULT_DF_REFERENCE_ARTIFACT_DIR,
    df_reference_rz_layer_dir: str | Path | None = DEFAULT_DF_REFERENCE_RZ_LAYER_DIR,
    df_reference_rz_layer_key: str = DEFAULT_DF_REFERENCE_RZ_LAYER_KEY,
) -> list[dict[str, Any]]:
    """Align screening best, actual LD-window costs, actual best, and max-LD."""
    df_cost_model = _normalize_df_cost_model(df_cost_model)
    reference_randomized_cost_mode = _normalize_reference_randomized_cost_mode(
        reference_randomized_cost_mode
    )
    reference_context = None
    if df_cost_model == "df_reference_rz_layers":
        reference_context = build_df_reference_cost_context(
            summaries,
            max_ld_rows,
            df_reference_artifact_dir=df_reference_artifact_dir,
            df_reference_rz_layer_dir=df_reference_rz_layer_dir,
            df_reference_rz_layer_key=df_reference_rz_layer_key,
        )
    pf_filter = None if pf_labels is None else {str(label) for label in pf_labels}
    max_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in max_ld_rows:
        if not _row_passes_filter(
            row,
            molecule_min=molecule_min,
            molecule_max=molecule_max,
            pf_filter=pf_filter,
        ):
            continue
        try:
            cost = cost_for_cgs_row(
                row,
                epsilon_total=epsilon_total,
                kappa_mode=kappa_mode,
                kappa_value=kappa_value,
                kappa_min=kappa_min,
                kappa_max=kappa_max,
                randomized_method=randomized_method,
                g_rand=g_rand,
                error_budget_rule=error_budget_rule,
                df_cost_model=df_cost_model,
                reference_context=reference_context,
                reference_randomized_cost_mode=reference_randomized_cost_mode,
            )
        except (KeyError, TypeError, ValueError):
            continue
        cost["_artifact_source"] = row.get("_artifact_source")
        max_by_key[(int(cost["molecule_type"]), str(cost["pf_label"]))] = cost

    comparisons: list[dict[str, Any]] = []
    for summary in summaries:
        if not _row_passes_filter(
            summary,
            molecule_min=molecule_min,
            molecule_max=molecule_max,
            pf_filter=pf_filter,
        ):
            continue
        molecule_type = int(summary["molecule_type"])
        pf_label = str(summary["pf_label"])
        if (
            df_cost_model == "df_reference_rz_layers"
            and (molecule_type, pf_label) not in (reference_context or {})
        ):
            continue
        screening = _best_recomputed_cost(
            summary.get("screening_candidates", []),
            cost_kind="screening",
            fallback=summary,
            epsilon_total=epsilon_total,
            kappa_mode=kappa_mode,
            kappa_value=kappa_value,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            randomized_method=randomized_method,
            g_rand=g_rand,
            error_budget_rule=error_budget_rule,
            df_cost_model=df_cost_model,
            reference_context=reference_context,
            reference_randomized_cost_mode=reference_randomized_cost_mode,
        )
        if screening is None and df_cost_model != "df_reference_rz_layers":
            screening = _normalize_cost_like(
                summary.get("screening_best"),
                cost_kind="screening",
                fallback=summary,
            )
        actual_costs = [
            _recompute_cost_like(
                cost,
                cost_kind="actual_window",
                fallback=summary,
                epsilon_total=epsilon_total,
                kappa_mode=kappa_mode,
                kappa_value=kappa_value,
                kappa_min=kappa_min,
                kappa_max=kappa_max,
                randomized_method=randomized_method,
                g_rand=g_rand,
                error_budget_rule=error_budget_rule,
                df_cost_model=df_cost_model,
                reference_context=reference_context,
                reference_randomized_cost_mode=reference_randomized_cost_mode,
            )
            for cost in summary.get("actual_costs", [])
            if isinstance(cost, Mapping)
        ]
        actual_costs = [cost for cost in actual_costs if cost is not None]
        actual_best = None
        if actual_costs:
            actual_best = min(actual_costs, key=lambda item: float(item["g_total"]))
            actual_best = {**actual_best, "cost_kind": "actual_best"}
        elif df_cost_model != "df_reference_rz_layers":
            actual_best = _normalize_cost_like(
                summary.get("actual_best"),
                cost_kind="actual_best",
                fallback=summary,
            )
        max_ld = _normalize_cost_like(
            max_by_key.get((molecule_type, pf_label)),
            cost_kind="max_ld",
            fallback=summary,
        )
        max_ld_cgs_optimized = _best_cost_with_replaced_cgs(
            summary.get("screening_candidates", []),
            c_gs_source=max_ld,
            cost_kind="max_ld_cgs_optimized",
            fallback=summary,
            epsilon_total=epsilon_total,
            kappa_mode=kappa_mode,
            kappa_value=kappa_value,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            randomized_method=randomized_method,
            g_rand=g_rand,
            error_budget_rule=error_budget_rule,
            df_cost_model=df_cost_model,
            reference_context=reference_context,
            reference_randomized_cost_mode=reference_randomized_cost_mode,
        )
        comparisons.append(
            {
                "molecule": _molecule_label(summary),
                "molecule_type": molecule_type,
                "pf_label": pf_label,
                "order": int(summary.get("order", pf_order(pf_label))),
                "df_rank_actual": _optional_int(summary.get("df_rank_actual")),
                "ld_anchor": _optional_int(summary.get("ld_anchor")),
                "screening_best": screening,
                "actual_costs": sorted(actual_costs, key=lambda item: int(item["ld"])),
                "actual_best": actual_best,
                "max_ld_cgs_optimized": max_ld_cgs_optimized,
                "max_ld": max_ld,
                "missing_actual_lds": [
                    int(ld) for ld in summary.get("missing_actual_lds", [])
                ],
                "_artifact_source": summary.get("_artifact_source"),
            }
        )
    comparisons.sort(key=lambda item: _system_pf_sort_key(item))
    return comparisons


def build_optimized_cost_records(
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten comparison groups into plot/export records."""
    records: list[dict[str, Any]] = []
    for comparison in comparisons:
        group_fields = {
            "molecule": comparison["molecule"],
            "molecule_type": int(comparison["molecule_type"]),
            "pf_label": str(comparison["pf_label"]),
            "order": int(comparison["order"]),
            "df_rank_actual": _optional_int(comparison.get("df_rank_actual")),
            "ld_anchor": _optional_int(comparison.get("ld_anchor")),
        }
        screening = comparison.get("screening_best")
        if isinstance(screening, Mapping):
            records.append(_merge_group_fields(screening, group_fields))
        for actual in comparison.get("actual_costs", []):
            if isinstance(actual, Mapping):
                actual_record = _merge_group_fields(actual, group_fields)
                best = comparison.get("actual_best")
                actual_record["is_actual_best"] = (
                    isinstance(best, Mapping)
                    and int(actual_record["ld"]) == int(best["ld"])
                )
                records.append(actual_record)
        actual_best = comparison.get("actual_best")
        if isinstance(actual_best, Mapping):
            records.append(_merge_group_fields(actual_best, group_fields))
        max_ld_cgs_optimized = comparison.get("max_ld_cgs_optimized")
        if isinstance(max_ld_cgs_optimized, Mapping):
            records.append(_merge_group_fields(max_ld_cgs_optimized, group_fields))
        max_ld = comparison.get("max_ld")
        if isinstance(max_ld, Mapping):
            records.append(_merge_group_fields(max_ld, group_fields))
    records.sort(key=_record_sort_key)
    return records


def build_grouping_deterministic_cost_records(
    *,
    molecule_min: int = 3,
    molecule_max: int = 13,
    pf_labels: Sequence[str] | None = None,
    epsilon_total: float = 1e-4,
    grouping_cost_unit: str = "rz_layers",
    grouping_qpe_beta: float = 1.0,
    use_original: bool = False,
) -> list[dict[str, Any]]:
    """
    Build deterministic-only grouping cost records with the linear error split.

    ``grouping_cost_unit="rz_layers"`` uses the grouping-Hamiltonian
    ``PF_RZ_LAYER`` table. ``"pauli_rotations"`` matches the existing
    grouping extrapolation plots.
    """
    if epsilon_total <= 0.0:
        raise ValueError("epsilon_total must be positive.")
    if grouping_qpe_beta <= 0.0:
        raise ValueError("grouping_qpe_beta must be positive.")
    if grouping_cost_unit == "pauli_rotations":
        step_cost_table = DECOMPO_NUM
    elif grouping_cost_unit == "rz_layers":
        step_cost_table = PF_RZ_LAYER
    else:
        raise ValueError(
            "grouping_cost_unit must be 'pauli_rotations' or 'rz_layers'."
        )

    labels = tuple(pf_labels) if pf_labels is not None else DEFAULT_PF_LABELS
    records: list[dict[str, Any]] = []
    for molecule_type in range(int(molecule_min), int(molecule_max) + 1):
        molecule = f"H{molecule_type}"
        molecule_step_costs = step_cost_table.get(molecule, {})
        for pf_label in labels:
            if pf_label not in P_DIR or pf_label not in molecule_step_costs:
                continue
            try:
                coeff, artifact_source = _load_grouping_coeff(
                    molecule_type,
                    pf_label,
                    use_original=use_original,
                )
            except (OSError, KeyError, TypeError, ValueError):
                continue
            if coeff <= 0.0:
                continue

            order = int(P_DIR[pf_label])
            step_cost = float(molecule_step_costs[pf_label])
            q_opt = order / (order + 1.0)
            eps_qpe = float(epsilon_total) * q_opt
            eps_trot = float(epsilon_total) * (1.0 - q_opt)
            deterministic_scale = step_cost * (coeff ** (1.0 / order))
            g_total = float(grouping_qpe_beta) * deterministic_scale / (
                eps_qpe * (eps_trot ** (1.0 / order))
            )
            record = {
                "molecule": molecule,
                "molecule_type": molecule_type,
                "pf_label": pf_label,
                "order": order,
                "ld": 0,
                "cost_kind": "grouping_deterministic",
                "g_total": float(g_total),
                "g_det": float(g_total),
                "g_rand": 0.0,
                "q_opt": float(q_opt),
                "kappa_opt": None,
                "eps_qpe_opt": eps_qpe,
                "eps_trot_opt": eps_trot,
                "b_opt": 0.0,
                "error_budget_rule": "linear",
                "c_gs_d": float(coeff),
                "lambda_r": 0.0,
                "df_cost_model": "not_applicable",
                "qpe_beta": float(grouping_qpe_beta),
                "deterministic_step_cost_value": step_cost,
                "grouping_cost_model": grouping_cost_unit,
                "grouping_cost_unit": grouping_cost_unit,
                "total_ref_rz_depth": (
                    int(step_cost) if grouping_cost_unit == "rz_layers" else None
                ),
                "source_kind": "grouping_full_deterministic",
                "_artifact_source": artifact_source,
            }
            if grouping_cost_unit == "pauli_rotations":
                record["pauli_rotations_per_step"] = int(step_cost)
            records.append(record)
    records.sort(key=_record_sort_key)
    return records


def build_df_reference_cost_context(
    summaries: Sequence[Mapping[str, Any]],
    max_ld_rows: Sequence[Mapping[str, Any]],
    *,
    df_reference_artifact_dir: str | Path | None = DEFAULT_DF_REFERENCE_ARTIFACT_DIR,
    df_reference_rz_layer_dir: str | Path | None = DEFAULT_DF_REFERENCE_RZ_LAYER_DIR,
    df_reference_rz_layer_key: str = DEFAULT_DF_REFERENCE_RZ_LAYER_KEY,
) -> dict[tuple[int, str], dict[str, Any]]:
    """
    Build the calibration table used by ``df_reference_rz_layers``.

    The current DF artifacts store Qiskit-decomposed RZ-depths for every LD.
    For comparison plots, we keep that LD-dependence but rescale each
    ``(molecule, PF)`` so that max-LD equals the DF reference project's
    full-deterministic RZ-layer metric. This deliberately does not use the
    grouping-Hamiltonian ``PF_RZ_LAYER`` table.
    """
    context: dict[tuple[int, str], dict[str, Any]] = {}
    reference_cache: dict[tuple[int, str], dict[str, Any]] = {}

    def reference_for(molecule_type: int, pf_label: str) -> dict[str, Any]:
        key = (int(molecule_type), str(pf_label))
        cached = reference_cache.get(key)
        if cached is not None:
            return cached
        reference = load_df_reference_step_layers(
            molecule_type,
            pf_label,
            df_reference_artifact_dir=df_reference_artifact_dir,
            df_reference_rz_layer_dir=df_reference_rz_layer_dir,
            df_reference_rz_layer_key=df_reference_rz_layer_key,
        )
        reference_cache[key] = reference
        return reference

    for row in max_ld_rows:
        try:
            molecule_type = int(row["molecule_type"])
            pf_label = str(row["pf_label"])
            key = (molecule_type, pf_label)
            raw_step = _total_ref_rz_depth(row)
            reference = reference_for(molecule_type, pf_label)
            reference_full = int(reference["reference_full_rz_depth"])
        except (OSError, KeyError, TypeError, ValueError):
            continue
        item = context.setdefault(key, {})
        item["reference_full_rz_depth"] = int(reference_full)
        item["df_reference_rz_layer_key"] = str(reference["df_reference_rz_layer_key"])
        item["df_reference_artifact_source"] = str(
            reference["df_reference_artifact_source"]
        )
        item["df_reference_artifact_kind"] = str(
            reference["df_reference_artifact_kind"]
        )
        item["max_raw_total_ref_rz_depth"] = int(raw_step)
        item["df_rank_actual"] = _optional_int(row.get("df_rank_actual", row.get("ld")))

    for summary in summaries:
        try:
            molecule_type = int(summary["molecule_type"])
            pf_label = str(summary["pf_label"])
            key = (molecule_type, pf_label)
            reference = reference_for(molecule_type, pf_label)
            reference_full = int(reference["reference_full_rz_depth"])
        except (OSError, KeyError, TypeError, ValueError):
            continue
        item = context.setdefault(key, {})
        item["reference_full_rz_depth"] = int(reference_full)
        item["df_reference_rz_layer_key"] = str(reference["df_reference_rz_layer_key"])
        item["df_reference_artifact_source"] = str(
            reference["df_reference_artifact_source"]
        )
        item["df_reference_artifact_kind"] = str(
            reference["df_reference_artifact_kind"]
        )
        df_rank = _optional_int(summary.get("df_rank_actual"))
        if df_rank is not None:
            item.setdefault("df_rank_actual", df_rank)
        for candidate in summary.get("screening_candidates", []):
            if not isinstance(candidate, Mapping):
                continue
            try:
                ld = int(candidate["ld"])
                raw_step = _total_ref_rz_depth(candidate)
            except (KeyError, TypeError, ValueError):
                continue
            if ld == 0:
                item["ld0_raw_total_ref_rz_depth"] = int(raw_step)
            candidate_df_rank = _optional_int(candidate.get("df_rank_actual", df_rank))
            if candidate_df_rank is not None:
                item.setdefault("df_rank_actual", candidate_df_rank)
                if ld == candidate_df_rank:
                    item.setdefault("max_raw_total_ref_rz_depth", int(raw_step))

    for item in context.values():
        max_raw = _optional_float(item.get("max_raw_total_ref_rz_depth"))
        reference_full = _optional_float(item.get("reference_full_rz_depth"))
        if max_raw is None or max_raw <= 0.0 or reference_full is None:
            continue
        scale = reference_full / max_raw
        item["reference_scale"] = scale
        ld0_raw = _optional_float(item.get("ld0_raw_total_ref_rz_depth"))
        if ld0_raw is not None:
            item["ld0_reference_rz_depth"] = _scaled_reference_step(
                ld0_raw,
                scale,
            )
    return context


def load_df_reference_step_layers(
    molecule_type: int,
    pf_label: str,
    *,
    df_reference_artifact_dir: str | Path | None = DEFAULT_DF_REFERENCE_ARTIFACT_DIR,
    df_reference_rz_layer_dir: str | Path | None = DEFAULT_DF_REFERENCE_RZ_LAYER_DIR,
    df_reference_rz_layer_key: str = DEFAULT_DF_REFERENCE_RZ_LAYER_KEY,
) -> dict[str, Any]:
    """
    Load a DF full-deterministic per-step RZ-layer metric from reference artifacts.

    The primary source is the older DF project's ``trotter_expo_coeff_df`` payload
    under ``rz_layers``. ``df_rz_layer`` is accepted as a fallback because some
    coefficient artifacts do not carry layer metadata.
    """
    ham_name = _df_reference_hamiltonian_name(int(molecule_type))
    search_roots = [
        ("trotter_expo_coeff_df", df_reference_artifact_dir),
        ("df_rz_layer", df_reference_rz_layer_dir),
    ]
    last_error: Exception | None = None
    for source_kind, raw_root in search_roots:
        if raw_root is None:
            continue
        root = Path(raw_root)
        if not root.exists():
            continue
        for target in _df_reference_target_names(ham_name, str(pf_label)):
            path = root / target
            if not path.exists():
                continue
            try:
                payload = load_data(str(path), gr=None)
                layer_key, layer_value = _pick_df_reference_rz_layer_value(
                    payload,
                    preferred_key=df_reference_rz_layer_key,
                )
            except Exception as exc:
                last_error = exc
                continue
            return {
                "molecule": f"H{int(molecule_type)}",
                "molecule_type": int(molecule_type),
                "pf_label": str(pf_label),
                "reference_full_rz_depth": int(round(float(layer_value))),
                "df_reference_rz_layer_key": str(layer_key),
                "df_reference_artifact_source": str(path),
                "df_reference_artifact_kind": source_kind,
            }
    if last_error is not None:
        raise ValueError(
            f"DF reference RZ-layer metric not found: H{int(molecule_type)} "
            f"{pf_label}"
        ) from last_error
    raise FileNotFoundError(
        f"DF reference artifact not found: H{int(molecule_type)} {pf_label}"
    )


def build_cost_ratio_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build long-form normalized cost-ratio records using max-LD as denominator."""
    by_group_kind: dict[tuple[int, str, str], dict[str, Any]] = {}
    for raw in records:
        if raw.get("g_total") is None:
            continue
        key = (
            int(raw["molecule_type"]),
            str(raw["pf_label"]),
            str(raw["cost_kind"]),
        )
        current = by_group_kind.get(key)
        row = dict(raw)
        if current is None or _source_priority_value(row) >= _source_priority_value(current):
            by_group_kind[key] = row

    ratio_rows: list[dict[str, Any]] = []
    group_keys = {
        (int(row["molecule_type"]), str(row["pf_label"]))
        for row in records
        if "molecule_type" in row and "pf_label" in row
    }
    for molecule_type, pf_label in sorted(
        group_keys, key=lambda item: (item[0], pf_order(item[1]), item[1])
    ):
        max_ld = by_group_kind.get((molecule_type, pf_label, "max_ld"))
        if max_ld is None:
            continue
        denominator = float(max_ld["g_total"])
        if denominator <= 0 or not math.isfinite(denominator):
            continue
        for numerator_kind, ratio_kind in (
            ("actual_best", "actual_best_over_max_ld"),
            ("screening", "screening_best_over_max_ld"),
        ):
            numerator = by_group_kind.get((molecule_type, pf_label, numerator_kind))
            if numerator is None:
                continue
            ratio_rows.append(
                {
                    "molecule": _molecule_label(numerator),
                    "molecule_type": molecule_type,
                    "pf_label": pf_label,
                    "order": int(numerator.get("order", pf_order(pf_label))),
                    "ratio_kind": ratio_kind,
                    "numerator_kind": numerator_kind,
                    "denominator_kind": "max_ld",
                    "g_total_ratio": float(numerator["g_total"]) / denominator,
                    "g_total_num": float(numerator["g_total"]),
                    "g_total_den": denominator,
                    "ld_num": int(numerator["ld"]),
                    "ld_den": int(max_ld["ld"]),
                    "q_opt_num": _optional_float(numerator.get("q_opt")),
                    "kappa_opt_num": _optional_float(numerator.get("kappa_opt")),
                }
            )
    ratio_rows.sort(
        key=lambda item: (
            int(item["molecule_type"]),
            pf_order(str(item["pf_label"])),
            str(item["pf_label"]),
            str(item["ratio_kind"]),
        )
    )
    return ratio_rows


def build_reference_max_ld_sanity_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return max-LD checks for ``df_reference_rz_layers`` records."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if (
            str(record.get("cost_kind")) != "max_ld"
            or str(record.get("df_cost_model"))
            not in {"df_reference_rz_layers", "reference_rz_layers"}
        ):
            continue
        step = _optional_float(record.get("deterministic_step_cost_value"))
        reference_full = _optional_float(record.get("reference_full_rz_depth"))
        if step is None or reference_full is None or reference_full <= 0.0:
            continue
        rows.append(
            {
                "molecule": _molecule_label(record),
                "molecule_type": int(record["molecule_type"]),
                "pf_label": str(record["pf_label"]),
                "order": int(record.get("order", pf_order(str(record["pf_label"])))),
                "ld": int(record["ld"]),
                "df_rank_actual": _optional_int(record.get("df_rank_actual")),
                "deterministic_step_cost_value": step,
                "reference_full_rz_depth": reference_full,
                "step_over_reference": step / reference_full,
                "lambda_r": _optional_float(record.get("lambda_r")),
                "g_rand": _optional_float(record.get("g_rand")),
                "g_total": _optional_float(record.get("g_total")),
                "raw_total_ref_rz_depth": _optional_int(
                    record.get("raw_total_ref_rz_depth")
                ),
                "reference_scale": _optional_float(record.get("reference_scale")),
                "df_reference_rz_layer_key": record.get("df_reference_rz_layer_key"),
                "df_reference_artifact_source": record.get(
                    "df_reference_artifact_source"
                ),
                "df_reference_artifact_kind": record.get("df_reference_artifact_kind"),
            }
        )
    rows.sort(
        key=lambda item: (
            int(item["molecule_type"]),
            pf_order(str(item["pf_label"])),
            str(item["pf_label"]),
        )
    )
    return rows


def write_records_json(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(list(records), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_records_csv(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in records]
    fieldnames = _csv_fieldnames(rows)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_cost_vs_ld_by_system_pf(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    image_format: str = "png",
    dpi: int = 180,
) -> list[Path]:
    """Write one cost-vs-LD plot per ``(molecule, pf_label)`` group."""
    plt = _pyplot()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_group: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for record in records:
        by_group.setdefault(
            (int(record["molecule_type"]), str(record["pf_label"])), []
        ).append(record)

    paths: list[Path] = []
    for (molecule_type, pf_label), group_records in sorted(
        by_group.items(), key=lambda item: (item[0][0], pf_order(item[0][1]), item[0][1])
    ):
        actual = _records_for_kind(group_records, "actual_window")
        screening = _first_record_for_kind(group_records, "screening")
        actual_best = _first_record_for_kind(group_records, "actual_best")
        max_ld = _first_record_for_kind(group_records, "max_ld")
        if not actual and screening is None and max_ld is None:
            continue

        fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
        if actual:
            ax.plot(
                [int(row["ld"]) for row in actual],
                [float(row["g_total"]) for row in actual],
                color="#2f6f9f",
                marker="o",
                linewidth=1.7,
                markersize=5,
                label="actual window",
            )
        if screening is not None:
            ax.axvline(
                int(screening["ld"]),
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                label="screening best LD",
            )
            ax.scatter(
                [int(screening["ld"])],
                [float(screening["g_total"])],
                color="#555555",
                marker="x",
                s=60,
                zorder=4,
                label="screening cost",
            )
        if actual_best is not None:
            ax.scatter(
                [int(actual_best["ld"])],
                [float(actual_best["g_total"])],
                color="#c2410c",
                marker="*",
                s=130,
                zorder=5,
                label="actual best",
            )
        if max_ld is not None:
            ax.scatter(
                [int(max_ld["ld"])],
                [float(max_ld["g_total"])],
                color="#166534",
                marker="D",
                s=58,
                zorder=5,
                label="max LD",
            )

        ax.set_title(f"H{molecule_type} {pf_label}")
        ax.set_xlabel("LD")
        ax.set_ylabel("g_total")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

        filename = f"H{molecule_type}_{_safe_label(pf_label)}_cost_vs_ld.{image_format}"
        path = out_dir / filename
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        paths.append(path)
    return paths


def plot_ratio_summary(
    ratio_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    image_format: str = "png",
    dpi: int = 180,
    write_per_pf: bool = True,
) -> list[Path]:
    """Write combined and per-PF normalized cost-ratio summary plots."""
    plt = _pyplot()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [dict(row) for row in ratio_records]
    if not records:
        return []

    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for pf_label in _sorted_pf_labels({str(row["pf_label"]) for row in records}):
        color = COLOR_MAP.get(pf_label)
        for ratio_kind, linestyle, marker, label_suffix in (
            ("actual_best_over_max_ld", "-", MARKER_MAP.get(pf_label, "o"), "actual"),
            ("screening_best_over_max_ld", "--", "x", "screening"),
        ):
            series = _ratio_series(records, pf_label, ratio_kind)
            if not series:
                continue
            ax.plot(
                [int(row["molecule_type"]) for row in series],
                [float(row["g_total_ratio"]) for row in series],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.6,
                markersize=5,
                label=f"{pf_label} {label_suffix}",
            )
    ax.axhline(1.0, color="#444444", linestyle=":", linewidth=1.0)
    ax.set_xlabel("molecule size")
    ax.set_ylabel("cost / max-LD cost")
    ax.set_title("DF optimized cost ratios")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    combined_path = out_dir / f"cost_ratio_summary_all_pf.{image_format}"
    fig.savefig(combined_path, dpi=int(dpi))
    plt.close(fig)
    paths.append(combined_path)

    if not write_per_pf:
        return paths

    for pf_label in _sorted_pf_labels({str(row["pf_label"]) for row in records}):
        fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
        color = COLOR_MAP.get(pf_label, "#2f6f9f")
        for ratio_kind, linestyle, marker, label in (
            ("actual_best_over_max_ld", "-", "o", "actual best / max LD"),
            ("screening_best_over_max_ld", "--", "x", "screening best / max LD"),
        ):
            series = _ratio_series(records, pf_label, ratio_kind)
            if not series:
                continue
            ax.plot(
                [int(row["molecule_type"]) for row in series],
                [float(row["g_total_ratio"]) for row in series],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.7,
                markersize=5,
                label=label,
            )
        ax.axhline(1.0, color="#444444", linestyle=":", linewidth=1.0)
        ax.set_xlabel("molecule size")
        ax.set_ylabel("cost / max-LD cost")
        ax.set_title(f"{pf_label} cost ratio")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        path = out_dir / f"cost_ratio_summary_{_safe_label(pf_label)}.{image_format}"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        paths.append(path)
    return paths


def plot_optimized_cost_by_pf(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    cost_kind: str = "actual_best",
    include_max_ld_cgs_optimized: bool = True,
    include_max_ld: bool = True,
    include_grouping_deterministic: bool = True,
    image_format: str = "png",
    dpi: int = 180,
) -> Path | None:
    """Write a PF-to-PF optimized-cost comparison over molecule size."""
    plt = _pyplot()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        dict(row)
        for row in records
        if str(row.get("cost_kind")) == str(cost_kind)
        and row.get("g_total") is not None
    ]
    if not selected:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    pf_labels = _sorted_pf_labels({str(row["pf_label"]) for row in selected})
    for pf_label in pf_labels:
        series = sorted(
            [row for row in selected if str(row["pf_label"]) == pf_label],
            key=lambda item: int(item["molecule_type"]),
        )
        ax.plot(
            [int(row["molecule_type"]) for row in series],
            [float(row["g_total"]) for row in series],
            color=COLOR_MAP.get(pf_label),
            marker=MARKER_MAP.get(pf_label, "o"),
            linewidth=1.7,
            markersize=5,
            label=f"{pf_label} optimized",
        )
        if include_max_ld_cgs_optimized:
            max_ld_cgs_series = _cost_series_for_pf(
                records,
                pf_label,
                "max_ld_cgs_optimized",
            )
            if max_ld_cgs_series:
                ax.plot(
                    [int(row["molecule_type"]) for row in max_ld_cgs_series],
                    [float(row["g_total"]) for row in max_ld_cgs_series],
                    color=COLOR_MAP.get(pf_label),
                    marker="s",
                    linestyle="-.",
                    linewidth=1.3,
                    markersize=4.5,
                    alpha=0.85,
                    label=f"{pf_label} max-LD Cgs opt",
                )
        if include_max_ld:
            max_ld_series = _cost_series_for_pf(records, pf_label, "max_ld")
            if max_ld_series:
                ax.plot(
                    [int(row["molecule_type"]) for row in max_ld_series],
                    [float(row["g_total"]) for row in max_ld_series],
                    color=COLOR_MAP.get(pf_label),
                    marker=MARKER_MAP.get(pf_label, "o"),
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                    linestyle="--",
                    linewidth=1.3,
                    markersize=5,
                    alpha=0.85,
                    label=f"{pf_label} max LD",
                )
        if include_grouping_deterministic:
            grouping_series = _cost_series_for_pf(
                records,
                pf_label,
                "grouping_deterministic",
            )
            if grouping_series:
                unit = str(grouping_series[0].get("grouping_cost_unit", "cost"))
                ax.plot(
                    [int(row["molecule_type"]) for row in grouping_series],
                    [float(row["g_total"]) for row in grouping_series],
                    color=COLOR_MAP.get(pf_label),
                    marker="^",
                    linestyle=":",
                    linewidth=1.5,
                    markersize=5,
                    alpha=0.9,
                    label=f"{pf_label} grouping det ({unit})",
                )

    ax.set_xlabel("molecule size")
    ax.set_ylabel("g_total")
    ax.set_title(f"DF optimized and max-LD cost by PF ({cost_kind})")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    path = out_dir / f"optimized_cost_by_pf_{_safe_label(cost_kind)}.{image_format}"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def _best_cost_with_replaced_cgs(
    candidates: Any,
    *,
    c_gs_source: Mapping[str, Any] | None,
    cost_kind: str,
    fallback: Mapping[str, Any],
    epsilon_total: float,
    kappa_mode: str,
    kappa_value: float,
    kappa_min: float,
    kappa_max: float,
    randomized_method: str,
    g_rand: float,
    error_budget_rule: str,
    df_cost_model: str,
    reference_context: Mapping[tuple[int, str], Mapping[str, Any]] | None,
    reference_randomized_cost_mode: str,
) -> dict[str, Any] | None:
    if not isinstance(c_gs_source, Mapping):
        return None
    c_gs_d = _optional_float(c_gs_source.get("c_gs_d"))
    if c_gs_d is None:
        return None
    costs: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        try:
            cost = _cost_for_candidate_with_cgs(
                raw_candidate,
                c_gs_d=c_gs_d,
                c_gs_source=c_gs_source,
                cost_kind=cost_kind,
                fallback=fallback,
                epsilon_total=epsilon_total,
                kappa_mode=kappa_mode,
                kappa_value=kappa_value,
                kappa_min=kappa_min,
                kappa_max=kappa_max,
                randomized_method=randomized_method,
                g_rand=g_rand,
                error_budget_rule=error_budget_rule,
                df_cost_model=df_cost_model,
                reference_context=reference_context,
                reference_randomized_cost_mode=reference_randomized_cost_mode,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if cost.get("g_total") is not None:
            costs.append(cost)
    if not costs:
        return None
    return min(costs, key=lambda item: float(item["g_total"]))


def _best_recomputed_cost(
    candidates: Any,
    *,
    cost_kind: str,
    fallback: Mapping[str, Any],
    epsilon_total: float,
    kappa_mode: str,
    kappa_value: float,
    kappa_min: float,
    kappa_max: float,
    randomized_method: str,
    g_rand: float,
    error_budget_rule: str,
    df_cost_model: str,
    reference_context: Mapping[tuple[int, str], Mapping[str, Any]] | None,
    reference_randomized_cost_mode: str,
) -> dict[str, Any] | None:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return None
    costs = [
        cost
        for raw in candidates
        if isinstance(raw, Mapping)
        for cost in [
            _recompute_cost_like(
                raw,
                cost_kind=cost_kind,
                fallback=fallback,
                epsilon_total=epsilon_total,
                kappa_mode=kappa_mode,
                kappa_value=kappa_value,
                kappa_min=kappa_min,
                kappa_max=kappa_max,
                randomized_method=randomized_method,
                g_rand=g_rand,
                error_budget_rule=error_budget_rule,
                df_cost_model=df_cost_model,
                reference_context=reference_context,
                reference_randomized_cost_mode=reference_randomized_cost_mode,
            )
        ]
        if cost is not None
    ]
    if not costs:
        return None
    return min(costs, key=lambda item: float(item["g_total"]))


def _recompute_cost_like(
    raw: Mapping[str, Any],
    *,
    cost_kind: str,
    fallback: Mapping[str, Any],
    epsilon_total: float,
    kappa_mode: str,
    kappa_value: float,
    kappa_min: float,
    kappa_max: float,
    randomized_method: str,
    g_rand: float,
    error_budget_rule: str,
    df_cost_model: str,
    reference_context: Mapping[tuple[int, str], Mapping[str, Any]] | None,
    reference_randomized_cost_mode: str,
) -> dict[str, Any] | None:
    try:
        molecule_type = int(raw.get("molecule_type", fallback["molecule_type"]))
        pf_label = str(raw.get("pf_label", fallback["pf_label"]))
        order = int(raw.get("order", fallback.get("order", pf_order(pf_label))))
        c_gs_d = float(
            raw.get("c_gs_d", raw.get("c_gs_d_screen", fallback["anchor_c_gs_d"]))
        )
        lambda_r = float(raw["lambda_r"])
        ld = int(raw["ld"])
    except (KeyError, TypeError, ValueError):
        return None

    try:
        cost_inputs = _df_cost_inputs_for_row(
            {**fallback, **raw},
            df_cost_model=df_cost_model,
            reference_context=reference_context,
            reference_randomized_cost_mode=reference_randomized_cost_mode,
            g_rand_input=g_rand,
        )
    except (KeyError, TypeError, ValueError):
        return None

    budget = optimize_error_budget_and_kappa(
        epsilon_total=float(epsilon_total),
        order=order,
        deterministic_step_cost_value=int(cost_inputs["deterministic_step_cost_value"]),
        c_gs=c_gs_d,
        lambda_r=lambda_r,
        randomized_method=str(randomized_method),
        g_rand=float(cost_inputs["g_rand_input"]),
        kappa_mode=str(kappa_mode),
        kappa_value=float(kappa_value),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),
        error_budget_rule=error_budget_rule,
    )
    return {
        "molecule": f"H{molecule_type}",
        "molecule_type": molecule_type,
        "pf_label": pf_label,
        "order": order,
        "ld": ld,
        "cost_kind": cost_kind,
        "g_total": budget.g_total,
        "g_det": budget.g_det,
        "g_rand": budget.g_rand,
        "q_opt": budget.q_ratio,
        "kappa_opt": budget.kappa,
        "eps_qpe_opt": budget.eps_qpe,
        "eps_trot_opt": budget.eps_trot,
        "b_opt": budget.b_value,
        "error_budget_rule": str(error_budget_rule),
        "boundary_hit_q": budget.boundary_hit_q,
        "boundary_hit_kappa": budget.boundary_hit_kappa,
        "c_gs_d": c_gs_d,
        "lambda_r": lambda_r,
        **cost_inputs,
        "df_rank_actual": _optional_int(
            raw.get("df_rank_actual", fallback.get("df_rank_actual"))
        ),
        "source_kind": str(raw.get("source_kind", fallback.get("source_kind", "unknown"))),
        "_artifact_source": raw.get("_artifact_source", fallback.get("_artifact_source")),
    }


def _cost_for_candidate_with_cgs(
    candidate: Mapping[str, Any],
    *,
    c_gs_d: float,
    c_gs_source: Mapping[str, Any],
    cost_kind: str,
    fallback: Mapping[str, Any],
    epsilon_total: float,
    kappa_mode: str,
    kappa_value: float,
    kappa_min: float,
    kappa_max: float,
    randomized_method: str,
    g_rand: float,
    error_budget_rule: str,
    df_cost_model: str,
    reference_context: Mapping[tuple[int, str], Mapping[str, Any]] | None,
    reference_randomized_cost_mode: str,
) -> dict[str, Any]:
    molecule_type = int(candidate.get("molecule_type", fallback["molecule_type"]))
    pf_label = str(candidate.get("pf_label", fallback["pf_label"]))
    order = int(candidate.get("order", fallback.get("order", pf_order(pf_label))))
    cost_inputs = _df_cost_inputs_for_row(
        {**fallback, **candidate},
        df_cost_model=df_cost_model,
        reference_context=reference_context,
        reference_randomized_cost_mode=reference_randomized_cost_mode,
        g_rand_input=g_rand,
    )
    budget = optimize_error_budget_and_kappa(
        epsilon_total=float(epsilon_total),
        order=order,
        deterministic_step_cost_value=int(cost_inputs["deterministic_step_cost_value"]),
        c_gs=float(c_gs_d),
        lambda_r=float(candidate["lambda_r"]),
        randomized_method=str(randomized_method),
        g_rand=float(cost_inputs["g_rand_input"]),
        kappa_mode=str(kappa_mode),
        kappa_value=float(kappa_value),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),
        error_budget_rule=error_budget_rule,
    )
    return {
        "molecule": f"H{molecule_type}",
        "molecule_type": molecule_type,
        "pf_label": pf_label,
        "order": order,
        "ld": int(candidate["ld"]),
        "cost_kind": cost_kind,
        "g_total": budget.g_total,
        "g_det": budget.g_det,
        "g_rand": budget.g_rand,
        "q_opt": budget.q_ratio,
        "kappa_opt": budget.kappa,
        "eps_qpe_opt": budget.eps_qpe,
        "eps_trot_opt": budget.eps_trot,
        "b_opt": budget.b_value,
        "error_budget_rule": str(error_budget_rule),
        "boundary_hit_q": budget.boundary_hit_q,
        "boundary_hit_kappa": budget.boundary_hit_kappa,
        "c_gs_d": float(c_gs_d),
        "cgs_source_kind": str(c_gs_source.get("source_kind", "max_ld")),
        "lambda_r": float(candidate["lambda_r"]),
        **cost_inputs,
        "df_rank_actual": _optional_int(
            candidate.get("df_rank_actual", fallback.get("df_rank_actual"))
        ),
        "source_kind": "max_ld_cgs_screening_optimization",
        "_artifact_source": fallback.get("_artifact_source"),
    }


def _df_cost_inputs_for_row(
    row: Mapping[str, Any],
    *,
    df_cost_model: str,
    reference_context: Mapping[tuple[int, str], Mapping[str, Any]] | None,
    reference_randomized_cost_mode: str,
    g_rand_input: float,
) -> dict[str, Any]:
    df_cost_model = _normalize_df_cost_model(df_cost_model)
    reference_randomized_cost_mode = _normalize_reference_randomized_cost_mode(
        reference_randomized_cost_mode
    )
    raw_step = _total_ref_rz_depth(row)
    if df_cost_model == "qiskit_decomposed_rz_depth":
        return {
            "df_cost_model": df_cost_model,
            "reference_randomized_cost_mode": "input",
            "raw_total_ref_rz_depth": int(raw_step),
            "total_ref_rz_depth": int(raw_step),
            "deterministic_step_cost_value": int(raw_step),
            "g_rand_input": float(g_rand_input),
        }

    molecule_type = int(row["molecule_type"])
    pf_label = str(row["pf_label"])
    ld = int(row["ld"])
    df_rank_actual = _optional_int(row.get("df_rank_actual"))
    key = (molecule_type, pf_label)
    if reference_context is None or key not in reference_context:
        raise KeyError(f"missing reference context for H{molecule_type} {pf_label}")
    ctx = reference_context[key]
    reference_full = int(ctx.get("reference_full_rz_depth") or 0)
    scale = _optional_float(ctx.get("reference_scale"))
    if reference_full <= 0 or scale is None:
        raise KeyError(f"incomplete reference context for H{molecule_type} {pf_label}")
    if df_rank_actual is None:
        df_rank_actual = _optional_int(ctx.get("df_rank_actual"))
    reference_step = _scaled_reference_step(raw_step, scale)
    if df_rank_actual is not None and ld == df_rank_actual:
        reference_step = int(reference_full)
    reference_tail = max(0.0, float(reference_full) - float(reference_step))
    reference_ld0 = _optional_float(ctx.get("ld0_reference_rz_depth"))
    sample_cost = _reference_randomized_sample_cost(
        mode=reference_randomized_cost_mode,
        g_rand_input=float(g_rand_input),
        lambda_r=_optional_float(row.get("lambda_r")),
        reference_full=float(reference_full),
        reference_ld0=reference_ld0,
        reference_tail=reference_tail,
        ld=ld,
        df_rank_actual=df_rank_actual,
    )
    return {
        "df_cost_model": df_cost_model,
        "reference_randomized_cost_mode": reference_randomized_cost_mode,
        "raw_total_ref_rz_depth": int(raw_step),
        "total_ref_rz_depth": int(reference_step),
        "deterministic_step_cost_value": int(reference_step),
        "reference_full_rz_depth": int(reference_full),
        "reference_scale": float(scale),
        "reference_tail_rz_depth": float(reference_tail),
        "df_reference_rz_layer_key": ctx.get("df_reference_rz_layer_key"),
        "df_reference_artifact_source": ctx.get("df_reference_artifact_source"),
        "df_reference_artifact_kind": ctx.get("df_reference_artifact_kind"),
        "g_rand_input": float(sample_cost),
    }


def _reference_randomized_sample_cost(
    *,
    mode: str,
    g_rand_input: float,
    lambda_r: float | None,
    reference_full: float,
    reference_ld0: float | None,
    reference_tail: float,
    ld: int,
    df_rank_actual: int | None,
) -> float:
    if lambda_r is not None and lambda_r <= 0.0:
        return 0.0
    if mode == "input":
        return float(g_rand_input)
    if df_rank_actual is None or df_rank_actual <= 0:
        return float(g_rand_input)
    one_body_cost = float(reference_ld0 or 0.0)
    fragment_total = max(0.0, float(reference_full) - one_body_cost)
    mean_fragment = fragment_total / float(df_rank_actual)
    if mode == "mean_fragment":
        return max(1.0, mean_fragment)
    tail_count = max(0, int(df_rank_actual) - int(ld))
    if mode == "tail_mean_fragment":
        if tail_count <= 0:
            return 0.0
        return max(1.0, float(reference_tail) / float(tail_count))
    if mode == "tail_total":
        return max(1.0, float(reference_tail))
    raise ValueError(f"Unsupported reference randomized cost mode: {mode}")


def _scaled_reference_step(raw_step: float, scale: float) -> int:
    return max(1, int(round(float(raw_step) * float(scale))))


def _normalize_df_cost_model(value: str) -> str:
    normalized = str(value)
    if normalized not in DF_COST_MODELS:
        raise ValueError(f"Unsupported DF cost model: {value}")
    if normalized == "reference_rz_layers":
        return "df_reference_rz_layers"
    return normalized


def _normalize_reference_randomized_cost_mode(value: str) -> str:
    normalized = str(value)
    if normalized not in REFERENCE_RANDOMIZED_COST_MODES:
        raise ValueError(f"Unsupported reference randomized cost mode: {value}")
    return normalized


def _df_reference_hamiltonian_name(molecule_type: int) -> str:
    return _grouping_hamiltonian_name(molecule_type)


def _df_reference_target_names(ham_name: str, pf_label: str) -> list[str]:
    return [
        f"{ham_name}_Operator_{pf_label}",
        f"{ham_name}_Operator_{pf_label}_ave",
    ]


def _pick_df_reference_rz_layer_value(
    payload: Any,
    *,
    preferred_key: str | None,
) -> tuple[str, float]:
    rz_layers: Mapping[str, Any]
    if isinstance(payload, Mapping) and isinstance(payload.get("rz_layers"), Mapping):
        rz_layers = payload["rz_layers"]
    elif isinstance(payload, Mapping):
        rz_layers = payload
    else:
        raise ValueError("DF reference payload must be a mapping.")

    candidate_keys: list[str] = []
    if preferred_key:
        candidate_keys.append(str(preferred_key))
    candidate_keys.extend(
        [
            "total_ref_rz_depth",
            "ref_rz_depth",
            "u_ref_rz_depth",
            "d_ref_rz_depth",
            "total_nonclifford_z_coloring_depth",
            "total_nonclifford_z_depth",
            "total_nonclifford_rz_depth",
            "total_rz_depth",
        ]
    )
    for key in dict.fromkeys(candidate_keys):
        if key not in rz_layers:
            continue
        value = _optional_float(rz_layers.get(key))
        if value is not None and value > 0.0:
            return key, value
    raise ValueError(
        "No positive DF reference RZ-layer metric found. "
        f"available keys={list(rz_layers.keys())}"
    )


def _preferred_paths_by_suffix(directory: Path, suffix: str) -> list[Path]:
    return preferred_final_or_partial_paths(
        list(directory.glob(f"*{suffix}.json"))
        + list(directory.glob(f"*{suffix}.partial.json"))
    )


def _load_rows_with_source(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in load_json_rows(path):
            row["_artifact_source"] = str(path)
            row["_artifact_source_is_partial"] = path.name.endswith(".partial.json")
            rows.append(row)
    return rows


def _deduplicate_rows(
    rows: Sequence[dict[str, Any]],
    key_fn: Any,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], tuple[tuple[int, int], dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            key = key_fn(row)
        except (KeyError, TypeError, ValueError):
            passthrough.append(dict(row))
            continue
        rank = (_source_priority_from_row(row), index)
        current = by_key.get(key)
        if current is None or rank >= current[0]:
            by_key[key] = (rank, dict(row))
    deduped = [entry[1] for entry in by_key.values()]
    deduped.extend(passthrough)
    deduped.sort(
        key=lambda item: (
            _source_priority_from_row(item),
            str(item.get("_artifact_source")),
            str(item),
        )
    )
    return deduped


def _summary_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return (int(row["molecule_type"]), str(row["pf_label"]))


def _cgs_key(row: Mapping[str, Any]) -> tuple[int, str, int, str]:
    return (
        int(row["molecule_type"]),
        str(row["pf_label"]),
        int(row["ld"]),
        str(row.get("source_kind", "unknown")),
    )


def _logical_artifact_name(path: Path) -> str:
    if path.name.endswith(".partial.json"):
        return path.name[: -len(".partial.json")] + ".json"
    return path.name


def _source_priority(path: Path) -> int:
    return 0 if path.name.endswith(".partial.json") else 1


def _source_priority_from_row(row: Mapping[str, Any]) -> int:
    return 0 if row.get("_artifact_source_is_partial") else 1


def _source_priority_value(row: Mapping[str, Any]) -> tuple[int, str]:
    source = str(row.get("_artifact_source", ""))
    return (_source_priority_from_row(row), source)


def _row_passes_filter(
    row: Mapping[str, Any],
    *,
    molecule_min: int | None,
    molecule_max: int | None,
    pf_filter: set[str] | None,
) -> bool:
    try:
        molecule_type = int(row["molecule_type"])
        pf_label = str(row["pf_label"])
    except (KeyError, TypeError, ValueError):
        return False
    if molecule_min is not None and molecule_type < int(molecule_min):
        return False
    if molecule_max is not None and molecule_type > int(molecule_max):
        return False
    if pf_filter is not None and pf_label not in pf_filter:
        return False
    return True


def _normalize_cost_like(
    raw: Any,
    *,
    cost_kind: str,
    fallback: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        molecule_type = int(raw.get("molecule_type", fallback["molecule_type"]))
        pf_label = str(raw.get("pf_label", fallback["pf_label"]))
        ld = int(raw["ld"])
    except (KeyError, TypeError, ValueError):
        return None

    c_gs_d = raw.get("c_gs_d", raw.get("c_gs_d_screen", fallback.get("anchor_c_gs_d")))
    record = {
        "molecule": _molecule_label({"molecule_type": molecule_type}),
        "molecule_type": molecule_type,
        "pf_label": pf_label,
        "order": int(raw.get("order", fallback.get("order", pf_order(pf_label)))),
        "ld": ld,
        "cost_kind": cost_kind,
        "g_total": _optional_float(raw.get("g_total")),
        "g_det": _optional_float(raw.get("g_det")),
        "g_rand": _optional_float(raw.get("g_rand")),
        "q_opt": _optional_float(raw.get("q_opt")),
        "kappa_opt": _optional_float(raw.get("kappa_opt")),
        "c_gs_d": _optional_float(c_gs_d),
        "error_budget_rule": str(
            raw.get("error_budget_rule", fallback.get("error_budget_rule", "stored"))
        ),
        "lambda_r": _optional_float(raw.get("lambda_r")),
        "df_cost_model": str(
            raw.get("df_cost_model", fallback.get("df_cost_model", "stored"))
        ),
        "reference_randomized_cost_mode": str(
            raw.get(
                "reference_randomized_cost_mode",
                fallback.get("reference_randomized_cost_mode", "stored"),
            )
        ),
        "raw_total_ref_rz_depth": _optional_int(raw.get("raw_total_ref_rz_depth")),
        "deterministic_step_cost_value": _optional_float(
            raw.get("deterministic_step_cost_value")
        ),
        "reference_full_rz_depth": _optional_int(raw.get("reference_full_rz_depth")),
        "reference_scale": _optional_float(raw.get("reference_scale")),
        "reference_tail_rz_depth": _optional_float(raw.get("reference_tail_rz_depth")),
        "df_reference_rz_layer_key": raw.get("df_reference_rz_layer_key"),
        "df_reference_artifact_source": raw.get("df_reference_artifact_source"),
        "df_reference_artifact_kind": raw.get("df_reference_artifact_kind"),
        "g_rand_input": _optional_float(raw.get("g_rand_input")),
        "total_ref_rz_depth": _optional_int(raw.get("total_ref_rz_depth")),
        "df_rank_actual": _optional_int(raw.get("df_rank_actual", fallback.get("df_rank_actual"))),
        "source_kind": str(raw.get("source_kind", fallback.get("source_kind", "unknown"))),
        "_artifact_source": raw.get("_artifact_source", fallback.get("_artifact_source")),
    }
    return record


def _merge_group_fields(
    record: Mapping[str, Any],
    group_fields: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(group_fields)
    merged.update(record)
    merged["molecule"] = _molecule_label(merged)
    merged["molecule_type"] = int(merged["molecule_type"])
    merged["pf_label"] = str(merged["pf_label"])
    merged["order"] = int(merged.get("order", pf_order(merged["pf_label"])))
    return merged


def _record_sort_key(record: Mapping[str, Any]) -> tuple[int, int, str, int, int]:
    return (
        int(record["molecule_type"]),
        pf_order(str(record["pf_label"])),
        str(record["pf_label"]),
        int(record["ld"]),
        COST_KIND_ORDER.get(str(record.get("cost_kind")), 99),
    )


def _load_grouping_coeff(
    molecule_type: int,
    pf_label: str,
    *,
    use_original: bool,
) -> tuple[float, str]:
    ham_name = _grouping_hamiltonian_name(molecule_type)
    last_error: Exception | None = None
    for target in _grouping_coeff_target_names(
        ham_name,
        pf_label,
        use_original=use_original,
    ):
        try:
            data = load_data(target, gr=True, use_original=use_original)
        except Exception as exc:
            last_error = exc
            continue
        coeff = _extract_grouping_coeff(data)
        source = str(pickle_dir(True, use_original=use_original) / target)
        return coeff, source
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"grouping coeff not found: {ham_name} / {pf_label}")


def _grouping_coeff_target_names(
    ham_name: str,
    pf_label: str,
    *,
    use_original: bool,
) -> list[str]:
    labels: list[str] = []
    if use_original:
        legacy = _GROUPING_ORIGINAL_PF_LABEL_MAP.get(pf_label)
        if legacy is not None:
            labels.append(legacy)
    labels.append(pf_label)
    labels = list(dict.fromkeys(labels))
    targets: list[str] = []
    for label in labels:
        targets.append(f"{ham_name}_Operator_{label}_ave")
        targets.append(f"{ham_name}_Operator_{label}")
    return targets


def _extract_grouping_coeff(data: Any) -> float:
    if isinstance(data, Mapping):
        data = data.get("coeff")
    coeff = float(data)
    if not math.isfinite(coeff):
        raise ValueError("grouping coeff must be finite.")
    return coeff


def _grouping_hamiltonian_name(molecule_type: int) -> str:
    molecule = f"H{int(molecule_type)}"
    if int(molecule_type) % 2 == 0:
        return f"{molecule}_sto-3g_singlet_distance_100_charge_0_grouping"
    return f"{molecule}_sto-3g_triplet_1+_distance_100_charge_1_grouping"


def _system_pf_sort_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(record["molecule_type"]),
        pf_order(str(record["pf_label"])),
        str(record["pf_label"]),
    )


def _total_ref_rz_depth(row: Mapping[str, Any]) -> int:
    value = row.get("total_ref_rz_depth")
    if value is None and isinstance(row.get("df_step_cost"), Mapping):
        value = row["df_step_cost"].get("total_ref_rz_depth")
    if value is None:
        raise KeyError("total_ref_rz_depth")
    return int(value)


def _molecule_label(row: Mapping[str, Any]) -> str:
    molecule = row.get("molecule")
    if molecule:
        return str(molecule)
    return f"H{int(row['molecule_type'])}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "molecule",
        "molecule_type",
        "pf_label",
        "order",
        "ld",
        "cost_kind",
        "ratio_kind",
        "g_total",
        "g_total_ratio",
        "g_det",
        "g_rand",
        "g_rand_input",
        "q_opt",
        "kappa_opt",
        "c_gs_d",
        "error_budget_rule",
        "lambda_r",
        "df_cost_model",
        "reference_randomized_cost_mode",
        "qpe_beta",
        "raw_total_ref_rz_depth",
        "total_ref_rz_depth",
        "deterministic_step_cost_value",
        "reference_full_rz_depth",
        "reference_scale",
        "reference_tail_rz_depth",
        "df_reference_rz_layer_key",
        "df_reference_artifact_source",
        "df_reference_artifact_kind",
        "grouping_cost_unit",
        "grouping_cost_model",
        "pauli_rotations_per_step",
        "df_rank_actual",
        "ld_anchor",
        "source_kind",
        "g_total_num",
        "g_total_den",
        "ld_num",
        "ld_den",
    ]
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _records_for_kind(
    records: Sequence[Mapping[str, Any]],
    cost_kind: str,
) -> list[Mapping[str, Any]]:
    selected = [
        record
        for record in records
        if record.get("cost_kind") == cost_kind and record.get("g_total") is not None
    ]
    return sorted(selected, key=lambda item: int(item["ld"]))


def _first_record_for_kind(
    records: Sequence[Mapping[str, Any]],
    cost_kind: str,
) -> Mapping[str, Any] | None:
    selected = _records_for_kind(records, cost_kind)
    return selected[0] if selected else None


def _ratio_series(
    records: Sequence[Mapping[str, Any]],
    pf_label: str,
    ratio_kind: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        [
            row
            for row in records
            if str(row.get("pf_label")) == pf_label
            and str(row.get("ratio_kind")) == ratio_kind
        ],
        key=lambda item: int(item["molecule_type"]),
    )


def _cost_series_for_pf(
    records: Sequence[Mapping[str, Any]],
    pf_label: str,
    cost_kind: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        [
            row
            for row in records
            if str(row.get("pf_label")) == pf_label
            and str(row.get("cost_kind")) == cost_kind
            and row.get("g_total") is not None
        ],
        key=lambda item: int(item["molecule_type"]),
    )


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def _sorted_pf_labels(labels: Iterable[str]) -> list[str]:
    return sorted(labels, key=lambda label: (pf_order(label), label))


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt

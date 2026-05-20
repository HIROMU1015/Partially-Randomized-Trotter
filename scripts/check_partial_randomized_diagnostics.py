from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

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
    PFLabel,
    pf_order,
)
from trotterlib.df_cost_plotting import (  # noqa: E402
    DEFAULT_ANCHOR_REFINEMENT_DIR,
    DEFAULT_MAX_LD_DIR,
    build_optimized_cost_records,
    collect_optimized_cost_comparisons,
    load_df_cost_plot_inputs,
)
from trotterlib.df_hamiltonian import build_df_h_d_from_molecule  # noqa: E402
from trotterlib.df_partial_randomized_pf import (  # noqa: E402
    _DF_COST_BASIS_GATES,
    _analytic_d_only_rz_cost,
    _build_d_only_cost_circuit,
    _rz_depth_from_circuit,
    get_or_compute_cached_df_cgs_fit,
    load_df_cgs_json_cache,
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)
from trotterlib.df_screening_cost import (  # noqa: E402
    build_rank_ordered_df_cost_blocks,
    df_screening_costs_for_all_ld,
)
from trotterlib.df_trotter.circuit import build_df_trotter_circuit  # noqa: E402
from trotterlib.partial_randomized_pf import (  # noqa: E402
    default_perturbation_t_values,
    randomized_prefactor_b0,
)


DEFAULT_OUTPUT_DIR = PARTIAL_RANDOMIZED_ARTIFACTS_DIR / "diagnostics"
DEFAULT_PF_LABELS = ("2nd", "4th", "4th(new_2)", "8th(Morales)")


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _parse_molecules(raw: str | None, *, default_min: int, default_max: int) -> tuple[int, ...]:
    if not raw:
        return tuple(range(int(default_min), int(default_max) + 1))
    values: list[int] = []
    for item in _parse_csv(raw):
        item = item.removeprefix("H")
        if "-" in item:
            lo_raw, hi_raw = item.split("-", 1)
            lo = int(lo_raw.removeprefix("H"))
            hi = int(hi_raw.removeprefix("H"))
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(item))
    return tuple(dict.fromkeys(values))


def _parse_ld_values(raw: str | None, *, rank: int | None = None) -> tuple[int, ...]:
    if not raw:
        return ()
    values: list[int] = []
    for item in _parse_csv(raw):
        lower = item.lower()
        if lower in {"max", "rank"}:
            if rank is None:
                raise ValueError("'max' LD requires a known DF rank.")
            values.append(int(rank))
            continue
        if "-" in item:
            lo_raw, hi_raw = item.split("-", 1)
            lo = int(lo_raw)
            hi = int(hi_raw)
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(item))
    return tuple(dict.fromkeys(values))


def _parse_targets(raw_targets: Sequence[str]) -> tuple[tuple[int, str, int], ...]:
    targets: list[tuple[int, str, int]] = []
    for raw in raw_targets:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Targets must be formatted as Hn:pf_label:ld, e.g. H8:8th(Morales):8."
            )
        molecule = int(parts[0].removeprefix("H"))
        pf_label = parts[1]
        ld = int(parts[2])
        targets.append((molecule, pf_label, ld))
    return tuple(targets)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        if abs(value.imag) < 1e-12:
            return float(value.real)
        return {"real": value.real, "imag": value.imag}
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


def _write_rows(base_path: Path, rows: Sequence[Mapping[str, Any]], payload: Any) -> None:
    _write_json(base_path.with_suffix(".json"), payload)
    _write_csv(base_path.with_suffix(".csv"), rows)


def _molecule_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(row["molecule_type"]),
        pf_order(str(row.get("pf_label", "2nd"))),
        str(row.get("pf_label", "")),
    )


def _row_molecule(row: Mapping[str, Any]) -> str:
    return str(row.get("molecule") or f"H{int(row['molecule_type'])}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _best_rows_by_molecule(
    rows: Iterable[Mapping[str, Any]],
    *,
    cost_kind: str = "actual_best",
) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("cost_kind")) != cost_kind:
            continue
        g_total = _safe_float(row.get("g_total"))
        if g_total is None:
            continue
        molecule_type = int(row["molecule_type"])
        if molecule_type not in best or g_total < float(best[molecule_type]["g_total"]):
            best[molecule_type] = dict(row)
    return [best[key] for key in sorted(best)]


def _trim_cost_row(row: Mapping[str, Any], *, scale: float, b0: float) -> dict[str, Any]:
    g_total = float(row["g_total"])
    g_rand = float(row.get("g_rand", 0.0))
    return {
        "g_rand_scale": float(scale),
        "b0": float(b0),
        "molecule": _row_molecule(row),
        "molecule_type": int(row["molecule_type"]),
        "pf_label": str(row["pf_label"]),
        "order": int(row["order"]),
        "cost_kind": str(row.get("cost_kind", "")),
        "ld": int(row["ld"]),
        "df_rank_actual": row.get("df_rank_actual"),
        "g_total": g_total,
        "g_det": float(row.get("g_det", 0.0)),
        "g_rand": g_rand,
        "g_rand_fraction": 0.0 if g_total == 0.0 else g_rand / g_total,
        "q_opt": row.get("q_opt"),
        "kappa_opt": row.get("kappa_opt"),
        "lambda_r": row.get("lambda_r"),
        "c_gs_d": row.get("c_gs_d"),
        "total_ref_rz_depth": row.get("total_ref_rz_depth"),
        "source_kind": row.get("source_kind"),
        "error_budget_rule": row.get("error_budget_rule"),
    }


def run_rand_sensitivity(args: argparse.Namespace) -> int:
    pf_labels = _parse_csv(args.pf_labels) or DEFAULT_PF_LABELS
    inputs = load_df_cost_plot_inputs(
        anchor_refinement_dir=args.anchor_refinement_dir,
        max_ld_dir=args.max_ld_dir,
    )
    rows_out: list[dict[str, Any]] = []
    scale_summaries: list[dict[str, Any]] = []
    for scale in _parse_float_csv(args.scales):
        g_rand = float(args.g_rand_base) * float(scale)
        b0 = randomized_prefactor_b0(str(args.randomized_method), g_rand)
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
            g_rand=g_rand,
            error_budget_rule=str(args.error_budget_rule),
            df_cost_model=str(args.df_cost_model),
            reference_randomized_cost_mode=str(args.reference_randomized_cost_mode),
        )
        records = build_optimized_cost_records(comparisons)
        best_rows = _best_rows_by_molecule(records, cost_kind=str(args.cost_kind))
        trimmed = [_trim_cost_row(row, scale=scale, b0=b0) for row in best_rows]
        rows_out.extend(trimmed)
        if trimmed:
            fractions = [float(row["g_rand_fraction"]) for row in trimmed]
            scale_summaries.append(
                {
                    "g_rand_scale": float(scale),
                    "g_rand_input": float(g_rand),
                    "b0": float(b0),
                    "num_best_rows": len(trimmed),
                    "best_pf_by_molecule": {
                        row["molecule"]: row["pf_label"] for row in trimmed
                    },
                    "best_ld_by_molecule": {row["molecule"]: row["ld"] for row in trimmed},
                    "g_rand_fraction_min": min(fractions),
                    "g_rand_fraction_median": float(np.median(fractions)),
                    "g_rand_fraction_max": max(fractions),
                }
            )
    payload = {
        "schema_version": 1,
        "diagnostic": "randomized_prefactor_sensitivity",
        "note": "Scaling g_rand scales B0 linearly for the current model.",
        "epsilon_total": float(args.epsilon_total),
        "error_budget_rule": str(args.error_budget_rule),
        "df_cost_model": str(args.df_cost_model),
        "cost_kind": str(args.cost_kind),
        "pf_labels": list(pf_labels),
        "summaries": scale_summaries,
        "best_rows": rows_out,
    }
    out_base = Path(args.output_dir) / "rand_sensitivity"
    _write_rows(out_base, rows_out, payload)
    print(f"wrote {out_base.with_suffix('.json')}")
    print(f"wrote {out_base.with_suffix('.csv')}")
    return 0


def _rank_positions(values: Sequence[float]) -> list[int]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: (-float(item[1]), item[0]))
    ranks = [0 for _ in values]
    for rank, (idx, _value) in enumerate(indexed, start=1):
        ranks[idx] = rank
    return ranks


def _spearman_from_values(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x = np.asarray(_rank_positions(x_values), dtype=np.float64)
    y = np.asarray(_rank_positions(y_values), dtype=np.float64)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _df_fragment_pauli_l1(
    hamiltonian: Any,
    block_index: int,
    *,
    coeff_tol: float,
    include_identity: bool,
) -> tuple[float, int]:
    from openfermion import FermionOperator
    from openfermion.transforms import jordan_wigner, normal_ordered

    lam = complex(hamiltonian.lambdas[int(block_index)])
    g_mat = np.asarray(hamiltonian.g_matrices[int(block_index)], dtype=np.complex128)
    op = FermionOperator()
    n = int(g_mat.shape[0])
    for p in range(n):
        for q in range(n):
            gpq = complex(g_mat[p, q])
            if abs(gpq) <= coeff_tol:
                continue
            for r in range(n):
                for s in range(n):
                    coeff = lam * gpq * complex(g_mat[r, s])
                    if abs(coeff) <= coeff_tol:
                        continue
                    op += FermionOperator(((p, 1), (q, 0), (r, 1), (s, 0)), coeff)
    qop = jordan_wigner(normal_ordered(op))
    l1 = 0.0
    term_count = 0
    for term, coeff in qop.terms.items():
        if not include_identity and len(term) == 0:
            continue
        value = abs(complex(coeff))
        if value <= coeff_tol:
            continue
        l1 += float(value)
        term_count += 1
    return l1, term_count


def _lambda_proxy_rows_for_molecule(
    molecule_type: int,
    *,
    include_pauli_l1: bool,
    pauli_l1_max_molecule: int,
    coeff_tol: float,
    include_identity: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hamiltonian, _sector = build_df_h_d_from_molecule(int(molecule_type))
    lambda_frob = []
    abs_lambda = []
    pauli_l1_values: list[float | None] = []
    pauli_term_counts: list[int | None] = []
    for idx in range(hamiltonian.n_blocks):
        lam = float(hamiltonian.lambdas[idx])
        g_mat = np.asarray(hamiltonian.g_matrices[idx])
        lambda_frob.append(float(abs(lam) * (np.linalg.norm(g_mat, ord="fro") ** 2)))
        abs_lambda.append(abs(lam))
        if include_pauli_l1 and int(molecule_type) <= int(pauli_l1_max_molecule):
            l1, n_terms = _df_fragment_pauli_l1(
                hamiltonian,
                idx,
                coeff_tol=float(coeff_tol),
                include_identity=include_identity,
            )
            pauli_l1_values.append(l1)
            pauli_term_counts.append(n_terms)
        else:
            pauli_l1_values.append(None)
            pauli_term_counts.append(None)

    rank_frob = _rank_positions(lambda_frob)
    rank_abs = _rank_positions(abs_lambda)
    rank_l1 = (
        _rank_positions([float(v or 0.0) for v in pauli_l1_values])
        if all(v is not None for v in pauli_l1_values)
        else [None for _ in pauli_l1_values]
    )
    rows: list[dict[str, Any]] = []
    for idx in range(hamiltonian.n_blocks):
        rows.append(
            {
                "molecule": f"H{int(molecule_type)}",
                "molecule_type": int(molecule_type),
                "df_rank_actual": int(hamiltonian.n_blocks),
                "fragment_index": int(idx),
                "lambda": float(hamiltonian.lambdas[idx]),
                "abs_lambda": abs_lambda[idx],
                "lambda_frobenius_squared": lambda_frob[idx],
                "pauli_l1_norm": pauli_l1_values[idx],
                "pauli_term_count": pauli_term_counts[idx],
                "rank_lambda_frobenius_squared": rank_frob[idx],
                "rank_abs_lambda": rank_abs[idx],
                "rank_pauli_l1_norm": rank_l1[idx],
            }
        )
    summary = {
        "molecule": f"H{int(molecule_type)}",
        "molecule_type": int(molecule_type),
        "df_rank_actual": int(hamiltonian.n_blocks),
        "spearman_lambda_frobenius_squared_vs_abs_lambda": _spearman_from_values(
            lambda_frob, abs_lambda
        ),
        "spearman_lambda_frobenius_squared_vs_pauli_l1": (
            _spearman_from_values(lambda_frob, [float(v or 0.0) for v in pauli_l1_values])
            if all(v is not None for v in pauli_l1_values)
            else None
        ),
        "spearman_abs_lambda_vs_pauli_l1": (
            _spearman_from_values(abs_lambda, [float(v or 0.0) for v in pauli_l1_values])
            if all(v is not None for v in pauli_l1_values)
            else None
        ),
    }
    return rows, summary


def run_lambda_proxy(args: argparse.Namespace) -> int:
    molecules = _parse_molecules(args.molecules, default_min=3, default_max=5)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for molecule_type in molecules:
        molecule_rows, summary = _lambda_proxy_rows_for_molecule(
            molecule_type,
            include_pauli_l1=bool(args.include_pauli_l1),
            pauli_l1_max_molecule=int(args.pauli_l1_max_molecule),
            coeff_tol=float(args.coeff_tol),
            include_identity=bool(args.include_identity),
        )
        rows.extend(molecule_rows)
        summaries.append(summary)
    payload = {
        "schema_version": 1,
        "diagnostic": "lambda_r_proxy_comparison",
        "molecules": [f"H{m}" for m in molecules],
        "include_pauli_l1": bool(args.include_pauli_l1),
        "pauli_l1_max_molecule": int(args.pauli_l1_max_molecule),
        "include_identity": bool(args.include_identity),
        "coeff_tol": float(args.coeff_tol),
        "summaries": summaries,
        "rows": rows,
    }
    out_base = Path(args.output_dir) / "lambda_proxy"
    _write_rows(out_base, rows, payload)
    print(f"wrote {out_base.with_suffix('.json')}")
    print(f"wrote {out_base.with_suffix('.csv')}")
    return 0


def _load_all_cgs_rows(anchor_refinement_dir: Path, max_ld_dir: Path) -> list[dict[str, Any]]:
    from trotterlib.df_cost_plotting import (  # noqa: WPS433
        load_anchor_cgs_rows,
        load_max_ld_cgs_rows,
        load_refinement_cgs_rows,
    )

    rows = []
    for source, source_rows in (
        ("anchor", load_anchor_cgs_rows(anchor_refinement_dir)),
        ("refinement", load_refinement_cgs_rows(anchor_refinement_dir)),
        ("max_ld", load_max_ld_cgs_rows(max_ld_dir)),
    ):
        for row in source_rows:
            if row.get("c_gs_d") is None:
                continue
            patched = dict(row)
            patched["diagnostic_source"] = source
            rows.append(patched)
    rows.sort(key=lambda row: (int(row["molecule_type"]), pf_order(str(row["pf_label"])), int(row["ld"])))
    return rows


def _cgs_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    order = int(row.get("order", pf_order(str(row["pf_label"]))))
    fit_slope = _safe_float(row.get("fit_slope"))
    fixed = _safe_float(row.get("fixed_order_coeff", row.get("c_gs_d")))
    free = _safe_float(row.get("fit_coeff"))
    slope_ratio = None if fit_slope is None else fit_slope / float(order)
    return {
        "molecule": _row_molecule(row),
        "molecule_type": int(row["molecule_type"]),
        "pf_label": str(row["pf_label"]),
        "order": order,
        "ld": int(row["ld"]),
        "df_rank_actual": row.get("df_rank_actual"),
        "source_kind": row.get("source_kind"),
        "diagnostic_source": row.get("diagnostic_source"),
        "c_gs_d": row.get("c_gs_d"),
        "fit_slope": fit_slope,
        "fit_slope_over_order": slope_ratio,
        "fit_slope_rel_error": None if slope_ratio is None else slope_ratio - 1.0,
        "fit_coeff": free,
        "fixed_order_coeff": fixed,
        "fit_coeff_over_fixed_order": (
            None if free is None or fixed in {None, 0.0} else free / fixed
        ),
        "num_t_values": len(row.get("t_values", []) or []),
        "t_values": row.get("t_values"),
        "perturbation_errors": row.get("perturbation_errors"),
        "lambda_r": row.get("lambda_r"),
        "total_ref_rz_depth": row.get("total_ref_rz_depth"),
        "_artifact_source": row.get("_artifact_source"),
    }


def run_cgs_fit_summary(args: argparse.Namespace) -> int:
    rows_raw = _load_all_cgs_rows(Path(args.anchor_refinement_dir), Path(args.max_ld_dir))
    molecule_filter = set(_parse_molecules(args.molecules, default_min=3, default_max=13)) if args.molecules else None
    pf_filter = set(_parse_csv(args.pf_labels)) if args.pf_labels else None
    rows = []
    for row in rows_raw:
        if molecule_filter is not None and int(row["molecule_type"]) not in molecule_filter:
            continue
        if pf_filter is not None and str(row["pf_label"]) not in pf_filter:
            continue
        rows.append(_cgs_summary_row(row))
    payload = {
        "schema_version": 1,
        "diagnostic": "cgs_fit_summary",
        "rows": rows,
        "summary": {
            "num_rows": len(rows),
            "num_with_fit_slope": sum(row["fit_slope"] is not None for row in rows),
        },
    }
    out_base = Path(args.output_dir) / "cgs_fit_summary"
    _write_rows(out_base, rows, payload)
    print(f"wrote {out_base.with_suffix('.json')}")
    print(f"wrote {out_base.with_suffix('.csv')}")
    return 0


def _window_values(base: Sequence[float], *, shift: float, scale: float) -> tuple[float, ...]:
    center = float(np.mean(np.asarray(base, dtype=np.float64)))
    out = tuple(center + (float(value) - center) * float(scale) + float(shift) for value in base)
    if any(value <= 0.0 for value in out):
        raise ValueError(f"Invalid shifted/scaled t-values: {out}")
    return out


def _fit_result_row(
    result: Any,
    *,
    molecule_type: int,
    window_label: str,
) -> dict[str, Any]:
    fixed = float(result.fit_coeff_fixed_order)
    free = None if result.fit_coeff is None else float(result.fit_coeff)
    slope = None if result.fit_slope is None else float(result.fit_slope)
    return {
        "window_label": window_label,
        "molecule": f"H{int(molecule_type)}",
        "molecule_type": int(molecule_type),
        "pf_label": str(result.pf_label),
        "order": int(result.order),
        "ld": int(result.ld),
        "df_rank_actual": int(result.df_rank_actual),
        "lambda_r": float(result.lambda_r),
        "c_gs_d": float(result.coeff),
        "fixed_order_coeff": fixed,
        "fit_slope": slope,
        "fit_slope_over_order": None if slope is None else slope / float(result.order),
        "fit_coeff": free,
        "fit_coeff_over_fixed_order": None if free is None or fixed == 0.0 else free / fixed,
        "t_values": list(result.t_values),
        "perturbation_errors": list(result.perturbation_errors),
        "evolution_backend": str(result.evolution_backend),
        "gpu_ids": list(result.gpu_ids),
        "parallel_times": bool(result.parallel_times),
        "processes": int(result.processes),
        "total_ref_rz_depth": (
            result.metadata.get("df_step_cost", {}).get("total_ref_rz_depth")
            if isinstance(result.metadata.get("df_step_cost"), Mapping)
            else None
        ),
    }


def run_cgs_window_sweep(args: argparse.Namespace) -> int:
    targets = _parse_targets(args.target)
    shifts = _parse_float_csv(args.shifts)
    scales = _parse_float_csv(args.window_scales)
    rows: list[dict[str, Any]] = []
    cache_document = load_df_cgs_json_cache(args.cache_path)
    for molecule_type, pf_label, ld in targets:
        hamiltonian, sector = build_df_h_d_from_molecule(int(molecule_type))
        ranked = rank_df_fragments(hamiltonian)
        partition = split_df_hamiltonian_by_ld(
            hamiltonian,
            int(ld),
            ranked_fragments=ranked,
        )
        base_t_values = default_perturbation_t_values(int(molecule_type), pf_label)
        for scale in scales:
            for shift in shifts:
                t_values = _window_values(base_t_values, shift=shift, scale=scale)
                label = f"scale={scale:g},shift={shift:g}"
                result = get_or_compute_cached_df_cgs_fit(
                    hamiltonian=hamiltonian,
                    sector=sector,
                    partition=partition,
                    pf_label=pf_label,
                    cache_document=cache_document,
                    cache_path=args.cache_path,
                    t_values=t_values,
                    evolution_backend=args.evolution_backend,
                    gpu_ids=tuple(_parse_csv(args.gpu_ids) or ("0",)),
                    chunk_splits=int(args.chunk_splits),
                    optimization_level=int(args.optimization_level),
                    matrix_free_backend=args.matrix_free_backend,
                    matrix_free_threads=args.matrix_free_threads,
                    matrix_free_block_chunk_size=args.matrix_free_block_chunk_size,
                    ground_state_ncv=args.ground_state_ncv,
                    ground_state_tol=float(args.ground_state_tol),
                    parallel_times=not args.no_parallel_times,
                    processes=args.processes,
                    use_parameterized_template=not args.no_parameterized_template,
                    use_ground_state_cache=not args.no_ground_state_cache,
                    debug=bool(args.debug),
                )
                rows.append(
                    _fit_result_row(
                        result,
                        molecule_type=int(molecule_type),
                        window_label=label,
                    )
                )
    payload = {
        "schema_version": 1,
        "diagnostic": "cgs_t_window_sweep",
        "targets": [
            {"molecule": f"H{m}", "molecule_type": m, "pf_label": pf, "ld": ld}
            for m, pf, ld in targets
        ],
        "shifts": list(shifts),
        "window_scales": list(scales),
        "rows": rows,
    }
    out_base = Path(args.output_dir) / "cgs_window_sweep"
    _write_rows(out_base, rows, payload)
    print(f"wrote {out_base.with_suffix('.json')}")
    print(f"wrote {out_base.with_suffix('.csv')}")
    return 0


def _active_blocks_for_ld(blocks: Sequence[Any], ld: int) -> list[Any]:
    has_one_body = bool(blocks and blocks[0].kind == "one_body_gaussian")
    stop = int(ld) + 1 if has_one_body else int(ld)
    return list(blocks[:stop])


def run_step_cost_sanity(args: argparse.Namespace) -> int:
    molecules = _parse_molecules(args.molecules, default_min=3, default_max=5)
    pf_labels = _parse_csv(args.pf_labels) or ("2nd", "4th(new_2)", "8th(Morales)")
    rows: list[dict[str, Any]] = []
    for molecule_type in molecules:
        hamiltonian, blocks = build_rank_ordered_df_cost_blocks(int(molecule_type))
        for pf_label in pf_labels:
            costs_by_ld = df_screening_costs_for_all_ld(
                hamiltonian=hamiltonian,
                blocks=blocks,
                pf_label=pf_label,
            )
            lds = _parse_ld_values(args.lds, rank=hamiltonian.n_blocks)
            if not lds:
                rank = int(hamiltonian.n_blocks)
                lds = tuple(dict.fromkeys([max(0, rank // 2), rank]))
            for ld in lds:
                if ld < 0 or ld > int(hamiltonian.n_blocks):
                    continue
                active_blocks = _active_blocks_for_ld(blocks, ld)
                proxy = costs_by_ld[int(ld)]
                analytic_d = _analytic_d_only_rz_cost(
                    active_blocks,
                    time=1.0,
                    num_qubits=hamiltonian.n_qubits,
                    pf_label=pf_label,
                )
                d_qc = _build_d_only_cost_circuit(
                    active_blocks,
                    time=1.0,
                    num_qubits=hamiltonian.n_qubits,
                    pf_label=pf_label,
                )
                circuit_d = _rz_depth_from_circuit(
                    d_qc,
                    basis_gates=_DF_COST_BASIS_GATES,
                    decompose_reps=int(args.decompose_reps),
                    optimization_level=int(args.optimization_level),
                )
                full_depth = None
                full_count = None
                full_transpiled_depth = None
                proxy_over_full = None
                if args.full_circuit:
                    full_qc = build_df_trotter_circuit(
                        active_blocks,
                        time=1.0,
                        num_qubits=hamiltonian.n_qubits,
                        pf_label=pf_label,
                    )
                    full_cost = _rz_depth_from_circuit(
                        full_qc,
                        basis_gates=_DF_COST_BASIS_GATES,
                        decompose_reps=int(args.decompose_reps),
                        optimization_level=int(args.optimization_level),
                    )
                    full_depth = int(full_cost["rz_depth"])
                    full_count = int(full_cost["rz_count"])
                    full_transpiled_depth = int(full_cost["transpiled_depth"])
                    proxy_over_full = (
                        None
                        if full_depth == 0
                        else float(proxy["total_ref_rz_depth"]) / float(full_depth)
                    )
                rows.append(
                    {
                        "molecule": f"H{int(molecule_type)}",
                        "molecule_type": int(molecule_type),
                        "pf_label": str(pf_label),
                        "order": pf_order(pf_label),
                        "ld": int(ld),
                        "df_rank_actual": int(hamiltonian.n_blocks),
                        "num_active_cost_blocks": len(active_blocks),
                        "u_ref_rz_depth_proxy": int(proxy["u_ref_rz_depth"]),
                        "d_ref_rz_depth_proxy": int(proxy["d_ref_rz_depth"]),
                        "total_ref_rz_depth_proxy": int(proxy["total_ref_rz_depth"]),
                        "u_ref_rz_count_proxy": int(proxy["u_ref_rz_count"]),
                        "d_ref_rz_count_proxy": int(proxy["d_ref_rz_count"]),
                        "total_ref_rz_count_proxy": int(proxy["total_ref_rz_count"]),
                        "analytic_d_rz_depth": int(analytic_d["rz_depth"]),
                        "analytic_d_rz_count": int(analytic_d["rz_count"]),
                        "circuit_d_rz_depth": int(circuit_d["rz_depth"]),
                        "circuit_d_rz_count": int(circuit_d["rz_count"]),
                        "d_depth_match": int(analytic_d["rz_depth"])
                        == int(circuit_d["rz_depth"]),
                        "d_count_match": int(analytic_d["rz_count"])
                        == int(circuit_d["rz_count"]),
                        "full_circuit_rz_depth": full_depth,
                        "full_circuit_rz_count": full_count,
                        "full_circuit_transpiled_depth": full_transpiled_depth,
                        "proxy_total_over_full_rz_depth": proxy_over_full,
                    }
                )
    payload = {
        "schema_version": 1,
        "diagnostic": "df_step_cost_sanity",
        "full_circuit_enabled": bool(args.full_circuit),
        "rows": rows,
    }
    out_base = Path(args.output_dir) / "step_cost_sanity"
    _write_rows(out_base, rows, payload)
    print(f"wrote {out_base.with_suffix('.json')}")
    print(f"wrote {out_base.with_suffix('.csv')}")
    return 0


def _add_common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostics for the DF partial-randomized PF result claims."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rand = subparsers.add_parser(
        "rand-sensitivity",
        help="Recompute existing refined costs under B0/g_rand scale factors.",
    )
    _add_common_output(rand)
    rand.add_argument("--anchor-refinement-dir", type=Path, default=DEFAULT_ANCHOR_REFINEMENT_DIR)
    rand.add_argument("--max-ld-dir", type=Path, default=DEFAULT_MAX_LD_DIR)
    rand.add_argument("--scales", default="0.1,1,10,100")
    rand.add_argument("--g-rand-base", type=float, default=PARTIAL_RANDOMIZED_DEFAULT_G_RAND)
    rand.add_argument("--epsilon-total", type=float, default=1e-4)
    rand.add_argument("--error-budget-rule", choices=("quadrature", "linear"), default="quadrature")
    rand.add_argument("--pf-labels", default=",".join(DEFAULT_PF_LABELS))
    rand.add_argument("--molecule-min", type=int, default=None)
    rand.add_argument("--molecule-max", type=int, default=None)
    rand.add_argument("--kappa-mode", choices=("fixed", "optimize"), default="optimize")
    rand.add_argument("--kappa-value", type=float, default=PARTIAL_RANDOMIZED_DEFAULT_KAPPA)
    rand.add_argument("--kappa-min", type=float, default=PARTIAL_RANDOMIZED_KAPPA_MIN)
    rand.add_argument("--kappa-max", type=float, default=PARTIAL_RANDOMIZED_KAPPA_MAX)
    rand.add_argument("--randomized-method", default=PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD)
    rand.add_argument(
        "--df-cost-model",
        choices=("qiskit_decomposed_rz_depth", "df_reference_rz_layers", "reference_rz_layers"),
        default="qiskit_decomposed_rz_depth",
    )
    rand.add_argument(
        "--reference-randomized-cost-mode",
        choices=("input", "mean_fragment", "tail_mean_fragment", "tail_total"),
        default="input",
    )
    rand.add_argument(
        "--cost-kind",
        choices=("actual_best", "actual_or_max_ld_cgs_optimized", "screening"),
        default="actual_best",
    )
    rand.set_defaults(func=run_rand_sensitivity)

    lam = subparsers.add_parser(
        "lambda-proxy",
        help="Compare DF lambda_R proxy rankings with abs(lambda) and optional Pauli L1.",
    )
    _add_common_output(lam)
    lam.add_argument("--molecules", default="H3-H5")
    lam.add_argument("--include-pauli-l1", action="store_true")
    lam.add_argument("--pauli-l1-max-molecule", type=int, default=5)
    lam.add_argument("--include-identity", action="store_true")
    lam.add_argument("--coeff-tol", type=float, default=1e-12)
    lam.set_defaults(func=run_lambda_proxy)

    fit = subparsers.add_parser(
        "cgs-fit-summary",
        help="Summarize stored Cgs fit slopes and fixed/free coefficient diagnostics.",
    )
    _add_common_output(fit)
    fit.add_argument("--anchor-refinement-dir", type=Path, default=DEFAULT_ANCHOR_REFINEMENT_DIR)
    fit.add_argument("--max-ld-dir", type=Path, default=DEFAULT_MAX_LD_DIR)
    fit.add_argument("--molecules", default=None)
    fit.add_argument("--pf-labels", default=None)
    fit.set_defaults(func=run_cgs_fit_summary)

    sweep = subparsers.add_parser(
        "cgs-window-sweep",
        help="Recompute selected Cgs fits under shifted/scaled t windows.",
    )
    _add_common_output(sweep)
    sweep.add_argument(
        "--target",
        action="append",
        required=True,
        help="Target formatted as Hn:pf_label:ld, e.g. H8:8th(Morales):8.",
    )
    sweep.add_argument("--shifts", default="-0.004,0,0.004")
    sweep.add_argument("--window-scales", default="1")
    sweep.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "cgs_window_sweep.cache.json",
    )
    sweep.add_argument("--evolution-backend", choices=("gpu", "cpu", "auto"), default="gpu")
    sweep.add_argument("--gpu-ids", default="0")
    sweep.add_argument("--chunk-splits", type=int, default=1)
    sweep.add_argument("--optimization-level", type=int, default=0)
    sweep.add_argument("--matrix-free-backend", choices=("auto", "numba", "tensor"), default="auto")
    sweep.add_argument("--matrix-free-threads", type=int, default=None)
    sweep.add_argument("--matrix-free-block-chunk-size", type=int, default=None)
    sweep.add_argument("--ground-state-ncv", type=int, default=None)
    sweep.add_argument("--ground-state-tol", type=float, default=1e-10)
    sweep.add_argument("--processes", type=int, default=None)
    sweep.add_argument("--no-parallel-times", action="store_true")
    sweep.add_argument("--no-parameterized-template", action="store_true")
    sweep.add_argument("--no-ground-state-cache", action="store_true")
    sweep.add_argument("--debug", action="store_true")
    sweep.set_defaults(func=run_cgs_window_sweep)

    step = subparsers.add_parser(
        "step-cost-sanity",
        help="Check analytic D-only depth and optionally compare proxy depth with full circuits.",
    )
    _add_common_output(step)
    step.add_argument("--molecules", default="H3-H5")
    step.add_argument("--pf-labels", default="2nd,4th(new_2),8th(Morales)")
    step.add_argument("--lds", default=None, help="Comma/range list. Use max for DF rank.")
    step.add_argument("--full-circuit", action="store_true")
    step.add_argument("--decompose-reps", type=int, default=8)
    step.add_argument("--optimization-level", type=int, default=0)
    step.set_defaults(func=run_step_cost_sanity)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

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
    pf_order,
)
from trotterlib.df_hamiltonian import build_df_h_d_from_molecule  # noqa: E402
from trotterlib.df_partial_randomized_pf import (  # noqa: E402
    get_or_compute_cached_df_cgs_fit,
    load_df_cgs_json_cache,
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)
from trotterlib.df_screening_cost import (  # noqa: E402
    df_screening_costs_for_all_ld,
    build_rank_ordered_df_cost_blocks,
)
from trotterlib.partial_randomized_pf import (  # noqa: E402
    optimize_error_budget_and_kappa,
    randomized_prefactor_b0,
)


ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "partial_randomized_pf"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "df_cgs" / "anchor_refinement"


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_molecules(raw: str | None, *, molecule_min: int, molecule_max: int) -> tuple[int, ...]:
    if raw:
        values: list[int] = []
        for item in _parse_csv(raw):
            if "-" in item:
                lo_raw, hi_raw = item.split("-", 1)
                lo = int(lo_raw.removeprefix("H"))
                hi = int(hi_raw.removeprefix("H"))
                step = 1 if hi >= lo else -1
                values.extend(range(lo, hi + step, step))
            else:
                values.append(int(item.removeprefix("H")))
        return tuple(dict.fromkeys(values))
    return tuple(range(int(molecule_min), int(molecule_max) + 1))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return [dict(item) for item in data["entries"] if isinstance(item, dict)]
    return []


def _result_row(result: Any, *, source_kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    df_step_cost = result.metadata.get("df_step_cost", {})
    ground_state_cache = result.metadata.get("ground_state_cache", {})
    row: dict[str, Any] = {
        "molecule_type": int(result.metadata.get("molecule_type", 0)),
        "df_rank_actual": int(result.df_rank_actual),
        "df_rank_requested": result.df_rank_requested,
        "df_tol_requested": result.df_tol_requested,
        "ld": int(result.ld),
        "lambda_r": float(result.lambda_r),
        "pf_label": str(result.pf_label),
        "order": int(result.order),
        "c_gs_d": float(result.coeff),
        "fit_slope": result.fit_slope,
        "fit_coeff": result.fit_coeff,
        "fixed_order_coeff": result.fit_coeff_fixed_order,
        "t_values": list(result.t_values),
        "perturbation_errors": list(result.perturbation_errors),
        "gpu_ids": list(result.gpu_ids),
        "parallel_times": bool(result.parallel_times),
        "processes": int(result.processes),
        "use_parameterized_template": result.metadata.get("use_parameterized_template"),
        "ground_state_cache": ground_state_cache,
        "df_step_cost": df_step_cost,
        "total_ref_rz_depth": (
            df_step_cost.get("total_ref_rz_depth")
            if isinstance(df_step_cost, dict)
            else None
        ),
        "total_ref_rz_count": (
            df_step_cost.get("total_ref_rz_count")
            if isinstance(df_step_cost, dict)
            else None
        ),
        "template_transpile_s": (
            result.metadata.get("parameterized_template_profile") or {}
        ).get("transpile_s"),
        "source_kind": source_kind,
    }
    if extra:
        row.update(extra)
    return row


def _patch_molecule_type(row: dict[str, Any], molecule_type: int) -> dict[str, Any]:
    row["molecule"] = f"H{int(molecule_type)}"
    row["molecule_type"] = int(molecule_type)
    return row


def _candidate_for_ld(
    *,
    molecule_type: int,
    pf_label: str,
    ld: int,
    hamiltonian: Any,
    ranked: Sequence[Any],
    anchor_row: dict[str, Any],
    step_costs: dict[str, int],
    epsilon_total: float,
    kappa_mode: str,
    kappa_value: float,
    kappa_min: float,
    kappa_max: float,
    randomized_method: str,
    g_rand: float,
    error_budget_rule: str,
) -> dict[str, Any]:
    partition = split_df_hamiltonian_by_ld(
        hamiltonian,
        int(ld),
        ranked_fragments=ranked,
    )
    budget = optimize_error_budget_and_kappa(
        epsilon_total=float(epsilon_total),
        order=pf_order(pf_label),
        deterministic_step_cost_value=int(step_costs["total_ref_rz_depth"]),
        c_gs=float(anchor_row["c_gs_d"]),
        lambda_r=float(partition.lambda_r),
        randomized_method=randomized_method,
        g_rand=float(g_rand),
        kappa_mode=kappa_mode,
        kappa_value=float(kappa_value),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),
        error_budget_rule=error_budget_rule,
    )
    return {
        "molecule": f"H{int(molecule_type)}",
        "molecule_type": int(molecule_type),
        "df_rank_actual": int(hamiltonian.n_blocks),
        "pf_label": str(pf_label),
        "order": pf_order(pf_label),
        "ld": int(ld),
        "ld_anchor": int(anchor_row["ld_anchor"]),
        "cgs_source_kind": str(anchor_row.get("source_kind", "screening_anchor")),
        "c_gs_d_screen": float(anchor_row["c_gs_d"]),
        "lambda_r": float(partition.lambda_r),
        **step_costs,
        "q_opt": budget.q_ratio,
        "eps_qpe_opt": budget.eps_qpe,
        "eps_trot_opt": budget.eps_trot,
        "kappa_opt": budget.kappa,
        "b_opt": budget.b_value,
        "boundary_hit_q": budget.boundary_hit_q,
        "boundary_hit_kappa": budget.boundary_hit_kappa,
        "error_budget_rule": str(error_budget_rule),
        "g_det": budget.g_det,
        "g_rand": budget.g_rand,
        "g_total": budget.g_total,
    }


def _anchor_screening_candidates(
    *,
    molecule_type: int,
    pf_label: str,
    anchor_row: dict[str, Any],
    epsilon_total: float,
    kappa_mode: str,
    kappa_value: float,
    kappa_min: float,
    kappa_max: float,
    randomized_method: str,
    g_rand: float,
    error_budget_rule: str,
) -> list[dict[str, Any]]:
    cost_hamiltonian, blocks = build_rank_ordered_df_cost_blocks(int(molecule_type))
    ranked = rank_df_fragments(cost_hamiltonian)
    costs_by_ld = df_screening_costs_for_all_ld(
        hamiltonian=cost_hamiltonian,
        blocks=blocks,
        pf_label=str(pf_label),
    )
    return [
        _candidate_for_ld(
            molecule_type=molecule_type,
            pf_label=pf_label,
            ld=ld,
            hamiltonian=cost_hamiltonian,
            ranked=ranked,
            anchor_row=anchor_row,
            step_costs=costs_by_ld[ld],
            epsilon_total=epsilon_total,
            kappa_mode=kappa_mode,
            kappa_value=kappa_value,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            randomized_method=randomized_method,
            g_rand=g_rand,
            error_budget_rule=error_budget_rule,
        )
        for ld in range(0, int(cost_hamiltonian.n_blocks) + 1)
    ]


def _actual_cost_for_row(
    row: dict[str, Any],
    *,
    epsilon_total: float,
    kappa_mode: str,
    kappa_value: float,
    kappa_min: float,
    kappa_max: float,
    randomized_method: str,
    g_rand: float,
    error_budget_rule: str,
) -> dict[str, Any]:
    budget = optimize_error_budget_and_kappa(
        epsilon_total=float(epsilon_total),
        order=int(row["order"]),
        deterministic_step_cost_value=int(row["total_ref_rz_depth"]),
        c_gs=float(row["c_gs_d"]),
        lambda_r=float(row["lambda_r"]),
        randomized_method=randomized_method,
        g_rand=float(g_rand),
        kappa_mode=kappa_mode,
        kappa_value=float(kappa_value),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),
        error_budget_rule=error_budget_rule,
    )
    return {
        "molecule": f"H{int(row['molecule_type'])}",
        "molecule_type": int(row["molecule_type"]),
        "pf_label": str(row["pf_label"]),
        "ld": int(row["ld"]),
        "order": int(row["order"]),
        "lambda_r": float(row["lambda_r"]),
        "c_gs_d": float(row["c_gs_d"]),
        "total_ref_rz_depth": int(row["total_ref_rz_depth"]),
        "q_opt": budget.q_ratio,
        "eps_qpe_opt": budget.eps_qpe,
        "eps_trot_opt": budget.eps_trot,
        "kappa_opt": budget.kappa,
        "b_opt": budget.b_value,
        "g_det": budget.g_det,
        "g_rand": budget.g_rand,
        "g_total": budget.g_total,
        "boundary_hit_q": budget.boundary_hit_q,
        "boundary_hit_kappa": budget.boundary_hit_kappa,
        "error_budget_rule": str(error_budget_rule),
        "source_kind": str(row.get("source_kind", "unknown")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run anchor Cgs, anchor-based LD screening, and LD-window Cgs "
            "refinement for DF H-chain systems."
        )
    )
    parser.add_argument("--molecules", type=str)
    parser.add_argument("--molecule-min", type=int, default=3)
    parser.add_argument("--molecule-max", type=int, default=14)
    parser.add_argument(
        "--pf-labels",
        default="2nd,4th,4th(new_2),8th(Morales)",
        help="Comma-separated PF labels.",
    )
    parser.add_argument("--ld-window", type=int, default=1)
    parser.add_argument("--epsilon-total", type=float, default=1e-4)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--optimization-level", type=int, default=0)
    parser.add_argument("--ground-state-tol", type=float, default=1e-10)
    parser.add_argument("--matrix-free-threads", type=int, default=None)
    parser.add_argument(
        "--no-parameterized-template",
        action="store_true",
        help="Build one concrete circuit per t instead of a parameterized template.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="H3_H14_df_anchor_refinement")
    parser.add_argument("--cache-path", type=Path)
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
        choices=("quadrature", "linear"),
        default="quadrature",
        help="Total-error composition used when selecting and comparing LD candidates.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    molecules = _parse_molecules(
        args.molecules,
        molecule_min=int(args.molecule_min),
        molecule_max=int(args.molecule_max),
    )
    pf_labels = _parse_csv(args.pf_labels)
    gpu_ids = _parse_csv(args.gpu_ids)
    cache_path = (
        args.cache_path
        if args.cache_path is not None
        else args.output_dir / f"{args.output_stem}.cgs_cache.json"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = args.output_dir / f"{args.output_stem}.anchor.partial.json"
    refinement_path = args.output_dir / f"{args.output_stem}.refinement.partial.json"
    summary_path = args.output_dir / f"{args.output_stem}.summary.partial.json"
    final_anchor_path = args.output_dir / f"{args.output_stem}.anchor.json"
    final_refinement_path = args.output_dir / f"{args.output_stem}.refinement.json"
    final_summary_path = args.output_dir / f"{args.output_stem}.summary.json"

    anchor_rows = _load_rows(anchor_path)
    refinement_rows = _load_rows(refinement_path)
    summary_rows = _load_rows(summary_path)
    anchor_by_key = {
        (int(row["molecule_type"]), str(row["pf_label"])): dict(row)
        for row in anchor_rows
        if "molecule_type" in row and "pf_label" in row
    }
    refinement_by_key = {
        (int(row["molecule_type"]), str(row["pf_label"]), int(row["ld"])): dict(row)
        for row in refinement_rows
        if "molecule_type" in row and "pf_label" in row and "ld" in row
    }
    summary_done = {
        (int(row["molecule_type"]), str(row["pf_label"]))
        for row in summary_rows
        if "molecule_type" in row and "pf_label" in row
    }

    print(
        json.dumps(
            {
                "event": "pipeline_start",
                "molecules": list(molecules),
                "pf_labels": list(pf_labels),
                "gpu_ids": list(gpu_ids),
                "ld_window": int(args.ld_window),
                "epsilon_total": float(args.epsilon_total),
                "error_budget_rule": str(args.error_budget_rule),
                "cache_path": str(cache_path),
                "use_parameterized_template": not bool(args.no_parameterized_template),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    cache_document = load_df_cgs_json_cache(cache_path)
    b0 = randomized_prefactor_b0(args.randomized_method, float(args.g_rand))

    for molecule_type in molecules:
        hamiltonian, sector = build_df_h_d_from_molecule(int(molecule_type))
        ranked = rank_df_fragments(hamiltonian)
        ld_anchor = int(hamiltonian.n_blocks) // 2
        print(
            json.dumps(
                {
                    "event": "molecule_loaded",
                    "molecule_type": int(molecule_type),
                    "df_rank_actual": int(hamiltonian.n_blocks),
                    "ld_anchor": ld_anchor,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        for pf_label in pf_labels:
            anchor_key = (int(molecule_type), str(pf_label))
            if anchor_key in anchor_by_key:
                anchor_row = anchor_by_key[anchor_key]
                print(
                    json.dumps(
                        {
                            "event": "anchor_skip_done",
                            "molecule_type": int(molecule_type),
                            "pf_label": str(pf_label),
                            "ld_anchor": int(anchor_row["ld"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                partition = split_df_hamiltonian_by_ld(
                    hamiltonian,
                    ld_anchor,
                    ranked_fragments=ranked,
                )
                print(
                    json.dumps(
                        {
                            "event": "anchor_start",
                            "molecule_type": int(molecule_type),
                            "pf_label": str(pf_label),
                            "ld_anchor": ld_anchor,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                result = get_or_compute_cached_df_cgs_fit(
                    hamiltonian=hamiltonian,
                    sector=sector,
                    partition=partition,
                    pf_label=str(pf_label),
                    cache_document=cache_document,
                    cache_path=cache_path,
                    evolution_backend="gpu",
                    gpu_ids=gpu_ids,
                    optimization_level=int(args.optimization_level),
                    parallel_times=True,
                    processes=args.processes,
                    use_parameterized_template=not bool(args.no_parameterized_template),
                    use_ground_state_cache=True,
                    ground_state_tol=float(args.ground_state_tol),
                    matrix_free_threads=args.matrix_free_threads,
                    debug=bool(args.debug),
                )
                anchor_row = _patch_molecule_type(
                    _result_row(
                        result,
                        source_kind="screening_anchor",
                        extra={
                            "ld_anchor": ld_anchor,
                            "is_screening_anchor": True,
                        },
                    ),
                    int(molecule_type),
                )
                anchor_rows.append(anchor_row)
                anchor_by_key[anchor_key] = anchor_row
                _write_json(anchor_path, anchor_rows)
                print(json.dumps({"event": "anchor_done", **anchor_row}, sort_keys=True), flush=True)

            candidates = _anchor_screening_candidates(
                molecule_type=int(molecule_type),
                pf_label=str(pf_label),
                anchor_row=anchor_row,
                epsilon_total=float(args.epsilon_total),
                kappa_mode=str(args.kappa_mode),
                kappa_value=float(args.kappa_value),
                kappa_min=float(args.kappa_min),
                kappa_max=float(args.kappa_max),
                randomized_method=str(args.randomized_method),
                g_rand=float(args.g_rand),
                error_budget_rule=str(args.error_budget_rule),
            )
            finite_candidates = [
                candidate
                for candidate in candidates
                if math.isfinite(float(candidate.get("g_total", math.inf)))
            ]
            best = min(finite_candidates, key=lambda item: float(item["g_total"]))
            lo = max(0, int(best["ld"]) - int(args.ld_window))
            hi = min(int(hamiltonian.n_blocks), int(best["ld"]) + int(args.ld_window))
            target_lds = tuple(range(lo, hi + 1))
            print(
                json.dumps(
                    {
                        "event": "screening_best",
                        "molecule_type": int(molecule_type),
                        "pf_label": str(pf_label),
                        "ld_screening_best": int(best["ld"]),
                        "g_total_screening": float(best["g_total"]),
                        "kappa_opt_screening": float(best["kappa_opt"]),
                        "error_budget_rule": str(args.error_budget_rule),
                        "target_lds": list(target_lds),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            exact_rows_by_ld: dict[int, dict[str, Any]] = {}
            if int(anchor_row["ld"]) in target_lds:
                exact_rows_by_ld[int(anchor_row["ld"])] = anchor_row

            for ld in target_lds:
                refine_key = (int(molecule_type), str(pf_label), int(ld))
                if int(ld) == int(anchor_row["ld"]):
                    continue
                if refine_key in refinement_by_key:
                    exact_rows_by_ld[int(ld)] = refinement_by_key[refine_key]
                    print(
                        json.dumps(
                            {
                                "event": "refinement_skip_done",
                                "molecule_type": int(molecule_type),
                                "pf_label": str(pf_label),
                                "ld": int(ld),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue

                partition = split_df_hamiltonian_by_ld(
                    hamiltonian,
                    int(ld),
                    ranked_fragments=ranked,
                )
                print(
                    json.dumps(
                        {
                            "event": "refinement_start",
                            "molecule_type": int(molecule_type),
                            "pf_label": str(pf_label),
                            "ld": int(ld),
                            "ld_screening_best": int(best["ld"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                result = get_or_compute_cached_df_cgs_fit(
                    hamiltonian=hamiltonian,
                    sector=sector,
                    partition=partition,
                    pf_label=str(pf_label),
                    cache_document=cache_document,
                    cache_path=cache_path,
                    evolution_backend="gpu",
                    gpu_ids=gpu_ids,
                    optimization_level=int(args.optimization_level),
                    parallel_times=True,
                    processes=args.processes,
                    use_parameterized_template=not bool(args.no_parameterized_template),
                    use_ground_state_cache=True,
                    ground_state_tol=float(args.ground_state_tol),
                    matrix_free_threads=args.matrix_free_threads,
                    debug=bool(args.debug),
                )
                row = _patch_molecule_type(
                    _result_row(
                        result,
                        source_kind="explicit_ld_refinement",
                        extra={
                            "ld_screening_best": int(best["ld"]),
                            "ld_window": int(args.ld_window),
                            "screening_g_total": float(best["g_total"]),
                            "screening_q_opt": best.get("q_opt"),
                            "screening_kappa_opt": best.get("kappa_opt"),
                        },
                    ),
                    int(molecule_type),
                )
                refinement_rows.append(row)
                refinement_by_key[refine_key] = row
                exact_rows_by_ld[int(ld)] = row
                _write_json(refinement_path, refinement_rows)
                print(json.dumps({"event": "refinement_done", **row}, sort_keys=True), flush=True)

            for ld in target_lds:
                refine_key = (int(molecule_type), str(pf_label), int(ld))
                if refine_key in refinement_by_key:
                    exact_rows_by_ld[int(ld)] = refinement_by_key[refine_key]

            actual_costs = [
                _actual_cost_for_row(
                    exact_rows_by_ld[int(ld)],
                    epsilon_total=float(args.epsilon_total),
                    kappa_mode=str(args.kappa_mode),
                    kappa_value=float(args.kappa_value),
                    kappa_min=float(args.kappa_min),
                    kappa_max=float(args.kappa_max),
                    randomized_method=str(args.randomized_method),
                    g_rand=float(args.g_rand),
                    error_budget_rule=str(args.error_budget_rule),
                )
                for ld in target_lds
                if int(ld) in exact_rows_by_ld
            ]
            actual_best = (
                min(actual_costs, key=lambda item: float(item["g_total"]))
                if actual_costs
                else None
            )
            summary_key = (int(molecule_type), str(pf_label))
            summary_row = {
                "molecule": f"H{int(molecule_type)}",
                "molecule_type": int(molecule_type),
                "pf_label": str(pf_label),
                "order": pf_order(str(pf_label)),
                "epsilon_total": float(args.epsilon_total),
                "randomized_method": str(args.randomized_method),
                "g_rand_input": float(args.g_rand),
                "error_budget_rule": str(args.error_budget_rule),
                "b0": float(b0),
                "kappa_mode": str(args.kappa_mode),
                "kappa_min": float(args.kappa_min),
                "kappa_max": float(args.kappa_max),
                "df_rank_actual": int(hamiltonian.n_blocks),
                "ld_anchor": int(anchor_row["ld"]),
                "anchor_c_gs_d": float(anchor_row["c_gs_d"]),
                "anchor_fit_slope": anchor_row.get("fit_slope"),
                "screening_best": best,
                "screening_candidates": candidates,
                "target_lds": list(target_lds),
                "actual_costs": actual_costs,
                "actual_best": actual_best,
                "missing_actual_lds": [
                    int(ld) for ld in target_lds if int(ld) not in exact_rows_by_ld
                ],
            }
            if summary_key in summary_done:
                summary_rows = [
                    summary_row
                    if (
                        int(row.get("molecule_type", -1)) == summary_key[0]
                        and str(row.get("pf_label")) == summary_key[1]
                    )
                    else row
                    for row in summary_rows
                ]
            else:
                summary_rows.append(summary_row)
                summary_done.add(summary_key)
            _write_json(summary_path, summary_rows)
            print(
                json.dumps(
                    {
                        "event": "comparison_done",
                        "molecule_type": int(molecule_type),
                        "pf_label": str(pf_label),
                        "ld_screening_best": int(best["ld"]),
                        "ld_actual_best": (
                            None if actual_best is None else int(actual_best["ld"])
                        ),
                        "g_total_actual_best": (
                            None if actual_best is None else float(actual_best["g_total"])
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    _write_json(final_anchor_path, anchor_rows)
    _write_json(final_refinement_path, refinement_rows)
    _write_json(final_summary_path, summary_rows)
    print(
        json.dumps(
            {
                "event": "pipeline_done",
                "anchor_path": str(final_anchor_path),
                "refinement_path": str(final_refinement_path),
                "summary_path": str(final_summary_path),
                "num_anchor_rows": len(anchor_rows),
                "num_refinement_rows": len(refinement_rows),
                "num_summary_rows": len(summary_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

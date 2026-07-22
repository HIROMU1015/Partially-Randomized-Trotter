#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trotterlib.pauli_partial_cgs_validation import (
    dense_ranked_pauli_terms,
    fit_partial_s2_cgs,
    full_h_target_state,
    optimized_cost_for_cgs,
)
from trotterlib.partial_randomized_pf import build_sorted_pauli_hamiltonian


DEFAULT_DELTAS = (0.05, 0.075, 0.1, 0.15, 0.2)
DEFAULT_OUTPUT = Path(
    "artifacts/partial_randomized_pf/diagnostics/"
    "H2_H3_s2_full_h_partial_cgs_all_prefixes.json"
)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("at least one molecule is required")
    return values


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("delta values must be positive")
    return values


def _parse_csv_methods(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not values or any(value not in {"qdrift", "rte"} for value in values):
        raise ValueError("randomized methods must be qdrift and/or rte")
    return values


def _best(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: float(row["g_total"]))


def _positive_span(
    fits: Sequence[dict[str, Any]],
    metric: str,
    *,
    floor: float,
    slope_metric: str | None = None,
    slope_tolerance: float | None = None,
    overlap_metric: str | None = None,
    min_overlap: float | None = None,
    required_cluster_rank: int | None = None,
    min_cluster_principal_overlap: float | None = None,
) -> dict[str, Any]:
    def slope_valid(row: dict[str, Any]) -> bool:
        if slope_metric is None or slope_tolerance is None:
            return True
        slope = row.get(slope_metric)
        return slope is not None and abs(float(slope) - 2.0) <= float(slope_tolerance)

    def overlap_valid(row: dict[str, Any]) -> bool:
        if overlap_metric is None or min_overlap is None:
            return True
        overlap = row.get(overlap_metric)
        return overlap is not None and float(overlap) >= float(min_overlap)

    def cluster_valid(row: dict[str, Any]) -> bool:
        if required_cluster_rank is not None and not all(
            int(rank) == int(required_cluster_rank)
            for rank in row.get("target_cluster_ranks", ())
        ):
            return False
        if min_cluster_principal_overlap is not None and float(
            row.get("min_target_cluster_principal_overlap", 0.0)
        ) < float(min_cluster_principal_overlap):
            return False
        return True

    values = [
        (int(row["ld"]), float(row[metric]))
        for row in fits
        if (
            int(row["ld"]) > 0
            and float(row[metric]) > float(floor)
            and slope_valid(row)
            and overlap_valid(row)
            and cluster_valid(row)
        )
    ]
    unresolved = [
        int(row["ld"])
        for row in fits
        if int(row["ld"]) > 0
        and (
            float(row[metric]) <= float(floor)
            or not slope_valid(row)
            or not overlap_valid(row)
            or not cluster_valid(row)
        )
    ]
    if not values:
        return {
            "defined": False,
            "reason": "no nonzero prefix passed the positivity and fit-validity filters",
            "num_positive": 0,
            "num_unresolved_nonzero_ld": len(unresolved),
            "unresolved_nonzero_lds": unresolved,
            "slope_tolerance": slope_tolerance,
            "min_overlap": min_overlap,
            "required_cluster_rank": required_cluster_rank,
            "min_cluster_principal_overlap": min_cluster_principal_overlap,
        }
    min_ld, c_min = min(values, key=lambda item: item[1])
    max_ld, c_max = max(values, key=lambda item: item[1])
    return {
        "defined": True,
        "reporting_floor": float(floor),
        "slope_tolerance": slope_tolerance,
        "min_overlap": min_overlap,
        "required_cluster_rank": required_cluster_rank,
        "min_cluster_principal_overlap": min_cluster_principal_overlap,
        "num_positive": len(values),
        "num_unresolved_nonzero_ld": len(unresolved),
        "unresolved_nonzero_lds": unresolved,
        "c_min": c_min,
        "c_min_ld": min_ld,
        "c_max": c_max,
        "c_max_ld": max_ld,
        "c_max_over_c_min": c_max / c_min,
        "sqrt_c_max_over_c_min": math.sqrt(c_max / c_min),
    }


def _cost_comparison(
    fits: Sequence[dict[str, Any]],
    *,
    metric: str,
    anchor_ld: int,
    epsilon_total: float,
    total_terms: int,
    step_cost_rule: str,
    randomized_method: str,
    g_rand: float,
    reporting_floor: float,
    slope_metric: str,
    max_delta: float,
    required_cluster_rank: int | None = None,
    min_cluster_projector_overlap: float | None = None,
    min_cluster_state_weight: float | None = None,
    slope_tolerance: float = 0.2,
    applicable: bool = True,
    inapplicable_reason: str | None = None,
) -> dict[str, Any]:
    if not applicable:
        return {
            "metric": metric,
            "step_cost_rule": step_cost_rule,
            "anchor_ld": int(anchor_ld),
            "epsilon_total": float(epsilon_total),
            "randomized_method": randomized_method,
            "max_delta": float(max_delta),
            "applicable": False,
            "reason": inapplicable_reason,
        }

    def is_second_order_valid(c_gs: float, slope: object) -> bool:
        return (
            c_gs > float(reporting_floor)
            and slope is not None
            and abs(float(slope) - 2.0) <= float(slope_tolerance)
        )

    def is_fit_valid(fit: dict[str, Any]) -> bool:
        if not is_second_order_valid(float(fit[metric]), fit.get(slope_metric)):
            return False
        if required_cluster_rank is not None and not all(
            int(rank) == int(required_cluster_rank)
            for rank in fit.get("target_cluster_ranks", ())
        ):
            return False
        if (
            min_cluster_projector_overlap is not None
            and float(fit.get("min_target_cluster_principal_overlap", 0.0))
            < float(min_cluster_projector_overlap)
        ):
            return False
        if (
            min_cluster_state_weight is not None
            and float(fit.get("min_target_cluster_ground_state_weight", 0.0))
            < float(min_cluster_state_weight)
        ):
            return False
        return True

    anchor_cgs = float(fits[int(anchor_ld)][metric])
    anchor_slope = fits[int(anchor_ld)].get(slope_metric)
    if not is_fit_valid(fits[int(anchor_ld)]):
        return {
            "metric": metric,
            "step_cost_rule": step_cost_rule,
            "anchor_ld": int(anchor_ld),
            "epsilon_total": float(epsilon_total),
            "randomized_method": randomized_method,
            "max_delta": float(max_delta),
            "applicable": False,
            "reason": (
                "anchor Cgs does not have a resolved second-order fit above "
                "the reporting floor"
            ),
            "anchor_c_gs": anchor_cgs,
            "anchor_fit_slope": anchor_slope,
        }

    exact_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    for fit in fits:
        common = {
            "ld": int(fit["ld"]),
            "total_terms": int(total_terms),
            "lambda_r": float(fit["lambda_r"]),
            "epsilon_total": float(epsilon_total),
            "step_cost_rule": step_cost_rule,
            "randomized_method": randomized_method,
            "g_rand": float(g_rand),
            "max_delta": float(max_delta),
        }
        exact_row = optimized_cost_for_cgs(c_gs=float(fit[metric]), **common)
        exact_row["known_exact_zero"] = int(fit["ld"]) == 0
        exact_row["second_order_fit_valid"] = is_second_order_valid(
            float(fit[metric]), fit.get(slope_metric)
        )
        exact_row["metric_fit_valid"] = is_fit_valid(fit)
        exact_rows.append(exact_row)
        anchor_rows.append(optimized_cost_for_cgs(c_gs=anchor_cgs, **common))

    valid_exact_rows = [
        row
        for row in exact_rows
        if bool(row["known_exact_zero"]) or bool(row["metric_fit_valid"])
    ]
    if not valid_exact_rows:
        return {
            "metric": metric,
            "step_cost_rule": step_cost_rule,
            "anchor_ld": int(anchor_ld),
            "epsilon_total": float(epsilon_total),
            "randomized_method": randomized_method,
            "max_delta": float(max_delta),
            "applicable": False,
            "reason": "no exact-model prefix has a usable fit",
            "anchor_c_gs": anchor_cgs,
            "anchor_fit_slope": anchor_slope,
            "exact_costs": exact_rows,
            "anchor_costs": anchor_rows,
        }

    exact_best = _best(valid_exact_rows)
    anchor_best = _best(anchor_rows)
    exact_at_anchor_choice = exact_rows[int(anchor_best["ld"])]
    anchor_total = float(anchor_best["g_total"])
    exact_total = float(exact_best["g_total"])
    exact_at_anchor_total = float(exact_at_anchor_choice["g_total"])
    exact_best_slope = fits[int(exact_best["ld"])].get(slope_metric)
    anchor_choice_exact_fit_valid = bool(
        exact_at_anchor_choice["known_exact_zero"]
        or exact_at_anchor_choice["metric_fit_valid"]
    )
    pointwise_relative_differences = []
    for exact, anchor in zip(exact_rows, anchor_rows):
        denominator = float(anchor["g_total"])
        if denominator > 0.0:
            pointwise_relative_differences.append(
                (float(exact["g_total"]) - denominator) / denominator
            )
    return {
        "metric": metric,
        "step_cost_rule": step_cost_rule,
        "epsilon_total": float(epsilon_total),
        "randomized_method": randomized_method,
        "max_delta": float(max_delta),
        "applicable": True,
        "anchor_ld": int(anchor_ld),
        "anchor_c_gs": anchor_cgs,
        "anchor_fit_slope": anchor_slope,
        "anchor_second_order_fit_valid": is_second_order_valid(
            anchor_cgs, anchor_slope
        ),
        "anchor_metric_fit_valid": is_fit_valid(fits[int(anchor_ld)]),
        "exact_best": exact_best,
        "exact_best_fit_slope": exact_best_slope,
        "exact_best_second_order_fit_valid": is_second_order_valid(
            float(exact_best["c_gs"]), exact_best_slope
        ),
        "exact_best_metric_fit_valid": bool(
            exact_best["known_exact_zero"] or exact_best["metric_fit_valid"]
        ),
        "anchor_best": anchor_best,
        "exact_cost_at_anchor_chosen_ld": exact_at_anchor_choice,
        "anchor_choice_exact_fit_valid": anchor_choice_exact_fit_valid,
        "optimal_ld_changed": int(exact_best["ld"]) != int(anchor_best["ld"]),
        "anchor_decision_regret": (
            (exact_at_anchor_total - exact_total) / exact_total
            if exact_total > 0.0 and anchor_choice_exact_fit_valid
            else None
        ),
        "anchor_model_value_gap": (
            anchor_total / exact_total - 1.0 if exact_total > 0.0 else None
        ),
        "anchor_calibration_gap_at_chosen_ld": (
            anchor_total / exact_at_anchor_total - 1.0
            if exact_at_anchor_total > 0.0 and anchor_choice_exact_fit_valid
            else None
        ),
        "exact_minus_anchor_best_total": exact_total - anchor_total,
        "exact_over_anchor_best_total": (
            exact_total / anchor_total if anchor_total > 0.0 else None
        ),
        "relative_best_total_difference": (
            (exact_total - anchor_total) / anchor_total if anchor_total > 0.0 else None
        ),
        "max_abs_pointwise_relative_cost_difference": (
            max(abs(value) for value in pointwise_relative_differences)
            if pointwise_relative_differences
            else None
        ),
        "exact_costs": exact_rows,
        "anchor_costs": anchor_rows,
    }


def _molecule_result(
    molecule_type: int,
    *,
    delta_values: Sequence[float],
    epsilon_values: Sequence[float],
    noise_floor: float,
    cgs_reporting_floor: float,
    randomized_methods: Sequence[str],
    g_rand: float,
    branch_overlap_threshold: float,
    paper_ground_state_weight_threshold: float,
) -> dict[str, Any]:
    sorted_hamiltonian = build_sorted_pauli_hamiltonian(int(molecule_type))
    target = full_h_target_state(sorted_hamiltonian)
    dense_terms = dense_ranked_pauli_terms(sorted_hamiltonian)
    fits = []
    for ld in range(sorted_hamiltonian.num_terms + 1):
        fit = fit_partial_s2_cgs(
            sorted_hamiltonian,
            dense_terms,
            target,
            ld=ld,
            delta_values=delta_values,
            noise_floor=float(noise_floor),
        )
        fits.append(fit.to_dict())
        print(
            json.dumps(
                {
                    "event": "prefix_done",
                    "molecule_type": int(molecule_type),
                    "ld": ld,
                    "total_terms": sorted_hamiltonian.num_terms,
                    "c_gs_signal": fit.c_gs_signal,
                    "signal_fit_slope": fit.signal_fit_slope,
                    "c_gs_qpe_rmse": fit.c_gs_qpe_rmse,
                    "qpe_rmse_fit_slope": fit.qpe_rmse_fit_slope,
                    "c_gs_ground_space_worst_rmse": fit.c_gs_ground_space_worst_rmse,
                    "ground_space_worst_rmse_fit_slope": fit.ground_space_worst_rmse_fit_slope,
                    "c_gs_target_cluster_ground": fit.c_gs_target_cluster_ground,
                    "target_cluster_ground_fit_slope": fit.target_cluster_ground_fit_slope,
                    "min_target_cluster_ground_state_weight": fit.min_target_cluster_ground_state_weight,
                    "min_target_cluster_principal_overlap": fit.min_target_cluster_principal_overlap,
                    "min_phase_branch_cut_clearance": fit.min_phase_branch_cut_clearance,
                    "c_gs_branch": fit.c_gs_branch,
                    "branch_fit_slope": fit.branch_fit_slope,
                    "min_branch_overlap": fit.min_branch_overlap,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    anchor_ld = sorted_hamiltonian.num_terms // 2
    comparisons = []
    branch_comparison_applicable = (
        target.full_space_target_energy_multiplicity == 1
    )
    branch_inapplicable_reason = (
        None
        if branch_comparison_applicable
        else (
            "the target energy is degenerate in the full Hilbert space, so the "
            "physical-sector target does not continue to a unique S2 eigenphase branch"
        )
    )
    anchor_fit = fits[anchor_ld]
    paper_comparison_applicable = (
        float(anchor_fit["c_gs_target_cluster_ground"])
        > float(cgs_reporting_floor)
        and anchor_fit["target_cluster_ground_fit_slope"] is not None
        and abs(float(anchor_fit["target_cluster_ground_fit_slope"]) - 2.0)
        <= 0.2
        and all(
            int(rank) == target.full_space_target_energy_multiplicity
            for rank in anchor_fit["target_cluster_ranks"]
        )
        and float(anchor_fit["min_target_cluster_principal_overlap"]) >= 0.9
        and float(anchor_fit["min_target_cluster_ground_state_weight"])
        >= float(paper_ground_state_weight_threshold)
    )
    paper_inapplicable_reason = (
        None
        if paper_comparison_applicable
        else (
            "anchor target-cluster ground branch lacks a resolved second-order "
            "coefficient, a stable target cluster, or sufficient prepared-state weight"
        )
    )
    metric_slopes = {
        "c_gs_qpe_rmse": "qpe_rmse_fit_slope",
        "c_gs_ground_space_worst_rmse": "ground_space_worst_rmse_fit_slope",
        "c_gs_target_cluster_ground": "target_cluster_ground_fit_slope",
    }
    for metric, slope_metric in metric_slopes.items():
        for epsilon_total in epsilon_values:
            for randomized_method in randomized_methods:
                for step_cost_rule in ("partial_s2", "repo_hd_only"):
                    comparisons.append(
                        _cost_comparison(
                            fits,
                            metric=metric,
                            anchor_ld=anchor_ld,
                            epsilon_total=float(epsilon_total),
                            total_terms=sorted_hamiltonian.num_terms,
                            step_cost_rule=step_cost_rule,
                            randomized_method=randomized_method,
                            g_rand=g_rand,
                            reporting_floor=cgs_reporting_floor,
                            slope_metric=slope_metric,
                            max_delta=max(delta_values),
                            required_cluster_rank=(
                                target.full_space_target_energy_multiplicity
                                if metric == "c_gs_target_cluster_ground"
                                else None
                            ),
                            min_cluster_projector_overlap=(
                                0.9
                                if metric == "c_gs_target_cluster_ground"
                                else None
                            ),
                            min_cluster_state_weight=(
                                paper_ground_state_weight_threshold
                                if metric == "c_gs_target_cluster_ground"
                                else None
                            ),
                            applicable=(
                                paper_comparison_applicable
                                if metric == "c_gs_target_cluster_ground"
                                else True
                            ),
                            inapplicable_reason=(
                                paper_inapplicable_reason
                                if metric == "c_gs_target_cluster_ground"
                                else None
                            ),
                        )
                    )

    target_sector = target.sector
    return {
        "molecule": f"H{int(molecule_type)}",
        "molecule_type": int(molecule_type),
        "ham_name": sorted_hamiltonian.ham_name,
        "num_qubits": sorted_hamiltonian.num_qubits,
        "total_terms": sorted_hamiltonian.num_terms,
        "identity_coeff": sorted_hamiltonian.identity_coeff,
        "anchor_ld": anchor_ld,
        "target": {
            "selection": "full_H_ground_state_in_physical_spin_charge_sector",
            "energy": target.energy,
            "global_ground_energy": target.global_ground_energy,
            "global_and_target_ground_differ": not math.isclose(
                target.energy, target.global_ground_energy, rel_tol=0.0, abs_tol=1e-9
            ),
            "residual_norm": target.residual_norm,
            "sector_dimension": target_sector.dimension,
            "n_electrons": target_sector.n_electrons,
            "nelec_alpha": target_sector.nelec_alpha,
            "nelec_beta": target_sector.nelec_beta,
            "sz_value": target_sector.sz_value,
            "sector_ground_multiplicity": target.sector_ground_multiplicity,
            "sector_excitation_gap": target.sector_excitation_gap,
            "full_space_target_energy_multiplicity": target.full_space_target_energy_multiplicity,
            "full_space_target_energy_gap": target.full_space_target_energy_gap,
            "target_eigenspace_projection_residual": target.target_eigenspace_projection_residual,
            "max_full_spectral_distance_from_target": target.max_full_spectral_distance_from_target,
            "phase_alias_margin_at_max_delta": (
                math.pi
                - max(delta_values)
                * target.max_full_spectral_distance_from_target
            ),
        },
        "cgs_definition": {
            "paper_compatible": "lowest quasi-energy in the full-H target-energy cluster; cost use additionally requires a stable cluster, second-order slope, and sufficient prepared-state weight",
            "prepared_state_qpe_rmse": "spectral QPE RMSE over S2 eigenphases weighted by the physical-sector full-H target state",
            "ground_space_worst_qpe_rmse": "largest QPE RMSE over the full-H target-energy eigenspace, including leakage outside that eigenspace",
            "ground_space_mean_qpe_rmse": "basis-invariant uniform average over the full-H target-energy eigenspace; diagnostic, not a worst-case guarantee",
            "single_branch_proxy": "per-delta maximum prepared-state-overlap branch; diagnostic only when the full-H target energy is degenerate",
            "diagnostic": "-phase(<psi_full|S2(H1,...,H_LD,H_R_exact)|psi_full>)/delta",
            "order_fixed": 2,
            "all_prefix_ratio": {
                "defined": False,
                "reason": "LD=0 gives exact exp(-i H delta), hence Cgs=0 and Cmax/Cmin is infinite/undefined; finite spans below exclude LD=0 and unresolved values",
            },
        },
        "qpe_rmse_cgs_span_positive": _positive_span(
            fits, "c_gs_qpe_rmse", floor=cgs_reporting_floor
        ),
        "qpe_rmse_cgs_span_slope_valid": _positive_span(
            fits,
            "c_gs_qpe_rmse",
            floor=cgs_reporting_floor,
            slope_metric="qpe_rmse_fit_slope",
            slope_tolerance=0.2,
        ),
        "ground_space_worst_rmse_cgs_span_slope_valid": _positive_span(
            fits,
            "c_gs_ground_space_worst_rmse",
            floor=cgs_reporting_floor,
            slope_metric="ground_space_worst_rmse_fit_slope",
            slope_tolerance=0.2,
        ),
        "target_cluster_ground_cgs_span_slope_valid": _positive_span(
            fits,
            "c_gs_target_cluster_ground",
            floor=cgs_reporting_floor,
            slope_metric="target_cluster_ground_fit_slope",
            slope_tolerance=0.2,
        ),
        "target_cluster_ground_cgs_span_algorithm_valid": _positive_span(
            fits,
            "c_gs_target_cluster_ground",
            floor=cgs_reporting_floor,
            slope_metric="target_cluster_ground_fit_slope",
            slope_tolerance=0.2,
            overlap_metric="min_target_cluster_ground_state_weight",
            min_overlap=paper_ground_state_weight_threshold,
            required_cluster_rank=target.full_space_target_energy_multiplicity,
            min_cluster_principal_overlap=0.9,
        ),
        "signal_cgs_span": _positive_span(
            fits, "c_gs_signal", floor=cgs_reporting_floor
        ),
        "branch_cgs_span": (
            _positive_span(fits, "c_gs_branch", floor=cgs_reporting_floor)
            if branch_comparison_applicable
            else {
                "defined": False,
                "reason": branch_inapplicable_reason,
            }
        ),
        "branch_cgs_span_slope_valid": (
            _positive_span(
                fits,
                "c_gs_branch",
                floor=cgs_reporting_floor,
                slope_metric="branch_fit_slope",
                slope_tolerance=0.2,
            )
            if branch_comparison_applicable
            else {
                "defined": False,
                "reason": branch_inapplicable_reason,
            }
        ),
        "branch_cgs_span_high_weight_slope_valid": (
            _positive_span(
                fits,
                "c_gs_branch",
                floor=cgs_reporting_floor,
                slope_metric="branch_fit_slope",
                slope_tolerance=0.2,
                overlap_metric="min_branch_overlap",
                min_overlap=branch_overlap_threshold,
            )
            if branch_comparison_applicable
            else {
                "defined": False,
                "reason": branch_inapplicable_reason,
            }
        ),
        "max_unitary_defect": max(
            max(float(value) for value in fit["unitary_defects"]) for fit in fits
        ),
        "fits": fits,
        "cost_comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full-H, exact-tail-block S2 Cgs for every Pauli prefix and "
            "compare an anchor-constant cost model with the per-LD exact model."
        )
    )
    parser.add_argument("--molecules", default="2,3")
    parser.add_argument(
        "--delta-values", default=",".join(str(value) for value in DEFAULT_DELTAS)
    )
    parser.add_argument("--epsilon-total", type=float, default=1e-4)
    parser.add_argument(
        "--epsilon-values",
        default=None,
        help="Optional comma-separated epsilon sweep; overrides --epsilon-total.",
    )
    parser.add_argument("--noise-floor", type=float, default=1e-12)
    parser.add_argument("--cgs-reporting-floor", type=float, default=1e-10)
    parser.add_argument("--randomized-method", choices=("qdrift", "rte"), default="qdrift")
    parser.add_argument(
        "--randomized-methods",
        default=None,
        help="Optional comma-separated method sweep; overrides --randomized-method.",
    )
    parser.add_argument("--g-rand", type=float, default=1.0)
    parser.add_argument("--branch-overlap-threshold", type=float, default=0.9)
    parser.add_argument(
        "--paper-ground-state-weight-threshold", type=float, default=0.536
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    molecules = _parse_csv_ints(args.molecules)
    delta_values = _parse_csv_floats(args.delta_values)
    epsilon_values = (
        _parse_csv_floats(args.epsilon_values)
        if args.epsilon_values is not None
        else (float(args.epsilon_total),)
    )
    randomized_methods = (
        _parse_csv_methods(args.randomized_methods)
        if args.randomized_methods is not None
        else (str(args.randomized_method),)
    )
    document = {
        "schema_version": 10,
        "artifact_type": "pauli_full_h_partial_s2_cgs_all_prefix_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": {
            "molecules": list(molecules),
            "pf_label": "2nd",
            "delta_values": list(delta_values),
            "epsilon_values": list(epsilon_values),
            "noise_floor": float(args.noise_floor),
            "cgs_reporting_floor": float(args.cgs_reporting_floor),
            "randomized_methods": list(randomized_methods),
            "g_rand": float(args.g_rand),
            "branch_overlap_threshold": float(args.branch_overlap_threshold),
            "paper_ground_state_weight_threshold": float(
                args.paper_ground_state_weight_threshold
            ),
            "error_budget_rule": "quadrature",
            "kappa_mode": "optimize",
            "deterministic_step_domain_rule": "delta <= max(delta_values)",
        },
        "molecules": [],
    }
    for molecule_type in molecules:
        document["molecules"].append(
            _molecule_result(
                molecule_type,
                delta_values=delta_values,
                epsilon_values=epsilon_values,
                noise_floor=float(args.noise_floor),
                cgs_reporting_floor=float(args.cgs_reporting_floor),
                randomized_methods=randomized_methods,
                g_rand=float(args.g_rand),
                branch_overlap_threshold=float(args.branch_overlap_threshold),
                paper_ground_state_weight_threshold=float(
                    args.paper_ground_state_weight_threshold
                ),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

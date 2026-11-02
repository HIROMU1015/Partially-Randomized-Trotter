"""WP01-D/C07 full-scope allocation and schedule optimization compute stage."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .df_partial_s2 import DFPartialS2Preparation
from .research_direction_ablation import (
    AffineAxisCostModel,
    PairCompiledCostModel,
    evaluate_ablation_scenario,
    fingerprint,
)
from .research_direction_full_scope import (
    AXES,
    POLICY_LABEL,
    validate_wp05a_artifact,
)
from .research_direction_full_scope_extension import validate_wp05b_artifact
from .research_direction_full_scope_replication import (
    validate_wp05br_artifact,
)
from .research_direction_prevalidation import validate_artifact


SCHEMA_VERSION = "research_direction_decision_cost_compute_v2"
METHOD = "wp01d_c07_full_scope_allocation_schedule_grid_v2"
DELTAS = (0.01, 0.02)
RTE_STEPS = (1, 2, 4, 8, 16, 32)
CALIBRATION_Q = (1, 2)
MAXIMUM_ALPHA_ITERATIONS = 20


def _semantic_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _axis_model(
    point_q1: Mapping[str, Any],
    point_q2: Mapping[str, Any],
    *,
    policy: str | None,
    axis: str,
) -> AffineAxisCostModel:
    if policy is None:
        q1 = point_q1["axes"][axis]["rz_count"]
        q2 = point_q2["axes"][axis]["rz_count"]
    else:
        q1 = point_q1["axes"][axis]["policies"][policy]["rz_count"]
        q2 = point_q2["axes"][axis]["policies"][policy]["rz_count"]
    return AffineAxisCostModel(
        q1_mean=float(q1["mean"]),
        q2_mean=float(q2["mean"]),
        q1_standard_error=float(q1["standard_error"]),
        q2_standard_error=float(q2["standard_error"]),
    )


def _pair_model(
    points: Mapping[str, Any],
    *,
    rte_steps: int,
    finite_taylor_order: int,
    policy: str | None,
    source_fingerprint: str,
    source_label: str,
) -> PairCompiledCostModel:
    q1 = points["1"]
    q2 = points["2"]
    model_fingerprint = _semantic_fingerprint(
        {
            "source_fingerprint": source_fingerprint,
            "source_label": source_label,
            "rte_steps": rte_steps,
            "finite_taylor_order": finite_taylor_order,
            "policy": policy,
            "q1": q1,
            "q2": q2,
        }
    )
    return PairCompiledCostModel(
        rte_steps=rte_steps,
        finite_taylor_order=finite_taylor_order,
        cosine=_axis_model(q1, q2, policy=policy, axis="cosine"),
        sine=_axis_model(q1, q2, policy=policy, axis="sine"),
        dataset_fingerprint=source_fingerprint,
        proxy_fingerprint=model_fingerprint,
    )


def build_full_scope_cost_models(
    wp05a: Mapping[str, Any],
    wp05b: Mapping[str, Any],
    wp05br: Mapping[str, Any],
) -> tuple[dict[tuple[int, float], tuple[PairCompiledCostModel, ...]], dict]:
    """Build q=1,2 full-wrapper RZ providers for each candidate and delta."""
    validate_wp05a_artifact(wp05a)
    validate_wp05b_artifact(wp05b)
    validate_wp05br_artifact(wp05br)
    if not wp05br["overall_pass"]:
        raise ValueError("WP05-bR must pass before WP01-D/C07 optimization.")

    models: dict[
        tuple[int, float], tuple[PairCompiledCostModel, ...]
    ] = {}
    sources: dict[str, Any] = {}
    for delta in DELTAS:
        candidate_models = []
        for rte_steps in RTE_STEPS:
            if delta == 0.02 and rte_steps == 32:
                points = wp05br["direct_randomized_ld3"]["points"]
                source = wp05br
                source_label = "WP05-bR_fresh_32_trajectory_r32"
            elif delta == 0.02:
                points = wp05a["randomized_ld3"][str(rte_steps)]["points"]
                source = wp05a
                source_label = "WP05-a_q1_q2"
            else:
                points = wp05b["comparison_delta_0p01"][
                    "direct_randomized_ld3"
                ][str(rte_steps)]["points"]
                source = wp05b
                source_label = "WP05-b_delta_0p01_q1_q2"
            candidate_models.append(
                _pair_model(
                    points,
                    rte_steps=rte_steps,
                    finite_taylor_order=2,
                    policy=POLICY_LABEL,
                    source_fingerprint=source["content_fingerprint"],
                    source_label=source_label,
                )
            )
            sources[f"ld3:delta{delta:g}:r{rte_steps}"] = {
                "artifact_fingerprint": source["content_fingerprint"],
                "source_label": source_label,
                "policy": POLICY_LABEL,
                "q_values": list(CALIBRATION_Q),
            }
        models[(3, delta)] = tuple(candidate_models)

        if delta == 0.02:
            deterministic_points = wp05a["deterministic_ld12"]["points"]
            source = wp05a
            source_label = "WP05-a_deterministic_q1_q2"
        else:
            deterministic_points = wp05b["comparison_delta_0p01"][
                "deterministic_ld12_points"
            ]
            source = wp05b
            source_label = "WP05-b_delta_0p01_deterministic_q1_q2"
        models[(12, delta)] = (
            _pair_model(
                deterministic_points,
                rte_steps=0,
                finite_taylor_order=0,
                policy=None,
                source_fingerprint=source["content_fingerprint"],
                source_label=source_label,
            ),
        )
        sources[f"ld12:delta{delta:g}:r0"] = {
            "artifact_fingerprint": source["content_fingerprint"],
            "source_label": source_label,
            "policy": "deterministic_tail_free",
            "q_values": list(CALIBRATION_Q),
        }
    return models, sources


def _candidate_configuration(
    wp01: Mapping[str, Any], *, ld: int, delta_time: float
) -> tuple[int, float]:
    matches = [
        row
        for row in wp01["candidates"]
        if int(row["ld"]) == ld
        and math.isclose(
            float(row["delta_time"]), delta_time, abs_tol=1e-15
        )
    ]
    if len(matches) != 1:
        raise ValueError("WP01 candidate is missing or duplicated.")
    row = matches[0]
    schedule = row["schedule"]
    if not schedule["all_rounds_feasible"] or not schedule["target_met"]:
        raise ValueError("WP01 candidate schedule is not feasible.")
    return int(schedule["maximum_round_index_M"]), float(row["pf_coefficient"])


def _scenario_summary(scenario: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "beta_pf_budget": float(scenario["beta_pf_budget"]),
        "beta_rte_budget": float(scenario["beta_rte_budget"]),
        "beta_stat_budget": float(scenario["beta_stat_budget"]),
        "total_compiled_rz_point_estimate": float(
            scenario["total_compiled_rz_point_estimate"]
        ),
        "conservative_calibration_95_half_width": float(
            scenario["conservative_calibration_95_half_width"]
        ),
        "local_5_percent_plus_calibration_interval": list(
            scenario["scenario_intervals"][
                "local_5_percent_plus_calibration"
            ]
        ),
        "total_shots": int(scenario["total_shots"]),
        "selected_r_k_pairs": list(scenario["selected_r_k_pairs"]),
        "alpha_converged": bool(scenario["alpha_converged"]),
        "final_round_cost_fraction": float(
            scenario["final_round_cost_fraction"]
        ),
        "last_three_round_cost_fraction": float(
            scenario["last_three_round_cost_fraction"]
        ),
    }


def _evaluate_profile(
    preparation: DFPartialS2Preparation,
    models: Sequence[PairCompiledCostModel],
    *,
    ld: int,
    delta_time: float,
    maximum_round_index: int,
    pf_coefficient: float,
    beta_pf: float,
    beta_rte: float,
    beta_rpe: float,
    alpha_total: float,
    rte_seed: int,
) -> dict[str, Any]:
    beta_stat = beta_rpe - beta_pf - beta_rte
    if beta_stat <= 0.0:
        raise RuntimeError("Non-positive statistical phase budget.")
    return evaluate_ablation_scenario(
        preparation,
        ld=ld,
        pf_coefficient=pf_coefficient,
        cost_models=models,
        beta_profile_label=(
            f"grid_pf_{beta_pf:.12g}_rte_{beta_rte:.12g}"
        ),
        beta_profile=(beta_pf, beta_rte, beta_stat),
        schedule_policy="round_compiled_rz",
        alpha_policy="cost_sensitivity_weighted",
        maximum_round_index=maximum_round_index,
        delta_time=delta_time,
        beta_rpe=beta_rpe,
        alpha_total=alpha_total,
        rte_seed=rte_seed,
        maximum_alpha_iterations=MAXIMUM_ALPHA_ITERATIONS,
    )


def _grid_search(
    preparation: DFPartialS2Preparation,
    models: Sequence[PairCompiledCostModel],
    *,
    ld: int,
    delta_time: float,
    maximum_round_index: int,
    pf_coefficient: float,
    beta_rpe: float,
    alpha_total: float,
    rte_seed: int,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    pf_values = np.linspace(0.0025, 0.08, 32)
    rte_values = (
        np.asarray([0.0])
        if ld == 12
        else np.linspace(0.001, 0.08, 32)
    )
    feasible = []
    failure_count = 0
    evaluation_count = 0

    def evaluate_grid(
        phase: str,
        pf_grid: Sequence[float],
        rte_grid: Sequence[float],
    ) -> list[dict[str, Any]]:
        nonlocal failure_count, evaluation_count
        rows = []
        total = len(pf_grid) * len(rte_grid)
        for beta_pf in pf_grid:
            for beta_rte in rte_grid:
                evaluation_count += 1
                if progress is not None and evaluation_count % 100 == 1:
                    progress(
                        f"ld={ld} delta={delta_time:g} {phase} "
                        f"evaluation={evaluation_count} phase_total={total}"
                    )
                try:
                    scenario = _evaluate_profile(
                        preparation,
                        models,
                        ld=ld,
                        delta_time=delta_time,
                        maximum_round_index=maximum_round_index,
                        pf_coefficient=pf_coefficient,
                        beta_pf=float(beta_pf),
                        beta_rte=float(beta_rte),
                        beta_rpe=beta_rpe,
                        alpha_total=alpha_total,
                        rte_seed=rte_seed,
                    )
                except RuntimeError:
                    failure_count += 1
                    continue
                if not scenario["alpha_converged"]:
                    failure_count += 1
                    continue
                rows.append(scenario)
        return rows

    coarse = evaluate_grid("coarse", pf_values, rte_values)
    if not coarse:
        raise RuntimeError("No feasible coarse beta allocation.")
    feasible.extend(coarse)
    coarse_best = min(
        coarse, key=lambda row: row["total_compiled_rz_point_estimate"]
    )
    pf_step = float(pf_values[1] - pf_values[0])
    best_pf = float(coarse_best["beta_pf_budget"])
    fine_pf = np.linspace(
        max(0.0001, best_pf - pf_step),
        min(beta_rpe - 0.0002, best_pf + pf_step),
        25,
    )
    if ld == 12:
        fine_rte = np.asarray([0.0])
    else:
        rte_step = float(rte_values[1] - rte_values[0])
        best_rte = float(coarse_best["beta_rte_budget"])
        fine_rte = np.linspace(
            max(0.0001, best_rte - rte_step),
            min(beta_rpe - 0.0002, best_rte + rte_step),
            25,
        )
    fine = evaluate_grid("fine", fine_pf, fine_rte)
    if not fine:
        raise RuntimeError("No feasible fine beta allocation.")
    feasible.extend(fine)
    boundary_refined = []
    for seed_scenario in sorted(
        feasible,
        key=lambda row: row["total_compiled_rz_point_estimate"],
    )[:20]:
        current = seed_scenario
        for _iteration in range(8):
            beta_pf = float(current["maximum_empirical_pf_phase_proxy"])
            beta_pf += max(1e-12, abs(beta_pf) * 1e-10)
            if ld == 12:
                beta_rte = 0.0
            else:
                beta_rte = float(current["maximum_finite_rte_phase_bound"])
                beta_rte += max(1e-12, abs(beta_rte) * 1e-10)
            try:
                tightened = _evaluate_profile(
                    preparation,
                    models,
                    ld=ld,
                    delta_time=delta_time,
                    maximum_round_index=maximum_round_index,
                    pf_coefficient=pf_coefficient,
                    beta_pf=beta_pf,
                    beta_rte=beta_rte,
                    beta_rpe=beta_rpe,
                    alpha_total=alpha_total,
                    rte_seed=rte_seed,
                )
            except RuntimeError:
                failure_count += 1
                break
            evaluation_count += 1
            if not tightened["alpha_converged"]:
                failure_count += 1
                break
            boundary_refined.append(tightened)
            old_pair = (
                float(current["beta_pf_budget"]),
                float(current["beta_rte_budget"]),
            )
            new_pair = (
                float(tightened["beta_pf_budget"]),
                float(tightened["beta_rte_budget"]),
            )
            current = tightened
            if all(
                math.isclose(old, new, rel_tol=0.0, abs_tol=1e-13)
                for old, new in zip(old_pair, new_pair, strict=True)
            ):
                break
    if not boundary_refined:
        raise RuntimeError("No constraint-boundary refinement succeeded.")
    feasible.extend(boundary_refined)
    best = min(
        feasible, key=lambda row: row["total_compiled_rz_point_estimate"]
    )
    return {
        "ld": ld,
        "delta_time": delta_time,
        "maximum_round_index_M": maximum_round_index,
        "q_max": 1 << maximum_round_index,
        "pf_coefficient": pf_coefficient,
        "coarse_grid_shape": [len(pf_values), len(rte_values)],
        "fine_grid_shape": [len(fine_pf), len(fine_rte)],
        "constraint_boundary_seed_count": 20,
        "constraint_boundary_refined_count": len(boundary_refined),
        "evaluation_count": evaluation_count,
        "infeasible_or_nonconverged_count": failure_count,
        "feasible_count": len(feasible),
        "coarse_best": _scenario_summary(coarse_best),
        "best": best,
    }


def _intervals_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) <= min(
        float(left[1]), float(right[1])
    )


def evaluate_wp01d_c07_compute(
    preparations: Mapping[int, DFPartialS2Preparation],
    wp01: Mapping[str, Any],
    wp05a: Mapping[str, Any],
    wp05b: Mapping[str, Any],
    wp05br: Mapping[str, Any],
    *,
    beta_rpe: float = 0.4,
    alpha_total: float = 0.05,
    rte_seed: int = 2026092208,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Optimize beta, alpha, and schedules using the validated full-scope cost."""
    validate_artifact(wp01)
    models, provider_sources = build_full_scope_cost_models(
        wp05a, wp05b, wp05br
    )
    candidates: dict[str, Any] = {}
    for ld in (3, 12):
        if preparations[ld].ld != ld:
            raise ValueError("Preparation L_D does not match candidate key.")
        for delta_time in DELTAS:
            maximum_round_index, pf_coefficient = _candidate_configuration(
                wp01, ld=ld, delta_time=delta_time
            )
            key = f"ld{ld}:delta{delta_time:g}"
            if progress is not None:
                progress(f"starting {key}")
            candidates[key] = _grid_search(
                preparations[ld],
                models[(ld, delta_time)],
                ld=ld,
                delta_time=delta_time,
                maximum_round_index=maximum_round_index,
                pf_coefficient=pf_coefficient,
                beta_rpe=beta_rpe,
                alpha_total=alpha_total,
                rte_seed=rte_seed,
                progress=progress,
            )
            if progress is not None:
                progress(f"completed {key}")

    best_by_ld = {
        str(ld): min(
            (
                row
                for row in candidates.values()
                if int(row["ld"]) == ld
            ),
            key=lambda row: row["best"][
                "total_compiled_rz_point_estimate"
            ],
        )
        for ld in (3, 12)
    }
    ld3 = best_by_ld["3"]["best"]
    ld12 = best_by_ld["12"]["best"]
    ld3_cost = float(ld3["total_compiled_rz_point_estimate"])
    ld12_cost = float(ld12["total_compiled_rz_point_estimate"])
    ld3_interval = ld3["scenario_intervals"][
        "local_5_percent_plus_calibration"
    ]
    ld12_interval = ld12["scenario_intervals"][
        "local_5_percent_plus_calibration"
    ]
    overlap = _intervals_overlap(ld3_interval, ld12_interval)
    checks = {
        "wp05br_replication_passed": bool(wp05br["overall_pass"]),
        "all_four_candidate_delta_grids_completed": len(candidates) == 4,
        "all_grids_have_feasible_points": all(
            int(row["feasible_count"]) > 0 for row in candidates.values()
        ),
        "all_selected_alpha_allocations_converged": all(
            bool(row["best"]["alpha_converged"])
            for row in candidates.values()
        ),
        "all_selected_beta_sums_match": all(
            math.isclose(
                float(row["best"]["beta_pf_budget"])
                + float(row["best"]["beta_rte_budget"])
                + float(row["best"]["beta_stat_budget"]),
                beta_rpe,
                abs_tol=1e-12,
            )
            for row in candidates.values()
        ),
        "all_selected_rounds_feasible": all(
            bool(row["best"]["all_rounds_feasible"])
            for row in candidates.values()
        ),
        "calculation_only_does_not_claim_final_scientific_superiority": True,
    }
    return {
        "configuration": {
            "comparison_task": "WP00 fixed H4 CA_over_10 task",
            "candidate_ld_values": [3, 12],
            "delta_times": list(DELTAS),
            "beta_rpe": beta_rpe,
            "alpha_total": alpha_total,
            "schedule_policy": "round_compiled_rz",
            "alpha_policy": "cost_sensitivity_weighted",
            "beta_search": "coarse_grid_plus_local_fine_grid",
            "rte_seed": rte_seed,
            "full_scope_cost_metric": "rz_count",
        },
        "provider_sources": provider_sources,
        "candidates": candidates,
        "best_by_ld": best_by_ld,
        "comparison": {
            "ld3_total_compiled_rz_point_estimate": ld3_cost,
            "ld12_total_compiled_rz_point_estimate": ld12_cost,
            "ld3_over_ld12_point_estimate_ratio": ld3_cost / ld12_cost,
            "point_preference": "L_D=3" if ld3_cost < ld12_cost else "L_D=12",
            "ld3_local_5_percent_plus_calibration_interval": list(
                ld3_interval
            ),
            "ld12_local_5_percent_plus_calibration_interval": list(
                ld12_interval
            ),
            "local_intervals_overlap": overlap,
            "directional_result": (
                "undetermined_intervals_overlap"
                if overlap
                else "conditional_local_intervals_separate"
            ),
        },
        "scope": {
            "full_controlled_hadamard_cost_provider_used": True,
            "q8_holdout_and_replication_required": True,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "noise_included": False,
            "calculation_stage_only": True,
            "synthesis_and_claim_review_pending": True,
            "final_scientific_superiority_claimed": False,
        },
        "limitations": [
            "Long-q costs remain an affine extrapolation from q=1,2 with direct q=4,8 holdouts and a focused r=32 q=8 replication.",
            "The local interval uses a shared conservative calibration sum plus a 5% discrepancy allowance; it is not a rigorous confidence interval.",
            "State preparation, backend execution, noise, and external reproduction are excluded.",
            "This compute artifact precedes WP01-D/C07 synthesis and does not by itself establish scientific superiority.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "point_preference": "L_D=3" if ld3_cost < ld12_cost else "L_D=12",
            "ld3_over_ld12_point_estimate_ratio": ld3_cost / ld12_cost,
            "local_intervals_overlap": overlap,
            "directional_result": (
                "undetermined_intervals_overlap"
                if overlap
                else "conditional_local_intervals_separate"
            ),
            "next_action": "WP01-D_C07_synthesis_and_claim_review",
        },
    }


def finalize_wp01d_compute_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP01-D/C07-compute",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp01d_compute_artifact(payload)
    return payload


def validate_wp01d_compute_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WP01-D/C07 compute schema.")
    if payload.get("method") != METHOD:
        raise ValueError("Unsupported WP01-D/C07 compute method.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP01-D/C07 compute fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP01-D/C07 compute status does not match checks.")
    scope = payload.get("scope", {})
    if scope.get("calculation_stage_only") is not True:
        raise ValueError("Artifact must remain a calculation-stage result.")
    if scope.get("final_scientific_superiority_claimed") is not False:
        raise ValueError("Compute stage cannot claim scientific superiority.")


def write_wp01d_compute_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_wp01d_compute_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

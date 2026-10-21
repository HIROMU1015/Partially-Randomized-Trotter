"""Focused M06/L08 compiler-transfer analysis and fixed-plan reaggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .research_direction_compiler_transfer_compute import (
    validate_compiler_transfer_compute_artifact,
)
from .research_direction_decision_cost import validate_wp01d_compute_artifact
from .research_direction_full_scope import AXES, METRICS, POLICY_LABEL, fingerprint
from .research_direction_full_scope_extension import fit_affine_holdouts
from .research_direction_full_scope_replication import validate_wp05br_artifact
from .research_direction_prevalidation import affine_prediction_with_standard_error
from .research_direction_proxy_precision import validate_m08_artifact


SCHEMA_VERSION = "research_direction_compiler_transfer_analysis_v1"
METHOD = "m06_l08_opt2_focused_analysis_and_reaggregation_v1"
HOLDOUT_Q = (16, 32)


def _point_metric(
    point: Mapping[str, Any], *, axis: str, metric: str, policy: str | None
) -> Mapping[str, Any]:
    axis_record = point["axes"][axis]
    if policy is None:
        return axis_record[metric]
    return axis_record["policies"][policy][metric]


def _baseline_ld3_points(
    wp05br: Mapping[str, Any], m08: Mapping[str, Any]
) -> dict[int, Mapping[str, Any]]:
    return {
        1: wp05br["direct_randomized_ld3"]["points"]["1"],
        2: wp05br["direct_randomized_ld3"]["points"]["2"],
        16: m08["direct_holdout_points"]["16"],
        32: m08["direct_holdout_points"]["32"],
    }


def _raw_points(
    raw: Mapping[str, Any], ld: int
) -> dict[int, Mapping[str, Any]]:
    return {
        int(q): point for q, point in raw["direct_points"][str(ld)].items()
    }


def _compiler_difference(
    baseline: Mapping[str, Any], transferred: Mapping[str, Any]
) -> dict[str, Any]:
    keys = sorted(set(baseline) | set(transferred))
    changed = {
        key: {"baseline": baseline.get(key), "transferred": transferred.get(key)}
        for key in keys
        if baseline.get(key) != transferred.get(key)
    }
    return {
        "changed_fields": changed,
        "only_optimization_level_changed": (
            set(changed) == {"optimization_level"}
            and changed["optimization_level"]
            == {"baseline": 1, "transferred": 2}
        ),
    }


def _same_trajectory_effect(
    baseline: Mapping[int, Mapping[str, Any]],
    transferred: Mapping[int, Mapping[str, Any]],
    *,
    policy: str | None,
) -> dict[str, Any]:
    rows = []
    for q_m in sorted(baseline):
        for axis in AXES:
            for metric in METRICS:
                old = float(
                    _point_metric(
                        baseline[q_m], axis=axis, metric=metric, policy=policy
                    )["mean"]
                )
                new = float(
                    _point_metric(
                        transferred[q_m], axis=axis, metric=metric, policy=policy
                    )["mean"]
                )
                rows.append(
                    {
                        "q_m": q_m,
                        "axis": axis,
                        "metric": metric,
                        "optimization_level_1_mean": old,
                        "optimization_level_2_mean": new,
                        "relative_change": (new - old) / old,
                        "level_2_over_level_1_ratio": new / old,
                    }
                )
    rz_rows = [row for row in rows if row["metric"] == "rz_count"]
    return {
        "rows": rows,
        "summary": {
            "minimum_relative_change_all_metrics": min(
                row["relative_change"] for row in rows
            ),
            "maximum_relative_change_all_metrics": max(
                row["relative_change"] for row in rows
            ),
            "minimum_relative_change_rz": min(
                row["relative_change"] for row in rz_rows
            ),
            "maximum_relative_change_rz": max(
                row["relative_change"] for row in rz_rows
            ),
            "mean_rz_level_2_over_level_1_ratio": math.fsum(
                row["level_2_over_level_1_ratio"] for row in rz_rows
            )
            / len(rz_rows),
        },
    }


def _maximum_holdout_error(
    models: Mapping[str, Any], *, metrics: Sequence[str]
) -> dict[str, Any]:
    rows = []
    for axis in AXES:
        for metric in metrics:
            for q_text, holdout in models[axis][metric]["holdouts"].items():
                rows.append(
                    {
                        "axis": axis,
                        "metric": metric,
                        "q_m": int(q_text),
                        "absolute_relative_error": float(
                            holdout["absolute_relative_error"]
                        ),
                    }
                )
    return max(rows, key=lambda row: row["absolute_relative_error"])


def _maximum_direct_rz_relative_standard_error(
    points: Mapping[int, Mapping[str, Any]], *, policy: str | None
) -> dict[str, Any]:
    rows = []
    for q_m in HOLDOUT_Q:
        for axis in AXES:
            record = _point_metric(
                points[q_m], axis=axis, metric="rz_count", policy=policy
            )
            rows.append(
                {
                    "q_m": q_m,
                    "axis": axis,
                    "relative_standard_error": abs(
                        float(record["standard_error"]) / float(record["mean"])
                    ),
                }
            )
    return max(rows, key=lambda row: row["relative_standard_error"])


def _predict_metric(
    points: Mapping[int, Mapping[str, Any]],
    *,
    q_m: int,
    axis: str,
    policy: str | None,
) -> tuple[float, float]:
    q1 = _point_metric(points[1], axis=axis, metric="rz_count", policy=policy)
    q2 = _point_metric(points[2], axis=axis, metric="rz_count", policy=policy)
    return affine_prediction_with_standard_error(
        q_m=q_m,
        q1_mean=float(q1["mean"]),
        q2_mean=float(q2["mean"]),
        q1_standard_error=float(q1["standard_error"]),
        q2_standard_error=float(q2["standard_error"]),
    )


def _fixed_plan_reaggregation(
    baseline_best: Mapping[str, Any],
    transferred_points: Mapping[int, Mapping[str, Any]],
    *,
    ld: int,
    policy: str | None,
    replace_only_r32: bool,
) -> dict[str, Any]:
    rows = []
    for original in baseline_best["rounds"]:
        replace = not replace_only_r32 or int(original["r_m"]) == 32
        if replace:
            point = 0.0
            standard_error_sum = 0.0
            axis_rows = {}
            for axis in AXES:
                mean, standard_error = _predict_metric(
                    transferred_points,
                    q_m=int(original["q_m"]),
                    axis=axis,
                    policy=policy,
                )
                shots = int(original["axes"][axis]["shots"])
                point += shots * mean
                standard_error_sum += shots * standard_error
                axis_rows[axis] = {
                    "shots": shots,
                    "predicted_rz_count_per_interrogation": mean,
                    "propagated_calibration_standard_error": standard_error,
                }
        else:
            point = float(original["compiled_rz_point_estimate"])
            standard_error_sum = float(
                original["calibration_standard_error_conservative_sum"]
            )
            axis_rows = {
                axis: {
                    "shots": int(original["axes"][axis]["shots"]),
                    "predicted_rz_count_per_interrogation": float(
                        original["axes"][axis][
                            "predicted_rz_count_per_interrogation"
                        ]
                    ),
                    "propagated_calibration_standard_error": float(
                        original["axes"][axis][
                            "propagated_calibration_standard_error"
                        ]
                    ),
                }
                for axis in AXES
            }
        rows.append(
            {
                "round_index": int(original["round_index"]),
                "q_m": int(original["q_m"]),
                "r_m": int(original["r_m"]),
                "transferred_compiler_applied": replace,
                "axes": axis_rows,
                "baseline_compiled_rz_point_estimate": float(
                    original["compiled_rz_point_estimate"]
                ),
                "reaggregated_compiled_rz_point_estimate": point,
                "reaggregated_calibration_standard_error_conservative_sum": (
                    standard_error_sum
                ),
            }
        )
    total = math.fsum(
        row["reaggregated_compiled_rz_point_estimate"] for row in rows
    )
    calibration_sum = math.fsum(
        row["reaggregated_calibration_standard_error_conservative_sum"]
        for row in rows
    )
    replaced = math.fsum(
        row["reaggregated_compiled_rz_point_estimate"]
        for row in rows
        if row["transferred_compiler_applied"]
    )
    baseline_total = float(baseline_best["total_compiled_rz_point_estimate"])
    return {
        "ld": ld,
        "schedule_shots_alpha_and_beta_held_fixed": True,
        "replace_only_r32": replace_only_r32,
        "baseline_total_compiled_rz_point_estimate": baseline_total,
        "reaggregated_total_compiled_rz_point_estimate": total,
        "relative_change_from_baseline": (total - baseline_total) / baseline_total,
        "conservative_calibration_standard_error_sum": calibration_sum,
        "conservative_calibration_95_half_width": 1.96 * calibration_sum,
        "transferred_compiler_cost_fraction": replaced / total,
        "rounds": rows,
    }


def _interval(cost: float, half_width: float, discrepancy: float) -> list[float]:
    return [
        max(0.0, (1.0 - discrepancy) * cost - half_width),
        (1.0 + discrepancy) * cost + half_width,
    ]


def _overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) <= min(
        float(left[1]), float(right[1])
    )


def _comparison(
    ld3_cost: float,
    ld3_half_width: float,
    ld12_cost: float,
    ld12_half_width: float,
    discrepancies: Mapping[str, float],
) -> dict[str, Any]:
    scenarios = {}
    for label, discrepancy in discrepancies.items():
        ld3_interval = _interval(ld3_cost, ld3_half_width, discrepancy)
        ld12_interval = _interval(ld12_cost, ld12_half_width, discrepancy)
        scenarios[label] = {
            "symmetric_relative_discrepancy": discrepancy,
            "ld3_interval": ld3_interval,
            "ld12_interval": ld12_interval,
            "intervals_overlap": _overlap(ld3_interval, ld12_interval),
            "signed_separation_gap": ld12_interval[0] - ld3_interval[1],
        }
    separation_limit = max(
        0.0,
        (
            ld12_cost
            - ld3_cost
            - ld3_half_width
            - ld12_half_width
        )
        / (ld3_cost + ld12_cost),
    )
    return {
        "ld3_total_compiled_rz_point_estimate": ld3_cost,
        "ld12_total_compiled_rz_point_estimate": ld12_cost,
        "ld3_over_ld12_point_estimate_ratio": ld3_cost / ld12_cost,
        "ld3_point_advantage_fraction": 1.0 - ld3_cost / ld12_cost,
        "point_preference": "L_D=3" if ld3_cost < ld12_cost else "L_D=12",
        "symmetric_discrepancy_separation_limit": separation_limit,
        "scenarios": scenarios,
    }


def evaluate_compiler_transfer_analysis(
    raw: Mapping[str, Any],
    wp05br: Mapping[str, Any],
    m08: Mapping[str, Any],
    wp01d: Mapping[str, Any],
) -> dict[str, Any]:
    """Analyze opt-level-2 points without pretending to have a full reoptimization."""
    validate_compiler_transfer_compute_artifact(raw)
    validate_wp05br_artifact(wp05br)
    validate_m08_artifact(m08)
    validate_wp01d_compute_artifact(wp01d)

    compiler_difference = _compiler_difference(
        wp05br["configuration"]["compiler"],
        raw["configuration"]["compiler"],
    )
    baseline_ld3 = _baseline_ld3_points(wp05br, m08)
    opt2_ld3 = _raw_points(raw, 3)
    opt2_ld12 = _raw_points(raw, 12)

    selected_effect = _same_trajectory_effect(
        baseline_ld3, opt2_ld3, policy=POLICY_LABEL
    )
    full_effect = _same_trajectory_effect(
        baseline_ld3, opt2_ld3, policy="full_basis_shared"
    )

    selected_models = fit_affine_holdouts(
        opt2_ld3, policy=POLICY_LABEL, holdout_q=HOLDOUT_Q
    )
    full_models = fit_affine_holdouts(
        opt2_ld3, policy="full_basis_shared", holdout_q=HOLDOUT_Q
    )
    deterministic_models = fit_affine_holdouts(
        opt2_ld12, policy=None, holdout_q=HOLDOUT_Q
    )
    selected_rz_error = _maximum_holdout_error(
        selected_models, metrics=("rz_count",)
    )
    selected_all_error = _maximum_holdout_error(
        selected_models, metrics=METRICS
    )
    full_rz_error = _maximum_holdout_error(
        full_models, metrics=("rz_count",)
    )
    full_all_error = _maximum_holdout_error(full_models, metrics=METRICS)
    deterministic_all_error = _maximum_holdout_error(
        deterministic_models, metrics=METRICS
    )
    selected_direct_rz_se = _maximum_direct_rz_relative_standard_error(
        opt2_ld3, policy=POLICY_LABEL
    )
    full_direct_rz_se = _maximum_direct_rz_relative_standard_error(
        opt2_ld3, policy="full_basis_shared"
    )
    direct_rz_se = max(
        (selected_direct_rz_se, full_direct_rz_se),
        key=lambda row: row["relative_standard_error"],
    )

    baseline_ld3_best = wp01d["best_by_ld"]["3"]["best"]
    baseline_ld12_best = wp01d["best_by_ld"]["12"]["best"]
    focused_ld3 = _fixed_plan_reaggregation(
        baseline_ld3_best,
        opt2_ld3,
        ld=3,
        policy=POLICY_LABEL,
        replace_only_r32=True,
    )
    focused_ld12 = _fixed_plan_reaggregation(
        baseline_ld12_best,
        opt2_ld12,
        ld=12,
        policy=None,
        replace_only_r32=False,
    )

    discrepancies = {
        "opt2_selected_policy_measured": float(
            selected_rz_error["absolute_relative_error"]
        ),
        "opt2_maximum_observed_rz": float(
            full_rz_error["absolute_relative_error"]
        ),
        "local_5_percent": 0.05,
        "transfer_25_percent": 0.25,
    }
    focused_comparison = _comparison(
        float(focused_ld3["reaggregated_total_compiled_rz_point_estimate"]),
        float(focused_ld3["conservative_calibration_95_half_width"]),
        float(focused_ld12["reaggregated_total_compiled_rz_point_estimate"]),
        float(focused_ld12["conservative_calibration_95_half_width"]),
        discrepancies,
    )

    uniform_ratio = float(
        selected_effect["summary"]["mean_rz_level_2_over_level_1_ratio"]
    )
    uniform_ld3_cost = (
        float(baseline_ld3_best["total_compiled_rz_point_estimate"])
        * uniform_ratio
    )
    uniform_ld3_half_width = (
        float(baseline_ld3_best["conservative_calibration_95_half_width"])
        * uniform_ratio
    )
    counterfactual_comparison = _comparison(
        uniform_ld3_cost,
        uniform_ld3_half_width,
        float(focused_ld12["reaggregated_total_compiled_rz_point_estimate"]),
        float(focused_ld12["conservative_calibration_95_half_width"]),
        discrepancies,
    )

    selected_scenario = focused_comparison["scenarios"][
        "opt2_selected_policy_measured"
    ]
    checks = {
        "all_input_artifacts_validate_and_pass": bool(
            wp05br["overall_pass"] and m08["overall_pass"] and wp01d["overall_pass"]
        ),
        "only_optimization_level_changed": bool(
            compiler_difference["only_optimization_level_changed"]
        ),
        "opt2_selected_rz_q16_q32_error_within_5_percent": (
            float(selected_rz_error["absolute_relative_error"]) <= 0.05
        ),
        "opt2_selected_all_metric_q16_q32_error_within_5_percent": (
            float(selected_all_error["absolute_relative_error"]) <= 0.05
        ),
        "opt2_full_rz_q16_q32_error_within_5_percent": (
            float(full_rz_error["absolute_relative_error"]) <= 0.05
        ),
        "opt2_direct_rz_relative_standard_error_within_2_percent": (
            float(direct_rz_se["relative_standard_error"]) <= 0.02
        ),
        "opt2_deterministic_holdouts_exact": math.isclose(
            float(deterministic_all_error["absolute_relative_error"]),
            0.0,
            abs_tol=1e-15,
        ),
        "focused_measured_discrepancy_intervals_overlap": bool(
            selected_scenario["intervals_overlap"]
        ),
        "no_full_opt2_reoptimization_or_superiority_claim": True,
    }
    return {
        "configuration": {
            **{
                key: raw["configuration"][key]
                for key in (
                    "molecule",
                    "geometry_angstrom",
                    "basis",
                    "n_qubits",
                    "df_rank",
                    "delta_time",
                    "finite_taylor_order",
                    "rte_steps",
                )
            },
            "compiler_difference": compiler_difference,
            "cost_metric": "rz_count",
            "fixed_plan_fields": ["round schedule", "shots", "alpha", "beta"],
        },
        "same_trajectory_compiler_effect": {
            "ld3_selected_policy": selected_effect,
            "ld3_full_basis": full_effect,
        },
        "opt2_proxy_validation": {
            "selected_policy_affine_models": selected_models,
            "full_basis_affine_models": full_models,
            "deterministic_ld12_affine_models": deterministic_models,
            "maximum_selected_policy_rz_error": selected_rz_error,
            "maximum_selected_policy_all_metric_error": selected_all_error,
            "maximum_full_basis_rz_error": full_rz_error,
            "maximum_full_basis_all_metric_error": full_all_error,
            "maximum_deterministic_all_metric_error": deterministic_all_error,
            "maximum_direct_rz_relative_standard_error": direct_rz_se,
        },
        "focused_fixed_plan_reaggregation": {
            "ld3": focused_ld3,
            "ld12": focused_ld12,
            "comparison": focused_comparison,
            "interpretation": (
                "r=32 direct transfer sensitivity; L_D=3 r<32 retains optimization-level-1 costs"
            ),
        },
        "uniform_ratio_transfer_counterfactual": {
            "evidence_class": "counterfactual_not_directly_validated_for_r_below_32",
            "ld3_uniform_rz_ratio": uniform_ratio,
            "comparison": counterfactual_comparison,
        },
        "decision": {
            "opt2_q16_q32_proxy_adequacy": "passed",
            "fixed_plan_point_preference": focused_comparison["point_preference"],
            "compiler_invariant_local_separation": "not_established",
            "robust_directional_result": (
                "undetermined_under_compiler_and_transfer_sensitivity"
            ),
            "next_action": "N07_P03_claim_scope_and_break_even_before_external_instance",
        },
        "scope": {
            "same_trajectory_opt1_opt2_comparison": True,
            "ld3_opt2_direct_rte_steps": [32],
            "ld3_opt2_r_below_32_directly_validated": False,
            "q_above_32_directly_validated": False,
            "schedule_shots_alpha_beta_reoptimized_under_opt2": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
        "limitations": [
            "The focused reaggregation is not a coherent all-round optimization-level-2 run because L_D=3 r<32 costs remain at optimization level 1.",
            "The uniform-ratio transfer is a sensitivity counterfactual, not direct evidence for r<32.",
            "Direct q validation ends at q=32; longer-q schedule costs remain affine extrapolations.",
            "The intervals are empirical discrepancy envelopes plus conservative calibration sums, not rigorous confidence intervals.",
            "State preparation, backend execution, noise, and final total-cost evaluation remain outside scope.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "maximum_selected_policy_q16_q32_rz_relative_error": float(
                selected_rz_error["absolute_relative_error"]
            ),
            "maximum_full_basis_q16_q32_rz_relative_error": float(
                full_rz_error["absolute_relative_error"]
            ),
            "maximum_direct_rz_relative_standard_error": float(
                direct_rz_se["relative_standard_error"]
            ),
            "focused_ld3_point_estimate": focused_comparison[
                "ld3_total_compiled_rz_point_estimate"
            ],
            "focused_ld12_point_estimate": focused_comparison[
                "ld12_total_compiled_rz_point_estimate"
            ],
            "focused_ld3_point_advantage_fraction": focused_comparison[
                "ld3_point_advantage_fraction"
            ],
            "focused_symmetric_discrepancy_separation_limit": focused_comparison[
                "symmetric_discrepancy_separation_limit"
            ],
            "focused_measured_discrepancy_intervals_overlap": bool(
                selected_scenario["intervals_overlap"]
            ),
            "robust_directional_result": (
                "undetermined_under_compiler_and_transfer_sensitivity"
            ),
            "next_action": "N07_P03_claim_scope_and_break_even_before_external_instance",
        },
    }


def finalize_compiler_transfer_analysis_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "M06-L08-analysis-reaggregation",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_compiler_transfer_analysis_artifact(payload)
    return payload


def validate_compiler_transfer_analysis_artifact(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported compiler-transfer analysis schema.")
    if payload.get("method") != METHOD:
        raise ValueError("Unsupported compiler-transfer analysis method.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("Compiler-transfer analysis fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("Compiler-transfer analysis status does not match checks.")
    scope = payload.get("scope", {})
    if scope.get("schedule_shots_alpha_beta_reoptimized_under_opt2") is not False:
        raise ValueError("Focused analysis cannot claim a full opt2 reoptimization.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("Focused analysis cannot claim final total cost.")
    if scope.get("scientific_superiority_claimed") is not False:
        raise ValueError("Focused analysis cannot claim scientific superiority.")


def write_compiler_transfer_analysis_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_compiler_transfer_analysis_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

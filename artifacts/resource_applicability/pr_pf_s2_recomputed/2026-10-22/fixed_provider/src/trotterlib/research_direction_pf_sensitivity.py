"""WP03 Product-Formula coefficient selection-sensitivity helpers.

The routines in this module hold the WP04 task, cost models, schedule policy,
and allocation policies fixed while changing only the empirical PF coefficient
input.  The resulting artifact is a model-conditional Gate-S1 diagnostic, not
a rigorous PF certification or a final total-cost evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_partial_s2 import DFPartialS2Preparation
from .pf_delta_validation import validate_pf_delta_payload
from .research_direction_ablation import (
    PairCompiledCostModel,
    evaluate_ablation_scenario,
)
from .rpe_target_round_horizon_validation import required_rpe_round_horizon


SCHEMA_VERSION = "research_direction_pf_sensitivity_v1"
METHOD = "wp03_pf_coefficient_selection_regret_v1"
COEFFICIENT_POLICIES = (
    "hd_surrogate",
    "paper_d6",
    "dominant_eigenphase",
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sign(value: float | None, *, atol: float = 1e-18) -> int:
    if value is None or abs(float(value)) <= atol:
        return 0
    return 1 if float(value) > 0.0 else -1


def extract_pf_coefficient_audit(
    payloads: Mapping[int, Mapping[str, Any]],
    *,
    common_window_hd_fits: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract comparable C_D, paper-D6, and dominant-phase coefficients."""
    if not payloads:
        raise ValueError("At least one PF payload is required.")
    if common_window_hd_fits is not None:
        windows = {
            tuple(float(value) for value in fit["delta_values"])
            for fit in common_window_hd_fits.values()
        }
        if len(windows) != 1:
            raise ValueError("All H_D refits must use one common delta window.")
    hashes = set()
    rows: dict[str, dict[str, Any]] = {}
    for ld, payload in sorted(payloads.items()):
        validate_pf_delta_payload(payload)
        observed_ld = int(payload["request"]["ld"])
        if observed_ld != int(ld):
            raise ValueError("PF payload L_D does not match its mapping key.")
        hashes.add(str(payload["hamiltonian"]["hamiltonian_hash"]))
        summary = payload["summary"]
        surrogate = payload["surrogate"]
        original_hd_coefficient = float(surrogate["coefficient"])
        common_fit = (
            None
            if common_window_hd_fits is None
            else common_window_hd_fits.get(int(ld))
        )
        if common_window_hd_fits is not None and common_fit is None:
            raise ValueError("A common-window H_D fit is missing for one L_D.")
        hd_coefficient = (
            original_hd_coefficient
            if common_fit is None
            else float(common_fit["fixed_second_order_coefficient"])
        )
        coefficients = {
            "hd_surrogate": hd_coefficient,
            "paper_d6": float(
                summary["scalable_pf_fixed_second_order_coefficient"]
            ),
            "dominant_eigenphase": float(
                summary["recommended_pf_fixed_second_order_coefficient"]
            ),
        }
        if any(value < 0.0 for value in coefficients.values()):
            raise ValueError("PF coefficient magnitudes must be non-negative.")

        conditioned_d6_signs = []
        dominant_signs = []
        conditioned_deltas = []
        ill_conditioned_deltas = []
        for point in payload["points"]:
            delta = float(point["delta_time"])
            d6 = point["cpu_qiskit_direct_tail_validation"][
                "paper_d6_perturbative_energy_bias"
            ]
            if bool(d6["well_conditioned"]):
                conditioned_deltas.append(delta)
                conditioned_d6_signs.append(_sign(d6["signed_energy_bias"]))
            else:
                ill_conditioned_deltas.append(delta)
            dominant_signs.append(
                _sign(
                    point["qpe_spectral_energy_distribution"][
                        "dominant_phase_cluster_signed_energy_bias"
                    ]
                )
            )

        hd_signed_biases = (
            surrogate["signed_calibration_energy_biases"]
            if common_fit is None
            else common_fit["signed_energy_biases"]
        )
        hd_signs = (
            [0]
            if hd_coefficient == 0.0
            else sorted({_sign(float(value)) for value in hd_signed_biases})
        )
        d6_signs = sorted(set(conditioned_d6_signs))
        eigen_signs = sorted(set(dominant_signs))
        d6_value = coefficients["paper_d6"]
        dominant_value = coefficients["dominant_eigenphase"]
        hd_value = coefficients["hd_surrogate"]
        rows[str(ld)] = {
            "ld": int(ld),
            "exact_rte_lambda_r": float(
                payload["partial_s2"]["exact_rte_lambda_r"]
            ),
            "coefficients": coefficients,
            "hd_surrogate_original_window_coefficient": (
                original_hd_coefficient
            ),
            "hd_surrogate_original_fit_window": list(
                payload["request"]["surrogate_calibration_times"]
            ),
            "hd_surrogate_common_window_refit": (
                None if common_fit is None else dict(common_fit)
            ),
            "hd_surrogate_relative_difference_vs_paper_d6": (
                None if d6_value == 0.0 else hd_value / d6_value - 1.0
            ),
            "paper_d6_relative_difference_vs_dominant_eigenphase": (
                None
                if dominant_value == 0.0
                else d6_value / dominant_value - 1.0
            ),
            "coefficient_range_relative_to_paper_d6": (
                None
                if d6_value == 0.0
                else (
                    max(coefficients.values()) - min(coefficients.values())
                )
                / d6_value
            ),
            "paper_d6_conditioned_deltas": conditioned_deltas,
            "paper_d6_ill_conditioned_deltas": ill_conditioned_deltas,
            "paper_d6_estimator_validation_pass": bool(
                summary["paper_d6_estimator_validation_pass"]
            ),
            "single_dominant_phase_cost_model_validation_pass": bool(
                summary["single_dominant_phase_cost_model_validation_pass"]
            ),
            "maximum_paper_d6_vs_dominant_point_relative_difference": float(
                summary[
                    "maximum_paper_d6_vs_dominant_eigenphase_relative_difference"
                ]
            ),
            "hd_surrogate_signed_bias_signs": hd_signs,
            "paper_d6_signed_bias_signs": d6_signs,
            "dominant_eigenphase_signed_bias_signs": eigen_signs,
            "signed_conventions_directly_aligned": d6_signs == eigen_signs,
            "signed_bias_interpretation": (
                "paper_D6_and_dominant_eigenphase_sign_conventions_are_not_"
                "directly_aligned;_coefficient_comparisons_use_magnitudes"
                if d6_signs != eigen_signs
                else "signed_conventions_match_on_the_executed_grid"
            ),
            "commutator_term_decomposition_available": False,
            "coefficient_gap_interpretation": (
                "C_D_to_full_partial_gap_combines_omitted_random_tail_and_"
                "mixed_commutator_effects;_this_artifact_does_not_identify_"
                "individual_commutator_contributions_or_cancellation"
            ),
            "validation_fingerprint": payload["validation_fingerprint"],
        }
    if len(hashes) != 1:
        raise ValueError("All WP03 PF payloads must use one Hamiltonian hash.")
    return {
        "hamiltonian_hash": next(iter(hashes)),
        "coefficient_policies": list(COEFFICIENT_POLICIES),
        "common_delta_window_comparison": common_window_hd_fits is not None,
        "common_delta_window": (
            None
            if common_window_hd_fits is None
            else list(
                next(iter(common_window_hd_fits.values()))["delta_values"]
            )
        ),
        "rows": rows,
        "all_paper_d6_estimators_pass": all(
            row["paper_d6_estimator_validation_pass"] for row in rows.values()
        ),
        "all_single_phase_cost_models_pass": all(
            row["single_dominant_phase_cost_model_validation_pass"]
            for row in rows.values()
        ),
        "all_signed_conventions_directly_aligned": all(
            row["signed_conventions_directly_aligned"]
            for row in rows.values()
        ),
    }


def _intervals_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) <= min(
        float(left[1]), float(right[1])
    )


def _relative_gap_interval(
    candidate: Sequence[float], reference: Sequence[float]
) -> list[float]:
    reference_low = float(reference[0])
    reference_high = float(reference[1])
    if reference_low <= 0.0:
        raise ValueError("Reference interval must be positive.")
    return [
        float(candidate[0]) / reference_high - 1.0,
        float(candidate[1]) / reference_low - 1.0,
    ]


def _compact_scenario(
    scenario: Mapping[str, Any],
    *,
    policy: str,
    coefficient: float,
    delta_time: float,
    maximum_round_index: int,
    q_max: int,
    beta_pf_budget: float,
) -> dict[str, Any]:
    threshold = float(beta_pf_budget / (q_max * delta_time**3))
    rounds = [
        {
            "round_index": int(row["round_index"]),
            "q_m": int(row["q_m"]),
            "r_m": int(row["r_m"]),
            "K_m": int(row["K_m"]),
            "empirical_pf_phase_proxy": float(
                row["empirical_pf_phase_proxy"]
            ),
            "compiled_rz_point_estimate": float(
                row["compiled_rz_point_estimate"]
            ),
        }
        for row in scenario["rounds"]
    ]
    return {
        "coefficient_policy": policy,
        "ld": int(scenario["ld"]),
        "delta_time": float(delta_time),
        "candidate_id": f"ld{scenario['ld']}_delta{delta_time:g}",
        "pf_coefficient": float(coefficient),
        "pf_coefficient_is_rigorous_bound": False,
        "maximum_round_index_M": int(maximum_round_index),
        "q_max": int(q_max),
        "all_rounds_feasible": bool(scenario["all_rounds_feasible"]),
        "beta_pf_budget": float(beta_pf_budget),
        "maximum_empirical_pf_phase_proxy": float(
            scenario["maximum_empirical_pf_phase_proxy"]
        ),
        "pf_phase_budget_slack": float(
            beta_pf_budget - scenario["maximum_empirical_pf_phase_proxy"]
        ),
        "maximum_feasible_pf_coefficient_at_q_max": threshold,
        "relative_coefficient_increase_to_pf_budget_boundary": (
            None if coefficient == 0.0 else threshold / coefficient - 1.0
        ),
        "total_compiled_rz_point_estimate": float(
            scenario["total_compiled_rz_point_estimate"]
        ),
        "conservative_calibration_95_half_width": float(
            scenario["conservative_calibration_95_half_width"]
        ),
        "scenario_intervals": scenario["scenario_intervals"],
        "total_shots": int(scenario["total_shots"]),
        "selected_r_k_pairs": list(scenario["selected_r_k_pairs"]),
        "minimum_conservative_radius_lower_bound": float(
            scenario["minimum_conservative_radius_lower_bound"]
        ),
        "final_round_cost_fraction": float(
            scenario["final_round_cost_fraction"]
        ),
        "last_three_round_cost_fraction": float(
            scenario["last_three_round_cost_fraction"]
        ),
        "integer_shot_rounding_relative_overhead": float(
            scenario["integer_shot_rounding_relative_overhead"]
        ),
        "rounds": rounds,
    }


def evaluate_wp03_sensitivity(
    preparations: Mapping[int, DFPartialS2Preparation],
    *,
    cost_models: Mapping[int, Sequence[PairCompiledCostModel]],
    coefficient_audit: Mapping[str, Any],
    delta_values: Sequence[float] = (0.01, 0.0125, 0.02),
    target_energy_precision: float = 1.6e-4,
    beta_rpe: float = 0.4,
    beta_pf_budget: float = 0.015,
    alpha_total: float = 0.05,
    rte_seed: int = 20260818,
) -> dict[str, Any]:
    """Evaluate candidate selection while changing only the PF coefficient."""
    costed_ld_values = tuple(sorted(preparations))
    if costed_ld_values != (3, 12):
        raise ValueError("WP03 cost sensitivity requires L_D=3 and L_D=12.")
    if any(delta <= 0.0 for delta in delta_values):
        raise ValueError("delta_values must be positive.")
    rows = coefficient_audit["rows"]
    scenarios: list[dict[str, Any]] = []
    for policy in COEFFICIENT_POLICIES:
        for ld in costed_ld_values:
            coefficient = float(rows[str(ld)]["coefficients"][policy])
            beta_profile = (
                (beta_pf_budget, 0.005, beta_rpe - beta_pf_budget - 0.005)
                if ld == 3
                else (beta_pf_budget, 0.0, beta_rpe - beta_pf_budget)
            )
            for delta in delta_values:
                horizon = required_rpe_round_horizon(
                    target_energy_precision=target_energy_precision,
                    beta_rpe=beta_rpe,
                    delta_time=float(delta),
                )
                maximum_round_index = int(
                    horizon["maximum_round_index_M"]
                )
                q_max = int(horizon["q_max"])
                scenario = evaluate_ablation_scenario(
                    preparations[ld],
                    ld=ld,
                    pf_coefficient=coefficient,
                    cost_models=cost_models[ld],
                    beta_profile_label=(
                        "wp04_rebalanced_common"
                        if ld == 3
                        else "wp04_deterministic_rebalanced"
                    ),
                    beta_profile=beta_profile,
                    schedule_policy="round_compiled_rz",
                    alpha_policy="cost_sensitivity_weighted",
                    maximum_round_index=maximum_round_index,
                    delta_time=float(delta),
                    beta_rpe=beta_rpe,
                    alpha_total=alpha_total,
                    rte_seed=rte_seed,
                )
                scenarios.append(
                    _compact_scenario(
                        scenario,
                        policy=policy,
                        coefficient=coefficient,
                        delta_time=float(delta),
                        maximum_round_index=maximum_round_index,
                        q_max=q_max,
                        beta_pf_budget=beta_pf_budget,
                    )
                )

    selections: dict[str, dict[str, Any]] = {}
    candidate_regret: dict[str, list[dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for policy in COEFFICIENT_POLICIES:
        policy_rows = [
            row for row in scenarios if row["coefficient_policy"] == policy
        ]
        best = min(
            policy_rows,
            key=lambda row: (
                float(row["total_compiled_rz_point_estimate"]),
                int(row["ld"]),
                float(row["delta_time"]),
            ),
        )
        best_cost = float(best["total_compiled_rz_point_estimate"])
        selections[policy] = {
            "candidate_id": best["candidate_id"],
            "ld": int(best["ld"]),
            "delta_time": float(best["delta_time"]),
            "total_compiled_rz_point_estimate": best_cost,
        }
        candidate_regret[policy] = [
            {
                "candidate_id": row["candidate_id"],
                "ld": int(row["ld"]),
                "delta_time": float(row["delta_time"]),
                "total_compiled_rz_point_estimate": float(
                    row["total_compiled_rz_point_estimate"]
                ),
                "relative_regret_vs_policy_best": float(
                    row["total_compiled_rz_point_estimate"] / best_cost - 1.0
                ),
            }
            for row in sorted(
                policy_rows,
                key=lambda row: (int(row["ld"]), float(row["delta_time"])),
            )
        ]

        best_by_ld = {
            ld: min(
                (row for row in policy_rows if int(row["ld"]) == ld),
                key=lambda row: float(row["total_compiled_rz_point_estimate"]),
            )
            for ld in costed_ld_values
        }
        ld3 = best_by_ld[3]
        ld12 = best_by_ld[12]
        ld3_cost = float(ld3["total_compiled_rz_point_estimate"])
        ld12_cost = float(ld12["total_compiled_rz_point_estimate"])
        comparisons[policy] = {
            "ld3_best_delta": float(ld3["delta_time"]),
            "ld12_best_delta": float(ld12["delta_time"]),
            "ld3_relative_regret_vs_ld12": ld3_cost / ld12_cost - 1.0,
            "local_5_percent_relative_gap_interval": _relative_gap_interval(
                ld3["scenario_intervals"][
                    "local_5_percent_plus_calibration"
                ],
                ld12["scenario_intervals"][
                    "local_5_percent_plus_calibration"
                ],
            ),
            "transfer_25_percent_relative_gap_interval": (
                _relative_gap_interval(
                    ld3["scenario_intervals"][
                        "transfer_25_percent_plus_calibration"
                    ],
                    ld12["scenario_intervals"][
                        "transfer_25_percent_plus_calibration"
                    ],
                )
            ),
            "local_5_percent_intervals_overlap": _intervals_overlap(
                ld3["scenario_intervals"][
                    "local_5_percent_plus_calibration"
                ],
                ld12["scenario_intervals"][
                    "local_5_percent_plus_calibration"
                ],
            ),
            "transfer_25_percent_intervals_overlap": _intervals_overlap(
                ld3["scenario_intervals"][
                    "transfer_25_percent_plus_calibration"
                ],
                ld12["scenario_intervals"][
                    "transfer_25_percent_plus_calibration"
                ],
            ),
        }

    cross_policy_regret = []
    for selection_policy, selection in selections.items():
        for evaluation_policy in COEFFICIENT_POLICIES:
            evaluation_rows = [
                row
                for row in scenarios
                if row["coefficient_policy"] == evaluation_policy
            ]
            chosen = next(
                row
                for row in evaluation_rows
                if row["candidate_id"] == selection["candidate_id"]
            )
            best_cost = min(
                float(row["total_compiled_rz_point_estimate"])
                for row in evaluation_rows
            )
            cross_policy_regret.append(
                {
                    "selection_policy": selection_policy,
                    "evaluation_policy": evaluation_policy,
                    "selected_candidate_id": selection["candidate_id"],
                    "relative_regret": float(
                        chosen["total_compiled_rz_point_estimate"]
                        / best_cost
                        - 1.0
                    ),
                }
            )

    selected_ids = {item["candidate_id"] for item in selections.values()}
    d6_selected = selections["paper_d6"]["candidate_id"]
    d6_selected_rows = [
        row
        for row in scenarios
        if row["coefficient_policy"] == "paper_d6"
        and row["candidate_id"] == d6_selected
    ]
    if len(d6_selected_rows) != 1:
        raise RuntimeError("The paper-D6 selected candidate is not unique.")
    d6_selected_row = d6_selected_rows[0]
    d6_headroom = float(
        d6_selected_row["relative_coefficient_increase_to_pf_budget_boundary"]
    )

    checks = {
        "all_expected_scenarios_present": len(scenarios)
        == len(COEFFICIENT_POLICIES) * len(costed_ld_values) * len(delta_values),
        "all_scenarios_feasible": all(
            bool(row["all_rounds_feasible"]) for row in scenarios
        ),
        "all_pf_phase_budgets_satisfied": all(
            float(row["pf_phase_budget_slack"]) >= -1e-15
            for row in scenarios
        ),
        "all_costs_positive": all(
            float(row["total_compiled_rz_point_estimate"]) > 0.0
            for row in scenarios
        ),
        "all_paper_d6_estimators_pass": bool(
            coefficient_audit["all_paper_d6_estimators_pass"]
        ),
        "all_single_phase_cost_models_pass": bool(
            coefficient_audit["all_single_phase_cost_models_pass"]
        ),
        "cross_policy_regrets_non_negative": all(
            float(row["relative_regret"]) >= -1e-15
            for row in cross_policy_regret
        ),
        "final_total_cost_evaluation_not_claimed": True,
    }
    return {
        "scope": {
            "comparison_task": "WP00 fixed H4 CA_over_10 task",
            "cost_scope": "WP04 no-state-preparation Hadamard RZ projection",
            "changed_input": "PF coefficient only",
            "fixed_schedule_policy": "round_compiled_rz",
            "fixed_alpha_policy": "cost_sensitivity_weighted",
            "q1_q2_direct_calibration_reused": True,
            "q_greater_than_2_cost_method": "axis_affine_q1_q2_extrapolation",
            "unused_long_q_holdout_available": False,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "final_total_cost_evaluation_performed": False,
            "decision_grade": False,
        },
        "configuration": {
            "coefficient_policies": list(COEFFICIENT_POLICIES),
            "coefficient_audit_ld_values": sorted(
                int(ld) for ld in coefficient_audit["rows"]
            ),
            "costed_ld_values": list(costed_ld_values),
            "delta_values": [float(value) for value in delta_values],
            "target_energy_precision_ha": float(target_energy_precision),
            "beta_rpe": float(beta_rpe),
            "beta_pf_budget": float(beta_pf_budget),
            "alpha_total": float(alpha_total),
            "rte_seed": int(rte_seed),
        },
        "coefficient_audit": coefficient_audit,
        "scenarios": scenarios,
        "selections": selections,
        "candidate_regret": candidate_regret,
        "cross_policy_regret": cross_policy_regret,
        "ld3_vs_ld12": comparisons,
        "selection_sensitivity": {
            "selected_candidate_ids": sorted(selected_ids),
            "ld_and_delta_selection_invariant": len(selected_ids) == 1,
            "paper_d6_selected_candidate": d6_selected,
            "paper_d6_selected_relative_coefficient_increase_to_boundary": (
                d6_headroom
            ),
            "paper_d6_to_dominant_gap_is_smaller_than_boundary_headroom": (
                max(
                    abs(
                        float(
                            coefficient_audit["rows"][str(ld)][
                                "paper_d6_relative_difference_vs_"
                                "dominant_eigenphase"
                            ]
                        )
                    )
                    for ld in costed_ld_values
                )
                < d6_headroom
            ),
            "directional_result": (
                "coefficient_choice_does_not_resolve_wp04_interval_overlap"
                if all(
                    item["transfer_25_percent_intervals_overlap"]
                    for item in comparisons.values()
                )
                else "coefficient_choice_changes_interval_separation"
            ),
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
    }


def finalize_wp03_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP03",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp03_artifact(payload)
    return payload


def validate_wp03_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WP03 PF-sensitivity schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "WP03":
        raise ValueError("Unsupported WP03 PF-sensitivity method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP03 content_fingerprint mismatch.")
    scope = payload.get("scope", {})
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("WP03 cannot claim a final total-cost evaluation.")
    if scope.get("decision_grade") is not False:
        raise ValueError("WP03 must remain non-decision-grade.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP03 overall status does not match its checks.")


def write_wp03_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_wp03_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

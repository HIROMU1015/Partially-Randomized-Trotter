"""G08 round-wise cost, risk, and proxy-uncertainty dominance analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .research_direction_ablation import fingerprint
from .research_direction_decision_cost import validate_wp01d_compute_artifact
from .research_direction_decision_synthesis import (
    validate_wp01d_synthesis_artifact,
)


SCHEMA_VERSION = "research_direction_round_dominance_v1"
METHOD = "g08_round_cost_risk_proxy_dominance_v1"


def _dominant_round(
    rows: list[Mapping[str, Any]], metric: str
) -> Mapping[str, Any]:
    return max(rows, key=lambda row: float(row[metric]))


def _fraction(rows: list[Mapping[str, Any]], metric: str, count: int) -> float:
    total = sum(float(row[metric]) for row in rows)
    if total == 0.0:
        return 0.0
    return sum(float(row[metric]) for row in rows[-count:]) / total


def evaluate_g08_round_dominance(
    compute: Mapping[str, Any],
    synthesis: Mapping[str, Any],
) -> dict[str, Any]:
    """Locate the rounds that dominate cost, model uncertainty, and error risk."""
    validate_wp01d_compute_artifact(compute)
    validate_wp01d_synthesis_artifact(synthesis)
    if synthesis["input_compute_fingerprint"] != compute["content_fingerprint"]:
        raise ValueError("G08 compute and synthesis inputs do not match.")
    if not compute["overall_pass"] or not synthesis["overall_pass"]:
        raise ValueError("G08 requires passing WP01-D/C07 inputs.")

    candidates: dict[str, Any] = {}
    for ld in ("3", "12"):
        selected = compute["best_by_ld"][ld]
        best = selected["best"]
        rows = list(best["rounds"])
        cost_round = _dominant_round(rows, "compiled_rz_point_estimate")
        pf_round = _dominant_round(rows, "empirical_pf_phase_proxy")
        rte_round = _dominant_round(rows, "finite_rte_phase_bound")
        calibration_round = _dominant_round(
            rows, "calibration_standard_error_conservative_sum"
        )
        calibration_sum = sum(
            float(row["calibration_standard_error_conservative_sum"])
            for row in rows
        )
        late_calibration_sum = sum(
            float(row["calibration_standard_error_conservative_sum"])
            for row in rows[-3:]
        )
        row_records = []
        total_cost = float(best["total_compiled_rz_point_estimate"])
        for row in rows:
            row_records.append(
                {
                    "round_index": int(row["round_index"]),
                    "q_m": int(row["q_m"]),
                    "r_m": int(row["r_m"]),
                    "K_m": int(row["K_m"]),
                    "compiled_rz_point_estimate": float(
                        row["compiled_rz_point_estimate"]
                    ),
                    "cost_fraction": (
                        float(row["compiled_rz_point_estimate"]) / total_cost
                    ),
                    "calibration_standard_error_conservative_sum": float(
                        row["calibration_standard_error_conservative_sum"]
                    ),
                    "calibration_uncertainty_fraction": (
                        0.0
                        if calibration_sum == 0.0
                        else float(
                            row[
                                "calibration_standard_error_conservative_sum"
                            ]
                        )
                        / calibration_sum
                    ),
                    "empirical_pf_phase_proxy": float(
                        row["empirical_pf_phase_proxy"]
                    ),
                    "finite_rte_phase_bound": float(
                        row["finite_rte_phase_bound"]
                    ),
                    "epsilon_coordinate": float(row["epsilon_coordinate"]),
                }
            )
        candidates[ld] = {
            "delta_time": float(selected["delta_time"]),
            "maximum_round_index": int(selected["maximum_round_index_M"]),
            "q_max": int(selected["q_max"]),
            "total_compiled_rz_point_estimate": total_cost,
            "last_round_cost_fraction": float(best["final_round_cost_fraction"]),
            "last_three_round_cost_fraction": _fraction(
                rows, "compiled_rz_point_estimate", 3
            ),
            "last_three_round_calibration_uncertainty_fraction": _fraction(
                rows, "calibration_standard_error_conservative_sum", 3
            ),
            "dominant_cost_round": int(cost_round["round_index"]),
            "dominant_pf_risk_round": int(pf_round["round_index"]),
            "dominant_rte_risk_round": int(rte_round["round_index"]),
            "dominant_calibration_uncertainty_round": int(
                calibration_round["round_index"]
            ),
            "pf_budget_usage_at_maximum": (
                float(pf_round["empirical_pf_phase_proxy"])
                / float(best["beta_pf_budget"])
            ),
            "rte_budget_usage_at_maximum": (
                0.0
                if float(best["beta_rte_budget"]) == 0.0
                else float(rte_round["finite_rte_phase_bound"])
                / float(best["beta_rte_budget"])
            ),
            "calibration_95_half_width": float(
                best["conservative_calibration_95_half_width"]
            ),
            "counterfactual_calibration_95_half_width_if_late_three_exact": (
                1.96 * (calibration_sum - late_calibration_sum)
            ),
            "counterfactual_calibration_95_half_width_if_early_rounds_exact": (
                1.96 * late_calibration_sum
            ),
            "rounds": row_records,
        }

    ld3 = candidates["3"]
    ld12 = candidates["12"]
    threshold = float(
        synthesis["comparison"][
            "maximum_symmetric_model_discrepancy_for_interval_separation"
        ]
    )
    checks = {
        "wp01d_compute_and_synthesis_pass": True,
        "last_three_rounds_hold_at_least_80_percent_cost_both_candidates": (
            float(ld3["last_three_round_cost_fraction"]) >= 0.8
            and float(ld12["last_three_round_cost_fraction"]) >= 0.8
        ),
        "last_three_rounds_hold_at_least_80_percent_ld3_calibration_uncertainty": (
            float(
                ld3["last_three_round_calibration_uncertainty_fraction"]
            )
            >= 0.8
        ),
        "final_round_dominates_cost_both_candidates": (
            int(ld3["dominant_cost_round"]) == 17
            and int(ld12["dominant_cost_round"]) == 17
        ),
        "ld3_cost_and_rte_risk_rounds_are_distinct": (
            int(ld3["dominant_cost_round"])
            != int(ld3["dominant_rte_risk_round"])
        ),
        "m08_target_is_late_round_r32_proxy": True,
        "final_total_cost_not_claimed": True,
    }
    return {
        "input_fingerprints": {
            "compute": compute["content_fingerprint"],
            "synthesis": synthesis["content_fingerprint"],
        },
        "candidates": candidates,
        "comparison": {
            "local_interval_separation_model_discrepancy_limit": threshold,
            "adopted_local_model_discrepancy": float(
                synthesis["comparison"]["adopted_local_model_discrepancy"]
            ),
            "remaining_model_discrepancy_margin": float(
                synthesis["comparison"][
                    "separation_margin_in_model_discrepancy"
                ]
            ),
            "maximum_cost_round_same_for_both_candidates": (
                int(ld3["dominant_cost_round"])
                == int(ld12["dominant_cost_round"])
            ),
            "maximum_cost_and_failure_risk_round_same_for_ld3": (
                int(ld3["dominant_cost_round"])
                == int(ld3["dominant_rte_risk_round"])
            ),
        },
        "m08_target": {
            "delta_time": 0.02,
            "rte_steps": [32],
            "direct_holdout_q_values": [16, 32],
            "pilot_trajectories_per_q": 8,
            "calibration_q_values": [1, 2],
            "diagnostic_q_values_already_available": [4, 8],
            "accuracy_threshold": 0.05,
            "separation_limit": threshold,
            "reason": (
                "The last three rounds use r=32 and dominate both total RZ "
                "cost and propagated calibration uncertainty. q=16 and q=32 "
                "extend the direct domain before considering q=64."
            ),
        },
        "decision": {
            "g08_result": "late_three_rounds_dominate_cost_and_proxy_uncertainty",
            "m08_should_run": True,
            "m08_scope": "delta_0p02_r32_q16_q32_fresh_pilot",
            "robust_directional_result_remains": (
                "undetermined_under_transfer_sensitivity"
            ),
        },
        "scope": {
            "existing_artifacts_only": True,
            "new_circuit_compilation_performed": False,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "final_total_cost_evaluation_performed": False,
        },
        "limitations": [
            "The round decomposition inherits the WP01-D affine proxy beyond directly validated q=8.",
            "The RTE-risk maximum and cost maximum are different rounds for L_D=3.",
            "G08 selects where to validate; it does not itself validate q>8 proxy accuracy.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "ld3_last_three_round_cost_fraction": float(
                ld3["last_three_round_cost_fraction"]
            ),
            "ld12_last_three_round_cost_fraction": float(
                ld12["last_three_round_cost_fraction"]
            ),
            "ld3_last_three_round_calibration_uncertainty_fraction": float(
                ld3["last_three_round_calibration_uncertainty_fraction"]
            ),
            "ld3_dominant_cost_round": int(ld3["dominant_cost_round"]),
            "ld3_dominant_rte_risk_round": int(
                ld3["dominant_rte_risk_round"]
            ),
            "next_action": "M08_delta0p02_r32_q16_q32_fresh_holdout",
        },
    }


def finalize_g08_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "G08",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_g08_artifact(payload)
    return payload


def validate_g08_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported G08 schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "G08":
        raise ValueError("Unsupported G08 method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("G08 content_fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("G08 overall status does not match checks.")
    scope = payload.get("scope", {})
    if scope.get("new_circuit_compilation_performed") is not False:
        raise ValueError("G08 must use existing artifacts only.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("G08 cannot claim final total cost.")


def write_g08_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_g08_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

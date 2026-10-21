"""WP01-D/C07 synthesis and claim review for the optimized cost comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .research_direction_ablation import fingerprint
from .research_direction_decision_cost import validate_wp01d_compute_artifact


SCHEMA_VERSION = "research_direction_decision_synthesis_v1"
METHOD = "wp01d_c07_conditional_interval_synthesis_v1"


def _overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) <= min(
        float(left[1]), float(right[1])
    )


def evaluate_wp01d_c07_synthesis(
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate local conditional preference from robust scientific claims."""
    validate_wp01d_compute_artifact(compute)
    if not compute["overall_pass"]:
        raise ValueError("WP01-D/C07 compute artifact did not pass.")
    ld3_row = compute["best_by_ld"]["3"]
    ld12_row = compute["best_by_ld"]["12"]
    ld3 = ld3_row["best"]
    ld12 = ld12_row["best"]
    cost3 = float(ld3["total_compiled_rz_point_estimate"])
    cost12 = float(ld12["total_compiled_rz_point_estimate"])
    calibration3 = float(ld3["conservative_calibration_95_half_width"])
    calibration12 = float(ld12["conservative_calibration_95_half_width"])
    local3 = ld3["scenario_intervals"]["local_5_percent_plus_calibration"]
    local12 = ld12["scenario_intervals"]["local_5_percent_plus_calibration"]
    transfer3 = ld3["scenario_intervals"][
        "transfer_25_percent_plus_calibration"
    ]
    transfer12 = ld12["scenario_intervals"][
        "transfer_25_percent_plus_calibration"
    ]
    local_overlap = _overlap(local3, local12)
    transfer_overlap = _overlap(transfer3, transfer12)
    maximum_symmetric_discrepancy_for_separation = (
        cost12 - cost3 - calibration3 - calibration12
    ) / (cost3 + cost12)
    local_interval_gap = float(local12[0]) - float(local3[1])
    checks = {
        "compute_artifact_passed": bool(compute["overall_pass"]),
        "both_candidates_select_delta_0p02": (
            float(ld3_row["delta_time"]) == 0.02
            and float(ld12_row["delta_time"]) == 0.02
        ),
        "local_5_percent_intervals_separate": not local_overlap,
        "transfer_25_percent_intervals_overlap": transfer_overlap,
        "local_separation_margin_is_positive": local_interval_gap > 0.0,
        "robust_scientific_superiority_not_claimed": True,
    }
    return {
        "input_compute_fingerprint": compute["content_fingerprint"],
        "selected_candidates": {
            "3": {
                "delta_time": float(ld3_row["delta_time"]),
                "total_compiled_rz_point_estimate": cost3,
                "total_shots": int(ld3["total_shots"]),
                "local_5_percent_plus_calibration_interval": list(local3),
                "transfer_25_percent_plus_calibration_interval": list(
                    transfer3
                ),
                "last_three_round_cost_fraction": float(
                    ld3["last_three_round_cost_fraction"]
                ),
            },
            "12": {
                "delta_time": float(ld12_row["delta_time"]),
                "total_compiled_rz_point_estimate": cost12,
                "total_shots": int(ld12["total_shots"]),
                "local_5_percent_plus_calibration_interval": list(local12),
                "transfer_25_percent_plus_calibration_interval": list(
                    transfer12
                ),
                "last_three_round_cost_fraction": float(
                    ld12["last_three_round_cost_fraction"]
                ),
            },
        },
        "comparison": {
            "point_preference": "L_D=3",
            "ld3_over_ld12_point_estimate_ratio": cost3 / cost12,
            "ld3_point_reduction_relative_to_ld12": (cost12 - cost3) / cost12,
            "local_5_percent_intervals_overlap": local_overlap,
            "local_interval_separation_gap_rz": local_interval_gap,
            "local_interval_separation_gap_over_ld12_point": (
                local_interval_gap / cost12
            ),
            "transfer_25_percent_intervals_overlap": transfer_overlap,
            "maximum_symmetric_model_discrepancy_for_interval_separation": (
                maximum_symmetric_discrepancy_for_separation
            ),
            "adopted_local_model_discrepancy": 0.05,
            "separation_margin_in_model_discrepancy": (
                maximum_symmetric_discrepancy_for_separation - 0.05
            ),
        },
        "decision": {
            "local_model_conditional_result": (
                "L_D=3_interval_is_lower_under_5_percent_discrepancy"
            ),
            "robust_directional_result": (
                "undetermined_under_transfer_sensitivity"
            ),
            "partial_randomization_scientific_superiority_established": False,
            "next_action": (
                "M08_G08_quantify_dominant_late_round_proxy_precision"
            ),
        },
        "scope": {
            "h4_ca_over_10_no_state_preparation_comparison": True,
            "full_controlled_hadamard_cost_provider_used": True,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "noise_included": False,
            "external_reproduction_included": False,
            "final_total_cost_evaluation_performed": False,
            "decision_grade_scientific_superiority_claimed": False,
        },
        "limitations": [
            "The local interval separation depends on a 5% discrepancy allowance and has only a 0.048 percentage-point discrepancy margin.",
            "The 25% transfer-sensitivity intervals overlap, so the direction is not robust to the documented extrapolation scenario.",
            "The comparison is H4-only, no-state-preparation, one compiler context, and uses affine q extrapolation beyond direct q=8 validation.",
            "State preparation, backend noise, fault-tolerant synthesis, and external reproduction remain outside scope.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "point_preference": "L_D=3",
            "ld3_point_reduction_relative_to_ld12": (cost12 - cost3) / cost12,
            "local_5_percent_intervals_overlap": local_overlap,
            "transfer_25_percent_intervals_overlap": transfer_overlap,
            "maximum_symmetric_model_discrepancy_for_interval_separation": (
                maximum_symmetric_discrepancy_for_separation
            ),
            "robust_directional_result": (
                "undetermined_under_transfer_sensitivity"
            ),
            "next_action": (
                "M08_G08_quantify_dominant_late_round_proxy_precision"
            ),
        },
    }


def finalize_wp01d_synthesis_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP01-D/C07-synthesis",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp01d_synthesis_artifact(payload)
    return payload


def validate_wp01d_synthesis_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WP01-D/C07 synthesis schema.")
    if payload.get("method") != METHOD:
        raise ValueError("Unsupported WP01-D/C07 synthesis method.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP01-D/C07 synthesis fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP01-D/C07 synthesis status does not match checks.")
    decision = payload.get("decision", {})
    if decision.get("partial_randomization_scientific_superiority_established"):
        raise ValueError("Synthesis cannot claim robust superiority.")
    scope = payload.get("scope", {})
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("Synthesis cannot claim final total cost.")


def write_wp01d_synthesis_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_wp01d_synthesis_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

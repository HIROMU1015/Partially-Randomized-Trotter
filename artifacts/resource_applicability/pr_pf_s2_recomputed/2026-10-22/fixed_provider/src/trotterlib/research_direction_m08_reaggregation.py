"""Reaggregate WP01-D/C07 intervals with measured M08 q<=32 discrepancy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .research_direction_ablation import fingerprint
from .research_direction_decision_cost import validate_wp01d_compute_artifact
from .research_direction_decision_synthesis import (
    validate_wp01d_synthesis_artifact,
)
from .research_direction_proxy_precision import validate_m08_artifact
from .research_direction_round_dominance import validate_g08_artifact


SCHEMA_VERSION = "research_direction_m08_reaggregation_v1"
METHOD = "wp01d_c07_m08_measured_discrepancy_reaggregation_v1"


def _interval(
    cost: float, calibration_half_width: float, discrepancy: float
) -> list[float]:
    half_width = discrepancy * cost + calibration_half_width
    return [cost - half_width, cost + half_width]


def _overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) <= min(
        float(left[1]), float(right[1])
    )


def evaluate_m08_reaggregation(
    compute: Mapping[str, Any],
    synthesis: Mapping[str, Any],
    g08: Mapping[str, Any],
    m08: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild comparison intervals without extending M08 beyond q=32."""
    validate_wp01d_compute_artifact(compute)
    validate_wp01d_synthesis_artifact(synthesis)
    validate_g08_artifact(g08)
    validate_m08_artifact(m08)
    if synthesis["input_compute_fingerprint"] != compute["content_fingerprint"]:
        raise ValueError("Synthesis does not bind the supplied compute artifact.")
    if g08["input_fingerprints"]["compute"] != compute["content_fingerprint"]:
        raise ValueError("G08 does not bind the supplied compute artifact.")
    if m08["source_evidence"]["g08"]["content_fingerprint"] != (
        g08["content_fingerprint"]
    ):
        raise ValueError("M08 does not bind the supplied G08 artifact.")
    if not all(
        bool(payload["overall_pass"])
        for payload in (compute, synthesis, g08, m08)
    ):
        raise ValueError("M08 reaggregation requires passing upstream artifacts.")

    ld3 = compute["best_by_ld"]["3"]["best"]
    ld12 = compute["best_by_ld"]["12"]["best"]
    costs = {
        "3": float(ld3["total_compiled_rz_point_estimate"]),
        "12": float(ld12["total_compiled_rz_point_estimate"]),
    }
    calibration = {
        "3": float(ld3["conservative_calibration_95_half_width"]),
        "12": float(ld12["conservative_calibration_95_half_width"]),
    }
    selected_rz = float(
        m08["summary"][
            "maximum_selected_policy_q16_q32_rz_relative_error"
        ]
    )
    selected_all = float(
        m08["summary"][
            "maximum_selected_policy_q16_q32_all_metric_relative_error"
        ]
    )
    full_rz = float(
        m08["summary"]["maximum_full_basis_q16_q32_rz_relative_error"]
    )
    max_observed_rz = max(selected_rz, full_rz)
    scenario_discrepancies = {
        "m08_selected_policy_rz_q_le_32": selected_rz,
        "m08_max_observed_rz_q_le_32": max_observed_rz,
        "local_5_percent": 0.05,
        "transfer_25_percent": 0.25,
    }
    scenarios: dict[str, Any] = {}
    for name, discrepancy in scenario_discrepancies.items():
        interval3 = _interval(costs["3"], calibration["3"], discrepancy)
        interval12 = _interval(costs["12"], calibration["12"], discrepancy)
        overlap = _overlap(interval3, interval12)
        gap = float(interval12[0]) - float(interval3[1])
        scenarios[name] = {
            "symmetric_model_discrepancy": discrepancy,
            "ld3_interval": interval3,
            "ld12_interval": interval12,
            "intervals_overlap": overlap,
            "separation_gap_rz": gap,
            "separation_gap_over_ld12_point": gap / costs["12"],
            "direct_evidence_domain": (
                "q_le_32"
                if name.startswith("m08_")
                else "scenario_assumption"
            ),
        }

    required = float(
        synthesis["comparison"][
            "maximum_symmetric_model_discrepancy_for_interval_separation"
        ]
    )
    checks = {
        "all_upstream_artifacts_pass_and_match": True,
        "m08_selected_rz_error_below_5_percent": selected_rz <= 0.05,
        "m08_all_metric_error_below_5_percent": selected_all <= 0.05,
        "m08_max_observed_rz_error_below_5_percent": max_observed_rz <= 0.05,
        "m08_selected_rz_scenario_intervals_separate": not bool(
            scenarios["m08_selected_policy_rz_q_le_32"]["intervals_overlap"]
        ),
        "m08_max_observed_rz_scenario_intervals_separate": not bool(
            scenarios["m08_max_observed_rz_q_le_32"]["intervals_overlap"]
        ),
        "local_5_percent_intervals_still_separate": not bool(
            scenarios["local_5_percent"]["intervals_overlap"]
        ),
        "transfer_25_percent_intervals_still_overlap": bool(
            scenarios["transfer_25_percent"]["intervals_overlap"]
        ),
        "m08_direct_domain_not_extended_beyond_q32": (
            int(m08["decision"]["validated_direct_q_maximum"]) == 32
        ),
        "robust_scientific_superiority_not_claimed": True,
    }
    return {
        "input_fingerprints": {
            "compute": compute["content_fingerprint"],
            "synthesis": synthesis["content_fingerprint"],
            "g08": g08["content_fingerprint"],
            "m08": m08["content_fingerprint"],
        },
        "point_estimates": {
            "3": {
                "compiled_rz": costs["3"],
                "calibration_95_half_width": calibration["3"],
                "delta_time": float(compute["best_by_ld"]["3"]["delta_time"]),
                "total_shots": int(ld3["total_shots"]),
            },
            "12": {
                "compiled_rz": costs["12"],
                "calibration_95_half_width": calibration["12"],
                "delta_time": float(compute["best_by_ld"]["12"]["delta_time"]),
                "total_shots": int(ld12["total_shots"]),
            },
        },
        "m08_measurements": {
            "selected_policy_q16_q32_rz_maximum_relative_error": selected_rz,
            "selected_policy_q16_q32_all_metric_maximum_relative_error": (
                selected_all
            ),
            "full_basis_q16_q32_rz_maximum_relative_error": full_rz,
            "maximum_observed_rz_relative_error": max_observed_rz,
            "validated_direct_q_maximum": 32,
            "schedule_q_maximum": int(m08["decision"]["schedule_q_maximum"]),
        },
        "scenarios": scenarios,
        "comparison": {
            "point_preference": "L_D=3",
            "ld3_point_reduction_relative_to_ld12": (
                costs["12"] - costs["3"]
            )
            / costs["12"],
            "maximum_symmetric_model_discrepancy_for_interval_separation": (
                required
            ),
            "m08_selected_rz_margin_to_separation_limit": (
                required - selected_rz
            ),
            "m08_max_observed_rz_margin_to_separation_limit": (
                required - max_observed_rz
            ),
        },
        "decision": {
            "local_q_le_32_measured_result": (
                "L_D=3_interval_is_lower_under_measured_m08_discrepancy"
            ),
            "local_5_percent_result": (
                "L_D=3_interval_is_lower_under_5_percent_discrepancy"
            ),
            "robust_directional_result": (
                "undetermined_under_transfer_sensitivity"
            ),
            "partial_randomization_scientific_superiority_established": False,
            "q64_followup_triggered_by_preregistered_m08_rule": False,
            "next_action": (
                "claim_review_or_external_transfer_validation_not_more_local_q_precision"
            ),
        },
        "scope": {
            "m08_measurement_used_only_for_q_le_32_scenario": True,
            "q_above_32_directly_validated": False,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "noise_included": False,
            "external_transfer_validated": False,
            "final_total_cost_evaluation_performed": False,
        },
        "limitations": [
            "The measured 2.47% and 3.29% discrepancies are direct evidence only through q=32.",
            "The selected schedule reaches q=131072, so the measured scenarios do not replace the 5% or 25% long-q scenarios.",
            "The point estimates are unchanged; this reaggregation updates decision intervals rather than rerunning circuit compilation or allocation optimization.",
            "The comparison remains H4-only, no-state-preparation, and one compiler context.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "m08_selected_policy_rz_discrepancy": selected_rz,
            "m08_max_observed_rz_discrepancy": max_observed_rz,
            "m08_selected_rz_intervals_overlap": bool(
                scenarios["m08_selected_policy_rz_q_le_32"][
                    "intervals_overlap"
                ]
            ),
            "m08_max_observed_rz_intervals_overlap": bool(
                scenarios["m08_max_observed_rz_q_le_32"][
                    "intervals_overlap"
                ]
            ),
            "local_5_percent_intervals_overlap": bool(
                scenarios["local_5_percent"]["intervals_overlap"]
            ),
            "transfer_25_percent_intervals_overlap": bool(
                scenarios["transfer_25_percent"]["intervals_overlap"]
            ),
            "robust_directional_result": (
                "undetermined_under_transfer_sensitivity"
            ),
        },
    }


def finalize_m08_reaggregation_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP01-D/C07-M08-reaggregation",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_m08_reaggregation_artifact(payload)
    return payload


def validate_m08_reaggregation_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported M08 reaggregation schema.")
    if payload.get("method") != METHOD:
        raise ValueError("Unsupported M08 reaggregation method.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("M08 reaggregation fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("M08 reaggregation status does not match checks.")
    decision = payload.get("decision", {})
    if decision.get("partial_randomization_scientific_superiority_established"):
        raise ValueError("M08 reaggregation cannot claim robust superiority.")
    scope = payload.get("scope", {})
    if scope.get("q_above_32_directly_validated") is not False:
        raise ValueError("M08 reaggregation cannot claim direct q>32 evidence.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("M08 reaggregation cannot claim final total cost.")


def write_m08_reaggregation_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_m08_reaggregation_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

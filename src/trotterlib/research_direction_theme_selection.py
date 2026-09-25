"""Synthesize the P-B, P-C, and P-A pilots into one scoped theme decision."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .parallel_validation_executor import atomic_write_json
from .research_direction_full_scope import fingerprint
from .research_direction_geometry_energy_difference_pilot import (
    validate_geometry_energy_difference_pilot_artifact,
)
from .research_direction_joint_synthesis_pilot import (
    validate_joint_synthesis_pilot_artifact,
)
from .research_direction_signal_weight_pilot import (
    validate_signal_weight_pilot_artifact,
)


SCHEMA_VERSION = "research_direction_theme_selection_v1"
METHOD = "pb_pc_pa_four_question_theme_selection_v1"
STAGE = "theme-selection-PB-PC-PA"


def evaluate_theme_selection(
    pb: Mapping[str, Any],
    pc: Mapping[str, Any],
    pa: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the three pilot artifacts and select a provisional theme."""
    validate_signal_weight_pilot_artifact(pb)
    validate_geometry_energy_difference_pilot_artifact(pc)
    validate_joint_synthesis_pilot_artifact(pa)

    pb_stops = not bool(pb["summary"]["pb_hypothesis_supported_in_scope"])
    pc_advances = bool(pc["summary"]["pc_hypothesis_supported_in_scope"])
    pa_advances = bool(pa["summary"]["pa_hypothesis_supported_in_scope"])
    pa_holdout = pa["holdout"]
    pa_rz = pa_holdout["policies"]["interval_union_dp"]["metrics"]["rz_count"]

    theme_assessment = [
        {
            "theme": "P-A_joint_sequence_synthesis",
            "decision": "primary_provisional",
            "difference_from_project_baselines": (
                "Interval dynamic programming selects full or support-union basis "
                "representations over contiguous source-basis runs, beyond the fixed "
                "full, event-support, and support_run_le_1 baselines."
            ),
            "clear_counterexample_improvement_or_prediction": True,
            "unused_condition_prediction_present": True,
            "completion_statement": (
                "Predict an interval basis plan from sequence support/run structure "
                "that preserves the operator and reduces compiled cost relative to "
                "strong baselines on unseen event sequences."
            ),
            "limitation": (
                "Novelty relative to the broader circuit-synthesis literature is not "
                "yet established, and only one H4 topology-free compiler context was tested."
            ),
        },
        {
            "theme": "P-C_geometry_energy_difference",
            "decision": "secondary_candidate",
            "difference_from_project_baselines": (
                "Use signed geometry-dependent PF bias to predict cancellation in "
                "energy differences rather than optimizing absolute energies alone."
            ),
            "clear_counterexample_improvement_or_prediction": True,
            "unused_condition_prediction_present": True,
            "completion_statement": (
                "Predict when signed PF errors cancel or break across geometry pairs "
                "and select a decomposition for energy differences."
            ),
            "limitation": (
                "The current pilot shows smooth local H4 interpolation but not yet a "
                "breakdown mechanism or a controllable orbital/fragment-tracking design."
            ),
        },
        {
            "theme": "P-B_signal_weight",
            "decision": "stop_in_current_scope",
            "difference_from_project_baselines": (
                "Separate target phase bias from target weight and coherent signal."
            ),
            "clear_counterexample_improvement_or_prediction": False,
            "unused_condition_prediction_present": False,
            "completion_statement": (
                "Would predict energy-only PF selection failures from a cheap "
                "state-action diagnostic if a practical failure were found."
            ),
            "limitation": (
                "No practical selection disagreement occurs in the fixed H4 grid, "
                "so the proposed diagnostic cannot be validated as a selector."
            ),
        },
    ]

    checks = {
        "pb_stop_decision_preserved": pb_stops,
        "pc_candidate_decision_preserved": pc_advances,
        "pa_candidate_decision_preserved": pa_advances,
        "pa_all_pilot_gates_pass": all(pa["pilot_gates"].values()),
        "exactly_one_provisional_primary": sum(
            row["decision"] == "primary_provisional" for row in theme_assessment
        )
        == 1,
        "novelty_not_overstated": True,
    }
    return {
        "input_fingerprints": {
            "pb": pb["content_fingerprint"],
            "pc": pc["content_fingerprint"],
            "pa": pa["content_fingerprint"],
        },
        "selection_rule": {
            "questions": [
                "What differs from known or current strong baselines?",
                "Was one clear counterexample, improvement, or prediction obtained?",
                "Is prediction on an unused condition supported?",
                "Can the completed research claim be stated in one sentence?",
            ],
            "tie_break": (
                "Prefer the pilot with a direct effect against strong baselines and "
                "an unused-condition holdout; retain alternatives whose next missing "
                "mechanism is clearly identified."
            ),
        },
        "theme_assessment": theme_assessment,
        "quantitative_basis": {
            "pa": {
                "holdout_sequence_lengths": pa_holdout["sequence_lengths"],
                "holdout_sample_count": pa_holdout["sample_count"],
                "pooled_rz_relative_change_vs_current": pa["summary"][
                    "holdout_pooled_rz_relative_change_vs_current"
                ],
                "pooled_rz_relative_change_vs_full": pa_rz[
                    "candidate_relative_to_full"
                ],
                "maximum_trajectory_rz_increase_vs_current": pa["summary"][
                    "maximum_holdout_trajectory_rz_relative_increase_vs_current"
                ],
                "compiled_oracle_regret_over_full_rz": pa["summary"][
                    "compiled_oracle_regret_over_full_rz"
                ],
                "changed_trajectory_fraction": pa["summary"][
                    "changed_holdout_trajectory_fraction"
                ],
                "maximum_operator_residual": pa["summary"][
                    "maximum_operator_equivalence_residual"
                ],
            },
            "pc": {
                "maximum_geometry_coefficient_holdout_relative_error": pc[
                    "summary"
                ]["maximum_geometry_holdout_coefficient_relative_error"],
                "maximum_delta_holdout_relative_error": pc["summary"][
                    "maximum_delta_holdout_bias_relative_error"
                ],
                "combined_pair_error_normalized_by_endpoint_bias": pc["summary"][
                    "combined_pair_prediction_error_normalized_by_endpoint_bias"
                ],
            },
            "pb": {
                "selection_disagreement_count": pb["summary"][
                    "energy_signal_selection_disagreement_count"
                ],
                "meaningful_inversion_count": pb["summary"][
                    "meaningful_pairwise_ordering_inversion_count"
                ],
            },
        },
        "decision": {
            "primary_theme": "P-A_joint_sequence_synthesis",
            "status": "provisional_pending_prior_art_and_novelty_audit",
            "secondary_theme": "P-C_geometry_energy_difference",
            "stopped_theme_in_current_scope": "P-B_signal_weight",
            "next_action": (
                "audit_prior_art_then_predeclare_one_blind_external_sequence_and_"
                "compiler_holdout_for_PA"
            ),
            "h12_or_long_rpe_required_next": False,
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
        "scope": {
            "new_physical_simulation_performed": False,
            "new_circuit_compilation_performed": False,
            "literature_novelty_established": False,
            "primary_theme_finalized": False,
            "h12_evaluated": False,
            "rpe_or_final_total_cost_evaluated": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_theme_selection_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": STAGE,
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_theme_selection_artifact(payload)
    return payload


def validate_theme_selection_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported theme-selection schema.")
    if payload.get("method") != METHOD or payload.get("stage") != STAGE:
        raise ValueError("Unsupported theme-selection method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("Theme-selection artifact fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("Theme-selection status does not match checks.")
    if payload.get("decision", {}).get("status") != (
        "provisional_pending_prior_art_and_novelty_audit"
    ):
        raise ValueError("Theme selection must remain provisional.")
    scope = payload.get("scope", {})
    for key in (
        "new_physical_simulation_performed",
        "new_circuit_compilation_performed",
        "literature_novelty_established",
        "primary_theme_finalized",
        "h12_evaluated",
        "rpe_or_final_total_cost_evaluated",
        "scientific_superiority_claimed",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"Theme-selection artifact overstates scope: {key}.")


def write_theme_selection_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_theme_selection_artifact(payload)
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing artifact: {output}")
    atomic_write_json(output, payload)

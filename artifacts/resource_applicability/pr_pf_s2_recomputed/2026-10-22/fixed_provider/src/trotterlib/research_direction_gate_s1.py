"""Gate-S1 synthesis for the research-direction screening work packages.

This module does not run a new physical simulation.  It validates the
fingerprinted WP00, WP02, WP01-S, WP04, and WP03 artifacts and turns their
documented results into a tamper-evident research-direction decision record.
The record chooses the next validation; it is not a final cost or superiority
result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .research_direction_ablation import validate_wp04_artifact
from .research_direction_pf_sensitivity import validate_wp03_artifact
from .research_direction_prevalidation import validate_artifact


SCHEMA_VERSION = "research_direction_gate_s1_v1"
METHOD = "wp00_wp02_wp01s_wp04_wp03_direction_synthesis_v1"


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _loo_increase(wp04: Mapping[str, Any], ld: int, factor: str) -> float:
    matches = [
        row
        for row in wp04["leave_one_out"][str(ld)]
        if row["removed_factor"] == factor
    ]
    _require(len(matches) == 1, f"WP04 L_D={ld} factor {factor!r} is missing.")
    return float(matches[0]["relative_cost_increase_when_removed"])


def build_gate_s1_body(
    *,
    wp00: Mapping[str, Any],
    wp02: Mapping[str, Any],
    wp01s: Mapping[str, Any],
    wp04: Mapping[str, Any],
    wp03: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate upstream artifacts and build the Gate-S1 decision body."""
    validate_artifact(wp00)
    validate_artifact(wp02)
    validate_artifact(wp01s)
    validate_wp04_artifact(wp04)
    validate_wp03_artifact(wp03)

    _require(wp00["stage"] == "WP00", "The WP00 artifact has the wrong stage.")
    _require(wp02["stage"] == "WP02", "The WP02 artifact has the wrong stage.")
    _require(
        wp01s["stage"] == "WP01-S",
        "The WP01-S artifact has the wrong stage.",
    )
    _require(bool(wp00["overall_pass"]), "WP00 did not pass.")
    _require(bool(wp02["overall_pass"]), "WP02 did not pass.")
    _require(bool(wp04["overall_pass"]), "WP04 did not pass.")
    _require(bool(wp03["overall_pass"]), "WP03 did not pass.")

    wp04_wp01 = wp04["source_evidence"]["wp01_screening"]
    wp03_wp01 = wp03["source_evidence"]["wp01_screening"]
    wp03_wp04 = wp03["source_evidence"]["wp04_ablation"]
    wp00_snapshot_sha = wp00["fixed_instance"]["snapshot"]["sha256"]
    _require(
        wp04_wp01["content_fingerprint"] == wp01s["content_fingerprint"],
        "WP04 is not bound to the supplied WP01-S artifact.",
    )
    _require(
        wp03_wp01["content_fingerprint"] == wp01s["content_fingerprint"],
        "WP03 is not bound to the supplied WP01-S artifact.",
    )
    _require(
        wp03_wp04["content_fingerprint"] == wp04["content_fingerprint"],
        "WP03 is not bound to the supplied WP04 artifact.",
    )
    _require(
        wp04["source_evidence"]["snapshot"]["sha256"] == wp00_snapshot_sha,
        "WP04 does not use the WP00 fixed snapshot.",
    )
    _require(
        wp03["source_evidence"]["snapshot"]["sha256"] == wp00_snapshot_sha,
        "WP03 does not use the WP00 fixed snapshot.",
    )

    full = wp04["full_setting"]
    wp04_summary = wp04["summary"]
    wp03_summary = wp03["summary"]
    provider = wp04["cost_provider_diagnostic"]

    intervals_overlap = bool(full["local_5_percent_intervals_overlap"]) and bool(
        full["transfer_25_percent_intervals_overlap"]
    )
    selection_invariant = not bool(
        wp03_summary["selection_changed_by_coefficient_policy"]
    )
    selected_candidate = str(wp03_summary["selected_candidate_for_all_policies"])
    _require(intervals_overlap, "Gate S1 expects both WP04 intervals to overlap.")
    _require(
        selection_invariant and selected_candidate == "ld12_delta0.02",
        "Gate S1 expects the documented invariant WP03 point selection.",
    )

    evidence_fingerprints = {
        "WP00": str(wp00["content_fingerprint"]),
        "WP02": str(wp02["content_fingerprint"]),
        "WP01-S": str(wp01s["content_fingerprint"]),
        "WP04": str(wp04["content_fingerprint"]),
        "WP03": str(wp03["content_fingerprint"]),
    }
    interval_evidence = {
        "ld3_rz_point_estimate": float(
            full["ld3_total_compiled_rz_point_estimate"]
        ),
        "ld12_rz_point_estimate": float(
            full["ld12_total_compiled_rz_point_estimate"]
        ),
        "ld12_point_reduction_relative_to_ld3": float(
            full["ld12_reduction_relative_to_ld3_point_estimate"]
        ),
        "local_5_percent_intervals_overlap": bool(
            full["local_5_percent_intervals_overlap"]
        ),
        "transfer_25_percent_intervals_overlap": bool(
            full["transfer_25_percent_intervals_overlap"]
        ),
        "paper_d6_local_relative_gap_interval": list(
            wp03_summary["paper_d6_local_5_percent_relative_gap_interval"]
        ),
        "paper_d6_transfer_relative_gap_interval": list(
            wp03_summary["paper_d6_transfer_25_percent_relative_gap_interval"]
        ),
        "judgement": "undetermined_not_tied",
    }
    mechanism_evidence = {
        "dominant_common_factor": str(wp04_summary["dominant_common_factor"]),
        "ld3_beta_removal_relative_cost_increase": _loo_increase(
            wp04, 3, "beta_reallocation"
        ),
        "ld12_beta_removal_relative_cost_increase": _loo_increase(
            wp04, 12, "beta_reallocation"
        ),
        "ld3_alpha_removal_relative_cost_increase": _loo_increase(
            wp04, 3, "alpha_reallocation"
        ),
        "ld12_alpha_removal_relative_cost_increase": _loo_increase(
            wp04, 12, "alpha_reallocation"
        ),
        "ld3_compiled_schedule_relative_rz_change_vs_fixed": float(
            provider["ld3_compiled_objective_schedule_relative_rz_change_vs_fixed"]
        ),
        "ld3_component_objective_schedule_relative_rz_change_vs_fixed": float(
            provider[
                "ld3_component_objective_schedule_relative_rz_change_vs_fixed"
            ]
        ),
        "judgement": (
            "observed_common_gain_is_dominated_by_beta_then_alpha;_a_"
            "partial_randomization_specific_advantage_is_not_demonstrated"
        ),
    }
    coefficient_evidence = {
        "selection_changed": not selection_invariant,
        "selected_candidate_for_all_policies": selected_candidate,
        "costed_max_paper_d6_vs_dominant_relative_gap": float(
            wp03_summary[
                "costed_max_paper_d6_vs_dominant_coefficient_relative_gap"
            ]
        ),
        "paper_d6_ld3_point_regret_vs_ld12": float(
            wp03_summary["paper_d6_ld3_point_regret_vs_ld12"]
        ),
        "minimum_pf_boundary_relative_headroom": min(
            float(value)
            for value in wp03_summary[
                "paper_d6_relative_headroom_to_pf_boundary_at_delta_0p02"
            ].values()
        ),
        "judgement": (
            "coefficient_choice_does_not_change_the_shortlist_or_resolve_"
            "the_cost_interval_overlap"
        ),
    }

    direction_decisions = [
        {
            "direction_id": "T1",
            "topic": "superiority_region_and_limits",
            "decision": "continue_conditionally_after_decision_bridge",
            "reason": (
                "The point estimate favors the deterministic endpoint, but both "
                "documented intervals overlap; retain the boundary question and "
                "do not claim either superiority or a tie."
            ),
            "reopen_or_advance_condition": (
                "WP05 full-scope intervals separate under equal optimization, or "
                "a reproducible condition-dependent boundary is identified."
            ),
        },
        {
            "direction_id": "T2",
            "topic": "state_specific_pf_error_and_partition_design",
            "decision": "narrow_to_required_coefficient_accuracy",
            "reason": (
                "C_D is insufficient as a final coefficient, but paper-D6 versus "
                "the dominant eigenphase does not change the current selection."
            ),
            "reopen_or_advance_condition": (
                "A candidate approaches the PF boundary, coefficient uncertainty "
                "changes ranking, or transfer to a new instance fails."
            ),
        },
        {
            "direction_id": "T3",
            "topic": "higher_order_partial_randomized_pf",
            "decision": "defer",
            "reason": (
                "The present decision is limited by circuit scope and structure, "
                "not by the tested second-order coefficient choice."
            ),
            "reopen_or_advance_condition": (
                "WP05 shows PF error or required round depth remains the dominant "
                "cost after full-scope calibration."
            ),
        },
        {
            "direction_id": "T4",
            "topic": "long_random_circuit_cost_prediction",
            "decision": "continue_high_priority",
            "reason": (
                "Long-q affine transfer and the full controlled-interrogation "
                "scope are the largest unresolved inputs to candidate ranking."
            ),
            "reopen_or_advance_condition": (
                "Complete WP06-a and then validate WP05 on unused q/scope "
                "holdouts with shared uncertainty."
            ),
        },
        {
            "direction_id": "T5",
            "topic": "finite_rte_and_rpe_schedule_optimization",
            "decision": "continue_narrow_scope",
            "reason": (
                "Beta and alpha allocation materially change both candidates, "
                "whereas the partial-specific compiled schedule effect is small."
            ),
            "reopen_or_advance_condition": (
                "Preserve the fair allocation rules and prioritize late rounds; "
                "broaden schedule search only if WP05 changes the cost ordering."
            ),
        },
        {
            "direction_id": "T6",
            "topic": "representation_and_sampling_joint_design",
            "decision": "run_wp06a_only_then_conditionally_defer",
            "reason": (
                "A cheap circuit-structure kill switch is justified, but broad "
                "representation or sampling exploration is not yet justified."
            ),
            "reopen_or_advance_condition": (
                "WP06-a reaches the preregistered structural trigger or exposes "
                "a ranking-changing implementation asymmetry."
            ),
        },
        {
            "direction_id": "T7",
            "topic": "reliable_resource_estimation_and_negative_results",
            "decision": "continue_primary_framing",
            "reason": (
                "Fair baseline optimization, provider-objective reversal, and "
                "overlapping uncertainty already identify reproducible ways that "
                "a screening estimate can give the wrong research impression."
            ),
            "reopen_or_advance_condition": (
                "Test whether the same corrections and limitations persist in "
                "WP05 and the later decision-grade comparison."
            ),
        },
    ]

    checks = {
        "all_upstream_artifacts_validate": True,
        "all_required_work_packages_present": set(evidence_fingerprints)
        == {"WP00", "WP02", "WP01-S", "WP04", "WP03"},
        "upstream_fingerprint_chain_matches": True,
        "fixed_snapshot_chain_matches": True,
        "both_cost_scenarios_overlap": intervals_overlap,
        "overlap_is_labeled_undetermined_not_tied": interval_evidence["judgement"]
        == "undetermined_not_tied",
        "coefficient_selection_is_invariant": selection_invariant,
        "next_action_is_single_and_preregistered": True,
        "final_total_cost_or_superiority_not_claimed": True,
    }

    return {
        "comparison_contract": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "hamiltonian_hash": str(
                wp00["fixed_instance"]["hamiltonian_hash"]
            ),
            "precision_task": "CA_over_10",
            "cost_scope": str(wp04["scope"]["circuit_cost_scope"]),
            "candidate_ld_values": [3, 12],
            "primary_delta_time": 0.02,
            "comparison_delta_time": 0.01,
            "sensitivity_delta_time": 0.0125,
        },
        "evidence_fingerprints": evidence_fingerprints,
        "gate_questions": {
            "candidate_cost_intervals": interval_evidence,
            "observed_gain_mechanism": mechanism_evidence,
            "pf_coefficient_selection": coefficient_evidence,
            "maximum_current_uncertainty": {
                "selected": (
                    "full_controlled_interrogation_cost_scope_and_circuit_structure"
                ),
                "components": [
                    "controlled_partial_s2_repetition_and_outer_boundaries",
                    "hadamard_wrapper_scope",
                    "long_q_cost_transfer_without_unused_holdout",
                    "shared_calibration_uncertainty_and_covariance",
                ],
                "why_not_pf_coefficient": (
                    "All three tested coefficient policies select the same "
                    "L_D and delta."
                ),
                "why_not_finite_rte_or_statistical_allocation": (
                    "Their largest common gains apply to both candidates and have "
                    "already been separated in WP04."
                ),
            },
        },
        "direction_decisions": direction_decisions,
        "shortlist": {
            "retain_for_wp05": [
                {"ld": 3, "delta_time": 0.02, "role": "partial_candidate"},
                {"ld": 12, "delta_time": 0.02, "role": "deterministic_endpoint"},
                {"ld": 3, "delta_time": 0.01, "role": "delta_comparator"},
                {"ld": 12, "delta_time": 0.01, "role": "delta_comparator"},
            ],
            "retain_as_sensitivity_not_first_batch": [
                {"ld": 3, "delta_time": 0.0125},
                {"ld": 12, "delta_time": 0.0125},
            ],
            "screened_out_for_current_model_conditional_task": [
                {
                    "ld": 0,
                    "reason": (
                        "WP01-S analytic component-application proxy; not a "
                        "general theoretical or compiled-RZ rejection"
                    ),
                }
            ],
            "primary_pf_coefficient_policy": "paper_d6",
            "small_system_reference_policy": "dominant_eigenphase",
            "screening_only_policy": "hd_surrogate",
        },
        "next_action": {
            "work_package": "WP06-a",
            "title": "representative_circuit_structure_kill_switch_pilot",
            "reason": (
                "It is the cheapest direct test of the largest ranking-relevant "
                "uncertainty before spending on a full-scope calibration."
            ),
            "inputs": (
                "one_or_two_representative_DF_Z_ZZ_events_and_short_sequences"
            ),
            "isolated_changes": [
                "full_gaussian_vs_support_restricted_basis_transform",
                "basis_transform_fusion",
                "known_scalar_and_phase_handling",
                "applicable_control_optimizations",
            ],
            "equivalence_requirement": (
                "exact_small_matrix_equivalence_including_known_phase_"
                "compensation_and_control_branch_relative_phase"
            ),
            "preregistered_decision_rule": {
                "eta_decision_relative_rz": 0.05,
                "eta_scope": (
                    "this_WP06a_circuit_structure_recalibration_decision_only"
                ),
                "reroute_before_wp05_if": [
                    "representative_RZ_change_at_least_eta_decision",
                    "candidate_ranking_reverses",
                    "asymptotic_q_slope_or_proxy_domain_changes",
                    "required_controlled_relative_phase_correction_is_missing_"
                    "from_the_current_structure",
                ],
                "otherwise": (
                    "retain_current_structure_and_proceed_to_WP05_while_"
                    "carrying_the_measured_residual_as_uncertainty"
                ),
                "rationale": (
                    "Five percent is approximately the current 4.96% point gap, "
                    "so a change of that size can alter the candidate decision."
                ),
            },
            "following_action": "WP05_full_controlled_interrogation_connection",
        },
        "scope": {
            "new_physical_simulation_performed": False,
            "new_circuit_compilation_performed": False,
            "research_direction_decision_record": True,
            "scientific_superiority_decision_grade": False,
            "final_total_cost_evaluation_performed": False,
            "evidence_status": "local_worktree_not_immutable_ci",
        },
        "limitations": [
            "Gate S1 synthesizes local H4 screening evidence and does not generalize to H12 or other geometries.",
            "The candidate intervals are sensitivity scenarios, not statistical confidence intervals.",
            "The compared costs omit state preparation and use q=1,2 affine extrapolation without an unused long-q holdout for these schedules.",
            "The 5% WP06-a trigger is a task-specific research-routing threshold, not a universal circuit-accuracy guarantee.",
            "No final total-cost or partial-randomization superiority evaluation has been performed.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "status": "Gate_S1_direction_synthesis_complete",
            "candidate_cost_judgement": "undetermined_not_tied",
            "partial_specific_advantage_demonstrated": False,
            "pf_coefficient_choice_changes_shortlist": False,
            "maximum_current_uncertainty": (
                "full_controlled_interrogation_cost_scope_and_circuit_structure"
            ),
            "next_action": "WP06-a",
            "following_action": "WP05",
            "active_primary_directions": ["T4", "T7"],
            "active_scoped_directions": ["T1", "T2", "T5", "T6"],
            "deferred_directions": ["T3"],
        },
    }


def finalize_gate_s1_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "Gate-S1",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_gate_s1_artifact(payload)
    return payload


def validate_gate_s1_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Gate-S1 schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "Gate-S1":
        raise ValueError("Unsupported Gate-S1 method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("Gate-S1 content_fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("Gate-S1 overall status does not match its checks.")
    scope = payload.get("scope", {})
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("Gate S1 cannot claim a final total-cost evaluation.")
    if scope.get("scientific_superiority_decision_grade") is not False:
        raise ValueError("Gate S1 cannot claim decision-grade superiority.")
    if payload.get("summary", {}).get("candidate_cost_judgement") != (
        "undetermined_not_tied"
    ):
        raise ValueError("Gate S1 must preserve the overlapping-interval result.")
    if payload.get("next_action", {}).get("work_package") != "WP06-a":
        raise ValueError("Gate S1 must select exactly WP06-a as its next action.")


def write_gate_s1_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_gate_s1_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

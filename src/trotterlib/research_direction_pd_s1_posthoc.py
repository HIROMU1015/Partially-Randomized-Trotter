"""Post-hoc interpretation of the frozen P-D S1 comparison artifact.

This module performs no Hamiltonian construction, diagonalization, RTE
sampling, or circuit compilation.  It preserves the preregistered S1 result
and derives a separately labelled interpretation from its saved candidates.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from . import research_direction_pd_fair_comparison as s1
from .research_direction_energy_tail_pareto import fingerprint


RESULT_SCHEMA = "research_direction_pd_s1_posthoc_reanalysis_v1"
METHOD = "pd_s1_frozen_artifact_posthoc_interpretation_v1"
SOURCE_RESULT_FINGERPRINT = (
    "0ba7764da7b7d8b7e195a5c315d3dc0a65c2c79ce01c51cf021a2685977487d2"
)
SOURCE_RESULT_FILE_SHA256 = (
    "6b8de6e255eb0796d93398c767017c2899837a63c7230e9956fb2d9beedbeaec"
)
PRIMARY_BASELINES = ("B1b", "B2", "B4")
DIAGNOSTIC_ABLATIONS = ("B0", "B1a")
SCOPES = ("nested", "native", "combined")
NEAR_OPTIMAL_RELATIVE_THRESHOLD = 0.05


def validate_source_result(payload: Mapping[str, Any]) -> None:
    """Validate both the original S1 contract and the fixed source identity."""

    s1.validate_result(payload)
    if payload.get("content_fingerprint") != SOURCE_RESULT_FINGERPRINT:
        raise ValueError("Post-hoc analysis requires the fixed P-D S1 v2 result.")
    classification = payload.get("classification", {})
    if classification.get("primary_case") != "B":
        raise ValueError("The frozen S1 primary Case B classification changed.")
    if classification.get("undetermined_boundary") is not True:
        raise ValueError("The frozen S1 boundary status changed.")
    if (
        classification.get("status")
        != "stop_s1_undetermined_boundary_no_go_decision"
    ):
        raise ValueError("The frozen S1 stop status changed.")


def _scope_rows(
    rows: Sequence[Mapping[str, Any]], scope: str
) -> list[Mapping[str, Any]]:
    if scope == "combined":
        return list(rows)
    return [row for row in rows if row["construction"] == scope]


def _row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {str(row["candidate_id"]): row for row in rows}


def _boundary_flags(
    row: Mapping[str, Any], scoped: Sequence[Mapping[str, Any]]
) -> dict[str, bool | None]:
    deltas = sorted({float(candidate["delta"]) for candidate in scoped})
    rte_steps = sorted(
        {int(candidate["rte_total_short_steps_per_outer_step"]) for candidate in scoped}
    )
    inner_values = sorted(
        {
            int(candidate["inner_hd_substeps"])
            for candidate in scoped
            if candidate["inner_hd_substeps"] is not None
        }
    )
    inner = row["inner_hd_substeps"]
    return {
        "delta_lower": float(row["delta"]) == deltas[0],
        "delta_upper": float(row["delta"]) == deltas[-1],
        "rte_steps_lower": int(row["rte_total_short_steps_per_outer_step"])
        == rte_steps[0],
        "rte_steps_upper": int(row["rte_total_short_steps_per_outer_step"])
        == rte_steps[-1],
        "inner_substeps_lower": (
            None if inner is None else int(inner) == inner_values[0]
        ),
        "inner_substeps_upper": (
            None if inner is None else int(inner) == inner_values[-1]
        ),
    }


def _candidate_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "construction": row["construction"],
        "formula_label": row["formula_label"],
        "delta": row["delta"],
        "inner_hd_substeps": row["inner_hd_substeps"],
        "rte_total_short_steps_per_outer_step": row[
            "rte_total_short_steps_per_outer_step"
        ],
        "finite_taylor_order": row["finite_taylor_order"],
        "deterministic_component_actions_total": row[
            "deterministic_component_actions_total"
        ],
        "b2_c1shot_proxy": row["b2_c1shot_proxy"],
        "b4_c1shot_expected_component_actions": row[
            "b4_c1shot_expected_component_actions"
        ],
        "b2_shot_factor": row["b2_shot_factor"],
        "b4_shot_factor": row["b4_shot_factor"],
        "b2_objective": row["b2_objective"],
        "b4_objective": row["b4_objective"],
        "b2_objective_relative_error_vs_b4": row[
            "b2_objective_relative_error_vs_b4"
        ],
        "b2_shot_factor_relative_error_vs_b4": row[
            "b2_shot_factor_relative_error_vs_b4"
        ],
        "deterministic_phase_bound_rad": row[
            "deterministic_phase_bound_rad"
        ],
        "finite_phase_error_bound_rad": row["finite_phase_error_bound_rad"],
        "b4_total_phase_error_bound_rad": row[
            "b4_total_phase_error_bound_rad"
        ],
        "b4_feasible": row["b4_feasible"],
    }


def _error_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparable = [
        row
        for row in rows
        if row.get("b2_objective_relative_error_vs_b4") is not None
        and row.get("b2_shot_factor_relative_error_vs_b4") is not None
    ]
    return {
        "candidate_count": len(rows),
        "comparable_candidate_count": len(comparable),
        "b2_proxy_accepts_b4_infeasible_count": sum(
            bool(row["deterministic_feasible"]) and not bool(row["b4_feasible"])
            for row in rows
        ),
        "max_abs_b2_objective_relative_error_vs_b4": (
            max(
                abs(float(row["b2_objective_relative_error_vs_b4"]))
                for row in comparable
            )
            if comparable
            else None
        ),
        "max_abs_b2_shot_factor_relative_error_vs_b4": (
            max(
                abs(float(row["b2_shot_factor_relative_error_vs_b4"]))
                for row in comparable
            )
            if comparable
            else None
        ),
    }


def _scope_reanalysis(
    source: Mapping[str, Any],
    scope: str,
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    block = source["primary_selection"][scope]
    scoped = _scope_rows(source["primary_k2_candidates"], scope)
    selected: dict[str, Any] = {}
    primary_ids: list[str] = []
    for model in (*DIAGNOSTIC_ABLATIONS, *PRIMARY_BASELINES):
        selection = block["selections"][model]
        if selection is None:
            selected[model] = None
            continue
        candidate_id = str(selection["candidate_id"])
        row = rows_by_id[candidate_id]
        regret = block["regret_against_b4"][model]
        selected[model] = {
            **_candidate_summary(row),
            "false_acceptance": regret["false_acceptance"],
            "regret_against_b4": regret["regret"],
            "boundary_flags": _boundary_flags(row, scoped),
        }
        if model in PRIMARY_BASELINES:
            primary_ids.append(candidate_id)

    reference = float(block["b4_reference_objective"])
    near_rows = sorted(
        (
            row
            for row in scoped
            if bool(row["b4_feasible"])
            and float(row["b4_objective"])
            <= reference * (1.0 + NEAR_OPTIMAL_RELATIVE_THRESHOLD)
        ),
        key=lambda row: (float(row["b4_objective"]), str(row["candidate_id"])),
    )
    near_candidates = []
    for row in near_rows:
        summary = _candidate_summary(row)
        summary["relative_gap_from_best_b4"] = float(
            float(row["b4_objective"]) / reference - 1.0
        )
        near_candidates.append(summary)

    return {
        "scope": scope,
        "candidate_count": len(scoped),
        "b4_feasible_count": sum(bool(row["b4_feasible"]) for row in scoped),
        "selected_models": selected,
        "primary_baseline_selection_consensus": len(set(primary_ids)) == 1,
        "primary_baseline_consensus_candidate_id": (
            primary_ids[0] if primary_ids and len(set(primary_ids)) == 1 else None
        ),
        "all_candidate_model_error_summary": _error_summary(scoped),
        "near_optimal_definition": {
            "reference_model": "B4",
            "relative_threshold": NEAR_OPTIMAL_RELATIVE_THRESHOLD,
            "reference_objective": reference,
        },
        "near_optimal_model_error_summary": _error_summary(near_rows),
        "near_optimal_candidates": near_candidates,
    }


def _b1a_inner_substep_diagnostic(
    source: Mapping[str, Any], rows_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    selected_id = str(
        source["primary_selection"]["nested"]["selections"]["B1a"][
            "candidate_id"
        ]
    )
    selected = rows_by_id[selected_id]
    series = sorted(
        (
            row
            for row in source["primary_k2_candidates"]
            if row["construction"] == "nested"
            and row["formula_label"] == selected["formula_label"]
            and float(row["delta"]) == float(selected["delta"])
            and int(row["rte_total_short_steps_per_outer_step"])
            == int(selected["rte_total_short_steps_per_outer_step"])
            and int(row["finite_taylor_order"])
            == int(selected["finite_taylor_order"])
        ),
        key=lambda row: int(row["inner_hd_substeps"]),
    )
    records = [
        {
            "candidate_id": row["candidate_id"],
            "inner_hd_substeps": row["inner_hd_substeps"],
            "outer_stage_count_total": row["outer_stage_count_total"],
            "deterministic_component_actions_total": row[
                "deterministic_component_actions_total"
            ],
            "deterministic_phase_bound_rad": row[
                "deterministic_phase_bound_rad"
            ],
            "finite_signal_error_bound": row["finite_signal_error_bound"],
            "finite_phase_error_bound_rad": row["finite_phase_error_bound_rad"],
            "b4_total_phase_error_bound_rad": row[
                "b4_total_phase_error_bound_rad"
            ],
            "b4_feasible": row["b4_feasible"],
        }
        for row in series
    ]
    finite_signal = float(selected["finite_signal_error_bound"])
    ideal_unit_radius_phase = (
        math.asin(finite_signal) if 0.0 <= finite_signal < 1.0 else None
    )
    phase_budget = float(source["configuration"]["total_phase_error_budget_rad"])
    return {
        "fixed_formula_label": selected["formula_label"],
        "fixed_delta": selected["delta"],
        "fixed_rte_total_short_steps_per_outer_step": selected[
            "rte_total_short_steps_per_outer_step"
        ],
        "fixed_finite_taylor_order": selected["finite_taylor_order"],
        "b1a_selected_candidate_id": selected_id,
        "b1a_objective_outer_stage_values": sorted(
            {int(row["outer_stage_count_total"]) for row in series}
        ),
        "records": records,
        "selected_finite_signal_error_bound": finite_signal,
        "ideal_unit_radius_finite_phase_bound_rad": ideal_unit_radius_phase,
        "phase_budget_rad": phase_budget,
        "same_tail_setting_can_be_made_b4_feasible_by_increasing_only_m_d": (
            False
            if ideal_unit_radius_phase is not None
            and ideal_unit_radius_phase > phase_budget
            else None
        ),
        "interpretation": (
            "B1a does not price inner work; its tie break moves along an "
            "objective-flat m_D axis.  Under the adopted B4 bound, the fixed "
            "tail contribution alone exceeds the phase budget even at unit "
            "signal radius."
        ),
    }


def _construction_decomposition(
    source: Mapping[str, Any], rows_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    selected_rows: dict[str, Mapping[str, Any]] = {}
    output: dict[str, Any] = {}
    for construction in ("nested", "native"):
        candidate_id = str(
            source["primary_selection"][construction]["selections"]["B4"][
                "candidate_id"
            ]
        )
        row = rows_by_id[candidate_id]
        selected_rows[construction] = row
        leading_tail_actions = int(row["outer_step_count"]) * int(
            row["rte_total_short_steps_per_outer_step"]
        )
        finite_tail_actions = float(
            row["b4_c1shot_expected_component_actions"]
        ) - int(row["deterministic_component_actions_total"])
        output[construction] = {
            **_candidate_summary(row),
            "leading_tail_component_actions_total": leading_tail_actions,
            "finite_expected_tail_component_actions_total": finite_tail_actions,
        }

    nested = selected_rows["nested"]
    native = selected_rows["native"]

    def ratio(field: str) -> float:
        return float(nested[field]) / float(native[field])

    return {
        "selected_points": output,
        "same_formula_label": nested["formula_label"] == native["formula_label"],
        "same_delta_r_and_k": (
            float(nested["delta"]) == float(native["delta"])
            and int(nested["rte_total_short_steps_per_outer_step"])
            == int(native["rte_total_short_steps_per_outer_step"])
            and int(nested["finite_taylor_order"])
            == int(native["finite_taylor_order"])
        ),
        "nested_over_native_ratios": {
            "deterministic_component_actions_total": ratio(
                "deterministic_component_actions_total"
            ),
            "b2_c1shot_proxy": ratio("b2_c1shot_proxy"),
            "b4_c1shot_expected_component_actions": ratio(
                "b4_c1shot_expected_component_actions"
            ),
            "b2_shot_factor": ratio("b2_shot_factor"),
            "b4_shot_factor": ratio("b4_shot_factor"),
            "b2_objective": ratio("b2_objective"),
            "b4_objective": ratio("b4_objective"),
        },
        "original_case_d_classifier_dimension": "selected_formula_label_only",
        "interpretation_limit": (
            "The ratio is an analytic component-action proxy comparison.  It "
            "does not establish compiled-circuit, measured-shot, or physical "
            "superiority of the native construction."
        ),
    }


def reanalyze_s1(source: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the frozen-data post-hoc interpretation body."""

    validate_source_result(source)
    rows = source["primary_k2_candidates"]
    rows_by_id = _row_index(rows)
    scopes = {
        scope: _scope_reanalysis(source, scope, rows_by_id) for scope in SCOPES
    }
    consensus = all(
        scopes[scope]["primary_baseline_selection_consensus"] for scope in SCOPES
    )
    primary_safe = all(
        scopes[scope]["selected_models"][model]["false_acceptance"] is False
        and float(
            scopes[scope]["selected_models"][model]["regret_against_b4"]
        )
        <= NEAR_OPTIMAL_RELATIVE_THRESHOLD
        for scope in SCOPES
        for model in PRIMARY_BASELINES
    )
    b1a_only_formal_failure = all(
        source["primary_selection"][scope]["regret_against_b4"]["B1b"][
            "false_acceptance"
        ]
        is False
        and source["primary_selection"][scope]["regret_against_b4"]["B1b"][
            "regret"
        ]
        <= NEAR_OPTIMAL_RELATIVE_THRESHOLD
        for scope in SCOPES
    ) and any(
        source["primary_selection"][scope]["regret_against_b4"]["B1a"][
            "false_acceptance"
        ]
        is True
        for scope in SCOPES
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "method": METHOD,
        "analysis_type": "post_hoc_reanalysis_of_frozen_s1_artifact",
        "source_result": {
            "content_fingerprint": source["content_fingerprint"],
            "expected_task_fingerprint": source["expected_task_fingerprint"],
            "file_sha256_expected": SOURCE_RESULT_FILE_SHA256,
            "formal_classification_preserved": dict(source["classification"]),
        },
        "fixed_rules": {
            "primary_baselines": list(PRIMARY_BASELINES),
            "diagnostic_ablations": list(DIAGNOSTIC_ABLATIONS),
            "near_optimal_relative_threshold": NEAR_OPTIMAL_RELATIVE_THRESHOLD,
            "source_candidate_grid_changed": False,
            "source_feasibility_rule_changed": False,
            "source_case_classification_changed": False,
        },
        "scope_reanalysis": scopes,
        "b1a_inner_substep_diagnostic": _b1a_inner_substep_diagnostic(
            source, rows_by_id
        ),
        "construction_decomposition": _construction_decomposition(
            source, rows_by_id
        ),
        "posthoc_interpretation": {
            "primary_baseline_consensus_in_every_scope": consensus,
            "primary_baselines_have_no_false_acceptance_or_decision_relevant_regret": primary_safe,
            "formal_case_b_and_boundary_are_caused_by_b1a_ablation_not_b1b": b1a_only_formal_failure,
            "selection_label": (
                "A_equivalent_on_frozen_candidate_set_under_B4"
                if consensus and primary_safe
                else "primary_baseline_selection_remains_unresolved"
            ),
            "not_a_preregistered_reclassification": True,
            "lower_r_boundary_limits_generalization": all(
                scopes[scope]["selected_models"]["B4"]["boundary_flags"][
                    "rte_steps_lower"
                ]
                is True
                for scope in SCOPES
            ),
        },
        "decision": {
            "unresolved_primary_baseline_selection_within_frozen_grid": not (
                consensus and primary_safe
            ),
            "additional_pd_s2_computation_authorized": False,
            "pd_active_development_status": "stop_without_s2",
            "next_step": (
                "audit_R3_prior_art_delta_and_freeze_a_separate_minimal_"
                "research_contract_before_any_new_pilot"
            ),
            "r3_adopted_as_primary_research": False,
            "r3_novelty_established": False,
        },
        "scope": {
            "new_hamiltonian_generated": False,
            "diagonalization_executed": False,
            "rte_sampling_executed": False,
            "circuit_compilation_executed": False,
            "new_candidate_grid_evaluated": False,
            "h12_evaluated": False,
            "long_rpe_evaluated": False,
            "final_total_cost_evaluated": False,
            "independent_holdout": False,
            "scientific_superiority_claimed": False,
        },
        "limitations": [
            "This is a post-hoc reanalysis, not the preregistered S1 result or an independent holdout.",
            "A-equivalent refers only to B1b/B2/B4 selection on the frozen candidate set under B4.",
            "The selected R=16 point is the lower evaluated R boundary, so untested R values are not covered.",
            "B4 is an analytic finite-RTE proxy, not compiled-circuit or measured-shot ground truth.",
            "The nested/native objective ratio mixes construction work and different deterministic error rules.",
            "R3 is only a next design candidate; novelty and a minimal validation contract remain unfixed.",
        ],
    }


def finalize_result(
    body: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    source_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        **dict(body),
        "provenance": dict(provenance),
        "source_evidence": [dict(row) for row in source_evidence],
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_result(payload)
    return payload


def validate_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("Unexpected P-D S1 post-hoc result schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-D S1 post-hoc result fingerprint mismatch.")
    source = payload.get("source_result", {})
    if source.get("content_fingerprint") != SOURCE_RESULT_FINGERPRINT:
        raise ValueError("P-D S1 post-hoc source fingerprint changed.")
    formal = source.get("formal_classification_preserved", {})
    if formal.get("primary_case") != "B" or formal.get("undetermined_boundary") is not True:
        raise ValueError("The formal S1 classification was not preserved.")
    interpretation = payload.get("posthoc_interpretation", {})
    if interpretation.get("not_a_preregistered_reclassification") is not True:
        raise ValueError("Post-hoc interpretation was presented as a reclassification.")
    if payload.get("decision", {}).get("additional_pd_s2_computation_authorized") is not False:
        raise ValueError("Post-hoc reanalysis must not authorize S2.")
    if any(value is not False for value in payload.get("scope", {}).values()):
        raise ValueError("Post-hoc result overstates the executed scope.")

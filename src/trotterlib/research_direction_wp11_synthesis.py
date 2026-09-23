"""WP11 scoped research-direction synthesis over completed screening evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .research_direction_ablation import fingerprint
from .research_direction_compiler_transfer_analysis import (
    validate_compiler_transfer_analysis_artifact,
)
from .research_direction_decision_cost import validate_wp01d_compute_artifact
from .research_direction_full_scope import validate_wp05a_artifact
from .research_direction_full_scope_extension import validate_wp05b_artifact
from .research_direction_full_scope_replication import validate_wp05br_artifact
from .research_direction_gate_s1 import validate_gate_s1_artifact
from .research_direction_m08_reaggregation import (
    validate_m08_reaggregation_artifact,
)
from .research_direction_round_dominance import validate_g08_artifact
from .research_direction_sequence_policy import validate_wp06b_artifact
from .research_direction_structure_pilot import validate_wp06a_artifact
from .research_direction_uncertainty_break_even import (
    validate_uncertainty_break_even_artifact,
)


SCHEMA_VERSION = "research_direction_wp11_synthesis_v1"
METHOD = "wp11_scoped_direction_and_single_followup_selection_v1"


def _fixed_instance_tuple(configuration: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        configuration["molecule"],
        float(configuration["geometry_angstrom"]),
        configuration["basis"],
        int(configuration["n_qubits"]),
        int(configuration["df_rank"]),
    )


def _require_fingerprint(
    container: Mapping[str, Any], key: str, expected: str, label: str
) -> None:
    observed = container[key]["content_fingerprint"]
    if observed != expected:
        raise ValueError(f"WP11 upstream fingerprint mismatch: {label}.")


def evaluate_wp11_synthesis(
    gate_s1: Mapping[str, Any],
    wp06a: Mapping[str, Any],
    wp06b: Mapping[str, Any],
    wp05a: Mapping[str, Any],
    wp05b: Mapping[str, Any],
    wp05br: Mapping[str, Any],
    wp01d: Mapping[str, Any],
    g08: Mapping[str, Any],
    m08_reaggregation: Mapping[str, Any],
    compiler_transfer: Mapping[str, Any],
    n07_p03: Mapping[str, Any],
) -> dict[str, Any]:
    """Synthesize completed evidence and select exactly one next discriminator."""
    validate_gate_s1_artifact(gate_s1)
    validate_wp06a_artifact(wp06a)
    validate_wp06b_artifact(wp06b)
    validate_wp05a_artifact(wp05a)
    validate_wp05b_artifact(wp05b)
    validate_wp05br_artifact(wp05br)
    validate_wp01d_compute_artifact(wp01d)
    validate_g08_artifact(g08)
    validate_m08_reaggregation_artifact(m08_reaggregation)
    validate_compiler_transfer_analysis_artifact(compiler_transfer)
    validate_uncertainty_break_even_artifact(n07_p03)

    fingerprints = {
        "gate_s1": gate_s1["content_fingerprint"],
        "wp06a": wp06a["content_fingerprint"],
        "wp06b": wp06b["content_fingerprint"],
        "wp05a": wp05a["content_fingerprint"],
        "wp05b": wp05b["content_fingerprint"],
        "wp05br": wp05br["content_fingerprint"],
        "wp01d": wp01d["content_fingerprint"],
        "g08": g08["content_fingerprint"],
        "m08_reaggregation": m08_reaggregation["content_fingerprint"],
        "compiler_transfer": compiler_transfer["content_fingerprint"],
        "n07_p03": n07_p03["content_fingerprint"],
    }

    _require_fingerprint(
        wp06a["source_evidence"], "gate_s1", fingerprints["gate_s1"], "WP06-a -> Gate S1"
    )
    _require_fingerprint(
        wp06b["source_evidence"], "wp06a", fingerprints["wp06a"], "WP06-b -> WP06-a"
    )
    _require_fingerprint(
        wp05a["source_evidence"], "wp06b", fingerprints["wp06b"], "WP05-a -> WP06-b"
    )
    _require_fingerprint(
        wp05b["source_evidence"], "wp05a", fingerprints["wp05a"], "WP05-b -> WP05-a"
    )
    _require_fingerprint(
        wp05b["source_evidence"], "wp06b", fingerprints["wp06b"], "WP05-b -> WP06-b"
    )
    _require_fingerprint(
        wp05br["source_evidence"], "wp05b", fingerprints["wp05b"], "WP05-bR -> WP05-b"
    )
    _require_fingerprint(
        wp01d["source_evidence"], "wp05a", fingerprints["wp05a"], "WP01-D -> WP05-a"
    )
    _require_fingerprint(
        wp01d["source_evidence"], "wp05b", fingerprints["wp05b"], "WP01-D -> WP05-b"
    )
    _require_fingerprint(
        wp01d["source_evidence"], "wp05br", fingerprints["wp05br"], "WP01-D -> WP05-bR"
    )
    if g08["input_fingerprints"]["compute"] != fingerprints["wp01d"]:
        raise ValueError("WP11 upstream fingerprint mismatch: G08 -> WP01-D.")
    if m08_reaggregation["input_fingerprints"]["compute"] != fingerprints["wp01d"]:
        raise ValueError("WP11 upstream fingerprint mismatch: M08 -> WP01-D.")
    if m08_reaggregation["input_fingerprints"]["g08"] != fingerprints["g08"]:
        raise ValueError("WP11 upstream fingerprint mismatch: M08 -> G08.")
    _require_fingerprint(
        compiler_transfer["source_evidence"],
        "wp01d",
        fingerprints["wp01d"],
        "M06/L08 -> WP01-D",
    )
    _require_fingerprint(
        compiler_transfer["source_evidence"],
        "wp05br",
        fingerprints["wp05br"],
        "M06/L08 -> WP05-bR",
    )
    expected_n07 = {
        "wp01d": fingerprints["wp01d"],
        "m08_reaggregation": fingerprints["m08_reaggregation"],
        "compiler_transfer": fingerprints["compiler_transfer"],
    }
    if n07_p03["input_fingerprints"] != expected_n07:
        raise ValueError("WP11 upstream fingerprint mismatch: N07/P03 inputs.")

    fixed_instances = {
        _fixed_instance_tuple(payload["configuration"])
        for payload in (wp06a, wp06b, wp05a, wp05b, wp05br, compiler_transfer, n07_p03)
    }
    gate_contract = gate_s1["comparison_contract"]
    fixed_instances.add(
        (
            gate_contract["molecule"],
            float(gate_contract["geometry_angstrom"]),
            gate_contract["basis"],
            int(gate_contract["n_qubits"]),
            int(gate_contract["df_rank"]),
        )
    )
    if len(fixed_instances) != 1:
        raise ValueError("WP11 inputs do not share one fixed physical instance.")

    ld3_best = wp01d["best_by_ld"]["3"]["best"]
    ld12_best = wp01d["best_by_ld"]["12"]["best"]
    selected_r_values = sorted({int(row["r_m"]) for row in ld3_best["rounds"]})
    direct_opt2_r_values = [int(compiler_transfer["configuration"]["rte_steps"])]
    missing_opt2_r_values = sorted(set(selected_r_values) - set(direct_opt2_r_values))
    direct_opt2_q_values = sorted(
        {
            int(row["q_m"])
            for row in compiler_transfer["same_trajectory_compiler_effect"][
                "ld3_selected_policy"
            ]["rows"]
        }
    )

    evidence_progression = [
        {
            "stage": "Gate_S1",
            "finding": "The screening comparison was undetermined rather than tied.",
            "routing_effect": "T4 and T7 became primary; T3 was deferred.",
        },
        {
            "stage": "WP06-a/WP06-b",
            "finding": (
                "Basis restriction was sequence dependent; the fixed singleton-run policy "
                "reduced holdout RZ by 10.67% and reversed the additive-bridge point ranking."
            ),
            "routing_effect": "Circuit structure must be an explicit cost-model input.",
        },
        {
            "stage": "WP05-a/b/R",
            "finding": (
                "The full controlled wrapper connected locally; an initial q=8 trigger was "
                "followed by an independent 32-trajectory replication with 0.52% selected RZ error."
            ),
            "routing_effect": "Proceed to candidate reoptimization while retaining the failed pilot.",
        },
        {
            "stage": "WP01-D/C07",
            "finding": (
                "After alpha, shots, and schedules were reoptimized, L_D=3 was 13.92% lower "
                "at the no-preparation optimization-level-1 point estimate."
            ),
            "routing_effect": "Retain L_D=3 and L_D=12; do not infer robust superiority.",
        },
        {
            "stage": "G08/M08",
            "finding": (
                "Late three rounds contained 90.94% of L_D=3 cost; q=16/32 direct holdouts "
                "passed, but q>32 transfer remained unmeasured."
            ),
            "routing_effect": "More local q precision was not the first discriminator.",
        },
        {
            "stage": "M06/L08",
            "finding": (
                "Optimization level 2 passed q<=32 proxy checks, but the coherent-evidence "
                "domain covered L_D=3 only at r=32 and the focused intervals overlapped."
            ),
            "routing_effect": "Compiler context became the nearest removable comparison asymmetry.",
        },
        {
            "stage": "N07/P03",
            "finding": (
                "L_D=3 used 2,376 more shots; positive common preparation work erodes its "
                "point advantage, and the opt2 focused interval overlaps already at P=0."
            ),
            "routing_effect": "Keep the robust result undetermined and avoid a superiority claim.",
        },
    ]

    directions = [
        {
            "direction_id": "T1",
            "topic": "superiority_region_and_limits",
            "wp11_decision": "rescope",
            "newly_learned": (
                "L_D=3 is the no-preparation point candidate after fair optimization, but its "
                "interval advantage is not invariant to compiler and transfer context."
            ),
            "relation_to_existing_work": (
                "The defensible contribution is a conditional boundary and failure analysis, "
                "not a general partially-randomized superiority statement."
            ),
            "supporting_evidence": ["wp01d", "m08_reaggregation", "compiler_transfer", "n07_p03"],
            "remaining_uncertainty": ["coherent_opt2_all_r", "q_above_32", "state_preparation", "external_instance"],
            "additional_validation_value": "high_only_after_compiler_context_is_coherent",
            "decision_change_condition": (
                "Advance only if a coherent compiler-context comparison separates intervals; "
                "otherwise retain a conditional or negative result."
            ),
        },
        {
            "direction_id": "T2",
            "topic": "state_specific_pf_error_and_partition_design",
            "wp11_decision": "continue_scoped",
            "newly_learned": (
                "Paper-D6 and dominant-eigenphase coefficient choices did not change the H4 shortlist."
            ),
            "relation_to_existing_work": (
                "The result supports a cheaper coefficient surrogate in this window; it is not a new PF theorem."
            ),
            "supporting_evidence": ["gate_s1", "wp01d"],
            "remaining_uncertainty": ["new_instance_transfer", "near_pf_boundary_cases"],
            "additional_validation_value": "low_for_current_H4_ranking",
            "decision_change_condition": (
                "Reopen when a candidate approaches the PF boundary or coefficient choice changes ranking."
            ),
        },
        {
            "direction_id": "T3",
            "topic": "higher_order_partial_randomized_pf",
            "wp11_decision": "hold",
            "newly_learned": "The current decision is dominated by compiled-cost context rather than second-order PF coefficient choice.",
            "relation_to_existing_work": (
                "No project-local evidence yet shows that a higher-order formula resolves the active bottleneck."
            ),
            "supporting_evidence": ["gate_s1", "n07_p03"],
            "remaining_uncertainty": ["higher_order_tail_cost_tradeoff"],
            "additional_validation_value": "low_until_pf_error_becomes_dominant",
            "decision_change_condition": (
                "Reopen only if coherent full-scope accounting identifies PF error or round depth as dominant."
            ),
        },
        {
            "direction_id": "T4",
            "topic": "long_random_circuit_cost_prediction",
            "wp11_decision": "continue_primary",
            "newly_learned": (
                "Sequence-aware circuit structure and compiler optimization materially change RZ cost; "
                "local affine q prediction works through q=32 but is not compiler invariant."
            ),
            "relation_to_existing_work": (
                "The contribution is an auditable domain-aware proxy workflow rather than assuming "
                "short-q linearity transfers across circuit and compiler contexts."
            ),
            "supporting_evidence": ["wp06a", "wp06b", "wp05a", "wp05br", "m08_reaggregation", "compiler_transfer"],
            "remaining_uncertainty": ["opt2_r_below_32", "q_above_32", "coupling_or_backend_context"],
            "additional_validation_value": "highest_for_coherent_opt2_all_r",
            "decision_change_condition": (
                "Reduce priority if the coherent all-r opt2 result still overlaps or if per-r compiler effects are uniform enough not to alter selection."
            ),
        },
        {
            "direction_id": "T5",
            "topic": "finite_rte_and_rpe_schedule_optimization",
            "wp11_decision": "continue_scoped",
            "newly_learned": (
                "Joint beta, alpha, shot, and round-schedule optimization changed the candidate gap; late rounds dominate cost but not RTE risk."
            ),
            "relation_to_existing_work": (
                "The relevant result is coupled discrete resource allocation, not an additive sum of isolated improvements."
            ),
            "supporting_evidence": ["gate_s1", "wp01d", "g08"],
            "remaining_uncertainty": ["compiler_consistent_reoptimization", "state_preparation_weighting"],
            "additional_validation_value": "high_as_part_of_selected_opt2_followup_not_as_separate_sweep",
            "decision_change_condition": (
                "Broaden schedule search only if coherent opt2 costs change the selected r or delta boundary."
            ),
        },
        {
            "direction_id": "T6",
            "topic": "representation_and_sampling_joint_design",
            "wp11_decision": "continue_scoped",
            "newly_learned": (
                "Universal support restriction failed; a training-fixed singleton-run policy passed holdouts and changed the point ranking."
            ),
            "relation_to_existing_work": (
                "The evidence favors a sequence-conditioned implementation rule, not a universal basis-restriction claim or new sampling theorem."
            ),
            "supporting_evidence": ["wp06a", "wp06b", "wp05a", "wp05br"],
            "remaining_uncertainty": ["other_instances", "coupling_context", "broader_sequence_classes"],
            "additional_validation_value": "moderate_only_if_selected_followup_exposes_policy_instability",
            "decision_change_condition": (
                "Reopen broad design only if the fixed policy fails coherent opt2 holdouts or an external instance."
            ),
        },
        {
            "direction_id": "T7",
            "topic": "reliable_resource_estimation_and_negative_results",
            "wp11_decision": "continue_primary",
            "newly_learned": (
                "Scope completion, fair reoptimization, direct holdouts, compiler change, and omitted preparation work each alter what can be claimed."
            ),
            "relation_to_existing_work": (
                "The strongest current contribution is a reproducible account of when simplified screening gives an unstable research conclusion."
            ),
            "supporting_evidence": list(fingerprints),
            "remaining_uncertainty": ["immutable_reproduction", "external_instance", "backend_noise", "final_total_cost"],
            "additional_validation_value": "high_for_one_coherent_opt2_discriminator",
            "decision_change_condition": (
                "Retain as primary even for a negative opt2 result; narrow it if findings reduce to one code-specific defect."
            ),
        },
    ]

    candidate_followups = [
        {
            "id": "all_r_coherent_opt2_reoptimization",
            "status": "selected",
            "uncertainty_addressed": "unmeasured optimization-level-2 context for L_D=3 r<32",
            "why_now": (
                "It removes a known within-comparison asymmetry before testing transfer to another physical instance."
            ),
            "what_remains_afterward": ["q_above_32_transfer", "state_preparation", "external_instance", "backend_noise"],
        },
        {
            "id": "external_instance_pilot",
            "status": "deferred_not_rejected",
            "uncertainty_addressed": "physical-instance transfer",
            "why_not_now": (
                "It would inherit the unresolved mixed opt1/opt2 r context and make a changed ranking ambiguous."
            ),
            "reopen_condition": "complete or stop the selected coherent opt2 discriminator",
        },
        {
            "id": "state_preparation_measurement",
            "status": "conditional",
            "uncertainty_addressed": "omitted per-shot work",
            "why_not_now": (
                "No preparation implementation is fixed, and the focused opt2 intervals overlap already at zero preparation work."
            ),
            "reopen_condition": (
                "a concrete candidate-specific preparation circuit is selected or plausible work approaches the 47,067,344 RZ-equivalent/shot point threshold"
            ),
        },
        {
            "id": "additional_q_above_32_holdout",
            "status": "conditional",
            "uncertainty_addressed": "long-q transfer",
            "why_not_now": (
                "The preregistered M08 q=64 trigger did not fire, while the missing opt2 r domain is a direct coherence defect."
            ),
            "reopen_condition": (
                "the coherent opt2 reoptimization retains a close ranking or selects a proxy whose error budget requires longer-q evidence"
            ),
        },
    ]

    selected_followup = {
        "id": "all_r_coherent_opt2_reoptimization",
        "title": "M06-F all-r coherent optimization-level-2 reoptimization",
        "compute_platform": "CPU_parallel_transpilation; GPU_not_required",
        "fixed_conditions": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "candidate_ld_values": [3, 12],
            "delta_values": [0.01, 0.02],
            "basis_gates": ["rz", "sx", "x", "cx"],
            "coupling_map": None,
            "transpiler_seed": 17,
            "optimization_level": 2,
            "circuit_scope": "complete controlled Hadamard wrapper",
            "sequence_policy": "support_run_le_1",
        },
        "evidence_reusable_without_recompile": {
            "ld3_r_values": direct_opt2_r_values,
            "direct_q_values": direct_opt2_q_values,
            "ld12_deterministic_provider": True,
        },
        "new_compute_scope": {
            "ld3_r_values": missing_opt2_r_values,
            "calibration_q_values": [1, 2],
            "holdout_policy": (
                "use q=4 or q=8 per r as a preregistered unused holdout; do not compile q>32 in the first batch"
            ),
            "trajectory_policy": "reuse matched physical trajectories where available and keep holdout seeds disjoint",
        },
        "analysis_scope": [
            "fit one optimization-level-2 proxy for every selected L_D=3 r value",
            "rerun the L_D=3/12 delta=0.01/0.02 beta-alpha-shot-schedule optimization coherently",
            "propagate measured per-r proxy discrepancy without averaging compiler contexts",
            "repeat N07/P03 interval and preparation break-even accounting on the coherent result",
        ],
        "decision_rule": {
            "advance_to_external_instance_pilot_if": (
                "the coherent opt2 candidate remains feasible and the ranking or interval conclusion is scientifically worth testing for transfer"
            ),
            "stop_local_compiler_refinement_if": (
                "measured-discrepancy intervals still overlap or the deterministic endpoint becomes preferred without a narrow unresolved compiler domain"
            ),
            "consider_long_q_followup_if": (
                "q>32 transfer is the remaining ranking-changing uncertainty after coherent reoptimization"
            ),
            "scientific_superiority_allowed_from_this_followup": False,
        },
        "not_part_of_this_followup": [
            "H12 or larger-system statevector simulation",
            "state-preparation circuit implementation",
            "backend or noise execution",
            "coupling-map transfer",
            "final total-cost evaluation",
        ],
    }

    checks = {
        "all_artifact_schemas_and_fingerprints_validate": True,
        "upstream_fingerprint_chain_matches": True,
        "fixed_h4_instance_matches": len(fixed_instances) == 1,
        "initial_wp05b_failure_is_preserved": wp05b["overall_pass"] is False,
        "independent_wp05br_followup_passes": bool(wp05br["overall_pass"]),
        "all_seven_directions_have_wp11_decisions": (
            {row["direction_id"] for row in directions} == {f"T{i}" for i in range(1, 8)}
        ),
        "exactly_one_next_followup_is_selected": (
            sum(row["status"] == "selected" for row in candidate_followups) == 1
        ),
        "selected_followup_targets_all_missing_opt2_r_values": (
            missing_opt2_r_values == [1, 2, 4, 8, 16]
        ),
        "external_pilot_is_deferred_not_rejected": (
            next(row for row in candidate_followups if row["id"] == "external_instance_pilot")["status"]
            == "deferred_not_rejected"
        ),
        "robust_direction_remains_undetermined": (
            n07_p03["decision"]["robust_directional_result"]
            == "undetermined_under_compiler_transfer_and_preparation_sensitivity"
        ),
        "no_new_compute_or_final_claim_in_wp11": True,
    }

    return {
        "comparison_contract": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "candidate_ld_values": [3, 12],
            "delta_time": 0.02,
            "precision_task": "CA_over_10",
            "cost_metric": "compiled_rz",
        },
        "input_fingerprints": fingerprints,
        "evidence_progression": evidence_progression,
        "direction_decisions": directions,
        "candidate_followups": candidate_followups,
        "selected_followup": selected_followup,
        "current_decision": {
            "no_preparation_point_preference": "L_D=3",
            "robust_interval_preference": "not_established",
            "robust_directional_result": (
                "undetermined_under_compiler_transfer_and_preparation_sensitivity"
            ),
            "primary_directions": ["T4", "T7"],
            "scoped_directions": ["T2", "T5", "T6"],
            "rescoped_directions": ["T1"],
            "held_directions": ["T3"],
            "selected_next_followup": "all_r_coherent_opt2_reoptimization",
        },
        "quantitative_basis": {
            "wp01d_ld3_point_reduction_relative_to_ld12": (
                1.0
                - float(ld3_best["total_compiled_rz_point_estimate"])
                / float(ld12_best["total_compiled_rz_point_estimate"])
            ),
            "ld3_last_three_round_cost_fraction": float(
                g08["summary"]["ld3_last_three_round_cost_fraction"]
            ),
            "m08_selected_rz_discrepancy": float(
                m08_reaggregation["summary"]["m08_selected_policy_rz_discrepancy"]
            ),
            "opt2_focused_point_advantage_ld3": float(
                compiler_transfer["summary"]["focused_ld3_point_advantage_fraction"]
            ),
            "opt2_focused_intervals_overlap": bool(
                compiler_transfer["summary"]["focused_measured_discrepancy_intervals_overlap"]
            ),
            "ld3_minus_ld12_shots": int(n07_p03["summary"]["ld3_minus_ld12_shots"]),
            "opt2_focused_preparation_point_break_even": float(
                n07_p03["summary"]["opt2_focused_point_break_even"]
            ),
            "selected_ld3_r_values": selected_r_values,
            "direct_opt2_ld3_r_values": direct_opt2_r_values,
            "missing_opt2_ld3_r_values": missing_opt2_r_values,
            "selected_schedule_q_maximum": max(int(row["q_m"]) for row in ld3_best["rounds"]),
        },
        "scope": {
            "existing_artifacts_only": True,
            "new_physical_simulation_performed": False,
            "new_circuit_compilation_performed": False,
            "full_opt2_reoptimization_performed": False,
            "external_instance_validated": False,
            "state_preparation_cost_measured": False,
            "q_above_32_directly_validated_under_opt2": False,
            "backend_execution_included": False,
            "noise_included": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
        "limitations": [
            "WP11 is a research-routing synthesis of local dirty-worktree artifacts, not a new numerical validation.",
            "The selected follow-up removes the all-r compiler-context asymmetry but does not by itself validate q>32 transfer.",
            "The existing physical evidence is one H4 snapshot, one geometry, one basis, and topology-free compiler contexts.",
            "The relation-to-existing-work statements delimit contribution type; a separate literature novelty review was not performed.",
            "State preparation, backend noise, fault-tolerant synthesis, external transfer, and final total cost remain outside scope.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "status": "WP11_scoped_direction_synthesis_complete",
            "primary_directions": ["T4", "T7"],
            "rescoped_direction": "T1",
            "held_direction": "T3",
            "robust_directional_result": (
                "undetermined_under_compiler_transfer_and_preparation_sensitivity"
            ),
            "selected_next_followup": "all_r_coherent_opt2_reoptimization",
            "new_opt2_ld3_r_values_required": missing_opt2_r_values,
            "external_instance_pilot_status": "deferred_not_rejected",
            "gpu_required_for_selected_followup": False,
        },
    }


def finalize_wp11_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP11-scoped-direction-synthesis",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp11_artifact(payload)
    return payload


def validate_wp11_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WP11 synthesis schema.")
    if payload.get("method") != METHOD:
        raise ValueError("Unsupported WP11 synthesis method.")
    if payload.get("stage") != "WP11-scoped-direction-synthesis":
        raise ValueError("Unsupported WP11 synthesis stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP11 synthesis fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP11 synthesis status does not match checks.")
    scope = payload.get("scope", {})
    for key in (
        "new_physical_simulation_performed",
        "new_circuit_compilation_performed",
        "full_opt2_reoptimization_performed",
        "external_instance_validated",
        "state_preparation_cost_measured",
        "final_total_cost_evaluation_performed",
        "scientific_superiority_claimed",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"WP11 synthesis cannot claim {key}.")
    selected = [
        row for row in payload.get("candidate_followups", []) if row.get("status") == "selected"
    ]
    if len(selected) != 1:
        raise ValueError("WP11 synthesis must select exactly one follow-up.")


def write_wp11_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_wp11_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

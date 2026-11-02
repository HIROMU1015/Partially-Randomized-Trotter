"""N07 uncertainty ledger and P03 preparation-cost break-even analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .research_direction_ablation import fingerprint
from .research_direction_compiler_transfer_analysis import (
    validate_compiler_transfer_analysis_artifact,
)
from .research_direction_decision_cost import validate_wp01d_compute_artifact
from .research_direction_m08_reaggregation import (
    validate_m08_reaggregation_artifact,
)


SCHEMA_VERSION = "research_direction_uncertainty_break_even_v1"
METHOD = "n07_p03_uncertainty_ledger_and_preparation_break_even_v1"


def _intervals_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) <= min(
        float(left[1]), float(right[1])
    )


def _break_even(
    *,
    ld3_cost: float,
    ld12_cost: float,
    ld3_interval: Sequence[float],
    ld12_interval: Sequence[float],
    ld3_shots: int,
    ld12_shots: int,
) -> dict[str, Any]:
    """Return point and interval break-even thresholds in RZ/shot units."""
    shot_difference = ld3_shots - ld12_shots
    if shot_difference <= 0:
        raise ValueError("P03 currently requires L_D=3 to use more shots.")
    point_work_advantage = ld12_cost - ld3_cost
    if point_work_advantage <= 0.0:
        raise ValueError("P03 currently requires the no-prep point preference L_D=3.")

    ld3_upper = float(ld3_interval[1])
    ld3_lower = float(ld3_interval[0])
    ld12_upper = float(ld12_interval[1])
    ld12_lower = float(ld12_interval[0])
    guaranteed_ld3_margin = ld12_lower - ld3_upper
    guaranteed_ld12_threshold_work = ld12_upper - ld3_lower

    return {
        "units": "compiled_rz_equivalent_per_shot",
        "ld3_total_shots": ld3_shots,
        "ld12_total_shots": ld12_shots,
        "ld3_minus_ld12_shot_count": shot_difference,
        "point_no_prep_work_advantage_ld3": point_work_advantage,
        "common_per_shot_preparation_point_break_even": (
            point_work_advantage / shot_difference
        ),
        "common_preparation_cost_point_preference": {
            "below_break_even": "L_D=3",
            "above_break_even": "L_D=12",
            "at_break_even": "equal_point_estimates",
        },
        "intervals_overlap_at_zero_preparation_cost": _intervals_overlap(
            ld3_interval, ld12_interval
        ),
        "common_per_shot_preparation_ld3_guaranteed_better_until": (
            guaranteed_ld3_margin / shot_difference
            if guaranteed_ld3_margin > 0.0
            else None
        ),
        "common_per_shot_preparation_ld12_guaranteed_better_from": max(
            0.0, guaranteed_ld12_threshold_work / shot_difference
        ),
        "candidate_specific_preparation_boundary": {
            "point_equality": (
                "N3*P3-N12*P12=ld12_cost-ld3_cost"
            ),
            "point_rhs_compiled_rz": point_work_advantage,
            "ld3_guaranteed_better_if_weighted_preparation_difference_below": (
                guaranteed_ld3_margin
            ),
            "ld12_guaranteed_better_if_weighted_preparation_difference_above": (
                guaranteed_ld12_threshold_work
            ),
            "weighted_preparation_difference_definition": (
                "N3*P3-N12*P12"
            ),
        },
    }


def _scenario(
    *,
    name: str,
    family: str,
    evidence_class: str,
    direct_domain: str,
    ld3_cost: float,
    ld12_cost: float,
    ld3_interval: Sequence[float],
    ld12_interval: Sequence[float],
    ld3_shots: int,
    ld12_shots: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "evidence_class": evidence_class,
        "direct_domain": direct_domain,
        "ld3_no_prep_compiled_rz": ld3_cost,
        "ld12_no_prep_compiled_rz": ld12_cost,
        "ld3_no_prep_interval": list(ld3_interval),
        "ld12_no_prep_interval": list(ld12_interval),
        "no_prep_intervals_overlap": _intervals_overlap(
            ld3_interval, ld12_interval
        ),
        "break_even": _break_even(
            ld3_cost=ld3_cost,
            ld12_cost=ld12_cost,
            ld3_interval=ld3_interval,
            ld12_interval=ld12_interval,
            ld3_shots=ld3_shots,
            ld12_shots=ld12_shots,
        ),
    }


def _scenario_from_comparison(
    *,
    name: str,
    family: str,
    evidence_class: str,
    direct_domain: str,
    comparison: Mapping[str, Any],
    scenario_name: str,
    ld3_shots: int,
    ld12_shots: int,
) -> dict[str, Any]:
    row = comparison["scenarios"][scenario_name]
    return _scenario(
        name=name,
        family=family,
        evidence_class=evidence_class,
        direct_domain=direct_domain,
        ld3_cost=float(comparison["ld3_total_compiled_rz_point_estimate"]),
        ld12_cost=float(comparison["ld12_total_compiled_rz_point_estimate"]),
        ld3_interval=row["ld3_interval"],
        ld12_interval=row["ld12_interval"],
        ld3_shots=ld3_shots,
        ld12_shots=ld12_shots,
    )


def evaluate_uncertainty_break_even(
    wp01d: Mapping[str, Any],
    m08_reaggregation: Mapping[str, Any],
    compiler_transfer: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate uncertainty classes and parameterize omitted preparation cost."""
    validate_wp01d_compute_artifact(wp01d)
    validate_m08_reaggregation_artifact(m08_reaggregation)
    validate_compiler_transfer_analysis_artifact(compiler_transfer)
    if m08_reaggregation["input_fingerprints"]["compute"] != (
        wp01d["content_fingerprint"]
    ):
        raise ValueError("M08 reaggregation does not bind the supplied WP01-D artifact.")
    if compiler_transfer["source_evidence"]["wp01d"][
        "content_fingerprint"
    ] != wp01d["content_fingerprint"]:
        raise ValueError("Compiler transfer does not bind the supplied WP01-D artifact.")
    if not all(
        bool(payload["overall_pass"])
        for payload in (wp01d, m08_reaggregation, compiler_transfer)
    ):
        raise ValueError("N07/P03 requires passing upstream artifacts.")

    ld3_best = wp01d["best_by_ld"]["3"]["best"]
    ld12_best = wp01d["best_by_ld"]["12"]["best"]
    shots3 = int(ld3_best["total_shots"])
    shots12 = int(ld12_best["total_shots"])
    opt1_cost3 = float(ld3_best["total_compiled_rz_point_estimate"])
    opt1_cost12 = float(ld12_best["total_compiled_rz_point_estimate"])

    scenarios: dict[str, Any] = {}
    for source_name, output_name, evidence_class, direct_domain in (
        (
            "m08_selected_policy_rz_q_le_32",
            "opt1_m08_selected_q_le_32_common_width",
            "measured_q_le_32_common_width_counterfactual_for_ld12",
            "L_D=3 r=32 q<=32; width applied symmetrically",
        ),
        (
            "m08_max_observed_rz_q_le_32",
            "opt1_m08_max_rz_q_le_32_common_width",
            "measured_q_le_32_common_width_counterfactual_for_ld12",
            "L_D=3 r=32 q<=32; width applied symmetrically",
        ),
        (
            "local_5_percent",
            "opt1_local_5_percent",
            "local_model_discrepancy_scenario",
            "scenario assumption",
        ),
        (
            "transfer_25_percent",
            "opt1_transfer_25_percent",
            "transfer_sensitivity_scenario",
            "scenario assumption",
        ),
    ):
        source = m08_reaggregation["scenarios"][source_name]
        scenarios[output_name] = _scenario(
            name=output_name,
            family="optimization_level_1_wp01d",
            evidence_class=evidence_class,
            direct_domain=direct_domain,
            ld3_cost=opt1_cost3,
            ld12_cost=opt1_cost12,
            ld3_interval=source["ld3_interval"],
            ld12_interval=source["ld12_interval"],
            ld3_shots=shots3,
            ld12_shots=shots12,
        )

    focused = compiler_transfer["focused_fixed_plan_reaggregation"][
        "comparison"
    ]
    for source_name, suffix, evidence_class, direct_domain in (
        (
            "opt2_selected_policy_measured",
            "selected_q_le_32",
            "focused_mixed_compiler_measured_discrepancy",
            "L_D=3 opt2 direct only at r=32 and q<=32",
        ),
        (
            "opt2_maximum_observed_rz",
            "maximum_rz_q_le_32",
            "focused_mixed_compiler_measured_discrepancy",
            "L_D=3 opt2 direct only at r=32 and q<=32",
        ),
        (
            "local_5_percent",
            "local_5_percent",
            "focused_mixed_compiler_local_scenario",
            "scenario assumption",
        ),
        (
            "transfer_25_percent",
            "transfer_25_percent",
            "focused_mixed_compiler_transfer_scenario",
            "scenario assumption",
        ),
    ):
        output_name = f"opt2_focused_{suffix}"
        scenarios[output_name] = _scenario_from_comparison(
            name=output_name,
            family="optimization_level_2_focused_fixed_plan",
            evidence_class=evidence_class,
            direct_domain=direct_domain,
            comparison=focused,
            scenario_name=source_name,
            ld3_shots=shots3,
            ld12_shots=shots12,
        )

    counterfactual = compiler_transfer["uniform_ratio_transfer_counterfactual"][
        "comparison"
    ]
    for source_name, suffix in (
        ("opt2_selected_policy_measured", "selected_q_le_32"),
        ("opt2_maximum_observed_rz", "maximum_rz_q_le_32"),
        ("local_5_percent", "local_5_percent"),
        ("transfer_25_percent", "transfer_25_percent"),
    ):
        output_name = f"opt2_uniform_ratio_counterfactual_{suffix}"
        scenarios[output_name] = _scenario_from_comparison(
            name=output_name,
            family="optimization_level_2_uniform_ratio_counterfactual",
            evidence_class="counterfactual_not_direct_evidence_for_r_below_32",
            direct_domain="ratio transfer to unmeasured L_D=3 r<32 rounds",
            comparison=counterfactual,
            scenario_name=source_name,
            ld3_shots=shots3,
            ld12_shots=shots12,
        )

    point_breaks = [
        float(row["break_even"]["common_per_shot_preparation_point_break_even"])
        for row in scenarios.values()
        if row["family"]
        != "optimization_level_2_uniform_ratio_counterfactual"
    ]
    opt1_local = scenarios["opt1_local_5_percent"]
    opt2_measured = scenarios["opt2_focused_selected_q_le_32"]
    opt2_counterfactual_measured = scenarios[
        "opt2_uniform_ratio_counterfactual_selected_q_le_32"
    ]

    uncertainty_ledger = [
        {
            "source": "calibration_sampling",
            "class": "sampling_uncertainty",
            "status": "quantified_local",
            "quantification": {
                "opt1_ld3_conservative_95_half_width": float(
                    ld3_best["conservative_calibration_95_half_width"]
                ),
                "opt1_ld12_conservative_95_half_width": float(
                    ld12_best["conservative_calibration_95_half_width"]
                ),
                "opt2_focused_ld3_conservative_95_half_width": float(
                    compiler_transfer["focused_fixed_plan_reaggregation"]["ld3"][
                        "conservative_calibration_95_half_width"
                    ]
                ),
            },
            "combination_rule": "conservative shared-fit sum; do not average as independent rounds",
            "decision_relevance": "included in every interval",
        },
        {
            "source": "affine_proxy_model_discrepancy",
            "class": "model_bias_sensitivity",
            "status": "quantified_only_through_q32",
            "quantification": {
                "opt1_m08_selected_rz": float(
                    m08_reaggregation["summary"][
                        "m08_selected_policy_rz_discrepancy"
                    ]
                ),
                "opt1_m08_max_observed_rz": float(
                    m08_reaggregation["summary"][
                        "m08_max_observed_rz_discrepancy"
                    ]
                ),
                "opt2_selected_rz": float(
                    compiler_transfer["summary"][
                        "maximum_selected_policy_q16_q32_rz_relative_error"
                    ]
                ),
                "opt2_max_observed_rz": float(
                    compiler_transfer["summary"][
                        "maximum_full_basis_q16_q32_rz_relative_error"
                    ]
                ),
            },
            "combination_rule": "symmetric discrepancy scenario plus calibration half-width; not quadrature with sampling SE",
            "decision_relevance": "opt2 focused measured interval already overlaps",
        },
        {
            "source": "compiler_context",
            "class": "systematic_context_sensitivity",
            "status": "partially_quantified",
            "quantification": {
                "ld3_focused_relative_change": float(
                    compiler_transfer["focused_fixed_plan_reaggregation"]["ld3"][
                        "relative_change_from_baseline"
                    ]
                ),
                "ld12_relative_change": float(
                    compiler_transfer["focused_fixed_plan_reaggregation"]["ld12"][
                        "relative_change_from_baseline"
                    ]
                ),
                "focused_point_advantage_ld3": float(
                    compiler_transfer["summary"][
                        "focused_ld3_point_advantage_fraction"
                    ]
                ),
            },
            "combination_rule": "report separate compiler contexts; do not turn context shift into sampling SE",
            "decision_relevance": "local separation is not compiler invariant",
        },
        {
            "source": "long_q_transfer",
            "class": "unmeasured_extrapolation",
            "status": "unresolved",
            "quantification": {"direct_q_maximum": 32, "schedule_q_maximum": 131072},
            "combination_rule": "retain explicit 5% and 25% scenarios",
            "decision_relevance": "blocks robust long-schedule superiority",
        },
        {
            "source": "opt2_r_below_32",
            "class": "unmeasured_compiler_domain",
            "status": "unresolved",
            "quantification": {"direct_opt2_r_values_for_ld3": [32]},
            "combination_rule": "focused mixed-context result and uniform-ratio counterfactual remain separate",
            "decision_relevance": "blocks a coherent full-opt2 candidate comparison",
        },
        {
            "source": "state_preparation",
            "class": "omitted_cost_parameter",
            "status": "parameterized_not_measured",
            "quantification": {
                "ld3_shots": shots3,
                "ld12_shots": shots12,
                "point_break_even_range_across_noncounterfactual_contexts": [
                    min(point_breaks),
                    max(point_breaks),
                ],
            },
            "combination_rule": "add P*N_shots; use separate P3 and P12 when preparations differ",
            "decision_relevance": "positive common P always reduces the L_D=3 point advantage because it uses more shots",
        },
        {
            "source": "external_snapshot_backend_noise",
            "class": "external_transfer_and_execution",
            "status": "unmeasured",
            "quantification": None,
            "combination_rule": "do not assign a numerical error bar without a new validation",
            "decision_relevance": "outside the current H4 local claim",
        },
    ]

    checks = {
        "upstream_artifacts_pass_and_bind": True,
        "shot_counts_are_fixed_and_ld3_uses_more_shots": shots3 > shots12,
        "all_scenarios_have_positive_finite_point_break_even": all(
            math.isfinite(
                float(
                    row["break_even"][
                        "common_per_shot_preparation_point_break_even"
                    ]
                )
            )
            and float(
                row["break_even"][
                    "common_per_shot_preparation_point_break_even"
                ]
            )
            > 0.0
            for row in scenarios.values()
        ),
        "opt1_local_has_small_guaranteed_ld3_preparation_region": (
            opt1_local["break_even"][
                "common_per_shot_preparation_ld3_guaranteed_better_until"
            ]
            is not None
        ),
        "opt2_focused_measured_has_no_guaranteed_ld3_region": (
            opt2_measured["break_even"][
                "common_per_shot_preparation_ld3_guaranteed_better_until"
            ]
            is None
        ),
        "uniform_ratio_counterfactual_changes_measured_interval_decision": (
            bool(opt2_measured["no_prep_intervals_overlap"])
            and not bool(
                opt2_counterfactual_measured["no_prep_intervals_overlap"]
            )
        ),
        "unmeasured_sources_remain_explicit": all(
            any(row["source"] == source for row in uncertainty_ledger)
            for source in (
                "long_q_transfer",
                "opt2_r_below_32",
                "external_snapshot_backend_noise",
            )
        ),
        "final_total_cost_and_superiority_not_claimed": True,
    }
    return {
        "configuration": {
            "comparison_task": "WP00 fixed H4 CA_over_10 task",
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "candidate_ld_values": [3, 12],
            "delta_time": 0.02,
            "cost_units": "compiled_rz",
            "preparation_parameter_units": "compiled_rz_equivalent_per_shot",
            "ld3_total_shots": shots3,
            "ld12_total_shots": shots12,
        },
        "input_fingerprints": {
            "wp01d": wp01d["content_fingerprint"],
            "m08_reaggregation": m08_reaggregation["content_fingerprint"],
            "compiler_transfer": compiler_transfer["content_fingerprint"],
        },
        "n07_uncertainty_ledger": uncertainty_ledger,
        "p03_break_even_scenarios": scenarios,
        "synthesis": {
            "no_prep_point_preference_in_all_scenarios": "L_D=3",
            "compiler_robust_interval_preference": "not_established",
            "common_preparation_cost_effect": (
                "always_erodes_ld3_advantage_because_ld3_uses_more_shots"
            ),
            "noncounterfactual_point_break_even_range": [
                min(point_breaks),
                max(point_breaks),
            ],
            "compiler_robust_guaranteed_ld3_preparation_interval": None,
            "dominant_next_decision_uncertainties": [
                "unmeasured_opt2_r_below_32",
                "q_above_32_transfer",
                "state_preparation_if_cost_approaches_point_break_even",
            ],
        },
        "decision": {
            "n07_status": "uncertainty_classes_separated",
            "p03_status": "preparation_cost_parameterized",
            "current_scientific_claim": (
                "H4_no_prep_point_preference_only; robust interval superiority not established"
            ),
            "robust_directional_result": (
                "undetermined_under_compiler_transfer_and_preparation_sensitivity"
            ),
            "next_action": (
                "WP11_scoped_direction_synthesis_before_full_opt2_or_external_pilot"
            ),
        },
        "scope": {
            "new_circuit_compilation_performed": False,
            "state_preparation_cost_measured": False,
            "state_preparation_cost_parameterized": True,
            "q_above_32_directly_validated": False,
            "ld3_opt2_r_below_32_directly_validated": False,
            "external_transfer_validated": False,
            "backend_execution_included": False,
            "noise_included": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
        "limitations": [
            "Break-even P is an RZ-equivalent accounting parameter, not a measured state-preparation circuit cost.",
            "The focused opt2 scenario mixes opt2 r=32 costs with opt1 r<32 costs for L_D=3.",
            "Measured proxy discrepancies end at q=32 while the selected schedule reaches q=131072.",
            "Candidate-specific preparation states require the two-dimensional P3/P12 boundary rather than a common P.",
            "No backend, noise, external snapshot, system-size, or final total-cost claim is made.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "ld3_minus_ld12_shots": shots3 - shots12,
            "opt1_point_break_even": float(
                scenarios["opt1_local_5_percent"]["break_even"][
                    "common_per_shot_preparation_point_break_even"
                ]
            ),
            "opt1_local_ld3_guaranteed_better_until": scenarios[
                "opt1_local_5_percent"
            ]["break_even"][
                "common_per_shot_preparation_ld3_guaranteed_better_until"
            ],
            "opt2_focused_point_break_even": float(
                scenarios["opt2_focused_selected_q_le_32"]["break_even"][
                    "common_per_shot_preparation_point_break_even"
                ]
            ),
            "opt2_focused_ld3_guaranteed_better_until": scenarios[
                "opt2_focused_selected_q_le_32"
            ]["break_even"][
                "common_per_shot_preparation_ld3_guaranteed_better_until"
            ],
            "compiler_robust_interval_preference": "not_established",
            "robust_directional_result": (
                "undetermined_under_compiler_transfer_and_preparation_sensitivity"
            ),
            "next_action": (
                "WP11_scoped_direction_synthesis_before_full_opt2_or_external_pilot"
            ),
        },
    }


def finalize_uncertainty_break_even_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "N07-P03-analysis-reaggregation",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_uncertainty_break_even_artifact(payload)
    return payload


def validate_uncertainty_break_even_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported N07/P03 schema.")
    if payload.get("method") != METHOD:
        raise ValueError("Unsupported N07/P03 method.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("N07/P03 fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("N07/P03 status does not match checks.")
    scope = payload.get("scope", {})
    if scope.get("state_preparation_cost_measured") is not False:
        raise ValueError("P03 cannot claim measured preparation cost.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("N07/P03 cannot claim final total cost.")
    if scope.get("scientific_superiority_claimed") is not False:
        raise ValueError("N07/P03 cannot claim scientific superiority.")


def write_uncertainty_break_even_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_uncertainty_break_even_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

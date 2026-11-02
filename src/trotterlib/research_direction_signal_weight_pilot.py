"""Screen PF energy-bias choices against target weight and coherent signal.

P-B is deliberately a small, no-new-physics-compute pilot.  It reuses the
committed H4 ``pf_delta_validation_v5`` artifacts and asks whether an
energy-only choice would be rejected by simple target-weight or signal-radius
criteria.  The result is a scoped discriminator, not a general statement
about product formulas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .parallel_validation_executor import atomic_write_json
from .pf_delta_validation import validate_pf_delta_payload
from .research_direction_full_scope import fingerprint


SCHEMA_VERSION = "research_direction_signal_weight_pilot_v1"
METHOD = "pb_existing_h4_pf_artifact_signal_weight_screening_v1"
STAGE = "P-B-signal-weight-screening"
EXPECTED_INPUT_SCHEMA = "pf_delta_validation_v5"
EXPECTED_LD_VALUES = tuple(range(12))
EXPECTED_DELTA_VALUES = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4)
EXPECTED_Q_VALUES = (1, 2, 4)

# Screening thresholds fixed for this pilot.  They are not universal QPE
# acceptance criteria.  w >= 0.995 implies the elementary 2w-1 radius lower
# bound is at least 0.99, matching the direct-signal screening threshold.
MINIMUM_TARGET_WEIGHT = 0.995
MINIMUM_Q_SIGNAL_RADIUS = 0.99
NEAR_TIE_BIAS_RELATIVE_MARGIN = 0.05
MEANINGFUL_LEAKAGE_REDUCTION_FACTOR = 2.0
MEANINGFUL_SIGNAL_RADIUS_IMPROVEMENT = 1.0e-3


def _float_tuple(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    norm_x = math.sqrt(sum(value * value for value in centered_x))
    norm_y = math.sqrt(sum(value * value for value in centered_y))
    if norm_x == 0.0 or norm_y == 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / (
        norm_x * norm_y
    )


def _candidate_row(
    payload: Mapping[str, Any], point: Mapping[str, Any]
) -> dict[str, Any]:
    spectrum = point["qpe_spectral_energy_distribution"]
    q_diagnostics = {
        int(result["q_m"]): result["single_dominant_phase_diagnostic"]
        for result in point["q_results"]
    }
    q_radii = {
        str(q_value): float(diagnostic["relative_physical_pf_signal_radius"])
        for q_value, diagnostic in sorted(q_diagnostics.items())
    }
    q_contaminations = {
        str(q_value): float(diagnostic["signal_phase_contamination"])
        for q_value, diagnostic in sorted(q_diagnostics.items())
    }
    q1_radius = q_radii["1"]
    state_action_loss_probability = max(0.0, 1.0 - q1_radius * q1_radius)
    target_weight = float(spectrum["dominant_phase_cluster_weight"])
    minimum_q_signal_radius = min(q_radii.values())
    row = {
        "ld": int(payload["request"]["ld"]),
        "delta": float(point["delta_time"]),
        "absolute_target_phase_bias": float(
            spectrum["dominant_phase_cluster_absolute_energy_bias"]
        ),
        "signed_target_phase_bias": float(
            spectrum["dominant_phase_cluster_signed_energy_bias"]
        ),
        "target_weight": target_weight,
        "target_weight_loss": max(0.0, 1.0 - target_weight),
        "qpe_energy_rmse": float(spectrum["qpe_energy_rmse"]),
        "qpe_energy_standard_deviation": float(
            spectrum["qpe_energy_standard_deviation"]
        ),
        "q_signal_radii": q_radii,
        "minimum_q_signal_radius": minimum_q_signal_radius,
        "maximum_q_signal_radius_loss": max(0.0, 1.0 - minimum_q_signal_radius),
        "q_signal_phase_contaminations": q_contaminations,
        "maximum_q_signal_phase_contamination": max(q_contaminations.values()),
        "q1_state_action_loss_probability": state_action_loss_probability,
        "q1_state_action_orthogonal_norm": math.sqrt(
            state_action_loss_probability
        ),
        "target_weight_screening_pass": target_weight >= MINIMUM_TARGET_WEIGHT,
        "q_signal_radius_screening_pass": (
            minimum_q_signal_radius >= MINIMUM_Q_SIGNAL_RADIUS
        ),
    }
    row["signal_screening_pass"] = bool(
        row["target_weight_screening_pass"]
        and row["q_signal_radius_screening_pass"]
    )
    return row


def _validate_input_context(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for payload in payloads:
        validate_pf_delta_payload(payload)
    lds = tuple(sorted(int(payload["request"]["ld"]) for payload in payloads))
    first = payloads[0]
    expected_hamiltonian = first["hamiltonian"]
    context_checks = {
        "input_count_12": len(payloads) == len(EXPECTED_LD_VALUES),
        "input_schema_v5": all(
            payload.get("schema_version") == EXPECTED_INPUT_SCHEMA
            for payload in payloads
        ),
        "ld_grid_exact": lds == EXPECTED_LD_VALUES,
        "h4_linear_chain": all(
            payload["hamiltonian"]["metadata"].get("molecule_type") == 4
            and float(payload["hamiltonian"]["metadata"].get("distance")) == 1.0
            for payload in payloads
        ),
        "sto3g_rank12_8qubit": all(
            payload["hamiltonian"]["metadata"].get("basis") == "sto-3g"
            and int(payload["hamiltonian"]["metadata"].get("df_rank_actual")) == 12
            and int(payload["hamiltonian"].get("n_qubits")) == 8
            for payload in payloads
        ),
        "same_hamiltonian_hash": all(
            payload["hamiltonian"]["hamiltonian_hash"]
            == expected_hamiltonian["hamiltonian_hash"]
            for payload in payloads
        ),
        "second_order_pf": all(
            payload["request"].get("product_formula") == "2nd"
            for payload in payloads
        ),
        "delta_grid_exact": all(
            _float_tuple(payload["request"]["validation_delta_times"])
            == EXPECTED_DELTA_VALUES
            for payload in payloads
        ),
        "q_grid_exact": all(
            tuple(int(value) for value in payload["request"]["q_values"])
            == EXPECTED_Q_VALUES
            for payload in payloads
        ),
        "required_weight_signal_numerics_pass": all(
            bool(payload["summary"][key])
            for payload in payloads
            for key in (
                "all_qpe_spectral_numerical_consistency_pass",
                "all_cpu_qiskit_matrix_consistency_pass",
                "all_signal_errors_within_numerical_operator_error",
                "all_signal_phase_contamination_within_analytic_bound",
                "all_signals_match_dominant_branch_within_tolerance",
            )
        ),
    }
    if not all(context_checks.values()):
        failed = sorted(key for key, value in context_checks.items() if not value)
        raise ValueError(f"P-B input context mismatch: {failed}")
    return context_checks


def evaluate_signal_weight_pilot(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate energy-only versus signal-screened choices on the fixed grid."""
    if not payloads:
        raise ValueError("P-B requires at least one PF validation artifact.")
    ordered = sorted(payloads, key=lambda payload: int(payload["request"]["ld"]))
    context_checks = _validate_input_context(ordered)
    rows = [
        _candidate_row(payload, point)
        for payload in ordered
        for point in payload["points"]
    ]
    rows.sort(key=lambda row: (row["delta"], row["ld"]))
    expected_row_count = len(EXPECTED_LD_VALUES) * len(EXPECTED_DELTA_VALUES)
    if len(rows) != expected_row_count:
        raise ValueError("P-B candidate grid has an unexpected row count.")

    per_delta: list[dict[str, Any]] = []
    for delta in EXPECTED_DELTA_VALUES:
        candidates = [row for row in rows if row["delta"] == delta]
        energy_choice = min(
            candidates, key=lambda row: (row["absolute_target_phase_bias"], row["ld"])
        )
        feasible = [row for row in candidates if row["signal_screening_pass"]]
        signal_choice = (
            min(
                feasible,
                key=lambda row: (row["absolute_target_phase_bias"], row["ld"]),
            )
            if feasible
            else None
        )
        near_tied = [
            row
            for row in candidates
            if row["absolute_target_phase_bias"]
            <= energy_choice["absolute_target_phase_bias"]
            * (1.0 + NEAR_TIE_BIAS_RELATIVE_MARGIN)
        ]
        best_near_tie_weight = max(
            near_tied, key=lambda row: (row["target_weight"], -row["ld"])
        )
        energy_leakage = energy_choice["target_weight_loss"]
        alternative_leakage = best_near_tie_weight["target_weight_loss"]
        leakage_reduction = (
            math.inf
            if alternative_leakage == 0.0 and energy_leakage > 0.0
            else (
                energy_leakage / alternative_leakage
                if alternative_leakage > 0.0
                else 1.0
            )
        )
        radius_improvement = (
            best_near_tie_weight["minimum_q_signal_radius"]
            - energy_choice["minimum_q_signal_radius"]
        )
        meaningful_near_tie_advantage = bool(
            best_near_tie_weight["ld"] != energy_choice["ld"]
            and leakage_reduction >= MEANINGFUL_LEAKAGE_REDUCTION_FACTOR
            and radius_improvement >= MEANINGFUL_SIGNAL_RADIUS_IMPROVEMENT
        )
        per_delta.append(
            {
                "delta": delta,
                "candidate_count": len(candidates),
                "signal_feasible_candidate_count": len(feasible),
                "energy_only_choice": {
                    "ld": energy_choice["ld"],
                    "absolute_target_phase_bias": energy_choice[
                        "absolute_target_phase_bias"
                    ],
                    "target_weight": energy_choice["target_weight"],
                    "minimum_q_signal_radius": energy_choice[
                        "minimum_q_signal_radius"
                    ],
                    "signal_screening_pass": energy_choice[
                        "signal_screening_pass"
                    ],
                },
                "signal_screened_energy_choice": (
                    None
                    if signal_choice is None
                    else {
                        "ld": signal_choice["ld"],
                        "absolute_target_phase_bias": signal_choice[
                            "absolute_target_phase_bias"
                        ],
                        "target_weight": signal_choice["target_weight"],
                        "minimum_q_signal_radius": signal_choice[
                            "minimum_q_signal_radius"
                        ],
                    }
                ),
                "selection_changed": bool(
                    signal_choice is None or signal_choice["ld"] != energy_choice["ld"]
                ),
                "near_tie_candidate_count": len(near_tied),
                "best_near_tie_weight_candidate_ld": best_near_tie_weight["ld"],
                "near_tie_leakage_reduction_factor": leakage_reduction,
                "near_tie_signal_radius_improvement": radius_improvement,
                "meaningful_near_tie_signal_advantage": meaningful_near_tie_advantage,
            }
        )

    inversions: list[dict[str, Any]] = []
    for delta in EXPECTED_DELTA_VALUES:
        candidates = [row for row in rows if row["delta"] == delta]
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                if left["absolute_target_phase_bias"] == right[
                    "absolute_target_phase_bias"
                ]:
                    continue
                energy_better, energy_worse = (
                    (left, right)
                    if left["absolute_target_phase_bias"]
                    < right["absolute_target_phase_bias"]
                    else (right, left)
                )
                if energy_better["target_weight"] >= energy_worse["target_weight"]:
                    continue
                better_leakage = energy_better["target_weight_loss"]
                worse_leakage = energy_worse["target_weight_loss"]
                leakage_factor = (
                    better_leakage / worse_leakage
                    if worse_leakage > 0.0
                    else math.inf
                )
                radius_disadvantage = (
                    energy_worse["minimum_q_signal_radius"]
                    - energy_better["minimum_q_signal_radius"]
                )
                inversions.append(
                    {
                        "delta": delta,
                        "energy_better_ld": energy_better["ld"],
                        "energy_worse_ld": energy_worse["ld"],
                        "energy_bias_ratio": (
                            energy_worse["absolute_target_phase_bias"]
                            / energy_better["absolute_target_phase_bias"]
                        ),
                        "energy_better_target_weight": energy_better["target_weight"],
                        "energy_worse_target_weight": energy_worse["target_weight"],
                        "energy_better_leakage_disadvantage_factor": leakage_factor,
                        "energy_better_signal_radius_disadvantage": radius_disadvantage,
                        "meaningful": bool(
                            leakage_factor >= MEANINGFUL_LEAKAGE_REDUCTION_FACTOR
                            and radius_disadvantage
                            >= MEANINGFUL_SIGNAL_RADIUS_IMPROVEMENT
                        ),
                    }
                )
    largest_inversion = (
        max(
            inversions,
            key=lambda row: (
                row["energy_better_leakage_disadvantage_factor"],
                row["energy_better_signal_radius_disadvantage"],
            ),
        )
        if inversions
        else None
    )

    weight_losses = [row["target_weight_loss"] for row in rows]
    action_losses = [row["q1_state_action_loss_probability"] for row in rows]
    positive_pairs = [
        (weight_loss, action_loss)
        for weight_loss, action_loss in zip(weight_losses, action_losses)
        if weight_loss > 0.0 and action_loss > 0.0
    ]
    log_correlation = _pearson(
        [math.log10(pair[0]) for pair in positive_pairs],
        [math.log10(pair[1]) for pair in positive_pairs],
    )
    minimum_weight_row = min(rows, key=lambda row: row["target_weight"])
    minimum_radius_row = min(rows, key=lambda row: row["minimum_q_signal_radius"])
    maximum_contamination_row = max(
        rows, key=lambda row: row["maximum_q_signal_phase_contamination"]
    )
    selection_disagreement = any(row["selection_changed"] for row in per_delta)
    meaningful_inversion = any(row["meaningful"] for row in inversions)
    meaningful_near_tie_advantage = any(
        row["meaningful_near_tie_signal_advantage"] for row in per_delta
    )
    hypothesis_supported = bool(
        selection_disagreement
        or meaningful_inversion
        or meaningful_near_tie_advantage
    )

    checks = {
        "input_artifacts_validate": True,
        "fixed_context_matches": all(context_checks.values()),
        "candidate_grid_complete": len(rows) == expected_row_count,
        "all_rows_finite": all(
            all(
                math.isfinite(float(row[key]))
                for key in (
                    "absolute_target_phase_bias",
                    "target_weight",
                    "minimum_q_signal_radius",
                    "q1_state_action_orthogonal_norm",
                )
            )
            for row in rows
        ),
        "all_energy_choices_signal_feasible": all(
            row["energy_only_choice"]["signal_screening_pass"]
            for row in per_delta
        ),
    }
    return {
        "configuration": {
            "molecule": "H4 linear chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "ld_values": list(EXPECTED_LD_VALUES),
            "delta_values": list(EXPECTED_DELTA_VALUES),
            "q_values": list(EXPECTED_Q_VALUES),
            "product_formula": "second_order_partial_S2_with_exact_tail_reference",
            "candidate_count": len(rows),
            "screening_thresholds": {
                "minimum_target_weight": MINIMUM_TARGET_WEIGHT,
                "minimum_q_signal_radius": MINIMUM_Q_SIGNAL_RADIUS,
                "near_tie_bias_relative_margin": NEAR_TIE_BIAS_RELATIVE_MARGIN,
                "meaningful_leakage_reduction_factor": (
                    MEANINGFUL_LEAKAGE_REDUCTION_FACTOR
                ),
                "meaningful_signal_radius_improvement": (
                    MEANINGFUL_SIGNAL_RADIUS_IMPROVEMENT
                ),
            },
        },
        "context_checks": context_checks,
        "candidate_rows": rows,
        "per_delta_selection": per_delta,
        "pairwise_energy_weight_ordering_inversions": inversions,
        "diagnostic": {
            "candidate": (
                "one-step state-action orthogonal norm derived from "
                "sqrt(1-|<psi|S(delta)|psi>|^2)"
            ),
            "log10_weight_loss_vs_log10_state_action_loss_pearson": log_correlation,
            "validated_as_selection_discriminator": False,
            "reason": (
                "The reference grid contains no practical weight or signal "
                "failure and no signal-aware selection change, so correlation "
                "alone cannot validate discriminative utility."
            ),
        },
        "summary": {
            "minimum_target_weight": dict(minimum_weight_row),
            "minimum_q_signal_radius": dict(minimum_radius_row),
            "maximum_q_signal_phase_contamination": dict(
                maximum_contamination_row
            ),
            "energy_signal_selection_disagreement_count": sum(
                int(row["selection_changed"]) for row in per_delta
            ),
            "pairwise_ordering_inversion_count": len(inversions),
            "meaningful_pairwise_ordering_inversion_count": sum(
                int(row["meaningful"]) for row in inversions
            ),
            "meaningful_near_tie_signal_advantage_count": sum(
                int(row["meaningful_near_tie_signal_advantage"])
                for row in per_delta
            ),
            "largest_pairwise_ordering_inversion": largest_inversion,
            "all_candidates_pass_signal_screening": all(
                row["signal_screening_pass"] for row in rows
            ),
            "pb_hypothesis_supported_in_scope": hypothesis_supported,
        },
        "decision": {
            "status": (
                "advance_pb"
                if hypothesis_supported
                else "do_not_advance_pb_on_current_h4_s2_prefix_grid"
            ),
            "pb_primary_theme_recommended": hypothesis_supported,
            "broader_signal_weight_research_rejected": False,
            "interpretation": (
                "Within the fixed H4 second-order partial-S2 prefix grid, "
                "energy-only choices remain signal-feasible and no meaningful "
                "energy/weight selection disagreement is observed. This meets "
                "the pilot stop condition for P-B in this scope, but is not a "
                "general no-go result for other PF families, gaps, or systems."
            ),
            "next_pilot": "P-C-geometry-signed-error",
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
        "scope": {
            "new_hamiltonian_or_pf_computation_performed": False,
            "new_circuit_compilation_performed": False,
            "existing_dense_reference_artifacts_reanalyzed": True,
            "state_action_candidate_uses_q1_overlap_only": True,
            "rte_redesign_performed": False,
            "compiler_remeasurement_performed": False,
            "p_b_general_hypothesis_rejected": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_signal_weight_pilot_artifact(
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
    validate_signal_weight_pilot_artifact(payload)
    return payload


def validate_signal_weight_pilot_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported P-B signal-weight schema.")
    if payload.get("method") != METHOD or payload.get("stage") != STAGE:
        raise ValueError("Unsupported P-B signal-weight method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-B signal-weight artifact fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("P-B signal-weight status does not match checks.")
    if payload.get("configuration", {}).get("candidate_count") != 72:
        raise ValueError("P-B artifact must contain the fixed 72-candidate grid.")
    scope = payload.get("scope", {})
    if scope.get("new_hamiltonian_or_pf_computation_performed") is not False:
        raise ValueError("P-B artifact cannot claim a new PF computation.")
    if scope.get("p_b_general_hypothesis_rejected") is not False:
        raise ValueError("P-B local pilot cannot reject the general hypothesis.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("P-B pilot cannot claim final total cost.")
    if scope.get("scientific_superiority_claimed") is not False:
        raise ValueError("P-B pilot cannot claim scientific superiority.")


def write_signal_weight_pilot_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_signal_weight_pilot_artifact(payload)
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing artifact: {output}")
    atomic_write_json(output, payload)


def read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload

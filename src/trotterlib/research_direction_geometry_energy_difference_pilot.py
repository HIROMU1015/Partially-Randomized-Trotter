"""P-C pilot for geometry-dependent signed PF energy errors.

The pilot keeps one H4/STO-3G/DF-rank-12/partial-S2 construction fixed and
asks whether a leading signed error coefficient can predict Product Formula
errors at unused geometries and an unused delta.  It is a theme-selection
experiment, not a potential-energy-surface or total-resource calculation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .parallel_validation_executor import atomic_write_json
from .pf_delta_validation import validate_pf_delta_payload
from .research_direction_full_scope import fingerprint


SCHEMA_VERSION = "research_direction_geometry_energy_difference_pilot_v1"
METHOD = "pc_h4_geometry_signed_pf_error_interpolation_v1"
STAGE = "P-C-geometry-signed-error"
EXPECTED_INPUT_SCHEMA = "pf_delta_validation_v5"
EXPECTED_GEOMETRIES = (0.8, 0.85, 1.0, 1.15, 1.2)
TRAINING_GEOMETRIES = (0.8, 1.0, 1.2)
HOLDOUT_GEOMETRIES = (0.85, 1.15)
FIT_DELTAS = (0.0125, 0.025, 0.05, 0.1, 0.2)
DELTA_HOLDOUT = 0.4
PAIR_REPORT_DELTA = 0.1

# Pilot gates fixed before generating the 0.80/0.85/1.15/1.20 Angstrom
# artifacts.  A discarded 0.90 Angstrom execution was used only to verify
# runtime and is not part of the evidence set.
MAXIMUM_GEOMETRY_HOLDOUT_COEFFICIENT_RELATIVE_ERROR = 0.10
MAXIMUM_DELTA_HOLDOUT_BIAS_RELATIVE_ERROR = 0.10
MAXIMUM_COMBINED_PAIR_NORMALIZED_ERROR = 0.10
MINIMUM_COEFFICIENT_SPAN_FRACTION = 0.10
MAXIMUM_MEANINGFUL_CANCELLATION_RATIO = 0.50
NUMERICAL_ATOL = 1.0e-14


def _relative_error(predicted: float, actual: float) -> float:
    return float(abs(predicted - actual) / max(abs(actual), NUMERICAL_ATOL))


def _fit_signed_second_order_coefficient(
    points: Mapping[float, float],
) -> float:
    numerator = sum(
        delta * delta * float(points[delta]) for delta in FIT_DELTAS
    )
    denominator = sum(delta**4 for delta in FIT_DELTAS)
    if denominator <= 0.0:
        raise ValueError("P-C second-order fit has an invalid denominator.")
    return float(numerator / denominator)


def _interpolate_training_coefficient(
    geometry: float, training: Mapping[float, float]
) -> float:
    if geometry in training:
        return float(training[geometry])
    ordered = sorted(training)
    for left, right in zip(ordered, ordered[1:]):
        if left < geometry < right:
            fraction = (geometry - left) / (right - left)
            return float(
                training[left]
                + fraction * (training[right] - training[left])
            )
    raise ValueError("P-C holdout geometry is outside the training domain.")


def _validate_input_context(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    for payload in payloads:
        validate_pf_delta_payload(payload)
    geometries = tuple(
        sorted(
            float(payload["hamiltonian"]["metadata"]["distance"])
            for payload in payloads
        )
    )
    checks = {
        "input_count_5": len(payloads) == len(EXPECTED_GEOMETRIES),
        "input_schema_v5": all(
            payload.get("schema_version") == EXPECTED_INPUT_SCHEMA
            for payload in payloads
        ),
        "geometry_grid_exact": geometries == EXPECTED_GEOMETRIES,
        "h4_linear_chain": all(
            int(payload["hamiltonian"]["metadata"].get("molecule_type")) == 4
            for payload in payloads
        ),
        "sto3g_rank12_8qubit": all(
            payload["hamiltonian"]["metadata"].get("basis") == "sto-3g"
            and int(payload["hamiltonian"]["metadata"].get("df_rank_actual"))
            == 12
            and int(payload["hamiltonian"].get("n_qubits")) == 8
            for payload in payloads
        ),
        "distinct_geometry_hamiltonians": len(
            {
                payload["hamiltonian"]["hamiltonian_hash"]
                for payload in payloads
            }
        )
        == len(EXPECTED_GEOMETRIES),
        "fixed_ld3_second_order_pf": all(
            int(payload["request"].get("ld")) == 3
            and payload["request"].get("product_formula") == "2nd"
            for payload in payloads
        ),
        "delta_grid_exact": all(
            tuple(
                float(value)
                for value in payload["request"]["validation_delta_times"]
            )
            == (*FIT_DELTAS, DELTA_HOLDOUT)
            for payload in payloads
        ),
        "q_grid_exact": all(
            tuple(int(value) for value in payload["request"]["q_values"])
            == (1, 2, 4)
            for payload in payloads
        ),
        "required_signed_bias_numerics_pass": all(
            bool(payload["summary"][key])
            for payload in payloads
            for key in (
                "all_qpe_spectral_numerical_consistency_pass",
                "all_cpu_qiskit_matrix_consistency_pass",
                "all_signal_phase_contamination_within_analytic_bound",
                "all_signals_match_dominant_branch_within_tolerance",
                "single_dominant_phase_approximation_validation_pass",
            )
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise ValueError(f"P-C input context mismatch: {failed}")
    return checks


def _geometry_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    geometry = float(payload["hamiltonian"]["metadata"]["distance"])
    biases = {
        float(point["delta_time"]): float(
            point["qpe_spectral_energy_distribution"][
                "dominant_phase_cluster_signed_energy_bias"
            ]
        )
        for point in payload["points"]
    }
    if tuple(sorted(biases)) != (*FIT_DELTAS, DELTA_HOLDOUT):
        raise ValueError("P-C input contains an unexpected delta grid.")
    coefficient = _fit_signed_second_order_coefficient(biases)
    delta_holdout_prediction = coefficient * DELTA_HOLDOUT**2
    delta_holdout_actual = biases[DELTA_HOLDOUT]
    return {
        "geometry_angstrom": geometry,
        "role": (
            "coefficient_training"
            if geometry in TRAINING_GEOMETRIES
            else "geometry_holdout"
        ),
        "exact_df_rank12_ground_energy_hartree": float(
            payload["hamiltonian"]["ground_energy"]
        ),
        "hamiltonian_hash": payload["hamiltonian"]["hamiltonian_hash"],
        "signed_target_phase_bias_by_delta_hartree": {
            str(delta): biases[delta] for delta in sorted(biases)
        },
        "signed_second_order_coefficient_hartree": coefficient,
        "delta_holdout": {
            "delta": DELTA_HOLDOUT,
            "actual_signed_bias_hartree": delta_holdout_actual,
            "predicted_signed_bias_hartree": delta_holdout_prediction,
            "relative_error": _relative_error(
                delta_holdout_prediction, delta_holdout_actual
            ),
        },
    }


def _pair_row(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    delta: float,
    coefficient_predictions: Mapping[float, float],
) -> dict[str, Any]:
    left_geometry = float(left["geometry_angstrom"])
    right_geometry = float(right["geometry_angstrom"])
    left_bias = float(left["signed_target_phase_bias_by_delta_hartree"][str(delta)])
    right_bias = float(
        right["signed_target_phase_bias_by_delta_hartree"][str(delta)]
    )
    actual_error = right_bias - left_bias
    predicted_error = delta**2 * (
        coefficient_predictions[right_geometry]
        - coefficient_predictions[left_geometry]
    )
    endpoint_scale = max(abs(left_bias), abs(right_bias), NUMERICAL_ATOL)
    exact_difference = float(
        right["exact_df_rank12_ground_energy_hartree"]
        - left["exact_df_rank12_ground_energy_hartree"]
    )
    return {
        "left_geometry_angstrom": left_geometry,
        "right_geometry_angstrom": right_geometry,
        "delta": delta,
        "exact_df_rank12_energy_difference_hartree": exact_difference,
        "pf_energy_difference_hartree": exact_difference + actual_error,
        "actual_signed_pf_difference_error_hartree": actual_error,
        "predicted_signed_pf_difference_error_hartree": predicted_error,
        "prediction_absolute_error_hartree": abs(predicted_error - actual_error),
        "prediction_error_normalized_by_endpoint_bias": float(
            abs(predicted_error - actual_error) / endpoint_scale
        ),
        "prediction_relative_error_to_difference": _relative_error(
            predicted_error, actual_error
        ),
        "signed_error_cancellation_ratio": float(abs(actual_error) / endpoint_scale),
        "uses_geometry_holdout": bool(
            left_geometry in HOLDOUT_GEOMETRIES
            or right_geometry in HOLDOUT_GEOMETRIES
        ),
        "uses_delta_holdout": delta == DELTA_HOLDOUT,
    }


def evaluate_geometry_energy_difference_pilot(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate smooth signed-error and energy-difference prediction gates."""
    if not payloads:
        raise ValueError("P-C requires five PF validation artifacts.")
    context_checks = _validate_input_context(payloads)
    rows = sorted(
        (_geometry_row(payload) for payload in payloads),
        key=lambda row: row["geometry_angstrom"],
    )
    by_geometry = {float(row["geometry_angstrom"]): row for row in rows}
    training = {
        geometry: float(
            by_geometry[geometry]["signed_second_order_coefficient_hartree"]
        )
        for geometry in TRAINING_GEOMETRIES
    }
    coefficient_predictions = {
        geometry: _interpolate_training_coefficient(geometry, training)
        for geometry in EXPECTED_GEOMETRIES
    }
    geometry_holdouts = []
    for geometry in HOLDOUT_GEOMETRIES:
        actual = float(
            by_geometry[geometry]["signed_second_order_coefficient_hartree"]
        )
        predicted = coefficient_predictions[geometry]
        geometry_holdouts.append(
            {
                "geometry_angstrom": geometry,
                "actual_signed_second_order_coefficient_hartree": actual,
                "predicted_signed_second_order_coefficient_hartree": predicted,
                "relative_error": _relative_error(predicted, actual),
            }
        )

    adjacent_pairs = [
        _pair_row(
            by_geometry[left],
            by_geometry[right],
            delta=PAIR_REPORT_DELTA,
            coefficient_predictions=coefficient_predictions,
        )
        for left, right in zip(EXPECTED_GEOMETRIES, EXPECTED_GEOMETRIES[1:])
    ]
    combined_holdout_pair = _pair_row(
        by_geometry[HOLDOUT_GEOMETRIES[0]],
        by_geometry[HOLDOUT_GEOMETRIES[1]],
        delta=DELTA_HOLDOUT,
        coefficient_predictions=coefficient_predictions,
    )
    coefficient_values = [
        float(row["signed_second_order_coefficient_hartree"]) for row in rows
    ]
    coefficient_span_fraction = float(
        (max(coefficient_values) - min(coefficient_values))
        / max(max(abs(value) for value in coefficient_values), NUMERICAL_ATOL)
    )
    maximum_geometry_holdout_error = max(
        row["relative_error"] for row in geometry_holdouts
    )
    maximum_delta_holdout_error = max(
        row["delta_holdout"]["relative_error"] for row in rows
    )
    minimum_adjacent_cancellation_ratio = min(
        row["signed_error_cancellation_ratio"] for row in adjacent_pairs
    )
    advance = bool(
        maximum_geometry_holdout_error
        <= MAXIMUM_GEOMETRY_HOLDOUT_COEFFICIENT_RELATIVE_ERROR
        and maximum_delta_holdout_error
        <= MAXIMUM_DELTA_HOLDOUT_BIAS_RELATIVE_ERROR
        and combined_holdout_pair[
            "prediction_error_normalized_by_endpoint_bias"
        ]
        <= MAXIMUM_COMBINED_PAIR_NORMALIZED_ERROR
        and coefficient_span_fraction >= MINIMUM_COEFFICIENT_SPAN_FRACTION
        and minimum_adjacent_cancellation_ratio
        <= MAXIMUM_MEANINGFUL_CANCELLATION_RATIO
    )
    checks = {
        "input_artifacts_validate": True,
        "fixed_context_matches": all(context_checks.values()),
        "geometry_grid_complete": len(rows) == len(EXPECTED_GEOMETRIES),
        "all_primary_values_finite": all(
            math.isfinite(value)
            for value in (
                *coefficient_values,
                maximum_geometry_holdout_error,
                maximum_delta_holdout_error,
                minimum_adjacent_cancellation_ratio,
                coefficient_span_fraction,
                combined_holdout_pair[
                    "prediction_error_normalized_by_endpoint_bias"
                ],
            )
        ),
    }
    return {
        "configuration": {
            "molecule": "H4 linear chain",
            "geometry_grid_angstrom": list(EXPECTED_GEOMETRIES),
            "coefficient_training_geometries_angstrom": list(
                TRAINING_GEOMETRIES
            ),
            "geometry_holdouts_angstrom": list(HOLDOUT_GEOMETRIES),
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "ld": 3,
            "product_formula": "second_order_partial_S2_exact_tail_reference",
            "coefficient_fit_deltas": list(FIT_DELTAS),
            "delta_holdout": DELTA_HOLDOUT,
            "pair_report_delta": PAIR_REPORT_DELTA,
            "gates": {
                "maximum_geometry_holdout_coefficient_relative_error": (
                    MAXIMUM_GEOMETRY_HOLDOUT_COEFFICIENT_RELATIVE_ERROR
                ),
                "maximum_delta_holdout_bias_relative_error": (
                    MAXIMUM_DELTA_HOLDOUT_BIAS_RELATIVE_ERROR
                ),
                "maximum_combined_pair_normalized_error": (
                    MAXIMUM_COMBINED_PAIR_NORMALIZED_ERROR
                ),
                "minimum_coefficient_span_fraction": (
                    MINIMUM_COEFFICIENT_SPAN_FRACTION
                ),
                "maximum_meaningful_cancellation_ratio": (
                    MAXIMUM_MEANINGFUL_CANCELLATION_RATIO
                ),
            },
        },
        "context_checks": context_checks,
        "geometry_rows": rows,
        "geometry_coefficient_holdouts": geometry_holdouts,
        "adjacent_pair_results_at_delta_0p1": adjacent_pairs,
        "combined_geometry_and_delta_holdout_pair": combined_holdout_pair,
        "summary": {
            "coefficient_minimum_hartree": min(coefficient_values),
            "coefficient_maximum_hartree": max(coefficient_values),
            "coefficient_span_fraction": coefficient_span_fraction,
            "maximum_geometry_holdout_coefficient_relative_error": (
                maximum_geometry_holdout_error
            ),
            "maximum_delta_holdout_bias_relative_error": (
                maximum_delta_holdout_error
            ),
            "combined_pair_prediction_error_normalized_by_endpoint_bias": (
                combined_holdout_pair[
                    "prediction_error_normalized_by_endpoint_bias"
                ]
            ),
            "minimum_adjacent_signed_error_cancellation_ratio": (
                minimum_adjacent_cancellation_ratio
            ),
            "pc_hypothesis_supported_in_scope": advance,
        },
        "decision": {
            "status": (
                "advance_pc_to_geometry_difference_design"
                if advance
                else "do_not_advance_pc_on_current_h4_geometry_grid"
            ),
            "pc_primary_theme_candidate": advance,
            "interpretation": (
                "This fixed H4 grid tests whether a signed leading PF-error "
                "curve predicts held-out geometries and a held-out delta, "
                "and whether energy differences cancel endpoint PF errors. "
                "It does not establish transfer beyond this geometry domain."
            ),
            "next_pilot": "P-A-joint-circuit-sequence-synthesis",
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
        "scope": {
            "new_hamiltonian_and_pf_computations_performed": True,
            "new_circuit_compilation_performed": False,
            "exact_energy_means_df_rank12_sector_reference": True,
            "potential_energy_surface_claimed": False,
            "geometry_transfer_beyond_0p8_to_1p2_angstrom_claimed": False,
            "rpe_or_rte_cost_evaluation_performed": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_geometry_energy_difference_pilot_artifact(
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
    validate_geometry_energy_difference_pilot_artifact(payload)
    return payload


def validate_geometry_energy_difference_pilot_artifact(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported P-C geometry pilot schema.")
    if payload.get("method") != METHOD or payload.get("stage") != STAGE:
        raise ValueError("Unsupported P-C geometry pilot method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-C geometry pilot artifact fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("P-C geometry pilot status does not match checks.")
    if len(payload.get("geometry_rows", [])) != len(EXPECTED_GEOMETRIES):
        raise ValueError("P-C artifact must contain the fixed geometry grid.")
    scope = payload.get("scope", {})
    if scope.get("exact_energy_means_df_rank12_sector_reference") is not True:
        raise ValueError("P-C exact-energy scope must remain DF-rank-12 specific.")
    for key in (
        "potential_energy_surface_claimed",
        "geometry_transfer_beyond_0p8_to_1p2_angstrom_claimed",
        "rpe_or_rte_cost_evaluation_performed",
        "final_total_cost_evaluation_performed",
        "scientific_superiority_claimed",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"P-C artifact overstates scope: {key}.")


def write_geometry_energy_difference_pilot_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_geometry_energy_difference_pilot_artifact(payload)
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing artifact: {output}")
    atomic_write_json(output, payload)


def read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload

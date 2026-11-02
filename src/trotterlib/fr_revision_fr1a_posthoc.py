"""Posthoc positive-scalar reanalysis of the completed finite-RTE FR-1 grid.

The analysis contract is frozen in
``docs/research/fr_revision_fr1a_posthoc_plan.md``.  This module only
reconstructs the existing 33 two-dimensional FR-1 conditions.  It does not
change the original gates or decision and it does not start FR-R1b.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import minimize_scalar

from .finite_rte_phase_amplitude import (
    _compose_local_error,
    _condition_record,
    _fingerprint,
    _norm_bound,
    _proposal_bound,
    _scalar_relative_error,
    validate_finite_rte_phase_amplitude_payload,
)


FR1A_SCHEMA_VERSION = "fr_revision_fr1a_posthoc_v1"
FR1A_EXPECTED_SCHEMA_VERSION = "fr_revision_fr1a_expected_v1"
FR1A_METHOD = "positive_scalar_posthoc_reanalysis_v1"
SOURCE_RESULT_FINGERPRINT = (
    "d96b200163f3a432652656ae97c65c837e01fe81323cd69528373dff16ce6152"
)
SOURCE_RESULT_SHA256 = (
    "6a81a0ba6e39ba0f5d79ba026a2a45c3c65709e071e9c6fa5606b3e99a89b0f7"
)
SOURCE_PREREGISTRATION_SHA256 = (
    "bc8066d7a31a3f46f476d2c591c7f0dad5e6f49023fc261dc518b61e2ec600a3"
)
POSTHOC_PLAN_SHA256 = (
    "a0aa2e75d2e304e008646f138211ed1e4da4d98def697f8503b07c3e7b56c95f"
)
SOURCE_CONDITION_COUNT = 33
SOURCE_STATE_COUNT = 99
SOURCE_APPLICABLE_METHOD_COUNT = 495
SOURCE_DECISION = "GO_FR2_MECHANISM_ONLY"
PHASE_BUDGETS = (1e-2, 1e-3, 1e-4)
RADIUS_FLOOR = 0.2
GAMMA_MIN = 0.5
GAMMA_MAX = 1.5
OPTIMIZER_ATOL = 1e-12
COMPARISON_ATOL = 1e-12

METHOD_ORDER = (
    "OLD_NORM",
    "STRONG_NORM",
    "OLD_FR",
    "SCALAR_NORM_COMMON",
    "SCALAR_FR_COMMON",
    "OPT_SCALAR_NORM",
    "OPT_SCALAR_FR",
    "INVOLUTION_POLAR",
    "DENSE_ORACLE",
)


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _complex_from_payload(payload: Mapping[str, Any]) -> complex:
    return complex(float(payload["real"]), float(payload["imag"]))


def _maximum_numeric_difference(left: Any, right: Any) -> float:
    """Compare reconstructed scientific records while requiring equal structure."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max(
            (_maximum_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max(
            (_maximum_numeric_difference(a, b) for a, b in zip(left, right, strict=True)),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left is right else math.inf
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def _audit_source(
    source: Mapping[str, Any],
    *,
    source_result_path: str | Path,
    frozen_preregistration_path: str | Path,
    posthoc_plan_path: str | Path,
) -> dict[str, Any]:
    validate_finite_rte_phase_amplitude_payload(source)
    result_sha = _file_sha256(source_result_path)
    prereg_sha = _file_sha256(frozen_preregistration_path)
    plan_sha = _file_sha256(posthoc_plan_path)
    summary = source.get("summary", {})
    checks = {
        "result_fingerprint_matches": bool(
            source.get("validation_fingerprint") == SOURCE_RESULT_FINGERPRINT
        ),
        "result_file_sha256_matches": bool(result_sha == SOURCE_RESULT_SHA256),
        "frozen_preregistration_sha256_matches": bool(
            prereg_sha == SOURCE_PREREGISTRATION_SHA256
        ),
        "posthoc_plan_sha256_matches": bool(plan_sha == POSTHOC_PLAN_SHA256),
        "condition_count_matches": bool(
            summary.get("condition_count") == SOURCE_CONDITION_COUNT
        ),
        "state_count_matches": bool(summary.get("state_record_count") == SOURCE_STATE_COUNT),
        "applicable_method_count_matches": bool(
            summary.get("applicable_method_record_count")
            == SOURCE_APPLICABLE_METHOD_COUNT
        ),
        "old_decision_matches": bool(summary.get("decision") == SOURCE_DECISION),
        "old_gates_preserved": bool(
            source.get("gates")
            == {
                "G0_semantic_consistency": True,
                "G1_all_applicable_bounds_sound": True,
                "G2_available_noncommuting_utility": False,
                "G3_conditioning_and_rejection": True,
                "G4_sign_cutoff_and_asymmetry_sound": True,
            }
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"FR-R1a frozen-source audit failed: {failed}")
    return {
        "checks": checks,
        "all_pass": True,
        "result_path": str(source_result_path),
        "result_fingerprint": source["validation_fingerprint"],
        "result_file_sha256": result_sha,
        "frozen_preregistration_path": str(frozen_preregistration_path),
        "frozen_preregistration_sha256": prereg_sha,
        "posthoc_plan_path": str(posthoc_plan_path),
        "posthoc_plan_sha256": plan_sha,
    }


def _reconstruct_source(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reconstructed: list[dict[str, Any]] = []
    maximum_difference = 0.0
    state_fingerprint_matches = 0
    for stored in source["conditions"]:
        regenerated = _condition_record(stored["condition"])
        difference = _maximum_numeric_difference(stored, regenerated)
        maximum_difference = max(maximum_difference, difference)
        if not math.isfinite(difference) or difference > 2e-12:
            raise ValueError(
                "FR-R1a deterministic source reconstruction mismatch for "
                f"{stored['condition']['condition_id']}: {difference}"
            )
        for old_state, new_state in zip(
            stored["state_records"], regenerated["state_records"], strict=True
        ):
            if old_state["state_fingerprint"] != new_state["state_fingerprint"]:
                raise ValueError(
                    "FR-R1a state fingerprint mismatch for "
                    f"{old_state['condition_id']} / {old_state['state_label']}"
                )
            state_fingerprint_matches += 1
        reconstructed.append(regenerated)
    return reconstructed, {
        "condition_count": len(reconstructed),
        "state_count": state_fingerprint_matches,
        "maximum_numeric_abs_difference": float(maximum_difference),
        "all_state_fingerprints_match": bool(
            state_fingerprint_matches == SOURCE_STATE_COUNT
        ),
        "all_pass": True,
    }


def _local_scalar_data(short_time: float, cutoff: int, occurrence_count: int) -> dict[str, Any]:
    values = np.asarray(
        [
            1.0 + _scalar_relative_error(-abs(short_time), cutoff),
            1.0 + _scalar_relative_error(abs(short_time), cutoff),
        ],
        dtype=np.complex128,
    )
    real_values = values.real
    f_minus = float(np.min(real_values) - 1.0)
    f_plus = float(np.max(real_values) - 1.0)
    c_mid = float((f_minus + f_plus) / 2.0)
    gamma_mid = float(1.0 + c_mid)
    if gamma_mid <= 0.0:
        raise ValueError("FR-R1a midpoint scalar is not positive.")
    representative = complex(values[np.argmax(values.imag)])
    gamma_norm_raw = float(abs(representative) ** 2 / representative.real)
    gamma_norm = float(min(GAMMA_MAX, max(GAMMA_MIN, gamma_norm_raw)))
    gamma_polar = float(abs(representative))
    return {
        "spectral_values": [
            {"real": float(value.real), "imag": float(value.imag)} for value in values
        ],
        "f_minus": f_minus,
        "f_plus": f_plus,
        "c_mid": c_mid,
        "gamma_mid": gamma_mid,
        "gamma_mid_product": float(gamma_mid**occurrence_count),
        "centered_hermitian_spectral_width": float(f_plus - f_minus),
        "gamma_opt_norm": gamma_norm,
        "gamma_opt_norm_unclipped": gamma_norm_raw,
        "gamma_opt_norm_product": float(gamma_norm**occurrence_count),
        "gamma_opt_norm_boundary_inconclusive": bool(
            abs(gamma_norm - GAMMA_MIN) <= OPTIMIZER_ATOL
            or abs(gamma_norm - GAMMA_MAX) <= OPTIMIZER_ATOL
        ),
        "gamma_opt_norm_certification": {
            "method": "analytic_two_point_involution_minimax",
            "absolute_tolerance": OPTIMIZER_ATOL,
            "finite_grid_maximum_used": False,
        },
        "gamma_involution_polar": gamma_polar,
        "gamma_involution_polar_product": float(gamma_polar**occurrence_count),
    }


def _relative_metrics(
    spectral_values: list[Mapping[str, Any]], gamma: float, occurrence_count: int
) -> dict[str, float]:
    values = np.asarray(
        [complex(float(v["real"]), float(v["imag"])) for v in spectral_values],
        dtype=np.complex128,
    )
    relative = values / float(gamma) - 1.0
    local_a = float(np.max(np.abs(relative.real)))
    local_b = float(np.max(np.abs(relative.imag)))
    local_e = float(np.max(np.abs(relative)))
    composed = _compose_local_error(local_e, occurrence_count)
    remainder = max(0.0, float(composed - occurrence_count * local_e))
    return {
        "a": local_a,
        "b": local_b,
        "e": local_e,
        "composed_error": float(composed),
        "product_remainder": remainder,
    }


def _add_physical_scaling(
    bound: Mapping[str, Any], *, gamma_product: float, normalization_product: float
) -> dict[str, Any]:
    output = deepcopy(dict(bound))
    output["gamma_product"] = float(gamma_product)
    output["normalization_product"] = float(normalization_product)
    output["physical_scalar_factor"] = float(gamma_product / normalization_product)
    if output["applicable"]:
        output["observed_radius_lower_bound"] = float(
            output["observed_radius_lower_bound"] * gamma_product
        )
        output["inapplicable_reason"] = None
    else:
        output["inapplicable_reason"] = "phase_or_radius_condition_failed"
    return output


def _scalar_norm_bound(
    *,
    rho: float,
    gamma: float,
    occurrence_count: int,
    spectral_values: list[Mapping[str, Any]],
    normalization_product: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    metrics = _relative_metrics(spectral_values, gamma, occurrence_count)
    raw = _norm_bound(
        rho_input=rho,
        composed_error=metrics["composed_error"],
        normalization_product=normalization_product,
    )
    return (
        _add_physical_scaling(
            raw,
            gamma_product=gamma**occurrence_count,
            normalization_product=normalization_product,
        ),
        metrics,
    )


def _scalar_fr_bound(
    *,
    rho: float,
    gamma: float,
    occurrence_count: int,
    spectral_values: list[Mapping[str, Any]],
    normalization_product: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    metrics = _relative_metrics(spectral_values, gamma, occurrence_count)
    raw = _proposal_bound(
        rho_input=rho,
        a_sum=occurrence_count * metrics["a"],
        b_sum=occurrence_count * metrics["b"],
        e_sum=occurrence_count * metrics["e"],
        product_remainder=metrics["product_remainder"],
        normalization_product=normalization_product,
    )
    return (
        _add_physical_scaling(
            raw,
            gamma_product=gamma**occurrence_count,
            normalization_product=normalization_product,
        ),
        metrics,
    )


def _phase_objective(
    *,
    rho: float,
    occurrence_count: int,
    spectral_values: list[Mapping[str, Any]],
    normalization_product: float,
) -> Callable[[float], float]:
    def objective(gamma: float) -> float:
        bound, _metrics = _scalar_fr_bound(
            rho=rho,
            gamma=float(gamma),
            occurrence_count=occurrence_count,
            spectral_values=spectral_values,
            normalization_product=normalization_product,
        )
        if not bound["applicable"]:
            return math.inf
        return float(bound["phase_upper_bound"])

    return objective


def _finite_interval(
    objective: Callable[[float], float], center: float, edge: float
) -> float:
    if math.isfinite(objective(edge)):
        return float(edge)
    finite = float(center)
    invalid = float(edge)
    for _ in range(80):
        midpoint = (finite + invalid) / 2.0
        if math.isfinite(objective(midpoint)):
            finite = midpoint
        else:
            invalid = midpoint
    return finite


def _optimize_fr_gamma(
    *,
    rho: float,
    occurrence_count: int,
    spectral_values: list[Mapping[str, Any]],
    normalization_product: float,
    gamma_mid: float,
) -> dict[str, Any]:
    objective = _phase_objective(
        rho=rho,
        occurrence_count=occurrence_count,
        spectral_values=spectral_values,
        normalization_product=normalization_product,
    )
    center = min(GAMMA_MAX, max(GAMMA_MIN, float(gamma_mid)))
    if not math.isfinite(objective(center)):
        return {
            "gamma": center,
            "objective": None,
            "applicable": False,
            "boundary_inconclusive": False,
            "certification": {
                "method": "analytic_involution_piecewise_bounded_search",
                "absolute_tolerance": OPTIMIZER_ATOL,
                "finite_grid_maximum_used": False,
                "reason": "midpoint_scalar_inapplicable",
            },
        }
    left = _finite_interval(objective, center, GAMMA_MIN)
    right = _finite_interval(objective, center, GAMMA_MAX)
    candidates: list[tuple[float, float]] = [(center, objective(center))]
    pieces = ((left, center), (center, right))
    optimizer_success = True
    for lower, upper in pieces:
        if upper - lower <= 2e-14:
            continue
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-14, "maxiter": 1000},
        )
        optimizer_success = optimizer_success and bool(result.success)
        if math.isfinite(float(result.fun)):
            candidates.append((float(result.x), float(result.fun)))
        for endpoint in (lower, upper):
            value = objective(endpoint)
            if math.isfinite(value):
                candidates.append((float(endpoint), float(value)))
    candidates.sort(key=lambda item: (item[1], abs(item[0] - 1.0), item[0]))
    best_gamma, best_value = candidates[0]
    tied = [
        item for item in candidates if abs(item[1] - best_value) <= OPTIMIZER_ATOL
    ]
    best_gamma, best_value = min(tied, key=lambda item: (abs(item[0] - 1.0), item[0]))
    boundary = bool(
        abs(best_gamma - GAMMA_MIN) <= OPTIMIZER_ATOL
        or abs(best_gamma - GAMMA_MAX) <= OPTIMIZER_ATOL
    )
    return {
        "gamma": float(best_gamma),
        "objective": float(best_value),
        "applicable": True,
        "boundary_inconclusive": boundary,
        "certification": {
            "method": "analytic_involution_piecewise_bounded_search",
            "absolute_tolerance": OPTIMIZER_ATOL,
            "finite_grid_maximum_used": False,
            "piece_count": 2,
            "applicable_interval": [float(left), float(right)],
            "optimizer_success": bool(optimizer_success),
        },
    }


def _decorate_method(
    method: Mapping[str, Any],
    *,
    label: str,
    information_level: str,
    oracle: bool,
    actual_phase: float,
    actual_radius: float,
    numerical_atol: float,
    boundary_inconclusive: bool = False,
) -> dict[str, Any]:
    output = deepcopy(dict(method))
    output["label"] = label
    output["information_level"] = information_level
    output["oracle"] = bool(oracle)
    output["boundary_inconclusive"] = bool(boundary_inconclusive)
    if output.get("applicable"):
        output["phase_bound_pass"] = bool(
            actual_phase <= float(output["phase_upper_bound"]) + numerical_atol
        )
        output["radius_bound_pass"] = bool(
            actual_radius + numerical_atol
            >= float(output["observed_radius_lower_bound"])
        )
    else:
        output["phase_bound_pass"] = None
        output["radius_bound_pass"] = None
        output.setdefault("inapplicable_reason", "bound_not_applicable")
    certifications: dict[str, bool] = {}
    for budget in PHASE_BUDGETS:
        certifications[f"{budget:.0e}"] = bool(
            output.get("applicable")
            and not boundary_inconclusive
            and float(output["phase_upper_bound"]) <= budget
            and float(output["observed_radius_lower_bound"]) >= RADIUS_FLOOR
        )
    output["certified_at_beta"] = certifications
    return output


def _state_methods(
    *,
    condition_record: Mapping[str, Any],
    state_record: Mapping[str, Any],
    scalar: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    count = int(condition_record["occurrence_count"])
    normalization = float(condition_record["normalization_product"])
    spectral_values = list(scalar["spectral_values"])
    rho_available = state_record["available_rho"]
    rho_reference = float(state_record["reference_radius"])
    actual_phase = float(state_record["actual_phase_error"])
    actual_radius = float(state_record["actual_observed_radius"])
    gamma_max_for_tol = max(
        float(scalar["gamma_mid_product"]),
        float(scalar["gamma_opt_norm_product"]),
        float(scalar["gamma_involution_polar_product"]),
    )
    numerical_atol = float(
        512.0
        * np.finfo(float).eps
        * max(1, count)
        * (
            1.0
            + max(
                abs(_complex_from_payload(state_record["reference_signal"])),
                abs(_complex_from_payload(state_record["corrected_signal"])),
                normalization,
                gamma_max_for_tol,
            )
        )
    )
    old_suffix = "AVAILABLE" if rho_available is not None else "REF"
    old_rho = float(rho_available) if rho_available is not None else rho_reference
    old_methods = state_record["methods"]
    methods: dict[str, Any] = {
        "OLD_NORM": _decorate_method(
            old_methods[f"OLD_NORM_{old_suffix}"],
            label="OLD-NORM",
            information_level="I0" if rho_available is not None else "I2",
            oracle=rho_available is None,
            actual_phase=actual_phase,
            actual_radius=actual_radius,
            numerical_atol=numerical_atol,
        ),
        "STRONG_NORM": _decorate_method(
            old_methods[f"STRONG_NORM_{old_suffix}"],
            label="STRONG-NORM",
            information_level="I1" if rho_available is not None else "I2",
            oracle=rho_available is None,
            actual_phase=actual_phase,
            actual_radius=actual_radius,
            numerical_atol=numerical_atol,
        ),
        "OLD_FR": _decorate_method(
            old_methods[f"PROPOSED_{old_suffix}"],
            label="OLD-FR",
            information_level="I1" if rho_available is not None else "I2",
            oracle=rho_available is None,
            actual_phase=actual_phase,
            actual_radius=actual_radius,
            numerical_atol=numerical_atol,
        ),
    }
    common_norm, common_metrics = _scalar_norm_bound(
        rho=old_rho,
        gamma=float(scalar["gamma_mid"]),
        occurrence_count=count,
        spectral_values=spectral_values,
        normalization_product=normalization,
    )
    common_fr, _ = _scalar_fr_bound(
        rho=old_rho,
        gamma=float(scalar["gamma_mid"]),
        occurrence_count=count,
        spectral_values=spectral_values,
        normalization_product=normalization,
    )
    opt_norm, opt_norm_metrics = _scalar_norm_bound(
        rho=old_rho,
        gamma=float(scalar["gamma_opt_norm"]),
        occurrence_count=count,
        spectral_values=spectral_values,
        normalization_product=normalization,
    )
    opt_fr_record = _optimize_fr_gamma(
        rho=old_rho,
        occurrence_count=count,
        spectral_values=spectral_values,
        normalization_product=normalization,
        gamma_mid=float(scalar["gamma_mid"]),
    )
    opt_fr, opt_fr_metrics = _scalar_fr_bound(
        rho=old_rho,
        gamma=float(opt_fr_record["gamma"]),
        occurrence_count=count,
        spectral_values=spectral_values,
        normalization_product=normalization,
    )
    polar, polar_metrics = _scalar_norm_bound(
        rho=old_rho,
        gamma=float(scalar["gamma_involution_polar"]),
        occurrence_count=count,
        spectral_values=spectral_values,
        normalization_product=normalization,
    )
    dense_opt = _optimize_fr_gamma(
        rho=rho_reference,
        occurrence_count=count,
        spectral_values=spectral_values,
        normalization_product=normalization,
        gamma_mid=float(scalar["gamma_mid"]),
    )
    dense, dense_metrics = _scalar_fr_bound(
        rho=rho_reference,
        gamma=float(dense_opt["gamma"]),
        occurrence_count=count,
        spectral_values=spectral_values,
        normalization_product=normalization,
    )
    practical_level = "I1" if rho_available is not None else "I2"
    practical_oracle = rho_available is None
    generated = {
        "SCALAR_NORM_COMMON": (common_norm, "SCALAR-NORM-COMMON", False),
        "SCALAR_FR_COMMON": (common_fr, "SCALAR-FR-COMMON", False),
        "OPT_SCALAR_NORM": (opt_norm, "OPT-SCALAR-NORM", bool(scalar["gamma_opt_norm_boundary_inconclusive"])),
        "OPT_SCALAR_FR": (opt_fr, "OPT-SCALAR-FR", bool(opt_fr_record["boundary_inconclusive"])),
        "INVOLUTION_POLAR": (polar, "INVOLUTION-POLAR", False),
    }
    for key, (bound, label, boundary) in generated.items():
        methods[key] = _decorate_method(
            bound,
            label=label,
            information_level=practical_level,
            oracle=practical_oracle,
            actual_phase=actual_phase,
            actual_radius=actual_radius,
            numerical_atol=numerical_atol,
            boundary_inconclusive=boundary,
        )
    methods["DENSE_ORACLE"] = _decorate_method(
        dense,
        label="DENSE-ORACLE",
        information_level="I2",
        oracle=True,
        actual_phase=actual_phase,
        actual_radius=actual_radius,
        numerical_atol=numerical_atol,
        boundary_inconclusive=bool(dense_opt["boundary_inconclusive"]),
    )
    diagnostics = {
        "rho_used_by_nonoracle_methods": old_rho,
        "rho_reference_oracle": rho_reference,
        "numerical_atol": numerical_atol,
        "common_gamma": float(scalar["gamma_mid"]),
        "opt_norm_gamma": float(scalar["gamma_opt_norm"]),
        "opt_fr_gamma": float(opt_fr_record["gamma"]),
        "polar_gamma": float(scalar["gamma_involution_polar"]),
        "dense_oracle_gamma": float(dense_opt["gamma"]),
        "common_metrics": common_metrics,
        "opt_norm_metrics": opt_norm_metrics,
        "opt_fr_metrics": opt_fr_metrics,
        "polar_metrics": polar_metrics,
        "dense_oracle_metrics": dense_metrics,
        "opt_fr_certification": opt_fr_record["certification"],
        "dense_oracle_certification": dense_opt["certification"],
        "common_fr_to_norm_ratio": (
            float(common_fr["phase_upper_bound"] / common_norm["phase_upper_bound"])
            if common_fr["applicable"]
            and common_norm["applicable"]
            and common_norm["phase_upper_bound"] > 0.0
            else None
        ),
        "optimized_fr_to_norm_ratio": (
            float(opt_fr["phase_upper_bound"] / opt_norm["phase_upper_bound"])
            if opt_fr["applicable"]
            and opt_norm["applicable"]
            and opt_norm["phase_upper_bound"] > 0.0
            else None
        ),
    }
    return methods, diagnostics


def build_fr1a_expected(source: Mapping[str, Any]) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "schema_version": FR1A_EXPECTED_SCHEMA_VERSION,
        "posthoc": True,
        "source_result_fingerprint": SOURCE_RESULT_FINGERPRINT,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_frozen_preregistration_sha256": SOURCE_PREREGISTRATION_SHA256,
        "posthoc_plan_sha256": POSTHOC_PLAN_SHA256,
        "expected_condition_count": SOURCE_CONDITION_COUNT,
        "expected_state_count": SOURCE_STATE_COUNT,
        "expected_condition_ids": [
            item["condition"]["condition_id"] for item in source["conditions"]
        ],
        "expected_state_labels": [
            "reference_eigenstate",
            "analytic_mixture_state",
            "physical_ground_state",
        ],
        "method_order": list(METHOD_ORDER),
        "phase_budgets_rad": list(PHASE_BUDGETS),
        "radius_floor": RADIUS_FLOOR,
        "gamma_interval": [GAMMA_MIN, GAMMA_MAX],
        "optimizer_absolute_tolerance": OPTIMIZER_ATOL,
        "old_decision_must_remain": SOURCE_DECISION,
        "fr1b_started": False,
        "h4_or_h12_evaluation_performed": False,
        "final_cost_evaluation_performed": False,
    }
    expected["expected_fingerprint"] = _fingerprint(expected)
    return expected


def run_fr_revision_fr1a_posthoc(
    source: Mapping[str, Any],
    *,
    source_result_path: str | Path,
    frozen_preregistration_path: str | Path,
    posthoc_plan_path: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_audit = _audit_source(
        source,
        source_result_path=source_result_path,
        frozen_preregistration_path=frozen_preregistration_path,
        posthoc_plan_path=posthoc_plan_path,
    )
    reconstructed, reconstruction = _reconstruct_source(source)
    conditions: list[dict[str, Any]] = []
    all_state_rows: list[dict[str, Any]] = []
    for condition_record in reconstructed:
        condition = condition_record["condition"]
        scalar = _local_scalar_data(
            float(condition_record["dimensionless_short_time"]),
            int(condition["cutoff"]),
            int(condition_record["occurrence_count"]),
        )
        state_rows: list[dict[str, Any]] = []
        for source_state in condition_record["state_records"]:
            methods, diagnostics = _state_methods(
                condition_record=condition_record,
                state_record=source_state,
                scalar=scalar,
            )
            row = {
                "condition_id": source_state["condition_id"],
                "scope": source_state["scope"],
                "state_label": source_state["state_label"],
                "state_fingerprint": source_state["state_fingerprint"],
                "available_rho": source_state["available_rho"],
                "reference_radius": source_state["reference_radius"],
                "actual_phase_error": source_state["actual_phase_error"],
                "actual_corrected_radius": float(
                    abs(_complex_from_payload(source_state["corrected_signal"]))
                ),
                "actual_physical_observed_radius": source_state[
                    "actual_observed_radius"
                ],
                "methods": methods,
                "diagnostics": diagnostics,
            }
            row["all_applicable_bounds_pass"] = all(
                method["phase_bound_pass"] is not False
                and method["radius_bound_pass"] is not False
                for method in methods.values()
            )
            state_rows.append(row)
            all_state_rows.append(row)
        conditions.append(
            {
                "condition": deepcopy(condition),
                "dimensionless_short_time": condition_record[
                    "dimensionless_short_time"
                ],
                "occurrence_count": condition_record["occurrence_count"],
                "normalization_product": condition_record["normalization_product"],
                "positive_scalar": scalar,
                "state_records": state_rows,
            }
        )
    main_rows = [
        row
        for row in all_state_rows
        if row["scope"] == "primary"
        and row["state_label"] == "analytic_mixture_state"
    ]
    soundness_failures = [
        {
            "condition_id": row["condition_id"],
            "state_label": row["state_label"],
            "method": key,
        }
        for row in all_state_rows
        for key, method in row["methods"].items()
        if method["phase_bound_pass"] is False or method["radius_bound_pass"] is False
    ]
    common_one_sided = []
    optimized_one_sided = []
    oracle_one_sided = []
    strict_common_improvement = []
    for row in main_rows:
        common_norm = row["methods"]["SCALAR_NORM_COMMON"]
        common_fr = row["methods"]["SCALAR_FR_COMMON"]
        opt_norm = row["methods"]["OPT_SCALAR_NORM"]
        opt_fr = row["methods"]["OPT_SCALAR_FR"]
        oracle = row["methods"]["DENSE_ORACLE"]
        if (
            common_norm["applicable"]
            and common_fr["applicable"]
            and common_fr["phase_upper_bound"]
            < common_norm["phase_upper_bound"] - COMPARISON_ATOL
        ):
            strict_common_improvement.append(row["condition_id"])
        for budget in PHASE_BUDGETS:
            key = f"{budget:.0e}"
            if common_fr["certified_at_beta"][key] and not common_norm[
                "certified_at_beta"
            ][key]:
                common_one_sided.append([row["condition_id"], key])
            if opt_fr["certified_at_beta"][key] and not opt_norm[
                "certified_at_beta"
            ][key]:
                optimized_one_sided.append([row["condition_id"], key])
            if oracle["certified_at_beta"][key] and not opt_norm[
                "certified_at_beta"
            ][key]:
                oracle_one_sided.append([row["condition_id"], key])
    scalar_explains = bool(
        not common_one_sided and not optimized_one_sided and not strict_common_improvement
    )
    residual = bool(strict_common_improvement or common_one_sided or optimized_one_sided)
    oracle_only = bool(oracle_one_sided and not common_one_sided and not optimized_one_sided)
    reconstruction_failure = not bool(source_audit["all_pass"] and reconstruction["all_pass"])
    if reconstruction_failure or soundness_failures:
        classification = "POSTHOC_BOUND_OR_RECONSTRUCTION_FAILURE"
    elif oracle_only:
        classification = "POSTHOC_ORACLE_ONLY_INCREMENT"
    elif residual:
        classification = "POSTHOC_RESIDUAL_FR_INCREMENT"
    else:
        classification = "POSTHOC_SCALAR_EXPLAINS_OLD_GAIN"
    payload: dict[str, Any] = {
        "schema_version": FR1A_SCHEMA_VERSION,
        "validation_method": FR1A_METHOD,
        "posthoc": True,
        "scope": "existing_fr1_33_condition_positive_scalar_explanation_audit_only",
        "provenance": dict(provenance or {}),
        "source_audit": source_audit,
        "reconstruction_audit": reconstruction,
        "contract": {
            "posthoc_plan": str(posthoc_plan_path),
            "phase_budgets_rad": list(PHASE_BUDGETS),
            "radius_floor": RADIUS_FLOOR,
            "gamma_interval": [GAMMA_MIN, GAMMA_MAX],
            "optimizer_absolute_tolerance": OPTIMIZER_ATOL,
            "old_decision_preserved": SOURCE_DECISION,
            "old_gates_preserved": deepcopy(source["gates"]),
            "fr1b_preregistration_modified": False,
        },
        "final_cost_evaluation_performed": False,
        "h4_or_h12_evaluation_performed": False,
        "circuit_compilation_performed": False,
        "monte_carlo_sampling_performed": False,
        "fr1b_started": False,
        "conditions": conditions,
        "summary": {
            "condition_count": len(conditions),
            "state_record_count": len(all_state_rows),
            "main_record_count": len(main_rows),
            "method_record_count": len(all_state_rows) * len(METHOD_ORDER),
            "soundness_failure_count": len(soundness_failures),
            "soundness_failures": soundness_failures,
            "strict_common_improvement_condition_ids": strict_common_improvement,
            "common_one_sided_certifications": common_one_sided,
            "optimized_one_sided_certifications": optimized_one_sided,
            "oracle_one_sided_certifications": oracle_one_sided,
            "scalar_explains_old_gain": scalar_explains,
            "residual_fr_increment": residual,
            "oracle_only_increment": oracle_only,
            "classification": classification,
            "old_decision": SOURCE_DECISION,
            "old_decision_changed": False,
            "execution_valid": bool(
                not reconstruction_failure
                and not soundness_failures
                and len(conditions) == SOURCE_CONDITION_COUNT
                and len(all_state_rows) == SOURCE_STATE_COUNT
                and len(main_rows) == 5
            ),
            "research_go_authorized": False,
            "fr1b_started": False,
        },
        "performance": {"total_seconds": float(time.perf_counter() - started)},
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_fr1a_expected(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FR1A_EXPECTED_SCHEMA_VERSION:
        raise ValueError("Unsupported FR-R1a expected schema.")
    fingerprint = payload.get("expected_fingerprint")
    without = deepcopy(dict(payload))
    without.pop("expected_fingerprint", None)
    if fingerprint != _fingerprint(without):
        raise ValueError("FR-R1a expected fingerprint mismatch.")
    if payload.get("fr1b_started") is not False:
        raise ValueError("FR-R1a expected specification cannot start FR-R1b.")


def validate_fr1a_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FR1A_SCHEMA_VERSION:
        raise ValueError("Unsupported FR-R1a result schema.")
    fingerprint = payload.get("validation_fingerprint")
    without = deepcopy(dict(payload))
    without.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without):
        raise ValueError("FR-R1a validation fingerprint mismatch.")
    if payload.get("posthoc") is not True:
        raise ValueError("FR-R1a result must be labeled posthoc.")
    for flag in (
        "final_cost_evaluation_performed",
        "h4_or_h12_evaluation_performed",
        "circuit_compilation_performed",
        "monte_carlo_sampling_performed",
        "fr1b_started",
    ):
        if payload.get(flag) is not False:
            raise ValueError(f"FR-R1a scope flag must remain false: {flag}")
    summary = payload.get("summary", {})
    if summary.get("old_decision") != SOURCE_DECISION:
        raise ValueError("FR-R1a changed the old FR-1 decision.")
    if summary.get("old_decision_changed") is not False:
        raise ValueError("FR-R1a must not reclassify the old FR-1 result.")
    if summary.get("condition_count") != SOURCE_CONDITION_COUNT:
        raise ValueError("FR-R1a condition count mismatch.")
    if summary.get("state_record_count") != SOURCE_STATE_COUNT:
        raise ValueError("FR-R1a state count mismatch.")


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Refusing to overwrite different FR-R1a artifact: {target}")
    target.write_text(serialized, encoding="utf-8")


def write_fr1a_expected(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_fr1a_expected(payload)
    _write_json(payload, path)


def write_fr1a_payload(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_fr1a_payload(payload)
    _write_json(payload, path)

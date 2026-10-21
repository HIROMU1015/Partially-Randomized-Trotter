"""Preregistered FR-R1b nonuniform-spectrum 4x4 validation.

The numerical contract is frozen in
``docs/research/fr_revision_nonuniform_preregistration.md``.  This module
implements only its 20 matrix conditions, 61 state rows, two semantic
controls, R0--R7 gates, and mandatory post-run stop.
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import minimize_scalar

from .finite_rte_phase_amplitude import (
    _array_fingerprint,
    _compose_local_error,
    _fingerprint,
    _norm_bound,
    _old_local_error,
    _operator_norm,
    _phase_distance,
    _proposal_bound,
    _strong_local_error,
)
from .fr_revision_fr1a_posthoc import (
    GAMMA_MAX,
    GAMMA_MIN,
    OPTIMIZER_ATOL,
    PHASE_BUDGETS,
    RADIUS_FLOOR,
    _scalar_fr_bound,
    _scalar_norm_bound,
)
from .rte import (
    InvolutoryTailTerm,
    enumerate_rte_events,
    event_unitary,
    exact_enumerated_event_mean_operator,
    finite_rte_distribution,
    finite_taylor_operator,
    normalize_involutory_tail,
)


FR_R1B_SCHEMA_VERSION = "fr_revision_nonuniform_r1b_v1"
FR_R1B_EXPECTED_SCHEMA_VERSION = "fr_revision_nonuniform_expected_v1"
FR_R1B_METHOD = "nonuniform_positive_scalar_phase_radius_validation_v1"
PREREGISTRATION_SHA256 = (
    "1bc2a72fa8dec98e2bdbe3504e65366d8daa93f7a7797837622ba7545607515e"
)
EXPECTED_MATRIX_CONDITIONS = 20
EXPECTED_STATE_ROWS = 61
EXPECTED_SEMANTIC_CONTROLS = 2
SEMANTIC_ATOL = 1e-12
SUPPLIED_RHO = 0.8
TOTAL_TIME = 0.8

STANDARD_STATE_LABELS = (
    "fixed_supplied_superposition",
    "fixed_q8_reference_eigenstate",
    "fixed_total_ground_state",
)
SPECIAL_STATE_LABEL = "signal_near_zero_stress_state"
METHOD_ORDER = (
    "OLD-NORM-I0",
    "STRONG-NORM-I0",
    "STRONG-NORM-I1",
    "OLD-FR-I1",
    "SCALAR-NORM-COMMON-I1",
    "SCALAR-FR-COMMON-I1",
    "OPT-SCALAR-NORM-I1",
    "OPT-SCALAR-FR-I1",
    "INVOLUTION-POLAR-I1",
    "DENSE-ORACLE-I2",
)

_I2 = np.eye(2, dtype=np.complex128)
_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_I4 = np.eye(4, dtype=np.complex128)
_ZI = np.kron(_Z, _I2)
_IZ = np.kron(_I2, _Z)
_XI = np.kron(_X, _I2)
_ZX = np.kron(_Z, _X)
_Q4 = np.diag(np.arange(4, dtype=float)).astype(np.complex128)


def _complex_payload(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def _canonicalize_vector(vector: np.ndarray) -> np.ndarray:
    output = np.asarray(vector, dtype=np.complex128).copy()
    output /= np.linalg.norm(output)
    absolute = np.abs(output)
    maximum = float(np.max(absolute))
    tied = np.flatnonzero(np.abs(absolute - maximum) <= 1e-14)
    pivot = int(tied[0])
    output *= np.exp(-1j * np.angle(output[pivot]))
    if output[pivot].real < 0.0:
        output *= -1.0
    return output


def _resolve_cluster(vectors: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(np.asarray(vectors, dtype=np.complex128))
    projected = basis.conj().T @ _Q4 @ basis
    values, rotations = np.linalg.eigh((projected + projected.conj().T) / 2.0)
    order = np.argsort(values, kind="stable")
    values = values[order]
    resolved = basis @ rotations[:, order]
    columns = [_canonicalize_vector(resolved[:, index]) for index in range(resolved.shape[1])]
    for cluster in _clusters(values):
        if len(cluster) <= 1:
            continue
        tied = [columns[index] for index in cluster]
        tied.sort(
            key=lambda vector: tuple(
                np.round(np.concatenate([vector.real, vector.imag]), 14).tolist()
            )
        )
        for index, vector in zip(cluster, tied, strict=True):
            columns[index] = vector
    return np.column_stack(columns)


def _clusters(values: np.ndarray, tolerance: float = 1e-12) -> list[list[int]]:
    output: list[list[int]] = []
    for index, value in enumerate(values):
        if not output or abs(float(value) - float(values[output[-1][0]])) > tolerance:
            output.append([index])
        else:
            output[-1].append(index)
    return output


def _canonical_hermitian_eigensystem(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.complex128))
    order = np.argsort(values, kind="stable")
    values = values[order]
    vectors = vectors[:, order]
    resolved: list[np.ndarray] = []
    for cluster in _clusters(values):
        block = vectors[:, cluster]
        if len(cluster) > 1:
            block = _resolve_cluster(block)
        else:
            block = _canonicalize_vector(block[:, 0])[:, None]
        resolved.extend(block[:, index] for index in range(block.shape[1]))
    output = np.column_stack(resolved)
    if not np.allclose(output.conj().T @ output, _I4, atol=2e-12, rtol=0.0):
        raise RuntimeError("Canonical Hermitian eigensystem is not orthonormal.")
    return values, output


def _principal_phase(value: complex) -> float:
    phase = float(np.angle(value))
    return math.pi if phase <= -math.pi + 1e-15 else phase


def _canonical_unitary_eigensystem(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=np.complex128))
    phases = np.asarray([_principal_phase(value) for value in values], dtype=float)
    order = np.argsort(phases, kind="stable")
    values = values[order]
    phases = phases[order]
    vectors = vectors[:, order]
    resolved_values: list[complex] = []
    resolved_vectors: list[np.ndarray] = []
    for cluster in _clusters(phases):
        block = vectors[:, cluster]
        if len(cluster) > 1:
            block = _resolve_cluster(block)
        else:
            block = _canonicalize_vector(block[:, 0])[:, None]
        resolved_vectors.extend(block[:, index] for index in range(block.shape[1]))
        resolved_values.extend(values[cluster[0]] for _ in cluster)
    output = np.column_stack(resolved_vectors)
    if not np.allclose(output.conj().T @ output, _I4, atol=2e-12, rtol=0.0):
        raise RuntimeError("Canonical unitary eigensystem is not orthonormal.")
    return np.asarray(resolved_values), output


def _hermitian_exponential(matrix: np.ndarray, time_value: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.complex128))
    return (vectors * np.exp(-1j * float(time_value) * values)) @ vectors.conj().T


def _tail(nu: float) -> np.ndarray:
    return ((1.0 + nu) / 2.0) * _ZI + ((1.0 - nu) / 2.0) * _IZ


def _deterministic(label: str) -> np.ndarray:
    if label == "c":
        return 0.7 * _ZI + 0.3 * _IZ
    if label == "nc":
        return 0.7 * _XI + 0.3 * _ZX
    raise ValueError(f"Unknown deterministic block: {label}")


def _nu_tag(nu: float) -> str:
    return {0.0: "0", 0.5: "0p5", 1.0: "1"}[float(nu)]


def _condition_id(
    *, nu: float, deterministic: str, q: int, sigma: int, cutoff: int, r: int, scope: str
) -> str:
    return (
        f"{scope}_nu{_nu_tag(nu)}_D{deterministic}_q{q}_"
        f"sigma{sigma:+d}_K{cutoff}_r{r}"
    )


def fixed_conditions() -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for nu in (0.0, 0.5, 1.0):
        for q in (2, 4, 8):
            for deterministic in ("c", "nc"):
                conditions.append(
                    {
                        "condition_id": _condition_id(
                            nu=nu,
                            deterministic=deterministic,
                            q=q,
                            sigma=1,
                            cutoff=2,
                            r=1,
                            scope="primary",
                        ),
                        "scope": "primary",
                        "nu": nu,
                        "deterministic_block": deterministic,
                        "q": q,
                        "sigma": 1,
                        "cutoff": 2,
                        "r": 1,
                        "total_time": TOTAL_TIME,
                    }
                )
    for scope, sigma, cutoff in (("negative_time_control", -1, 2), ("k4_control", 1, 4)):
        conditions.append(
            {
                "condition_id": _condition_id(
                    nu=0.5,
                    deterministic="nc",
                    q=4,
                    sigma=sigma,
                    cutoff=cutoff,
                    r=1,
                    scope=scope,
                ),
                "scope": scope,
                "nu": 0.5,
                "deterministic_block": "nc",
                "q": 4,
                "sigma": sigma,
                "cutoff": cutoff,
                "r": 1,
                "total_time": TOTAL_TIME,
            }
        )
    identifiers = [condition["condition_id"] for condition in conditions]
    if len(conditions) != EXPECTED_MATRIX_CONDITIONS or len(set(identifiers)) != len(identifiers):
        raise RuntimeError("FR-R1b fixed matrix-condition contract mismatch.")
    return conditions


def expected_state_ids(conditions: list[Mapping[str, Any]]) -> list[str]:
    identifiers: list[str] = []
    for condition in conditions:
        identifiers.extend(
            f"{condition['condition_id']}__{label}" for label in STANDARD_STATE_LABELS
        )
        if (
            condition["scope"] == "primary"
            and condition["nu"] == 0.0
            and condition["deterministic_block"] == "nc"
            and condition["q"] == 4
        ):
            identifiers.append(f"{condition['condition_id']}__{SPECIAL_STATE_LABEL}")
    if len(identifiers) != EXPECTED_STATE_ROWS or len(set(identifiers)) != len(identifiers):
        raise RuntimeError("FR-R1b fixed state-row contract mismatch.")
    return identifiers


def build_fr_r1b_expected(
    *, provenance: Mapping[str, Any], source_sha256: Mapping[str, str]
) -> dict[str, Any]:
    conditions = fixed_conditions()
    state_ids = expected_state_ids(conditions)
    specification: dict[str, Any] = {
        "schema_version": FR_R1B_EXPECTED_SCHEMA_VERSION,
        "preregistered_not_result_adaptive": True,
        "provenance": dict(provenance),
        "source_sha256": dict(source_sha256),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "condition_ids": [condition["condition_id"] for condition in conditions],
        "state_ids": state_ids,
        "semantic_control_ids": ["semantic_tau-0p2_K2", "semantic_tau+0p2_K2"],
        "method_order": list(METHOD_ORDER),
        "phase_budgets_rad": list(PHASE_BUDGETS),
        "radius_floor": RADIUS_FLOOR,
        "supplied_rho": SUPPLIED_RHO,
        "gamma_interval": [GAMMA_MIN, GAMMA_MAX],
        "optimizer_absolute_tolerance": OPTIMIZER_ATOL,
        "expected_matrix_conditions": EXPECTED_MATRIX_CONDITIONS,
        "expected_state_rows": EXPECTED_STATE_ROWS,
        "expected_semantic_controls": EXPECTED_SEMANTIC_CONTROLS,
        "expected_gates": [f"R{index}" for index in range(8)],
        "mandatory_stop_after_result": True,
        "h4_or_h12_evaluation_performed": False,
        "final_cost_evaluation_performed": False,
    }
    specification["configuration_fingerprint"] = _fingerprint(
        {
            "conditions": conditions,
            "state_ids": state_ids,
            "semantic_control_ids": specification["semantic_control_ids"],
            "method_order": specification["method_order"],
            "phase_budgets_rad": specification["phase_budgets_rad"],
            "radius_floor": RADIUS_FLOOR,
            "supplied_rho": SUPPLIED_RHO,
            "gamma_interval": specification["gamma_interval"],
            "gates": specification["expected_gates"],
        }
    )
    specification["expected_fingerprint"] = _fingerprint(specification)
    return specification


def validate_fr_r1b_expected(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FR_R1B_EXPECTED_SCHEMA_VERSION:
        raise ValueError("Unsupported FR-R1b expected schema.")
    fingerprint = payload.get("expected_fingerprint")
    without = deepcopy(dict(payload))
    without.pop("expected_fingerprint", None)
    if fingerprint != _fingerprint(without):
        raise ValueError("FR-R1b expected fingerprint mismatch.")
    if payload.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("FR-R1b preregistration hash mismatch.")
    if len(payload.get("condition_ids", [])) != EXPECTED_MATRIX_CONDITIONS:
        raise ValueError("FR-R1b expected condition count mismatch.")
    if len(payload.get("state_ids", [])) != EXPECTED_STATE_ROWS:
        raise ValueError("FR-R1b expected state count mismatch.")
    if len(payload.get("semantic_control_ids", [])) != EXPECTED_SEMANTIC_CONTROLS:
        raise ValueError("FR-R1b semantic-control count mismatch.")
    if payload.get("mandatory_stop_after_result") is not True:
        raise ValueError("FR-R1b expected specification must require a stop.")


def _fixed_states(nu: float, deterministic: str, sigma: int) -> dict[str, np.ndarray]:
    h = _tail(nu)
    h_d = _deterministic(deterministic)
    _values, vectors = _canonical_hermitian_eigensystem(h_d + sigma * h)
    supplied = math.sqrt(0.9) * vectors[:, 0]
    supplied += math.sqrt(0.1 / 3.0) * np.sum(vectors[:, 1:], axis=1)
    supplied /= np.linalg.norm(supplied)
    ground = vectors[:, 0].copy()
    delta = TOTAL_TIME / 8.0
    half = _hermitian_exponential(h_d, delta / 2.0)
    exact_tail = _hermitian_exponential(h, sigma * delta)
    u_q8 = np.linalg.matrix_power(half @ exact_tail @ half, 8)
    _unitary_values, unitary_vectors = _canonical_unitary_eigensystem(u_q8)
    reference = unitary_vectors[:, 0].copy()
    return {
        "fixed_supplied_superposition": _canonicalize_vector(supplied),
        "fixed_q8_reference_eigenstate": _canonicalize_vector(reference),
        "fixed_total_ground_state": _canonicalize_vector(ground),
    }


def _near_zero_state(unitary: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    eigenvalues, eigenvectors = _canonical_unitary_eigensystem(unitary)
    phases = [_principal_phase(value) for value in eigenvalues]
    candidates: list[tuple[float, int, int]] = []
    for left in range(4):
        for right in range(left + 1, 4):
            distance = abs(float(np.angle(np.exp(1j * (phases[left] - phases[right])))))
            candidates.append((distance, left, right))
    maximum = max(item[0] for item in candidates)
    _distance, left, right = min(
        (item for item in candidates if abs(item[0] - maximum) <= 1e-14),
        key=lambda item: (item[1], item[2]),
    )
    state = (eigenvectors[:, left] + eigenvectors[:, right]) / math.sqrt(2.0)
    return _canonicalize_vector(state), {
        "selected_index_pair": [left, right],
        "circular_phase_distance": float(maximum),
    }


def _spectral_values(nu: float, tau: float, cutoff: int) -> list[dict[str, float]]:
    values = (1.0, nu, -nu, -1.0)
    output = []
    for eigenvalue in values:
        x = tau * eigenvalue
        polynomial = sum(
            (-1j * x) ** degree / math.factorial(degree)
            for degree in range(cutoff + 2)
        )
        value = complex(np.exp(1j * x) * polynomial)
        output.append({"real": float(value.real), "imag": float(value.imag)})
    return output


def _relative_metrics(
    spectral_values: list[Mapping[str, Any]], gamma: float, count: int
) -> dict[str, float]:
    values = np.asarray(
        [complex(float(value["real"]), float(value["imag"])) for value in spectral_values]
    )
    relative = values / float(gamma) - 1.0
    local_a = float(np.max(np.abs(relative.real)))
    local_b = float(np.max(np.abs(relative.imag)))
    local_e = float(np.max(np.abs(relative)))
    composed = _compose_local_error(local_e, count)
    return {
        "a": local_a,
        "b": local_b,
        "e": local_e,
        "composed_error": float(composed),
        "product_remainder": max(0.0, float(composed - count * local_e)),
    }


def _norm_objective(spectral_values: list[Mapping[str, Any]], gamma: float) -> float:
    return _relative_metrics(spectral_values, gamma, 1)["e"]


def _optimize_norm_gamma(spectral_values: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = [complex(float(value["real"]), float(value["imag"])) for value in spectral_values]
    candidates = {GAMMA_MIN, GAMMA_MAX, 1.0}
    for value in values:
        if value.real > 0.0:
            candidates.add(float(abs(value) ** 2 / value.real))
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            denominator = 2.0 * (left.real - right.real)
            if abs(denominator) > 1e-18:
                candidates.add(float((abs(left) ** 2 - abs(right) ** 2) / denominator))
    allowed = sorted(value for value in candidates if GAMMA_MIN <= value <= GAMMA_MAX)
    scored = [(gamma, _norm_objective(spectral_values, gamma)) for gamma in allowed]
    best_value = min(value for _gamma, value in scored)
    tied = [item for item in scored if abs(item[1] - best_value) <= OPTIMIZER_ATOL]
    gamma, value = min(tied, key=lambda item: (abs(item[0] - 1.0), item[0]))
    boundary = bool(
        abs(gamma - GAMMA_MIN) <= OPTIMIZER_ATOL
        or abs(gamma - GAMMA_MAX) <= OPTIMIZER_ATOL
    )
    return {
        "gamma": float(gamma),
        "objective": float(value),
        "boundary_inconclusive": boundary,
        "certification": {
            "method": "analytic_finite_spectral_envelope_candidate_enumeration",
            "absolute_tolerance": OPTIMIZER_ATOL,
            "candidate_count": len(allowed),
            "finite_grid_maximum_used": False,
        },
    }


def _fr_objective(
    *,
    rho: float,
    count: int,
    spectral_values: list[Mapping[str, Any]],
    normalization: float,
) -> Callable[[float], float]:
    def objective(gamma: float) -> float:
        bound, _metrics = _scalar_fr_bound(
            rho=rho,
            gamma=float(gamma),
            occurrence_count=count,
            spectral_values=spectral_values,
            normalization_product=normalization,
        )
        return float(bound["phase_upper_bound"]) if bound["applicable"] else math.inf

    return objective


def _fr_breakpoints(spectral_values: list[Mapping[str, Any]], gamma_mid: float) -> list[float]:
    values = [complex(float(value["real"]), float(value["imag"])) for value in spectral_values]
    points = {GAMMA_MIN, GAMMA_MAX, float(gamma_mid), 1.0}
    for value in values:
        points.add(float(value.real))
        if value.real > 0.0:
            points.add(float(abs(value) ** 2 / value.real))
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            points.add(float((left.real + right.real) / 2.0))
            denominator = 2.0 * (left.real - right.real)
            if abs(denominator) > 1e-18:
                points.add(float((abs(left) ** 2 - abs(right) ** 2) / denominator))
    return sorted(value for value in points if GAMMA_MIN <= value <= GAMMA_MAX)


def _optimize_fr_gamma(
    *,
    rho: float,
    count: int,
    spectral_values: list[Mapping[str, Any]],
    normalization: float,
    gamma_mid: float,
) -> dict[str, Any]:
    objective = _fr_objective(
        rho=rho,
        count=count,
        spectral_values=spectral_values,
        normalization=normalization,
    )
    points = _fr_breakpoints(spectral_values, gamma_mid)
    candidates: list[tuple[float, float]] = []
    optimizer_success = True
    for point in points:
        value = objective(point)
        if math.isfinite(value):
            candidates.append((point, value))
    for lower, upper in zip(points[:-1], points[1:], strict=True):
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
    if not candidates:
        return {
            "gamma": float(gamma_mid),
            "objective": None,
            "applicable": False,
            "boundary_inconclusive": False,
            "certification": {
                "method": "piecewise_analytic_spectral_envelope_search",
                "absolute_tolerance": OPTIMIZER_ATOL,
                "finite_grid_maximum_used": False,
                "optimizer_success": False,
            },
        }
    best_value = min(value for _gamma, value in candidates)
    tied = [item for item in candidates if abs(item[1] - best_value) <= OPTIMIZER_ATOL]
    gamma, value = min(tied, key=lambda item: (abs(item[0] - 1.0), item[0]))
    boundary = bool(
        abs(gamma - GAMMA_MIN) <= OPTIMIZER_ATOL
        or abs(gamma - GAMMA_MAX) <= OPTIMIZER_ATOL
    )
    return {
        "gamma": float(gamma),
        "objective": float(value),
        "applicable": True,
        "boundary_inconclusive": boundary,
        "certification": {
            "method": "piecewise_analytic_spectral_envelope_search",
            "absolute_tolerance": OPTIMIZER_ATOL,
            "finite_grid_maximum_used": False,
            "analytic_breakpoint_count": len(points),
            "piece_count": max(0, len(points) - 1),
            "optimizer_success": bool(optimizer_success),
        },
    }


def _unavailable_method(label: str, information_level: str, reason: str) -> dict[str, Any]:
    return {
        "label": label,
        "information_level": information_level,
        "oracle": information_level == "I2",
        "applicable": False,
        "inapplicable_reason": reason,
        "phase_upper_bound": None,
        "observed_radius_lower_bound": None,
        "phase_bound_pass": None,
        "radius_bound_pass": None,
        "boundary_inconclusive": False,
        "certified_at_beta": {f"{budget:.0e}": False for budget in PHASE_BUDGETS},
    }


def _decorate_method(
    method: Mapping[str, Any],
    *,
    label: str,
    information_level: str,
    oracle: bool,
    actual_phase: float | None,
    actual_radius: float,
    numerical_atol: float,
    boundary_inconclusive: bool = False,
) -> dict[str, Any]:
    output = deepcopy(dict(method))
    output.update(
        {
            "label": label,
            "information_level": information_level,
            "oracle": bool(oracle),
            "boundary_inconclusive": bool(boundary_inconclusive),
        }
    )
    if output.get("applicable") and actual_phase is not None:
        output["phase_bound_pass"] = bool(
            actual_phase <= float(output["phase_upper_bound"]) + numerical_atol
        )
        output["radius_bound_pass"] = bool(
            actual_radius + numerical_atol >= float(output["observed_radius_lower_bound"])
        )
        output["inapplicable_reason"] = None
    else:
        output["applicable"] = False
        output["phase_bound_pass"] = None
        output["radius_bound_pass"] = None
        output["inapplicable_reason"] = (
            "undefined_reference_phase" if actual_phase is None else "bound_not_applicable"
        )
    output["certified_at_beta"] = {
        f"{budget:.0e}": bool(
            output["applicable"]
            and not boundary_inconclusive
            and float(output["phase_upper_bound"]) <= budget
            and float(output["observed_radius_lower_bound"]) >= RADIUS_FLOOR
        )
        for budget in PHASE_BUDGETS
    }
    return output


def _state_methods(
    *,
    condition: Mapping[str, Any],
    spectral_values: list[Mapping[str, Any]],
    scalar: Mapping[str, Any],
    state_label: str,
    available_rho: float | None,
    reference_radius: float,
    actual_phase: float | None,
    actual_radius: float,
    numerical_atol: float,
    normalization: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    count = int(condition["q"]) * int(condition["r"])
    methods: dict[str, Any] = {}
    if available_rho is None:
        levels = {
            "OLD-NORM-I0": "I0",
            "STRONG-NORM-I0": "I0",
            "STRONG-NORM-I1": "I1",
            "OLD-FR-I1": "I1",
            "SCALAR-NORM-COMMON-I1": "I1",
            "SCALAR-FR-COMMON-I1": "I1",
            "OPT-SCALAR-NORM-I1": "I1",
            "OPT-SCALAR-FR-I1": "I1",
            "INVOLUTION-POLAR-I1": "I1",
        }
        for label in METHOD_ORDER[:-1]:
            methods[label] = _unavailable_method(
                label, levels[label], "no_available_rho_certificate"
            )
    else:
        rho = float(available_rho)
        eta = abs(float(condition["sigma"]) * (TOTAL_TIME / int(condition["q"])) / int(condition["r"]))
        old_local = _old_local_error(eta, int(condition["cutoff"]))
        strong_i0_local = _strong_local_error(eta, int(condition["cutoff"]))
        unscaled = _relative_metrics(spectral_values, 1.0, count)
        old_norm = _norm_bound(
            rho_input=rho,
            composed_error=_compose_local_error(old_local, count),
            normalization_product=normalization,
        )
        strong_i0 = _norm_bound(
            rho_input=rho,
            composed_error=_compose_local_error(strong_i0_local, count),
            normalization_product=normalization,
        )
        strong_i1 = _norm_bound(
            rho_input=rho,
            composed_error=unscaled["composed_error"],
            normalization_product=normalization,
        )
        old_fr = _proposal_bound(
            rho_input=rho,
            a_sum=count * unscaled["a"],
            b_sum=count * unscaled["b"],
            e_sum=count * unscaled["e"],
            product_remainder=unscaled["product_remainder"],
            normalization_product=normalization,
        )
        common_norm, common_metrics = _scalar_norm_bound(
            rho=rho,
            gamma=float(scalar["gamma_mid"]),
            occurrence_count=count,
            spectral_values=spectral_values,
            normalization_product=normalization,
        )
        common_fr, _ = _scalar_fr_bound(
            rho=rho,
            gamma=float(scalar["gamma_mid"]),
            occurrence_count=count,
            spectral_values=spectral_values,
            normalization_product=normalization,
        )
        opt_norm, opt_norm_metrics = _scalar_norm_bound(
            rho=rho,
            gamma=float(scalar["gamma_opt_norm"]),
            occurrence_count=count,
            spectral_values=spectral_values,
            normalization_product=normalization,
        )
        opt_fr_record = _optimize_fr_gamma(
            rho=rho,
            count=count,
            spectral_values=spectral_values,
            normalization=normalization,
            gamma_mid=float(scalar["gamma_mid"]),
        )
        opt_fr, opt_fr_metrics = _scalar_fr_bound(
            rho=rho,
            gamma=float(opt_fr_record["gamma"]),
            occurrence_count=count,
            spectral_values=spectral_values,
            normalization_product=normalization,
        )
        practical = (
            ("OLD-NORM-I0", old_norm, "I0", False),
            ("STRONG-NORM-I0", strong_i0, "I0", False),
            ("STRONG-NORM-I1", strong_i1, "I1", False),
            ("OLD-FR-I1", old_fr, "I1", False),
            ("SCALAR-NORM-COMMON-I1", common_norm, "I1", False),
            ("SCALAR-FR-COMMON-I1", common_fr, "I1", False),
            (
                "OPT-SCALAR-NORM-I1",
                opt_norm,
                "I1",
                bool(scalar["gamma_opt_norm_boundary_inconclusive"]),
            ),
            (
                "OPT-SCALAR-FR-I1",
                opt_fr,
                "I1",
                bool(opt_fr_record["boundary_inconclusive"]),
            ),
        )
        for label, bound, level, boundary in practical:
            methods[label] = _decorate_method(
                bound,
                label=label,
                information_level=level,
                oracle=False,
                actual_phase=actual_phase,
                actual_radius=actual_radius,
                numerical_atol=numerical_atol,
                boundary_inconclusive=boundary,
            )
        if float(condition["nu"]) == 1.0:
            polar, _polar_metrics = _scalar_norm_bound(
                rho=rho,
                gamma=float(scalar["gamma_involution_polar"]),
                occurrence_count=count,
                spectral_values=spectral_values,
                normalization_product=normalization,
            )
            methods["INVOLUTION-POLAR-I1"] = _decorate_method(
                polar,
                label="INVOLUTION-POLAR-I1",
                information_level="I1",
                oracle=False,
                actual_phase=actual_phase,
                actual_radius=actual_radius,
                numerical_atol=numerical_atol,
            )
        else:
            methods["INVOLUTION-POLAR-I1"] = _unavailable_method(
                "INVOLUTION-POLAR-I1", "I1", "tail_is_not_involution"
            )
    dense_diagnostic: dict[str, Any]
    if actual_phase is None:
        methods["DENSE-ORACLE-I2"] = _unavailable_method(
            "DENSE-ORACLE-I2", "I2", "undefined_reference_phase"
        )
        dense_diagnostic = {"applicable": False}
    else:
        dense_opt = _optimize_fr_gamma(
            rho=reference_radius,
            count=count,
            spectral_values=spectral_values,
            normalization=normalization,
            gamma_mid=float(scalar["gamma_mid"]),
        )
        dense, dense_metrics = _scalar_fr_bound(
            rho=reference_radius,
            gamma=float(dense_opt["gamma"]),
            occurrence_count=count,
            spectral_values=spectral_values,
            normalization_product=normalization,
        )
        methods["DENSE-ORACLE-I2"] = _decorate_method(
            dense,
            label="DENSE-ORACLE-I2",
            information_level="I2",
            oracle=True,
            actual_phase=actual_phase,
            actual_radius=actual_radius,
            numerical_atol=numerical_atol,
            boundary_inconclusive=bool(dense_opt["boundary_inconclusive"]),
        )
        dense_diagnostic = {
            "applicable": True,
            "gamma": dense_opt["gamma"],
            "metrics": dense_metrics,
            "certification": dense_opt["certification"],
        }
    diagnostics = {
        "state_label": state_label,
        "available_rho": available_rho,
        "reference_radius_oracle": reference_radius,
        "dense_oracle": dense_diagnostic,
    }
    if available_rho is not None:
        diagnostics.update(
            {
                "common_metrics": common_metrics,
                "opt_norm_metrics": opt_norm_metrics,
                "opt_fr_metrics": opt_fr_metrics,
                "opt_fr_gamma": opt_fr_record["gamma"],
                "opt_fr_certification": opt_fr_record["certification"],
            }
        )
    if list(methods) != list(METHOD_ORDER):
        raise RuntimeError("FR-R1b method order mismatch.")
    return methods, diagnostics


def _scalar_record(
    spectral_values: list[Mapping[str, Any]], count: int
) -> dict[str, Any]:
    real_relative = [float(value["real"]) - 1.0 for value in spectral_values]
    f_minus = min(real_relative)
    f_plus = max(real_relative)
    c_mid = (f_minus + f_plus) / 2.0
    gamma_mid = 1.0 + c_mid
    if gamma_mid <= 0.0:
        raise RuntimeError("FR-R1b midpoint scalar is not positive.")
    opt = _optimize_norm_gamma(spectral_values)
    magnitudes = [
        abs(complex(float(value["real"]), float(value["imag"])))
        for value in spectral_values
    ]
    gamma_polar = float(sum(magnitudes) / len(magnitudes))
    return {
        "spectral_values": deepcopy(spectral_values),
        "f_minus": float(f_minus),
        "f_plus": float(f_plus),
        "c_mid": float(c_mid),
        "gamma_mid": float(gamma_mid),
        "gamma_mid_product": float(gamma_mid**count),
        "centered_hermitian_spectral_width": float(f_plus - f_minus),
        "gamma_opt_norm": opt["gamma"],
        "gamma_opt_norm_product": float(opt["gamma"] ** count),
        "gamma_opt_norm_boundary_inconclusive": opt["boundary_inconclusive"],
        "gamma_opt_norm_certification": opt["certification"],
        "gamma_involution_polar": gamma_polar,
        "gamma_involution_polar_product": float(gamma_polar**count),
        "involution_polar_magnitude_spread": float(max(magnitudes) - min(magnitudes)),
    }


def _condition_record(
    condition: Mapping[str, Any], state_cache: Mapping[tuple[float, str, int], Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    nu = float(condition["nu"])
    deterministic = str(condition["deterministic_block"])
    q = int(condition["q"])
    sigma = int(condition["sigma"])
    cutoff = int(condition["cutoff"])
    r = int(condition["r"])
    delta = TOTAL_TIME / q
    tau = sigma * delta / r
    count = q * r
    h = _tail(nu)
    h_d = _deterministic(deterministic)
    half = _hermitian_exponential(h_d, delta / 2.0)
    exact_tail = _hermitian_exponential(h, sigma * delta)
    short_polynomial = finite_taylor_operator(h, tau, cutoff)
    corrected_tail = np.linalg.matrix_power(short_polynomial, r)
    exact_step = half @ exact_tail @ half
    corrected_step = half @ corrected_tail @ half
    exact_operator = np.linalg.matrix_power(exact_step, q)
    corrected_operator = np.linalg.matrix_power(corrected_step, q)
    distribution = finite_rte_distribution(tau, cutoff)
    normalization = float(
        math.exp(count * math.log(distribution.exact_finite_distribution))
    )
    mean_operator = corrected_operator / normalization
    spectral_values = _spectral_values(nu, tau, cutoff)
    scalar = _scalar_record(spectral_values, count)
    short_exact = _hermitian_exponential(h, tau)
    matrix_relative = short_exact.conj().T @ short_polynomial
    spectral_e = max(
        abs(complex(value["real"], value["imag"]) - 1.0)
        for value in spectral_values
    )
    state_map = dict(state_cache[(nu, deterministic, sigma)])
    near_zero_metadata = None
    if (
        condition["scope"] == "primary"
        and nu == 0.0
        and deterministic == "nc"
        and q == 4
    ):
        near_zero, near_zero_metadata = _near_zero_state(exact_operator)
        state_map[SPECIAL_STATE_LABEL] = near_zero
    state_records: list[dict[str, Any]] = []
    for state_label, state in state_map.items():
        reference_signal = complex(np.vdot(state, exact_operator @ state))
        corrected_signal = complex(np.vdot(state, corrected_operator @ state))
        observed_signal = corrected_signal / normalization
        reference_radius = float(abs(reference_signal))
        if 1.0 < reference_radius <= 1.0 + 2e-12:
            reference_radius = 1.0
        actual_phase = _phase_distance(corrected_signal, reference_signal)
        available_rho = SUPPLIED_RHO if state_label == STANDARD_STATE_LABELS[0] else None
        gamma_for_tolerance = max(
            1.0,
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
                    abs(reference_signal),
                    abs(corrected_signal),
                    normalization,
                    gamma_for_tolerance,
                )
            )
        )
        methods, diagnostics = _state_methods(
            condition=condition,
            spectral_values=spectral_values,
            scalar=scalar,
            state_label=state_label,
            available_rho=available_rho,
            reference_radius=reference_radius,
            actual_phase=actual_phase,
            actual_radius=float(abs(observed_signal)),
            numerical_atol=numerical_atol,
            normalization=normalization,
        )
        state_id = f"{condition['condition_id']}__{state_label}"
        record = {
            "state_id": state_id,
            "condition_id": condition["condition_id"],
            "scope": condition["scope"],
            "state_label": state_label,
            "state_fingerprint": _array_fingerprint(state),
            "state_construction_uses_dense_toy_information": True,
            "available_rho": available_rho,
            "available_rho_valid": bool(
                available_rho is None or available_rho <= reference_radius + numerical_atol
            ),
            "reference_signal": _complex_payload(reference_signal),
            "corrected_signal": _complex_payload(corrected_signal),
            "observed_signal": _complex_payload(observed_signal),
            "reference_radius": reference_radius,
            "actual_phase_error": actual_phase,
            "actual_corrected_radius": float(abs(corrected_signal)),
            "actual_physical_observed_radius": float(abs(observed_signal)),
            "numerical_atol": numerical_atol,
            "methods": methods,
            "diagnostics": diagnostics,
        }
        if state_label == SPECIAL_STATE_LABEL:
            record["near_zero_construction"] = near_zero_metadata
        record["all_applicable_bounds_pass"] = all(
            method["phase_bound_pass"] is not False
            and method["radius_bound_pass"] is not False
            for method in methods.values()
        )
        state_records.append(record)
    return {
        "condition": dict(condition),
        "delta": float(delta),
        "signed_short_time": float(tau),
        "occurrence_count": count,
        "normalization_per_occurrence": float(distribution.exact_finite_distribution),
        "normalization_product": normalization,
        "operator_fingerprints": {
            "exact_operator": _array_fingerprint(exact_operator),
            "corrected_operator": _array_fingerprint(corrected_operator),
            "mean_operator": _array_fingerprint(mean_operator),
        },
        "exact_unitarity_defect": _operator_norm(exact_operator.conj().T @ exact_operator - _I4),
        "spectral_vs_matrix_local_error_abs_difference": float(
            abs(spectral_e - _operator_norm(matrix_relative - _I4))
        ),
        "positive_scalar": scalar,
        "state_records": state_records,
    }


def _semantic_controls() -> dict[str, Any]:
    nu = 0.5
    h = _tail(nu)
    tail = normalize_involutory_tail(
        "fr_r1b_semantic_tail",
        (
            InvolutoryTailTerm("ZI", 0.75, _ZI),
            InvolutoryTailTerm("IZ", 0.25, _IZ),
        ),
    )
    operator_map = {
        component.component_id: operator
        for component, operator in zip(tail.components, tail.operators, strict=True)
    }
    records = []
    for tau in (-0.2, 0.2):
        distribution = finite_rte_distribution(tau, 2)
        events = enumerate_rte_events(tail.components, distribution)
        enumerated = exact_enumerated_event_mean_operator(events, operator_map)
        polynomial = finite_taylor_operator(h, tau, 2)
        expected = polynomial / distribution.exact_finite_distribution
        ordinary_residual = _operator_norm(enumerated - expected)
        controlled_enumerated = np.zeros((8, 8), dtype=np.complex128)
        for event in events:
            controlled = np.zeros((8, 8), dtype=np.complex128)
            controlled[:4, :4] = _I4
            controlled[4:, 4:] = event_unitary(event, operator_map)
            controlled_enumerated += event.event_probability * controlled
        controlled_expected = np.zeros((8, 8), dtype=np.complex128)
        controlled_expected[:4, :4] = _I4
        controlled_expected[4:, 4:] = expected
        controlled_residual = _operator_norm(controlled_enumerated - controlled_expected)
        spectrum = _spectral_values(nu, tau, 2)
        scalar = _scalar_record(spectrum, 1)
        gamma = float(scalar["gamma_mid"])
        reconstructed = gamma * (polynomial / gamma) / distribution.exact_finite_distribution
        scalar_residual = _operator_norm(reconstructed - expected)
        records.append(
            {
                "semantic_control_id": f"semantic_tau{'+' if tau > 0 else '-'}0p2_K2",
                "signed_short_time": tau,
                "event_count": len(events),
                "probability_sum": float(sum(event.event_probability for event in events)),
                "ordinary_operator_residual": ordinary_residual,
                "controlled_relative_phase_residual": controlled_residual,
                "controlled_positive_scalar_residual": scalar_residual,
                "gamma_mid": gamma,
                "pass": bool(
                    ordinary_residual <= SEMANTIC_ATOL
                    and controlled_residual <= SEMANTIC_ATOL
                    and scalar_residual <= SEMANTIC_ATOL
                ),
            }
        )
    return {
        "records": records,
        "all_pass": all(record["pass"] for record in records),
        "maximum_residual": max(
            max(
                record["ordinary_operator_residual"],
                record["controlled_relative_phase_residual"],
                record["controlled_positive_scalar_residual"],
            )
            for record in records
        ),
    }


def _expected_matches_run(expected: Mapping[str, Any], conditions: list[dict[str, Any]]) -> bool:
    condition_ids = [record["condition"]["condition_id"] for record in conditions]
    state_ids = [state["state_id"] for record in conditions for state in record["state_records"]]
    return bool(
        condition_ids == expected["condition_ids"]
        and state_ids == expected["state_ids"]
        and list(METHOD_ORDER) == expected["method_order"]
    )


def run_fr_revision_nonuniform(
    expected: Mapping[str, Any], *, provenance: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    validate_fr_r1b_expected(expected)
    started = time.perf_counter()
    conditions_spec = fixed_conditions()
    cache_keys = sorted(
        {
            (
                float(condition["nu"]),
                str(condition["deterministic_block"]),
                int(condition["sigma"]),
            )
            for condition in conditions_spec
        }
    )
    state_cache = {key: _fixed_states(*key) for key in cache_keys}
    conditions = [_condition_record(condition, state_cache) for condition in conditions_spec]
    semantic = _semantic_controls()
    rows = [state for condition in conditions for state in condition["state_records"]]
    primary_supplied = [
        row
        for row in rows
        if row["scope"] == "primary"
        and row["state_label"] == "fixed_supplied_superposition"
    ]
    soundness_failures = [
        {"state_id": row["state_id"], "method": label}
        for row in rows
        for label, method in row["methods"].items()
        if method["phase_bound_pass"] is False or method["radius_bound_pass"] is False
    ]
    r0 = bool(
        len(conditions) == EXPECTED_MATRIX_CONDITIONS
        and len(rows) == EXPECTED_STATE_ROWS
        and len(semantic["records"]) == EXPECTED_SEMANTIC_CONTROLS
        and semantic["all_pass"]
        and _expected_matches_run(expected, conditions)
    )
    r1 = bool(
        len(primary_supplied) == 18
        and all(row["available_rho_valid"] for row in primary_supplied)
    )
    r2 = not soundness_failures
    widths_by_nu: dict[str, list[float]] = {}
    for condition in conditions:
        widths_by_nu.setdefault(str(condition["condition"]["nu"]), []).append(
            float(condition["positive_scalar"]["centered_hermitian_spectral_width"])
        )
    involution_width_zero = all(width <= 2e-14 for width in widths_by_nu["1.0"])
    nonuniform_width_positive = any(
        width > 2e-14 for nu in ("0.0", "0.5") for width in widths_by_nu[nu]
    )
    r3 = bool(involution_width_zero and nonuniform_width_positive)
    r4_witnesses: list[dict[str, Any]] = []
    optimized_strict_witnesses: list[dict[str, Any]] = []
    r5_witnesses: list[dict[str, Any]] = []
    for row in primary_supplied:
        condition = next(
            item["condition"] for item in conditions if item["condition"]["condition_id"] == row["condition_id"]
        )
        if float(condition["nu"]) not in (0.0, 0.5):
            continue
        common_norm = row["methods"]["SCALAR-NORM-COMMON-I1"]
        common_fr = row["methods"]["SCALAR-FR-COMMON-I1"]
        opt_norm = row["methods"]["OPT-SCALAR-NORM-I1"]
        opt_fr = row["methods"]["OPT-SCALAR-FR-I1"]
        if (
            common_norm["applicable"]
            and common_fr["applicable"]
            and common_fr["phase_upper_bound"]
            < common_norm["phase_upper_bound"] - row["numerical_atol"]
        ):
            r4_witnesses.append(
                {
                    "state_id": row["state_id"],
                    "norm": common_norm["phase_upper_bound"],
                    "fr": common_fr["phase_upper_bound"],
                }
            )
        if (
            opt_norm["applicable"]
            and opt_fr["applicable"]
            and opt_fr["phase_upper_bound"]
            < opt_norm["phase_upper_bound"] - row["numerical_atol"]
        ):
            optimized_strict_witnesses.append(
                {
                    "state_id": row["state_id"],
                    "norm": opt_norm["phase_upper_bound"],
                    "fr": opt_fr["phase_upper_bound"],
                }
            )
        for budget in PHASE_BUDGETS:
            key = f"{budget:.0e}"
            if opt_fr["certified_at_beta"][key] and not opt_norm["certified_at_beta"][key]:
                r5_witnesses.append({"state_id": row["state_id"], "beta": budget})
    r4 = bool(r4_witnesses)
    r5 = bool(r5_witnesses)
    r6 = bool(
        all(
            row["available_rho"] == SUPPLIED_RHO
            and not row["methods"]["SCALAR-FR-COMMON-I1"]["oracle"]
            and not row["methods"]["OPT-SCALAR-FR-I1"]["oracle"]
            for row in primary_supplied
        )
    )
    control_rows = [
        row for row in rows if row["scope"] in {"negative_time_control", "k4_control"}
    ]
    polar_rows = [
        row
        for row in primary_supplied
        if "_nu1_" in row["condition_id"]
    ]
    r7 = bool(
        all(row["all_applicable_bounds_pass"] for row in control_rows)
        and all(
            row["methods"]["INVOLUTION-POLAR-I1"]["applicable"]
            and row["methods"]["INVOLUTION-POLAR-I1"]["phase_bound_pass"]
            and row["methods"]["INVOLUTION-POLAR-I1"]["radius_bound_pass"]
            for row in polar_rows
        )
        and semantic["all_pass"]
    )
    gates = {
        "R0_completeness_and_semantics": r0,
        "R1_input_certificate": r1,
        "R2_soundness": r2,
        "R3_nonuniform_mechanism": r3,
        "R4_same_information_common_scalar_gain": r4,
        "R5_decision_relevance": r5,
        "R6_oracle_independence": r6,
        "R7_control_transfer": r7,
    }
    scalar_norm_explains_all = not optimized_strict_witnesses
    if not (r0 and r1 and r2 and r7):
        decision = "STOP_FR_R_INPUT_OR_SOUNDNESS"
    elif not r3 or not r4 or scalar_norm_explains_all:
        decision = "STOP_FR_R_INVOLUTION_OR_SCALAR_ONLY"
    elif not r5 or not r6:
        decision = "MECHANISM_ONLY_NO_PRACTICAL_GO"
    else:
        decision = "GO_FR_R2_CANDIDATE_CONDITIONAL_ON_SUPPLIED_STATE"
    applicable_methods = [
        method for row in rows for method in row["methods"].values() if method["applicable"]
    ]
    payload: dict[str, Any] = {
        "schema_version": FR_R1B_SCHEMA_VERSION,
        "validation_method": FR_R1B_METHOD,
        "scope": "preregistered_nonuniform_four_by_four_toy_only",
        "provenance": dict(provenance or {}),
        "expected_fingerprint": expected["expected_fingerprint"],
        "configuration_fingerprint": expected["configuration_fingerprint"],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "contract": {
            "phase_budgets_rad": list(PHASE_BUDGETS),
            "radius_floor": RADIUS_FLOOR,
            "supplied_rho": SUPPLIED_RHO,
            "gamma_interval": [GAMMA_MIN, GAMMA_MAX],
            "mandatory_stop_after_result": True,
            "supplied_state_conditioned_claim_only": True,
        },
        "final_cost_evaluation_performed": False,
        "h4_or_h12_evaluation_performed": False,
        "circuit_compilation_performed": False,
        "monte_carlo_sampling_performed": False,
        "fr_r2_started": False,
        "semantic_controls": semantic,
        "conditions": conditions,
        "gates": gates,
        "gate_witnesses": {
            "R4": r4_witnesses,
            "optimized_strict_gain": optimized_strict_witnesses,
            "R5": r5_witnesses,
        },
        "summary": {
            "matrix_condition_count": len(conditions),
            "state_record_count": len(rows),
            "semantic_control_count": len(semantic["records"]),
            "primary_supplied_state_count": len(primary_supplied),
            "method_record_count": len(rows) * len(METHOD_ORDER),
            "applicable_method_record_count": len(applicable_methods),
            "soundness_failure_count": len(soundness_failures),
            "soundness_failures": soundness_failures,
            "minimum_primary_supplied_reference_radius": min(
                row["reference_radius"] for row in primary_supplied
            ),
            "maximum_semantic_residual": semantic["maximum_residual"],
            "maximum_spectral_vs_matrix_local_error_abs_difference": max(
                condition["spectral_vs_matrix_local_error_abs_difference"]
                for condition in conditions
            ),
            "involution_radial_width_zero": involution_width_zero,
            "nonuniform_radial_width_positive": nonuniform_width_positive,
            "same_information_common_scalar_witness_count": len(r4_witnesses),
            "optimized_strict_gain_witness_count": len(optimized_strict_witnesses),
            "one_sided_decision_witness_count": len(r5_witnesses),
            "scalar_norm_explains_all_optimized_differences": scalar_norm_explains_all,
            "decision": decision,
            "execution_valid": bool(r0 and r1 and r2 and r7),
            "research_go_authorized": bool(
                decision == "GO_FR_R2_CANDIDATE_CONDITIONAL_ON_SUPPLIED_STATE"
            ),
            "mandatory_stop_observed": True,
            "fr_r2_started": False,
        },
        "performance": {"total_seconds": float(time.perf_counter() - started)},
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_fr_r1b_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FR_R1B_SCHEMA_VERSION:
        raise ValueError("Unsupported FR-R1b result schema.")
    fingerprint = payload.get("validation_fingerprint")
    without = deepcopy(dict(payload))
    without.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without):
        raise ValueError("FR-R1b validation fingerprint mismatch.")
    for flag in (
        "final_cost_evaluation_performed",
        "h4_or_h12_evaluation_performed",
        "circuit_compilation_performed",
        "monte_carlo_sampling_performed",
        "fr_r2_started",
    ):
        if payload.get(flag) is not False:
            raise ValueError(f"FR-R1b scope flag must remain false: {flag}")
    summary = payload.get("summary", {})
    if summary.get("matrix_condition_count") != EXPECTED_MATRIX_CONDITIONS:
        raise ValueError("FR-R1b result matrix-condition count mismatch.")
    if summary.get("state_record_count") != EXPECTED_STATE_ROWS:
        raise ValueError("FR-R1b result state count mismatch.")
    if summary.get("semantic_control_count") != EXPECTED_SEMANTIC_CONTROLS:
        raise ValueError("FR-R1b result semantic-control count mismatch.")
    if summary.get("mandatory_stop_observed") is not True:
        raise ValueError("FR-R1b mandatory stop was not observed.")


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Refusing to overwrite different FR-R1b artifact: {target}")
    target.write_text(serialized, encoding="utf-8")


def write_fr_r1b_expected(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_fr_r1b_expected(payload)
    _write_json(payload, path)


def write_fr_r1b_payload(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_fr_r1b_payload(payload)
    _write_json(payload, path)

"""FR-1 validation of finite-RTE phase/radius separation on fixed 2x2 toys.

The numerical grid and decision gates are preregistered in
``docs/research/finite_rte_phase_amplitude_fr1_preregistration.md``.  This
module deliberately does not construct chemistry Hamiltonians, compile
circuits, estimate a full RPE cost, or continue to FR-2.
"""

from __future__ import annotations

import cmath
import hashlib
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .rte import (
    InvolutoryTailTerm,
    enumerate_rte_events,
    event_unitary,
    exact_enumerated_event_mean_operator,
    finite_rte_distribution,
    finite_taylor_operator,
    normalize_involutory_tail,
)


FINITE_RTE_PHASE_AMPLITUDE_SCHEMA_VERSION = "finite_rte_phase_amplitude_fr1_v1"
FINITE_RTE_PHASE_AMPLITUDE_METHOD = (
    "relative_hermitian_antihermitian_phase_radius_bound_v1"
)
NUMERICAL_MULTIPLIER = 512.0
SEMANTIC_ATOL = 1e-12
PRIMARY_PHASE_TARGET = 1e-3
PRIMARY_RATIO_TARGET = 0.5
MAX_ANALYTIC_SPECTRAL_TIME = 0.8

_I2 = np.eye(2, dtype=np.complex128)
_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.complex128))
    return hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest()


def _complex_payload(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def _operator_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=np.complex128), ord=2))


def _pauli_exponential(matrix: np.ndarray, time_value: float) -> np.ndarray:
    """Exponentiate a traceless 2x2 Hermitian Pauli combination analytically."""
    operator = np.asarray(matrix, dtype=np.complex128)
    if operator.shape != (2, 2):
        raise ValueError("FR-1 Pauli exponential requires a 2x2 matrix.")
    if not np.allclose(operator, operator.conj().T, atol=1e-14, rtol=0.0):
        raise ValueError("FR-1 Pauli exponential requires a Hermitian matrix.")
    trace = complex(np.trace(operator))
    if abs(trace) > 1e-13:
        raise ValueError("FR-1 analytic Pauli exponential requires zero trace.")
    scale_squared = float(np.trace(operator @ operator).real / 2.0)
    if scale_squared < -1e-14:
        raise ValueError("Invalid Pauli scale.")
    scale = math.sqrt(max(0.0, scale_squared))
    if scale == 0.0:
        return _I2.copy()
    argument = float(time_value) * scale
    return (
        math.cos(argument) * _I2
        - 1j * math.sin(argument) / scale * operator
    )


def _paired_normalization(tau: float, cutoff: int) -> float:
    return float(finite_rte_distribution(float(tau), int(cutoff)).exact_finite_distribution)


def _scalar_relative_error(x: float, cutoff: int) -> complex:
    polynomial = sum(
        (-1j * float(x)) ** degree / math.factorial(degree)
        for degree in range(int(cutoff) + 2)
    )
    return complex(cmath.exp(1j * float(x)) * polynomial - 1.0)


def _strong_local_error(eta: float, cutoff: int) -> float:
    """Exact interval supremum for the preregistered K and eta range.

    For x >= 0, the derivative of |d_K(x)|^2 is

    K=0: 2*x*(1-cos(x)),
    K=2: x^3*(x^2+2*cos(x)-2)/6,
    K=4: x^5*(x^4-12*x^2+24*(1-cos(x)))/1440.

    The first two are non-negative by 1-cos(x) <= x^2/2.  For K=4 and
    |x| <= 0.8, the alternating cosine bound gives a lower derivative
    factor x^6*(56-x^2)/1680 >= 0.  The even magnitude therefore reaches
    its interval supremum at |x|=eta; no finite-grid maximum is used.
    """
    eta = abs(float(eta))
    cutoff = int(cutoff)
    if cutoff not in (0, 2, 4):
        raise ValueError("FR-1 analytic spectral supremum supports K=0,2,4 only.")
    if eta > MAX_ANALYTIC_SPECTRAL_TIME + 1e-15:
        raise ValueError("FR-1 analytic spectral proof is restricted to eta <= 0.8.")
    return float(abs(_scalar_relative_error(eta, cutoff)))


def _old_local_error(eta: float, cutoff: int) -> float:
    """Positive-term Taylor tail without cancellation."""
    eta = abs(float(eta))
    degree = int(cutoff) + 2
    term = eta**degree / math.factorial(degree)
    total = term
    for next_degree in range(degree + 1, degree + 10000):
        if term == 0.0:
            break
        term *= eta / next_degree
        updated = total + term
        if updated == total:
            break
        total = updated
    return float(total)


def _compose_local_error(local_error: float, occurrence_count: int) -> float:
    local_error = float(local_error)
    count = int(occurrence_count)
    if local_error == 0.0 or count == 0:
        return 0.0
    return float(math.expm1(count * math.log1p(local_error)))


def _phase_distance(left: complex, right: complex) -> float | None:
    if abs(left) < 1e-12 or abs(right) < 1e-12:
        return None
    return float(abs(np.angle(complex(left) * np.conj(complex(right)))) )


def _proposal_bound(
    *,
    rho_input: float,
    a_sum: float,
    b_sum: float,
    e_sum: float,
    product_remainder: float,
    normalization_product: float,
) -> dict[str, Any]:
    rho = float(rho_input)
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho_input must lie in (0, 1].")
    kappa = math.sqrt(max(0.0, 1.0 - rho * rho)) / rho
    off_diagonal = kappa * (float(e_sum) + float(product_remainder))
    lower_real = 1.0 - float(a_sum) - float(product_remainder) - off_diagonal
    upper_imag = float(b_sum) + float(product_remainder) + off_diagonal
    applicable = bool(lower_real > 0.0)
    return {
        "rho_input": rho,
        "kappa": float(kappa),
        "a_sum": float(a_sum),
        "b_sum": float(b_sum),
        "e_sum": float(e_sum),
        "product_remainder": float(product_remainder),
        "lower_real": float(lower_real),
        "upper_imag_abs": float(upper_imag),
        "applicable": applicable,
        "phase_upper_bound": (
            float(math.atan2(upper_imag, lower_real)) if applicable else None
        ),
        "observed_radius_lower_bound": (
            float(rho * lower_real / normalization_product)
            if applicable
            else None
        ),
    }


def _norm_bound(
    *,
    rho_input: float,
    composed_error: float,
    normalization_product: float,
) -> dict[str, Any]:
    rho = float(rho_input)
    error = float(composed_error)
    applicable = bool(error < rho)
    return {
        "rho_input": rho,
        "composed_operator_error": error,
        "applicable": applicable,
        "phase_upper_bound": (
            float(math.asin(min(1.0, error / rho))) if applicable else None
        ),
        "observed_radius_lower_bound": (
            float((rho - error) / normalization_product)
            if applicable
            else None
        ),
    }


def _canonical_eigensystem(unitary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(np.asarray(unitary, dtype=np.complex128))
    order = np.argsort(np.angle(eigenvalues), kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for column in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, column]
        vector /= np.linalg.norm(vector)
        pivot = int(np.argmax(np.abs(vector)))
        phase = np.angle(vector[pivot])
        vector *= np.exp(-1j * phase)
        if vector[pivot].real < 0.0:
            vector *= -1.0
        eigenvectors[:, column] = vector
    overlap = eigenvectors.conj().T @ eigenvectors
    if not np.allclose(overlap, _I2, atol=2e-12, rtol=0.0):
        raise RuntimeError("The unitary eigensystem was not orthonormal.")
    return eigenvalues, eigenvectors


def _states_for_reference(
    unitary: np.ndarray,
    physical_hamiltonian: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
    _eigenvalues, eigenvectors = _canonical_eigensystem(unitary)
    reference = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    mixture = (
        math.sqrt(0.9) * eigenvectors[:, 0]
        + math.sqrt(0.1) * eigenvectors[:, 1]
    )
    mixture /= np.linalg.norm(mixture)
    _physical_values, physical_vectors = np.linalg.eigh(physical_hamiltonian)
    physical = np.asarray(physical_vectors[:, 0], dtype=np.complex128)
    states = {
        "reference_eigenstate": reference,
        "analytic_mixture_state": mixture,
        "physical_ground_state": physical,
    }
    available_rho = {
        "reference_eigenstate": 1.0,
        "analytic_mixture_state": 0.8,
        "physical_ground_state": None,
    }
    return states, available_rho


def _state_record(
    *,
    condition: Mapping[str, Any],
    state_label: str,
    state: np.ndarray,
    available_rho: float | None,
    exact_operator: np.ndarray,
    corrected_operator: np.ndarray,
    normalization_product: float,
    local: Mapping[str, float],
) -> dict[str, Any]:
    reference_signal = complex(np.vdot(state, exact_operator @ state))
    corrected_signal = complex(np.vdot(state, corrected_operator @ state))
    observed_signal = corrected_signal / normalization_product
    reference_radius_raw = float(abs(reference_signal))
    reference_radius = reference_radius_raw
    if 1.0 < reference_radius <= 1.0 + 2e-12:
        reference_radius = 1.0
    actual_phase = _phase_distance(corrected_signal, reference_signal)
    if actual_phase is None:
        raise RuntimeError("FR-1 fixed grid produced an undefined phase.")
    actual_radius = float(abs(observed_signal))
    count = int(condition["r"]) * int(condition["q"])
    numerical_atol = float(
        NUMERICAL_MULTIPLIER
        * np.finfo(float).eps
        * max(1, int(condition["q"]) * (int(condition["r"]) + 2))
        * (
            1.0
            + max(
                abs(reference_signal),
                abs(corrected_signal),
                normalization_product,
            )
        )
    )
    product_remainder = float(local["product_remainder"])
    proposed_ref = _proposal_bound(
        rho_input=reference_radius,
        a_sum=count * float(local["a"]),
        b_sum=count * float(local["b"]),
        e_sum=count * float(local["e"]),
        product_remainder=product_remainder,
        normalization_product=normalization_product,
    )
    old_ref = _norm_bound(
        rho_input=reference_radius,
        composed_error=float(local["old_composed"]),
        normalization_product=normalization_product,
    )
    strong_ref = _norm_bound(
        rho_input=reference_radius,
        composed_error=float(local["strong_composed"]),
        normalization_product=normalization_product,
    )
    methods: dict[str, Any] = {
        "PROPOSED_REF": proposed_ref,
        "OLD_NORM_REF": old_ref,
        "STRONG_NORM_REF": strong_ref,
    }
    if available_rho is not None:
        methods["PROPOSED_AVAILABLE"] = _proposal_bound(
            rho_input=float(available_rho),
            a_sum=count * float(local["a"]),
            b_sum=count * float(local["b"]),
            e_sum=count * float(local["e"]),
            product_remainder=product_remainder,
            normalization_product=normalization_product,
        )
        methods["OLD_NORM_AVAILABLE"] = _norm_bound(
            rho_input=float(available_rho),
            composed_error=float(local["old_composed"]),
            normalization_product=normalization_product,
        )
        methods["STRONG_NORM_AVAILABLE"] = _norm_bound(
            rho_input=float(available_rho),
            composed_error=float(local["strong_composed"]),
            normalization_product=normalization_product,
        )
    all_applicable_pass = True
    for method in methods.values():
        if method["applicable"]:
            phase_pass = bool(
                actual_phase <= float(method["phase_upper_bound"]) + numerical_atol
            )
            radius_pass = bool(
                actual_radius + numerical_atol
                >= float(method["observed_radius_lower_bound"])
            )
        else:
            phase_pass = None
            radius_pass = None
        method["phase_bound_pass"] = phase_pass
        method["radius_bound_pass"] = radius_pass
        if phase_pass is False or radius_pass is False:
            all_applicable_pass = False
    available_rho_valid = bool(
        available_rho is None
        or float(available_rho) <= reference_radius + numerical_atol
    )
    return {
        "condition_id": condition["condition_id"],
        "scope": condition["scope"],
        "state_label": state_label,
        "state_fingerprint": _array_fingerprint(state),
        "available_rho": available_rho,
        "available_rho_valid": available_rho_valid,
        "reference_signal": _complex_payload(reference_signal),
        "corrected_signal": _complex_payload(corrected_signal),
        "observed_signal": _complex_payload(observed_signal),
        "reference_radius": reference_radius,
        "reference_radius_raw": reference_radius_raw,
        "actual_phase_error": actual_phase,
        "actual_observed_radius": actual_radius,
        "normalization_product": float(normalization_product),
        "numerical_atol": numerical_atol,
        "methods": methods,
        "all_applicable_bounds_pass": all_applicable_pass,
    }


def _condition_record(condition: Mapping[str, Any]) -> dict[str, Any]:
    theta = float(condition["theta"])
    sigma = int(condition["sigma"])
    total_time = float(condition["total_time"])
    q = int(condition["q"])
    r = int(condition["r"])
    cutoff = int(condition["cutoff"])
    step_time = total_time / q
    tail_time = sigma * step_time / r
    h_d = 0.7 * _Z
    h_r = math.cos(theta) * _Z + math.sin(theta) * _X
    deterministic_half = _pauli_exponential(h_d, step_time / 2.0)
    deterministic_full = _pauli_exponential(h_d, step_time)
    exact_tail = _pauli_exponential(h_r, sigma * step_time)
    short_exact_tail = _pauli_exponential(h_r, tail_time)
    polynomial = finite_taylor_operator(h_r, tail_time, cutoff)
    polynomial_power = np.linalg.matrix_power(polynomial, r)
    if bool(condition["asymmetric"]):
        exact_step = deterministic_full @ exact_tail
        corrected_step = deterministic_full @ polynomial_power
    else:
        exact_step = deterministic_half @ exact_tail @ deterministic_half
        corrected_step = deterministic_half @ polynomial_power @ deterministic_half
    exact_operator = np.linalg.matrix_power(exact_step, q)
    corrected_operator = np.linalg.matrix_power(corrected_step, q)
    distribution = finite_rte_distribution(tail_time, cutoff)
    occurrence_count = q * r
    normalization_product = float(
        math.exp(occurrence_count * math.log(distribution.exact_finite_distribution))
    )
    relative = short_exact_tail.conj().T @ polynomial - _I2
    hermitian = (relative + relative.conj().T) / 2.0
    antihermitian_coordinate = (relative - relative.conj().T) / (2.0j)
    local_a = _operator_norm(hermitian)
    local_b = _operator_norm(antihermitian_coordinate)
    local_e = _operator_norm(relative)
    product_remainder = float(
        _compose_local_error(local_e, occurrence_count) - occurrence_count * local_e
    )
    eta = abs(tail_time)
    old_local = _old_local_error(eta, cutoff)
    strong_local = _strong_local_error(eta, cutoff)
    local = {
        "a": local_a,
        "b": local_b,
        "e": local_e,
        "old_local": old_local,
        "strong_local": strong_local,
        "old_composed": _compose_local_error(old_local, occurrence_count),
        "strong_composed": _compose_local_error(strong_local, occurrence_count),
        "product_remainder": max(0.0, product_remainder),
        "matrix_vs_scalar_strong_abs_difference": abs(local_e - strong_local),
    }
    physical_hamiltonian = h_d + sigma * h_r
    states, available_rhos = _states_for_reference(
        exact_operator,
        physical_hamiltonian,
    )
    state_records = [
        _state_record(
            condition=condition,
            state_label=label,
            state=state,
            available_rho=available_rhos[label],
            exact_operator=exact_operator,
            corrected_operator=corrected_operator,
            normalization_product=normalization_product,
            local=local,
        )
        for label, state in states.items()
    ]
    return {
        "condition": dict(condition),
        "dimensionless_short_time": float(tail_time),
        "occurrence_count": occurrence_count,
        "paired_normalization_per_occurrence": float(
            distribution.exact_finite_distribution
        ),
        "normalization_product": normalization_product,
        "local_relative_error": local,
        "exact_unitarity_defect": _operator_norm(
            exact_operator.conj().T @ exact_operator - _I2
        ),
        "state_records": state_records,
    }


def _fixed_conditions() -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []

    def add(
        scope: str,
        *,
        theta: float,
        sigma: int = 1,
        total_time: float = 0.8,
        q: int = 1,
        r: int = 1,
        cutoff: int = 2,
        asymmetric: bool = False,
        suffix: str = "",
    ) -> None:
        condition_id = (
            f"{scope}_theta{theta / math.pi:.12g}pi_sigma{sigma:+d}_"
            f"T{total_time:.12g}_q{q}_r{r}_K{cutoff}_asym{int(asymmetric)}{suffix}"
        )
        conditions.append(
            {
                "condition_id": condition_id,
                "scope": scope,
                "theta": float(theta),
                "sigma": int(sigma),
                "total_time": float(total_time),
                "q": int(q),
                "r": int(r),
                "cutoff": int(cutoff),
                "asymmetric": bool(asymmetric),
            }
        )

    for q in (1, 2, 4, 8, 16):
        add("primary", theta=math.pi / 3.0, q=q)
        add("commuting_control", theta=0.0, q=q)
    for q in (2, 8):
        add("strong_noncommuting_control", theta=math.pi / 2.0, q=q)
    for r in (1, 2, 4):
        add("short_step_control", theta=math.pi / 3.0, q=2, r=r)
    add("negative_time_control", theta=math.pi / 3.0, q=2, sigma=-1)
    add("asymmetric_control", theta=math.pi / 3.0, q=2, asymmetric=True)
    for cutoff in (0, 4):
        for q in (2, 8):
            add("cutoff_control", theta=math.pi / 3.0, q=q, cutoff=cutoff)
    for cutoff in (0, 2, 4):
        for delta in (0.05, 0.1, 0.2, 0.4):
            add(
                "local_order_calibration",
                theta=math.pi / 3.0,
                total_time=delta,
                q=1,
                cutoff=cutoff,
                suffix=f"_delta{delta:.12g}",
            )
    identifiers = [item["condition_id"] for item in conditions]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("FR-1 fixed condition identifiers are not unique.")
    return conditions


def _semantic_checks() -> dict[str, Any]:
    tail = normalize_involutory_tail(
        "fr1_identity_semantics",
        (
            InvolutoryTailTerm("I", 0.2, _I2),
            InvolutoryTailTerm("X", 0.4, _X),
            InvolutoryTailTerm("Z", 0.4, _Z),
        ),
    )
    operator_map = {
        component.component_id: operator
        for component, operator in zip(tail.components, tail.operators, strict=True)
    }
    records: list[dict[str, Any]] = []
    for cutoff in (0, 2):
        for tau in (-0.2, 0.2):
            distribution = finite_rte_distribution(tau, cutoff)
            events = enumerate_rte_events(tail.components, distribution)
            enumerated = exact_enumerated_event_mean_operator(events, operator_map)
            expected = finite_taylor_operator(
                tail.normalized_hamiltonian,
                tau,
                cutoff,
            ) / distribution.exact_finite_distribution
            ordinary_residual = _operator_norm(enumerated - expected)
            controlled_enumerated = np.zeros((4, 4), dtype=np.complex128)
            for event in events:
                event_matrix = event_unitary(event, operator_map)
                controlled = np.zeros((4, 4), dtype=np.complex128)
                controlled[:2, :2] = _I2
                controlled[2:, 2:] = event_matrix
                controlled_enumerated += event.event_probability * controlled
            controlled_expected = np.zeros((4, 4), dtype=np.complex128)
            controlled_expected[:2, :2] = _I2
            controlled_expected[2:, 2:] = expected
            controlled_residual = _operator_norm(
                controlled_enumerated - controlled_expected
            )
            records.append(
                {
                    "cutoff": cutoff,
                    "dimensionless_time": tau,
                    "event_count": len(events),
                    "probability_sum": float(
                        math.fsum(event.event_probability for event in events)
                    ),
                    "ordinary_operator_residual": ordinary_residual,
                    "controlled_relative_phase_residual": controlled_residual,
                    "pass": bool(
                        ordinary_residual <= SEMANTIC_ATOL
                        and controlled_residual <= SEMANTIC_ATOL
                    ),
                }
            )
    return {
        "tail_lambda": float(tail.lambda_r),
        "normalized_hamiltonian_fingerprint": _array_fingerprint(
            tail.normalized_hamiltonian
        ),
        "records": records,
        "maximum_ordinary_operator_residual": max(
            item["ordinary_operator_residual"] for item in records
        ),
        "maximum_controlled_relative_phase_residual": max(
            item["controlled_relative_phase_residual"] for item in records
        ),
        "all_pass": all(item["pass"] for item in records),
    }


def _local_order_summary(condition_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cutoff in (0, 2, 4):
        selected = [
            record
            for record in condition_records
            if record["condition"]["scope"] == "local_order_calibration"
            and record["condition"]["cutoff"] == cutoff
        ]
        selected.sort(key=lambda item: abs(item["dimensionless_short_time"]))
        times = np.asarray(
            [abs(item["dimensionless_short_time"]) for item in selected],
            dtype=float,
        )
        radial = np.asarray(
            [item["local_relative_error"]["a"] for item in selected], dtype=float
        )
        tangential = np.asarray(
            [item["local_relative_error"]["b"] for item in selected], dtype=float
        )
        radial_slope = float(np.polyfit(np.log(times), np.log(radial), 1)[0])
        tangential_slope = float(
            np.polyfit(np.log(times), np.log(tangential), 1)[0]
        )
        output.append(
            {
                "cutoff": cutoff,
                "delta_values": times.tolist(),
                "radial_values": radial.tolist(),
                "tangential_values": tangential.tolist(),
                "radial_loglog_slope": radial_slope,
                "tangential_loglog_slope": tangential_slope,
                "expected_local_orders": [cutoff + 2, cutoff + 3],
                "gate_relevant": False,
            }
        )
    return output


def run_finite_rte_phase_amplitude_fr1(
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the preregistered FR-1 grid and return a fingerprinted payload."""
    started = time.perf_counter()
    semantic = _semantic_checks()
    conditions = _fixed_conditions()
    condition_records = [_condition_record(item) for item in conditions]
    state_records = [
        state_record
        for condition_record in condition_records
        for state_record in condition_record["state_records"]
    ]
    all_bounds_sound = all(
        record["all_applicable_bounds_pass"] and record["available_rho_valid"]
        for record in state_records
    )
    primary_records = [
        record
        for record in state_records
        if record["scope"] == "primary"
        and record["state_label"] == "analytic_mixture_state"
    ]
    utility_records: list[dict[str, Any]] = []
    for record in primary_records:
        proposed = record["methods"]["PROPOSED_AVAILABLE"]
        strong = record["methods"]["STRONG_NORM_AVAILABLE"]
        ratio = None
        ratio_trigger = False
        phase_target_trigger = False
        if proposed["applicable"] and strong["applicable"]:
            ratio = float(
                proposed["phase_upper_bound"] / strong["phase_upper_bound"]
            )
            ratio_trigger = bool(ratio <= PRIMARY_RATIO_TARGET)
        if proposed["applicable"]:
            phase_target_trigger = bool(
                proposed["phase_upper_bound"] <= PRIMARY_PHASE_TARGET
                and (
                    not strong["applicable"]
                    or strong["phase_upper_bound"] > PRIMARY_PHASE_TARGET
                )
            )
        utility_records.append(
            {
                "condition_id": record["condition_id"],
                "proposed_phase_upper_bound": proposed["phase_upper_bound"],
                "strong_phase_upper_bound": strong["phase_upper_bound"],
                "proposed_to_strong_ratio": ratio,
                "positive_proposed_radius_lower_bound": bool(
                    proposed["applicable"]
                    and proposed["observed_radius_lower_bound"] > 0.0
                ),
                "ratio_trigger": ratio_trigger,
                "phase_target_trigger": phase_target_trigger,
                "utility_pass": bool(
                    proposed["applicable"]
                    and proposed["observed_radius_lower_bound"] > 0.0
                    and (ratio_trigger or phase_target_trigger)
                ),
            }
        )
    g2 = any(item["utility_pass"] for item in utility_records)
    g3 = all(record["available_rho_valid"] for record in state_records)
    transfer_scopes = {
        "negative_time_control",
        "asymmetric_control",
        "cutoff_control",
    }
    transfer_records = [
        record for record in state_records if record["scope"] in transfer_scopes
    ]
    g4 = all(record["all_applicable_bounds_pass"] for record in transfer_records)
    gates = {
        "G0_semantic_consistency": bool(semantic["all_pass"]),
        "G1_all_applicable_bounds_sound": bool(all_bounds_sound),
        "G2_available_noncommuting_utility": bool(g2),
        "G3_conditioning_and_rejection": bool(g3),
        "G4_sign_cutoff_and_asymmetry_sound": bool(g4),
    }
    reference_oracle_utility = False
    for record in state_records:
        if record["scope"] != "primary":
            continue
        proposed = record["methods"]["PROPOSED_REF"]
        strong = record["methods"]["STRONG_NORM_REF"]
        if proposed["applicable"] and strong["applicable"]:
            ratio = proposed["phase_upper_bound"] / strong["phase_upper_bound"]
            reference_oracle_utility = reference_oracle_utility or bool(
                proposed["observed_radius_lower_bound"] > 0.0
                and (
                    ratio <= PRIMARY_RATIO_TARGET
                    or (
                        proposed["phase_upper_bound"] <= PRIMARY_PHASE_TARGET
                        and strong["phase_upper_bound"] > PRIMARY_PHASE_TARGET
                    )
                )
            )
    if not (
        gates["G0_semantic_consistency"]
        and gates["G1_all_applicable_bounds_sound"]
        and gates["G3_conditioning_and_rejection"]
        and gates["G4_sign_cutoff_and_asymmetry_sound"]
    ):
        decision = "STOP_FR1_BOUND_INVALID"
    elif gates["G2_available_noncommuting_utility"]:
        decision = "GO_FR2_AVAILABLE"
    elif reference_oracle_utility:
        decision = "GO_FR2_MECHANISM_ONLY"
    else:
        decision = "STOP_FR1_NO_NONCOMMUTING_GAIN"
    applicable_method_records = [
        method
        for record in state_records
        for method in record["methods"].values()
        if method["applicable"]
    ]
    payload: dict[str, Any] = {
        "schema_version": FINITE_RTE_PHASE_AMPLITUDE_SCHEMA_VERSION,
        "validation_method": FINITE_RTE_PHASE_AMPLITUDE_METHOD,
        "scope": "fr1_fixed_two_by_two_mechanism_validation_only",
        "final_cost_evaluation_performed": False,
        "h4_or_h12_evaluation_performed": False,
        "circuit_compilation_performed": False,
        "monte_carlo_sampling_performed": False,
        "provenance": dict(provenance or {}),
        "contract": {
            "preregistration": (
                "docs/research/finite_rte_phase_amplitude_fr1_preregistration.md"
            ),
            "parent_contract": (
                "docs/research/finite_rte_phase_amplitude_contract.md"
            ),
            "primary_ratio_target": PRIMARY_RATIO_TARGET,
            "primary_phase_target_rad": PRIMARY_PHASE_TARGET,
            "semantic_atol": SEMANTIC_ATOL,
            "available_rho_main": 0.8,
            "post_fr1_stop_required": True,
        },
        "analytic_strong_norm": {
            "maximum_dimensionless_time": MAX_ANALYTIC_SPECTRAL_TIME,
            "cutoffs": [0, 2, 4],
            "finite_grid_maximum_used": False,
            "stationary_point_argument": (
                "|d_K(x)|^2 is even and nondecreasing on [0,0.8] for K=0,2,4; "
                "the exact endpoint value is the interval supremum"
            ),
        },
        "semantic_checks": semantic,
        "conditions": condition_records,
        "local_order_diagnostic": _local_order_summary(condition_records),
        "utility_records": utility_records,
        "gates": gates,
        "summary": {
            "condition_count": len(condition_records),
            "state_record_count": len(state_records),
            "applicable_method_record_count": len(applicable_method_records),
            "bound_violation_count": sum(
                method["phase_bound_pass"] is False
                or method["radius_bound_pass"] is False
                for record in state_records
                for method in record["methods"].values()
                if method["applicable"]
            ),
            "maximum_matrix_vs_scalar_strong_abs_difference": max(
                record["local_relative_error"][
                    "matrix_vs_scalar_strong_abs_difference"
                ]
                for record in condition_records
            ),
            "minimum_actual_observed_radius": min(
                record["actual_observed_radius"] for record in state_records
            ),
            "minimum_reference_radius": min(
                record["reference_radius"] for record in state_records
            ),
            "minimum_primary_proposed_to_strong_ratio": min(
                item["proposed_to_strong_ratio"]
                for item in utility_records
                if item["proposed_to_strong_ratio"] is not None
            ),
            "reference_or_oracle_utility_observed": bool(reference_oracle_utility),
            "execution_valid": bool(
                gates["G0_semantic_consistency"]
                and gates["G1_all_applicable_bounds_sound"]
                and gates["G3_conditioning_and_rejection"]
                and gates["G4_sign_cutoff_and_asymmetry_sound"]
            ),
            "decision": decision,
            "fr2_started": False,
        },
        "performance": {"total_seconds": float(time.perf_counter() - started)},
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    return payload


def validate_finite_rte_phase_amplitude_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FINITE_RTE_PHASE_AMPLITUDE_SCHEMA_VERSION:
        raise ValueError("Unsupported finite-RTE phase/amplitude schema.")
    fingerprint = payload.get("validation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("validation_fingerprint must be a SHA-256 hex string.")
    without_fingerprint = deepcopy(dict(payload))
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Finite-RTE phase/amplitude fingerprint mismatch.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("FR-1 cannot contain a final cost evaluation.")
    if payload.get("h4_or_h12_evaluation_performed") is not False:
        raise ValueError("FR-1 cannot contain an H4 or H12 evaluation.")
    if payload.get("circuit_compilation_performed") is not False:
        raise ValueError("FR-1 cannot contain circuit compilation.")
    if payload.get("monte_carlo_sampling_performed") is not False:
        raise ValueError("FR-1 cannot contain Monte Carlo sampling.")
    if payload.get("summary", {}).get("fr2_started") is not False:
        raise ValueError("FR-1 payload must preserve the post-FR-1 stop.")


def write_finite_rte_phase_amplitude_payload(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_finite_rte_phase_amplitude_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

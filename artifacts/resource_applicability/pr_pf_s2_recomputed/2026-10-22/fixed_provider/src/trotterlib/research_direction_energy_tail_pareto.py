"""P-D audit of product-formula energy bias versus randomized-tail burden.

The validation deliberately separates three questions:

* K01: are the finite coefficient lists implemented with their advertised
  practical order on a fixed noncommuting toy problem?
* K02: does raising only the internal H_D formula leave the outer H_D/H_R
  Strang split at second order?
* K03/P-D: on one development split and one frozen holdout split, does an
  energy-only choice differ from a choice that includes random-tail exposure?

The H4 comparison treats exp(-i H_D t) and exp(-i H_R t) as exact dense
two-block evolutions.  It is therefore a screening reference, not an
implementation of a higher-order finite-RTE circuit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

from .df_hamiltonian import PhysicalSector
from .df_partial_randomized_pf import (
    df_hamiltonian_hash,
    select_df_h_d,
    split_df_hamiltonian_by_ld,
)
from .df_partial_s2 import prepare_df_partial_s2
from .finite_rte_signal_validation import dense_df_operator_in_sector
from .pf_decomposition import iter_pf_steps
from .product_formula import _get_w_list
from .research_direction_full_scope import fingerprint
from .rte import finite_rte_distribution
from .rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


EXPECTED_SCHEMA = "research_direction_energy_tail_pareto_expected_v1"
RESULT_SCHEMA = "research_direction_energy_tail_pareto_v1"
METHOD = "pd_k01_k03_two_block_energy_tail_pareto_v1"

FORMULA_SPECS: tuple[dict[str, Any], ...] = (
    {
        "label": "2nd",
        "display_name": "second-order Strang",
        "order": 2,
        "source_id": "canonical_strang_local_registry",
        "source_version": "local_product_formula_py_at_frozen_source_hash",
        "processor": False,
    },
    {
        "label": "4th",
        "display_name": "fourth-order Yoshida triple jump",
        "order": 4,
        "source_id": "yoshida_composition_as_reproduced_in_local_2026_pdf",
        "source_version": "local_2026_pdf_at_frozen_source_hash",
        "processor": False,
    },
    {
        "label": "4th(new_2)",
        "display_name": "project fourth-order m=2 candidate",
        "order": 4,
        "source_id": "evaluation_of_gate_numbers_2026_m2_coefficients",
        "source_version": "eight_decimal_coefficients_in_local_2026_pdf",
        "processor": False,
    },
    {
        "label": "8th(Yoshida)",
        "display_name": "eighth-order Yoshida composition",
        "order": 8,
        "source_id": "yoshida_coefficients_as_reproduced_in_local_2026_pdf",
        "source_version": "local_2026_pdf_at_frozen_source_hash",
        "processor": False,
    },
    {
        "label": "8th(Morales)",
        "display_name": "eighth-order Morales m=8 composition",
        "order": 8,
        "source_id": "morales_arxiv_2210_15817_v1_table_ii",
        "source_version": "arXiv_v1_2022_10_28_local_pdf",
        "processor": False,
    },
)
FORMULA_LABELS = tuple(spec["label"] for spec in FORMULA_SPECS)

DEVELOPMENT_LD = 3
BLIND_HOLDOUT_LD = 4
N_ELECTRONS = 4
DIRECT_DELTAS = (0.025, 0.05, 0.1, 0.2, 0.25, 0.32, 0.4)
COEFFICIENT_WINDOWS = {
    2: (0.025, 0.05, 0.1, 0.2),
    4: (0.025, 0.05, 0.1, 0.2),
    8: (0.2, 0.25, 0.32, 0.4),
}
DECISION_DELTA = 0.4
DECISION_ENERGY_TOLERANCE = 1.0e-6
MINIMUM_TARGET_WEIGHT = 0.9995

TOY_LOCAL_TIMES = (0.2, 0.25, 0.32, 0.4)
TOY_GLOBAL_TOTAL_TIME = 0.8
TOY_GLOBAL_STEP_COUNTS = (2, 3, 4)
ORDER_SLOPE_TOLERANCE = 0.35
NESTED_GLOBAL_MINIMUM_SLOPE_SEPARATION = 1.5

RTE_TOTAL_SHORT_STEPS = 64
RTE_TAYLOR_ORDER = 2
RTE_AUDIT_DELTAS = (0.02, 0.4)
MINIMUM_GAMMA_REDUCTION = 0.20
NUMERICAL_ATOL = 1.0e-9
BIAS_FIT_FLOOR = 1.0e-13

GATE_THRESHOLDS = {
    "coefficient_sum_absolute_tolerance": 1.0e-12,
    "coefficient_symmetry_absolute_tolerance": 1.0e-12,
    "order_slope_absolute_tolerance": ORDER_SLOPE_TOLERANCE,
    "nested_global_minimum_slope_separation": (
        NESTED_GLOBAL_MINIMUM_SLOPE_SEPARATION
    ),
    "decision_delta": DECISION_DELTA,
    "decision_energy_tolerance_hartree": DECISION_ENERGY_TOLERANCE,
    "minimum_target_weight": MINIMUM_TARGET_WEIGHT,
    "minimum_gamma_reduction": MINIMUM_GAMMA_REDUCTION,
    "finite_rte_total_short_steps_per_outer_step": RTE_TOTAL_SHORT_STEPS,
    "finite_rte_taylor_order": RTE_TAYLOR_ORDER,
}


_TOY_A = np.asarray(
    [
        [0.7, 0.2 + 0.1j, -0.15],
        [0.2 - 0.1j, -0.3, 0.05 + 0.12j],
        [-0.15, 0.05 - 0.12j, 0.2],
    ],
    dtype=np.complex128,
)
_TOY_B = np.asarray(
    [
        [-0.2, -0.11 + 0.07j, 0.18 - 0.04j],
        [-0.11 - 0.07j, 0.6, -0.09],
        [0.18 + 0.04j, -0.09, -0.4],
    ],
    dtype=np.complex128,
)


def _log_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.size < 2 or y.size != x.size or np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("A log slope needs matching positive grids with at least two points.")
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def _pf_unitary(
    terms: Sequence[np.ndarray],
    pf_label: str,
    time_value: float,
    *,
    eigensystems: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    if not terms:
        raise ValueError("At least one Hamiltonian term is required.")
    dimension = int(np.asarray(terms[0]).shape[0])
    unitary = np.eye(dimension, dtype=np.complex128)
    for term_index, weight in iter_pf_steps(len(terms), _get_w_list(pf_label)):
        scaled_time = float(time_value) * float(weight)
        if eigensystems is None:
            factor = expm(-1j * scaled_time * np.asarray(terms[term_index]))
        else:
            eigenvalues, eigenvectors = eigensystems[term_index]
            factor = (
                eigenvectors
                * np.exp(-1j * scaled_time * eigenvalues)[np.newaxis, :]
            ) @ eigenvectors.conj().T
        unitary = factor @ unitary
    return unitary


def composition_tail_weights(pf_label: str) -> tuple[float, ...]:
    """Return the H_R coefficients b_j for a two-block symmetric formula."""
    return tuple(
        float(weight)
        for term_index, weight in iter_pf_steps(2, _get_w_list(pf_label))
        if term_index == 1
    )


def _matrix_payload(matrix: np.ndarray) -> dict[str, list[list[float]]]:
    array = np.asarray(matrix, dtype=np.complex128)
    return {
        "real": np.asarray(array.real, dtype=float).tolist(),
        "imag": np.asarray(array.imag, dtype=float).tolist(),
    }


def formula_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in FORMULA_SPECS:
        label = str(spec["label"])
        parameters = tuple(float(value) for value in _get_w_list(label))
        tail_weights = composition_tail_weights(label)
        palindrome_error = max(
            (
                abs(left - right)
                for left, right in zip(tail_weights, reversed(tail_weights))
            ),
            default=0.0,
        )
        order = int(spec["order"])
        rows.append(
            {
                **spec,
                "parameter_list_w0_through_wm": list(parameters),
                "parameter_list_float_hex": [value.hex() for value in parameters],
                "tail_coefficients_in_circuit_order": list(tail_weights),
                "tail_coefficient_float_hex": [value.hex() for value in tail_weights],
                "tail_occurrence_count": len(tail_weights),
                "deterministic_occurrence_count": len(tail_weights) + 1,
                "full_exponential_stage_count": 2 * len(tail_weights) + 1,
                "negative_tail_occurrence_count": sum(
                    value < 0.0 for value in tail_weights
                ),
                "coefficient_sum": float(math.fsum(tail_weights)),
                "coefficient_sum_error": float(abs(math.fsum(tail_weights) - 1.0)),
                "palindrome_max_absolute_error": float(palindrome_error),
                "gamma_r": float(math.fsum(abs(value) for value in tail_weights)),
                "sum_b_squared": float(math.fsum(value * value for value in tail_weights)),
                "odd_moment_residuals": {
                    str(power): float(math.fsum(value**power for value in tail_weights))
                    for power in (3, 5, 7)
                    if power < order
                },
                "coefficient_precision_note": (
                    "practical numerical-order candidate; exact symbolic order is not "
                    "claimed from rounded coefficients"
                    if label == "4th(new_2)"
                    else "published/local decimal coefficients tested on a fixed window"
                ),
            }
        )
    return rows


def _matched_eigenphase_biases(
    unitary: np.ndarray,
    reference_eigenvalues: np.ndarray,
    reference_eigenvectors: np.ndarray,
    time_value: float,
) -> list[float]:
    pf_eigenvalues, pf_eigenvectors = np.linalg.eig(unitary)
    overlaps = np.abs(reference_eigenvectors.conj().T @ pf_eigenvectors) ** 2
    reference_indices, pf_indices = linear_sum_assignment(-overlaps)
    output = [math.nan] * len(reference_eigenvalues)
    for reference_index, pf_index in zip(reference_indices, pf_indices, strict=True):
        reference_energy = float(reference_eigenvalues[reference_index])
        pf_energy = -float(np.angle(pf_eigenvalues[pf_index])) / float(time_value)
        pf_energy += (2.0 * math.pi / float(time_value)) * round(
            (reference_energy - pf_energy) * float(time_value) / (2.0 * math.pi)
        )
        output[int(reference_index)] = abs(pf_energy - reference_energy)
    return [float(value) for value in output]


def evaluate_formula_order_registry() -> dict[str, Any]:
    exact_hamiltonian = _TOY_A + _TOY_B
    exact_eigenvalues, exact_eigenvectors = np.linalg.eigh(exact_hamiltonian)
    registry = formula_registry()
    rows: list[dict[str, Any]] = []
    for registry_row in registry:
        label = str(registry_row["label"])
        order = int(registry_row["order"])
        local_errors: list[float] = []
        branch_biases: list[list[float]] = []
        for time_value in TOY_LOCAL_TIMES:
            observed = _pf_unitary((_TOY_A, _TOY_B), label, time_value)
            exact = expm(-1j * float(time_value) * exact_hamiltonian)
            local_errors.append(float(np.linalg.norm(observed - exact, ord=2)))
            branch_biases.append(
                _matched_eigenphase_biases(
                    observed,
                    exact_eigenvalues,
                    exact_eigenvectors,
                    time_value,
                )
            )
        local_slope = _log_slope(TOY_LOCAL_TIMES, local_errors)

        global_errors: list[float] = []
        global_step_sizes: list[float] = []
        exact_global = expm(-1j * TOY_GLOBAL_TOTAL_TIME * exact_hamiltonian)
        for step_count in TOY_GLOBAL_STEP_COUNTS:
            step_size = TOY_GLOBAL_TOTAL_TIME / int(step_count)
            one_step = _pf_unitary((_TOY_A, _TOY_B), label, step_size)
            global_errors.append(
                float(
                    np.linalg.norm(
                        np.linalg.matrix_power(one_step, int(step_count)) - exact_global,
                        ord=2,
                    )
                )
            )
            global_step_sizes.append(float(step_size))
        global_slope = _log_slope(global_step_sizes, global_errors)

        branch_slopes: list[float | None] = []
        branch_array = np.asarray(branch_biases, dtype=float)
        for branch_index in range(branch_array.shape[1]):
            values = branch_array[:, branch_index]
            mask = values > BIAS_FIT_FLOOR
            branch_slopes.append(
                _log_slope(np.asarray(TOY_LOCAL_TIMES)[mask], values[mask])
                if np.count_nonzero(mask) >= 2
                else None
            )
        normalization_pass = bool(
            registry_row["coefficient_sum_error"]
            <= GATE_THRESHOLDS["coefficient_sum_absolute_tolerance"]
            and registry_row["palindrome_max_absolute_error"]
            <= GATE_THRESHOLDS["coefficient_symmetry_absolute_tolerance"]
        )
        local_order_pass = bool(
            abs(local_slope - (order + 1)) <= ORDER_SLOPE_TOLERANCE
        )
        global_order_pass = bool(abs(global_slope - order) <= ORDER_SLOPE_TOLERANCE)
        rows.append(
            {
                **registry_row,
                "toy_local_times": list(TOY_LOCAL_TIMES),
                "toy_local_operator_errors": local_errors,
                "toy_local_operator_error_slope": local_slope,
                "expected_local_operator_error_slope": order + 1,
                "toy_eigenphase_energy_biases_by_time": branch_biases,
                "toy_eigenphase_energy_bias_slopes": branch_slopes,
                "expected_generic_eigenphase_energy_bias_slope": order,
                "eigenphase_slope_is_nongating": True,
                "toy_global_total_time": TOY_GLOBAL_TOTAL_TIME,
                "toy_global_step_counts": list(TOY_GLOBAL_STEP_COUNTS),
                "toy_global_step_sizes": global_step_sizes,
                "toy_global_operator_errors": global_errors,
                "toy_global_operator_error_slope": global_slope,
                "expected_global_operator_error_slope": order,
                "normalization_and_symmetry_pass": normalization_pass,
                "local_order_pass": local_order_pass,
                "global_order_pass": global_order_pass,
                "formula_eligible_for_pd": bool(
                    normalization_pass and local_order_pass and global_order_pass
                ),
            }
        )
    return {
        "toy_matrices": {"a": _matrix_payload(_TOY_A), "b": _matrix_payload(_TOY_B)},
        "commutator_spectral_norm": float(
            np.linalg.norm(_TOY_A @ _TOY_B - _TOY_B @ _TOY_A, ord=2)
        ),
        "formulae": rows,
        "all_formulae_eligible": all(row["formula_eligible_for_pd"] for row in rows),
    }


def evaluate_nested_vs_global_order() -> dict[str, Any]:
    a1 = np.diag(np.diag(_TOY_A)).astype(np.complex128)
    a2 = _TOY_A - a1
    h_d = a1 + a2
    h_r = _TOY_B
    exact_hamiltonian = h_d + h_r
    constructions = {
        "outer_s2_exact_hd": lambda time_value: _pf_unitary(
            (h_d, h_r), "2nd", time_value
        ),
        "outer_s2_inner_hd_fourth": lambda time_value: (
            _pf_unitary((a1, a2), "4th", time_value / 2.0)
            @ expm(-1j * float(time_value) * h_r)
            @ _pf_unitary((a1, a2), "4th", time_value / 2.0)
        ),
        "global_hd_hr_fourth": lambda time_value: _pf_unitary(
            (h_d, h_r), "4th", time_value
        ),
        "global_three_term_fourth": lambda time_value: _pf_unitary(
            (a1, a2, h_r), "4th", time_value
        ),
    }
    rows: dict[str, dict[str, Any]] = {}
    for name, builder in constructions.items():
        errors = [
            float(
                np.linalg.norm(
                    builder(time_value)
                    - expm(-1j * float(time_value) * exact_hamiltonian),
                    ord=2,
                )
            )
            for time_value in TOY_LOCAL_TIMES
        ]
        rows[name] = {
            "times": list(TOY_LOCAL_TIMES),
            "operator_errors": errors,
            "local_operator_error_slope": _log_slope(TOY_LOCAL_TIMES, errors),
        }
    nested_slope = rows["outer_s2_inner_hd_fourth"]["local_operator_error_slope"]
    global_slope = rows["global_hd_hr_fourth"]["local_operator_error_slope"]
    nested_second_order_pass = abs(nested_slope - 3.0) <= ORDER_SLOPE_TOLERANCE
    global_fourth_order_pass = abs(global_slope - 5.0) <= ORDER_SLOPE_TOLERANCE
    separation_pass = (
        global_slope - nested_slope >= NESTED_GLOBAL_MINIMUM_SLOPE_SEPARATION
    )
    return {
        "inner_commutator_spectral_norm": float(
            np.linalg.norm(a1 @ a2 - a2 @ a1, ord=2)
        ),
        "outer_commutator_spectral_norm": float(
            np.linalg.norm(h_d @ h_r - h_r @ h_d, ord=2)
        ),
        "constructions": rows,
        "nested_hd_only_remains_outer_second_order_pass": bool(
            nested_second_order_pass
        ),
        "global_fourth_order_pass": bool(global_fourth_order_pass),
        "slope_separation_pass": bool(separation_pass),
        "overall_pass": bool(
            nested_second_order_pass and global_fourth_order_pass and separation_pass
        ),
    }


def _integer_allocation(
    coefficients: Sequence[float],
    *,
    total_steps: int,
    policy: str,
) -> tuple[int, ...]:
    count = len(coefficients)
    if count <= 0 or total_steps < count:
        raise ValueError("total_steps must give every occurrence at least one step.")
    remaining = int(total_steps) - count
    if policy == "equal":
        weights = [1.0] * count
    elif policy == "absolute_time_proportional":
        weights = [abs(float(value)) for value in coefficients]
    else:
        raise ValueError(f"Unsupported allocation policy: {policy}")
    weight_sum = math.fsum(weights)
    quotas = [remaining * value / weight_sum for value in weights]
    floors = [int(math.floor(value)) for value in quotas]
    allocation = [1 + value for value in floors]
    leftover = int(total_steps) - sum(allocation)
    order = sorted(
        range(count),
        key=lambda index: (-(quotas[index] - floors[index]), index),
    )
    for index in order[:leftover]:
        allocation[index] += 1
    return tuple(allocation)


def _finite_rte_burden(
    coefficients: Sequence[float],
    *,
    lambda_r: float,
    delta: float,
    policy: str,
) -> dict[str, Any]:
    allocation = _integer_allocation(
        coefficients,
        total_steps=RTE_TOTAL_SHORT_STEPS,
        policy=policy,
    )
    log_normalization = 0.0
    truncation_bound = 0.0
    occurrences: list[dict[str, Any]] = []
    for coefficient, rte_steps in zip(coefficients, allocation, strict=True):
        tau = float(lambda_r) * float(delta) * float(coefficient) / int(rte_steps)
        distribution = finite_rte_distribution(tau, RTE_TAYLOR_ORDER)
        contribution = int(rte_steps) * math.log(
            distribution.exact_finite_distribution
        )
        residual = int(rte_steps) * distribution.step_truncation_residual_bound
        log_normalization += contribution
        truncation_bound += residual
        occurrences.append(
            {
                "tail_coefficient": float(coefficient),
                "rte_steps": int(rte_steps),
                "dimensionless_short_step_time": tau,
                "finite_distribution_normalization": (
                    distribution.exact_finite_distribution
                ),
                "log_normalization_contribution": contribution,
                "truncation_residual_contribution_bound": residual,
            }
        )
    return {
        "allocation_policy": policy,
        "total_short_steps": sum(allocation),
        "allocation": list(allocation),
        "log_normalization": float(log_normalization),
        "attenuation": float(math.exp(-log_normalization)),
        "summed_truncation_residual_bound": float(truncation_bound),
        "occurrences": occurrences,
    }


def evaluate_finite_rte_burden(
    registry_rows: Sequence[Mapping[str, Any]],
    *,
    lambda_r: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for registry_row in registry_rows:
        coefficients = tuple(
            float(value)
            for value in registry_row["tail_coefficients_in_circuit_order"]
        )
        for delta in RTE_AUDIT_DELTAS:
            equal = _finite_rte_burden(
                coefficients,
                lambda_r=lambda_r,
                delta=delta,
                policy="equal",
            )
            proportional = _finite_rte_burden(
                coefficients,
                lambda_r=lambda_r,
                delta=delta,
                policy="absolute_time_proportional",
            )
            gamma = float(registry_row["gamma_r"])
            rows.append(
                {
                    "label": registry_row["label"],
                    "delta": float(delta),
                    "lambda_r": float(lambda_r),
                    "gamma_r": gamma,
                    "continuous_gamma_squared_exponent_diagnostic": float(
                        lambda_r**2 * delta**2 * gamma**2 / RTE_TOTAL_SHORT_STEPS
                    ),
                    "equal_allocation": equal,
                    "absolute_time_proportional_allocation": proportional,
                    "proportional_log_normalization_nonworse": bool(
                        proportional["log_normalization"]
                        <= equal["log_normalization"] + 1.0e-14
                    ),
                }
            )
    return rows


def _dominant_phase_metrics(
    unitary: np.ndarray,
    state: np.ndarray,
    target_energy: float,
    delta: float,
) -> dict[str, Any]:
    pf_eigenvalues, pf_eigenvectors = np.linalg.eig(unitary)
    overlaps = np.abs(pf_eigenvectors.conj().T @ state) ** 2
    target_index = int(np.argmax(overlaps))
    raw_energy = -float(np.angle(pf_eigenvalues[target_index])) / float(delta)
    raw_energy += (2.0 * math.pi / float(delta)) * round(
        (float(target_energy) - raw_energy) * float(delta) / (2.0 * math.pi)
    )
    unitary_defect = float(
        np.linalg.norm(
            unitary.conj().T @ unitary - np.eye(unitary.shape[0]),
            ord=2,
        )
    )
    return {
        "signed_energy_bias_hartree": float(raw_energy - target_energy),
        "absolute_energy_bias_hartree": float(abs(raw_energy - target_energy)),
        "target_branch_weight": float(overlaps[target_index] / np.sum(overlaps)),
        "unitary_defect_spectral_norm": unitary_defect,
    }


def _pareto_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        point = (
            float(row["absolute_energy_bias_hartree"]),
            float(row["gamma_r"]),
            int(row["full_exponential_stage_count"]),
        )
        dominated = False
        for other in rows:
            if other["label"] == row["label"]:
                continue
            other_point = (
                float(other["absolute_energy_bias_hartree"]),
                float(other["gamma_r"]),
                int(other["full_exponential_stage_count"]),
            )
            if all(left <= right for left, right in zip(other_point, point)) and any(
                left < right for left, right in zip(other_point, point)
            ):
                dominated = True
                break
        if not dominated:
            labels.append(str(row["label"]))
    return labels


def _split_decision(
    formula_rows: Sequence[Mapping[str, Any]],
    registry_by_label: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    at_delta: list[dict[str, Any]] = []
    for formula_row in formula_rows:
        label = str(formula_row["label"])
        point = next(
            item
            for item in formula_row["points"]
            if math.isclose(float(item["delta"]), DECISION_DELTA)
        )
        registry = registry_by_label[label]
        at_delta.append(
            {
                "label": label,
                **dict(point),
                "gamma_r": float(registry["gamma_r"]),
                "full_exponential_stage_count": int(
                    registry["full_exponential_stage_count"]
                ),
                "formula_eligible_for_pd": bool(
                    registry["formula_eligible_for_pd"]
                ),
            }
        )
    feasible = [
        row
        for row in at_delta
        if row["formula_eligible_for_pd"]
        and row["absolute_energy_bias_hartree"] <= DECISION_ENERGY_TOLERANCE
        and row["target_branch_weight"] >= MINIMUM_TARGET_WEIGHT
        and row["unitary_defect_spectral_norm"] <= NUMERICAL_ATOL
    ]
    pareto = _pareto_labels(at_delta)
    if not feasible:
        return {
            "decision_delta": DECISION_DELTA,
            "energy_tolerance_hartree": DECISION_ENERGY_TOLERANCE,
            "rows": at_delta,
            "feasible_labels": [],
            "pareto_labels": pareto,
            "energy_only_label": None,
            "tail_aware_label": None,
            "selection_reversal": False,
            "gamma_reduction": None,
            "tail_stage_nonincrease": False,
            "selection_gate_pass": False,
        }
    energy_choice = min(
        feasible,
        key=lambda row: (
            row["absolute_energy_bias_hartree"],
            row["full_exponential_stage_count"],
            row["label"],
        ),
    )
    tail_choice = min(
        feasible,
        key=lambda row: (
            row["gamma_r"],
            row["full_exponential_stage_count"],
            row["absolute_energy_bias_hartree"],
            row["label"],
        ),
    )
    gamma_reduction = 1.0 - float(tail_choice["gamma_r"]) / float(
        energy_choice["gamma_r"]
    )
    stage_nonincrease = bool(
        tail_choice["full_exponential_stage_count"]
        <= energy_choice["full_exponential_stage_count"]
    )
    reversal = energy_choice["label"] != tail_choice["label"]
    selection_pass = bool(
        len(feasible) >= 2
        and reversal
        and gamma_reduction >= MINIMUM_GAMMA_REDUCTION
        and stage_nonincrease
        and energy_choice["label"] in pareto
        and tail_choice["label"] in pareto
    )
    return {
        "decision_delta": DECISION_DELTA,
        "energy_tolerance_hartree": DECISION_ENERGY_TOLERANCE,
        "rows": at_delta,
        "feasible_labels": [str(row["label"]) for row in feasible],
        "pareto_labels": pareto,
        "energy_only_label": str(energy_choice["label"]),
        "tail_aware_label": str(tail_choice["label"]),
        "selection_reversal": bool(reversal),
        "gamma_reduction": float(gamma_reduction),
        "tail_stage_nonincrease": stage_nonincrease,
        "selection_gate_pass": selection_pass,
    }


def _evaluate_h4_split(
    hamiltonian: Any,
    sector: PhysicalSector,
    full_hamiltonian: np.ndarray,
    ground_energy: float,
    ground_state: np.ndarray,
    *,
    ld: int,
    role: str,
    registry_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=1.0e-12,
    )
    h_d = select_df_h_d(hamiltonian, partition)
    dense_h_d = dense_df_operator_in_sector(
        h_d,
        sector,
        matrix_free_backend="python",
    )
    dense_h_r = full_hamiltonian - dense_h_d
    eigensystems = (np.linalg.eigh(dense_h_d), np.linalg.eigh(dense_h_r))
    registry_by_label = {str(row["label"]): row for row in registry_rows}
    formula_rows: list[dict[str, Any]] = []
    for label in FORMULA_LABELS:
        registry = registry_by_label[label]
        order = int(registry["order"])
        points: list[dict[str, Any]] = []
        for delta in DIRECT_DELTAS:
            unitary = _pf_unitary(
                (dense_h_d, dense_h_r),
                label,
                delta,
                eigensystems=eigensystems,
            )
            points.append(
                {
                    "delta": float(delta),
                    **_dominant_phase_metrics(
                        unitary,
                        ground_state,
                        ground_energy,
                        delta,
                    ),
                }
            )
        coefficient_window = COEFFICIENT_WINDOWS[order]
        coefficient_points = [
            point
            for point in points
            if point["delta"] in coefficient_window
            and point["absolute_energy_bias_hartree"] > BIAS_FIT_FLOOR
        ]
        point_coefficients = [
            point["absolute_energy_bias_hartree"] / point["delta"] ** order
            for point in coefficient_points
        ]
        coefficient_slope = (
            _log_slope(
                [point["delta"] for point in coefficient_points],
                [point["absolute_energy_bias_hartree"] for point in coefficient_points],
            )
            if len(coefficient_points) >= 2
            else None
        )
        formula_rows.append(
            {
                "label": label,
                "order": order,
                "points": points,
                "coefficient_window": list(coefficient_window),
                "coefficient_point_count_above_numerical_floor": len(
                    coefficient_points
                ),
                "empirical_point_coefficient_max": (
                    max(point_coefficients) if point_coefficients else None
                ),
                "empirical_bias_log_slope": coefficient_slope,
                "coefficient_is_rigorous_bound": False,
            }
        )
    decision = _split_decision(formula_rows, registry_by_label)
    finite_rte_rows = evaluate_finite_rte_burden(
        registry_rows,
        lambda_r=preparation.exact_rte_lambda_r,
    )
    return {
        "ld": int(ld),
        "role": role,
        "deterministic_block_indices": list(partition.deterministic_block_indices),
        "randomized_block_indices": list(partition.randomized_block_indices),
        "ranking_proxy_lambda_r": float(partition.ranking_proxy_lambda_r),
        "exact_rte_lambda_r": float(preparation.exact_rte_lambda_r),
        "dense_partition_reconstruction_error_frobenius": float(
            np.linalg.norm(full_hamiltonian - dense_h_d - dense_h_r, ord="fro")
        ),
        "formulae": formula_rows,
        "finite_rte_burden": finite_rte_rows,
        "finite_rte_proportional_allocation_all_nonworse": all(
            row["proportional_log_normalization_nonworse"]
            for row in finite_rte_rows
        ),
        "decision": decision,
    }


def expected_task_manifest_body() -> dict[str, Any]:
    tasks = [
        {
            "task_id": f"LD{ld}_{label.replace('(', '_').replace(')', '').replace(' ', '_')}",
            "ld": ld,
            "role": role,
            "formula_label": label,
            "direct_deltas": list(DIRECT_DELTAS),
        }
        for ld, role in (
            (DEVELOPMENT_LD, "development_previously_inspected"),
            (BLIND_HOLDOUT_LD, "blind_holdout_not_inspected_before_freeze"),
        )
        for label in FORMULA_LABELS
    ]
    return {
        "schema_version": EXPECTED_SCHEMA,
        "method": METHOD,
        "configuration": {
            "molecule": "H4 linear chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "n_electrons": N_ELECTRONS,
            "df_rank": 12,
            "development_ld": DEVELOPMENT_LD,
            "blind_holdout_ld": BLIND_HOLDOUT_LD,
            "formula_labels": list(FORMULA_LABELS),
            "direct_deltas": list(DIRECT_DELTAS),
            "coefficient_windows": {
                str(key): list(value) for key, value in COEFFICIENT_WINDOWS.items()
            },
            "toy_local_times": list(TOY_LOCAL_TIMES),
            "toy_global_total_time": TOY_GLOBAL_TOTAL_TIME,
            "toy_global_step_counts": list(TOY_GLOBAL_STEP_COUNTS),
            "finite_rte_audit_deltas": list(RTE_AUDIT_DELTAS),
            "gate_thresholds": dict(GATE_THRESHOLDS),
        },
        "tasks": tasks,
        "task_count": len(tasks),
        "exploration_disclosure": {
            "development_ld3_candidate_grid_inspected_before_freeze": True,
            "blind_ld4_candidate_grid_inspected_before_freeze": False,
            "reason": (
                "LD3 was used to choose a numerically resolved delta window; LD4 is "
                "the frozen transfer holdout for the research decision."
            ),
        },
        "decision_rules": [
            "advance_pd_as_conditional_candidate_pending_signed_time_and_inner_hd_validation",
            "stop_pd_reversal_did_not_transfer",
            "stop_pd_no_randomization_specific_selection_difference",
            "stop_pd_due_formula_or_order_definition_failure",
        ],
        "scope": {
            "two_block_exact_hd_hr_reference": True,
            "finite_rte_normalization_analytic_audit": True,
            "finite_rte_sampled_operator_evaluated": False,
            "negative_time_circuit_oracle_evaluated": False,
            "internal_hd_fragment_error_in_h4_pareto_evaluated": False,
            "compiled_circuit_cost_evaluated": False,
            "rpe_total_cost_evaluated": False,
            "h12_evaluated": False,
            "global_pf_optimality_claimed": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_expected_task_manifest(
    body: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {**dict(body), "provenance": dict(provenance)}
    payload["content_fingerprint"] = fingerprint(payload)
    validate_expected_task_manifest(payload)
    return payload


def validate_expected_task_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("Unexpected P-D expected-task schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-D expected-task fingerprint mismatch.")
    config = payload.get("configuration", {})
    if tuple(config.get("formula_labels", ())) != FORMULA_LABELS:
        raise ValueError("P-D formula registry changed after freeze.")
    if tuple(config.get("direct_deltas", ())) != DIRECT_DELTAS:
        raise ValueError("P-D direct delta grid changed after freeze.")
    if config.get("gate_thresholds") != GATE_THRESHOLDS:
        raise ValueError("P-D gate thresholds changed after freeze.")
    if int(payload.get("task_count", -1)) != 2 * len(FORMULA_LABELS):
        raise ValueError("P-D expected task count changed.")
    disclosure = payload.get("exploration_disclosure", {})
    if disclosure.get("blind_ld4_candidate_grid_inspected_before_freeze") is not False:
        raise ValueError("P-D blind holdout disclosure changed.")


def evaluate_energy_tail_pareto(
    snapshot_path: str | Path,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_expected_task_manifest(expected_manifest)
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(snapshot_path)
    if hamiltonian.n_qubits != 8 or hamiltonian.n_blocks != 12:
        raise ValueError("P-D requires the fixed H4 rank-12 snapshot.")
    sector = PhysicalSector.number_sector(
        n_qubits=hamiltonian.n_qubits,
        n_electrons=N_ELECTRONS,
    )
    full_hamiltonian = dense_df_operator_in_sector(
        hamiltonian,
        sector,
        matrix_free_backend="python",
    )
    exact_eigenvalues, exact_eigenvectors = np.linalg.eigh(full_hamiltonian)
    ground_energy = float(exact_eigenvalues[0])
    ground_state = np.asarray(exact_eigenvectors[:, 0], dtype=np.complex128)

    order_audit = evaluate_formula_order_registry()
    nested_audit = evaluate_nested_vs_global_order()
    registry_rows = order_audit["formulae"]
    development = _evaluate_h4_split(
        hamiltonian,
        sector,
        full_hamiltonian,
        ground_energy,
        ground_state,
        ld=DEVELOPMENT_LD,
        role="development_previously_inspected",
        registry_rows=registry_rows,
    )
    blind = _evaluate_h4_split(
        hamiltonian,
        sector,
        full_hamiltonian,
        ground_energy,
        ground_state,
        ld=BLIND_HOLDOUT_LD,
        role="blind_holdout_not_inspected_before_freeze",
        registry_rows=registry_rows,
    )

    formula_registry_pass = bool(order_audit["all_formulae_eligible"])
    nested_global_pass = bool(nested_audit["overall_pass"])
    finite_rte_pass = bool(
        development["finite_rte_proportional_allocation_all_nonworse"]
        and blind["finite_rte_proportional_allocation_all_nonworse"]
    )
    development_reversal_pass = bool(
        development["decision"]["selection_gate_pass"]
    )
    blind_reversal_pass = bool(blind["decision"]["selection_gate_pass"])
    numerical_pass = all(
        point["unitary_defect_spectral_norm"] <= NUMERICAL_ATOL
        for split in (development, blind)
        for row in split["formulae"]
        for point in row["points"]
    )
    target_weight_pass = all(
        point["target_branch_weight"] >= MINIMUM_TARGET_WEIGHT
        for split in (development, blind)
        for row in split["formulae"]
        for point in row["points"]
    )
    gates = {
        "formula_registry_and_order_pass": formula_registry_pass,
        "nested_vs_global_distinction_pass": nested_global_pass,
        "finite_rte_allocation_audit_pass": finite_rte_pass,
        "development_selection_reversal_pass": development_reversal_pass,
        "blind_selection_reversal_pass": blind_reversal_pass,
        "all_direct_unitaries_numerically_consistent_pass": numerical_pass,
        "all_target_branch_weights_pass": target_weight_pass,
    }
    overall_pass = all(gates.values())
    if not formula_registry_pass or not nested_global_pass:
        status = "stop_pd_due_formula_or_order_definition_failure"
    elif development_reversal_pass and not blind_reversal_pass:
        status = "stop_pd_reversal_did_not_transfer"
    elif not development_reversal_pass:
        status = "stop_pd_no_randomization_specific_selection_difference"
    elif overall_pass:
        status = (
            "advance_pd_as_conditional_candidate_pending_signed_time_and_inner_hd_validation"
        )
    else:
        status = "stop_pd_no_randomization_specific_selection_difference"

    return {
        "schema_version": RESULT_SCHEMA,
        "method": METHOD,
        "expected_task_fingerprint": expected_manifest["content_fingerprint"],
        "configuration": dict(expected_manifest["configuration"]),
        "exploration_disclosure": dict(expected_manifest["exploration_disclosure"]),
        "hamiltonian": {
            "molecule": "H4 linear chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": int(hamiltonian.n_qubits),
            "n_electrons": N_ELECTRONS,
            "df_rank": int(hamiltonian.n_blocks),
            "hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
            "ground_energy_hartree": ground_energy,
            "sector_dimension": int(sector.dimension),
        },
        "formula_registry_and_order": order_audit,
        "nested_vs_global_order": nested_audit,
        "splits": [development, blind],
        "gates": gates,
        "overall_pass": bool(overall_pass),
        "decision": {
            "status": status,
            "thresholds_changed_after_results": False,
            "current_primary_theme": (
                "P-D-conditional-candidate" if overall_pass else "none_confirmed"
            ),
            "next_required_gate": (
                "signed_negative_time_rte_oracle_and_real_internal_hd_error"
                if overall_pass
                else "redefine_research_question_R3_R6_or_R8"
            ),
        },
        "scope": dict(expected_manifest["scope"]),
        "limitations": [
            "LD3 was a disclosed development cell; LD4 is the frozen transfer holdout.",
            "The H4 Pareto calculation uses exact dense H_D and H_R exponentials.",
            "Finite-RTE results here audit normalization and truncation bounds only.",
            "Negative-time sampled operators, controlled phases, compiled circuits, and RPE total cost were not evaluated.",
            "The finite candidate family does not establish global product-formula optimality.",
        ],
    }


def finalize_energy_tail_pareto_artifact(
    body: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    source_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        **dict(body),
        "provenance": dict(provenance),
        "source_evidence": [dict(item) for item in source_evidence],
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_energy_tail_pareto_artifact(payload)
    return payload


def validate_energy_tail_pareto_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("Unexpected P-D result schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-D result fingerprint mismatch.")
    if payload.get("configuration", {}).get("gate_thresholds") != GATE_THRESHOLDS:
        raise ValueError("P-D result does not preserve frozen gate thresholds.")
    if payload.get("decision", {}).get("thresholds_changed_after_results") is not False:
        raise ValueError("P-D result changed thresholds after observing results.")
    scope = payload.get("scope", {})
    forbidden_true = (
        "finite_rte_sampled_operator_evaluated",
        "negative_time_circuit_oracle_evaluated",
        "internal_hd_fragment_error_in_h4_pareto_evaluated",
        "compiled_circuit_cost_evaluated",
        "rpe_total_cost_evaluated",
        "h12_evaluated",
        "global_pf_optimality_claimed",
        "scientific_superiority_claimed",
    )
    if any(scope.get(key) is not False for key in forbidden_true):
        raise ValueError("P-D result overstates scope.")
    if len(payload.get("splits", ())) != 2:
        raise ValueError("P-D result must contain development and blind splits.")


def read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object.")
    return payload


def write_json_nonoverwriting(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)

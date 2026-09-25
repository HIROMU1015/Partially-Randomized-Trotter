"""P-D signed-time and internal-H_D Go/No-Go validation.

This module deliberately keeps the validation small.  It checks the paired
finite-RTE construction on a two-dimensional oracle and restores a
fragment-level second-order approximation inside each outer H_D occurrence
for one fixed H4 snapshot.
"""

from __future__ import annotations

from itertools import product
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from trotterlib import research_direction_energy_tail_pareto as pd


EXPECTED_SCHEMA = "research_direction_pd_realization_expected_tasks_v1"
RESULT_SCHEMA = "research_direction_pd_realization_v1"
METHOD = "pd_signed_time_internal_hd_go_no_go_v1"

DEVELOPMENT_LD = 3
TRANSFER_LD = 4
FRESH_HOLDOUT_LD = 5
LD_ROLES = (
    (DEVELOPMENT_LD, "development_previously_inspected"),
    (TRANSFER_LD, "internal_hd_transfer_outer_exact_previously_inspected"),
    (FRESH_HOLDOUT_LD, "fresh_holdout_not_inspected_before_freeze"),
)
DIAGNOSTIC_DELTAS = (0.2, 0.4)
DECISION_DELTA = 0.4
ENERGY_TOLERANCE = 1.0e-6
MINIMUM_TARGET_WEIGHT = 0.9995
MINIMUM_BURDEN_REDUCTION = 0.20
INTERNAL_HD_SUBSTEPS = 32
RTE_TAYLOR_ORDER = 2
RTE_TOTAL_SHORT_STEPS = 64
D1_SAMPLE_COUNT = 100_000
D1_BASE_SEED = 271_828
D1_OPERATOR_ATOL = 1.0e-12
D1_PHASE_ATOL = 1.0e-12
D1_SAMPLE_ABS_TOL = 5.0e-3
D1_SAMPLE_Z_TOL = 4.0
RECONSTRUCTION_ATOL = 1.0e-12
UNITARY_ATOL = 1.0e-10

GATE_THRESHOLDS = {
    "decision_delta": DECISION_DELTA,
    "diagnostic_deltas": list(DIAGNOSTIC_DELTAS),
    "energy_tolerance_hartree": ENERGY_TOLERANCE,
    "minimum_target_branch_weight": MINIMUM_TARGET_WEIGHT,
    "minimum_tail_burden_reduction": MINIMUM_BURDEN_REDUCTION,
    "internal_hd_substeps_per_outer_occurrence": INTERNAL_HD_SUBSTEPS,
    "finite_rte_taylor_order": RTE_TAYLOR_ORDER,
    "finite_rte_total_short_steps_per_outer_step": RTE_TOTAL_SHORT_STEPS,
    "d1_sample_count": D1_SAMPLE_COUNT,
    "d1_operator_absolute_tolerance": D1_OPERATOR_ATOL,
    "d1_phase_absolute_tolerance_rad": D1_PHASE_ATOL,
    "d1_sample_absolute_tolerance": D1_SAMPLE_ABS_TOL,
    "d1_sample_standardized_tolerance": D1_SAMPLE_Z_TOL,
    "fragment_reconstruction_frobenius_tolerance": RECONSTRUCTION_ATOL,
    "unitary_defect_spectral_tolerance": UNITARY_ATOL,
}

_IDENTITY = np.eye(2, dtype=np.complex128)
_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_D1_IDENTITY_COEFFICIENT = 0.17
_D1_COMPONENT_COEFFICIENTS = (0.53, -0.31, 0.16)
_D1_COMPONENT_MATRICES = (_X, _Z, _Y)


def _matrix_payload(matrix: np.ndarray) -> dict[str, list[list[float]]]:
    value = np.asarray(matrix, dtype=np.complex128)
    return {
        "real": value.real.astype(float).tolist(),
        "imag": value.imag.astype(float).tolist(),
    }


def _largest_remainder_allocation(
    coefficients: Sequence[float], total_steps: int = RTE_TOTAL_SHORT_STEPS
) -> tuple[int, ...]:
    if not coefficients or total_steps < len(coefficients):
        raise ValueError("Every RTE occurrence needs at least one short step.")
    remaining = int(total_steps) - len(coefficients)
    magnitudes = [abs(float(value)) for value in coefficients]
    scale = math.fsum(magnitudes)
    quotas = [remaining * value / scale for value in magnitudes]
    floors = [int(math.floor(value)) for value in quotas]
    allocation = [1 + value for value in floors]
    leftover = int(total_steps) - sum(allocation)
    order = sorted(
        range(len(coefficients)),
        key=lambda index: (-(quotas[index] - floors[index]), index),
    )
    for index in order[:leftover]:
        allocation[index] += 1
    return tuple(allocation)


def signed_time_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in pd.formula_registry():
        coefficients = tuple(
            float(value) for value in row["tail_coefficients_in_circuit_order"]
        )
        allocation = _largest_remainder_allocation(coefficients)
        for occurrence_index, (coefficient, rte_steps) in enumerate(
            zip(coefficients, allocation, strict=True)
        ):
            if coefficient >= 0.0:
                continue
            signed_short_time = DECISION_DELTA * coefficient / int(rte_steps)
            tasks.append(
                {
                    "task_id": f"D1_{row['label']}_{occurrence_index}",
                    "formula_label": str(row["label"]),
                    "tail_occurrence_index": int(occurrence_index),
                    "tail_coefficient": float(coefficient),
                    "rte_steps_for_occurrence": int(rte_steps),
                    "signed_short_time": float(signed_short_time),
                    "positive_control_time": float(abs(signed_short_time)),
                    "sample_count": D1_SAMPLE_COUNT,
                    "seed": D1_BASE_SEED + len(tasks),
                }
            )
    if not tasks:
        raise ValueError("The fixed P-D formula family has no negative tail coefficient.")
    return tasks


def expected_task_manifest_body() -> dict[str, Any]:
    d1_tasks = signed_time_tasks()
    d2_tasks = [
        {
            "task_id": f"D23_LD{ld}_{label.replace('(', '_').replace(')', '')}",
            "ld": int(ld),
            "role": role,
            "formula_label": label,
            "deltas": list(DIAGNOSTIC_DELTAS),
        }
        for ld, role in LD_ROLES
        for label in pd.FORMULA_LABELS
    ]
    return {
        "schema_version": EXPECTED_SCHEMA,
        "method": METHOD,
        "configuration": {
            "molecule": "H4 linear chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "n_electrons": 4,
            "df_rank": 12,
            "formula_labels": list(pd.FORMULA_LABELS),
            "ld_roles": [
                {"ld": int(ld), "role": role} for ld, role in LD_ROLES
            ],
            "gate_thresholds": dict(GATE_THRESHOLDS),
            "tail_allocation_policy": "absolute_time_proportional_largest_remainder",
            "inner_hd_formula": "second_order_symmetric_fixed_df_prefix_order",
        },
        "exploration_disclosure": {
            "ld3_outer_exact_and_internal_hd_seen_before_freeze": True,
            "ld4_outer_exact_seen_before_freeze": True,
            "ld4_internal_hd_seen_before_freeze": False,
            "ld5_outer_exact_or_internal_hd_seen_before_freeze": False,
        },
        "d1_tasks": d1_tasks,
        "d2_d3_tasks": d2_tasks,
        "task_count": len(d1_tasks) + len(d2_tasks),
        "decision_rules": [
            "advance_pd_to_formal_primary_candidate_then_stop_for_research_redesign",
            "stop_pd_signed_time_rte_not_validated",
            "stop_or_narrow_pd_after_internal_hd_error",
            "stop_pd_selection_difference_did_not_transfer",
        ],
        "scope": {
            "signed_time_finite_rte_small_matrix_oracle": True,
            "dense_controlled_semantics_checked": True,
            "fragment_level_internal_hd_error_checked": True,
            "fresh_ld5_holdout_checked": True,
            "qiskit_controlled_circuit_compiled": False,
            "finite_rte_h4_sampled_operator_evaluated": False,
            "full_rpe_total_cost_evaluated": False,
            "h12_evaluated": False,
            "backend_or_noise_evaluated": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_expected_task_manifest(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {**dict(body), "provenance": dict(provenance)}
    payload["content_fingerprint"] = pd.fingerprint(payload)
    validate_expected_task_manifest(payload)
    return payload


def validate_expected_task_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("Unexpected P-D realization expected-task schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != pd.fingerprint(unsigned):
        raise ValueError("P-D realization expected-task fingerprint mismatch.")
    configuration = payload.get("configuration", {})
    if tuple(configuration.get("formula_labels", ())) != pd.FORMULA_LABELS:
        raise ValueError("P-D realization formula family changed after freeze.")
    if configuration.get("gate_thresholds") != GATE_THRESHOLDS:
        raise ValueError("P-D realization thresholds changed after freeze.")
    if list(payload.get("d1_tasks", ())) != signed_time_tasks():
        raise ValueError("P-D realization signed-time tasks changed after freeze.")
    expected_count = len(payload.get("d1_tasks", ())) + len(
        payload.get("d2_d3_tasks", ())
    )
    if int(payload.get("task_count", -1)) != expected_count:
        raise ValueError("P-D realization task count changed after freeze.")
    disclosure = payload.get("exploration_disclosure", {})
    if disclosure.get("ld5_outer_exact_or_internal_hd_seen_before_freeze") is not False:
        raise ValueError("P-D realization fresh-holdout disclosure changed.")


def _d1_component_data() -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[float, ...]]:
    coefficients = np.asarray(_D1_COMPONENT_COEFFICIENTS, dtype=float)
    lambda_value = float(np.sum(np.abs(coefficients)))
    probabilities = tuple(float(value) for value in np.abs(coefficients) / lambda_value)
    signed_components = tuple(
        (1.0 if coefficient >= 0.0 else -1.0) * matrix
        for coefficient, matrix in zip(
            coefficients, _D1_COMPONENT_MATRICES, strict=True
        )
    )
    h_nonidentity = sum(
        (
            float(coefficient) * matrix
            for coefficient, matrix in zip(
                coefficients, _D1_COMPONENT_MATRICES, strict=True
            )
        ),
        np.zeros_like(_IDENTITY),
    )
    return h_nonidentity, signed_components, probabilities


def _paired_event_distribution(
    signed_time: float,
) -> tuple[list[float], list[np.ndarray], float]:
    h_nonidentity, components, component_probabilities = _d1_component_data()
    lambda_value = float(sum(abs(value) for value in _D1_COMPONENT_COEFFICIENTS))
    tau = lambda_value * float(signed_time)
    distribution = pd.finite_rte_distribution(tau, RTE_TAYLOR_ORDER)
    probabilities: list[float] = []
    matrices: list[np.ndarray] = []
    identity_phase = np.exp(-1.0j * _D1_IDENTITY_COEFFICIENT * signed_time)

    for order, order_probability in zip(
        distribution.orders, distribution.order_probabilities, strict=True
    ):
        sequence_length = int(order)
        for indices in product(range(len(components)), repeat=sequence_length + 1):
            sequence_indices = indices[:-1]
            extra_index = indices[-1]
            sequence = _IDENTITY.copy()
            component_probability = 1.0
            for component_index in sequence_indices:
                sequence = components[component_index] @ sequence
                component_probability *= component_probabilities[component_index]
            component_probability *= component_probabilities[extra_index]
            ratio = tau / (sequence_length + 1)
            rotation = (
                _IDENTITY - 1.0j * ratio * components[extra_index]
            ) / math.hypot(1.0, ratio)
            event = (
                identity_phase
                * ((-1.0) ** (sequence_length // 2))
                * rotation
                @ sequence
            )
            probabilities.append(float(order_probability * component_probability))
            matrices.append(np.asarray(event, dtype=np.complex128))

    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1.0e-14):
        raise ValueError("D1 exhaustive event probabilities do not sum to one.")
    return probabilities, matrices, float(distribution.exact_finite_distribution)


def _finite_taylor_oracle(signed_time: float, normalization: float) -> np.ndarray:
    h_nonidentity, _, _ = _d1_component_data()
    value = _IDENTITY.copy()
    power = _IDENTITY.copy()
    for degree in range(1, RTE_TAYLOR_ORDER + 2):
        power = h_nonidentity @ power
        value = value + ((-1.0j * signed_time) ** degree / math.factorial(degree)) * power
    identity_phase = np.exp(-1.0j * _D1_IDENTITY_COEFFICIENT * signed_time)
    return identity_phase * value / float(normalization)


def _controlled(operator: np.ndarray) -> np.ndarray:
    operator = np.asarray(operator, dtype=np.complex128)
    dimension = int(operator.shape[0])
    value = np.zeros((2 * dimension, 2 * dimension), dtype=np.complex128)
    value[:dimension, :dimension] = np.eye(dimension, dtype=np.complex128)
    value[dimension:, dimension:] = operator
    return value


def _max_sample_residual_diagnostics(
    probabilities: np.ndarray,
    matrices: np.ndarray,
    exhaustive_mean: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, float, float, bool]:
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(int(sample_count), probabilities)
    sampled = np.tensordot(counts / float(sample_count), matrices, axes=(0, 0))
    errors = sampled - exhaustive_mean
    max_abs = float(np.max(np.abs(errors)))
    max_z = 0.0
    component_pass = True
    for extractor in (np.real, np.imag):
        values = extractor(matrices)
        mean = extractor(exhaustive_mean)
        variance = np.tensordot(
            probabilities,
            (values - mean[np.newaxis, :, :]) ** 2,
            axes=(0, 0),
        )
        standard_error = np.sqrt(np.maximum(variance, 0.0) / float(sample_count))
        component_error = np.abs(extractor(errors))
        z = np.zeros_like(component_error)
        positive = standard_error > 1.0e-15
        z[positive] = component_error[positive] / standard_error[positive]
        z[~positive] = np.where(component_error[~positive] <= 1.0e-14, 0.0, np.inf)
        max_z = max(max_z, float(np.max(z)))
        component_pass = bool(
            component_pass
            and np.all(
                (component_error <= D1_SAMPLE_ABS_TOL) | (z <= D1_SAMPLE_Z_TOL)
            )
        )
    return sampled, max_abs, max_z, component_pass


def evaluate_signed_time_task(task: Mapping[str, Any]) -> dict[str, Any]:
    signed_time = float(task["signed_short_time"])
    probabilities, events, normalization = _paired_event_distribution(signed_time)
    probability_array = np.asarray(probabilities, dtype=float)
    event_array = np.asarray(events, dtype=np.complex128)
    exhaustive = np.tensordot(probability_array, event_array, axes=(0, 0))
    oracle = _finite_taylor_oracle(signed_time, normalization)

    positive_probabilities, positive_events, positive_normalization = (
        _paired_event_distribution(abs(signed_time))
    )
    positive_mean = np.tensordot(
        np.asarray(positive_probabilities, dtype=float),
        np.asarray(positive_events, dtype=np.complex128),
        axes=(0, 0),
    )
    positive_oracle = _finite_taylor_oracle(abs(signed_time), positive_normalization)

    controlled_events = np.asarray([_controlled(event) for event in events])
    controlled_mean = np.tensordot(
        probability_array, controlled_events, axes=(0, 0)
    )
    controlled_reference = _controlled(exhaustive)
    samplewise_controlled_residual = max(
        float(np.linalg.norm(controlled - _controlled(event), ord=2))
        for controlled, event in zip(controlled_events, events, strict=True)
    )
    sampled, max_sample_abs, max_sample_z, sampled_pass = (
        _max_sample_residual_diagnostics(
            probability_array,
            event_array,
            exhaustive,
            sample_count=int(task["sample_count"]),
            seed=int(task["seed"]),
        )
    )
    phase_expected = -_D1_IDENTITY_COEFFICIENT * signed_time
    phase_observed = float(np.angle(np.exp(-1.0j * _D1_IDENTITY_COEFFICIENT * signed_time)))
    phase_residual = abs(
        float(np.angle(np.exp(1.0j * (phase_observed - phase_expected))))
    )
    ordinary_residual = float(np.linalg.norm(exhaustive - oracle, ord=2))
    positive_residual = float(np.linalg.norm(positive_mean - positive_oracle, ord=2))
    adjoint_residual = float(
        np.linalg.norm(exhaustive - positive_mean.conj().T, ord=2)
    )
    controlled_residual = float(
        np.linalg.norm(controlled_mean - controlled_reference, ord=2)
    )
    probability_pass = bool(
        np.all(probability_array >= 0.0)
        and math.isclose(float(np.sum(probability_array)), 1.0, abs_tol=1.0e-14)
        and normalization > 0.0
    )
    task_pass = bool(
        ordinary_residual <= D1_OPERATOR_ATOL
        and positive_residual <= D1_OPERATOR_ATOL
        and adjoint_residual <= D1_OPERATOR_ATOL
        and samplewise_controlled_residual <= D1_OPERATOR_ATOL
        and controlled_residual <= D1_OPERATOR_ATOL
        and phase_residual <= D1_PHASE_ATOL
        and sampled_pass
        and probability_pass
    )
    return {
        **dict(task),
        "event_count": len(events),
        "finite_distribution_normalization": normalization,
        "minimum_event_probability": float(np.min(probability_array)),
        "probability_and_normalization_pass": probability_pass,
        "ordinary_oracle_residual_spectral_norm": ordinary_residual,
        "positive_oracle_residual_spectral_norm": positive_residual,
        "signed_adjoint_residual_spectral_norm": adjoint_residual,
        "samplewise_controlled_block_residual_spectral_norm": (
            samplewise_controlled_residual
        ),
        "exhaustive_controlled_residual_spectral_norm": controlled_residual,
        "identity_relative_phase_residual_rad": phase_residual,
        "sampled_mean_max_absolute_error": max_sample_abs,
        "sampled_mean_max_absolute_standardized_residual": max_sample_z,
        "sampled_mean_gate_pass": sampled_pass,
        "exhaustive_mean": _matrix_payload(exhaustive),
        "finite_taylor_oracle": _matrix_payload(oracle),
        "sampled_mean": _matrix_payload(sampled),
        "task_pass": task_pass,
    }


def evaluate_d1(expected_manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = [evaluate_signed_time_task(task) for task in expected_manifest["d1_tasks"]]
    return {
        "toy": {
            "identity_coefficient": _D1_IDENTITY_COEFFICIENT,
            "nonidentity_coefficients": list(_D1_COMPONENT_COEFFICIENTS),
            "lambda_nonidentity": float(
                sum(abs(value) for value in _D1_COMPONENT_COEFFICIENTS)
            ),
            "noncommuting_commutator_norm": float(
                np.linalg.norm(_X @ _Z - _Z @ _X, ord=2)
            ),
        },
        "rows": rows,
        "task_count": len(rows),
        "completed_tasks": len(rows),
        "failed_tasks": sum(not row["task_pass"] for row in rows),
        "overall_pass": all(row["task_pass"] for row in rows),
    }


def _dense_hd(
    hamiltonian: Any,
    sector: Any,
    ld: int,
) -> tuple[np.ndarray, Any]:
    partition = pd.split_df_hamiltonian_by_ld(hamiltonian, int(ld))
    h_d = pd.select_df_h_d(hamiltonian, partition)
    dense = pd.dense_df_operator_in_sector(
        h_d,
        sector,
        matrix_free_backend="python",
    )
    return np.asarray(dense, dtype=np.complex128), partition


def _fragment_terms(hamiltonian: Any, sector: Any, ld: int) -> tuple[list[np.ndarray], float]:
    cumulative = [_dense_hd(hamiltonian, sector, index)[0] for index in range(ld + 1)]
    terms = [cumulative[0]] + [
        cumulative[index] - cumulative[index - 1] for index in range(1, ld + 1)
    ]
    reconstruction = sum(terms, np.zeros_like(cumulative[-1]))
    residual = float(np.linalg.norm(reconstruction - cumulative[-1], ord="fro"))
    return terms, residual


def _internal_hd_unitary(
    terms: Sequence[np.ndarray],
    eigensystems: Sequence[tuple[np.ndarray, np.ndarray]],
    signed_time: float,
) -> np.ndarray:
    one_step = pd._pf_unitary(
        terms,
        "2nd",
        float(signed_time) / INTERNAL_HD_SUBSTEPS,
        eigensystems=eigensystems,
    )
    return np.linalg.matrix_power(one_step, INTERNAL_HD_SUBSTEPS)


def _realized_outer_unitary(
    h_d_terms: Sequence[np.ndarray],
    h_r_eigensystem: tuple[np.ndarray, np.ndarray],
    formula_label: str,
    delta: float,
) -> np.ndarray:
    dimension = int(h_d_terms[0].shape[0])
    result = np.eye(dimension, dtype=np.complex128)
    h_d_eigensystems = tuple(np.linalg.eigh(term) for term in h_d_terms)
    h_r_values, h_r_vectors = h_r_eigensystem
    for term_index, weight in pd.iter_pf_steps(2, pd._get_w_list(formula_label)):
        signed_time = float(delta) * float(weight)
        if term_index == 0:
            factor = _internal_hd_unitary(
                h_d_terms,
                h_d_eigensystems,
                signed_time,
            )
        else:
            factor = (
                h_r_vectors
                * np.exp(-1.0j * signed_time * h_r_values)[np.newaxis, :]
            ) @ h_r_vectors.conj().T
        result = factor @ result
    return result


def _pareto_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    output: list[str] = []
    for row in rows:
        point = (
            float(row["absolute_energy_bias_hartree"]),
            float(row["tail_log_normalization"]),
            int(row["realized_stage_proxy"]),
        )
        dominated = False
        for other in rows:
            if other["label"] == row["label"]:
                continue
            other_point = (
                float(other["absolute_energy_bias_hartree"]),
                float(other["tail_log_normalization"]),
                int(other["realized_stage_proxy"]),
            )
            if all(left <= right for left, right in zip(other_point, point)) and any(
                left < right for left, right in zip(other_point, point)
            ):
                dominated = True
                break
        if not dominated:
            output.append(str(row["label"]))
    return output


def _realized_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decision_rows = [
        dict(row) for row in rows if math.isclose(float(row["delta"]), DECISION_DELTA)
    ]
    feasible = [
        row
        for row in decision_rows
        if row["absolute_energy_bias_hartree"] <= ENERGY_TOLERANCE
        and row["target_branch_weight"] >= MINIMUM_TARGET_WEIGHT
        and row["unitary_defect_spectral_norm"] <= UNITARY_ATOL
    ]
    pareto = _pareto_labels(decision_rows)
    if len(feasible) < 2:
        return {
            "decision_delta": DECISION_DELTA,
            "rows": decision_rows,
            "feasible_labels": [str(row["label"]) for row in feasible],
            "pareto_labels": pareto,
            "energy_only_label": None,
            "tail_aware_label": None,
            "selection_reversal": False,
            "tail_burden_reduction": None,
            "tail_stage_nonincrease": False,
            "selection_gate_pass": False,
        }
    energy_choice = min(
        feasible,
        key=lambda row: (
            row["absolute_energy_bias_hartree"],
            row["realized_stage_proxy"],
            row["label"],
        ),
    )
    tail_choice = min(
        feasible,
        key=lambda row: (
            row["tail_log_normalization"],
            row["tail_summed_truncation_residual_bound"],
            row["gamma_r"],
            row["realized_stage_proxy"],
            row["label"],
        ),
    )
    energy_burden = float(energy_choice["tail_log_normalization"])
    tail_burden = float(tail_choice["tail_log_normalization"])
    burden_reduction = (
        1.0 - tail_burden / energy_burden if energy_burden > 0.0 else 0.0
    )
    reversal = energy_choice["label"] != tail_choice["label"]
    stage_nonincrease = bool(
        tail_choice["realized_stage_proxy"]
        <= energy_choice["realized_stage_proxy"]
    )
    gate = bool(
        reversal
        and burden_reduction >= MINIMUM_BURDEN_REDUCTION
        and stage_nonincrease
        and energy_choice["label"] in pareto
        and tail_choice["label"] in pareto
    )
    return {
        "decision_delta": DECISION_DELTA,
        "rows": decision_rows,
        "feasible_labels": [str(row["label"]) for row in feasible],
        "pareto_labels": pareto,
        "energy_only_label": str(energy_choice["label"]),
        "tail_aware_label": str(tail_choice["label"]),
        "selection_reversal": reversal,
        "tail_burden_reduction": float(burden_reduction),
        "tail_stage_nonincrease": stage_nonincrease,
        "selection_gate_pass": gate,
    }


def evaluate_internal_hd_split(
    hamiltonian: Any,
    sector: Any,
    full_hamiltonian: np.ndarray,
    ground_energy: float,
    ground_state: np.ndarray,
    *,
    ld: int,
    role: str,
    registry_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    h_d, _ = _dense_hd(hamiltonian, sector, ld)
    h_r = np.asarray(full_hamiltonian - h_d, dtype=np.complex128)
    h_d_terms, reconstruction_residual = _fragment_terms(hamiltonian, sector, ld)
    exact_eigensystems = (np.linalg.eigh(h_d), np.linalg.eigh(h_r))
    h_r_eigensystem = exact_eigensystems[1]

    exact_reference = pd._evaluate_h4_split(
        hamiltonian,
        sector,
        full_hamiltonian,
        ground_energy,
        ground_state,
        ld=ld,
        role=role,
        registry_rows=registry_rows,
    )
    registry_by_label = {str(row["label"]): row for row in registry_rows}
    inner_stage_count = INTERNAL_HD_SUBSTEPS * (2 * len(h_d_terms) - 1)
    rows: list[dict[str, Any]] = []
    for label in pd.FORMULA_LABELS:
        registry = registry_by_label[label]
        for delta in DIAGNOSTIC_DELTAS:
            exact_outer = pd._pf_unitary(
                (h_d, h_r),
                label,
                delta,
                eigensystems=exact_eigensystems,
            )
            realized = _realized_outer_unitary(
                h_d_terms,
                h_r_eigensystem,
                label,
                delta,
            )
            exact_metrics = pd._dominant_phase_metrics(
                exact_outer, ground_state, ground_energy, delta
            )
            realized_metrics = pd._dominant_phase_metrics(
                realized, ground_state, ground_energy, delta
            )
            burden = pd._finite_rte_burden(
                tuple(
                    float(value)
                    for value in registry["tail_coefficients_in_circuit_order"]
                ),
                lambda_r=float(exact_reference["exact_rte_lambda_r"]),
                delta=delta,
                policy="absolute_time_proportional",
            )
            deterministic_occurrences = int(
                registry["deterministic_occurrence_count"]
            )
            realized_stage_proxy = (
                deterministic_occurrences * inner_stage_count
                + RTE_TOTAL_SHORT_STEPS
            )
            rows.append(
                {
                    "label": label,
                    "delta": float(delta),
                    "gamma_r": float(registry["gamma_r"]),
                    "outer_full_exponential_stage_count": int(
                        registry["full_exponential_stage_count"]
                    ),
                    "deterministic_occurrence_count": deterministic_occurrences,
                    "inner_stages_per_hd_occurrence": inner_stage_count,
                    "realized_stage_proxy": int(realized_stage_proxy),
                    "tail_log_normalization": float(burden["log_normalization"]),
                    "tail_attenuation": float(burden["attenuation"]),
                    "tail_summed_truncation_residual_bound": float(
                        burden["summed_truncation_residual_bound"]
                    ),
                    "exact_hd_metrics": exact_metrics,
                    **realized_metrics,
                    "realized_vs_exact_outer_operator_residual_spectral_norm": float(
                        np.linalg.norm(realized - exact_outer, ord=2)
                    ),
                }
            )
    decision = _realized_decision(rows)
    all_unitary_pass = all(
        row["unitary_defect_spectral_norm"] <= UNITARY_ATOL for row in rows
    )
    all_weight_pass = all(
        row["target_branch_weight"] >= MINIMUM_TARGET_WEIGHT for row in rows
    )
    return {
        "ld": int(ld),
        "role": role,
        "fragment_term_count": len(h_d_terms),
        "fragment_reconstruction_residual_frobenius": reconstruction_residual,
        "fragment_reconstruction_pass": reconstruction_residual
        <= RECONSTRUCTION_ATOL,
        "all_unitary_pass": all_unitary_pass,
        "all_target_weight_pass": all_weight_pass,
        "exact_rte_lambda_r": float(exact_reference["exact_rte_lambda_r"]),
        "rows": rows,
        "decision": decision,
        "split_gate_pass": bool(
            reconstruction_residual <= RECONSTRUCTION_ATOL
            and all_unitary_pass
            and all_weight_pass
            and decision["selection_gate_pass"]
        ),
    }


def evaluate_pd_realization(
    snapshot_path: str | Path,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_expected_task_manifest(expected_manifest)
    d1 = evaluate_d1(expected_manifest)
    hamiltonian = pd.load_connected_cluster_hamiltonian_snapshot(snapshot_path)
    if hamiltonian.n_qubits != 8 or hamiltonian.n_blocks != 12:
        raise ValueError("P-D realization requires the fixed H4 rank-12 snapshot.")
    sector = pd.PhysicalSector.number_sector(
        n_qubits=hamiltonian.n_qubits,
        n_electrons=4,
    )
    full_hamiltonian = pd.dense_df_operator_in_sector(
        hamiltonian,
        sector,
        matrix_free_backend="python",
    )
    full_hamiltonian = np.asarray(full_hamiltonian, dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eigh(full_hamiltonian)
    ground_energy = float(eigenvalues[0])
    ground_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    registry_audit = pd.evaluate_formula_order_registry()
    registry_rows = registry_audit["formulae"]
    splits = [
        evaluate_internal_hd_split(
            hamiltonian,
            sector,
            full_hamiltonian,
            ground_energy,
            ground_state,
            ld=ld,
            role=role,
            registry_rows=registry_rows,
        )
        for ld, role in LD_ROLES
    ]
    d2_pass = bool(d1["overall_pass"] and all(split["split_gate_pass"] for split in splits[:2]))
    d3_pass = bool(d2_pass and splits[2]["split_gate_pass"])
    if not d1["overall_pass"]:
        status = "stop_pd_signed_time_rte_not_validated"
    elif not d2_pass:
        status = "stop_or_narrow_pd_after_internal_hd_error"
    elif not d3_pass:
        status = "stop_pd_selection_difference_did_not_transfer"
    else:
        status = (
            "advance_pd_to_formal_primary_candidate_then_stop_for_research_redesign"
        )
    gates = {
        "d1_signed_time_rte_and_controlled_phase_pass": bool(d1["overall_pass"]),
        "d2_internal_hd_development_and_transfer_pass": d2_pass,
        "d3_fresh_ld5_selection_difference_pass": d3_pass,
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "method": METHOD,
        "expected_task_fingerprint": expected_manifest["content_fingerprint"],
        "configuration": dict(expected_manifest["configuration"]),
        "exploration_disclosure": dict(
            expected_manifest["exploration_disclosure"]
        ),
        "hamiltonian": {
            "molecule": "H4 linear chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": int(hamiltonian.n_qubits),
            "n_electrons": 4,
            "df_rank": int(hamiltonian.n_blocks),
            "hamiltonian_hash": pd.df_hamiltonian_hash(hamiltonian),
            "ground_energy_hartree": ground_energy,
            "sector_dimension": int(sector.dimension),
        },
        "d1_signed_time": d1,
        "d2_d3_internal_hd_splits": splits,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "decision": {
            "status": status,
            "thresholds_changed_after_results": False,
            "stop_after_this_validation_for_research_redesign": True,
            "next_action": (
                "research_rq_novelty_endpoint_and_validation_redesign"
                if all(gates.values())
                else "stop_pd_and_redefine_r3_r6_or_r8"
            ),
        },
        "scope": dict(expected_manifest["scope"]),
        "limitations": [
            "D1 is a dense small-matrix exhaustive and fixed-seed sampling oracle, not a compiled Qiskit circuit benchmark.",
            "D2/D3 restore fragment-level internal H_D product-formula error while keeping each H_R occurrence exact in the energy operator comparison.",
            "The H4 finite-RTE contribution is an analytic normalization and truncation audit; a sampled H4 operator is not evaluated.",
            "The realized-stage quantity is a fixed operation-count proxy, not compiled depth or final total cost.",
            "H12, long RPE, backend noise, global PF optimality, and scientific superiority are not evaluated.",
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
    payload["content_fingerprint"] = pd.fingerprint(payload)
    validate_result(payload)
    return payload


def validate_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("Unexpected P-D realization result schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != pd.fingerprint(unsigned):
        raise ValueError("P-D realization result fingerprint mismatch.")
    if payload.get("configuration", {}).get("gate_thresholds") != GATE_THRESHOLDS:
        raise ValueError("P-D realization result changed frozen thresholds.")
    decision = payload.get("decision", {})
    if decision.get("thresholds_changed_after_results") is not False:
        raise ValueError("P-D realization result changed thresholds after results.")
    if decision.get("stop_after_this_validation_for_research_redesign") is not True:
        raise ValueError("P-D realization must stop for research redesign.")
    scope = payload.get("scope", {})
    forbidden_true = (
        "qiskit_controlled_circuit_compiled",
        "finite_rte_h4_sampled_operator_evaluated",
        "full_rpe_total_cost_evaluated",
        "h12_evaluated",
        "backend_or_noise_evaluated",
        "scientific_superiority_claimed",
    )
    if any(scope.get(key) is not False for key in forbidden_true):
        raise ValueError("P-D realization result overstates scope.")
    splits = payload.get("d2_d3_internal_hd_splits", ())
    if [row.get("ld") for row in splits] != [3, 4, 5]:
        raise ValueError("P-D realization result lost the fixed LD roles.")


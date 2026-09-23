"""WP05-a full controlled-interrogation connection for the WP06-b policy.

The validation in this module pairs the established full-basis implementation
with the fixed ``support_run_le_1`` policy on identical finite-RTE
trajectories.  It compiles complete measurement-bearing Hadamard wrappers, but
does not include state preparation, backend execution, quantum shots, or a
decision-grade long-round cost claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.quantum_info import Statevector

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import DFPartialS2Preparation
from .df_partial_s2_repeated import (
    DFPartialS2RepeatedRequest,
    QiskitDFPartialS2RepeatedCircuitBuilder,
)
from .df_partial_s2_repeated_cost import (
    DFPartialS2RepeatedTrajectoryStream,
    make_exact_df_partial_s2_repeated_trajectory_stream,
    make_monte_carlo_df_partial_s2_repeated_trajectory_stream,
)
from .research_direction_prevalidation import (
    affine_prediction_with_standard_error,
)
from .research_direction_sequence_policy import (
    make_run_threshold_basis_plan,
    register_support_restricted_bases,
)
from .rpe_hadamard_compiled_cost_benchmark import (
    QiskitRPEHadamardBenchmarkCircuitBuilder,
)
from .rpe_hadamard_interrogation import RPEHadamardInterrogationRequest
from .rpe_resource_accounting import RPE_COST_METRICS
from .rte import CompilerSettings, finite_rte_distribution, make_rte_config
from .rte_compiled_cost import transpile_and_measure_cost


SCHEMA_VERSION = "research_direction_full_scope_v1"
METHOD = "wp05a_full_controlled_interrogation_policy_connection_v1"
POLICY_LABEL = "support_run_le_1"
POLICY_RUN_THRESHOLD = 1
CALIBRATION_Q = (1, 2)
HOLDOUT_Q = (4,)
AXES = ("cosine", "sine")
METRICS = tuple(RPE_COST_METRICS)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _derived_seed(master_seed: int, *parts: object) -> int:
    encoded = json.dumps(
        [int(master_seed), *parts],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 1:
        raise ValueError("At least one value is required.")
    variance = 0.0 if array.size == 1 else float(np.var(array, ddof=1))
    return {
        "mean": float(np.mean(array)),
        "standard_error": float(math.sqrt(variance / array.size)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _metric_record(cost: Any) -> dict[str, float]:
    return {metric: float(getattr(cost, metric)) for metric in METRICS}


def _materialize_stream(
    stream: DFPartialS2RepeatedTrajectoryStream,
) -> tuple[DFPartialS2RepeatedRequest, ...]:
    records = tuple(stream.records)
    if len(records) != stream.expected_record_count:
        raise RuntimeError("Trajectory stream count differs from preflight metadata.")
    if any(weight is not None for _request, weight in records):
        raise ValueError("WP05-a randomized streams must be unweighted Monte Carlo.")
    return tuple(request for request, _weight in records)


def apply_run_threshold_policy(
    request: DFPartialS2RepeatedRequest,
    support_definitions: Mapping[Any, Any],
    *,
    maximum_support_run_length: int = POLICY_RUN_THRESHOLD,
    training_fingerprint: str,
) -> DFPartialS2RepeatedRequest:
    """Return the same sampled trajectory with explicit per-step basis plans."""
    if request.preparation.is_deterministic_only:
        raise ValueError("A deterministic request does not accept an RTE basis policy.")
    plans = tuple(
        make_run_threshold_basis_plan(
            occurrence,
            support_definitions,
            maximum_support_run_length=maximum_support_run_length,
            training_fingerprint=training_fingerprint,
        )
        for occurrence in request.rte_occurrences
    )
    return replace(request, rte_basis_plans=plans)


def _finite_inputs(
    preparation: DFPartialS2Preparation,
    *,
    delta_time: float,
    rte_steps: int,
    finite_taylor_order: int,
) -> tuple[Any, Any]:
    distribution = finite_rte_distribution(
        preparation.exact_rte_lambda_r * delta_time / rte_steps,
        finite_taylor_order,
    )
    tolerance = max(
        math.nextafter(
            distribution.step_truncation_residual_bound,
            math.inf,
        ),
        math.ulp(0.0),
    )
    return make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=delta_time,
        rte_steps=rte_steps,
        truncation_tolerance=tolerance,
        finite_taylor_order=finite_taylor_order,
    )


def _statevector_action_max_abs_difference(
    left: Any,
    right: Any,
    *,
    seed: int,
) -> float:
    if left.num_qubits != right.num_qubits:
        raise ValueError("Statevector probes require equal circuit dimensions.")
    rng = np.random.default_rng(seed)
    size = 1 << left.num_qubits
    vector = rng.normal(size=size) + 1j * rng.normal(size=size)
    vector /= np.linalg.norm(vector)
    initial = Statevector(vector)
    left_output = np.asarray(initial.evolve(left).data)
    right_output = np.asarray(initial.evolve(right).data)
    return float(np.max(np.abs(left_output - right_output), initial=0.0))


def _compile_paired_requests(
    requests: Sequence[DFPartialS2RepeatedRequest],
    support_definitions: Mapping[Any, Any],
    compiler: CompilerSettings,
    *,
    training_fingerprint: str,
    maximum_repetition_count: int,
    operator_probe_requested: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    repeated_builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    wrapper_builder = QiskitRPEHadamardBenchmarkCircuitBuilder(
        maximum_repetition_count=maximum_repetition_count
    )
    raw: dict[str, dict[str, list[float]]] = {
        axis: {
            f"{policy}:{metric}": []
            for policy in ("full_basis_shared", POLICY_LABEL)
            for metric in METRICS
        }
        for axis in AXES
    }
    raw_delta: dict[str, dict[str, list[float]]] = {
        axis: {metric: [] for metric in METRICS} for axis in AXES
    }
    plan_fingerprints: list[str] = []
    evolution_fingerprints: dict[str, list[str]] = {
        "full_basis_shared": [],
        POLICY_LABEL: [],
    }
    support_application_counts: list[int] = []
    full_application_counts: list[int] = []
    operator_probe: dict[str, Any] | None = None

    for request in requests:
        selected_request = apply_run_threshold_policy(
            request,
            support_definitions,
            training_fingerprint=training_fingerprint,
        )
        full_evolution = repeated_builder.build(
            request,
            construction_policy="boundary_optimized",
        )
        selected_evolution = repeated_builder.build(
            selected_request,
            construction_policy="boundary_optimized",
        )
        evolution_fingerprints["full_basis_shared"].append(
            full_evolution.circuit_semantics_fingerprint
        )
        evolution_fingerprints[POLICY_LABEL].append(
            selected_evolution.circuit_semantics_fingerprint
        )
        support_count = sum(
            step.rte_support_restricted_application_count
            for step in selected_evolution.step_results
        )
        full_count = sum(
            step.rte_full_basis_application_count
            for step in selected_evolution.step_results
        )
        support_application_counts.append(support_count)
        full_application_counts.append(full_count)
        plan_fingerprints.extend(
            plan.plan_fingerprint
            for plan in selected_request.rte_basis_plans
            if plan is not None
        )

        for axis in AXES:
            policy_costs: dict[str, dict[str, float]] = {}
            for policy, evolution in (
                ("full_basis_shared", full_evolution),
                (POLICY_LABEL, selected_evolution),
            ):
                wrapper = wrapper_builder.build(
                    RPEHadamardInterrogationRequest(
                        evolution=evolution,
                        axis=axis,
                        include_measurement=True,
                    )
                )
                # The wrapper semantics fingerprint already binds every nested
                # evolution fingerprint.  Supplying it here avoids an expensive
                # recursive serialization of deeply nested Qiskit definitions;
                # it does not alter transpilation or the measured metrics.
                cost = _metric_record(
                    transpile_and_measure_cost(
                        wrapper.circuit,
                        compiler,
                        circuit_fingerprint=(
                            wrapper.compiler_independent_fingerprint
                        ),
                        actual_circuit_fingerprint=(
                            wrapper.compiler_independent_fingerprint
                        ),
                    )
                )
                policy_costs[policy] = cost
                for metric, value in cost.items():
                    raw[axis][f"{policy}:{metric}"].append(value)
            for metric in METRICS:
                raw_delta[axis][metric].append(
                    policy_costs[POLICY_LABEL][metric]
                    - policy_costs["full_basis_shared"][metric]
                )

        if (
            operator_probe_requested
            and operator_probe is None
            and support_count > 0
        ):
            cosine_full = wrapper_builder.build(
                RPEHadamardInterrogationRequest(
                    evolution=full_evolution,
                    axis="cosine",
                    include_measurement=False,
                )
            )
            cosine_selected = wrapper_builder.build(
                RPEHadamardInterrogationRequest(
                    evolution=selected_evolution,
                    axis="cosine",
                    include_measurement=False,
                )
            )
            sine_full = wrapper_builder.build(
                RPEHadamardInterrogationRequest(
                    evolution=full_evolution,
                    axis="sine",
                    include_measurement=False,
                )
            )
            sine_selected = wrapper_builder.build(
                RPEHadamardInterrogationRequest(
                    evolution=selected_evolution,
                    axis="sine",
                    include_measurement=False,
                )
            )
            operator_probe = {
                "trajectory_seed": request.trajectory_seed,
                "q_m": request.repetition_count,
                "support_restricted_application_count": support_count,
                "evolution_random_state_max_abs_difference": (
                    _statevector_action_max_abs_difference(
                        full_evolution.circuit,
                        selected_evolution.circuit,
                        seed=2026092251,
                    )
                ),
                "cosine_wrapper_random_state_max_abs_difference": (
                    _statevector_action_max_abs_difference(
                        cosine_full.circuit,
                        cosine_selected.circuit,
                        seed=2026092252,
                    )
                ),
                "sine_wrapper_random_state_max_abs_difference": (
                    _statevector_action_max_abs_difference(
                        sine_full.circuit,
                        sine_selected.circuit,
                        seed=2026092253,
                    )
                ),
                "full_rte_relative_phase": full_evolution.rte_relative_phase,
                "selected_rte_relative_phase": (
                    selected_evolution.rte_relative_phase
                ),
                "cosine_signal_component": cosine_selected.signal_component,
                "sine_signal_component": sine_selected.signal_component,
                "cosine_estimator_definition": (
                    cosine_selected.estimator_definition
                ),
                "sine_estimator_definition": sine_selected.estimator_definition,
                "additional_control_applied": (
                    cosine_selected.additional_control_applied
                    or sine_selected.additional_control_applied
                ),
            }

    axes: dict[str, Any] = {}
    for axis in AXES:
        policies: dict[str, Any] = {}
        for policy in ("full_basis_shared", POLICY_LABEL):
            policies[policy] = {
                metric: _statistics(raw[axis][f"{policy}:{metric}"])
                for metric in METRICS
            }
        axes[axis] = {
            "policies": policies,
            "paired_delta_selected_minus_full": {
                metric: _statistics(raw_delta[axis][metric])
                for metric in METRICS
            },
        }
    return (
        {
            "sample_count": len(requests),
            "axes": axes,
            "support_restricted_application_count": _statistics(
                support_application_counts
            ),
            "full_basis_application_count": _statistics(full_application_counts),
            "basis_plan_fingerprint_digest": hashlib.sha256(
                "".join(plan_fingerprints).encode()
            ).hexdigest(),
            "evolution_semantics_digest": {
                policy: hashlib.sha256("".join(values).encode()).hexdigest()
                for policy, values in evolution_fingerprints.items()
            },
        },
        operator_probe,
    )


def _compile_deterministic_point(
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    *,
    delta_time: float,
    q_m: int,
) -> dict[str, Any]:
    stream = make_exact_df_partial_s2_repeated_trajectory_stream(
        preparation,
        delta_time,
        q_m,
        None,
        None,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        construction_policy="boundary_optimized",
        maximum_trajectories=1,
    )
    records = tuple(stream.records)
    if len(records) != 1 or records[0][1] != 1.0:
        raise RuntimeError("Deterministic endpoint did not produce one exact circuit.")
    evolution = QiskitDFPartialS2RepeatedCircuitBuilder().build(
        records[0][0],
        construction_policy="boundary_optimized",
    )
    wrapper_builder = QiskitRPEHadamardBenchmarkCircuitBuilder(
        maximum_repetition_count=max(HOLDOUT_Q)
    )
    axes: dict[str, Any] = {}
    for axis in AXES:
        wrapper = wrapper_builder.build(
            RPEHadamardInterrogationRequest(
                evolution=evolution,
                axis=axis,
                include_measurement=True,
            )
        )
        axes[axis] = {
            metric: {
                "mean": value,
                "standard_error": 0.0,
                "minimum": value,
                "maximum": value,
            }
            for metric, value in _metric_record(
                transpile_and_measure_cost(
                    wrapper.circuit,
                    compiler,
                    circuit_fingerprint=wrapper.compiler_independent_fingerprint,
                    actual_circuit_fingerprint=(
                        wrapper.compiler_independent_fingerprint
                    ),
                )
            ).items()
        }
    return {"sample_count": 1, "axes": axes}


def _fit_affine_models(
    points: Mapping[int, Mapping[str, Any]],
    *,
    policy: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis in AXES:
        result[axis] = {}
        for metric in METRICS:
            if policy is None:
                q1 = points[1]["axes"][axis][metric]
                q2 = points[2]["axes"][axis][metric]
                q4 = points[4]["axes"][axis][metric]
            else:
                q1 = points[1]["axes"][axis]["policies"][policy][metric]
                q2 = points[2]["axes"][axis]["policies"][policy][metric]
                q4 = points[4]["axes"][axis]["policies"][policy][metric]
            prediction, prediction_se = affine_prediction_with_standard_error(
                q_m=4,
                q1_mean=float(q1["mean"]),
                q2_mean=float(q2["mean"]),
                q1_standard_error=float(q1["standard_error"]),
                q2_standard_error=float(q2["standard_error"]),
            )
            actual = float(q4["mean"])
            slope = float(q2["mean"]) - float(q1["mean"])
            intercept = 2.0 * float(q1["mean"]) - float(q2["mean"])
            result[axis][metric] = {
                "formula": "slope*q_m+intercept",
                "slope": slope,
                "intercept": intercept,
                "q4_prediction": prediction,
                "q4_prediction_standard_error": prediction_se,
                "q4_direct_mean": actual,
                "q4_absolute_relative_error": (
                    0.0 if actual == 0.0 else abs(prediction - actual) / actual
                ),
            }
    return result


def _bridge_diagnostics(
    points: Mapping[int, Mapping[str, Any]],
    wp06b: Mapping[str, Any],
    *,
    rte_steps: int,
) -> dict[str, Any]:
    central = wp06b["proxy_recalibration"]["by_rte_steps"][str(rte_steps)][
        "metrics"
    ]
    rows: dict[str, Any] = {}
    normalized_residuals: list[float] = []
    for q_m, point in points.items():
        axes: dict[str, Any] = {}
        for axis in AXES:
            metrics: dict[str, Any] = {}
            for metric in METRICS:
                actual = float(
                    point["axes"][axis]["paired_delta_selected_minus_full"][
                        metric
                    ]["mean"]
                )
                predicted = q_m * float(
                    central[metric]["central_occurrence_paired_delta"]
                )
                baseline = float(
                    point["axes"][axis]["policies"]["full_basis_shared"][
                        metric
                    ]["mean"]
                )
                residual = actual - predicted
                normalized = (
                    0.0 if baseline == 0.0 else abs(residual) / baseline
                )
                normalized_residuals.append(normalized)
                metrics[metric] = {
                    "central_additive_prediction": predicted,
                    "direct_full_wrapper_paired_delta": actual,
                    "wrapper_boundary_residual": residual,
                    "absolute_residual_over_full_wrapper": normalized,
                }
            axes[axis] = metrics
        rows[str(q_m)] = {"q_m": q_m, "axes": axes}
    return {
        "normalization": "absolute_residual_divided_by_full_wrapper_mean",
        "points": rows,
        "maximum_absolute_residual_over_full_wrapper_all_metrics": max(
            normalized_residuals
        ),
        "maximum_rz_absolute_residual_over_full_wrapper": max(
            float(rows[str(q)]["axes"][axis]["rz_count"][
                "absolute_residual_over_full_wrapper"
            ])
            for q in points
            for axis in AXES
        ),
    }


def _predict(model: Mapping[str, Any], q_m: int) -> tuple[float, float]:
    value = float(model["slope"]) * q_m + float(model["intercept"])
    # Reconstruct the two calibration SEs from the stored q=4 propagated SE is
    # impossible, so ranking callers supply models augmented below.
    se = math.hypot(
        (2.0 - q_m) * float(model["q1_standard_error"]),
        (q_m - 1.0) * float(model["q2_standard_error"]),
    )
    return value, se


def _augment_model_errors(
    models: dict[str, Any],
    points: Mapping[int, Mapping[str, Any]],
    *,
    policy: str | None,
) -> None:
    for axis in AXES:
        for metric in METRICS:
            if policy is None:
                q1 = points[1]["axes"][axis][metric]
                q2 = points[2]["axes"][axis][metric]
            else:
                q1 = points[1]["axes"][axis]["policies"][policy][metric]
                q2 = points[2]["axes"][axis]["policies"][policy][metric]
            models[axis][metric]["q1_standard_error"] = float(
                q1["standard_error"]
            )
            models[axis][metric]["q2_standard_error"] = float(
                q2["standard_error"]
            )


def _fixed_wp04_ranking_bridge(
    wp04: Mapping[str, Any],
    selected_models: Mapping[int, Mapping[str, Any]],
    full_models: Mapping[int, Mapping[str, Any]],
    deterministic_models: Mapping[str, Any],
) -> dict[str, Any]:
    scenarios = {row["scenario_id"]: row for row in wp04["scenarios"]}
    ids = wp04["full_setting"]["scenario_ids"]
    ld3 = scenarios[ids["3"]]
    ld12 = scenarios[ids["12"]]

    def total(
        scenario: Mapping[str, Any],
        model_provider: Any,
    ) -> dict[str, Any]:
        total_value = 0.0
        conservative_se = 0.0
        rows = []
        for round_row in scenario["rounds"]:
            q_m = int(round_row["q_m"])
            model_set = model_provider(round_row)
            round_value = 0.0
            round_se = 0.0
            axes = {}
            for axis in AXES:
                prediction, se = _predict(model_set[axis]["rz_count"], q_m)
                if prediction < 0.0:
                    raise ValueError("Affine full-scope RZ prediction became negative.")
                shots = int(round_row["axes"][axis]["shots"])
                round_value += shots * prediction
                round_se += shots * se
                axes[axis] = {
                    "shots": shots,
                    "predicted_rz_count_per_interrogation": prediction,
                    "propagated_calibration_standard_error": se,
                }
            total_value += round_value
            conservative_se += round_se
            rows.append(
                {
                    "round_index": int(round_row["round_index"]),
                    "q_m": q_m,
                    "r_m": int(round_row["r_m"]),
                    "axes": axes,
                    "round_rz_point_estimate": round_value,
                }
            )
        half_width = 0.05 * total_value + 1.96 * conservative_se
        return {
            "total_compiled_rz_point_estimate": total_value,
            "conservative_propagated_calibration_standard_error": (
                conservative_se
            ),
            "local_5_percent_plus_calibration_interval": [
                total_value - half_width,
                total_value + half_width,
            ],
            "rounds": rows,
        }

    selected = total(
        ld3,
        lambda row: selected_models[int(row["r_m"])],
    )
    full = total(
        ld3,
        lambda row: full_models[int(row["r_m"])],
    )
    deterministic = total(ld12, lambda _row: deterministic_models)
    left = selected["local_5_percent_plus_calibration_interval"]
    right = deterministic["local_5_percent_plus_calibration_interval"]
    overlap = max(left[0], right[0]) <= min(left[1], right[1])
    old_ld3 = float(ld3["total_compiled_rz_point_estimate"])
    old_ld12 = float(ld12["total_compiled_rz_point_estimate"])
    return {
        "scope": (
            "fixed_WP04_rounds_shots_alpha_with_q1_q2_full_wrapper_affine_"
            "models;_q_greater_than_4_is_unvalidated_extrapolation"
        ),
        "old_wp04": {"ld3": old_ld3, "ld12": old_ld12},
        "direct_full_basis_recalibration_ld3": full,
        "selected_policy_ld3": selected,
        "direct_deterministic_ld12": deterministic,
        "selected_policy_ld3_relative_to_old_wp04": (
            selected["total_compiled_rz_point_estimate"] / old_ld3 - 1.0
        ),
        "selected_point_preference": (
            "L_D=3"
            if selected["total_compiled_rz_point_estimate"]
            < deterministic["total_compiled_rz_point_estimate"]
            else "L_D=12"
        ),
        "selected_vs_deterministic_point_ratio": (
            selected["total_compiled_rz_point_estimate"]
            / deterministic["total_compiled_rz_point_estimate"]
        ),
        "local_intervals_overlap": overlap,
        "alpha_reoptimized": False,
        "shot_counts_reoptimized": False,
        "q_greater_than_4_directly_transpiled": False,
        "decision_grade": False,
    }


def evaluate_wp05a_full_scope(
    hamiltonian: DFHamiltonian,
    ld3_preparation: DFPartialS2Preparation,
    ld12_preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    wp06b: Mapping[str, Any],
    wp04: Mapping[str, Any],
    *,
    delta_time: float = 0.02,
    finite_taylor_order: int = 2,
    schedule_rte_steps: Sequence[int] = (1, 2, 4, 8, 16, 32),
    sample_count: int = 8,
    master_seed: int = 2026092205,
    accuracy_threshold: float = 0.05,
    equivalence_atol: float = 1e-10,
) -> dict[str, Any]:
    """Compile paired full wrappers and assess the WP06-b additive bridge."""
    if ld3_preparation.ld != 3 or ld12_preparation.ld != 12:
        raise ValueError("WP05-a requires the fixed L_D=3 and L_D=12 candidates.")
    decision = wp06b["decision"]
    if (
        decision["selected_policy"] != POLICY_LABEL
        or int(decision["selected_maximum_support_run_length"])
        != POLICY_RUN_THRESHOLD
        or not decision["selected_policy_approved_for_WP05_validation_path"]
    ):
        raise ValueError("WP06-b did not approve the fixed WP05-a policy input.")
    if wp06b["physical_instance"]["hamiltonian_hash"] != (
        ld3_preparation.hamiltonian_hash
    ):
        raise ValueError("WP06-b and WP05-a Hamiltonian hashes differ.")
    training_fingerprint = str(
        wp06b["training_selection"]["training_fingerprint"]
    )
    support_definitions, proof_records = register_support_restricted_bases(
        hamiltonian,
        ld3_preparation,
    )
    q_values = (*CALIBRATION_Q, *HOLDOUT_Q)
    randomized_results: dict[str, Any] = {}
    operator_probe: dict[str, Any] | None = None
    selected_models: dict[int, Any] = {}
    full_models: dict[int, Any] = {}
    bridge_results: dict[str, Any] = {}

    for rte_steps in schedule_rte_steps:
        config, distribution = _finite_inputs(
            ld3_preparation,
            delta_time=delta_time,
            rte_steps=int(rte_steps),
            finite_taylor_order=finite_taylor_order,
        )
        points: dict[int, Any] = {}
        for q_m in q_values:
            seed = _derived_seed(master_seed, "ld3", int(rte_steps), q_m)
            stream = make_monte_carlo_df_partial_s2_repeated_trajectory_stream(
                ld3_preparation,
                delta_time,
                q_m,
                config,
                distribution,
                sample_count=sample_count,
                seed=seed,
                maximum_samples=sample_count,
                controlled=True,
                ancilla_qubit=ld3_preparation.num_system_qubits,
                construction_policy="boundary_optimized",
            )
            compiled, probe = _compile_paired_requests(
                _materialize_stream(stream),
                support_definitions,
                compiler,
                training_fingerprint=training_fingerprint,
                maximum_repetition_count=max(q_values),
                operator_probe_requested=(operator_probe is None and q_m == 1),
            )
            compiled.update(
                {
                    "q_m": q_m,
                    "partition": (
                        "calibration" if q_m in CALIBRATION_Q else "holdout"
                    ),
                    "master_seed": seed,
                }
            )
            points[q_m] = compiled
            if probe is not None and operator_probe is None:
                operator_probe = {
                    "rte_steps": int(rte_steps),
                    **probe,
                }
        selected = _fit_affine_models(points, policy=POLICY_LABEL)
        full = _fit_affine_models(points, policy="full_basis_shared")
        _augment_model_errors(selected, points, policy=POLICY_LABEL)
        _augment_model_errors(full, points, policy="full_basis_shared")
        bridge = _bridge_diagnostics(points, wp06b, rte_steps=int(rte_steps))
        randomized_results[str(rte_steps)] = {
            "rte_steps": int(rte_steps),
            "points": {str(q): points[q] for q in q_values},
            "affine_models": {
                "full_basis_shared": full,
                POLICY_LABEL: selected,
            },
            "additive_bridge": bridge,
        }
        selected_models[int(rte_steps)] = selected
        full_models[int(rte_steps)] = full
        bridge_results[str(rte_steps)] = bridge

    deterministic_points = {
        q_m: _compile_deterministic_point(
            ld12_preparation,
            compiler,
            delta_time=delta_time,
            q_m=q_m,
        )
        for q_m in q_values
    }
    deterministic_models = _fit_affine_models(
        deterministic_points,
        policy=None,
    )
    _augment_model_errors(
        deterministic_models,
        deterministic_points,
        policy=None,
    )

    ranking = _fixed_wp04_ranking_bridge(
        wp04,
        selected_models,
        full_models,
        deterministic_models,
    )
    selected_q4_rz_errors = [
        float(
            randomized_results[str(r)]["affine_models"][POLICY_LABEL][axis][
                "rz_count"
            ]["q4_absolute_relative_error"]
        )
        for r in schedule_rte_steps
        for axis in AXES
    ]
    selected_q4_all_errors = [
        float(
            randomized_results[str(r)]["affine_models"][POLICY_LABEL][axis][
                metric
            ]["q4_absolute_relative_error"]
        )
        for r in schedule_rte_steps
        for axis in AXES
        for metric in METRICS
    ]
    full_q4_rz_errors = [
        float(
            randomized_results[str(r)]["affine_models"]["full_basis_shared"][
                axis
            ]["rz_count"]["q4_absolute_relative_error"]
        )
        for r in schedule_rte_steps
        for axis in AXES
    ]
    bridge_rz_residuals = [
        float(bridge_results[str(r)]["maximum_rz_absolute_residual_over_full_wrapper"])
        for r in schedule_rte_steps
    ]
    proof_residual = max(
        float(row["preserved_columns_max_abs_residual"])
        for row in proof_records
    )
    statevector_residual = (
        math.inf
        if operator_probe is None
        else max(
            float(operator_probe["evolution_random_state_max_abs_difference"]),
            float(operator_probe["cosine_wrapper_random_state_max_abs_difference"]),
            float(operator_probe["sine_wrapper_random_state_max_abs_difference"]),
        )
    )
    phase_residual = (
        math.inf
        if operator_probe is None
        else abs(
            float(operator_probe["full_rte_relative_phase"])
            - float(operator_probe["selected_rte_relative_phase"])
        )
    )
    checks = {
        "wp06b_policy_propagates_through_complete_partial_s2": all(
            int(
                randomized_results[str(r)]["points"][str(q)][
                    "support_restricted_application_count"
                ]["maximum"]
            )
            > 0
            for r in schedule_rte_steps
            for q in q_values
        ),
        "support_basis_certificates_pass": proof_residual <= equivalence_atol,
        "controlled_evolution_and_wrapper_statevector_probe_passes": (
            statevector_residual <= equivalence_atol
        ),
        "relative_ancilla_phase_is_preserved": phase_residual <= equivalence_atol,
        "cosine_and_sine_semantics_are_explicit": (
            operator_probe is not None
            and operator_probe["cosine_signal_component"] == "real"
            and operator_probe["sine_signal_component"] == "imaginary"
            and not operator_probe["additional_control_applied"]
        ),
        "additive_bridge_rz_residual_within_5_percent_of_full_wrapper": max(
            bridge_rz_residuals
        )
        <= accuracy_threshold,
        "selected_policy_q4_rz_holdout_within_5_percent": max(
            selected_q4_rz_errors
        )
        <= accuracy_threshold,
        "deterministic_q4_rz_holdout_within_5_percent": max(
            float(
                deterministic_models[axis]["rz_count"][
                    "q4_absolute_relative_error"
                ]
            )
            for axis in AXES
        )
        <= accuracy_threshold,
            "full_wrapper_scope_excludes_state_preparation_and_backend_execution": True,
        "long_q_ranking_is_labeled_non_decision_grade": not ranking[
            "decision_grade"
        ],
        "final_total_cost_evaluation_not_claimed": True,
    }
    overall_pass = all(checks.values())
    next_action = (
        "WP05b_extend_selected_policy_to_q8_and_delta_0p01"
        if overall_pass
        else "refine_full_wrapper_boundary_model_before_scope_extension"
    )
    return {
        "configuration": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": len(hamiltonian.lambdas),
            "candidate_ld_values": [3, 12],
            "delta_time": delta_time,
            "finite_taylor_order": finite_taylor_order,
            "schedule_rte_steps": [int(value) for value in schedule_rte_steps],
            "calibration_q": list(CALIBRATION_Q),
            "holdout_q": list(HOLDOUT_Q),
            "sample_count_per_randomized_point": sample_count,
            "master_seed": master_seed,
            "accuracy_threshold": accuracy_threshold,
            "equivalence_atol": equivalence_atol,
            "compiler": {
                "basis_gates": list(compiler.basis_gates),
                "backend_name": compiler.backend_name,
                "coupling_map": compiler.coupling_map,
                "optimization_level": compiler.optimization_level,
                "layout_method": compiler.layout_method,
                "routing_method": compiler.routing_method,
                "transpiler_seed": compiler.transpiler_seed,
                "qiskit_version": compiler.qiskit_version,
            },
        },
        "physical_instance": {
            "hamiltonian_hash": ld3_preparation.hamiltonian_hash,
            "ld3_partition_hash": ld3_preparation.partition_hash,
            "ld3_preparation_hash": ld3_preparation.preparation_hash,
            "ld3_tail_hash": ld3_preparation.tail_extraction.tail_hash,
            "ld12_partition_hash": ld12_preparation.partition_hash,
            "ld12_preparation_hash": ld12_preparation.preparation_hash,
            "ld12_tail_hash": ld12_preparation.tail_extraction.tail_hash,
        },
        "policy_input": {
            "policy": POLICY_LABEL,
            "maximum_support_run_length": POLICY_RUN_THRESHOLD,
            "wp06b_training_fingerprint": training_fingerprint,
            "production_default_changed": False,
            "support_basis_definition_count": len(support_definitions),
            "maximum_preserved_columns_residual": proof_residual,
        },
        "randomized_ld3": randomized_results,
        "deterministic_ld12": {
            "points": {str(q): deterministic_points[q] for q in q_values},
            "affine_models": deterministic_models,
        },
        "operator_and_axis_semantics_probe": operator_probe,
        "holdout_summary": {
            "maximum_selected_policy_q4_rz_relative_error": max(
                selected_q4_rz_errors
            ),
            "maximum_selected_policy_q4_all_metric_relative_error": max(
                selected_q4_all_errors
            ),
            "maximum_full_basis_q4_rz_relative_error": max(full_q4_rz_errors),
            "maximum_deterministic_q4_rz_relative_error": max(
                float(
                    deterministic_models[axis]["rz_count"][
                        "q4_absolute_relative_error"
                    ]
                )
                for axis in AXES
            ),
        },
        "additive_bridge_summary": {
            "maximum_rz_absolute_residual_over_full_wrapper": max(
                bridge_rz_residuals
            ),
            "by_rte_steps": bridge_results,
            "wrapper_boundary_terms_observed": any(
                float(
                    bridge_results[str(r)]["points"][str(q)]["axes"][axis][
                        "rz_count"
                    ]["wrapper_boundary_residual"]
                )
                != 0.0
                for r in schedule_rte_steps
                for q in q_values
                for axis in AXES
            ),
        },
        "fixed_wp04_ranking_sensitivity": ranking,
        "decision": {
            "status": (
                "WP05a_full_scope_connection_passed"
                if overall_pass
                else "WP05a_requires_boundary_model_refinement"
            ),
            "selected_policy_retained": overall_pass,
            "point_preference_in_nondecision_long_q_bridge": ranking[
                "selected_point_preference"
            ],
            "local_intervals_overlap": ranking["local_intervals_overlap"],
            "next_action": next_action,
        },
        "scope": {
            "complete_controlled_partial_s2_retranspiled": True,
            "complete_measurement_bearing_hadamard_wrapper_retranspiled": True,
            "transpile_cost_identity_uses_compiler_independent_semantics_fingerprint": True,
            "cosine_and_sine_axes_included": True,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "quantum_shots_executed": 0,
            "q4_is_independent_holdout": True,
            "q_greater_than_4_directly_transpiled": False,
            "alpha_reoptimized": False,
            "shot_counts_reoptimized": False,
            "decision_grade": False,
            "final_total_cost_evaluation_performed": False,
        },
        "limitations": [
            "The direct full-wrapper validation is limited to one H4 rank-12 snapshot, delta=0.02, q=1,2,4, and one topology-free Qiskit compiler context.",
            "The fixed WP04 ranking sensitivity extrapolates q=1,2 affine fits beyond the directly validated q=4 holdout and is not decision-grade.",
            "WP04 rounds, shots, alpha allocation, and statistical model are held fixed; they are not reoptimized here.",
            "No state preparation, backend, noise, quantum shots, final total cost, or scientific superiority is evaluated.",
        ],
        "checks": checks,
        "overall_pass": overall_pass,
        "summary": {
            "status": (
                "WP05a_full_scope_connection_passed"
                if overall_pass
                else "WP05a_requires_boundary_model_refinement"
            ),
            "maximum_selected_policy_q4_rz_relative_error": max(
                selected_q4_rz_errors
            ),
            "maximum_additive_bridge_rz_residual_over_full_wrapper": max(
                bridge_rz_residuals
            ),
            "maximum_random_state_action_residual": statevector_residual,
            "point_preference_in_nondecision_long_q_bridge": ranking[
                "selected_point_preference"
            ],
            "local_intervals_overlap": ranking["local_intervals_overlap"],
            "next_action": next_action,
        },
    }


def finalize_wp05a_artifact(
    body: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP05-a",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp05a_artifact(payload)
    return payload


def validate_wp05a_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WP05-a schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "WP05-a":
        raise ValueError("Unsupported WP05-a method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP05-a content_fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP05-a overall status does not match its checks.")
    scope = payload.get("scope", {})
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("WP05-a cannot claim a final total-cost evaluation.")
    if scope.get("decision_grade") is not False:
        raise ValueError("WP05-a must remain non-decision-grade.")
    if payload.get("policy_input", {}).get("production_default_changed") is not False:
        raise ValueError("WP05-a must not silently change the production default.")


def write_wp05a_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_wp05a_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

"""WP05-b q=8 and delta=0.01 full controlled-scope extension."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import DFPartialS2Preparation
from .df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder
from .df_partial_s2_repeated_cost import (
    make_exact_df_partial_s2_repeated_trajectory_stream,
    make_monte_carlo_df_partial_s2_repeated_trajectory_stream,
)
from .research_direction_full_scope import (
    AXES,
    METRICS,
    POLICY_LABEL,
    POLICY_RUN_THRESHOLD,
    _compile_paired_requests,
    _derived_seed,
    _finite_inputs,
    _materialize_stream,
    _metric_record,
    fingerprint,
    validate_wp05a_artifact,
)
from .research_direction_prevalidation import affine_prediction_with_standard_error
from .research_direction_sequence_policy import register_support_restricted_bases
from .rpe_hadamard_compiled_cost_benchmark import (
    QiskitRPEHadamardBenchmarkCircuitBuilder,
)
from .rpe_hadamard_interrogation import RPEHadamardInterrogationRequest
from .rte import CompilerSettings
from .rte_compiled_cost import transpile_and_measure_cost


SCHEMA_VERSION = "research_direction_full_scope_extension_v1"
METHOD = "wp05b_q8_and_delta_0p01_full_scope_extension_v1"
REFERENCE_DELTA = 0.02
COMPARISON_DELTA = 0.01
CALIBRATION_Q = (1, 2)
HOLDOUT_Q = (4, 8)


def _point_metric(
    point: Mapping[str, Any],
    *,
    axis: str,
    metric: str,
    policy: str | None,
) -> Mapping[str, Any]:
    if policy is None:
        return point["axes"][axis][metric]
    return point["axes"][axis]["policies"][policy][metric]


def fit_affine_holdouts(
    points: Mapping[int, Mapping[str, Any]],
    *,
    policy: str | None,
    holdout_q: Sequence[int] = HOLDOUT_Q,
) -> dict[str, Any]:
    """Fit q=1,2 and score direct q=4 and q=8 holdouts."""
    if 1 not in points or 2 not in points:
        raise ValueError("Affine calibration requires direct q=1 and q=2 points.")
    result: dict[str, Any] = {}
    for axis in AXES:
        result[axis] = {}
        for metric in METRICS:
            q1 = _point_metric(points[1], axis=axis, metric=metric, policy=policy)
            q2 = _point_metric(points[2], axis=axis, metric=metric, policy=policy)
            holdouts: dict[str, Any] = {}
            for q_m in holdout_q:
                if int(q_m) not in points:
                    raise ValueError(f"Missing direct q={q_m} holdout point.")
                direct = _point_metric(
                    points[int(q_m)], axis=axis, metric=metric, policy=policy
                )
                prediction, prediction_se = affine_prediction_with_standard_error(
                    q_m=int(q_m),
                    q1_mean=float(q1["mean"]),
                    q2_mean=float(q2["mean"]),
                    q1_standard_error=float(q1["standard_error"]),
                    q2_standard_error=float(q2["standard_error"]),
                )
                actual = float(direct["mean"])
                holdouts[str(q_m)] = {
                    "q_m": int(q_m),
                    "prediction": prediction,
                    "prediction_standard_error": prediction_se,
                    "direct_mean": actual,
                    "absolute_relative_error": (
                        0.0 if actual == 0.0 else abs(prediction - actual) / actual
                    ),
                }
            result[axis][metric] = {
                "formula": "slope*q_m+intercept",
                "slope": float(q2["mean"]) - float(q1["mean"]),
                "intercept": 2.0 * float(q1["mean"]) - float(q2["mean"]),
                "q1_standard_error": float(q1["standard_error"]),
                "q2_standard_error": float(q2["standard_error"]),
                "holdouts": holdouts,
            }
    return result


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
        records[0][0], construction_policy="boundary_optimized"
    )
    wrapper_builder = QiskitRPEHadamardBenchmarkCircuitBuilder(
        maximum_repetition_count=8
    )
    axes: dict[str, Any] = {}
    for axis in AXES:
        wrapper = wrapper_builder.build(
            RPEHadamardInterrogationRequest(
                evolution=evolution, axis=axis, include_measurement=True
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
                    actual_circuit_fingerprint=wrapper.compiler_independent_fingerprint,
                )
            ).items()
        }
    return {"sample_count": 1, "q_m": q_m, "axes": axes}


def _compile_randomized_delta(
    preparation: DFPartialS2Preparation,
    support_definitions: Mapping[Any, Any],
    compiler: CompilerSettings,
    *,
    delta_time: float,
    q_values: Sequence[int],
    schedule_rte_steps: Sequence[int],
    finite_taylor_order: int,
    sample_count: int,
    master_seed: int,
    training_fingerprint: str,
    probe: bool,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    results: dict[str, Any] = {}
    operator_probe: dict[str, Any] | None = None
    for rte_steps in schedule_rte_steps:
        config, distribution = _finite_inputs(
            preparation,
            delta_time=delta_time,
            rte_steps=int(rte_steps),
            finite_taylor_order=finite_taylor_order,
        )
        points: dict[str, Any] = {}
        for q_m in q_values:
            if progress is not None:
                progress(f"delta={delta_time:g} r={int(rte_steps)} q={int(q_m)}")
            seed = _derived_seed(
                master_seed, "wp05b", delta_time, int(rte_steps), int(q_m)
            )
            stream = make_monte_carlo_df_partial_s2_repeated_trajectory_stream(
                preparation,
                delta_time,
                int(q_m),
                config,
                distribution,
                sample_count=sample_count,
                seed=seed,
                maximum_samples=sample_count,
                controlled=True,
                ancilla_qubit=preparation.num_system_qubits,
                construction_policy="boundary_optimized",
            )
            compiled, candidate_probe = _compile_paired_requests(
                _materialize_stream(stream),
                support_definitions,
                compiler,
                training_fingerprint=training_fingerprint,
                maximum_repetition_count=8,
                operator_probe_requested=probe and operator_probe is None,
            )
            compiled.update(
                {
                    "q_m": int(q_m),
                    "partition": (
                        "calibration" if int(q_m) in CALIBRATION_Q else "holdout"
                    ),
                    "master_seed": seed,
                }
            )
            points[str(q_m)] = compiled
            if candidate_probe is not None and operator_probe is None:
                operator_probe = {
                    "delta_time": delta_time,
                    "rte_steps": int(rte_steps),
                    **candidate_probe,
                }
        results[str(rte_steps)] = {"rte_steps": int(rte_steps), "points": points}
    return results, operator_probe


def _models(records: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for rte_steps, row in records.items():
        points = {int(q): value for q, value in row["points"].items()}
        output[rte_steps] = {
            policy: fit_affine_holdouts(points, policy=policy)
            for policy in ("full_basis_shared", POLICY_LABEL)
        }
    return output


def _errors(
    models: Mapping[str, Any],
    *,
    policy: str,
    q_m: int,
    metric: str | None,
) -> list[float]:
    metrics = METRICS if metric is None else (metric,)
    return [
        float(
            row[policy][axis][name]["holdouts"][str(q_m)][
                "absolute_relative_error"
            ]
        )
        for row in models.values()
        for axis in AXES
        for name in metrics
    ]


def _deterministic_errors(
    models: Mapping[str, Any], *, q_m: int, metric: str | None
) -> list[float]:
    metrics = METRICS if metric is None else (metric,)
    return [
        float(
            models[axis][name]["holdouts"][str(q_m)]["absolute_relative_error"]
        )
        for axis in AXES
        for name in metrics
    ]


def _merge_reference(
    wp05a: Mapping[str, Any],
    q8_records: Mapping[str, Any],
    schedule_rte_steps: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rte_steps in schedule_rte_steps:
        key = str(rte_steps)
        upstream = wp05a["randomized_ld3"][key]["points"]
        result[key] = {
            "rte_steps": int(rte_steps),
            "points": {
                "1": upstream["1"],
                "2": upstream["2"],
                "4": upstream["4"],
                "8": q8_records[key]["points"]["8"],
            },
        }
    return result


def _policy_reduction(records_by_delta: Mapping[str, Mapping[str, Any]]) -> dict:
    rows = []
    for delta_label, records in records_by_delta.items():
        for rte_steps, row in records.items():
            for q_label, point in row["points"].items():
                for axis in AXES:
                    full = float(
                        point["axes"][axis]["policies"]["full_basis_shared"][
                            "rz_count"
                        ]["mean"]
                    )
                    selected = float(
                        point["axes"][axis]["policies"][POLICY_LABEL]["rz_count"][
                            "mean"
                        ]
                    )
                    rows.append(
                        {
                            "delta_label": delta_label,
                            "rte_steps": int(rte_steps),
                            "q_m": int(q_label),
                            "axis": axis,
                            "selected_minus_full_over_full": (
                                0.0 if full == 0.0 else (selected - full) / full
                            ),
                        }
                    )
    values = [float(row["selected_minus_full_over_full"]) for row in rows]
    return {
        "rows": rows,
        "minimum_selected_minus_full_over_full": min(values),
        "maximum_selected_minus_full_over_full": max(values),
        "selected_mean_rz_not_worse_at_every_direct_point": all(
            value <= 0.0 for value in values
        ),
    }


def _cross_delta(
    reference: Mapping[str, Any], comparison: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    for rte_steps in sorted(reference, key=int):
        for axis in AXES:
            for policy in ("full_basis_shared", POLICY_LABEL):
                for metric in METRICS:
                    ref = float(
                        reference[rte_steps]["points"]["8"]["axes"][axis][
                            "policies"
                        ][policy][metric]["mean"]
                    )
                    comp = float(
                        comparison[rte_steps]["points"]["8"]["axes"][axis][
                            "policies"
                        ][policy][metric]["mean"]
                    )
                    rows.append(
                        {
                            "rte_steps": int(rte_steps),
                            "axis": axis,
                            "policy": policy,
                            "metric": metric,
                            "delta_0p02_mean": ref,
                            "delta_0p01_mean": comp,
                            "delta_0p01_relative_to_delta_0p02": (
                                0.0 if ref == 0.0 else comp / ref - 1.0
                            ),
                        }
                    )
    selected_rz = [
        float(row["delta_0p01_relative_to_delta_0p02"])
        for row in rows
        if row["policy"] == POLICY_LABEL and row["metric"] == "rz_count"
    ]
    return {
        "comparison": (
            "direct_q8_means_with_delta_specific_trajectory_distributions"
        ),
        "rows": rows,
        "selected_policy_rz_relative_change_range": [
            min(selected_rz),
            max(selected_rz),
        ],
    }


def _reference_bridge(
    reference: Mapping[str, Any], wp06b: Mapping[str, Any]
) -> dict[str, Any]:
    rows = {}
    rz_residuals = []
    for rte_steps, row in reference.items():
        central = wp06b["proxy_recalibration"]["by_rte_steps"][rte_steps][
            "metrics"
        ]
        axes = {}
        for axis in AXES:
            metrics = {}
            point = row["points"]["8"]
            for metric in METRICS:
                actual = float(
                    point["axes"][axis]["paired_delta_selected_minus_full"][
                        metric
                    ]["mean"]
                )
                predicted = 8.0 * float(
                    central[metric]["central_occurrence_paired_delta"]
                )
                baseline = float(
                    point["axes"][axis]["policies"]["full_basis_shared"][metric][
                        "mean"
                    ]
                )
                residual = actual - predicted
                normalized = 0.0 if baseline == 0.0 else abs(residual) / baseline
                metrics[metric] = {
                    "central_additive_prediction": predicted,
                    "direct_full_wrapper_paired_delta": actual,
                    "wrapper_boundary_residual": residual,
                    "absolute_residual_over_full_wrapper": normalized,
                }
                if metric == "rz_count":
                    rz_residuals.append(normalized)
            axes[axis] = metrics
        rows[rte_steps] = {"q_m": 8, "axes": axes}
    return {
        "domain": "delta_0p02_q8_only",
        "delta_0p01_not_evaluated_against_delta_0p02_central_bridge": True,
        "by_rte_steps": rows,
        "maximum_rz_absolute_residual_over_full_wrapper": max(rz_residuals),
    }


def evaluate_wp05b_scope_extension(
    hamiltonian: DFHamiltonian,
    ld3_preparation: DFPartialS2Preparation,
    ld12_preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    wp05a: Mapping[str, Any],
    wp06b: Mapping[str, Any],
    *,
    finite_taylor_order: int = 2,
    schedule_rte_steps: Sequence[int] = (1, 2, 4, 8, 16, 32),
    sample_count: int = 8,
    master_seed: int = 2026092206,
    accuracy_threshold: float = 0.05,
    equivalence_atol: float = 1e-10,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run independent q=8 and delta=0.01 full-wrapper holdouts."""
    validate_wp05a_artifact(wp05a)
    expected_r = tuple(int(value) for value in schedule_rte_steps)
    if not wp05a["overall_pass"]:
        raise ValueError("WP05-a must pass before WP05-b.")
    if float(wp05a["configuration"]["delta_time"]) != REFERENCE_DELTA:
        raise ValueError("WP05-a reference delta is not 0.02.")
    if tuple(wp05a["configuration"]["calibration_q"]) != CALIBRATION_Q:
        raise ValueError("WP05-a calibration q values changed.")
    if tuple(wp05a["configuration"]["schedule_rte_steps"]) != expected_r:
        raise ValueError("WP05-a and WP05-b RTE-step grids differ.")
    if ld3_preparation.ld != 3 or ld12_preparation.ld != 12:
        raise ValueError("WP05-b requires L_D=3 and L_D=12.")
    if wp05a["physical_instance"]["hamiltonian_hash"] != (
        ld3_preparation.hamiltonian_hash
    ):
        raise ValueError("WP05-a and WP05-b Hamiltonian hashes differ.")
    decision = wp06b["decision"]
    if (
        decision["selected_policy"] != POLICY_LABEL
        or int(decision["selected_maximum_support_run_length"])
        != POLICY_RUN_THRESHOLD
        or not decision["selected_policy_approved_for_WP05_validation_path"]
    ):
        raise ValueError("WP06-b did not approve the fixed policy input.")

    training_fingerprint = str(
        wp06b["training_selection"]["training_fingerprint"]
    )
    support_definitions, proof_records = register_support_restricted_bases(
        hamiltonian, ld3_preparation
    )
    reference_q8, _ = _compile_randomized_delta(
        ld3_preparation,
        support_definitions,
        compiler,
        delta_time=REFERENCE_DELTA,
        q_values=(8,),
        schedule_rte_steps=schedule_rte_steps,
        finite_taylor_order=finite_taylor_order,
        sample_count=sample_count,
        master_seed=master_seed,
        training_fingerprint=training_fingerprint,
        probe=False,
        progress=progress,
    )
    comparison, operator_probe = _compile_randomized_delta(
        ld3_preparation,
        support_definitions,
        compiler,
        delta_time=COMPARISON_DELTA,
        q_values=(*CALIBRATION_Q, *HOLDOUT_Q),
        schedule_rte_steps=schedule_rte_steps,
        finite_taylor_order=finite_taylor_order,
        sample_count=sample_count,
        master_seed=master_seed,
        training_fingerprint=training_fingerprint,
        probe=True,
        progress=progress,
    )
    reference = _merge_reference(wp05a, reference_q8, schedule_rte_steps)
    reference_models = _models(reference)
    comparison_models = _models(comparison)

    if progress is not None:
        progress("deterministic L_D=12 reference/comparison points")
    upstream_det = wp05a["deterministic_ld12"]["points"]
    reference_det_points = {
        1: upstream_det["1"],
        2: upstream_det["2"],
        4: upstream_det["4"],
        8: _compile_deterministic_point(
            ld12_preparation, compiler, delta_time=REFERENCE_DELTA, q_m=8
        ),
    }
    comparison_det_points = {
        q_m: _compile_deterministic_point(
            ld12_preparation, compiler, delta_time=COMPARISON_DELTA, q_m=q_m
        )
        for q_m in (*CALIBRATION_Q, *HOLDOUT_Q)
    }
    reference_det_models = fit_affine_holdouts(
        reference_det_points, policy=None
    )
    comparison_det_models = fit_affine_holdouts(
        comparison_det_points, policy=None
    )

    selected_q8_rz = [
        *_errors(
            reference_models, policy=POLICY_LABEL, q_m=8, metric="rz_count"
        ),
        *_errors(
            comparison_models, policy=POLICY_LABEL, q_m=8, metric="rz_count"
        ),
    ]
    selected_q8_all = [
        *_errors(reference_models, policy=POLICY_LABEL, q_m=8, metric=None),
        *_errors(comparison_models, policy=POLICY_LABEL, q_m=8, metric=None),
    ]
    full_q8_rz = [
        *_errors(
            reference_models,
            policy="full_basis_shared",
            q_m=8,
            metric="rz_count",
        ),
        *_errors(
            comparison_models,
            policy="full_basis_shared",
            q_m=8,
            metric="rz_count",
        ),
    ]
    comparison_q4_rz = _errors(
        comparison_models, policy=POLICY_LABEL, q_m=4, metric="rz_count"
    )
    comparison_q4_all = _errors(
        comparison_models, policy=POLICY_LABEL, q_m=4, metric=None
    )
    deterministic_q8_rz = [
        *_deterministic_errors(reference_det_models, q_m=8, metric="rz_count"),
        *_deterministic_errors(comparison_det_models, q_m=8, metric="rz_count"),
    ]
    reduction = _policy_reduction(
        {
            "delta_0p02_q8": reference_q8,
            "delta_0p01_q1_q2_q4_q8": comparison,
        }
    )
    cross_delta = _cross_delta(reference_q8, comparison)
    bridge = _reference_bridge(reference_q8, wp06b)
    proof_residual = max(
        float(row["preserved_columns_max_abs_residual"]) for row in proof_records
    )
    statevector_residual = (
        math.inf
        if operator_probe is None
        else max(
            float(operator_probe["evolution_random_state_max_abs_difference"]),
            float(
                operator_probe["cosine_wrapper_random_state_max_abs_difference"]
            ),
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
        "upstream_wp05a_passed": bool(wp05a["overall_pass"]),
        "q8_directly_transpiled_at_both_deltas": all(
            "8" in records[str(r)]["points"]
            for records in (reference_q8, comparison)
            for r in schedule_rte_steps
        ),
        "delta_0p01_q1_q2_q4_q8_directly_transpiled": all(
            all(str(q) in comparison[str(r)]["points"] for q in (1, 2, 4, 8))
            for r in schedule_rte_steps
        ),
        "support_basis_certificates_pass": proof_residual <= equivalence_atol,
        "delta_0p01_controlled_and_wrapper_probe_passes": (
            statevector_residual <= equivalence_atol
            and phase_residual <= equivalence_atol
        ),
        "selected_policy_q8_rz_holdout_within_5_percent": (
            max(selected_q8_rz) <= accuracy_threshold
        ),
        "selected_policy_q8_all_metrics_within_5_percent": (
            max(selected_q8_all) <= accuracy_threshold
        ),
        "full_basis_q8_rz_holdout_within_5_percent": (
            max(full_q8_rz) <= accuracy_threshold
        ),
        "delta_0p01_selected_q4_rz_holdout_within_5_percent": (
            max(comparison_q4_rz) <= accuracy_threshold
        ),
        "delta_0p01_selected_q4_all_metrics_within_5_percent": (
            max(comparison_q4_all) <= accuracy_threshold
        ),
        "deterministic_q8_rz_holdout_within_5_percent": (
            max(deterministic_q8_rz) <= accuracy_threshold
        ),
        "delta_0p02_q8_additive_bridge_rz_within_5_percent": (
            float(bridge["maximum_rz_absolute_residual_over_full_wrapper"])
            <= accuracy_threshold
        ),
        "selected_policy_mean_rz_not_worse_at_direct_extension_points": bool(
            reduction["selected_mean_rz_not_worse_at_every_direct_point"]
        ),
        "production_default_remains_full_basis": True,
        "reoptimization_and_final_total_cost_not_claimed": True,
    }
    overall_pass = all(checks.values())
    next_action = (
        "WP01-D_C07_reoptimize_alpha_shots_and_reassess_candidate_intervals"
        if overall_pass
        else "refine_full_scope_proxy_or_policy_before_resource_reoptimization"
    )
    return {
        "configuration": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": len(hamiltonian.lambdas),
            "candidate_ld_values": [3, 12],
            "reference_delta_time": REFERENCE_DELTA,
            "comparison_delta_time": COMPARISON_DELTA,
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
        "physical_instance": dict(wp05a["physical_instance"]),
        "policy_input": {
            "policy": POLICY_LABEL,
            "maximum_support_run_length": POLICY_RUN_THRESHOLD,
            "wp06b_training_fingerprint": training_fingerprint,
            "production_default_changed": False,
            "support_basis_definition_count": len(support_definitions),
            "maximum_preserved_columns_residual": proof_residual,
        },
        "upstream_wp05a": {
            "content_fingerprint": wp05a["content_fingerprint"],
            "q1_q2_q4_reference_data_reused_without_retranspilation": True,
        },
        "reference_delta_0p02": {
            "direct_q8_randomized_ld3": reference_q8,
            "combined_affine_models": reference_models,
            "deterministic_ld12_points": {
                str(q): reference_det_points[q] for q in (1, 2, 4, 8)
            },
            "deterministic_ld12_affine_models": reference_det_models,
            "q8_additive_bridge": bridge,
        },
        "comparison_delta_0p01": {
            "direct_randomized_ld3": comparison,
            "affine_models": comparison_models,
            "deterministic_ld12_points": {
                str(q): comparison_det_points[q] for q in (1, 2, 4, 8)
            },
            "deterministic_ld12_affine_models": comparison_det_models,
        },
        "operator_and_axis_semantics_probe": operator_probe,
        "holdout_summary": {
            "maximum_selected_policy_q8_rz_relative_error_both_deltas": max(
                selected_q8_rz
            ),
            "maximum_selected_policy_q8_all_metric_relative_error_both_deltas": max(
                selected_q8_all
            ),
            "maximum_full_basis_q8_rz_relative_error_both_deltas": max(full_q8_rz),
            "maximum_delta_0p01_selected_policy_q4_rz_relative_error": max(
                comparison_q4_rz
            ),
            "maximum_delta_0p01_selected_policy_q4_all_metric_relative_error": max(
                comparison_q4_all
            ),
            "maximum_deterministic_q8_rz_relative_error_both_deltas": max(
                deterministic_q8_rz
            ),
            "maximum_delta_0p02_q8_additive_bridge_rz_residual": bridge[
                "maximum_rz_absolute_residual_over_full_wrapper"
            ],
            "maximum_random_state_action_residual": statevector_residual,
        },
        "policy_rz_reduction": reduction,
        "q8_cross_delta_comparison": cross_delta,
        "decision": {
            "status": (
                "WP05b_scope_extension_passed"
                if overall_pass
                else "WP05b_scope_extension_requires_refinement"
            ),
            "selected_policy_retained": overall_pass,
            "next_action": next_action,
        },
        "scope": {
            "complete_controlled_partial_s2_retranspiled": True,
            "complete_measurement_bearing_hadamard_wrapper_retranspiled": True,
            "cosine_and_sine_axes_included": True,
            "q8_directly_transpiled": True,
            "delta_0p01_directly_transpiled": True,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "quantum_shots_executed": 0,
            "alpha_reoptimized": False,
            "shot_counts_reoptimized": False,
            "round_schedule_reoptimized": False,
            "candidate_intervals_recomputed": False,
            "decision_grade": False,
            "final_total_cost_evaluation_performed": False,
        },
        "limitations": [
            "One H4 rank-12 snapshot, L_D=3/12, delta=0.02/0.01, q<=8, and one topology-free Qiskit compiler context.",
            "The delta=0.02 q=1,2,4 evidence is reused from fingerprint-validated WP05-a; only q=8 is retranspiled there.",
            "The WP06-b additive bridge is assessed at delta=0.02 q=8 only and is not transferred to delta=0.01.",
            "Alpha, shots, and schedules are not reoptimized; candidate intervals are not recomputed and the result remains non-decision-grade.",
            "No state preparation, backend, noise, quantum shots, final total cost, or scientific superiority is evaluated.",
        ],
        "checks": checks,
        "overall_pass": overall_pass,
        "summary": {
            "status": (
                "WP05b_scope_extension_passed"
                if overall_pass
                else "WP05b_scope_extension_requires_refinement"
            ),
            "maximum_selected_policy_q8_rz_relative_error_both_deltas": max(
                selected_q8_rz
            ),
            "maximum_selected_policy_q8_all_metric_relative_error_both_deltas": max(
                selected_q8_all
            ),
            "maximum_delta_0p01_selected_policy_q4_rz_relative_error": max(
                comparison_q4_rz
            ),
            "maximum_delta_0p02_q8_additive_bridge_rz_residual": bridge[
                "maximum_rz_absolute_residual_over_full_wrapper"
            ],
            "selected_policy_direct_rz_relative_change_range": [
                reduction["minimum_selected_minus_full_over_full"],
                reduction["maximum_selected_minus_full_over_full"],
            ],
            "next_action": next_action,
        },
    }


def finalize_wp05b_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP05-b",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp05b_artifact(payload)
    return payload


def validate_wp05b_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WP05-b schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "WP05-b":
        raise ValueError("Unsupported WP05-b method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP05-b content_fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP05-b overall status does not match its checks.")
    scope = payload.get("scope", {})
    if scope.get("q8_directly_transpiled") is not True:
        raise ValueError("WP05-b must directly transpile q=8.")
    if scope.get("delta_0p01_directly_transpiled") is not True:
        raise ValueError("WP05-b must directly transpile delta=0.01.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("WP05-b cannot claim a final total-cost evaluation.")
    if scope.get("decision_grade") is not False:
        raise ValueError("WP05-b remains non-decision-grade before reoptimization.")
    if payload.get("policy_input", {}).get("production_default_changed") is not False:
        raise ValueError("WP05-b must not silently change the production default.")


def write_wp05b_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_wp05b_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

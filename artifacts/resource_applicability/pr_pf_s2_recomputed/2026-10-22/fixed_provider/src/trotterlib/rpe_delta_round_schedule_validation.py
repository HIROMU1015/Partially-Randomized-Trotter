"""Delta screening and round-specific finite-RTE schedule validation.

This module follows the target-round-horizon audit.  It screens only delta
values that were explicitly exercised by the H4 PF-delta validation, builds a
round-specific ``(r_m, K_m)`` schedule from analytic resource constraints, and
then checks the selected schedule against the small-sector matrix reference.

The workload used to rank feasible candidates counts randomized Hamiltonian-
component applications weighted by the provisional shot count.  It is a
screening proxy, not a compiled-circuit cost and not a final total cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian, PhysicalSector
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
from .finite_rte_signal_validation import validate_finite_rte_signals
from .rpe_resource_accounting import (
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundSpecification,
    evaluate_rpe_round_candidate,
)
from .rpe_target_round_horizon_validation import required_rpe_round_horizon
from .rte import finite_rte_distribution


SCHEMA_VERSION = "rpe_delta_round_schedule_validation_v1"
METHOD = "executed_delta_pf_screen_and_roundwise_finite_rte_matrix_validation_v1"


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


def _positive_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _unique_positive(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(sorted({float(value) for value in values}))
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values.")
    return result


def _unique_counts(
    values: Sequence[int], *, name: str, even: bool = False
) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if not result or any(value < (0 if even else 1) for value in result):
        raise ValueError(f"{name} contains an invalid count.")
    if even and any(value % 2 for value in result):
        raise ValueError(f"{name} must contain non-negative even values.")
    return result


def screen_delta_candidate(
    *,
    delta_time: float,
    target_energy_precision: float,
    beta_rpe: float,
    beta_pf_budget: float,
    pf_coefficient: float,
) -> dict[str, Any]:
    """Screen one delta using the empirical ``C q delta**3`` PF model."""
    delta = _positive_float(delta_time, name="delta_time")
    coefficient = _positive_float(pf_coefficient, name="pf_coefficient")
    budget = _positive_float(beta_pf_budget, name="beta_pf_budget")
    horizon = required_rpe_round_horizon(
        target_energy_precision=target_energy_precision,
        beta_rpe=beta_rpe,
        delta_time=delta,
    )
    beta_pf = float(coefficient * int(horizon["q_max"]) * delta**3)
    return {
        "delta_time": delta,
        **horizon,
        "empirical_pf_phase_proxy_at_q_max": beta_pf,
        "beta_pf_budget": budget,
        "empirical_pf_screen_pass": beta_pf <= budget + 1e-15,
    }


def _physical_result(point: Mapping[str, Any]) -> Mapping[str, Any]:
    for item in point["state_results"]:
        if item["state_label"] == "physical_df_ground_state":
            return item
    raise ValueError("physical_df_ground_state result is missing.")


def _expected_component_applications(tau: float, cutoff: int) -> float:
    distribution = finite_rte_distribution(tau, cutoff)
    return float(
        sum(
            (order + 1) * probability
            for order, probability in zip(
                distribution.orders,
                distribution.order_probabilities,
                strict=True,
            )
        )
    )


def validate_rpe_delta_round_schedule(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    ld: int,
    target_energy_precision: float,
    delta_candidates: Sequence[float],
    calibration_delta_values: Sequence[float],
    disjoint_validation_delta_values: Sequence[float],
    pf_coefficient: float,
    pf_coefficient_source: str,
    beta_rpe: float = 0.4,
    beta_pf_budget: float = 0.02,
    beta_rte_budget: float = 0.02,
    beta_stat_budget: float = 0.36,
    alpha_total: float = 0.05,
    rte_step_values: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    finite_taylor_orders: Sequence[int] = (0, 2, 4, 6, 8),
    rte_seed: int = 20260818,
    near_tie_relative_tolerance: float = 0.05,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and matrix-check provisional schedules for PF-feasible deltas."""
    started = time.perf_counter()
    if hamiltonian.n_qubits != sector.n_qubits:
        raise ValueError("Hamiltonian and sector n_qubits differ.")
    deltas = _unique_positive(delta_candidates, name="delta_candidates")
    calibration = set(
        _unique_positive(calibration_delta_values, name="calibration_delta_values")
    )
    validation = set(
        _unique_positive(
            disjoint_validation_delta_values,
            name="disjoint_validation_delta_values",
        )
    )
    if calibration & validation:
        raise ValueError("Calibration and disjoint-validation delta sets overlap.")
    if set(deltas) != calibration | validation:
        raise ValueError("delta_candidates must equal the two declared source sets.")
    r_grid = _unique_counts(rte_step_values, name="rte_step_values")
    k_grid = _unique_counts(
        finite_taylor_orders,
        name="finite_taylor_orders",
        even=True,
    )
    beta = _positive_float(beta_rpe, name="beta_rpe")
    beta_pf = _positive_float(beta_pf_budget, name="beta_pf_budget")
    beta_rte = _positive_float(beta_rte_budget, name="beta_rte_budget")
    beta_stat = _positive_float(beta_stat_budget, name="beta_stat_budget")
    alpha = _positive_float(alpha_total, name="alpha_total")
    if alpha >= 1.0:
        raise ValueError("alpha_total must be smaller than one.")
    if beta_pf + beta_rte + beta_stat > beta + 1e-15:
        raise ValueError("Phase-budget components exceed beta_rpe.")
    tie_tolerance = float(near_tie_relative_tolerance)
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("near_tie_relative_tolerance must be non-negative.")

    partition = split_df_hamiltonian_by_ld(hamiltonian, int(ld))
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
    )
    pf_model = RPEPFErrorModel(
        float(pf_coefficient),
        pf_coefficient_source,
        False,
    )
    sampling_policy = RPEHadamardSamplingPolicy(
        rte_trajectory_mode="fresh_iid_per_hadamard_shot",
        independent_bounded_outcomes_within_each_round_axis=True,
    )

    delta_screen = []
    for delta in deltas:
        item = screen_delta_candidate(
            delta_time=delta,
            target_energy_precision=target_energy_precision,
            beta_rpe=beta,
            beta_pf_budget=beta_pf,
            pf_coefficient=pf_model.coefficient,
        )
        item["source_role"] = (
            "pf_surrogate_calibration_grid"
            if delta in calibration
            else "pf_disjoint_validation_grid"
        )
        delta_screen.append(item)

    analytic_schedules: list[dict[str, Any]] = []
    for screen in delta_screen:
        if not screen["empirical_pf_screen_pass"]:
            continue
        delta = float(screen["delta_time"])
        maximum_round = int(screen["maximum_round_index_M"])
        alpha_axis = float(alpha / (2 * (maximum_round + 1)))
        allocation = RPEErrorAllocation(
            beta_pf,
            beta_rte,
            beta_stat,
            alpha_axis,
            alpha_axis,
        )
        rounds: list[dict[str, Any]] = []
        for round_index in range(maximum_round + 1):
            candidates = []
            specification = RPERoundSpecification(round_index, delta)
            for rte_steps in r_grid:
                for cutoff in k_grid:
                    candidate = evaluate_rpe_round_candidate(
                        preparation,
                        specification,
                        allocation,
                        pf_model,
                        beta_rpe=beta,
                        rte_steps_per_occurrence=rte_steps,
                        finite_taylor_order=cutoff,
                        cost_metric="rz_count",
                        rte_seed=int(rte_seed),
                        hadamard_sampling_policy=sampling_policy,
                    )
                    if not candidate.feasible:
                        continue
                    expected_apps = _expected_component_applications(
                        candidate.tau_m,
                        cutoff,
                    )
                    total_shots = int(
                        candidate.cosine_shots + candidate.sine_shots  # type: ignore[operator]
                    )
                    per_shot_apps = float(
                        candidate.q_m * rte_steps * expected_apps
                    )
                    workload = float(total_shots * per_shot_apps)
                    candidates.append(
                        (
                            workload,
                            rte_steps,
                            cutoff,
                            candidate,
                            expected_apps,
                            total_shots,
                            per_shot_apps,
                        )
                    )
            if not candidates:
                rounds.append(
                    {
                        "round_index": round_index,
                        "q_m": specification.q_m,
                        "analytic_feasible": False,
                    }
                )
                continue
            selected = min(candidates, key=lambda value: value[:3])
            (
                workload,
                rte_steps,
                cutoff,
                candidate,
                expected_apps,
                total_shots,
                per_shot_apps,
            ) = selected
            rounds.append(
                {
                    "round_index": round_index,
                    "q_m": candidate.q_m,
                    "t_m": candidate.t_m,
                    "r_m": rte_steps,
                    "K_m": cutoff,
                    "tau_m": candidate.tau_m,
                    "analytic_feasible": True,
                    "analytic_feasible_grid_candidate_count": len(candidates),
                    "empirical_pf_phase_proxy": candidate.beta_pf,
                    "finite_rte_phase_bound": candidate.beta_rte,
                    "attenuation": candidate.attenuation,
                    "conservative_radius_lower_bound": (
                        candidate.rho_observed_lower_bound
                    ),
                    "cosine_shots": candidate.cosine_shots,
                    "sine_shots": candidate.sine_shots,
                    "total_axis_shots": total_shots,
                    "expected_component_applications_per_rte_short_step": (
                        expected_apps
                    ),
                    "randomized_component_applications_per_shot_proxy": (
                        per_shot_apps
                    ),
                    "shot_weighted_randomized_component_application_proxy": (
                        workload
                    ),
                }
            )
        all_feasible = all(item["analytic_feasible"] for item in rounds)
        feasible_rounds = [item for item in rounds if item["analytic_feasible"]]
        analytic_schedules.append(
            {
                "delta_time": delta,
                "source_role": screen["source_role"],
                "maximum_round_index_M": maximum_round,
                "q_max": int(screen["q_max"]),
                "alpha_axis_uniform": alpha_axis,
                "all_rounds_analytic_feasible": all_feasible,
                "rounds": rounds,
                "total_shots": sum(
                    int(item["total_axis_shots"]) for item in feasible_rounds
                ),
                "minimum_analytic_radius_lower_bound": min(
                    float(item["conservative_radius_lower_bound"])
                    for item in feasible_rounds
                ),
                "maximum_axis_pair_shots": max(
                    int(item["total_axis_shots"]) for item in feasible_rounds
                ),
                "total_shot_weighted_randomized_component_application_proxy": sum(
                    float(
                        item[
                            "shot_weighted_randomized_component_application_proxy"
                        ]
                    )
                    for item in feasible_rounds
                ),
                "selected_r_values": sorted(
                    {int(item["r_m"]) for item in feasible_rounds}
                ),
                "selected_K_values": sorted(
                    {int(item["K_m"]) for item in feasible_rounds}
                ),
            }
        )

    matrix_validations: list[dict[str, Any]] = []
    for schedule in analytic_schedules:
        if not schedule["all_rounds_analytic_feasible"]:
            continue
        rounds = schedule["rounds"]
        selected_pairs = {
            int(item["round_index"]): (int(item["r_m"]), int(item["K_m"]))
            for item in rounds
        }
        matrix_payload = validate_finite_rte_signals(
            hamiltonian,
            sector,
            ld=int(ld),
            delta_time=float(schedule["delta_time"]),
            q_values=tuple(int(item["q_m"]) for item in rounds),
            rte_step_values=tuple(schedule["selected_r_values"]),
            finite_taylor_orders=tuple(schedule["selected_K_values"]),
            beta_rpe=beta,
            beta_pf_budget=beta_pf,
            beta_rte_budget=beta_rte,
            beta_stat_budget=beta_stat,
            alpha_total=alpha,
            seed=int(rte_seed),
            provenance={"role": "selected_schedule_matrix_reference"},
        )
        selected_points = []
        for point in matrix_payload["points"]:
            round_index = int(point["round_index"])
            if (int(point["r_m"]), int(point["K_m"])) != selected_pairs[
                round_index
            ]:
                continue
            physical = _physical_result(point)
            selected_points.append(
                {
                    "round_index": round_index,
                    "q_m": int(point["q_m"]),
                    "r_m": int(point["r_m"]),
                    "K_m": int(point["K_m"]),
                    "attenuation": float(point["attenuation"]),
                    "round_signal_error_bound": float(
                        point["round_signal_error_bound"]
                    ),
                    "corrected_operator_error_spectral_norm": float(
                        point["corrected_operator_error_spectral_norm"]
                    ),
                    "operator_error_bound_pass": bool(
                        point["operator_error_bound_pass"]
                    ),
                    "finite_rte_signal_bound_pass": bool(
                        physical["finite_rte_signal_bound_pass"]
                    ),
                    "pf_vs_exact_phase_error": float(
                        physical["pf_vs_exact_phase_error"]
                    ),
                    "pf_phase_budget_pass": bool(
                        physical["pf_phase_error_within_provisional_budget"]
                    ),
                    "finite_rte_phase_error": float(
                        physical["finite_rte_phase_error"]
                    ),
                    "finite_rte_phase_error_bound": float(
                        physical["finite_rte_phase_error_bound"]
                    ),
                    "finite_rte_phase_bound_pass": bool(
                        physical["finite_rte_phase_bound_pass"]
                    ),
                    "finite_rte_phase_budget_pass": bool(
                        physical[
                            "finite_rte_phase_bound_within_provisional_budget"
                        ]
                    ),
                    "reference_signal_radius": float(
                        physical["reference_signal_radius"]
                    ),
                    "observed_attenuated_radius": float(
                        physical["observed_attenuated_radius"]
                    ),
                    "conservative_radius_lower_bound": float(
                        physical["conservative_radius_lower_bound"]
                    ),
                    "conservative_radius_bound_pass": bool(
                        physical["conservative_radius_bound_pass"]
                    ),
                }
            )
        selected_points.sort(key=lambda item: item["round_index"])
        q_max_point = selected_points[-1]
        screen = next(
            item
            for item in delta_screen
            if item["delta_time"] == schedule["delta_time"]
        )
        matrix_validations.append(
            {
                "delta_time": float(schedule["delta_time"]),
                "finite_rte_signal_validation_fingerprint": matrix_payload[
                    "validation_fingerprint"
                ],
                "selected_point_count": len(selected_points),
                "points": selected_points,
                "minimum_observed_attenuated_radius": min(
                    item["observed_attenuated_radius"] for item in selected_points
                ),
                "maximum_actual_pf_phase_error": max(
                    item["pf_vs_exact_phase_error"] for item in selected_points
                ),
                "maximum_finite_rte_phase_bound": max(
                    item["finite_rte_phase_error_bound"]
                    for item in selected_points
                ),
                "q_max_actual_pf_phase_error": q_max_point[
                    "pf_vs_exact_phase_error"
                ],
                "q_max_empirical_pf_phase_proxy": screen[
                    "empirical_pf_phase_proxy_at_q_max"
                ],
                "empirical_pf_proxy_covers_q_max_actual_phase": bool(
                    q_max_point["pf_vs_exact_phase_error"]
                    <= screen["empirical_pf_phase_proxy_at_q_max"] + 1e-12
                ),
                "all_selected_matrix_checks_pass": all(
                    item["operator_error_bound_pass"]
                    and item["finite_rte_signal_bound_pass"]
                    and item["pf_phase_budget_pass"]
                    and item["finite_rte_phase_bound_pass"]
                    and item["finite_rte_phase_budget_pass"]
                    and item["conservative_radius_bound_pass"]
                    and item["conservative_radius_lower_bound"] > 0.0
                    for item in selected_points
                ),
            }
        )

    feasible_schedules = [
        item for item in analytic_schedules if item["all_rounds_analytic_feasible"]
    ]
    minimum_proxy = min(
        float(item["total_shot_weighted_randomized_component_application_proxy"])
        for item in feasible_schedules
    )
    near_tie_deltas = [
        float(item["delta_time"])
        for item in feasible_schedules
        if float(item["total_shot_weighted_randomized_component_application_proxy"])
        <= minimum_proxy * (1.0 + tie_tolerance)
    ]
    proxy_best = min(
        feasible_schedules,
        key=lambda item: (
            float(item["total_shot_weighted_randomized_component_application_proxy"]),
            float(item["delta_time"]),
        ),
    )
    checks = {
        "at_least_one_delta_passes_empirical_pf_screen": any(
            item["empirical_pf_screen_pass"] for item in delta_screen
        ),
        "all_pf_passing_deltas_have_complete_analytic_schedules": all(
            item["all_rounds_analytic_feasible"] for item in analytic_schedules
        ),
        "all_analytic_schedules_have_matrix_validation": (
            len(matrix_validations) == len(feasible_schedules)
        ),
        "all_selected_matrix_checks_pass": all(
            item["all_selected_matrix_checks_pass"]
            for item in matrix_validations
        ),
        "empirical_pf_proxy_covers_all_q_max_matrix_phases": all(
            item["empirical_pf_proxy_covers_q_max_actual_phase"]
            for item in matrix_validations
        ),
        "selected_r_grid_not_capped": all(
            max(item["selected_r_values"]) < max(r_grid)
            for item in feasible_schedules
        ),
        "selected_K_grid_not_capped": all(
            max(item["selected_K_values"]) < max(k_grid)
            for item in feasible_schedules
        ),
    }
    overall_pass = all(checks.values())
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "scope": {
            "empirical_pf_delta_screen_performed": True,
            "round_specific_finite_rte_schedule_constructed": True,
            "small_system_selected_schedule_matrix_validation_performed": True,
            "q_greater_than_8_circuit_compilation_performed": False,
            "compiled_cost_proxy_used_for_selection": False,
            "fresh_iid_shot_simulation_performed": False,
            "final_total_cost_evaluation_performed": False,
            "pf_coefficient_is_rigorous_bound": False,
            "target_precision_is_provisional_config_value_not_normative_choice": True,
        },
        "system": {
            "molecule": f"H{hamiltonian.metadata.get('molecule_type')}",
            "basis": hamiltonian.metadata.get("basis"),
            "distance_angstrom": hamiltonian.metadata.get("distance"),
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": hamiltonian.n_blocks,
            "ld": int(ld),
            "hamiltonian_hash": preparation.hamiltonian_hash,
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
            "sector_dimension": sector.dimension,
            "sector_n_electrons": sector.n_electrons,
        },
        "configuration": {
            "target_energy_precision_ha": float(target_energy_precision),
            "beta_rpe": beta,
            "beta_pf_budget": beta_pf,
            "beta_rte_budget": beta_rte,
            "beta_stat_budget": beta_stat,
            "alpha_total": alpha,
            "alpha_policy": "uniform_across_all_round_axes_per_delta_candidate",
            "pf_coefficient": pf_model.coefficient,
            "pf_coefficient_source": pf_model.source,
            "pf_coefficient_is_rigorous_bound": pf_model.is_rigorous_bound,
            "delta_candidates": list(deltas),
            "calibration_delta_values": sorted(calibration),
            "disjoint_validation_delta_values": sorted(validation),
            "rte_step_values": list(r_grid),
            "finite_taylor_orders": list(k_grid),
            "rte_seed": int(rte_seed),
            "near_tie_relative_tolerance": tie_tolerance,
            "schedule_selection_objective": (
                "sum_over_rounds_of_total_axis_shots_times_q_m_times_r_m_"
                "times_expected_component_applications_per_short_step"
            ),
            "schedule_selection_objective_limit": (
                "randomized_component_application_screen_only_not_compiled_cost"
            ),
        },
        "delta_pf_screen": delta_screen,
        "analytic_round_schedules": analytic_schedules,
        "selected_schedule_matrix_validations": matrix_validations,
        "summary": {
            "pf_screen_passing_deltas": [
                item["delta_time"]
                for item in delta_screen
                if item["empirical_pf_screen_pass"]
            ],
            "pf_screen_failing_deltas": [
                item["delta_time"]
                for item in delta_screen
                if not item["empirical_pf_screen_pass"]
            ],
            "proxy_best_delta": float(proxy_best["delta_time"]),
            "near_tie_deltas_within_provisional_5_percent": near_tie_deltas,
            "disjoint_validation_grid_passing_deltas": [
                item["delta_time"]
                for item in delta_screen
                if item["source_role"] == "pf_disjoint_validation_grid"
                and item["empirical_pf_screen_pass"]
            ],
            "minimum_matrix_observed_radius_over_selected_schedules": min(
                item["minimum_observed_attenuated_radius"]
                for item in matrix_validations
            ),
            "maximum_matrix_actual_pf_phase_over_selected_schedules": max(
                item["maximum_actual_pf_phase_error"]
                for item in matrix_validations
            ),
            "maximum_matrix_finite_rte_phase_bound_over_selected_schedules": max(
                item["maximum_finite_rte_phase_bound"]
                for item in matrix_validations
            ),
            "checks": checks,
            "overall_pass": overall_pass,
            "interpretation": (
                "feasible_round_specific_schedules_exist_for_three_executed_"
                "delta_values_but_compiled_cost_is_needed_to_rank_near_tied_"
                "delta_0p01_and_0p02"
            ),
            "next_action": (
                "calibrate_or_validate_compiled_cost_for_the_near_tied_delta_"
                "shortlist_before_total_cost_optimization"
            ),
        },
        "performance": {"elapsed_seconds": time.perf_counter() - started},
        "provenance": dict(provenance or {}),
    }
    payload["content_fingerprint"] = _fingerprint(payload)
    validate_rpe_delta_round_schedule_payload(payload)
    return payload


def validate_rpe_delta_round_schedule_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("method") != METHOD:
        raise ValueError("Unsupported RPE delta-round-schedule artifact.")
    unsigned = dict(payload)
    fingerprint = unsigned.pop("content_fingerprint", None)
    if fingerprint != _fingerprint(unsigned):
        raise ValueError("RPE delta-round-schedule content_fingerprint mismatch.")
    configuration = payload.get("configuration", {})
    screen = payload.get("delta_pf_screen", ())
    if len(screen) != len(configuration.get("delta_candidates", ())):
        raise ValueError("Delta-screen count does not match the configured grid.")
    for item in screen:
        expected = screen_delta_candidate(
            delta_time=float(item["delta_time"]),
            target_energy_precision=float(
                configuration["target_energy_precision_ha"]
            ),
            beta_rpe=float(configuration["beta_rpe"]),
            beta_pf_budget=float(configuration["beta_pf_budget"]),
            pf_coefficient=float(configuration["pf_coefficient"]),
        )
        for key, value in expected.items():
            if item.get(key) != value:
                raise ValueError(f"Delta-screen mismatch at {item['delta_time']}: {key}.")
    summary = payload.get("summary", {})
    checks = summary.get("checks", {})
    if summary.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("RPE delta-round-schedule overall status mismatch.")
    scope = payload.get("scope", {})
    if scope.get("q_greater_than_8_circuit_compilation_performed") is not False:
        raise ValueError("This validation cannot claim q>8 circuit compilation.")
    if scope.get("compiled_cost_proxy_used_for_selection") is not False:
        raise ValueError("This validation cannot claim compiled-cost selection.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("This validation cannot claim a final total cost.")


def write_rpe_delta_round_schedule_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_rpe_delta_round_schedule_payload(payload)
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

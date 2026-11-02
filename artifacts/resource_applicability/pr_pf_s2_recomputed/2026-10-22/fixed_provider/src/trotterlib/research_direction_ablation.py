"""WP04 schedule, phase-budget, and failure-allocation ablation helpers.

This validation stays inside the model-conditional scope established by
WP01-S.  It reuses the same short-q affine compiled-cost calibrations and
separates schedule, beta allocation, alpha allocation, and schedule-objective
effects.  Nothing in this module is a final total-cost estimator.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binom

from .df_hamiltonian import DFHamiltonian, PhysicalSector
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import (
    DFPartialS2Preparation,
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from .finite_rte_signal_validation import (
    _circuit_operator_in_openfermion_sector,
    _explicit_cutoff_tolerance,
    _qiskit_to_openfermion_sector_permutation,
    dense_df_operator_in_sector,
)
from .research_direction_prevalidation import (
    affine_prediction_with_standard_error,
)
from .rpe_resource_accounting import (
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundSpecification,
    evaluate_rpe_round_candidate,
)
from .rte import finite_rte_distribution


SCHEMA_VERSION = "research_direction_ablation_v1"
METHOD = "wp04_schedule_beta_alpha_provider_ablation_v1"
SCHEDULE_POLICIES = (
    "fixed_compiled_rz",
    "round_component_applications",
    "round_compiled_rz",
)
ALPHA_POLICIES = ("uniform", "cost_sensitivity_weighted")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AffineAxisCostModel:
    """Two-point affine RZ model and independent calibration errors."""

    q1_mean: float
    q2_mean: float
    q1_standard_error: float
    q2_standard_error: float

    def __post_init__(self) -> None:
        for name in ("q1_mean", "q2_mean"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        for name in ("q1_standard_error", "q2_standard_error"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)

    def predict(self, q_m: int) -> tuple[float, float]:
        prediction, standard_error = affine_prediction_with_standard_error(
            q_m=q_m,
            q1_mean=self.q1_mean,
            q2_mean=self.q2_mean,
            q1_standard_error=self.q1_standard_error,
            q2_standard_error=self.q2_standard_error,
        )
        if prediction < 0.0:
            raise ValueError("Affine compiled-cost prediction became negative.")
        return prediction, standard_error


@dataclass(frozen=True)
class PairCompiledCostModel:
    """Axis-specific RZ models for one finite-RTE ``(r, K)`` pair."""

    rte_steps: int
    finite_taylor_order: int
    cosine: AffineAxisCostModel
    sine: AffineAxisCostModel
    dataset_fingerprint: str
    proxy_fingerprint: str

    def __post_init__(self) -> None:
        if self.rte_steps < 0 or self.finite_taylor_order < 0:
            raise ValueError("r and K must be non-negative.")
        if self.finite_taylor_order % 2:
            raise ValueError("K must be even.")
        for name in ("dataset_fingerprint", "proxy_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 fingerprint.")

    @property
    def pair(self) -> tuple[int, int]:
        return self.rte_steps, self.finite_taylor_order

    def predict(self, axis: str, q_m: int) -> tuple[float, float]:
        if axis == "cosine":
            return self.cosine.predict(q_m)
        if axis == "sine":
            return self.sine.predict(q_m)
        raise ValueError(f"Unsupported axis: {axis}.")


def _expected_component_applications(tau: float, cutoff: int) -> float:
    distribution = finite_rte_distribution(tau, cutoff)
    return float(
        math.fsum(
            (order + 1) * probability
            for order, probability in zip(
                distribution.orders,
                distribution.order_probabilities,
                strict=True,
            )
        )
    )


def _continuous_hoeffding_shots(epsilon: float, alpha: float) -> float:
    return float(2.0 / (epsilon * epsilon) * math.log(2.0 / alpha))


def _evaluate_pair_round(
    preparation: DFPartialS2Preparation,
    pf_model: RPEPFErrorModel,
    model: PairCompiledCostModel,
    *,
    round_index: int,
    delta_time: float,
    beta_rpe: float,
    beta_profile: tuple[float, float, float],
    alpha_cosine: float,
    alpha_sine: float,
    rte_seed: int,
) -> dict[str, Any] | None:
    beta_pf, beta_rte, beta_stat = beta_profile
    candidate = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(round_index, delta_time),
        RPEErrorAllocation(
            beta_pf,
            beta_rte,
            beta_stat,
            alpha_cosine,
            alpha_sine,
        ),
        pf_model,
        beta_rpe=beta_rpe,
        rte_steps_per_occurrence=model.rte_steps,
        finite_taylor_order=model.finite_taylor_order,
        cost_metric="rz_count",
        rte_seed=rte_seed,
        hadamard_sampling_policy=RPEHadamardSamplingPolicy(
            rte_trajectory_mode="fresh_iid_per_hadamard_shot",
            independent_bounded_outcomes_within_each_round_axis=True,
        ),
    )
    if not candidate.feasible:
        return None
    if (
        candidate.cosine_shots is None
        or candidate.sine_shots is None
        or candidate.epsilon_coordinate is None
    ):
        raise RuntimeError("A feasible candidate is missing shot-accounting data.")

    axes: dict[str, dict[str, float | int]] = {}
    compiled_cost = 0.0
    continuous_cost = 0.0
    calibration_standard_error = 0.0
    for axis, shots, alpha in (
        ("cosine", candidate.cosine_shots, alpha_cosine),
        ("sine", candidate.sine_shots, alpha_sine),
    ):
        expected_cost, standard_error = model.predict(axis, candidate.q_m)
        continuous_shots = _continuous_hoeffding_shots(
            candidate.epsilon_coordinate,
            alpha,
        )
        compiled_cost += shots * expected_cost
        continuous_cost += continuous_shots * expected_cost
        calibration_standard_error += shots * standard_error
        axes[axis] = {
            "alpha": float(alpha),
            "shots": int(shots),
            "continuous_hoeffding_shots": continuous_shots,
            "predicted_rz_count_per_interrogation": expected_cost,
            "propagated_calibration_standard_error": standard_error,
        }

    expected_applications = (
        0.0
        if preparation.is_deterministic_only
        else _expected_component_applications(
            candidate.tau_m,
            model.finite_taylor_order,
        )
    )
    total_shots = candidate.cosine_shots + candidate.sine_shots
    component_proxy = float(
        total_shots
        * candidate.q_m
        * model.rte_steps
        * expected_applications
    )
    return {
        "round_index": round_index,
        "q_m": candidate.q_m,
        "r_m": model.rte_steps,
        "K_m": model.finite_taylor_order,
        "tau_m": candidate.tau_m,
        "attenuation": candidate.attenuation,
        "conservative_radius_lower_bound": candidate.rho_observed_lower_bound,
        "epsilon_coordinate": candidate.epsilon_coordinate,
        "empirical_pf_phase_proxy": candidate.beta_pf,
        "finite_rte_phase_bound": candidate.beta_rte,
        "expected_component_applications_per_short_step": expected_applications,
        "shot_weighted_component_application_proxy": component_proxy,
        "compiled_rz_point_estimate": compiled_cost,
        "continuous_shot_compiled_rz_point_estimate": continuous_cost,
        "calibration_standard_error_conservative_sum": (
            calibration_standard_error
        ),
        "axes": axes,
    }


def _select_schedule(
    preparation: DFPartialS2Preparation,
    pf_model: RPEPFErrorModel,
    cost_models: Sequence[PairCompiledCostModel],
    *,
    maximum_round_index: int,
    delta_time: float,
    beta_rpe: float,
    beta_profile: tuple[float, float, float],
    alpha_by_axis: Mapping[tuple[int, str], float],
    schedule_policy: str,
    rte_seed: int,
) -> list[dict[str, Any]]:
    if schedule_policy not in SCHEDULE_POLICIES:
        raise ValueError(f"Unsupported schedule policy: {schedule_policy}.")
    table: dict[tuple[int, int], list[dict[str, Any] | None]] = {}
    for model in cost_models:
        table[model.pair] = [
            _evaluate_pair_round(
                preparation,
                pf_model,
                model,
                round_index=round_index,
                delta_time=delta_time,
                beta_rpe=beta_rpe,
                beta_profile=beta_profile,
                alpha_cosine=alpha_by_axis[(round_index, "cosine")],
                alpha_sine=alpha_by_axis[(round_index, "sine")],
                rte_seed=rte_seed,
            )
            for round_index in range(maximum_round_index + 1)
        ]

    if schedule_policy == "fixed_compiled_rz":
        feasible_pairs = [
            pair
            for pair, rows in table.items()
            if all(row is not None for row in rows)
        ]
        if not feasible_pairs:
            raise RuntimeError("No fixed (r,K) pair is feasible for all rounds.")
        selected_pair = min(
            feasible_pairs,
            key=lambda pair: (
                math.fsum(
                    float(row["compiled_rz_point_estimate"])
                    for row in table[pair]
                    if row is not None
                ),
                pair,
            ),
        )
        return [row for row in table[selected_pair] if row is not None]

    objective_key = (
        "shot_weighted_component_application_proxy"
        if schedule_policy == "round_component_applications"
        else "compiled_rz_point_estimate"
    )
    rounds = []
    for round_index in range(maximum_round_index + 1):
        feasible = [
            (pair, rows[round_index])
            for pair, rows in table.items()
            if rows[round_index] is not None
        ]
        if not feasible:
            raise RuntimeError(f"No feasible candidate at round {round_index}.")
        pair, selected = min(
            feasible,
            key=lambda item: (
                float(item[1][objective_key]),  # type: ignore[index]
                float(item[1]["compiled_rz_point_estimate"]),  # type: ignore[index]
                item[0],
            ),
        )
        del pair
        if selected is None:  # pragma: no cover - guarded by feasible filter
            raise RuntimeError("Selected schedule row unexpectedly vanished.")
        rounds.append(selected)
    return rounds


def _weighted_alpha(
    rounds: Sequence[Mapping[str, Any]],
    *,
    alpha_total: float,
) -> dict[tuple[int, str], float]:
    weights: dict[tuple[int, str], float] = {}
    for row in rounds:
        epsilon = float(row["epsilon_coordinate"])
        for axis in ("cosine", "sine"):
            expected_cost = float(
                row["axes"][axis]["predicted_rz_count_per_interrogation"]
            )
            weights[(int(row["round_index"]), axis)] = float(
                2.0 * expected_cost / (epsilon * epsilon)
            )
    total_weight = math.fsum(weights.values())
    return {
        key: float(alpha_total * weight / total_weight)
        for key, weight in weights.items()
    }


def evaluate_ablation_scenario(
    preparation: DFPartialS2Preparation,
    *,
    ld: int,
    pf_coefficient: float,
    cost_models: Sequence[PairCompiledCostModel],
    beta_profile_label: str,
    beta_profile: tuple[float, float, float],
    schedule_policy: str,
    alpha_policy: str,
    maximum_round_index: int = 17,
    delta_time: float = 0.02,
    beta_rpe: float = 0.4,
    alpha_total: float = 0.05,
    rte_seed: int = 20260818,
    maximum_alpha_iterations: int = 20,
    alpha_convergence_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Evaluate one WP04 factorial cell with a fixed compiled-cost family."""
    if alpha_policy not in ALPHA_POLICIES:
        raise ValueError(f"Unsupported alpha policy: {alpha_policy}.")
    if len(beta_profile) != 3 or any(value < 0.0 for value in beta_profile):
        raise ValueError("beta_profile must contain three non-negative values.")
    if not math.isclose(sum(beta_profile), beta_rpe, abs_tol=1e-14):
        raise ValueError("beta_profile must sum to beta_rpe.")
    if not cost_models:
        raise ValueError("At least one compiled-cost model is required.")
    if alpha_convergence_tolerance <= 0.0:
        raise ValueError("alpha_convergence_tolerance must be positive.")
    axis_count = 2 * (maximum_round_index + 1)
    uniform_alpha = alpha_total / axis_count
    alpha_by_axis = {
        (round_index, axis): uniform_alpha
        for round_index in range(maximum_round_index + 1)
        for axis in ("cosine", "sine")
    }
    pf_model = RPEPFErrorModel(
        float(pf_coefficient),
        "same_snapshot_paper_d6_empirical_coefficient",
        False,
    )

    convergence_rows: list[dict[str, Any]] = []
    alpha_converged = alpha_policy == "uniform"
    if alpha_policy == "cost_sensitivity_weighted":
        for iteration in range(1, maximum_alpha_iterations + 1):
            selected = _select_schedule(
                preparation,
                pf_model,
                cost_models,
                maximum_round_index=maximum_round_index,
                delta_time=delta_time,
                beta_rpe=beta_rpe,
                beta_profile=beta_profile,
                alpha_by_axis=alpha_by_axis,
                schedule_policy=schedule_policy,
                rte_seed=rte_seed,
            )
            updated = _weighted_alpha(selected, alpha_total=alpha_total)
            maximum_change = max(
                abs(updated[key] - alpha_by_axis[key]) for key in updated
            )
            convergence_rows.append(
                {
                    "iteration": iteration,
                    "maximum_alpha_change": maximum_change,
                    "selected_r_k_pairs": [
                        f"r{row['r_m']}_k{row['K_m']}" for row in selected
                    ],
                }
            )
            alpha_by_axis = updated
            if maximum_change <= alpha_convergence_tolerance:
                alpha_converged = True
                break
    selected = _select_schedule(
        preparation,
        pf_model,
        cost_models,
        maximum_round_index=maximum_round_index,
        delta_time=delta_time,
        beta_rpe=beta_rpe,
        beta_profile=beta_profile,
        alpha_by_axis=alpha_by_axis,
        schedule_policy=schedule_policy,
        rte_seed=rte_seed,
    )
    if alpha_policy == "cost_sensitivity_weighted":
        consistency = _weighted_alpha(selected, alpha_total=alpha_total)
        alpha_converged = alpha_converged and max(
            abs(consistency[key] - alpha_by_axis[key]) for key in consistency
        ) <= alpha_convergence_tolerance

    total_cost = math.fsum(
        float(row["compiled_rz_point_estimate"]) for row in selected
    )
    continuous_cost = math.fsum(
        float(row["continuous_shot_compiled_rz_point_estimate"])
        for row in selected
    )
    calibration_se = math.fsum(
        float(row["calibration_standard_error_conservative_sum"])
        for row in selected
    )
    calibration_half_width = 1.96 * calibration_se
    total_shots = sum(
        int(row["axes"][axis]["shots"])
        for row in selected
        for axis in ("cosine", "sine")
    )
    final_round_cost = float(selected[-1]["compiled_rz_point_estimate"])
    last_three_cost = math.fsum(
        float(row["compiled_rz_point_estimate"]) for row in selected[-3:]
    )
    return {
        "scenario_id": (
            f"ld{ld}:{schedule_policy}:{beta_profile_label}:{alpha_policy}"
        ),
        "ld": int(ld),
        "schedule_policy": schedule_policy,
        "schedule_selection_cost_provider": (
            "q1_q2_affine_full_hadamard_rz"
            if schedule_policy in ("fixed_compiled_rz", "round_compiled_rz")
            else "analytic_randomized_component_applications"
        ),
        "beta_profile": beta_profile_label,
        "beta_pf_budget": beta_profile[0],
        "beta_rte_budget": beta_profile[1],
        "beta_stat_budget": beta_profile[2],
        "alpha_policy": alpha_policy,
        "alpha_total_allocated": math.fsum(alpha_by_axis.values()),
        "alpha_convergence_tolerance": alpha_convergence_tolerance,
        "alpha_iterations": convergence_rows,
        "alpha_converged": alpha_converged,
        "all_rounds_feasible": True,
        "rounds": selected,
        "selected_r_k_pairs": sorted(
            {f"r{row['r_m']}_k{row['K_m']}" for row in selected}
        ),
        "total_shots": total_shots,
        "total_compiled_rz_point_estimate": total_cost,
        "continuous_shot_compiled_rz_point_estimate": continuous_cost,
        "integer_shot_rounding_relative_overhead": (
            (total_cost - continuous_cost) / continuous_cost
        ),
        "conservative_calibration_95_half_width": calibration_half_width,
        "scenario_intervals": {
            "local_5_percent_plus_calibration": [
                max(0.0, 0.95 * total_cost - calibration_half_width),
                1.05 * total_cost + calibration_half_width,
            ],
            "transfer_25_percent_plus_calibration": [
                max(0.0, 0.75 * total_cost - calibration_half_width),
                1.25 * total_cost + calibration_half_width,
            ],
        },
        "total_component_application_proxy": math.fsum(
            float(row["shot_weighted_component_application_proxy"])
            for row in selected
        ),
        "minimum_conservative_radius_lower_bound": min(
            float(row["conservative_radius_lower_bound"])
            for row in selected
        ),
        "maximum_empirical_pf_phase_proxy": max(
            float(row["empirical_pf_phase_proxy"]) for row in selected
        ),
        "maximum_finite_rte_phase_bound": max(
            float(row["finite_rte_phase_bound"] or 0.0) for row in selected
        ),
        "final_round_cost_fraction": final_round_cost / total_cost,
        "last_three_round_cost_fraction": last_three_cost / total_cost,
    }


def _scenario_lookup(
    scenarios: Sequence[Mapping[str, Any]], scenario_id: str
) -> Mapping[str, Any]:
    matches = [item for item in scenarios if item["scenario_id"] == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"Scenario {scenario_id!r} is missing or duplicated.")
    return matches[0]


def _step_ledger(
    scenarios: Sequence[Mapping[str, Any]],
    specs: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    baseline = float(
        _scenario_lookup(scenarios, specs[0][1])[
            "total_compiled_rz_point_estimate"
        ]
    )
    previous = None
    for label, scenario_id in specs:
        scenario = _scenario_lookup(scenarios, scenario_id)
        cost = float(scenario["total_compiled_rz_point_estimate"])
        rows.append(
            {
                "step": label,
                "scenario_id": scenario_id,
                "total_compiled_rz_point_estimate": cost,
                "relative_reduction_from_previous": (
                    None if previous is None else (previous - cost) / previous
                ),
                "relative_reduction_from_baseline": (baseline - cost) / baseline,
            }
        )
        previous = cost
    return rows


def build_wp04_ablation_body(
    preparations: Mapping[int, DFPartialS2Preparation],
    *,
    pf_coefficients: Mapping[int, float],
    cost_models: Mapping[int, Sequence[PairCompiledCostModel]],
    maximum_round_index: int = 17,
    delta_time: float = 0.02,
    beta_rpe: float = 0.4,
    alpha_total: float = 0.05,
    rte_seed: int = 20260818,
) -> dict[str, Any]:
    """Build the complete WP04 factorial, sequential, and LOO ledgers."""
    beta_profiles = {
        "legacy": (0.08, 0.08, 0.24),
        "guarded": (0.02, 0.02, 0.36),
        "rebalanced_common": (0.015, 0.005, 0.38),
        "deterministic_rebalanced": (0.015, 0.0, 0.385),
    }
    profiles_by_ld = {
        3: ("legacy", "guarded", "rebalanced_common"),
        12: (
            "legacy",
            "guarded",
            "rebalanced_common",
            "deterministic_rebalanced",
        ),
    }
    scenarios = []
    for ld in (3, 12):
        for schedule_policy in SCHEDULE_POLICIES:
            for profile_label in profiles_by_ld[ld]:
                for alpha_policy in ALPHA_POLICIES:
                    scenarios.append(
                        evaluate_ablation_scenario(
                            preparations[ld],
                            ld=ld,
                            pf_coefficient=pf_coefficients[ld],
                            cost_models=cost_models[ld],
                            beta_profile_label=profile_label,
                            beta_profile=beta_profiles[profile_label],
                            schedule_policy=schedule_policy,
                            alpha_policy=alpha_policy,
                            maximum_round_index=maximum_round_index,
                            delta_time=delta_time,
                            beta_rpe=beta_rpe,
                            alpha_total=alpha_total,
                            rte_seed=rte_seed,
                        )
                    )

    sequential_specs = {
        3: (
            (
                "fixed_schedule_legacy_beta_uniform_alpha",
                "ld3:fixed_compiled_rz:legacy:uniform",
            ),
            (
                "round_component_schedule",
                "ld3:round_component_applications:legacy:uniform",
            ),
            (
                "rebalanced_beta",
                "ld3:round_component_applications:rebalanced_common:uniform",
            ),
            (
                "weighted_alpha",
                (
                    "ld3:round_component_applications:rebalanced_common:"
                    "cost_sensitivity_weighted"
                ),
            ),
            (
                "compiled_cost_aligned_schedule_selection",
                (
                    "ld3:round_compiled_rz:rebalanced_common:"
                    "cost_sensitivity_weighted"
                ),
            ),
        ),
        12: (
            (
                "fixed_schedule_legacy_beta_uniform_alpha",
                "ld12:fixed_compiled_rz:legacy:uniform",
            ),
            (
                "round_component_schedule",
                "ld12:round_component_applications:legacy:uniform",
            ),
            (
                "rebalanced_beta",
                (
                    "ld12:round_component_applications:"
                    "deterministic_rebalanced:uniform"
                ),
            ),
            (
                "weighted_alpha",
                (
                    "ld12:round_component_applications:"
                    "deterministic_rebalanced:cost_sensitivity_weighted"
                ),
            ),
            (
                "compiled_cost_aligned_schedule_selection",
                (
                    "ld12:round_compiled_rz:deterministic_rebalanced:"
                    "cost_sensitivity_weighted"
                ),
            ),
        ),
    }
    sequential = {
        str(ld): _step_ledger(scenarios, specs)
        for ld, specs in sequential_specs.items()
    }

    full_ids = {
        3: (
            "ld3:round_compiled_rz:rebalanced_common:"
            "cost_sensitivity_weighted"
        ),
        12: (
            "ld12:round_compiled_rz:deterministic_rebalanced:"
            "cost_sensitivity_weighted"
        ),
    }
    loo_ids = {
        3: {
            "round_schedule": (
                "ld3:fixed_compiled_rz:rebalanced_common:"
                "cost_sensitivity_weighted"
            ),
            "beta_reallocation": (
                "ld3:round_compiled_rz:legacy:cost_sensitivity_weighted"
            ),
            "alpha_reallocation": (
                "ld3:round_compiled_rz:rebalanced_common:uniform"
            ),
            "compiled_cost_aligned_selection": (
                "ld3:round_component_applications:rebalanced_common:"
                "cost_sensitivity_weighted"
            ),
        },
        12: {
            "round_schedule": (
                "ld12:fixed_compiled_rz:deterministic_rebalanced:"
                "cost_sensitivity_weighted"
            ),
            "beta_reallocation": (
                "ld12:round_compiled_rz:legacy:cost_sensitivity_weighted"
            ),
            "alpha_reallocation": (
                "ld12:round_compiled_rz:deterministic_rebalanced:uniform"
            ),
            "compiled_cost_aligned_selection": (
                "ld12:round_component_applications:"
                "deterministic_rebalanced:cost_sensitivity_weighted"
            ),
        },
    }
    leave_one_out = {}
    for ld in (3, 12):
        full = _scenario_lookup(scenarios, full_ids[ld])
        full_cost = float(full["total_compiled_rz_point_estimate"])
        leave_one_out[str(ld)] = [
            {
                "removed_factor": factor,
                "scenario_id": scenario_id,
                "total_compiled_rz_point_estimate": float(
                    _scenario_lookup(scenarios, scenario_id)[
                        "total_compiled_rz_point_estimate"
                    ]
                ),
                "relative_cost_increase_when_removed": (
                    float(
                        _scenario_lookup(scenarios, scenario_id)[
                            "total_compiled_rz_point_estimate"
                        ]
                    )
                    - full_cost
                )
                / full_cost,
            }
            for factor, scenario_id in loo_ids[ld].items()
        ]

    full_ld3 = _scenario_lookup(scenarios, full_ids[3])
    full_ld12 = _scenario_lookup(scenarios, full_ids[12])
    ld3_cost = float(full_ld3["total_compiled_rz_point_estimate"])
    ld12_cost = float(full_ld12["total_compiled_rz_point_estimate"])
    local_overlap = _intervals_overlap(
        full_ld3["scenario_intervals"]["local_5_percent_plus_calibration"],
        full_ld12["scenario_intervals"]["local_5_percent_plus_calibration"],
    )
    transfer_overlap = _intervals_overlap(
        full_ld3["scenario_intervals"]["transfer_25_percent_plus_calibration"],
        full_ld12["scenario_intervals"]["transfer_25_percent_plus_calibration"],
    )

    interactions = {}
    full_profiles = ((3, "rebalanced_common"), (12, "deterministic_rebalanced"))
    for ld, full_profile in full_profiles:
        legacy_uniform = _scenario_lookup(
            scenarios, f"ld{ld}:round_compiled_rz:legacy:uniform"
        )
        legacy_weighted = _scenario_lookup(
            scenarios,
            f"ld{ld}:round_compiled_rz:legacy:cost_sensitivity_weighted",
        )
        full_uniform = _scenario_lookup(
            scenarios, f"ld{ld}:round_compiled_rz:{full_profile}:uniform"
        )
        full_weighted = _scenario_lookup(scenarios, full_ids[ld])
        alpha_gain_legacy = float(
            legacy_uniform["total_compiled_rz_point_estimate"]
            - legacy_weighted["total_compiled_rz_point_estimate"]
        )
        alpha_gain_rebalanced = float(
            full_uniform["total_compiled_rz_point_estimate"]
            - full_weighted["total_compiled_rz_point_estimate"]
        )
        interactions[str(ld)] = {
            "alpha_gain_at_legacy_beta": alpha_gain_legacy,
            "alpha_gain_at_rebalanced_beta": alpha_gain_rebalanced,
            "beta_alpha_additive_interaction_rz": (
                alpha_gain_rebalanced - alpha_gain_legacy
            ),
        }

    ld3_fixed_full = _scenario_lookup(
        scenarios,
        "ld3:fixed_compiled_rz:rebalanced_common:cost_sensitivity_weighted",
    )
    ld3_component_full = _scenario_lookup(
        scenarios,
        (
            "ld3:round_component_applications:rebalanced_common:"
            "cost_sensitivity_weighted"
        ),
    )
    provider_diagnostic = {
        "ld3_component_objective_schedule_relative_rz_change_vs_fixed": (
            float(ld3_component_full["total_compiled_rz_point_estimate"])
            / float(ld3_fixed_full["total_compiled_rz_point_estimate"])
            - 1.0
        ),
        "ld3_compiled_objective_schedule_relative_rz_change_vs_fixed": (
            float(full_ld3["total_compiled_rz_point_estimate"])
            / float(ld3_fixed_full["total_compiled_rz_point_estimate"])
            - 1.0
        ),
        "schedule_objective_changes_direction": bool(
            float(ld3_component_full["total_compiled_rz_point_estimate"])
            > float(ld3_fixed_full["total_compiled_rz_point_estimate"])
            and float(full_ld3["total_compiled_rz_point_estimate"])
            < float(ld3_fixed_full["total_compiled_rz_point_estimate"])
        ),
        "interpretation": (
            "component_application_and_compiled_RZ_schedule_objectives_are_"
            "reported_as_separate_factors"
        ),
    }

    checks = {
        "all_scenarios_feasible": all(
            bool(item["all_rounds_feasible"]) for item in scenarios
        ),
        "all_alpha_allocations_converged": all(
            bool(item["alpha_converged"]) for item in scenarios
        ),
        "all_alpha_sums_match": all(
            math.isclose(
                float(item["alpha_total_allocated"]),
                alpha_total,
                abs_tol=1e-14,
            )
            for item in scenarios
        ),
        "all_beta_sums_match": all(
            math.isclose(
                float(item["beta_pf_budget"])
                + float(item["beta_rte_budget"])
                + float(item["beta_stat_budget"]),
                beta_rpe,
                abs_tol=1e-14,
            )
            for item in scenarios
        ),
        "all_pf_budgets_satisfied": all(
            float(item["maximum_empirical_pf_phase_proxy"])
            <= float(item["beta_pf_budget"]) + 1e-15
            for item in scenarios
        ),
        "all_rte_budgets_satisfied": all(
            float(item["maximum_finite_rte_phase_bound"])
            <= float(item["beta_rte_budget"]) + 1e-15
            for item in scenarios
        ),
        "all_radius_lower_bounds_positive": all(
            float(item["minimum_conservative_radius_lower_bound"]) > 0.0
            for item in scenarios
        ),
        "schedule_objective_difference_detected": bool(
            provider_diagnostic["schedule_objective_changes_direction"]
        ),
    }
    return {
        "scope": {
            "comparison_task": "WP00 fixed H4 CA_over_10 task",
            "circuit_cost_scope": (
                "single_hadamard_interrogation_without_state_preparation"
            ),
            "q1_q2_direct_calibration_reused": True,
            "q_greater_than_2_cost_method": "axis_affine_q1_q2_extrapolation",
            "unused_long_q_holdout_available": False,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "final_total_cost_evaluation_performed": False,
            "decision_grade": False,
        },
        "configuration": {
            "candidate_ld_values": [3, 12],
            "delta_time": delta_time,
            "maximum_round_index_M": maximum_round_index,
            "q_max": 1 << maximum_round_index,
            "beta_rpe": beta_rpe,
            "beta_profiles": {
                key: {
                    "beta_pf": value[0],
                    "beta_rte": value[1],
                    "beta_stat": value[2],
                    "applicability": (
                        "deterministic_endpoint_only"
                        if key == "deterministic_rebalanced"
                        else "both_candidates"
                    ),
                }
                for key, value in beta_profiles.items()
            },
            "alpha_total": alpha_total,
            "alpha_policies": list(ALPHA_POLICIES),
            "schedule_policies": list(SCHEDULE_POLICIES),
            "factorial_cell_count": len(scenarios),
            "rte_seed": rte_seed,
            "pf_coefficients": {
                str(key): float(value) for key, value in pf_coefficients.items()
            },
            "pf_coefficients_are_rigorous_bounds": False,
        },
        "scenarios": scenarios,
        "sequential_ablation": sequential,
        "leave_one_out": leave_one_out,
        "factor_interactions": interactions,
        "cost_provider_diagnostic": provider_diagnostic,
        "full_setting": {
            "scenario_ids": {str(key): value for key, value in full_ids.items()},
            "ld3_total_compiled_rz_point_estimate": ld3_cost,
            "ld12_total_compiled_rz_point_estimate": ld12_cost,
            "ld3_over_ld12_point_estimate_ratio": ld3_cost / ld12_cost,
            "ld12_reduction_relative_to_ld3_point_estimate": (
                (ld3_cost - ld12_cost) / ld3_cost
            ),
            "local_5_percent_intervals_overlap": local_overlap,
            "transfer_25_percent_intervals_overlap": transfer_overlap,
            "directional_result": (
                "undetermined_between_intermediate_and_deterministic_endpoint"
                if transfer_overlap
                else "conditional_intervals_separate"
            ),
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
    }


def _intervals_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) <= min(
        float(left[1]), float(right[1])
    )


def deterministic_physical_signal_diagnostic(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    ld: int,
    delta_time: float,
    q_values: Sequence[int],
    seed: int = 20260818,
) -> dict[str, Any]:
    """Compute physical-ground-state signal radii for a tail-free endpoint."""
    dense = dense_df_operator_in_sector(hamiltonian, sector)
    _eigenvalues, eigenvectors = np.linalg.eigh(dense)
    physical_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, ld),
        identity_policy="extract_identity_phase",
    )
    if not preparation.is_deterministic_only:
        raise ValueError("The deterministic signal diagnostic requires no RTE tail.")
    request = make_df_partial_s2_step_request(
        preparation,
        step_time=delta_time,
        rte_steps=1,
        truncation_tolerance=_explicit_cutoff_tolerance(0.0, 0),
        finite_taylor_order=0,
        seed=seed,
    )
    parts = QiskitDFPartialS2CircuitBuilder().build_additive_circuits(request)
    permutation = _qiskit_to_openfermion_sector_permutation(sector)
    forward = _circuit_operator_in_openfermion_sector(
        parts.forward_deterministic_half,
        sector,
        permutation,
    )
    reverse = _circuit_operator_in_openfermion_sector(
        parts.reverse_deterministic_half,
        sector,
        permutation,
    )
    step = reverse.matrix @ forward.matrix
    identity = np.eye(sector.dimension, dtype=np.complex128)
    rows = []
    for q_m in q_values:
        evolved_state = np.linalg.matrix_power(step, int(q_m)) @ physical_state
        signal = complex(np.vdot(physical_state, evolved_state))
        rows.append(
            {
                "q_m": int(q_m),
                "signal": {"real": float(signal.real), "imag": float(signal.imag)},
                "observed_radius": float(abs(signal)),
                "unit_radius_model_absolute_error": float(abs(abs(signal) - 1.0)),
            }
        )
    return {
        "method": "exact_sector_matrix_power_tail_free_partial_s2",
        "unitary_defect_spectral_norm": float(
            np.linalg.norm(step.conj().T @ step - identity, ord=2)
        ),
        "minimum_observed_radius": min(row["observed_radius"] for row in rows),
        "maximum_unit_radius_model_absolute_error": max(
            row["unit_radius_model_absolute_error"] for row in rows
        ),
        "points": rows,
    }


def selected_matrix_signal_diagnostic(
    matrix_payload: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the selected physical-state signals from a finite-RTE grid."""
    selected = {
        (int(row["q_m"]), int(row["r_m"]), int(row["K_m"]))
        for row in scenario["rounds"]
    }
    points = []
    for point in matrix_payload["points"]:
        key = (int(point["q_m"]), int(point["r_m"]), int(point["K_m"]))
        if key not in selected:
            continue
        physical = next(
            item
            for item in point["state_results"]
            if item["state_label"] == "physical_df_ground_state"
        )
        points.append(
            {
                "round_index": int(point["round_index"]),
                "q_m": key[0],
                "r_m": key[1],
                "K_m": key[2],
                "signal": physical["attenuated_event_mean_signal"],
                "observed_radius": float(physical["observed_attenuated_radius"]),
                "conservative_radius_lower_bound": float(
                    physical["conservative_radius_lower_bound"]
                ),
                "observed_minus_bound": float(
                    physical["observed_attenuated_radius"]
                    - physical["conservative_radius_lower_bound"]
                ),
                "radius_bound_pass": bool(
                    physical["conservative_radius_bound_pass"]
                ),
            }
        )
    points.sort(key=lambda item: item["round_index"])
    if len(points) != len(scenario["rounds"]):
        raise ValueError("Matrix signal grid does not cover the selected schedule.")
    return {
        "method": "selected_points_from_finite_rte_sector_matrix_grid",
        "matrix_validation_fingerprint": matrix_payload["validation_fingerprint"],
        "all_radius_bounds_pass": all(item["radius_bound_pass"] for item in points),
        "minimum_observed_radius": min(item["observed_radius"] for item in points),
        "minimum_conservative_radius_lower_bound": min(
            item["conservative_radius_lower_bound"] for item in points
        ),
        "minimum_observed_minus_bound": min(
            item["observed_minus_bound"] for item in points
        ),
        "points": points,
    }


def exact_coordinate_failure_probability(
    *, mean: float, shots: int, epsilon_coordinate: float
) -> float:
    """Exact two-sided failure probability for a +/-1 sample mean."""
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if not -1.0 - 1e-12 <= mean <= 1.0 + 1e-12:
        raise ValueError("mean must lie in [-1,1].")
    if epsilon_coordinate <= 0.0:
        raise ValueError("epsilon_coordinate must be positive.")
    probability_plus = float(np.clip((1.0 + mean) / 2.0, 0.0, 1.0))
    lower_exclusive = shots * (1.0 + mean - epsilon_coordinate) / 2.0
    upper_exclusive = shots * (1.0 + mean + epsilon_coordinate) / 2.0
    success_min = max(0, math.floor(lower_exclusive) + 1)
    success_max = min(shots, math.ceil(upper_exclusive) - 1)
    if success_min > success_max:
        return 1.0
    success = float(
        binom.cdf(success_max, shots, probability_plus)
        - binom.cdf(success_min - 1, shots, probability_plus)
    )
    return float(min(1.0, max(0.0, 1.0 - success)))


def exact_binomial_minimum_shots(
    *,
    mean: float,
    epsilon_coordinate: float,
    alpha: float,
    hoeffding_shots: int,
) -> int:
    """Find the first shot count meeting alpha up to the Hoeffding count."""
    for shots in range(1, hoeffding_shots + 1):
        if exact_coordinate_failure_probability(
            mean=mean,
            shots=shots,
            epsilon_coordinate=epsilon_coordinate,
        ) <= alpha:
            return shots
    raise RuntimeError("Hoeffding shots did not satisfy the exact binomial tail.")


def statistical_bound_diagnostic(
    scenario: Mapping[str, Any],
    signal_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate radius, integer-ceiling, and Hoeffding conservatism."""
    signals = {
        int(item["q_m"]): complex(
            float(item["signal"]["real"]),
            float(item["signal"]["imag"]),
        )
        for item in signal_diagnostic["points"]
    }
    axis_rows = []
    actual_radius_cost = 0.0
    exact_binomial_cost = 0.0
    exact_union_at_hoeffding = 0.0
    for row in scenario["rounds"]:
        q_m = int(row["q_m"])
        signal = signals[q_m]
        actual_coordinate = float(
            abs(signal)
            * math.sin(float(scenario["beta_stat_budget"]))
            / math.sqrt(2.0)
        )
        for axis, mean in (("cosine", signal.real), ("sine", signal.imag)):
            axis_payload = row["axes"][axis]
            alpha = float(axis_payload["alpha"])
            hoeffding_shots = int(axis_payload["shots"])
            epsilon = float(row["epsilon_coordinate"])
            expected_cost = float(
                axis_payload["predicted_rz_count_per_interrogation"]
            )
            exact_failure = exact_coordinate_failure_probability(
                mean=float(mean),
                shots=hoeffding_shots,
                epsilon_coordinate=epsilon,
            )
            exact_shots = exact_binomial_minimum_shots(
                mean=float(mean),
                epsilon_coordinate=epsilon,
                alpha=alpha,
                hoeffding_shots=hoeffding_shots,
            )
            actual_radius_shots = int(
                math.ceil(_continuous_hoeffding_shots(actual_coordinate, alpha))
            )
            actual_radius_cost += actual_radius_shots * expected_cost
            exact_binomial_cost += exact_shots * expected_cost
            exact_union_at_hoeffding += exact_failure
            axis_rows.append(
                {
                    "q_m": q_m,
                    "axis": axis,
                    "true_mean": float(mean),
                    "alpha": alpha,
                    "epsilon_coordinate_bound": epsilon,
                    "actual_radius_coordinate": actual_coordinate,
                    "hoeffding_shots": hoeffding_shots,
                    "actual_radius_hoeffding_shots": actual_radius_shots,
                    "exact_binomial_minimum_shots": exact_shots,
                    "exact_failure_at_hoeffding_shots": exact_failure,
                    "exact_failure_within_allocated_alpha": exact_failure <= alpha,
                }
            )
    point_cost = float(scenario["total_compiled_rz_point_estimate"])
    return {
        "method": (
            "exact_binomial_coordinate_tails_on_sector_matrix_signals_with_"
            "the_selected_bound_based_coordinate_tolerance"
        ),
        "axis_count": len(axis_rows),
        "all_exact_failures_within_allocated_alpha": all(
            item["exact_failure_within_allocated_alpha"] for item in axis_rows
        ),
        "exact_union_bound_at_hoeffding_shots": exact_union_at_hoeffding,
        "selected_alpha_total": float(scenario["alpha_total_allocated"]),
        "actual_radius_hoeffding_rz_counterfactual": actual_radius_cost,
        "actual_radius_relative_change_vs_conservative_bound": (
            actual_radius_cost / point_cost - 1.0
        ),
        "exact_binomial_minimum_rz_counterfactual": exact_binomial_cost,
        "exact_binomial_relative_reduction_vs_hoeffding": (
            (point_cost - exact_binomial_cost) / point_cost
        ),
        "integer_shot_rounding_relative_overhead": float(
            scenario["integer_shot_rounding_relative_overhead"]
        ),
        "axes": axis_rows,
    }


def finalize_wp04_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP04",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp04_artifact(payload)
    return payload


def validate_wp04_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported research-direction ablation schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "WP04":
        raise ValueError("Unsupported research-direction ablation method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP04 content_fingerprint mismatch.")
    final_cost_claim = payload.get("scope", {}).get(
        "final_total_cost_evaluation_performed"
    )
    if final_cost_claim is not False:
        raise ValueError("WP04 cannot claim a final total-cost evaluation.")
    if payload.get("scope", {}).get("decision_grade") is not False:
        raise ValueError("WP04 must remain non-decision-grade.")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("WP04 scenarios are missing.")
    if payload.get("configuration", {}).get("factorial_cell_count") != len(scenarios):
        raise ValueError("WP04 factorial-cell count mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP04 overall status does not match checks.")


def write_wp04_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_wp04_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

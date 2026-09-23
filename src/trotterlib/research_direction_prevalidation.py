"""WP00/WP02/WP01-S helpers for the research-direction prevalidation gate.

The artifacts produced here are deliberately screening artifacts.  They bind
all reused evidence to one Hamiltonian snapshot, audit the required RPE
horizon, and compare a small candidate set with a short-q affine compiled-cost
model.  They do not claim a final resource estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_partial_s2 import DFPartialS2Preparation
from .rpe_resource_accounting import (
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundSpecification,
    evaluate_rpe_round_candidate,
)
from .rpe_target_round_horizon_validation import required_rpe_round_horizon
from .rte import finite_rte_distribution


SCHEMA_VERSION = "research_direction_prevalidation_v1"
SCHEDULE_METHOD = "analytic_component_application_screen_v1"


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


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize_artifact(
    *,
    stage: str,
    body: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        **dict(body),
        "provenance": dict(provenance or {}),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported research-direction prevalidation schema.")
    if payload.get("stage") not in ("WP00", "WP02", "WP01-S"):
        raise ValueError("Unsupported research-direction prevalidation stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("Prevalidation content_fingerprint mismatch.")
    if payload["stage"] == "WP01-S":
        scope = payload.get("scope", {})
        if scope.get("final_total_cost_evaluation_performed") is not False:
            raise ValueError("WP01-S cannot claim final total-cost evaluation.")
        if scope.get("decision_grade") is not False:
            raise ValueError("WP01-S must remain non-decision-grade.")


def write_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_horizon_audit(
    *,
    precision_scenarios: Sequence[tuple[str, float]],
    delta_values: Sequence[float],
    beta_rpe: float,
    beta_pf_budget: float,
    pf_coefficient: float,
    direct_wrapper_q_max: int,
    existing_schedule_precision: float,
    existing_schedule_deltas: Sequence[float],
) -> list[dict[str, Any]]:
    """Build the precision/delta horizon and empirical PF coverage matrix."""
    schedule_deltas = {float(value) for value in existing_schedule_deltas}
    rows: list[dict[str, Any]] = []
    for label, epsilon in precision_scenarios:
        for delta in delta_values:
            horizon = required_rpe_round_horizon(
                target_energy_precision=float(epsilon),
                beta_rpe=float(beta_rpe),
                delta_time=float(delta),
            )
            q_max = int(horizon["q_max"])
            pf_phase = float(pf_coefficient * q_max * float(delta) ** 3)
            exact_existing_schedule = bool(
                math.isclose(float(epsilon), float(existing_schedule_precision))
                and float(delta) in schedule_deltas
            )
            rows.append(
                {
                    "precision_label": label,
                    "target_energy_precision_ha": float(epsilon),
                    "delta_time": float(delta),
                    **horizon,
                    "empirical_pf_phase_proxy_at_q_max": pf_phase,
                    "beta_pf_budget": float(beta_pf_budget),
                    "empirical_pf_screen_pass": (
                        pf_phase <= float(beta_pf_budget) + 1e-15
                    ),
                    "direct_full_wrapper_q_max": int(direct_wrapper_q_max),
                    "q_max_directly_compiled": q_max <= direct_wrapper_q_max,
                    "existing_round_schedule_exact_condition_match": (
                        exact_existing_schedule
                    ),
                    "coverage_status": (
                        "existing_schedule_and_matrix_checks_reused"
                        if exact_existing_schedule
                        else "horizon_only_requires_new_schedule_or_reallocation"
                    ),
                }
            )
    return rows


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


def select_analytic_round_schedule(
    preparation: DFPartialS2Preparation,
    *,
    delta_time: float,
    target_energy_precision: float,
    pf_coefficient: float,
    beta_rpe: float = 0.4,
    beta_pf_budget: float = 0.02,
    beta_rte_budget: float = 0.02,
    beta_stat_budget: float = 0.36,
    alpha_total: float = 0.05,
    rte_step_values: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    finite_taylor_orders: Sequence[int] = (0, 2, 4, 6, 8),
    rte_seed: int = 20260818,
) -> dict[str, Any]:
    """Select a provisional feasible schedule with the existing cheap proxy."""
    horizon = required_rpe_round_horizon(
        target_energy_precision=target_energy_precision,
        beta_rpe=beta_rpe,
        delta_time=delta_time,
    )
    maximum_round = int(horizon["maximum_round_index_M"])
    alpha_axis = float(alpha_total / (2 * (maximum_round + 1)))
    allocation = RPEErrorAllocation(
        beta_pf_budget,
        beta_rte_budget,
        beta_stat_budget,
        alpha_axis,
        alpha_axis,
    )
    pf_model = RPEPFErrorModel(
        float(pf_coefficient),
        "same_snapshot_paper_d6_empirical_coefficient",
        False,
    )
    sampling_policy = RPEHadamardSamplingPolicy(
        rte_trajectory_mode="fresh_iid_per_hadamard_shot",
        independent_bounded_outcomes_within_each_round_axis=True,
    )
    grid = (
        ((0, 0),)
        if preparation.is_deterministic_only
        else tuple(
            (int(r_m), int(k_m))
            for r_m in rte_step_values
            for k_m in finite_taylor_orders
        )
    )
    normalized_r_grid = tuple(sorted({pair[0] for pair in grid}))
    normalized_k_grid = tuple(sorted({pair[1] for pair in grid}))
    rounds: list[dict[str, Any]] = []
    failed_evaluation_count = 0
    for round_index in range(maximum_round + 1):
        specification = RPERoundSpecification(round_index, delta_time)
        feasible: list[tuple[float, int, int, Any, float]] = []
        for r_m, k_m in grid:
            try:
                candidate = evaluate_rpe_round_candidate(
                    preparation,
                    specification,
                    allocation,
                    pf_model,
                    beta_rpe=beta_rpe,
                    rte_steps_per_occurrence=r_m,
                    finite_taylor_order=k_m,
                    cost_metric="rz_count",
                    rte_seed=rte_seed,
                    hadamard_sampling_policy=sampling_policy,
                )
            except (OverflowError, ValueError, ZeroDivisionError):
                failed_evaluation_count += 1
                continue
            if not candidate.feasible:
                continue
            total_shots = int(candidate.cosine_shots + candidate.sine_shots)
            expected_applications = (
                0.0
                if preparation.is_deterministic_only
                else _expected_component_applications(candidate.tau_m, k_m)
            )
            per_shot_proxy = float(
                candidate.q_m * r_m * expected_applications
            )
            objective = float(total_shots * per_shot_proxy)
            feasible.append(
                (objective, r_m, k_m, candidate, expected_applications)
            )
        if not feasible:
            return {
                "method": SCHEDULE_METHOD,
                **horizon,
                "delta_time": float(delta_time),
                "all_rounds_feasible": False,
                "failed_round_index": round_index,
                "failed_evaluation_count": failed_evaluation_count,
                "rounds": rounds,
            }
        objective, r_m, k_m, selected, expected_applications = min(
            feasible, key=lambda item: item[:3]
        )
        rounds.append(
            {
                "round_index": round_index,
                "q_m": selected.q_m,
                "r_m": r_m,
                "K_m": k_m,
                "tau_m": selected.tau_m,
                "attenuation": selected.attenuation,
                "conservative_radius_lower_bound": (
                    selected.rho_observed_lower_bound
                ),
                "finite_rte_phase_bound": selected.beta_rte,
                "empirical_pf_phase_proxy": selected.beta_pf,
                "cosine_shots": selected.cosine_shots,
                "sine_shots": selected.sine_shots,
                "total_axis_shots": (
                    selected.cosine_shots + selected.sine_shots
                ),
                "expected_component_applications_per_short_step": (
                    expected_applications
                ),
                "shot_weighted_randomized_component_application_proxy": (
                    objective
                ),
            }
        )
    return {
        "method": SCHEDULE_METHOD,
        **horizon,
        "delta_time": float(delta_time),
        "all_rounds_feasible": True,
        "failed_evaluation_count": failed_evaluation_count,
        "rte_step_values": list(normalized_r_grid),
        "finite_taylor_orders": list(normalized_k_grid),
        "rounds": rounds,
        "total_shots": sum(int(row["total_axis_shots"]) for row in rounds),
        "minimum_attenuation": min(float(row["attenuation"]) for row in rounds),
        "minimum_conservative_radius_lower_bound": min(
            float(row["conservative_radius_lower_bound"]) for row in rounds
        ),
        "selected_r_k_pairs": sorted(
            {f"r{row['r_m']}_k{row['K_m']}" for row in rounds}
        ),
        "maximum_selected_r": max(int(row["r_m"]) for row in rounds),
        "maximum_selected_K": max(int(row["K_m"]) for row in rounds),
        "selected_r_grid_boundary_hit": (
            not preparation.is_deterministic_only
            and max(int(row["r_m"]) for row in rounds) == max(normalized_r_grid)
        ),
        "selected_K_grid_boundary_hit": (
            not preparation.is_deterministic_only
            and max(int(row["K_m"]) for row in rounds) == max(normalized_k_grid)
        ),
        "total_shot_weighted_randomized_component_application_proxy": (
            math.fsum(
                float(row["shot_weighted_randomized_component_application_proxy"])
                for row in rounds
            )
        ),
    }


def affine_prediction_with_standard_error(
    *,
    q_m: int,
    q1_mean: float,
    q2_mean: float,
    q1_standard_error: float,
    q2_standard_error: float,
) -> tuple[float, float]:
    """Predict C(q) from q=1,2 and propagate independent calibration SEs."""
    q_value = int(q_m)
    if q_value < 1 or q_value & (q_value - 1):
        raise ValueError("q_m must be a positive power of two.")
    coefficient_q1 = 2.0 - q_value
    coefficient_q2 = q_value - 1.0
    prediction = coefficient_q1 * float(q1_mean) + coefficient_q2 * float(q2_mean)
    standard_error = math.hypot(
        coefficient_q1 * float(q1_standard_error),
        coefficient_q2 * float(q2_standard_error),
    )
    return float(prediction), float(standard_error)

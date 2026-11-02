"""Target-precision RPE round-horizon and fixed-schedule feasibility audit.

This validation answers two questions that are deliberately separate:

1. How many RPE rounds are required by ``beta_RPE/(q_max*delta) <= epsilon_E``?
2. Can the previously validated fixed H4 schedule (delta=0.1, r=4, K=2)
   simply be extended to that q value?

The second question is evaluated with the small-system matrix reference only.
No q>8 circuit is compiled and no final total cost is evaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian, PhysicalSector
from .finite_rte_signal_validation import validate_finite_rte_signals
from .rpe_four_round_phase_validation import (
    validate_rpe_four_round_phase_payload,
)


SCHEMA_VERSION = "rpe_target_round_horizon_validation_v1"
METHOD = "target_precision_horizon_and_fixed_schedule_matrix_diagnostic_v1"


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


def required_rpe_round_horizon(
    *,
    target_energy_precision: float,
    beta_rpe: float,
    delta_time: float,
) -> dict[str, Any]:
    """Return the minimal power-of-two RPE horizon for one target precision."""
    epsilon = float(target_energy_precision)
    beta = float(beta_rpe)
    delta = float(delta_time)
    if not (math.isfinite(epsilon) and epsilon > 0.0):
        raise ValueError("target_energy_precision must be positive and finite.")
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError("beta_rpe must be positive and finite.")
    if not (math.isfinite(delta) and delta > 0.0):
        raise ValueError("delta_time must be positive and finite.")

    required_q_continuous = beta / (delta * epsilon)
    maximum_round_index = max(
        0,
        int(math.ceil(math.log2(required_q_continuous) - 1e-14)),
    )
    q_max = 1 << maximum_round_index
    # Guard exact-power boundaries against floating-point roundoff.
    while beta / (q_max * delta) > epsilon:
        maximum_round_index += 1
        q_max <<= 1
    while (
        maximum_round_index > 0
        and beta / ((q_max >> 1) * delta) <= epsilon
    ):
        maximum_round_index -= 1
        q_max >>= 1

    achieved = beta / (q_max * delta)
    previous = None if q_max == 1 else beta / ((q_max >> 1) * delta)
    return {
        "target_energy_precision_ha": epsilon,
        "required_q_continuous": required_q_continuous,
        "maximum_round_index_M": maximum_round_index,
        "round_count_M_plus_one": maximum_round_index + 1,
        "q_max": q_max,
        "achieved_energy_resolution_ha": achieved,
        "previous_q_energy_resolution_ha": previous,
        "target_met": achieved <= epsilon,
        "previous_power_fails": previous is None or previous > epsilon,
        "minimal_power_of_two_horizon": bool(
            achieved <= epsilon and (previous is None or previous > epsilon)
        ),
    }


def _physical_result(point: Mapping[str, Any]) -> Mapping[str, Any]:
    for item in point["state_results"]:
        if item["state_label"] == "physical_df_ground_state":
            return item
    raise ValueError("physical_df_ground_state result is missing.")


def validate_rpe_target_round_horizon(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    four_round_payload: Mapping[str, Any],
    *,
    target_energy_precision: float,
    chemical_accuracy: float,
    illustrative_energy_precision: float = 0.50,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit the required q range and the old fixed schedule at that horizon."""
    started = time.perf_counter()
    validate_rpe_four_round_phase_payload(four_round_payload)
    config = four_round_payload["configuration"]
    system = four_round_payload["system"]
    q_validated = tuple(int(value) for value in config["q_values"])
    if q_validated != (1, 2, 4, 8):
        raise ValueError("The source validation must cover q=(1,2,4,8).")

    beta_rpe = float(config["beta_rpe"])
    delta_time = float(config["delta_time"])
    scenarios = [
        (
            "illustrative_0p50_ha",
            "dimensionless_example_from_supplement_not_adopted_target",
            float(illustrative_energy_precision),
        ),
        (
            "chemical_accuracy",
            "chemical_accuracy_sensitivity_comparison",
            float(chemical_accuracy),
        ),
        (
            "provisional_config_target_ca_over_10",
            "existing_config_TARGET_ERROR_used_as_provisional_main_condition",
            float(target_energy_precision),
        ),
    ]
    horizon_results = []
    for label, role, epsilon in scenarios:
        horizon_results.append(
            {
                "label": label,
                "role": role,
                **required_rpe_round_horizon(
                    target_energy_precision=epsilon,
                    beta_rpe=beta_rpe,
                    delta_time=delta_time,
                ),
            }
        )
    target_horizon = horizon_results[-1]
    q_diagnostics = tuple(
        sorted({max(q_validated), *(item["q_max"] for item in horizon_results[1:])})
    )

    signal_payload = validate_finite_rte_signals(
        hamiltonian,
        sector,
        ld=int(system["ld"]),
        delta_time=delta_time,
        q_values=q_diagnostics,
        rte_step_values=(int(config["rte_steps_per_occurrence"]),),
        finite_taylor_orders=(int(config["finite_taylor_order"]),),
        beta_rpe=beta_rpe,
        beta_pf_budget=float(config["beta_pf_budget"]),
        beta_rte_budget=float(config["beta_rte_budget"]),
        beta_stat_budget=float(config["beta_stat_budget"]),
        alpha_total=float(config["alpha_total"]),
    )

    diagnostics: list[dict[str, Any]] = []
    for point in signal_payload["points"]:
        physical = _physical_result(point)
        fixed_schedule_accepted = bool(
            physical["pf_phase_error_within_provisional_budget"]
            and physical["finite_rte_phase_bound_within_provisional_budget"]
            and physical["conservative_radius_lower_bound"] > 0.0
        )
        diagnostics.append(
            {
                "q_m": int(point["q_m"]),
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
                "pf_phase_error": float(physical["pf_vs_exact_phase_error"]),
                "pf_phase_budget": float(config["beta_pf_budget"]),
                "pf_phase_budget_pass": bool(
                    physical["pf_phase_error_within_provisional_budget"]
                ),
                "finite_rte_phase_error_bound": float(
                    physical["finite_rte_phase_error_bound"]
                ),
                "finite_rte_phase_budget": float(config["beta_rte_budget"]),
                "finite_rte_phase_budget_pass": bool(
                    physical["finite_rte_phase_bound_within_provisional_budget"]
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
                "fixed_schedule_accepted": fixed_schedule_accepted,
            }
        )

    diagnostic_by_q = {item["q_m"]: item for item in diagnostics}
    source_q8 = four_round_payload["physical_signals_and_exact_probabilities"][
        "rounds"
    ][-1]
    target_diagnostic = diagnostic_by_q[int(target_horizon["q_max"])]
    q8_diagnostic = diagnostic_by_q[8]
    checks = {
        "all_horizons_are_minimal_powers_of_two": all(
            item["minimal_power_of_two_horizon"] for item in horizon_results
        ),
        "illustrative_0p50_condition_reproduces_q8": (
            horizon_results[0]["q_max"] == 8
        ),
        "provisional_target_is_ca_over_10": math.isclose(
            float(target_energy_precision),
            float(chemical_accuracy) / 10.0,
            rel_tol=1e-14,
            abs_tol=0.0,
        ),
        "required_target_q_exceeds_current_q8_validation": (
            int(target_horizon["q_max"]) > max(q_validated)
        ),
        "q8_matrix_signal_matches_prior_physical_result": math.isclose(
            q8_diagnostic["observed_attenuated_radius"],
            float(source_q8["observed_signal_radius"]),
            rel_tol=1e-12,
            abs_tol=1e-14,
        ),
        "all_matrix_operator_bounds_pass": all(
            item["operator_error_bound_pass"] for item in diagnostics
        ),
        "fixed_schedule_rejected_at_provisional_target_q": (
            not target_diagnostic["fixed_schedule_accepted"]
        ),
        "pf_budget_identifies_target_q_rejection": (
            not target_diagnostic["pf_phase_budget_pass"]
        ),
    }
    overall_pass = all(checks.values())
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "scope": {
            "target_precision_round_selection_performed": True,
            "fixed_schedule_large_q_matrix_diagnostic_performed": True,
            "q_greater_than_8_circuit_compilation_performed": False,
            "q_greater_than_8_cost_proxy_validated": False,
            "large_q_fresh_iid_shot_simulation_performed": False,
            "final_total_cost_evaluation_performed": False,
            "target_precision_is_provisional_config_value_not_normative_choice": True,
        },
        "system": dict(system),
        "configuration": {
            "beta_rpe": beta_rpe,
            "delta_time": delta_time,
            "rte_steps_per_occurrence": int(
                config["rte_steps_per_occurrence"]
            ),
            "finite_taylor_order": int(config["finite_taylor_order"]),
            "beta_pf_budget": float(config["beta_pf_budget"]),
            "beta_rte_budget": float(config["beta_rte_budget"]),
            "beta_stat_budget": float(config["beta_stat_budget"]),
            "alpha_total": float(config["alpha_total"]),
            "source_validated_q_values": list(q_validated),
        },
        "round_horizons": horizon_results,
        "fixed_schedule_matrix_diagnostics": diagnostics,
        "source_evidence": {
            "four_round_phase_content_fingerprint": four_round_payload[
                "content_fingerprint"
            ],
            "finite_rte_signal_validation_fingerprint": signal_payload[
                "validation_fingerprint"
            ],
        },
        "summary": {
            "provisional_target_energy_precision_ha": float(
                target_energy_precision
            ),
            "required_maximum_round_index_M": int(
                target_horizon["maximum_round_index_M"]
            ),
            "required_round_count": int(target_horizon["round_count_M_plus_one"]),
            "required_q_max": int(target_horizon["q_max"]),
            "current_validated_q_max": max(q_validated),
            "current_q8_energy_resolution_ha": beta_rpe
            / (max(q_validated) * delta_time),
            "fixed_schedule_target_q_accepted": bool(
                target_diagnostic["fixed_schedule_accepted"]
            ),
            "next_action": (
                "reoptimize_delta_and_round_specific_r_K_before_any_large_q_"
                "compiled_cost_holdout"
            ),
            "checks": checks,
            "overall_pass": overall_pass,
            "interpretation": (
                "round_horizon_determined_but_fixed_q8_schedule_cannot_be_"
                "extrapolated_to_the_provisional_target_horizon"
            ),
        },
        "performance": {"elapsed_seconds": time.perf_counter() - started},
        "provenance": dict(provenance or {}),
    }
    payload["content_fingerprint"] = _fingerprint(payload)
    validate_rpe_target_round_horizon_payload(payload)
    return payload


def validate_rpe_target_round_horizon_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("method") != METHOD:
        raise ValueError("Unsupported RPE target-round-horizon artifact.")
    unsigned = dict(payload)
    fingerprint = unsigned.pop("content_fingerprint", None)
    if fingerprint != _fingerprint(unsigned):
        raise ValueError("RPE target-round-horizon content_fingerprint mismatch.")
    horizons = payload.get("round_horizons", ())
    if len(horizons) != 3:
        raise ValueError("Exactly three precision scenarios are required.")
    for item in horizons:
        expected = required_rpe_round_horizon(
            target_energy_precision=float(item["target_energy_precision_ha"]),
            beta_rpe=float(payload["configuration"]["beta_rpe"]),
            delta_time=float(payload["configuration"]["delta_time"]),
        )
        for key, value in expected.items():
            if item.get(key) != value:
                raise ValueError(f"Round-horizon mismatch for {item['label']}: {key}.")
    checks = payload.get("summary", {}).get("checks", {})
    expected_overall = bool(checks) and all(bool(value) for value in checks.values())
    if payload.get("summary", {}).get("overall_pass") != expected_overall:
        raise ValueError("RPE target-round-horizon overall status mismatch.")
    scope = payload.get("scope", {})
    if scope.get("q_greater_than_8_circuit_compilation_performed") is not False:
        raise ValueError("This validation cannot claim q>8 circuit compilation.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("This validation cannot claim a final total cost.")


def write_rpe_target_round_horizon_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_rpe_target_round_horizon_payload(payload)
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

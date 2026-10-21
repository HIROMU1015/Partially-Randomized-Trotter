"""Physical-signal and branch-reconstruction validation for four RPE rounds.

The validation extends the fixed H4 q=(1,2,4,8) accounting result without
changing its resource allocation.  It computes the physical finite-RTE signal
at q=8, audits one fresh-IID trajectory per Hadamard shot for all four rounds,
and applies a sequential nearest-branch phase reconstruction to analytic,
explicit-shot, and marginal Monte Carlo data.

This is a fixed-condition local validation.  It does not choose the number of
rounds required by a target energy precision or evaluate a final optimized
total cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .df_hamiltonian import DFHamiltonian, PhysicalSector
from .df_partial_randomized_pf import df_hamiltonian_hash
from .finite_rte_signal_validation import validate_finite_rte_signals
from .rpe_four_round_accounting_validation import (
    validate_rpe_four_round_accounting_payload,
)
from .rpe_hadamard_failure_validation import (
    _axis_distribution,
    _clopper_pearson_upper,
    _complex_from_payload,
    _complex_payload,
    _explicit_fresh_iid_validation,
    _round_exact_phase_probability,
    _strip_axis_arrays,
)


SCHEMA_VERSION = "rpe_four_round_phase_validation_v1"
METHOD = "physical_q8_fresh_iid_and_sequential_branch_reconstruction_v1"


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


def _wrap_phase(value: float | np.ndarray) -> float | np.ndarray:
    wrapped = (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi
    if np.ndim(value) == 0:
        return float(wrapped)
    return wrapped


def _phase_distance(left: float | np.ndarray, right: float) -> float | np.ndarray:
    distance = np.abs(np.angle(np.exp(1j * (np.asarray(left) - float(right)))))
    if np.ndim(left) == 0:
        return float(distance)
    return distance


def reconstruct_rpe_phase(
    measured_round_phases: Sequence[float],
    q_values: Sequence[int],
) -> dict[str, Any]:
    """Select the phase branch nearest to the preceding RPE estimate."""
    phases = tuple(float(value) for value in measured_round_phases)
    q_grid = tuple(int(value) for value in q_values)
    if len(phases) != len(q_grid) or not phases:
        raise ValueError("measured_round_phases and q_values must have equal nonzero length.")
    if q_grid[0] != 1 or any(value <= 0 or value & (value - 1) for value in q_grid):
        raise ValueError("q_values must start at one and contain positive powers of two.")
    if any(right != 2 * left for left, right in zip(q_grid, q_grid[1:])):
        raise ValueError("q_values must be consecutive powers of two.")

    estimate = float(_wrap_phase(phases[0]))
    rounds = [
        {
            "q_m": q_grid[0],
            "measured_round_phase": float(_wrap_phase(phases[0])),
            "selected_branch": 0,
            "base_phase_estimate": estimate,
            "distance_from_previous": 0.0,
        }
    ]
    for phase, q_m in zip(phases[1:], q_grid[1:]):
        principal = float(_wrap_phase(phase))
        candidates = np.asarray(
            [
                _wrap_phase((principal + 2.0 * math.pi * branch) / q_m)
                for branch in range(q_m)
            ],
            dtype=np.float64,
        )
        distances = np.asarray(_phase_distance(candidates, estimate), dtype=np.float64)
        selected = int(np.argmin(distances))
        new_estimate = float(candidates[selected])
        rounds.append(
            {
                "q_m": q_m,
                "measured_round_phase": principal,
                "selected_branch": selected,
                "base_phase_estimate": new_estimate,
                "distance_from_previous": float(distances[selected]),
                "next_smallest_branch_distance": float(
                    np.partition(distances, 1)[1]
                ),
            }
        )
        estimate = new_estimate
    return {"rounds": rounds, "final_phase_estimate": estimate}


def _vectorized_reconstruction(
    measured_round_phases: Sequence[np.ndarray],
    q_values: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    q_grid = tuple(int(value) for value in q_values)
    phases = tuple(np.asarray(value, dtype=np.float64) for value in measured_round_phases)
    if not phases or len(phases) != len(q_grid):
        raise ValueError("Phase arrays and q_values must have equal nonzero length.")
    if any(item.shape != phases[0].shape for item in phases):
        raise ValueError("Every measured phase array must have the same shape.")
    estimate = np.asarray(_wrap_phase(phases[0]), dtype=np.float64)
    selected_branches = [np.zeros(estimate.shape, dtype=np.int64)]
    ambiguous = np.zeros(estimate.shape, dtype=bool)
    for phase, q_m in zip(phases[1:], q_grid[1:]):
        principal = np.asarray(_wrap_phase(phase), dtype=np.float64)
        branches = np.arange(q_m, dtype=np.float64)
        candidates = _wrap_phase(
            (principal[..., None] + 2.0 * math.pi * branches) / q_m
        )
        distances = np.abs(
            np.angle(np.exp(1j * (np.asarray(candidates) - estimate[..., None])))
        )
        order = np.argsort(distances, axis=-1)
        selected = order[..., 0]
        runner_up = order[..., 1]
        best_distance = np.take_along_axis(
            distances, selected[..., None], axis=-1
        )[..., 0]
        next_distance = np.take_along_axis(
            distances, runner_up[..., None], axis=-1
        )[..., 0]
        ambiguous |= np.isclose(best_distance, next_distance, rtol=0.0, atol=1e-12)
        estimate = np.take_along_axis(
            np.asarray(candidates), selected[..., None], axis=-1
        )[..., 0]
        selected_branches.append(selected.astype(np.int64))
    return estimate, selected_branches, ambiguous


def _oracle_branch(
    measured_phase: np.ndarray,
    q_m: int,
    reference_phase: float,
) -> np.ndarray:
    principal = np.asarray(_wrap_phase(measured_phase), dtype=np.float64)
    candidates = _wrap_phase(
        (
            principal[..., None]
            + 2.0 * math.pi * np.arange(q_m, dtype=np.float64)
        )
        / q_m
    )
    distances = np.abs(
        np.angle(np.exp(1j * (np.asarray(candidates) - reference_phase)))
    )
    return np.argmin(distances, axis=-1).astype(np.int64)


def _marginal_branch_monte_carlo(
    round_inputs: Sequence[Mapping[str, Any]],
    *,
    reference_phase: float,
    beta_rpe: float,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed))
    measured_phases: list[np.ndarray] = []
    q_values: list[int] = []
    zero_signal = np.zeros(repetitions, dtype=bool)
    coordinate_failure = np.zeros(repetitions, dtype=bool)
    for item in round_inputs:
        q_m = int(item["q_m"])
        shots = int(item["shots"])
        signal = complex(item["true_signal"])
        cosine_plus = rng.binomial(shots, (1.0 + signal.real) / 2.0, repetitions)
        sine_plus = rng.binomial(shots, (1.0 + signal.imag) / 2.0, repetitions)
        cosine = 2.0 * cosine_plus / shots - 1.0
        sine = 2.0 * sine_plus / shots - 1.0
        estimates = cosine + 1j * sine
        zero_signal |= np.abs(estimates) == 0.0
        coordinate_failure |= (
            np.abs(cosine - signal.real) >= float(item["epsilon_coordinate"])
        ) | (
            np.abs(sine - signal.imag) >= float(item["epsilon_coordinate"])
        )
        measured_phases.append(np.angle(estimates))
        q_values.append(q_m)

    final_estimate, selected, ambiguous = _vectorized_reconstruction(
        measured_phases, q_values
    )
    branch_failure = np.zeros(repetitions, dtype=bool)
    per_round_branch_failures: list[dict[str, Any]] = []
    for index, (phase, q_m) in enumerate(zip(measured_phases, q_values)):
        oracle = _oracle_branch(phase, q_m, reference_phase)
        failures = selected[index] != oracle
        branch_failure |= failures
        per_round_branch_failures.append(
            {
                "q_m": q_m,
                "failure_count": int(np.count_nonzero(failures)),
                "failure_rate": float(np.mean(failures)),
            }
        )
    final_error = np.asarray(_phase_distance(final_estimate, reference_phase))
    final_tolerance = float(beta_rpe / q_values[-1])
    final_failure = zero_signal | ambiguous | (final_error > final_tolerance)
    count = int(np.count_nonzero(final_failure))
    branch_count = int(np.count_nonzero(branch_failure))
    return {
        "repetitions": repetitions,
        "seed": seed,
        "reference_phase": reference_phase,
        "final_phase_tolerance": final_tolerance,
        "final_failure_count": count,
        "final_failure_rate": float(count / repetitions),
        "final_failure_one_sided_95_percent_upper": _clopper_pearson_upper(
            count, repetitions, confidence=0.95
        ),
        "branch_failure_count": branch_count,
        "branch_failure_rate": float(branch_count / repetitions),
        "coordinate_failure_count": int(np.count_nonzero(coordinate_failure)),
        "zero_signal_count": int(np.count_nonzero(zero_signal)),
        "ambiguous_branch_count": int(np.count_nonzero(ambiguous)),
        "maximum_final_phase_error": float(np.max(final_error)),
        "per_round_branch_failures": per_round_branch_failures,
    }


def _explicit_reconstruction(
    explicit: Mapping[str, Any],
    q_values: Sequence[int],
    *,
    reference_phase: float,
    beta_rpe: float,
) -> dict[str, Any]:
    by_q_axis = {
        (int(item["q_m"]), str(item["axis"])): item
        for item in explicit["axis_results"]
    }
    signals = []
    for q_m in q_values:
        signals.append(
            complex(
                float(by_q_axis[(q_m, "cosine")]["measurement_sample_mean"]),
                float(by_q_axis[(q_m, "sine")]["measurement_sample_mean"]),
            )
        )
    reconstruction = reconstruct_rpe_phase(
        [float(np.angle(value)) for value in signals], q_values
    )
    error = float(
        _phase_distance(reconstruction["final_phase_estimate"], reference_phase)
    )
    return {
        "measured_complex_coordinates": [
            {"q_m": q_m, "value": _complex_payload(value)}
            for q_m, value in zip(q_values, signals)
        ],
        "reconstruction": reconstruction,
        "reference_phase": reference_phase,
        "final_phase_error": error,
        "final_phase_tolerance": float(beta_rpe / q_values[-1]),
        "within_final_phase_tolerance": bool(
            error <= beta_rpe / q_values[-1]
        ),
    }


def validate_rpe_four_round_phase_reconstruction(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    accounting_payload: Mapping[str, Any],
    *,
    marginal_repetitions: int = 100_000,
    marginal_seed: int = 20260920,
    explicit_trajectory_seed: int = 20260921,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate q=8 physical signal and four-round branch reconstruction."""
    started = time.perf_counter()
    validate_rpe_four_round_accounting_payload(accounting_payload)
    if not accounting_payload["summary"]["overall_pass"]:
        raise ValueError("The source four-round accounting did not pass.")
    system = accounting_payload["system"]
    config = accounting_payload["configuration"]
    rounds = sorted(accounting_payload["rounds"], key=lambda item: item["q_m"])
    q_values = tuple(int(item["q_m"]) for item in rounds)
    if q_values != (1, 2, 4, 8):
        raise ValueError("This validation requires q=(1,2,4,8).")
    if df_hamiltonian_hash(hamiltonian) != system["hamiltonian_hash"]:
        raise ValueError("Hamiltonian differs from the accounting source.")
    if hamiltonian.n_qubits != system["num_system_qubits"]:
        raise ValueError("System size differs from the accounting source.")
    if hamiltonian.n_blocks != system["df_rank"]:
        raise ValueError("DF rank differs from the accounting source.")
    if any(item["cosine_shots"] != item["sine_shots"] for item in rounds):
        raise ValueError("The explicit sampler currently requires equal axis shots.")
    repetitions = int(marginal_repetitions)
    if repetitions < 1:
        raise ValueError("marginal_repetitions must be positive.")

    beta_pf = float(config["beta_pf_budget"])
    beta_rte = float(config["beta_rte_budget"])
    beta_stat = float(config["beta_stat_budget"])
    beta_rpe = float(config["beta_rpe"])
    alpha_total = float(accounting_payload["limited_aggregation"]["total_alpha_budget"])
    if beta_pf + beta_rte + beta_stat > beta_rpe + 1e-15:
        raise ValueError("Phase budgets exceed beta_rpe.")

    signal_payload = validate_finite_rte_signals(
        hamiltonian,
        sector,
        ld=int(system["ld"]),
        delta_time=float(config["delta_time"]),
        q_values=q_values,
        rte_step_values=(int(config["rte_steps_per_occurrence"]),),
        finite_taylor_orders=(int(config["finite_taylor_order"]),),
        beta_rpe=beta_rpe,
        beta_pf_budget=beta_pf,
        beta_rte_budget=beta_rte,
        beta_stat_budget=beta_stat,
        alpha_total=alpha_total,
        seed=explicit_trajectory_seed,
        provenance={
            "parent_validation": METHOD,
            "outer_provenance": dict(provenance or {}),
        },
    )
    if not signal_payload["summary"]["overall_pass"]:
        raise ValueError("The four-round finite-RTE signal validation failed.")

    point_by_q = {int(item["q_m"]): item for item in signal_payload["points"]}
    round_inputs: list[dict[str, Any]] = []
    exact_rounds: list[dict[str, Any]] = []
    analytic_phases: list[float] = []
    exact_phases: list[float] = []
    for plan in rounds:
        q_m = int(plan["q_m"])
        point = point_by_q[q_m]
        physical = next(
            item
            for item in point["state_results"]
            if item["state_label"] == "physical_df_ground_state"
        )
        signal = _complex_from_payload(physical["attenuated_event_mean_signal"])
        exact_signal = _complex_from_payload(physical["exact_signal"])
        shots = int(plan["cosine_shots"])
        radius_lower = float(physical["conservative_radius_lower_bound"])
        epsilon = float(radius_lower * math.sin(beta_stat) / math.sqrt(2.0))
        cosine = _axis_distribution(signal.real, shots, epsilon)
        sine = _axis_distribution(signal.imag, shots, epsilon)
        phase_probability, implication_violations = _round_exact_phase_probability(
            cosine, sine, signal, beta_stat
        )
        round_inputs.append(
            {
                "q_m": q_m,
                "shots": shots,
                "epsilon_coordinate": epsilon,
                "beta_stat_budget": beta_stat,
                "true_signal": signal,
                "cosine_exact_failure_probability": cosine[
                    "exact_coordinate_failure_probability"
                ],
                "sine_exact_failure_probability": sine[
                    "exact_coordinate_failure_probability"
                ],
                "exact_phase_failure_probability": phase_probability,
            }
        )
        analytic_phases.append(float(np.angle(signal)))
        exact_phases.append(float(np.angle(exact_signal)))
        systematic_phase_error = float(
            abs(np.angle(signal * np.conj(exact_signal)))
        )
        exact_rounds.append(
            {
                "q_m": q_m,
                "shots_per_axis": shots,
                "alpha_cosine": float(plan["alpha_cosine"]),
                "alpha_sine": float(plan["alpha_sine"]),
                "exact_signal": _complex_payload(exact_signal),
                "attenuated_event_mean_signal": _complex_payload(signal),
                "observed_signal_radius": float(abs(signal)),
                "conservative_radius_lower_bound": radius_lower,
                "epsilon_coordinate": epsilon,
                "systematic_phase_error_from_exact_signal": systematic_phase_error,
                "systematic_phase_budget": beta_pf + beta_rte,
                "cosine": _strip_axis_arrays(cosine),
                "sine": _strip_axis_arrays(sine),
                "exact_statistical_phase_failure_probability": phase_probability,
                "coordinate_success_but_statistical_phase_failure_grid_points": (
                    implication_violations
                ),
            }
        )

    reference_phase = exact_phases[0]
    analytic_reconstruction = reconstruct_rpe_phase(analytic_phases, q_values)
    analytic_error = float(
        _phase_distance(
            analytic_reconstruction["final_phase_estimate"], reference_phase
        )
    )
    combined_coordinate_failure = float(
        1.0
        - math.prod(
            (1.0 - item["cosine_exact_failure_probability"])
            * (1.0 - item["sine_exact_failure_probability"])
            for item in round_inputs
        )
    )
    combined_statistical_phase_failure = float(
        1.0
        - math.prod(
            1.0 - item["exact_phase_failure_probability"]
            for item in round_inputs
        )
    )

    explicit = _explicit_fresh_iid_validation(
        hamiltonian,
        sector,
        ld=int(system["ld"]),
        delta_time=float(config["delta_time"]),
        q_values=q_values,
        rte_steps=int(config["rte_steps_per_occurrence"]),
        finite_taylor_order=int(config["finite_taylor_order"]),
        round_inputs=round_inputs,
        seed=explicit_trajectory_seed,
    )
    explicit_reconstruction = _explicit_reconstruction(
        explicit,
        q_values,
        reference_phase=reference_phase,
        beta_rpe=beta_rpe,
    )
    marginal = _marginal_branch_monte_carlo(
        round_inputs,
        reference_phase=reference_phase,
        beta_rpe=beta_rpe,
        repetitions=repetitions,
        seed=marginal_seed,
    )

    exact_checks = {
        "all_axis_coordinate_failures_within_allocated_alpha": all(
            item[axis]["exact_coordinate_failure_probability"]
            <= item[f"alpha_{axis}"]
            for item in exact_rounds
            for axis in ("cosine", "sine")
        ),
        "all_round_statistical_phase_failures_within_axis_union": all(
            item["exact_statistical_phase_failure_probability"]
            <= item["alpha_cosine"] + item["alpha_sine"]
            for item in exact_rounds
        ),
        "all_systematic_phase_errors_within_pf_plus_rte_budget": all(
            item["systematic_phase_error_from_exact_signal"]
            <= item["systematic_phase_budget"] + 1e-12
            for item in exact_rounds
        ),
        "coordinate_success_implies_statistical_phase_success": all(
            item[
                "coordinate_success_but_statistical_phase_failure_grid_points"
            ]
            == 0
            for item in exact_rounds
        ),
        "combined_coordinate_failure_within_total_alpha": (
            combined_coordinate_failure <= alpha_total
        ),
        "combined_statistical_phase_failure_within_total_alpha": (
            combined_statistical_phase_failure <= alpha_total
        ),
    }
    explicit_checks = {
        "fresh_seed_per_hadamard_shot": bool(
            explicit["all_trajectory_seeds_unique_across_axes_and_rounds"]
        ),
        "trajectory_means_match_analytic_signal_within_five_standard_errors": (
            explicit["maximum_trajectory_mean_absolute_z"] <= 5.0
        ),
        "conditional_measurements_match_marginal_binomial_intervals": bool(
            explicit["all_measurement_counts_inside_marginal_99p9_intervals"]
        ),
    }
    reconstruction_checks = {
        "analytic_four_round_reconstruction_within_beta_over_qmax": (
            analytic_error <= beta_rpe / q_values[-1]
        ),
        "marginal_branch_failure_upper_within_total_alpha": (
            marginal["final_failure_one_sided_95_percent_upper"] <= alpha_total
        ),
        "no_ambiguous_branches_in_marginal_replications": (
            marginal["ambiguous_branch_count"] == 0
        ),
        "explicit_four_round_reconstruction_completed": bool(
            explicit_reconstruction["reconstruction"]["rounds"]
        ),
    }
    overall = bool(
        all(exact_checks.values())
        and all(explicit_checks.values())
        and all(reconstruction_checks.values())
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "scope": {
            "description": "fixed_h4_q1_q2_q4_q8_physical_signal_and_branch_validation",
            "q8_physical_signal_evaluated": True,
            "fresh_iid_trajectory_batch_performed": True,
            "four_round_branch_reconstruction_performed": True,
            "target_precision_round_selection_performed": False,
            "final_total_cost_evaluation_performed": False,
            "backend_execution_performed": False,
            "noise_model_included": False,
        },
        "system": {
            **dict(system),
            "sector_dimension": sector.dimension,
            "sector_n_electrons": sector.n_electrons,
        },
        "configuration": {
            "q_values": list(q_values),
            "delta_time": float(config["delta_time"]),
            "rte_steps_per_occurrence": int(
                config["rte_steps_per_occurrence"]
            ),
            "finite_taylor_order": int(config["finite_taylor_order"]),
            "beta_pf_budget": beta_pf,
            "beta_rte_budget": beta_rte,
            "beta_stat_budget": beta_stat,
            "beta_rpe": beta_rpe,
            "alpha_total": alpha_total,
            "marginal_repetitions": repetitions,
            "marginal_seed": marginal_seed,
            "explicit_trajectory_seed": explicit_trajectory_seed,
        },
        "source_evidence": {
            "four_round_accounting_content_fingerprint": accounting_payload[
                "content_fingerprint"
            ],
            "finite_rte_signal_validation_fingerprint": signal_payload[
                "validation_fingerprint"
            ],
        },
        "physical_signals_and_exact_probabilities": {
            "rounds": exact_rounds,
            "combined_eight_axis_coordinate_failure_probability": (
                combined_coordinate_failure
            ),
            "combined_four_round_statistical_phase_failure_probability": (
                combined_statistical_phase_failure
            ),
            "checks": exact_checks,
        },
        "analytic_branch_reconstruction": {
            "reference_one_step_phase": reference_phase,
            "round_signal_phases": [
                {"q_m": q_m, "phase": phase}
                for q_m, phase in zip(q_values, analytic_phases)
            ],
            "reconstruction": analytic_reconstruction,
            "final_phase_error": analytic_error,
            "final_phase_tolerance": beta_rpe / q_values[-1],
        },
        "marginal_branch_monte_carlo": marginal,
        "explicit_fresh_iid_trajectory_batch": {
            **explicit,
            "checks": explicit_checks,
            "single_batch_branch_reconstruction": explicit_reconstruction,
            "single_batch_success_is_diagnostic_not_acceptance_criterion": True,
        },
        "summary": {
            "all_finite_rte_signal_checks_pass": signal_payload["summary"][
                "overall_pass"
            ],
            "all_exact_probability_checks_pass": all(exact_checks.values()),
            "all_explicit_fresh_iid_checks_pass": all(explicit_checks.values()),
            "all_branch_reconstruction_checks_pass": all(
                reconstruction_checks.values()
            ),
            "reconstruction_checks": reconstruction_checks,
            "overall_pass": overall,
            "interpretation": (
                "fixed_four_round_end_to_end_signal_statistics_and_phase_"
                "reconstruction_not_target_precision_or_final_total_cost"
            ),
        },
        "performance": {"elapsed_seconds": time.perf_counter() - started},
        "provenance": dict(provenance or {}),
    }
    payload["content_fingerprint"] = _fingerprint(payload)
    validate_rpe_four_round_phase_payload(payload)
    return payload


def validate_rpe_four_round_phase_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("method") != METHOD:
        raise ValueError("Unsupported four-round phase-validation artifact.")
    unsigned = dict(payload)
    fingerprint = unsigned.pop("content_fingerprint", None)
    if fingerprint != _fingerprint(unsigned):
        raise ValueError("Four-round phase content_fingerprint mismatch.")
    summary = payload.get("summary", {})
    exact = payload.get("physical_signals_and_exact_probabilities", {}).get(
        "checks", {}
    )
    explicit = payload.get("explicit_fresh_iid_trajectory_batch", {}).get(
        "checks", {}
    )
    reconstruction = summary.get("reconstruction_checks", {})
    expected_exact = bool(exact) and all(bool(value) for value in exact.values())
    expected_explicit = bool(explicit) and all(
        bool(value) for value in explicit.values()
    )
    expected_reconstruction = bool(reconstruction) and all(
        bool(value) for value in reconstruction.values()
    )
    if summary.get("all_exact_probability_checks_pass") != expected_exact:
        raise ValueError("Four-round exact-probability summary mismatch.")
    if summary.get("all_explicit_fresh_iid_checks_pass") != expected_explicit:
        raise ValueError("Four-round explicit-IID summary mismatch.")
    if summary.get("all_branch_reconstruction_checks_pass") != expected_reconstruction:
        raise ValueError("Four-round reconstruction summary mismatch.")
    expected_overall = bool(
        summary.get("all_finite_rte_signal_checks_pass")
        and expected_exact
        and expected_explicit
        and expected_reconstruction
    )
    if summary.get("overall_pass") != expected_overall:
        raise ValueError("Four-round phase-validation overall status mismatch.")
    if payload["scope"].get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("This validation cannot claim a final total cost.")


def write_rpe_four_round_phase_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_rpe_four_round_phase_payload(payload)
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

"""Validate short-round Hadamard shot counts and failure allocations.

The validation has two deliberately separate layers.

1.  The marginal layer uses the analytically averaged finite-RTE signal.  A
    fresh independently randomized RTE trajectory for every quantum shot makes
    each Hadamard outcome an IID Bernoulli variable with that marginal mean.
    Exact binomial probabilities and a vectorized Monte Carlo replication then
    test the Hoeffding shot allocation.
2.  The explicit layer samples one fresh RTE trajectory per Hadamard shot in a
    small physical sector, applies it to the ground state, and samples the
    conditional Hadamard outcome.  This audits the fresh-IID implementation and
    the event-to-signal convention; it is not used to estimate a rare failure
    probability from only one production-sized batch.

This module does not reconstruct an RPE estimate across rounds, execute a
backend, include noise, or evaluate a final total cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import binom

from .df_hamiltonian import DFHamiltonian, PhysicalSector
from .df_partial_randomized_pf import (
    df_hamiltonian_hash,
    split_df_hamiltonian_by_ld,
)
from .df_partial_s2 import (
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from .df_rte_tail import basis_change_unitary
from .finite_rte_signal_validation import (
    _circuit_operator_in_openfermion_sector,
    _explicit_cutoff_tolerance,
    _matrix_in_openfermion_sector,
    _qiskit_to_openfermion_sector_permutation,
    dense_df_operator_in_sector,
    validate_finite_rte_signals,
)
from .rte import (
    RTEEvent,
    make_rte_config,
    require_integer_count,
    sample_rte_events,
)


RPE_HADAMARD_FAILURE_SCHEMA_VERSION = (
    "rpe_hadamard_failure_validation_v1"
)
RPE_HADAMARD_FAILURE_METHOD = (
    "exact_binomial_and_fresh_iid_explicit_trajectory_v1"
)


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


def _complex_from_payload(payload: Mapping[str, Any]) -> complex:
    return complex(float(payload["real"]), float(payload["imag"]))


def _complex_payload(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def _phase_distance(left: complex, right: complex) -> float:
    if abs(left) == 0.0 or abs(right) == 0.0:
        return math.pi
    return float(abs(np.angle(complex(left) * np.conj(complex(right)))))


def _clopper_pearson_interval(
    failures: int,
    trials: int,
    *,
    confidence: float,
) -> tuple[float, float]:
    failures = require_integer_count(failures, name="failures")
    trials = require_integer_count(trials, name="trials", minimum=1)
    if failures > trials:
        raise ValueError("failures cannot exceed trials.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly in (0, 1).")
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0
        if failures == 0
        else float(beta_distribution.ppf(tail, failures, trials - failures + 1))
    )
    upper = (
        1.0
        if failures == trials
        else float(
            beta_distribution.ppf(
                1.0 - tail,
                failures + 1,
                trials - failures,
            )
        )
    )
    return lower, upper


def _clopper_pearson_upper(
    failures: int,
    trials: int,
    *,
    confidence: float,
) -> float:
    failures = require_integer_count(failures, name="failures")
    trials = require_integer_count(trials, name="trials", minimum=1)
    if failures > trials:
        raise ValueError("failures cannot exceed trials.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly in (0, 1).")
    if failures == trials:
        return 1.0
    return float(
        beta_distribution.ppf(
            confidence,
            failures + 1,
            trials - failures,
        )
    )


def _axis_distribution(
    mean: float,
    shots: int,
    epsilon_coordinate: float,
) -> dict[str, Any]:
    if not -1.0 - 1e-12 <= mean <= 1.0 + 1e-12:
        raise ValueError("Hadamard coordinate mean lies outside [-1, 1].")
    probability_plus = float(np.clip((1.0 + mean) / 2.0, 0.0, 1.0))
    plus_counts = np.arange(shots + 1, dtype=np.int64)
    sample_means = 2.0 * plus_counts / shots - 1.0
    probabilities = np.asarray(
        binom.pmf(plus_counts, shots, probability_plus),
        dtype=np.float64,
    )
    failure_mask = np.abs(sample_means - mean) >= epsilon_coordinate
    failure_probability = float(math.fsum(probabilities[failure_mask]))
    return {
        "mean": float(mean),
        "probability_plus_one": probability_plus,
        "shots": int(shots),
        "epsilon_coordinate": float(epsilon_coordinate),
        "exact_coordinate_failure_probability": failure_probability,
        "_sample_means": sample_means,
        "_probabilities": probabilities,
        "_failure_mask": failure_mask,
    }


def _strip_axis_arrays(axis: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in axis.items() if not key.startswith("_")}


def _round_exact_phase_probability(
    cosine: Mapping[str, Any],
    sine: Mapping[str, Any],
    true_signal: complex,
    beta_stat_budget: float,
) -> tuple[float, int]:
    cosine_values = np.asarray(cosine["_sample_means"], dtype=np.float64)
    sine_values = np.asarray(sine["_sample_means"], dtype=np.float64)
    cosine_probabilities = np.asarray(cosine["_probabilities"], dtype=np.float64)
    sine_probabilities = np.asarray(sine["_probabilities"], dtype=np.float64)
    estimates = cosine_values[:, None] + 1j * sine_values[None, :]
    phase_errors = np.abs(np.angle(estimates * np.conj(true_signal)))
    zero_mask = np.abs(estimates) == 0.0
    phase_failures = zero_mask | (phase_errors > beta_stat_budget)
    probability_grid = cosine_probabilities[:, None] * sine_probabilities[None, :]
    probability = float(math.fsum(probability_grid[phase_failures].ravel()))

    coordinate_success = ~np.asarray(cosine["_failure_mask"])[:, None] & (
        ~np.asarray(sine["_failure_mask"])[None, :]
    )
    implication_violations = int(np.count_nonzero(coordinate_success & phase_failures))
    return probability, implication_violations


def _simulate_marginal_experiments(
    round_inputs: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    round_results: list[dict[str, Any]] = []
    combined_coordinate_failure = np.zeros(repetitions, dtype=bool)
    combined_phase_failure = np.zeros(repetitions, dtype=bool)
    for item in round_inputs:
        shots = int(item["shots"])
        epsilon_coordinate = float(item["epsilon_coordinate"])
        beta_stat_budget = float(item["beta_stat_budget"])
        true_signal = complex(item["true_signal"])
        cosine_probability = (1.0 + true_signal.real) / 2.0
        sine_probability = (1.0 + true_signal.imag) / 2.0
        cosine_plus = rng.binomial(shots, cosine_probability, size=repetitions)
        sine_plus = rng.binomial(shots, sine_probability, size=repetitions)
        cosine_estimates = 2.0 * cosine_plus / shots - 1.0
        sine_estimates = 2.0 * sine_plus / shots - 1.0
        cosine_failures = (
            np.abs(cosine_estimates - true_signal.real) >= epsilon_coordinate
        )
        sine_failures = (
            np.abs(sine_estimates - true_signal.imag) >= epsilon_coordinate
        )
        estimates = cosine_estimates + 1j * sine_estimates
        phase_errors = np.abs(np.angle(estimates * np.conj(true_signal)))
        phase_failures = (np.abs(estimates) == 0.0) | (
            phase_errors > beta_stat_budget
        )
        implication_violations = (~cosine_failures & ~sine_failures) & phase_failures
        combined_coordinate_failure |= cosine_failures | sine_failures
        combined_phase_failure |= phase_failures

        axes: dict[str, Any] = {}
        for axis, failures, exact_probability in (
            (
                "cosine",
                cosine_failures,
                item["cosine_exact_failure_probability"],
            ),
            (
                "sine",
                sine_failures,
                item["sine_exact_failure_probability"],
            ),
        ):
            count = int(np.count_nonzero(failures))
            lower, upper = _clopper_pearson_interval(
                count,
                repetitions,
                confidence=confidence,
            )
            axes[axis] = {
                "failure_count": count,
                "failure_rate": float(count / repetitions),
                "two_sided_clopper_pearson_interval": [lower, upper],
                "exact_probability_inside_interval": bool(
                    lower <= exact_probability <= upper
                ),
                "one_sided_95_percent_upper": _clopper_pearson_upper(
                    count,
                    repetitions,
                    confidence=0.95,
                ),
            }
        phase_count = int(np.count_nonzero(phase_failures))
        phase_lower, phase_upper = _clopper_pearson_interval(
            phase_count,
            repetitions,
            confidence=confidence,
        )
        round_results.append(
            {
                "q_m": int(item["q_m"]),
                "axes": axes,
                "phase_failure_count": phase_count,
                "phase_failure_rate": float(phase_count / repetitions),
                "phase_two_sided_clopper_pearson_interval": [
                    phase_lower,
                    phase_upper,
                ],
                "exact_phase_probability_inside_interval": bool(
                    phase_lower
                    <= item["exact_phase_failure_probability"]
                    <= phase_upper
                ),
                "phase_one_sided_95_percent_upper": _clopper_pearson_upper(
                    phase_count,
                    repetitions,
                    confidence=0.95,
                ),
                "coordinate_success_but_phase_failure_count": int(
                    np.count_nonzero(implication_violations)
                ),
            }
        )

    combined_coordinate_count = int(np.count_nonzero(combined_coordinate_failure))
    combined_phase_count = int(np.count_nonzero(combined_phase_failure))
    combined = {
        "coordinate_failure_count": combined_coordinate_count,
        "coordinate_failure_rate": float(combined_coordinate_count / repetitions),
        "coordinate_one_sided_95_percent_upper": _clopper_pearson_upper(
            combined_coordinate_count,
            repetitions,
            confidence=0.95,
        ),
        "phase_failure_count": combined_phase_count,
        "phase_failure_rate": float(combined_phase_count / repetitions),
        "phase_one_sided_95_percent_upper": _clopper_pearson_upper(
            combined_phase_count,
            repetitions,
            confidence=0.95,
        ),
    }
    return round_results, combined


def _component_operators_in_sector(
    preparation: Any,
    sector: PhysicalSector,
    permutation: np.ndarray,
) -> dict[str, np.ndarray]:
    full_dimension = 1 << sector.n_qubits
    basis_states = np.arange(full_dimension, dtype=np.uint64)
    result: dict[str, np.ndarray] = {}
    for component in preparation.rte_preparation.symbolic_tail.components:
        signs = np.ones(full_dimension, dtype=np.float64)
        parity = np.zeros(full_dimension, dtype=np.uint64)
        for qubit in component.diagonal_pauli_support:
            parity ^= (basis_states >> int(qubit)) & 1
        signs[parity.astype(bool)] = -1.0
        basis = basis_change_unitary(
            preparation.tail_extraction.basis_definition(component.basis_id),
            max_dense_qubits=8,
        )
        full = component.coefficient_sign * (
            (basis * signs[np.newaxis, :]) @ basis.conj().T
        )
        restricted = _matrix_in_openfermion_sector(full, sector, permutation)
        if restricted.leakage_frobenius_norm > 1e-10:
            raise ValueError("An RTE component leaks outside the physical sector.")
        result[component.component_id] = restricted.matrix
    return result


def _apply_event_to_state(
    event: RTEEvent,
    operators: Mapping[str, np.ndarray],
    state: np.ndarray,
) -> np.ndarray:
    value = state
    for component_id in event.product_component_ids:
        value = operators[component_id] @ value
    rotation = operators[event.rotation_component_id]
    value = (
        math.cos(event.rotation_angle) * value
        - 1j * math.sin(event.rotation_angle) * (rotation @ value)
    )
    return event.phase * value


def _trajectory_signal(
    *,
    initial_state: np.ndarray,
    forward: np.ndarray,
    reverse: np.ndarray,
    events: Sequence[RTEEvent],
    operators: Mapping[str, np.ndarray],
    q_m: int,
    rte_steps: int,
) -> complex:
    expected_events = q_m * rte_steps
    if len(events) != expected_events:
        raise ValueError("Explicit trajectory event count mismatch.")
    value = np.asarray(initial_state, dtype=np.complex128)
    offset = 0
    for _ in range(q_m):
        value = forward @ value
        for event in events[offset : offset + rte_steps]:
            value = _apply_event_to_state(event, operators, value)
        offset += rte_steps
        value = reverse @ value
    return complex(np.vdot(initial_state, value))


def _seed_for_shot(master_seed: int, q_m: int, axis: str, shot: int) -> int:
    encoded = f"{master_seed}:{q_m}:{axis}:{shot}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _explicit_fresh_iid_validation(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    ld: int,
    delta_time: float,
    q_values: Sequence[int],
    rte_steps: int,
    finite_taylor_order: int,
    round_inputs: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    dense_hamiltonian = dense_df_operator_in_sector(hamiltonian, sector)
    _eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
    initial_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, ld),
        identity_policy="extract_identity_phase",
    )
    permutation = _qiskit_to_openfermion_sector_permutation(sector)
    reference_request = make_df_partial_s2_step_request(
        preparation,
        step_time=delta_time,
        rte_steps=1,
        truncation_tolerance=_explicit_cutoff_tolerance(
            preparation.exact_rte_lambda_r * delta_time,
            0,
        ),
        finite_taylor_order=0,
        seed=seed,
    )
    additive = QiskitDFPartialS2CircuitBuilder().build_additive_circuits(
        reference_request
    )
    forward = _circuit_operator_in_openfermion_sector(
        additive.forward_deterministic_half,
        sector,
        permutation,
    ).matrix
    reverse = _circuit_operator_in_openfermion_sector(
        additive.reverse_deterministic_half,
        sector,
        permutation,
    ).matrix
    operators = _component_operators_in_sector(preparation, sector, permutation)
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=delta_time,
        rte_steps=rte_steps,
        truncation_tolerance=_explicit_cutoff_tolerance(
            preparation.exact_rte_lambda_r * delta_time / rte_steps,
            finite_taylor_order,
        ),
        finite_taylor_order=finite_taylor_order,
        seed=seed,
    )

    results: list[dict[str, Any]] = []
    all_seeds: list[int] = []
    for item in round_inputs:
        q_m = int(item["q_m"])
        if q_m not in q_values:
            raise ValueError("Explicit q does not match validated q_values.")
        true_signal = complex(item["true_signal"])
        shots = int(item["shots"])
        for axis in ("cosine", "sine"):
            trajectory_signals = np.empty(shots, dtype=np.complex128)
            outcomes = np.empty(shots, dtype=np.int8)
            measurement_rng = np.random.Generator(
                np.random.PCG64(_seed_for_shot(seed, q_m, axis, shots + 1))
            )
            axis_seeds: list[int] = []
            for shot in range(shots):
                trajectory_seed = _seed_for_shot(seed, q_m, axis, shot)
                axis_seeds.append(trajectory_seed)
                events = sample_rte_events(
                    preparation.rte_preparation.symbolic_tail.components,
                    distribution,
                    sample_count=q_m * rte_steps,
                    seed=trajectory_seed,
                )
                signal = _trajectory_signal(
                    initial_state=initial_state,
                    forward=forward,
                    reverse=reverse,
                    events=events,
                    operators=operators,
                    q_m=q_m,
                    rte_steps=rte_steps,
                )
                trajectory_signals[shot] = signal
                conditional_mean = signal.real if axis == "cosine" else signal.imag
                probability_plus = float(
                    np.clip((1.0 + conditional_mean) / 2.0, 0.0, 1.0)
                )
                outcomes[shot] = (
                    1 if measurement_rng.random() < probability_plus else -1
                )
            all_seeds.extend(axis_seeds)
            trajectory_mean = complex(np.mean(trajectory_signals))
            centered = trajectory_signals - trajectory_mean
            complex_se = float(
                math.sqrt(
                    math.fsum(float(abs(value) ** 2) for value in centered)
                    / (shots - 1)
                    / shots
                )
            )
            trajectory_z = (
                0.0
                if complex_se == 0.0 and trajectory_mean == true_signal
                else (
                    math.inf
                    if complex_se == 0.0
                    else float(abs(trajectory_mean - true_signal) / complex_se)
                )
            )
            target_coordinate = true_signal.real if axis == "cosine" else true_signal.imag
            plus_count = int(np.count_nonzero(outcomes == 1))
            marginal_probability = float((1.0 + target_coordinate) / 2.0)
            lower_plus = int(binom.ppf(0.0005, shots, marginal_probability))
            upper_plus = int(binom.ppf(0.9995, shots, marginal_probability))
            results.append(
                {
                    "q_m": q_m,
                    "axis": axis,
                    "shot_count": shots,
                    "fresh_trajectory_seed_count": len(axis_seeds),
                    "fresh_trajectory_seeds_unique": len(set(axis_seeds))
                    == len(axis_seeds),
                    "trajectory_seed_digest": hashlib.sha256(
                        ",".join(str(value) for value in axis_seeds).encode("utf-8")
                    ).hexdigest(),
                    "analytic_marginal_signal": _complex_payload(true_signal),
                    "sample_trajectory_mean_signal": _complex_payload(
                        trajectory_mean
                    ),
                    "trajectory_mean_complex_standard_error": complex_se,
                    "trajectory_mean_absolute_z": trajectory_z,
                    "measurement_plus_count": plus_count,
                    "measurement_sample_mean": float(np.mean(outcomes)),
                    "target_coordinate_mean": float(target_coordinate),
                    "marginal_binomial_99p9_central_plus_count_interval": [
                        lower_plus,
                        upper_plus,
                    ],
                    "measurement_count_inside_marginal_interval": bool(
                        lower_plus <= plus_count <= upper_plus
                    ),
                }
            )
    return {
        "rte_config": {
            "dimensionless_step_time": config.dimensionless_step_time,
            "rte_steps": config.rte_steps,
            "finite_taylor_order": config.finite_taylor_order,
            "distribution_normalization": config.distribution_normalization,
        },
        "total_fresh_trajectory_count": len(all_seeds),
        "all_trajectory_seeds_unique_across_axes_and_rounds": len(set(all_seeds))
        == len(all_seeds),
        "maximum_trajectory_mean_absolute_z": max(
            float(item["trajectory_mean_absolute_z"]) for item in results
        ),
        "all_measurement_counts_inside_marginal_99p9_intervals": all(
            bool(item["measurement_count_inside_marginal_interval"])
            for item in results
        ),
        "axis_results": results,
    }


def validate_rpe_hadamard_failure_allocation(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    ld: int,
    delta_time: float,
    q_values: Sequence[int] = (1, 2, 4),
    rte_steps_per_occurrence: int = 4,
    finite_taylor_order: int = 2,
    beta_rpe: float = 0.40,
    beta_pf_budget: float = 0.08,
    beta_rte_budget: float = 0.08,
    beta_stat_budget: float = 0.24,
    alpha_total: float = 0.05,
    marginal_repetitions: int = 100_000,
    marginal_seed: int = 20260902,
    explicit_trajectory_seed: int = 20260903,
    monte_carlo_interval_confidence: float = 0.99,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate provisional short-round statistical failure allocations."""
    started = time.perf_counter()
    ld_value = require_integer_count(ld, name="ld")
    r_value = require_integer_count(
        rte_steps_per_occurrence,
        name="rte_steps_per_occurrence",
        minimum=1,
    )
    k_value = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    repetitions = require_integer_count(
        marginal_repetitions,
        name="marginal_repetitions",
        minimum=1,
    )
    q_grid = tuple(
        sorted(
            set(
                require_integer_count(value, name="q_value", minimum=1)
                for value in q_values
            )
        )
    )
    if not q_grid or any(value & (value - 1) for value in q_grid):
        raise ValueError("q_values must be positive powers of two.")
    if beta_pf_budget + beta_rte_budget + beta_stat_budget > beta_rpe + 1e-15:
        raise ValueError("Phase budgets exceed beta_rpe.")
    if not 0.0 < alpha_total < 1.0:
        raise ValueError("alpha_total must lie strictly in (0, 1).")

    signal_payload = validate_finite_rte_signals(
        hamiltonian,
        sector,
        ld=ld_value,
        delta_time=delta_time,
        q_values=q_grid,
        rte_step_values=(r_value,),
        finite_taylor_orders=(k_value,),
        beta_rpe=beta_rpe,
        beta_pf_budget=beta_pf_budget,
        beta_rte_budget=beta_rte_budget,
        beta_stat_budget=beta_stat_budget,
        alpha_total=alpha_total,
        seed=explicit_trajectory_seed,
        provenance={
            "parent_validation": RPE_HADAMARD_FAILURE_METHOD,
            "outer_provenance": dict(provenance or {}),
        },
    )
    alpha_axis = float(alpha_total / (2 * len(q_grid)))
    round_inputs: list[dict[str, Any]] = []
    exact_rounds: list[dict[str, Any]] = []
    for point in signal_payload["points"]:
        physical = next(
            item
            for item in point["state_results"]
            if item["state_label"] == "physical_df_ground_state"
        )
        true_signal = _complex_from_payload(physical["attenuated_event_mean_signal"])
        shots = physical["provisional_shots_reference_radius_per_axis"]
        if shots is None:
            raise ValueError("Signal validation did not produce a finite shot count.")
        radius_lower = float(physical["conservative_radius_lower_bound"])
        epsilon_coordinate = float(
            radius_lower * math.sin(beta_stat_budget) / math.sqrt(2.0)
        )
        cosine = _axis_distribution(true_signal.real, shots, epsilon_coordinate)
        sine = _axis_distribution(true_signal.imag, shots, epsilon_coordinate)
        phase_probability, implication_violations = _round_exact_phase_probability(
            cosine,
            sine,
            true_signal,
            beta_stat_budget,
        )
        round_input = {
            "q_m": int(point["q_m"]),
            "shots": int(shots),
            "epsilon_coordinate": epsilon_coordinate,
            "beta_stat_budget": float(beta_stat_budget),
            "true_signal": true_signal,
            "cosine_exact_failure_probability": cosine[
                "exact_coordinate_failure_probability"
            ],
            "sine_exact_failure_probability": sine[
                "exact_coordinate_failure_probability"
            ],
            "exact_phase_failure_probability": phase_probability,
        }
        round_inputs.append(round_input)
        exact_rounds.append(
            {
                "round_index": int(point["round_index"]),
                "q_m": int(point["q_m"]),
                "shots_per_axis": int(shots),
                "attenuated_event_mean_signal": _complex_payload(true_signal),
                "observed_signal_radius": float(abs(true_signal)),
                "conservative_radius_lower_bound": radius_lower,
                "epsilon_coordinate": epsilon_coordinate,
                "cosine": _strip_axis_arrays(cosine),
                "sine": _strip_axis_arrays(sine),
                "exact_phase_failure_probability": phase_probability,
                "phase_failure_union_budget": float(2.0 * alpha_axis),
                "coordinate_success_but_phase_failure_grid_points": (
                    implication_violations
                ),
            }
        )

    exact_combined_coordinate = float(
        1.0
        - math.prod(
            (1.0 - item["cosine_exact_failure_probability"])
            * (1.0 - item["sine_exact_failure_probability"])
            for item in round_inputs
        )
    )
    exact_combined_phase = float(
        1.0
        - math.prod(
            1.0 - item["exact_phase_failure_probability"]
            for item in round_inputs
        )
    )
    marginal_rounds, marginal_combined = _simulate_marginal_experiments(
        round_inputs,
        repetitions=repetitions,
        seed=marginal_seed,
        confidence=monte_carlo_interval_confidence,
    )
    explicit = _explicit_fresh_iid_validation(
        hamiltonian,
        sector,
        ld=ld_value,
        delta_time=delta_time,
        q_values=q_grid,
        rte_steps=r_value,
        finite_taylor_order=k_value,
        round_inputs=round_inputs,
        seed=explicit_trajectory_seed,
    )

    exact_checks = {
        "all_axis_coordinate_failures_within_allocated_alpha": all(
            round_result[axis]["exact_coordinate_failure_probability"]
            <= alpha_axis
            for round_result in exact_rounds
            for axis in ("cosine", "sine")
        ),
        "all_round_phase_failures_within_axis_union_budget": all(
            item["exact_phase_failure_probability"] <= 2.0 * alpha_axis
            for item in exact_rounds
        ),
        "combined_coordinate_failure_within_alpha_total": (
            exact_combined_coordinate <= alpha_total
        ),
        "combined_phase_failure_within_alpha_total": (
            exact_combined_phase <= alpha_total
        ),
        "coordinate_bounds_imply_phase_budget_on_discrete_grid": all(
            item["coordinate_success_but_phase_failure_grid_points"] == 0
            for item in exact_rounds
        ),
    }
    marginal_checks = {
        "all_axis_exact_probabilities_inside_99_percent_mc_intervals": all(
            axis_result["exact_probability_inside_interval"]
            for item in marginal_rounds
            for axis_result in item["axes"].values()
        ),
        "all_round_exact_phase_probabilities_inside_99_percent_mc_intervals": all(
            item["exact_phase_probability_inside_interval"]
            for item in marginal_rounds
        ),
        "no_coordinate_success_but_phase_failure_samples": all(
            item["coordinate_success_but_phase_failure_count"] == 0
            for item in marginal_rounds
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
    overall_pass = bool(
        signal_payload["summary"]["overall_pass"]
        and all(exact_checks.values())
        and all(marginal_checks.values())
        and all(explicit_checks.values())
    )
    payload: dict[str, Any] = {
        "schema_version": RPE_HADAMARD_FAILURE_SCHEMA_VERSION,
        "validation_method": RPE_HADAMARD_FAILURE_METHOD,
        "scope": "short_round_virtual_hadamard_measurement_statistics_only",
        "final_cost_evaluation_performed": False,
        "full_rpe_phase_reconstruction_performed": False,
        "backend_execution_performed": False,
        "noise_model_included": False,
        "hamiltonian": {
            "hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": hamiltonian.n_blocks,
            "metadata": dict(hamiltonian.metadata),
            "sector_dimension": sector.dimension,
            "sector_n_electrons": sector.n_electrons,
        },
        "request": {
            "ld": ld_value,
            "delta_time": float(delta_time),
            "q_values": list(q_grid),
            "rte_steps_per_occurrence": r_value,
            "finite_taylor_order": k_value,
            "beta_rpe": float(beta_rpe),
            "allocation": {
                "beta_pf_budget": float(beta_pf_budget),
                "beta_rte_budget": float(beta_rte_budget),
                "beta_stat_budget": float(beta_stat_budget),
                "alpha_total": float(alpha_total),
                "alpha_per_round_axis": alpha_axis,
            },
            "marginal_repetitions": repetitions,
            "marginal_seed": marginal_seed,
            "explicit_trajectory_seed": explicit_trajectory_seed,
            "monte_carlo_interval_confidence": float(
                monte_carlo_interval_confidence
            ),
        },
        "finite_rte_signal_validation_fingerprint": signal_payload[
            "validation_fingerprint"
        ],
        "exact_binomial": {
            "rounds": exact_rounds,
            "combined_six_axis_coordinate_failure_probability": (
                exact_combined_coordinate
            ),
            "combined_three_round_phase_failure_probability": exact_combined_phase,
            "checks": exact_checks,
        },
        "marginal_monte_carlo": {
            "interpretation": (
                "iid_bernoulli_replications_from_analytic_finite_rte_marginal"
            ),
            "rounds": marginal_rounds,
            "combined": marginal_combined,
            "checks": marginal_checks,
        },
        "explicit_fresh_iid_trajectory_batch": {
            "interpretation": (
                "one_production_sized_batch_per_axis_for_sampler_and_signal_audit_"
                "not_a_rare_failure_probability_estimate"
            ),
            **explicit,
            "checks": explicit_checks,
        },
        "summary": {
            "all_finite_rte_signal_checks_pass": signal_payload["summary"][
                "overall_pass"
            ],
            "all_exact_probability_checks_pass": all(exact_checks.values()),
            "all_marginal_monte_carlo_checks_pass": all(
                marginal_checks.values()
            ),
            "all_explicit_fresh_iid_checks_pass": all(explicit_checks.values()),
            "overall_pass": overall_pass,
            "interpretation": (
                "short_round_statistical_allocation_validated_not_end_to_end_rpe"
            ),
        },
        "performance": {"elapsed_seconds": time.perf_counter() - started},
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    validate_rpe_hadamard_failure_payload(payload)
    return payload


def validate_rpe_hadamard_failure_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RPE_HADAMARD_FAILURE_SCHEMA_VERSION:
        raise ValueError("Unsupported Hadamard failure-validation schema.")
    if payload.get("validation_method") != RPE_HADAMARD_FAILURE_METHOD:
        raise ValueError("Unsupported Hadamard failure-validation method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("Failure validation cannot contain a final cost result.")
    exact_checks = payload.get("exact_binomial", {}).get("checks", {})
    marginal_checks = payload.get("marginal_monte_carlo", {}).get("checks", {})
    explicit_checks = payload.get("explicit_fresh_iid_trajectory_batch", {}).get(
        "checks", {}
    )
    expected_exact = bool(exact_checks) and all(bool(value) for value in exact_checks.values())
    expected_marginal = bool(marginal_checks) and all(
        bool(value) for value in marginal_checks.values()
    )
    expected_explicit = bool(explicit_checks) and all(
        bool(value) for value in explicit_checks.values()
    )
    summary = payload.get("summary", {})
    if summary.get("all_exact_probability_checks_pass") != expected_exact:
        raise ValueError("Exact probability check summary mismatch.")
    if summary.get("all_marginal_monte_carlo_checks_pass") != expected_marginal:
        raise ValueError("Marginal Monte Carlo check summary mismatch.")
    if summary.get("all_explicit_fresh_iid_checks_pass") != expected_explicit:
        raise ValueError("Explicit fresh-IID check summary mismatch.")
    expected_overall = bool(
        summary.get("all_finite_rte_signal_checks_pass")
        and expected_exact
        and expected_marginal
        and expected_explicit
    )
    if summary.get("overall_pass") != expected_overall:
        raise ValueError("Hadamard failure-validation overall status mismatch.")
    fingerprint = payload.get("validation_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Hadamard failure-validation fingerprint mismatch.")


def write_rpe_hadamard_failure_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_rpe_hadamard_failure_payload(payload)
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

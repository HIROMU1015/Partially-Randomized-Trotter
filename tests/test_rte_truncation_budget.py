from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

from trotterlib.rte import (
    RPETruncationSummary,
    RPERound,
    RTEOccurrenceParameters,
    compose_truncation_residual_bounds,
    finite_taylor_operator,
    occurrence_truncation_residual_bound,
    select_rpe_round_taylor_orders,
    step_taylor_truncation_residual_bound,
    summarize_rpe_round_truncation,
)


X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _operator_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, ord=2))


def test_one_step_and_repeated_actual_errors_obey_composed_bounds() -> None:
    tau = 0.31
    order = 2
    step_bound = step_taylor_truncation_residual_bound(tau, order)
    finite_step = finite_taylor_operator(X, tau, order)
    exact_step = expm(-1j * tau * X)
    assert _operator_error(finite_step, exact_step) <= step_bound

    rte_steps = 5
    finite_occurrence = np.linalg.matrix_power(finite_step, rte_steps)
    exact_occurrence = expm(-1j * tau * rte_steps * X)
    occurrence_bound = occurrence_truncation_residual_bound(step_bound, rte_steps)
    assert _operator_error(finite_occurrence, exact_occurrence) <= occurrence_bound


def test_composed_bound_is_stable_for_tiny_values_and_overflow() -> None:
    tiny = compose_truncation_residual_bounds(((1e-16, 1000, 1),))
    np.testing.assert_allclose(tiny, math.expm1(1000 * math.log1p(1e-16)))
    assert math.isinf(compose_truncation_residual_bounds(((1.0, 100_000, 1),)))


def test_heterogeneous_composition_actual_error_obeys_total_bound() -> None:
    first_tau, first_order, first_steps = 0.19, 2, 3
    second_tau, second_order, second_steps = -0.23, 4, 2
    first_step = finite_taylor_operator(X, first_tau, first_order)
    second_step = finite_taylor_operator(Z, second_tau, second_order)
    finite = np.linalg.matrix_power(second_step, second_steps) @ np.linalg.matrix_power(
        first_step, first_steps
    )
    exact = expm(-1j * second_tau * second_steps * Z) @ expm(
        -1j * first_tau * first_steps * X
    )
    first_bound = step_taylor_truncation_residual_bound(first_tau, first_order)
    second_bound = step_taylor_truncation_residual_bound(second_tau, second_order)
    total_bound = compose_truncation_residual_bounds(
        (
            (first_bound, first_steps, 1),
            (second_bound, second_steps, 1),
        )
    )

    assert _operator_error(finite, exact) <= total_bound


def test_partial_s2_repetition_is_reflected_in_round_bound() -> None:
    parameters = RTEOccurrenceParameters(
        occurrence_id="tail-center",
        tail_id="tail",
        tail_hash="hash",
        lambda_r=1.0,
        evolution_time=0.6,
        rte_steps=3,
        round_occurrence_count=4,
    )
    selected = select_rpe_round_taylor_orders(
        (parameters,),
        target_round_truncation_error=1e-5,
    ).occurrences[0]
    rpe_round = RPERound(
        round_index=2,
        effective_evolution_time=2.4,
        partial_s2_repetitions=4,
        tail_evolutions=4,
        rte_total_steps=12,
        attenuation_factor=0.9,
        measurement_axis="X",
        required_shots=100,
    )
    summary = summarize_rpe_round_truncation(
        (selected,),
        target_round_truncation_error=1e-5,
        rpe_round=rpe_round,
    )
    one_occurrence_bound = occurrence_truncation_residual_bound(
        selected.step_truncation_residual_bound,
        selected.rte_steps,
    )

    assert isinstance(summary, RPETruncationSummary)
    assert summary.budget.partial_s2_repetitions == 4
    assert summary.budget.expected_tail_evolutions == 4
    assert summary.budget.total_short_step_count == 12
    assert summary.round_truncation_residual_bound > one_occurrence_bound
    np.testing.assert_allclose(
        summary.round_truncation_residual_bound,
        occurrence_truncation_residual_bound(
            selected.step_truncation_residual_bound,
            12,
        ),
    )


def test_round_budget_selects_minimal_even_orders_for_equal_log_allocation() -> None:
    occurrences = (
        RTEOccurrenceParameters(
            occurrence_id="left-tail",
            tail_id="tail-a",
            tail_hash="hash-a",
            lambda_r=1.0,
            evolution_time=0.6,
            rte_steps=3,
            round_occurrence_count=2,
        ),
        RTEOccurrenceParameters(
            occurrence_id="right-tail",
            tail_id="tail-b",
            tail_hash="hash-b",
            lambda_r=0.8,
            evolution_time=-0.5,
            rte_steps=2,
            round_occurrence_count=1,
        ),
    )
    rpe_round = RPERound(
        round_index=5,
        effective_evolution_time=1.7,
        partial_s2_repetitions=2,
        tail_evolutions=3,
        rte_total_steps=8,
        attenuation_factor=0.8,
        measurement_axis="Y",
        required_shots=200,
    )
    summary = select_rpe_round_taylor_orders(
        occurrences,
        target_round_truncation_error=1e-7,
        rpe_round=rpe_round,
    )
    allocated = summary.budget.allocated_step_error_bound

    assert allocated is not None
    assert summary.meets_round_budget
    assert summary.round_truncation_residual_bound <= 1e-7
    assert summary.budget.allocation_policy == "equal_log_budget_per_short_step"
    assert summary.budget.total_short_step_count == 8
    for record in summary.occurrences:
        assert record.finite_taylor_order >= 0
        assert record.finite_taylor_order % 2 == 0
        assert record.step_truncation_residual_bound <= allocated
        if record.finite_taylor_order >= 2:
            lower = step_taylor_truncation_residual_bound(
                record.dimensionless_step_time,
                record.finite_taylor_order - 2,
            )
            assert lower > allocated

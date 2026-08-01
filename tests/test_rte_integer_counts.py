from __future__ import annotations

import numpy as np
import pytest

from trotterlib.rte import (
    InvolutoryTailTerm,
    RPERound,
    RPERoundTruncationBudget,
    RTEComponent,
    RTEOccurrenceParameters,
    RTEOccurrenceTruncation,
    choose_finite_taylor_order,
    compose_truncation_residual_bounds,
    finite_rte_distribution,
    make_rte_config,
    normalize_involutory_tail,
    occurrence_truncation_residual_bound,
    require_integer_count,
    sample_rte_events,
)


X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


@pytest.mark.parametrize("value", (True, False, 2.0, 2.5, "2"))
def test_integer_count_rejects_bool_float_and_string(value) -> None:
    with pytest.raises(TypeError):
        require_integer_count(value, name="count")


def test_integer_count_accepts_python_and_numpy_integers() -> None:
    assert require_integer_count(0, name="count") == 0
    assert require_integer_count(2, name="count") == 2
    assert require_integer_count(np.int64(3), name="count") == 3
    with pytest.raises(ValueError):
        require_integer_count(-1, name="count")
    with pytest.raises(ValueError):
        require_integer_count(0, name="count", minimum=1)


@pytest.mark.parametrize("value", (True, 2.0, 2.5, "2", -1, 0))
def test_positive_rte_step_apis_reject_invalid_counts(value) -> None:
    tail = normalize_involutory_tail(
        "tail",
        (InvolutoryTailTerm("x", 1.0, X),),
    )
    with pytest.raises((TypeError, ValueError)):
        make_rte_config(
            tail,
            evolution_time=0.0,
            rte_steps=value,
            truncation_tolerance=1.0,
            finite_taylor_order=0,
        )
    with pytest.raises((TypeError, ValueError)):
        RTEOccurrenceParameters(
            occurrence_id="occ",
            tail_id="tail",
            tail_hash="hash",
            lambda_r=1.0,
            evolution_time=0.1,
            rte_steps=value,
        )


@pytest.mark.parametrize("value", (True, 2.0, 2.5, "2", -1, 0))
def test_occurrence_multiplicity_requires_a_positive_integer(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        RTEOccurrenceParameters(
            occurrence_id="occ",
            tail_id="tail",
            tail_hash="hash",
            lambda_r=1.0,
            evolution_time=0.1,
            rte_steps=1,
            round_occurrence_count=value,
        )


def test_numpy_counts_are_normalized_and_zero_is_allowed_where_valid() -> None:
    tail = normalize_involutory_tail(
        "tail",
        (InvolutoryTailTerm("x", 1.0, X),),
    )
    config, _ = make_rte_config(
        tail,
        evolution_time=0.0,
        rte_steps=np.int64(2),
        truncation_tolerance=1.0,
        finite_taylor_order=np.int64(0),
    )
    assert config.rte_steps == 2
    assert type(config.rte_steps) is int
    assert occurrence_truncation_residual_bound(0.1, np.int64(0)) == 0.0
    assert compose_truncation_residual_bounds(((0.1, 0, 4),)) == 0.0
    assert compose_truncation_residual_bounds(((0.1, 2, 0),)) == 0.0


@pytest.mark.parametrize("value", (True, 2.0, 2.5, "2", -1))
def test_composed_bound_counts_never_silently_truncate(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        occurrence_truncation_residual_bound(0.1, value)
    with pytest.raises((TypeError, ValueError)):
        compose_truncation_residual_bounds(((0.1, value, 1),))
    with pytest.raises((TypeError, ValueError)):
        compose_truncation_residual_bounds(((0.1, 1, value),))


def test_finite_taylor_and_maximum_orders_require_integers() -> None:
    distribution = finite_rte_distribution(0.1, np.int64(2))
    assert distribution.finite_taylor_order == 2
    assert choose_finite_taylor_order(0.0, 1e-6, maximum_order=np.int64(0)) == 0
    for value in (True, 2.0, 2.5, "2", -1):
        with pytest.raises((TypeError, ValueError)):
            finite_rte_distribution(0.1, value)
        with pytest.raises((TypeError, ValueError)):
            choose_finite_taylor_order(0.1, 1e-6, maximum_order=value)
    with pytest.raises(ValueError, match="even"):
        finite_rte_distribution(0.1, 3)


def test_sample_count_requires_an_integer_but_allows_zero() -> None:
    component = RTEComponent(
        component_id="x",
        probability=1.0,
        coefficient_abs=1.0,
        coefficient_sign=1,
    )
    distribution = finite_rte_distribution(0.1, 2)
    assert sample_rte_events(
        (component,),
        distribution,
        sample_count=np.int64(0),
        seed=0,
    ) == ()
    for value in (True, 1.0, 1.5, "1", -1):
        with pytest.raises((TypeError, ValueError)):
            sample_rte_events(
                (component,),
                distribution,
                sample_count=value,
                seed=0,
            )


def test_rpe_and_truncation_dataclasses_validate_all_counts() -> None:
    occurrence = RTEOccurrenceTruncation(
        occurrence_id="occ",
        tail_id="tail",
        tail_hash="hash",
        lambda_r=1.0,
        evolution_time=0.1,
        rte_steps=np.int64(2),
        finite_taylor_order=np.int64(2),
        dimensionless_step_time=0.05,
        step_truncation_residual_bound=1e-6,
        occurrence_truncation_residual_bound=2e-6,
        round_occurrence_count=np.int64(3),
        round_contribution_residual_bound=6e-6,
    )
    assert occurrence.rte_steps == 2
    assert occurrence.round_occurrence_count == 3

    budget = RPERoundTruncationBudget(
        round_index=np.int64(0),
        target_round_truncation_error=1e-5,
        allocation_policy="user_selected_orders",
        total_short_step_count=np.int64(6),
        allocated_log_error_per_short_step=None,
        allocated_step_error_bound=None,
        partial_s2_repetitions=np.int64(0),
        expected_tail_evolutions=np.int64(3),
    )
    assert budget.total_short_step_count == 6

    rpe_round = RPERound(
        round_index=np.int64(0),
        effective_evolution_time=0.0,
        partial_s2_repetitions=np.int64(0),
        tail_evolutions=np.int64(0),
        rte_total_steps=np.int64(0),
        attenuation_factor=1.0,
        measurement_axis="X",
        required_shots=np.int64(1),
    )
    assert rpe_round.required_shots == 1

    with pytest.raises(TypeError):
        RPERound(
            round_index=True,
            effective_evolution_time=0.0,
            partial_s2_repetitions=0,
            tail_evolutions=0,
            rte_total_steps=0,
            attenuation_factor=1.0,
            measurement_axis="X",
            required_shots=1,
        )
    with pytest.raises(ValueError):
        RPERound(
            round_index=0,
            effective_evolution_time=0.0,
            partial_s2_repetitions=0,
            tail_evolutions=0,
            rte_total_steps=0,
            attenuation_factor=1.0,
            measurement_axis="X",
            required_shots=0,
        )
    with pytest.raises(ValueError):
        RPERoundTruncationBudget(
            round_index=0,
            target_round_truncation_error=1e-5,
            allocation_policy="user_selected_orders",
            total_short_step_count=0,
            allocated_log_error_per_short_step=None,
            allocated_step_error_bound=None,
        )


@pytest.mark.parametrize("value", (True, 2.0, 2.5, "2", -1, 0))
def test_occurrence_truncation_rejects_invalid_positive_counts(value) -> None:
    fields = {
        "occurrence_id": "occ",
        "tail_id": "tail",
        "tail_hash": "hash",
        "lambda_r": 1.0,
        "evolution_time": 0.1,
        "rte_steps": 1,
        "finite_taylor_order": 2,
        "dimensionless_step_time": 0.1,
        "step_truncation_residual_bound": 1e-6,
        "occurrence_truncation_residual_bound": 1e-6,
        "round_occurrence_count": 1,
        "round_contribution_residual_bound": 1e-6,
    }
    with pytest.raises((TypeError, ValueError)):
        RTEOccurrenceTruncation(**{**fields, "rte_steps": value})
    with pytest.raises((TypeError, ValueError)):
        RTEOccurrenceTruncation(**{**fields, "round_occurrence_count": value})


@pytest.mark.parametrize("value", (True, 2.0, 2.5, "2", -1))
def test_rpe_round_count_fields_reject_noninteger_or_negative(value) -> None:
    base = {
        "round_index": 0,
        "effective_evolution_time": 0.0,
        "partial_s2_repetitions": 0,
        "tail_evolutions": 0,
        "rte_total_steps": 0,
        "attenuation_factor": 1.0,
        "measurement_axis": "X",
        "required_shots": 1,
    }
    for name in (
        "round_index",
        "partial_s2_repetitions",
        "tail_evolutions",
        "rte_total_steps",
    ):
        with pytest.raises((TypeError, ValueError)):
            RPERound(**{**base, name: value})
    with pytest.raises((TypeError, ValueError)):
        RPERound(**{**base, "required_shots": value})

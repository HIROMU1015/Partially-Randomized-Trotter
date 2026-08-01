from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest
import qiskit

import trotterlib.df_partial_s2_repeated_cost as repeated_cost_module
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import DFPartialS2StepRequest, prepare_df_partial_s2
from trotterlib.df_partial_s2_cost import estimate_exact_compiled_partial_s2_cost
from trotterlib.df_partial_s2_repeated import (
    DFPartialS2RepeatedRequest,
    QiskitDFPartialS2RepeatedCircuitBuilder,
    make_df_partial_s2_repeated_request,
)
from trotterlib.df_partial_s2_repeated_cost import (
    estimate_exact_compiled_repeated_partial_s2_cost,
    estimate_monte_carlo_compiled_repeated_partial_s2_cost,
)
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.rte import CompilerSettings, enumerate_rte_events, make_rte_config
from trotterlib.rte_compiled_cost import (
    TranspiledCircuitCostCache,
    transpile_and_measure_cost,
)


METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)


def _compiler(seed: int = 17, optimization_level: int = 1) -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=optimization_level,
        layout_method=None,
        routing_method=None,
        transpiler_seed=seed,
        qiskit_version=qiskit.__version__,
    )


def _case(*, rte_steps: int = 1, deterministic: bool = False):
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "repeated-partial-s2-cost"},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1 if deterministic else 0),
        identity_policy="extract_identity_phase",
    )
    if deterministic:
        return preparation, None, None, 0.2
    step_time = 1.2 * rte_steps / preparation.exact_rte_lambda_r
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=step_time,
        rte_steps=rte_steps,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    return preparation, config, distribution, step_time


def _step_request(preparation, config, distribution, events, *, seed=0):
    tail = preparation.rte_preparation.symbolic_tail
    occurrence = DFRTEEventSequenceCircuitRequest(
        events=tuple(events),
        component_specs=preparation.rte_preparation.component_specs,
        tail_id=tail.tail_id,
        tail_hash=tail.tail_hash,
        occurrence_rte_steps=config.rte_steps,
    )
    return DFPartialS2StepRequest(
        preparation=preparation,
        step_time=config.evolution_time,
        rte_config=config,
        rte_distribution=distribution,
        rte_occurrence=occurrence,
        seed=seed,
    )


def _compiled_repeated_cost(request, compiler):
    result = QiskitDFPartialS2RepeatedCircuitBuilder().build(request)
    return transpile_and_measure_cost(
        result.circuit,
        compiler,
        circuit_fingerprint=result.compiler_independent_fingerprint,
    )


def test_repetition_one_exact_cost_matches_existing_level_five_step() -> None:
    preparation, config, distribution, step_time = _case()
    compiler = _compiler()
    existing = estimate_exact_compiled_partial_s2_cost(
        preparation,
        step_time,
        config,
        distribution,
        compiler,
    )
    repeated = estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        1,
        config,
        distribution,
        compiler,
        construction_policy="raw_concatenation",
    )

    assert repeated.repetition_count == 1
    assert repeated.expected_cost.fidelity_level == 5
    for metric in METRICS:
        assert getattr(repeated.expected_cost, metric) == pytest.approx(
            getattr(existing.expected_cost, metric)
        )
        assert getattr(repeated.cross_step_nonadditive_difference, metric) == (
            pytest.approx(0.0)
        )


def test_exact_repeated_expectation_matches_manual_trajectory_weighting() -> None:
    preparation, config, distribution, step_time = _case()
    compiler = _compiler()
    events = enumerate_rte_events(
        preparation.rte_preparation.symbolic_tail.components,
        distribution,
    )
    step_sequences = tuple(itertools.product(events, repeat=config.rte_steps))
    trajectories = tuple(itertools.product(step_sequences, repeat=2))
    costs = []
    probabilities = []
    for trajectory in trajectories:
        steps = tuple(
            _step_request(
                preparation,
                config,
                distribution,
                event_sequence,
                seed=index,
            )
            for index, event_sequence in enumerate(trajectory)
        )
        request = DFPartialS2RepeatedRequest.from_step_requests(
            steps,
            construction_policy="boundary_optimized",
        )
        costs.append(_compiled_repeated_cost(request, compiler))
        probabilities.append(
            math.prod(
                event.event_probability
                for event_sequence in trajectory
                for event in event_sequence
            )
        )
    manual = {
        metric: math.fsum(
            probability * float(getattr(cost, metric))
            for probability, cost in zip(probabilities, costs, strict=True)
        )
        for metric in METRICS
    }
    estimate = estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        2,
        config,
        distribution,
        compiler,
        maximum_trajectories=4,
    )

    assert len(step_sequences) == 2
    assert len(trajectories) == 4
    assert math.fsum(probabilities) == pytest.approx(1.0)
    assert estimate.trajectory_probability_sum == pytest.approx(1.0)
    assert estimate.enumerated_trajectory_count == 4
    assert estimate.single_step_event_sequence_count == 2
    assert estimate.trajectory_space_size == 4
    assert estimate.standard_error is None
    for metric in METRICS:
        assert getattr(estimate.expected_cost, metric) == pytest.approx(
            manual[metric]
        )
        assert getattr(estimate.cross_step_nonadditive_difference, metric) == (
            pytest.approx(
                getattr(estimate.expected_cost, metric)
                - getattr(estimate.matched_per_step_expected_cost, metric)
            )
        )
        assert getattr(estimate.boundary_optimization_difference, metric) == (
            pytest.approx(
                getattr(estimate.boundary_optimized_expected_cost, metric)
                - getattr(estimate.raw_concatenation_expected_cost, metric)
            )
        )


def test_exact_trajectory_limit_is_checked_before_product_or_circuit_build(
    monkeypatch,
) -> None:
    preparation, config, distribution, step_time = _case()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("enumeration or circuit construction began before preflight")

    monkeypatch.setattr(repeated_cost_module.itertools, "product", forbidden)
    monkeypatch.setattr(
        repeated_cost_module.QiskitDFPartialS2RepeatedCircuitBuilder,
        "build",
        forbidden,
    )
    with pytest.raises(ValueError, match=r"4 trajectories \(M=2, q=2\)"):
        estimate_exact_compiled_repeated_partial_s2_cost(
            preparation,
            step_time,
            2,
            config,
            distribution,
            _compiler(),
            maximum_trajectories=3,
        )


def test_repeated_untranspiled_size_limit_precedes_transpilation(monkeypatch) -> None:
    preparation, config, distribution, step_time = _case()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("transpile began before the size-limit check")

    monkeypatch.setattr(
        repeated_cost_module.TranspiledCircuitCostCache,
        "get_or_transpile",
        forbidden,
    )
    with pytest.raises(ValueError, match="repeated partial-S2 circuit"):
        estimate_exact_compiled_repeated_partial_s2_cost(
            preparation,
            step_time,
            2,
            config,
            distribution,
            _compiler(),
            maximum_trajectories=4,
            maximum_untranspiled_circuit_size=1,
        )


def test_monte_carlo_is_unweighted_reproducible_and_reports_statistics() -> None:
    preparation, config, distribution, step_time = _case()
    compiler = _compiler()
    sample_count = 40
    seed = 8
    first = estimate_monte_carlo_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        2,
        config,
        distribution,
        compiler,
        sample_count=sample_count,
        seed=seed,
    )
    second = estimate_monte_carlo_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        2,
        config,
        distribution,
        compiler,
        sample_count=sample_count,
        seed=seed,
    )
    assert first.sampled_trajectory_seeds is not None
    values = []
    sampled_probabilities = []
    for trajectory_seed in first.sampled_trajectory_seeds:
        request = make_df_partial_s2_repeated_request(
            preparation,
            step_time=step_time,
            repetition_count=2,
            rte_config=config,
            rte_distribution=distribution,
            seed=trajectory_seed,
            construction_policy="boundary_optimized",
        )
        cost = _compiled_repeated_cost(request, compiler)
        values.append(float(cost.rz_count))
        sampled_probabilities.append(
            math.prod(
                event.event_probability
                for step in request.iter_step_requests()
                for event in step.rte_occurrence.events
            )
        )
    expected_mean = float(np.mean(values))
    expected_variance = float(np.var(values, ddof=1))
    statistics = dict(first.full_metric_statistics)["rz_count"]
    double_weighted = math.fsum(
        probability * value
        for probability, value in zip(
            sampled_probabilities,
            values,
            strict=True,
        )
    ) / sample_count

    assert first.sample_count == sample_count
    assert first.master_seed == seed
    assert first.expected_cost.rz_count == pytest.approx(expected_mean)
    assert statistics.unbiased_sample_variance == pytest.approx(expected_variance)
    assert statistics.standard_error == pytest.approx(
        math.sqrt(expected_variance / sample_count)
    )
    assert first.standard_error is not None
    assert first.expected_cost.rz_count != pytest.approx(double_weighted)
    assert first.expected_cost == second.expected_cost
    assert first.standard_error == second.standard_error
    assert first.sampled_trajectory_seeds == second.sampled_trajectory_seeds
    assert statistics.minimum <= statistics.mean <= statistics.maximum


def test_monte_carlo_converges_to_exact_short_trajectory() -> None:
    preparation, config, distribution, step_time = _case()
    compiler = _compiler()
    exact = estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        2,
        config,
        distribution,
        compiler,
        maximum_trajectories=4,
    )
    sampled = estimate_monte_carlo_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        2,
        config,
        distribution,
        compiler,
        sample_count=300,
        seed=23,
    )
    assert sampled.standard_error is not None
    assert abs(sampled.expected_cost.rz_count - exact.expected_cost.rz_count) <= (
        5.0 * sampled.standard_error.rz_count + 1e-12
    )


def test_controlled_full_repeated_cost_exposes_cross_step_and_boundary_savings() -> None:
    preparation, config, distribution, step_time = _case()
    estimate = estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        2,
        config,
        distribution,
        _compiler(),
        controlled=True,
        ancilla_qubit=1,
        maximum_trajectories=4,
    )
    assert estimate.expected_cost.rz_count == pytest.approx(14.663494501049938)
    assert estimate.matched_per_step_expected_cost.rz_count == pytest.approx(
        18.66349450104994
    )
    assert estimate.cross_step_nonadditive_difference.rz_count == pytest.approx(-4.0)
    assert estimate.raw_concatenation_expected_cost.rz_count == pytest.approx(
        17.66349450104994
    )
    assert estimate.boundary_optimization_difference.rz_count == pytest.approx(-3.0)


def test_cache_reuses_identical_trajectories_and_separates_conditions() -> None:
    preparation, config, distribution, step_time = _case()
    cache = TranspiledCircuitCostCache()
    first = estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        2,
        config,
        distribution,
        _compiler(),
        maximum_trajectories=4,
        cache=cache,
    )
    cache_size = len(cache)
    repeated = estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        2,
        config,
        distribution,
        _compiler(),
        maximum_trajectories=4,
        cache=cache,
    )
    assert len(cache) == cache_size
    assert repeated.transpile_cache_hit_count > first.transpile_cache_hit_count

    estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        1,
        config,
        distribution,
        _compiler(),
        maximum_trajectories=2,
        cache=cache,
    )
    repetition_separated_size = len(cache)
    assert repetition_separated_size > cache_size

    estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        1,
        config,
        distribution,
        _compiler(),
        controlled=True,
        ancilla_qubit=1,
        maximum_trajectories=2,
        cache=cache,
    )
    controlled_size = len(cache)
    assert controlled_size > repetition_separated_size
    estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        1,
        config,
        distribution,
        _compiler(seed=18),
        controlled=True,
        ancilla_qubit=1,
        maximum_trajectories=2,
        cache=cache,
    )
    assert len(cache) > controlled_size


def test_deterministic_repeated_exact_cost_has_one_trajectory() -> None:
    preparation, config, distribution, step_time = _case(deterministic=True)
    estimate = estimate_exact_compiled_repeated_partial_s2_cost(
        preparation,
        step_time,
        3,
        config,
        distribution,
        _compiler(),
    )
    assert estimate.single_event_space_size == 1
    assert estimate.single_step_event_sequence_count == 1
    assert estimate.trajectory_space_size == 1
    assert estimate.enumerated_trajectory_count == 1
    assert estimate.trajectory_probability_sum == 1.0
    assert estimate.attenuation.total_attenuation == 1.0
    assert estimate.truncation.repeated_partial_s2_residual_bound == 0.0

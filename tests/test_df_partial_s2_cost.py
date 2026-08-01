from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import (
    DFPartialS2StepRequest,
    QiskitDFPartialS2CircuitBuilder,
    prepare_df_partial_s2,
)
from trotterlib.df_partial_s2_cost import (
    estimate_exact_compiled_partial_s2_cost,
    estimate_monte_carlo_compiled_partial_s2_cost,
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


def _compiler(seed: int = 17) -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=seed,
        qiskit_version=qiskit.__version__,
    )


def _case(*, rte_steps: int = 2, ld: int = 0):
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "partial-s2-cost"},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, ld),
        identity_policy="extract_identity_phase",
    )
    if preparation.is_deterministic_only:
        return preparation, None, None
    step_time = 1.2 * rte_steps / preparation.exact_rte_lambda_r
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=step_time,
        rte_steps=rte_steps,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    return preparation, config, distribution


def _request(
    preparation,
    config,
    distribution,
    events,
    *,
    controlled: bool = False,
    seed: int = 0,
) -> DFPartialS2StepRequest:
    tail = preparation.rte_preparation.symbolic_tail
    occurrence = DFRTEEventSequenceCircuitRequest(
        events=tuple(events),
        component_specs=preparation.rte_preparation.component_specs,
        controlled=controlled,
        ancilla_qubit=1 if controlled else None,
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
        controlled=controlled,
        ancilla_qubit=1 if controlled else None,
        seed=seed,
    )


def _compiled_full_cost(request, compiler):
    result = QiskitDFPartialS2CircuitBuilder().build_step(request)
    return transpile_and_measure_cost(
        result.circuit,
        compiler,
        circuit_fingerprint=result.compiler_independent_fingerprint,
    )


def test_exact_partial_s2_expectation_matches_manual_sequence_weighting() -> None:
    preparation, config, distribution = _case(rte_steps=2)
    compiler = _compiler()
    events = enumerate_rte_events(
        preparation.rte_preparation.symbolic_tail.components,
        distribution,
    )
    sequences = tuple(itertools.product(events, repeat=config.rte_steps))
    costs = tuple(
        _compiled_full_cost(
            _request(preparation, config, distribution, sequence),
            compiler,
        )
        for sequence in sequences
    )
    probabilities = tuple(
        math.prod(event.event_probability for event in sequence)
        for sequence in sequences
    )
    manual = {
        name: math.fsum(
            probability * float(getattr(cost, name))
            for probability, cost in zip(probabilities, costs, strict=True)
        )
        for name in METRICS
    }
    cache = TranspiledCircuitCostCache()
    estimate = estimate_exact_compiled_partial_s2_cost(
        preparation,
        config.evolution_time,
        config,
        distribution,
        compiler,
        maximum_event_sequences=10,
        cache=cache,
    )
    repeated = estimate_exact_compiled_partial_s2_cost(
        preparation,
        config.evolution_time,
        config,
        distribution,
        compiler,
        maximum_event_sequences=10,
        cache=cache,
    )

    assert len(events) == 2
    assert len(sequences) == 4
    assert math.fsum(probabilities) == pytest.approx(1.0)
    assert estimate.event_sequence_probability_sum == pytest.approx(1.0)
    assert estimate.enumerated_event_sequence_count == 4
    assert estimate.standard_error is None
    assert estimate.sample_count is None
    assert estimate.expected_cost.fidelity_level == 5
    assert estimate.expected_cost.estimate_kind == (
        "exact_compiled_partial_s2_expectation"
    )
    for name in METRICS:
        assert getattr(estimate.expected_cost, name) == pytest.approx(manual[name])
        assert getattr(estimate.nonadditive_difference, name) == pytest.approx(
            getattr(estimate.expected_cost, name)
            - getattr(estimate.additive_expected_cost, name)
        )
    assert repeated.transpile_cache_hit_count == 4 * len(sequences)
    assert repeated.unique_full_step_circuit_count == 4


def test_exact_partial_s2_sequence_and_size_limits_are_preflighted() -> None:
    preparation, config, distribution = _case(rte_steps=2)
    with pytest.raises(ValueError, match="4 sequences"):
        estimate_exact_compiled_partial_s2_cost(
            preparation,
            config.evolution_time,
            config,
            distribution,
            _compiler(),
            maximum_event_sequences=3,
        )
    with pytest.raises(ValueError, match="size limit"):
        estimate_exact_compiled_partial_s2_cost(
            preparation,
            config.evolution_time,
            config,
            distribution,
            _compiler(),
            maximum_event_sequences=4,
            maximum_untranspiled_circuit_size=1,
        )


def test_monte_carlo_partial_s2_statistics_are_unweighted_and_reproducible() -> None:
    preparation, config, distribution = _case(rte_steps=2)
    compiler = _compiler()
    sample_count = 100
    seed = 8
    rng = random.Random(seed)
    sampled_sequences = tuple(
        preparation.rte_preparation.sample_events(
            distribution,
            sample_count=config.rte_steps,
            seed=rng.randrange(0, 2**63),
        )
        for _ in range(sample_count)
    )
    cost_by_orders = {}
    values = []
    sampled_probabilities = []
    for sequence in sampled_sequences:
        key = tuple(event.taylor_order for event in sequence)
        if key not in cost_by_orders:
            cost_by_orders[key] = _compiled_full_cost(
                _request(
                    preparation,
                    config,
                    distribution,
                    sequence,
                    controlled=True,
                ),
                compiler,
            )
        values.append(float(cost_by_orders[key].rz_count))
        sampled_probabilities.append(
            math.prod(event.event_probability for event in sequence)
        )
    values_array = np.asarray(values)
    expected_mean = float(np.mean(values_array))
    expected_variance = float(np.var(values_array, ddof=1))
    expected_standard_error = math.sqrt(expected_variance / sample_count)

    first = estimate_monte_carlo_compiled_partial_s2_cost(
        preparation,
        config.evolution_time,
        config,
        distribution,
        compiler,
        sample_count=sample_count,
        seed=seed,
        controlled=True,
        ancilla_qubit=1,
    )
    second = estimate_monte_carlo_compiled_partial_s2_cost(
        preparation,
        config.evolution_time,
        config,
        distribution,
        compiler,
        sample_count=sample_count,
        seed=seed,
        controlled=True,
        ancilla_qubit=1,
    )
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
    assert first.seed == seed
    assert first.expected_cost.rz_count == pytest.approx(expected_mean)
    assert statistics.unbiased_sample_variance == pytest.approx(expected_variance)
    assert statistics.standard_error == pytest.approx(expected_standard_error)
    assert first.standard_error is not None
    assert first.standard_error.rz_count == pytest.approx(expected_standard_error)
    assert first.expected_cost.rz_count != pytest.approx(double_weighted)
    assert first.expected_cost == second.expected_cost
    assert first.standard_error == second.standard_error
    assert first.unique_full_step_circuit_count == len(cost_by_orders)
    assert first.transpile_cache_hit_count > 0
    for name in METRICS:
        assert getattr(first.nonadditive_difference, name) == pytest.approx(
            getattr(first.expected_cost, name)
            - getattr(first.additive_expected_cost, name)
        )
        assert getattr(first.additive_expected_cost, name) == pytest.approx(
            getattr(first.forward_half_expected_cost, name)
            + getattr(first.rte_occurrence_expected_cost, name)
            + getattr(first.reverse_half_expected_cost, name)
        )


def test_monte_carlo_converges_to_exact_small_case() -> None:
    preparation, config, distribution = _case(rte_steps=1)
    compiler = _compiler()
    exact = estimate_exact_compiled_partial_s2_cost(
        preparation,
        config.evolution_time,
        config,
        distribution,
        compiler,
        controlled=True,
        ancilla_qubit=1,
    )
    sampled = estimate_monte_carlo_compiled_partial_s2_cost(
        preparation,
        config.evolution_time,
        config,
        distribution,
        compiler,
        sample_count=1_000,
        seed=23,
        controlled=True,
        ancilla_qubit=1,
    )

    assert sampled.standard_error is not None
    assert abs(
        sampled.expected_cost.rz_count - exact.expected_cost.rz_count
    ) <= 5.0 * sampled.standard_error.rz_count + 1e-12


def test_cache_separates_controlled_partial_s2_steps() -> None:
    preparation, config, distribution = _case(rte_steps=1)
    compiler = _compiler()
    cache = TranspiledCircuitCostCache()
    estimate_monte_carlo_compiled_partial_s2_cost(
        preparation,
        config.evolution_time,
        config,
        distribution,
        compiler,
        sample_count=2,
        seed=3,
        cache=cache,
    )
    uncontrolled_size = len(cache)
    estimate_monte_carlo_compiled_partial_s2_cost(
        preparation,
        config.evolution_time,
        config,
        distribution,
        compiler,
        sample_count=2,
        seed=3,
        controlled=True,
        ancilla_qubit=1,
        cache=cache,
    )
    assert len(cache) > uncontrolled_size


def test_deterministic_only_exact_cost_is_one_level_five_sequence() -> None:
    preparation, config, distribution = _case(ld=1)
    estimate = estimate_exact_compiled_partial_s2_cost(
        preparation,
        0.2,
        config,
        distribution,
        _compiler(),
    )

    assert estimate.single_event_space_size == 1
    assert estimate.enumerated_event_sequence_count == 1
    assert estimate.event_sequence_probability_sum == 1.0
    assert estimate.expected_cost.fidelity_level == 5
    assert estimate.basis_reuse_policy == "none"

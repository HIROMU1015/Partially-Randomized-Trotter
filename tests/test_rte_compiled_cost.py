from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator

import trotterlib.rte_compiled_cost as compiled_cost_module
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.df_rte_tail import (
    extract_df_diagonal_tail,
    extraction_to_normalized_rte_tail,
    prepare_df_rte_event_inputs,
)
from trotterlib.df_trotter.model import DFBlock
from trotterlib.rte import (
    CompilerSettings,
    enumerate_rte_events,
    event_unitary,
    finite_rte_distribution,
    make_rte_config,
)
from trotterlib.rte_compiled_cost import (
    CompiledMetricAccumulator,
    TranspiledCircuitCostCache,
    canonical_qiskit_circuit_fingerprint,
    compiler_settings_hash,
    estimate_compiled_occurrence_cost,
    estimate_exact_compiled_event_cost,
    estimate_monte_carlo_compiled_event_cost,
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


class _OneShotIterator:
    def __init__(self, values):
        self._iterator = iter(values)
        self._iterated = False

    def __iter__(self):
        if self._iterated:
            raise AssertionError("one-shot iterator was traversed more than once")
        self._iterated = True
        return self

    def __next__(self):
        return next(self._iterator)

    def __len__(self):
        raise AssertionError("streaming code requested len(iterator)")


def _compiler(*, seed: int = 17, optimization_level: int = 1) -> CompilerSettings:
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


def _one_component_case(*, rte_steps: int = 3):
    block = DFBlock(
        U_ops=((RYGate(0.23), (0,)),),
        eta=np.asarray([1.0]),
        lam=0.7,
    )
    extraction = extract_df_diagonal_tail(
        "compiled-one-component",
        (block,),
        identity_policy="extract_identity_phase",
    )
    preparation = prepare_df_rte_event_inputs(extraction)
    evolution_time = 1.2 * rte_steps / preparation.symbolic_tail.lambda_r
    config, distribution = make_rte_config(
        preparation.symbolic_tail,
        evolution_time=evolution_time,
        rte_steps=rte_steps,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    return extraction, preparation, config, distribution


def test_transpiled_cost_uses_actual_integer_counts_and_filtered_depths() -> None:
    circuit = QuantumCircuit(2)
    circuit.rz(0.1, 0)
    circuit.cx(0, 1)
    circuit.rz(0.2, 0)
    circuit.rz(0.3, 1)
    compiler = _compiler(optimization_level=0)

    first = transpile_and_measure_cost(circuit, compiler)
    second = transpile_and_measure_cost(circuit, compiler)

    assert first.posttranspile_gate_counts == (("cx", 1), ("rz", 3))
    assert first.rz_count == 3
    assert first.rz_depth == 2
    assert first.cx_count == 1
    assert first.cx_depth == 1
    assert first.total_depth == 3
    assert first.circuit_size == 4
    assert all(isinstance(getattr(first, name), int) for name in METRICS)
    assert first.compiler == compiler
    assert first.compiler_settings_hash == compiler_settings_hash(compiler)
    assert first.circuit_fingerprint == canonical_qiskit_circuit_fingerprint(
        circuit
    )
    assert first.posttranspile_gate_counts == second.posttranspile_gate_counts
    assert np.allclose(
        Operator(first.transpiled_circuit).data,
        Operator(circuit).data,
        atol=1e-12,
    )


def test_zero_weight_record_does_not_affect_weighted_minimum_or_maximum() -> None:
    zero_weight = SimpleNamespace(**{name: 999.0 for name in METRICS})
    positive_weight = SimpleNamespace(**{name: 3.0 for name in METRICS})
    accumulator = CompiledMetricAccumulator(weighted=True)
    accumulator.update(zero_weight, weight=0.0)
    accumulator.update(positive_weight, weight=1.0)

    for _name, statistics in accumulator.finalize():
        assert statistics.mean == 3.0
        assert statistics.minimum == 3.0
        assert statistics.maximum == 3.0


def test_recursive_circuit_hash_includes_custom_gate_definition() -> None:
    definition_x = QuantumCircuit(1)
    definition_x.x(0)
    definition_h = QuantumCircuit(1)
    definition_h.h(0)

    circuit_x = QuantumCircuit(1)
    circuit_x.append(definition_x.to_gate(label="same-label"), [0])
    circuit_x_copy = QuantumCircuit(1)
    circuit_x_copy.append(definition_x.copy().to_gate(label="same-label"), [0])
    circuit_h = QuantumCircuit(1)
    circuit_h.append(definition_h.to_gate(label="same-label"), [0])

    assert canonical_qiskit_circuit_fingerprint(
        circuit_x
    ) == canonical_qiskit_circuit_fingerprint(circuit_x_copy)
    assert canonical_qiskit_circuit_fingerprint(
        circuit_x
    ) != canonical_qiskit_circuit_fingerprint(circuit_h)


def test_cost_cache_is_bounded_and_drops_transpiled_circuit_bodies() -> None:
    cache = TranspiledCircuitCostCache(maximum_entries=1)
    compiler = _compiler(optimization_level=0)
    first = QuantumCircuit(1)
    first.rz(0.1, 0)
    second = QuantumCircuit(1)
    second.rz(0.2, 0)

    miss, _key, cached = cache.get_or_transpile(
        first,
        compiler,
        circuit_fingerprint="first",
    )
    assert cached is False
    assert miss.transpiled_circuit is not None
    cache.get_or_transpile(second, compiler, circuit_fingerprint="second")

    assert len(cache) == 1
    assert cache.eviction_count == 1
    assert next(iter(cache._costs.values())).transpiled_circuit is None


def test_cache_validates_backend_context_before_lookup() -> None:
    backend = GenericBackendV2(
        num_qubits=1,
        basis_gates=["rz", "sx", "x"],
        seed=11,
    )
    compiler = replace(
        _compiler(optimization_level=0),
        basis_gates=("rz", "sx", "x"),
        backend_name=backend.name,
    )
    circuit = QuantumCircuit(1)
    circuit.rz(0.2, 0)
    cache = TranspiledCircuitCostCache()
    cache.get_or_transpile(
        circuit,
        compiler,
        circuit_fingerprint="backend-bound",
        backend=backend,
    )

    with pytest.raises(ValueError, match="requires the corresponding backend"):
        cache.get_or_transpile(
            circuit,
            compiler,
            circuit_fingerprint="backend-bound",
            backend=None,
        )


def test_symbolic_circuit_and_uncanonical_backend_bypass_cache(monkeypatch) -> None:
    compiler = _compiler(optimization_level=0)
    symbolic = QuantumCircuit(1)
    symbolic.rz(Parameter("theta"), 0)
    cache = TranspiledCircuitCostCache()
    for _ in range(2):
        _cost, _key, cached = cache.get_or_transpile(
            symbolic,
            compiler,
            circuit_fingerprint="symbolic",
        )
        assert cached is False
    assert len(cache) == 0
    assert cache.bypass_count == 2

    backend = GenericBackendV2(
        num_qubits=1,
        basis_gates=["rz", "sx", "x"],
        seed=11,
    )
    backend_compiler = replace(
        compiler,
        basis_gates=("rz", "sx", "x"),
        backend_name=backend.name,
    )
    numeric = QuantumCircuit(1)
    numeric.rz(0.2, 0)
    monkeypatch.setattr(
        compiled_cost_module,
        "_canonical_backend_fingerprint_or_none",
        lambda _backend: None,
    )
    backend_cache = TranspiledCircuitCostCache()
    for _ in range(2):
        _cost, _key, cached = backend_cache.get_or_transpile(
            numeric,
            backend_compiler,
            circuit_fingerprint="uncanonical-backend",
            backend=backend,
        )
        assert cached is False
    assert len(backend_cache) == 0
    assert backend_cache.bypass_count == 2


def test_same_backend_name_with_different_targets_has_distinct_cache_keys() -> None:
    first_backend = GenericBackendV2(
        num_qubits=2,
        basis_gates=["rz", "sx", "x", "cx"],
        seed=11,
    )
    second_backend = GenericBackendV2(
        num_qubits=2,
        basis_gates=["rz", "sx", "x", "cx"],
        seed=12,
    )
    assert first_backend.name == second_backend.name
    compiler = replace(_compiler(), backend_name=first_backend.name)
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    cache = TranspiledCircuitCostCache()
    _cost, first_key, first_cached = cache.get_or_transpile(
        circuit,
        compiler,
        circuit_fingerprint="target-a",
        backend=first_backend,
    )
    _cost, second_key, second_cached = cache.get_or_transpile(
        circuit,
        compiler,
        circuit_fingerprint="target-b",
        backend=second_backend,
    )
    assert first_cached is False
    assert second_cached is False
    assert first_key != second_key


def test_transpile_preserves_controlled_identity_relative_phase() -> None:
    block = DFBlock(
        U_ops=((RYGate(0.19), (0,)),),
        eta=np.asarray([1.0]),
        lam=0.7,
    )
    extraction = extract_df_diagonal_tail(
        "controlled-identity",
        (block,),
        identity_policy="faithful_identity_in_tail",
    )
    preparation = prepare_df_rte_event_inputs(extraction)
    _config, distribution = make_rte_config(
        preparation.symbolic_tail,
        evolution_time=0.2,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    event = next(
        item
        for item in enumerate_rte_events(
            preparation.symbolic_tail.components,
            distribution,
            max_events=100,
        )
        if item.taylor_order == 0 and item.application_sequence[0].is_identity
    )
    built = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    ).build_event(
        preparation.request_for_event(event, controlled=True, ancilla_qubit=1)
    )
    cost = transpile_and_measure_cost(
        built.circuit,
        _compiler(),
        circuit_fingerprint=built.circuit_fingerprint,
    )
    dense_tail = extraction_to_normalized_rte_tail(extraction)
    operators = dict(
        zip(
            (item.component_id for item in dense_tail.components),
            dense_tail.operators,
            strict=True,
        )
    )
    event_matrix = event_unitary(event, operators)
    identity = np.eye(2, dtype=np.complex128)
    zero = np.zeros_like(identity)
    expected = np.block([[identity, zero], [zero, event_matrix]])

    assert built.relative_ancilla_phase != 0.0
    assert np.allclose(
        Operator(cost.transpiled_circuit).data,
        expected,
        atol=1e-12,
    )


def test_exact_compiled_expectation_is_weighted_once_and_cached() -> None:
    _extraction, preparation, _config, distribution = _one_component_case()
    compiler = _compiler()
    events = enumerate_rte_events(
        preparation.symbolic_tail.components,
        distribution,
    )
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    event_costs = []
    for event in events:
        built = builder.build_event(
            preparation.request_for_event(
                event,
                controlled=True,
                ancilla_qubit=1,
            )
        )
        event_costs.append(
            transpile_and_measure_cost(
                built.circuit,
                compiler,
                circuit_fingerprint=built.circuit_fingerprint,
            )
        )
    manual = {
        name: math.fsum(
            event.event_probability * float(getattr(cost, name))
            for event, cost in zip(events, event_costs, strict=True)
        )
        for name in METRICS
    }
    cache = TranspiledCircuitCostCache()
    estimate = estimate_exact_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        controlled=True,
        ancilla_qubit=1,
        cache=cache,
    )
    repeated = estimate_exact_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        controlled=True,
        ancilla_qubit=1,
        cache=cache,
    )

    assert estimate.event_probability_sum == pytest.approx(1.0)
    assert estimate.standard_error is None
    assert estimate.sample_count is None
    assert estimate.enumerated_event_count == len(events) == 2
    assert estimate.expected_cost.fidelity_level == 3
    assert estimate.expected_cost.estimate_kind == "exact_compiled_expectation"
    assert estimate.planned_build_requests == estimate.actual_build_requests == 2
    assert estimate.planned_transpile_requests == estimate.actual_cache_requests == 2
    assert estimate.actual_built_instruction_total <= (
        estimate.planned_instruction_applications
    )
    for name in METRICS:
        assert getattr(estimate.expected_cost, name) == pytest.approx(manual[name])
    assert len(cache) == 2
    assert repeated.unique_compiled_circuit_count == 2
    assert repeated.transpile_cache_hit_count == 2


def test_event_total_work_limit_rejects_before_builder_and_cache(monkeypatch) -> None:
    _extraction, preparation, _config, distribution = _one_component_case()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("work began before the total-work preflight")

    monkeypatch.setattr(QiskitDFRTEEventCircuitBuilder, "build_event", forbidden)
    monkeypatch.setattr(TranspiledCircuitCostCache, "get_or_transpile", forbidden)
    monkeypatch.setattr(compiled_cost_module, "transpile_and_measure_cost", forbidden)
    with pytest.raises(ValueError, match="build requests"):
        estimate_exact_compiled_event_cost(
            preparation,
            distribution,
            _compiler(),
            maximum_build_requests=1,
        )


def test_exact_compiled_expectation_normalizes_roundoff_and_keeps_raw_sum(
    monkeypatch,
) -> None:
    _extraction, preparation, _config, distribution = _one_component_case()
    compiler = _compiler()
    baseline = estimate_exact_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        controlled=True,
        ancilla_qubit=1,
    )
    events = enumerate_rte_events(
        preparation.symbolic_tail.components,
        distribution,
    )
    scale = 1.0 + 5e-13
    scaled_events = tuple(
        replace(event, event_probability=event.event_probability * scale)
        for event in events
    )
    event_iterator = _OneShotIterator(scaled_events)
    monkeypatch.setattr(
        compiled_cost_module,
        "iter_rte_events",
        lambda *_args, **_kwargs: event_iterator,
    )

    estimate = estimate_exact_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        controlled=True,
        ancilla_qubit=1,
    )

    assert estimate.event_probability_sum != 1.0
    assert estimate.event_probability_sum == pytest.approx(
        scale,
        rel=0.0,
        abs=1e-15,
    )
    for name in METRICS:
        assert getattr(estimate.expected_cost, name) == pytest.approx(
            getattr(baseline.expected_cost, name),
            abs=1e-12,
        )


def test_monte_carlo_statistics_are_unweighted_reproducible_and_convergent() -> None:
    _extraction, preparation, _config, distribution = _one_component_case()
    compiler = _compiler()
    sample_count = 100
    seed = 8
    events = preparation.sample_events(
        distribution,
        sample_count=sample_count,
        seed=seed,
    )
    conditional_rz = {0: 5.0, 2: 6.0}
    values = np.asarray([conditional_rz[event.taylor_order] for event in events])
    expected_mean = float(np.mean(values))
    expected_variance = float(np.var(values, ddof=1))
    expected_standard_error = math.sqrt(expected_variance / sample_count)

    first = estimate_monte_carlo_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        sample_count=sample_count,
        seed=seed,
        controlled=True,
        ancilla_qubit=1,
    )
    second = estimate_monte_carlo_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        sample_count=sample_count,
        seed=seed,
        controlled=True,
        ancilla_qubit=1,
    )
    rz_statistics = dict(first.metric_statistics)["rz_count"]
    double_weighted = math.fsum(
        event.event_probability * value
        for event, value in zip(events, values, strict=True)
    ) / sample_count

    assert first.sample_count == sample_count
    assert first.seed == seed
    assert first.ancilla_qubit == 1
    assert first.maximum_work_items == 10_000
    assert first.event_stream_rolling_digest == second.event_stream_rolling_digest
    assert first.expected_cost.rz_count == pytest.approx(expected_mean)
    assert rz_statistics.unbiased_sample_variance == pytest.approx(expected_variance)
    assert rz_statistics.standard_error == pytest.approx(expected_standard_error)
    assert first.standard_error is not None
    assert first.standard_error.rz_count == pytest.approx(expected_standard_error)
    assert first.expected_cost.rz_count != pytest.approx(double_weighted)
    assert first.expected_cost == second.expected_cost
    assert first.standard_error == second.standard_error

    with pytest.raises(ValueError, match="maximum_samples"):
        estimate_monte_carlo_compiled_event_cost(
            preparation,
            distribution,
            compiler,
            sample_count=2,
            seed=seed,
            maximum_samples=1,
        )
    assert first.unique_compiled_circuit_count == 2
    assert first.transpile_cache_hit_count == sample_count - 2

    exact = estimate_exact_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        controlled=True,
        ancilla_qubit=1,
    )
    large_sample = estimate_monte_carlo_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        sample_count=2_000,
        seed=23,
        controlled=True,
        ancilla_qubit=1,
    )
    assert large_sample.standard_error is not None
    assert abs(
        large_sample.expected_cost.rz_count - exact.expected_cost.rz_count
    ) <= 5.0 * large_sample.standard_error.rz_count


def test_short_occurrence_reports_nonadditive_compiled_cost_and_limits() -> None:
    _extraction, preparation, config, distribution = _one_component_case(
        rte_steps=3
    )
    compiler = _compiler()
    estimate = estimate_compiled_occurrence_cost(
        preparation,
        config,
        distribution,
        compiler,
        sequence_sample_count=12,
        seed=3,
    )

    assert estimate.event_count_per_sample == config.rte_steps == 3
    assert estimate.sample_count == 12
    assert estimate.sequence_expected_cost.fidelity_level == 4
    assert estimate.additive_expected_cost.fidelity_level == 4
    assert estimate.nonadditive_difference.fidelity_level == 4
    for name in METRICS:
        assert getattr(estimate.nonadditive_difference, name) == pytest.approx(
            getattr(estimate.sequence_expected_cost, name)
            - getattr(estimate.additive_expected_cost, name)
        )
    assert estimate.sequence_expected_cost.rz_count < (
        estimate.additive_expected_cost.rz_count
    )
    assert estimate.nonadditive_difference.rz_count < 0.0
    assert estimate.unique_sequence_circuit_count >= 1
    assert estimate.transpile_cache_hit_count > 0
    assert estimate.planned_build_requests == estimate.actual_build_requests == 48
    assert estimate.planned_transpile_requests == estimate.actual_cache_requests == 48
    assert estimate.actual_built_instruction_total <= (
        estimate.planned_instruction_applications
    )

    with pytest.raises(ValueError, match="maximum_rte_steps"):
        estimate_compiled_occurrence_cost(
            preparation,
            config,
            distribution,
            compiler,
            sequence_sample_count=1,
            seed=3,
            maximum_rte_steps=2,
        )
    with pytest.raises(ValueError, match="size limit"):
        estimate_compiled_occurrence_cost(
            preparation,
            config,
            distribution,
            compiler,
            sequence_sample_count=1,
            seed=3,
            maximum_untranspiled_circuit_size=1,
        )


def test_occurrence_request_validates_config_and_distribution_identity() -> None:
    _extraction, preparation, config, distribution = _one_component_case()
    request = preparation.sample_occurrence_request(
        config,
        distribution,
        seed=4,
    )
    assert len(request.events) == config.rte_steps
    assert request.occurrence_rte_steps == config.rte_steps

    with pytest.raises(ValueError, match="tail hash"):
        preparation.sample_occurrence_request(
            replace(config, tail_hash="wrong"),
            distribution,
            seed=4,
        )
    with pytest.raises(ValueError, match="tail ID"):
        preparation.sample_occurrence_request(
            replace(config, tail_id="wrong"),
            distribution,
            seed=4,
        )
    with pytest.raises(ValueError, match="Taylor cutoffs"):
        preparation.sample_occurrence_request(
            config,
            finite_rte_distribution(1.2, 4),
            seed=4,
        )
    with pytest.raises(ValueError, match="step times"):
        preparation.sample_occurrence_request(
            config,
            finite_rte_distribution(0.9, 2),
            seed=4,
        )
    with pytest.raises(ValueError, match="order weights"):
        replace(
            distribution,
            exact_finite_distribution=(
                1.01 * distribution.exact_finite_distribution
            ),
        )


def test_compiler_hash_separates_settings_and_version_mismatch_is_rejected() -> None:
    compiler = _compiler(seed=1)
    assert compiler_settings_hash(compiler) != compiler_settings_hash(
        replace(compiler, transpiler_seed=2)
    )
    circuit = QuantumCircuit(1)
    with pytest.raises(ValueError, match="qiskit_version"):
        transpile_and_measure_cost(
            circuit,
            replace(compiler, qiskit_version="different"),
        )


def test_cache_separates_compiler_control_and_reuse_conditions() -> None:
    _extraction, preparation, _config, distribution = _one_component_case()
    compiler = _compiler(seed=1)
    cache = TranspiledCircuitCostCache()

    estimate_exact_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        cache=cache,
    )
    initial_size = len(cache)
    estimate_exact_compiled_event_cost(
        preparation,
        distribution,
        replace(compiler, transpiler_seed=2),
        cache=cache,
    )
    after_compiler_change = len(cache)
    estimate_exact_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        controlled=True,
        ancilla_qubit=1,
        cache=cache,
    )
    after_control_change = len(cache)
    estimate_exact_compiled_event_cost(
        preparation,
        distribution,
        compiler,
        cancel_adjacent_equal_bases=False,
        cache=cache,
    )

    assert initial_size == 2
    assert after_compiler_change == 4
    assert after_control_change == 6
    # Reuse policy is provenance, not circuit semantics.  One no-op-policy
    # event is therefore deduplicated by the actual recursive circuit hash.
    assert len(cache) == 7
    assert cache.hit_count >= 1

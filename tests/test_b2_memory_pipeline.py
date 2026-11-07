"""Artificial whole-wrapper regressions; no molecular data or price MC."""
import gc
import os
import resource
import time
import weakref
from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import SXGate
from qiskit.quantum_info import Operator

from b2_cost_method_cases import artificial_round, event_pattern, expected_controlled, measurement_error
from trotterlib.b2_p1_compiled_cost import price_explicit_trajectory, price_streamed_trajectory, iter_price_streamed_trajectories
from trotterlib.b2_streamed_native_cost import cached_compiler, measured_metrics, stream_native
from trotterlib.o1_m1_resource_bridge import MatchedNoPrepCostProvider
from trotterlib.rpe_hadamard_interrogation import wrap_controlled_no_prep
from trotterlib.rte_compiled_cost import (TranspiledCircuitCostCache, canonical_qiskit_circuit_fingerprint,
    measure_compiled_metrics, transpile_and_measure_cost)
from trotterlib.streamed_circuit_fingerprint import streamed_canonical_qiskit_circuit_fingerprint as streamed_hash
from trottertracks.o1_m1_parallel_price import ParallelMatchedNoPrepCostProvider
from trottertracks.resource_applicability.ax2a_native_df import build_native_hadamard_wrapper


@pytest.fixture(autouse=True)
def artificial_only(monkeypatch):
    def forbidden(*a, **k): raise AssertionError('No H4, molecular I/O or price MC')
    monkeypatch.setattr(np, 'load', forbidden)
    import trotterlib.df_hamiltonian as h
    monkeypatch.setattr(h, 'solve_df_ground_state', forbidden)
    monkeypatch.setattr(h, 'build_df_h_d_from_molecule', forbidden)
    monkeypatch.setattr(MatchedNoPrepCostProvider, 'monte_carlo', forbidden)
    monkeypatch.setattr(MatchedNoPrepCostProvider, 'monte_carlo_with_builder', forbidden)
    monkeypatch.setattr(ParallelMatchedNoPrepCostProvider, 'monte_carlo', forbidden)


@pytest.mark.parametrize('seed', [7, 21, 101])
def test_one_pass_metrics_match_qiskit_with_sync_and_classical_wires(seed):
    rng = np.random.default_rng(seed)
    c = QuantumCircuit(5, 2)
    for _ in range(170):
        i, j = rng.choice(5, size=2, replace=False)
        op = rng.integers(6)
        if op == 0: c.cx(int(i), int(j))
        elif op == 1: c.rz(float(rng.normal()), int(i))
        elif op == 2: c.sx(int(i))
        elif op == 3: c.x(int(i))
        elif op == 4: c.barrier(int(i), int(j))
        else: c.measure(int(i), int(j)%2)
    assert measure_compiled_metrics(c) == measured_metrics(c)
    c.h(0)  # unsupported common-path operation must retain reference semantics
    assert measure_compiled_metrics(c) == measured_metrics(c)
    c.x(1).c_if(c.cregs[0], 1)
    assert measure_compiled_metrics(c) == measured_metrics(c)
    assert measure_compiled_metrics(QuantumCircuit()) == measured_metrics(QuantumCircuit())


def test_axis_lifetimes_and_full_wrapper_meaning():
    h, r = artificial_round(0, 2)
    evolution = stream_native(r, event_pattern(r, 'mixed'), fingerprint_function=streamed_hash)
    assert np.max(abs(Operator(evolution.circuit).data - expected_controlled(h, r, tuple(event_pattern(r, 'mixed'))))) < 1e-10
    old_wrappers, old_outputs = [], []
    def wrapper(value, axis):
        gc.collect()
        assert all(w() is None for w in old_wrappers + old_outputs)
        c = build_native_hadamard_wrapper(value, axis=axis,
            max_instructions=len(value.circuit.data)+4, fingerprint_function=streamed_hash)
        old_wrappers.append(weakref.ref(c))
        return c
    def compile(circuit, compiler, **kw):
        cost = transpile_and_measure_cost(circuit, compiler, metric_function=measure_compiled_metrics, **kw)
        errors = measurement_error(circuit, cost.transpiled_circuit)
        assert max(errors.values()) < 1e-10
        assert cost.transpiled_circuit.count_ops()['measure'] == 1
        assert measure_compiled_metrics(cost.transpiled_circuit) == measured_metrics(cost.transpiled_circuit)
        old_outputs.append(weakref.ref(cost.transpiled_circuit))
        return cost
    provider = MatchedNoPrepCostProvider(r.context.compiler, compile_function=compile, memory_efficient=True)
    pair = provider.compile_pair(evolution, wrapper_builder=wrapper)
    gc.collect()
    assert all(w() is None for w in old_wrappers + old_outputs)
    assert all(c.transpiled_circuit is None for c in pair.values())


@pytest.mark.parametrize('index', [0, 1, 3])
def test_streamed_price_matches_all_six_metrics_identity_and_scope(index):
    _, r = artificial_round(index, 2)
    reference = price_explicit_trajectory(r, MatchedNoPrepCostProvider(r.context.compiler), event_pattern(r, 'mixed'))
    provider = MatchedNoPrepCostProvider(r.context.compiler, memory_efficient=True)
    candidate = price_streamed_trajectory(r, provider, event_pattern(r, 'mixed'))
    for name in ('axes', 'trajectory_identity', 'evolution_fingerprint', 'evolution_instructions',
            'context_hash', 'preparation_hash', 'round_binding', 'compiler_settings_hash',
            'circuit_cost_scope', 'state_preparation_included', 'evaluation_method'):
        assert candidate[name] == reference[name]
    assert candidate['trajectory'] is None and candidate['trajectory_event_count'] == r.q*r.r
    assert candidate['expected_trajectory_cost'] is candidate['classical_sample_count'] is candidate['total_rpe_cost'] is None
    again = price_streamed_trajectory(r, provider, event_pattern(r, 'mixed'), retain_trajectory=True)
    assert again['trajectory'] == reference['trajectory']
    assert provider.cost_cache.miss_count == 2 and provider.cost_cache.hit_count == 2
    assert all(c.transpiled_circuit is None for c in provider.cost_cache._costs.values())


def test_cache_rehashes_each_input_and_preserves_persistent_keys(tmp_path):
    _, r = artificial_round(0, 2)
    calls = []
    def hasher(c): calls.append(1); return streamed_hash(c)
    path = tmp_path/'metrics.sqlite'
    cache = TranspiledCircuitCostCache(persistent_path=path, fingerprint_function=hasher,
        metric_function=measure_compiled_metrics)
    compile = cached_compiler(cache)
    body = QuantumCircuit(3); body.crz(.2, 2, 0)
    c = wrap_controlled_no_prep(body, ancilla_qubit=2, axis='cosine')
    a = compile(c, r.context.compiler); b = compile(c, r.context.compiler)
    assert len(calls) == 2 and cache.miss_count == 1 and cache.hit_count == 1
    assert a.actual_circuit_fingerprint == canonical_qiskit_circuit_fingerprint(c)
    assert b.transpiled_circuit is None
    c.rz(.123, 0); changed = compile(c, r.context.compiler)
    assert changed.actual_circuit_fingerprint != a.actual_circuit_fingerprint and cache.miss_count == 2
    compile(c, replace(r.context.compiler, transpiler_seed=8))
    assert cache.miss_count == 3
    fresh = TranspiledCircuitCostCache(persistent_path=path, fingerprint_function=streamed_hash,
        metric_function=measure_compiled_metrics)
    restored = cached_compiler(fresh)(c, r.context.compiler)
    assert fresh.persistent_hit_count == 1 and restored.transpiled_circuit is None
    assert restored.actual_circuit_fingerprint == changed.actual_circuit_fingerprint
    # Local operation memo must never hide definition mutation between calls.
    gate = SXGate(); definition = gate.definition
    before = streamed_hash(QuantumCircuit(1))
    c2 = QuantumCircuit(1); c2.sx(0)
    baseline = streamed_hash(c2)
    try:
        definition.rz(.071, 0)
        assert streamed_hash(c2) == canonical_qiskit_circuit_fingerprint(c2) != baseline
    finally:
        definition.data.pop()
    assert streamed_hash(c2) == baseline and streamed_hash(QuantumCircuit(1)) == before


def test_lazy_trajectory_tasks_and_bad_events_do_not_compile():
    _, r = artificial_round(0, 2)
    provider = MatchedNoPrepCostProvider(r.context.compiler, memory_efficient=True)
    consumed = []
    def tasks():
        for i in range(3): consumed.append(i); yield event_pattern(r, 'mixed')
    output = iter_price_streamed_trajectories(r, provider, tasks())
    next(output); assert consumed == [0]
    list(output); assert consumed == [0, 1, 2]
    assert provider.cost_cache.miss_count == 2 and provider.cost_cache.hit_count == 4
    for events in ([], [*event_pattern(r, 'mixed'), next(event_pattern(r, 'mixed'))]):
        with pytest.raises(ValueError): price_streamed_trajectory(r, provider, events)
    assert provider.cost_cache.miss_count == 2


def scheduler_task(value):
    time.sleep(.005*(3-value%3))
    return os.getpid(), value, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@pytest.mark.parametrize('probe', [False, True])
def test_existing_scheduler_consumes_bounded_tasks_in_order(probe):
    _, r = artificial_round(0, 2)
    # Two worker slots are the fixture under test, not a production resource cap.
    provider = ParallelMatchedNoPrepCostProvider(r.context.compiler, max_workers=2)
    consumed = []
    def tasks():
        for i in range(9): consumed.append(i); yield i
    stream = provider._iter_map(scheduler_task, tasks(), memory_probe=probe, task_count=9)
    result = [next(stream)]
    assert len(consumed) <= 2
    result.extend(stream)
    assert result == list(range(9))
    assert provider.last_execution['task_count'] == 9
    assert provider.last_execution['peak_pending_futures'] <= 2
    assert provider._map(scheduler_task, [], memory_probe=probe) == []
    assert provider._map(scheduler_task, [3], memory_probe=probe) == [3]

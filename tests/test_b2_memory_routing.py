"""Price entry wiring only: replace MC dispatch; never perform price MC."""
import pytest
import gc
import os
import resource
import time
import weakref
from types import SimpleNamespace
from qiskit import QuantumCircuit
from b2_cost_method_cases import artificial_round, event_pattern
from trotterlib.b2_p1_compiled_cost import request_instruction_bound
from trotterlib.b2_streamed_native_cost import structural_instruction_bound
from trotterlib.b2_streamed_native_cost import measured_metrics
from trotterlib.o1_m1_resource_bridge import MatchedNoPrepCostProvider
from trotterlib.streamed_circuit_fingerprint import streamed_canonical_qiskit_circuit_fingerprint
from trottertracks.o1_m1_parallel_price import ParallelMatchedNoPrepCostProvider
from trotterlib.rte_compiled_cost import measure_compiled_metrics
from trotterlib.rte_compiled_cost import transpile_and_measure_cost
from trotterlib.b2_p1_compiled_cost import price_explicit_trajectory
from trotterlib.rpe_hadamard_interrogation import wrap_controlled_no_prep


@pytest.mark.parametrize('efficient', [False, True])
def test_b2_entry_selects_streaming_without_executing_price_mc(monkeypatch, efficient):
    _, round_ = artificial_round(0, 2)
    provider = MatchedNoPrepCostProvider(round_.context.compiler, memory_efficient=efficient)
    observed = {}
    sentinel = object()
    def dispatch(*args, **kwargs):
        observed.update(kwargs)
        return sentinel
    monkeypatch.setattr(provider, 'monte_carlo_with_builder', dispatch)
    bound = structural_instruction_bound(round_)+4
    assert round_.price(provider, sample_count=2, seed=7, max_instructions=bound) is sentinel
    assert observed.get('stream_events', False) is efficient
    assert observed['sample_count'] == 2 and observed['seed'] == 7
    # Exercise only the supplied fixed-event builders, without invoking a sampler,
    # a compiler, statistical reduction, or a price MC implementation.
    evolution = observed['build_trajectory'](event_pattern(round_, 'mixed'))
    request = round_.explicit_request(event_pattern(round_, 'mixed'))
    reference = round_.native(request, max_instructions=request_instruction_bound(request))
    assert evolution.circuit_fingerprint == reference.circuit_fingerprint
    for axis in ('cosine', 'sine'):
        wrapped = observed['wrapper_builder'](evolution, axis)
        assert wrapped.count_ops()['measure'] == 1
        assert wrapped.num_clbits == 1
        assert streamed_canonical_qiskit_circuit_fingerprint(wrapped)
    if efficient:
        with pytest.raises(RuntimeError, match='PRE_BUILD_INSTRUCTION_BUDGET'):
            round_.price(provider, sample_count=2, seed=7, max_instructions=1)


class Payload:
    def __init__(self, value):
        self.value = value
        self.buffer = bytearray(1024*1024)


def payload_worker(payload):
    return os.getpid(), payload.value, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def test_scheduler_releases_completed_first_task_payload():
    _, round_ = artificial_round(0, 2)
    # The fixture has two slots and three ordinary data tasks; no price worker.
    provider = ParallelMatchedNoPrepCostProvider(round_.context.compiler, max_workers=2)
    references = []
    def tasks():
        for i in range(3):
            item = Payload(i)
            references.append(weakref.ref(item))
            yield item
    stream = provider._iter_map(payload_worker, tasks(), task_count=3)
    assert next(stream) == 0
    for _ in range(20):
        gc.collect()
        if references[0]() is None:
            break
        time.sleep(.01)
    assert references[0]() is None
    assert list(stream) == [1, 2]


def test_conditional_basis_metrics_keep_classical_dependencies():
    c = QuantumCircuit(3, 1)
    c.rz(.2, 0); c.measure(0, 0); c.x(1).c_if(c.cregs[0], 1)
    c.cx(1, 2); c.barrier(); c.rz(.3, 2)
    assert measure_compiled_metrics(c) == measured_metrics(c)


def test_existing_duck_provider_keeps_legacy_entries():
    from test_b2_rpe_connection import fake_compile
    _, round_ = artificial_round(0, 2)
    sentinel = object()
    provider = SimpleNamespace(compiler=round_.context.compiler, backend_hash='no_backend',
        backend=None, compile_function=fake_compile, monte_carlo_with_builder=lambda *a, **k:sentinel)
    assert round_.price(provider, sample_count=2, seed=7,
        max_instructions=structural_instruction_bound(round_)+4) is sentinel
    priced = price_explicit_trajectory(round_, provider, event_pattern(round_, 'mixed'))
    assert set(priced['axes']) == {'cosine', 'sine'}
    assert priced['expected_trajectory_cost'] is None


def test_pair_releases_axis_objects_without_observer_gc():
    _, round_ = artificial_round(0, 2)
    body = QuantumCircuit(3); body.crz(.2, 2, 0)
    references = []
    def wrapper(c, axis):
        assert all(ref() is None for ref in references)
        w = wrap_controlled_no_prep(c, ancilla_qubit=2, axis=axis)
        references.append(weakref.ref(w))
        return w
    def compile(c, settings, **kw):
        cost = transpile_and_measure_cost(c, settings, metric_function=measure_compiled_metrics, **kw)
        references.append(weakref.ref(cost.transpiled_circuit))
        return cost
    provider = MatchedNoPrepCostProvider(round_.context.compiler,
        memory_efficient=True, compile_function=compile)
    pair = provider.compile_pair(body, wrapper_builder=wrapper)
    assert all(ref() is None for ref in references)
    assert all(c.transpiled_circuit is None for c in pair.values())

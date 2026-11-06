"""Synthetic correctness/cache/partition checks; no saved molecular data."""
from dataclasses import replace
from itertools import islice
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from b2_cost_method_cases import artificial_round,event_pattern,expected_controlled
from trotterlib.b2_p1_compiled_cost import request_instruction_bound
from trotterlib.b2_streamed_native_cost import (stream_native,iter_b2_step_requests,
    cached_compiler,split_compile,measured_metrics)
from trotterlib.rte import iter_sample_rte_events
from trotterlib.rte_compiled_cost import (TranspiledCircuitCostCache,CompiledMetricAccumulator,
    canonical_qiskit_circuit_fingerprint,transpile_and_measure_cost)
from trottertracks.resource_applicability.ax2a_native_df import build_native_hadamard_wrapper


@pytest.fixture(autouse=True)
def no_saved_inputs(monkeypatch):
    def forbidden(*a,**k):
        raise AssertionError('No molecular inputs/solvers')
    monkeypatch.setattr(np,'load',forbidden)
    import trotterlib.df_hamiltonian as h
    monkeypatch.setattr(h,'solve_df_ground_state',forbidden)
    monkeypatch.setattr(h,'build_df_h_d_from_molecule',forbidden)


@pytest.mark.parametrize('index',[0,1,2])
@pytest.mark.parametrize('ld',[2,3])
def test_stream_has_identical_gate_identity_and_controlled_meaning(index,ld):
    h,r = artificial_round(index,ld)
    request = r.explicit_request(tuple(event_pattern(r,'mixed')))
    original = r.native(request,max_instructions=request_instruction_bound(request))
    streamed = stream_native(r,event_pattern(r,'mixed'))
    assert streamed.circuit_fingerprint == original.circuit_fingerprint
    assert streamed.circuit.data == original.circuit.data
    assert np.max(abs(Operator(streamed.circuit).data-
        expected_controlled(h,r,tuple(event_pattern(r,'mixed'))))) < 1e-10


def test_requests_consume_only_r_events_at_a_time():
    _,r = artificial_round(3,2)
    consumed = []
    def source():
        for event in event_pattern(r,'mixed'):
            consumed.append(1)
            yield event
    requests = iter_b2_step_requests(r,source())
    next(requests)
    assert len(consumed)==r.r
    next(requests)
    assert len(consumed)==2*r.r
    assert sum(1 for _ in requests)==r.q-2
    assert len(consumed)==r.q*r.r


@pytest.mark.parametrize('defect',['short','extra','phase'])
def test_stream_refuses_incomplete_extra_or_noncanonical_events(defect):
    _,r = artificial_round(0,2)
    events = list(event_pattern(r,'mixed'))
    if defect=='short': events.pop()
    elif defect=='extra': events.append(events[0])
    else: events[0]=replace(events[0],rotation_angle=events[0].rotation_angle+.1)
    with pytest.raises(ValueError):
        stream_native(r,iter(events))


def test_cache_preserves_repeated_observations_and_keys_actual_circuit(tmp_path):
    _,r = artificial_round(0,2)
    evolution=stream_native(r,event_pattern(r,'mixed'))
    wrapper=build_native_hadamard_wrapper(evolution,axis='cosine',max_instructions=len(evolution.circuit.data)+4)
    cache=TranspiledCircuitCostCache(persistent_path=tmp_path/'cost.sqlite')
    compile=cached_compiler(cache)
    a=compile(wrapper,r.context.compiler)
    b=compile(wrapper,r.context.compiler)
    assert cache.miss_count==1 and cache.hit_count==1
    accumulator=CompiledMetricAccumulator(weighted=False)
    for cost in (a,b,b): accumulator.update(cost)
    assert accumulator.count==3
    assert dict(accumulator.finalize())['cx_count'].mean==a.cx_count
    fresh=TranspiledCircuitCostCache(persistent_path=tmp_path/'cost.sqlite')
    restored=cached_compiler(fresh)(wrapper,r.context.compiler)
    assert fresh.persistent_hit_count==1 and restored.cx_count==a.cx_count
    changed=wrapper.copy();changed.rz(.123,0)
    compile(changed,r.context.compiler)
    assert cache.miss_count==2  # caller labels never stand in for gate identity


def test_existing_sampler_sequence_survives_streaming_without_resampling():
    _,r = artificial_round(1,2)
    def draw(): return iter_sample_rte_events(r.tail.components,r.distribution,
        sample_count=r.q*r.r,seed=771)
    expected=tuple(draw())
    observed=tuple(e for step in iter_b2_step_requests(r,draw()) for e in step.rte_occurrence.events)
    assert observed==expected


def test_split_compile_is_semantically_valid_but_not_whole_price():
    _,r=artificial_round(0,2)
    circuit=QuantumCircuit(3,1)
    circuit.h(2);circuit.sx(0);circuit.rz(.2,0);circuit.x(1);circuit.cx(0,1)
    circuit.cx(0,1);circuit.rz(-.2,0);circuit.x(1);circuit.h(2);circuit.measure(2,0)
    whole=transpile_and_measure_cost(circuit,r.context.compiler)
    split,sums=split_compile(circuit,r.context.compiler,pieces=2)
    assert measured_metrics(split)['cx_count'] != whole.cx_count
    a=Operator(circuit.remove_final_measurements(inplace=False)).data
    b=Operator(split.remove_final_measurements(inplace=False)).data
    assert np.max(abs(np.kron(a.conj(),a)-np.kron(b.conj(),b)))<1e-10
    assert split.count_ops()['measure']==1

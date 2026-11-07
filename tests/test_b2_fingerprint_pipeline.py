"""Opt-in fingerprint integration: artificial two-mode B2 only."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from b2_cost_method_cases import artificial_round, event_pattern, expected_controlled
from trotterlib.b2_p1_compiled_cost import request_instruction_bound
from trotterlib.b2_streamed_native_cost import stream_native
from trotterlib.o1_m1_resource_bridge import _compile_pair
from trotterlib.rpe_resource_accounting import RPE_COST_METRICS
from trotterlib.rte_compiled_cost import canonical_qiskit_circuit_fingerprint as legacy
from trotterlib.rte_compiled_cost import transpile_and_measure_cost
from trotterlib.streamed_circuit_fingerprint import streamed_canonical_qiskit_circuit_fingerprint as streamed
from trottertracks.resource_applicability.ax2a_native_df import build_native_hadamard_wrapper


@pytest.fixture(autouse=True)
def no_molecular_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Artificial circuits only')
    monkeypatch.setattr(np, 'load', forbidden)
    import trotterlib.df_hamiltonian as h
    monkeypatch.setattr(h, 'solve_df_ground_state', forbidden)
    monkeypatch.setattr(h, 'build_df_h_d_from_molecule', forbidden)


@pytest.mark.parametrize('m', [0, 1, 3])
def test_opt_in_native_and_xy_keep_identity_and_controlled_meaning(m):
    h, r = artificial_round(m, 2)
    events = tuple(event_pattern(r, 'mixed'))
    request = r.explicit_request(events)
    reference = r.native(request, max_instructions=request_instruction_bound(request))
    calls = []
    def hasher(circuit):
        calls.append(len(circuit.data))
        return streamed(circuit)
    candidate = stream_native(r, iter(events), fingerprint_function=hasher)
    assert len(calls) == 1
    assert candidate.circuit_fingerprint == reference.circuit_fingerprint
    assert candidate.circuit.data == reference.circuit.data
    assert candidate.circuit.global_phase == reference.circuit.global_phase
    assert np.max(abs(Operator(candidate.circuit).data - expected_controlled(h, r, events))) < 1e-10
    for axis in ('cosine', 'sine'):
        a = build_native_hadamard_wrapper(reference, axis=axis,
            max_instructions=len(reference.circuit.data) + 4)
        b = build_native_hadamard_wrapper(candidate, axis=axis,
            max_instructions=len(candidate.circuit.data) + 4, fingerprint_function=hasher)
        assert a.data == b.data and a.global_phase == b.global_phase
        assert streamed(b) == legacy(a)
        assert b.num_qubits == 3 and b.num_clbits == 1 and b.count_ops()['measure'] == 1
    assert len(calls) == 3


@pytest.mark.parametrize('hasher', [legacy, streamed])
def test_wrapper_mutation_check_remains_enforced(hasher):
    _, r = artificial_round(0, 2)
    evolution = stream_native(r, event_pattern(r, 'mixed'), fingerprint_function=hasher)
    evolution.circuit.rz(.123, 0)
    with pytest.raises(ValueError, match='mutated'):
        build_native_hadamard_wrapper(evolution, axis='cosine',
            max_instructions=len(evolution.circuit.data) + 4, fingerprint_function=hasher)


def test_opt_in_hasher_error_is_propagated_without_fallback():
    _, r = artificial_round(0, 2)
    def failed(circuit):
        raise RuntimeError('hasher failed')
    with pytest.raises(RuntimeError, match='hasher failed'):
        stream_native(r, event_pattern(r, 'mixed'), fingerprint_function=failed)
    evolution = stream_native(r, event_pattern(r, 'mixed'))
    with pytest.raises(RuntimeError, match='hasher failed'):
        build_native_hadamard_wrapper(evolution, axis='cosine',
            max_instructions=len(evolution.circuit.data) + 4, fingerprint_function=failed)


def test_existing_pair_compiler_uses_selected_hash_and_keeps_both_compiled_axes():
    _, r = artificial_round(0, 2)
    evolution = stream_native(r, event_pattern(r, 'mixed'), fingerprint_function=streamed)
    def wrapper(value, axis):
        return build_native_hadamard_wrapper(value, axis=axis,
            max_instructions=len(value.circuit.data)+4, fingerprint_function=streamed)
    old = _compile_pair(evolution, r.context.compiler,
        compile_function=transpile_and_measure_cost, backend=None, wrapper_builder=wrapper)
    seen = []
    def hash_wrapper(circuit):
        value = streamed(circuit); seen.append(value)
        return value
    new = _compile_pair(evolution, r.context.compiler,
        compile_function=transpile_and_measure_cost, backend=None,
        wrapper_builder=wrapper, fingerprint_function=hash_wrapper)
    assert len(seen) == 2
    for axis in ('cosine', 'sine'):
        assert new[axis].actual_circuit_fingerprint == seen[('cosine','sine').index(axis)]
        assert streamed(new[axis].transpiled_circuit) == legacy(old[axis].transpiled_circuit)
        assert {k:getattr(new[axis],k) for k in RPE_COST_METRICS} == {k:getattr(old[axis],k) for k in RPE_COST_METRICS}

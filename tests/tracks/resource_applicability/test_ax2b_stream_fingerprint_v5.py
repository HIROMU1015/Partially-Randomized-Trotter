"""Canonical byte compatibility and memory checks on fixed synthetic circuits."""
import builtins
import gc
import hashlib
import json
from pathlib import Path
import tracemalloc

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import RZGate, UnitaryGate

from trotterlib.rte_compiled_cost import _circuit_payload, canonical_qiskit_circuit_fingerprint as legacy
from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import write_canonical_bytes, canonical_qiskit_circuit_fingerprint as streaming


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    original, path_open = builtins.open, Path.open
    def check(path):
        if isinstance(path, (str, Path)) and (str(path).endswith('.npz') or '/artifacts/' in str(path)):
            raise AssertionError('Scientific file access forbidden')
    def guarded(path, *args, **kwargs):
        check(path); return original(path, *args, **kwargs)
    def guarded_path(path, *args, **kwargs):
        check(path); return path_open(path, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Scientific loading, sampling or compilation forbidden')
    monkeypatch.setattr(builtins, 'open', guarded)
    monkeypatch.setattr(Path, 'open', guarded_path)
    monkeypatch.setattr(np, 'load', forbidden)
    import qiskit
    import trotterlib.rte_compiled_cost as cost
    import trotterlib.rte as rte
    monkeypatch.setattr(qiskit, 'transpile', forbidden)
    monkeypatch.setattr(cost, 'transpile', forbidden)
    monkeypatch.setattr(rte, 'sample_rte_events', forbidden)


def fixture(kind):
    circuit = QuantumCircuit(3, 2, name='合成')
    circuit.global_phase = -.17
    if kind == 'empty':
        return circuit
    if kind == 'nested':
        inner = QuantumCircuit(1, name='定義')
        inner.rz(.23, 0); inner.p(-.19, 0); inner.global_phase = .31
        outer = QuantumCircuit(2, name='outer')
        outer.append(inner.to_gate().control(ctrl_state=0), [0, 1])
        for _ in range(3): circuit.append(outer.to_gate().control(), [0, 1, 2])
    elif kind == 'array':
        circuit.append(UnitaryGate(np.diag([1., 1j]), label='fixture'), [2])
    elif kind == 'conditional':
        circuit.rz(.27, 1).c_if(circuit.clbits[1], 1)
        circuit.x(2).c_if(circuit.clbits[0], 1)
    elif kind == 'leaves':
        operation = Instruction('literal', 1, 0, [True, 2, '数値', None, -.0, complex(.2, -.3), [.2, 1]])
        circuit.append(operation, [0])
    else:
        circuit.h(0); circuit.cx(0, 1); circuit.crz(-.31, 1, 2)
        circuit.barrier(); circuit.reset(2); circuit.measure(0, 1)
    return circuit


@pytest.mark.parametrize('kind', ['empty', 'nested', 'array', 'conditional', 'leaves', 'standard'])
def test_every_canonical_byte_and_hash_match_legacy(kind):
    circuit = fixture(kind)
    expected = json.dumps(_circuit_payload(circuit), sort_keys=True, separators=(',', ':')).encode()
    actual = bytearray()
    write_canonical_bytes(circuit, actual.extend)
    assert bytes(actual) == expected
    assert streaming(circuit) == legacy(circuit) == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize('mutation', ['global_phase', 'angle', 'definition', 'control_state', 'qubits', 'clbits'])
def test_parameter_phase_nested_control_and_register_changes_affect_same_identity(mutation):
    first, second = fixture('standard'), fixture('standard')
    if mutation == 'global_phase': second.global_phase += .01
    elif mutation == 'angle': second.rz(.013, 0)
    elif mutation == 'definition':
        one, two = QuantumCircuit(1, name='same'), QuantumCircuit(1, name='same')
        one.x(0); two.h(0)
        first.append(one.to_gate(), [0]); second.append(two.to_gate(), [0])
    elif mutation == 'control_state':
        first.append(RZGate(.17).control(ctrl_state=0), [0, 1])
        second.append(RZGate(.17).control(ctrl_state=1), [0, 1])
    elif mutation == 'qubits': first.rz(.013, 0); second.rz(.013, 1)
    else: first.measure(1, 0); second.measure(1, 1)
    assert streaming(first) == legacy(first)
    assert streaming(second) == legacy(second)
    assert streaming(first) != streaming(second)


@pytest.mark.parametrize('kind', ['symbolic', 'nan', 'object_array', 'layout', 'calibration', 'recursive', 'classical_register'])
def test_legacy_rejections_are_preserved(kind):
    circuit = QuantumCircuit(1)
    if kind == 'symbolic': circuit.rz(Parameter('x'), 0)
    elif kind == 'nan': circuit.append(Instruction('literal', 1, 0, [float('nan')]), [0])
    elif kind == 'object_array': circuit.append(Instruction('literal', 1, 0, [np.array(['x'], dtype=object)]), [0])
    elif kind == 'layout': circuit._layout = object()
    elif kind == 'calibration': circuit._calibrations = {'x': {'synthetic': True}}
    elif kind == 'classical_register':
        circuit = QuantumCircuit(1, 2)
        circuit.x(0).c_if(circuit.cregs[0], 2)
    else:
        gate = Gate('cycle', 1, [])
        definition = QuantumCircuit(1)
        definition.append(gate, [0], copy=False)
        gate.definition = definition
        circuit.append(gate, [0], copy=False)
    for function in (legacy, streaming):
        with pytest.raises((ValueError, TypeError, CircuitError)):
            function(circuit)


def test_repeated_definition_python_memory_is_smaller_without_changing_bytes():
    inner = QuantumCircuit(1)
    inner.rz(.13, 0); inner.p(-.17, 0)
    # Observe total traced peak AND allocations retained after hashing. Qiskit
    # also constructs definition graphs while traversing ordinary gates, so
    # the 64 KiB buffer does not imply a 64 KiB total-process memory bound.
    outer = QuantumCircuit(2)
    for _ in range(4): outer.append(inner.to_gate(), [0])
    outer.cx(0, 1)
    gate = outer.to_gate()
    circuit = QuantumCircuit(2)
    for _ in range(200): circuit.append(gate, [0, 1])
    # Materialize lazy Qiskit definitions before observing serializer memory.
    expected = legacy(circuit)
    def peak(function):
        gc.collect(); tracemalloc.start()
        try:
            result = function(circuit)
            retained, maximum = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result == expected
        return maximum, retained
    (old_peak, _), (new_peak, new_retained) = peak(legacy), peak(streaming)
    assert new_peak < old_peak / 4, (old_peak, new_peak)
    # This difference is an observed transient excess, not a decomposition of
    # native/Qiskit memory. Whole payload/string/bytes allocation is absent.
    assert new_peak - new_retained < 512 * 1024, (new_peak, new_retained)


@pytest.mark.parametrize('change', ['phase', 'definition', 'nested_parameter', 'array', 'layout', 'recursive'])
def test_call_local_reuse_cannot_hide_mutations_between_hash_calls(change):
    definition = QuantumCircuit(1)
    definition.rz(.13, 0)
    matrix = np.diag([1., 1j]).astype(complex)
    definition.append(UnitaryGate(matrix), [0])
    gate = Gate('shared_definition', 1, [])
    gate.definition = definition
    circuit = QuantumCircuit(1)
    for _ in range(6): circuit.append(gate, [0], copy=False)
    before = streaming(circuit)
    assert before == legacy(circuit)
    if change == 'phase': definition.global_phase += .017
    elif change == 'definition': definition.x(0)
    elif change == 'nested_parameter':
        item = definition.data[0]
        changed = item.operation.copy()
        changed.params = [changed.params[0] + .021]
        definition.data[0] = item.replace(operation=changed)
    elif change == 'array': definition.data[1].operation.params[0][0, 0] = -1
    elif change == 'layout': definition._layout = object()
    else: definition.append(gate, [0], copy=False)
    if change in ('layout', 'recursive'):
        for fn in (streaming, legacy):
            with pytest.raises((ValueError, TypeError, CircuitError)): fn(circuit)
    else:
        after = streaming(circuit)
        assert after == legacy(circuit) and after != before


def test_cache_bytes_and_entry_limits_do_not_change_identity():
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import (
        _Writer, _HashSink, CACHE_BYTES, CACHE_ENTRIES, CACHE_ENTRY_BYTES,
    )
    circuit = QuantumCircuit(1)
    for index in range(300):
        definition = QuantumCircuit(1)
        definition.rz(.013 * index, 0)
        gate = Gate('distinct', 1, [])
        gate.definition = definition
        circuit.append(gate, [0], copy=False)
    sink = _HashSink(); writer = _Writer(sink)
    writer.circuit(circuit); sink.flush()
    assert sink.hasher.hexdigest() == legacy(circuit)
    assert writer.cache_bytes == sum(len(item[1]) for item in writer.cache.values()) <= CACHE_BYTES
    assert len(writer.cache) <= CACHE_ENTRIES
    assert all(len(item[1]) <= CACHE_ENTRY_BYTES for item in writer.cache.values())
    assert writer.capture_reserved == 0


def test_oversize_definition_streams_without_entering_byte_cache():
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import _Writer, _HashSink
    definition = QuantumCircuit(1)
    for index in range(256): definition.rz(.031 * index, 0)
    gate = Gate('large', 1, []); gate.definition = definition
    circuit = QuantumCircuit(1)
    for _ in range(2): circuit.append(gate, [0], copy=False)
    sink = _HashSink(); writer = _Writer(sink)
    writer.circuit(circuit); sink.flush()
    assert id(definition) not in writer.cache
    assert sink.hasher.hexdigest() == legacy(circuit)
    assert writer.capture_reserved == 0


def test_hash_buffer_stays_bounded_even_for_oversize_chunks():
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import _HashSink, HASH_BUFFER_BYTES
    chunks = [b'fixed' * 17000, b'small' * 12000, b'other' * 15000]
    sink = _HashSink()
    for chunk in chunks:
        sink.write(chunk)
        assert len(sink.buffer) <= HASH_BUFFER_BYTES
    sink.flush()
    assert sink.hasher.hexdigest() == hashlib.sha256(b''.join(chunks)).hexdigest()


def test_legacy_control_flow_parameter_rejection_is_preserved():
    circuit = QuantumCircuit(1, 1)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.rz(.23, 0)
    # The legacy numeric-parameter helper rejects the QuantumCircuit objects
    # stored in IfElseOp.params; v5 does not silently add control-flow support.
    for function in (legacy, streaming):
        with pytest.raises(TypeError, match='Unsupported circuit parameter type'):
            function(circuit)

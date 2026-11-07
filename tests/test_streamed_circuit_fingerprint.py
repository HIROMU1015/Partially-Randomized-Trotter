"""Exact JSON compatibility tests on artificial circuits only."""

import hashlib
import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.library import RZGate, RZZGate, UnitaryGate, XXPlusYYGate

from b2_cost_method_cases import artificial_round, event_pattern
from trotterlib.b2_streamed_native_cost import stream_native
from trotterlib.rte_compiled_cost import (
    _circuit_payload, canonical_qiskit_circuit_fingerprint,
)
from trotterlib.streamed_circuit_fingerprint import (
    iter_canonical_qiskit_circuit_bytes,
    streamed_canonical_qiskit_circuit_fingerprint as streamed,
)
from trottertracks.resource_applicability.ax2a_native_df import build_native_hadamard_wrapper


def same_bytes(circuit):
    expected = json.dumps(_circuit_payload(circuit), sort_keys=True,
                          separators=(",", ":")).encode()
    assert b"".join(iter_canonical_qiskit_circuit_bytes(circuit)) == expected
    assert streamed(circuit) == hashlib.sha256(expected).hexdigest()
    assert streamed(circuit) == canonical_qiskit_circuit_fingerprint(circuit)


@pytest.mark.parametrize('kind', ['empty', 'primitive', 'controlled', 'custom',
                                  'array', 'condition'])
def test_exact_legacy_json_bytes(kind):
    q = QuantumCircuit(3, 2)
    q.global_phase = -.123
    if kind == 'primitive':
        q.rz(-0.0, 0); q.sx(1); q.x(2); q.cx(2, 0); q.barrier(); q.measure(1, 1)
    elif kind == 'controlled':
        q.append(RZZGate(.33).control(1, ctrl_state=0), [2, 0, 1])
        q.append(XXPlusYYGate(-.4, .2).control(), [2, 1, 0])
    elif kind == 'custom':
        d = QuantumCircuit(2, name='日本語"\\'); d.global_phase = .25
        d.h(0); d.rz(-.0, 1); d.cx(1, 0)
        q.append(d.to_gate().control(), [2, 0, 1])
    elif kind == 'array':
        q.append(UnitaryGate(np.array([[1, 0], [0, 1j]], dtype=complex)), [0])
        q.append(Instruction('numeric', 1, 0, [True, None, 2, .3, 1j, '日本語',
            [np.float32(.2), -0.0], np.arange(6, dtype=np.int16).reshape(2, 3)]), [1])
    elif kind == 'condition':
        q.x(0).c_if(q.clbits[1], True)
    same_bytes(q)


@pytest.mark.parametrize('m', [0, 1, 3])
def test_artificial_b2_native_and_xy_wrappers(m):
    _, r = artificial_round(m, 2)
    evolution = stream_native(r, event_pattern(r, 'mixed'))
    same_bytes(evolution.circuit)
    for axis in ('cosine', 'sine'):
        same_bytes(build_native_hadamard_wrapper(evolution, axis=axis,
            max_instructions=len(evolution.circuit.data) + 4))


@pytest.mark.parametrize('defect', ['symbolic', 'layout', 'calibrations', 'nan',
                                    'object_array', 'recursive_definition', 'control_flow'])
def test_legacy_rejections_are_preserved(defect):
    q = QuantumCircuit(1, 1)
    if defect == 'symbolic':
        q.rz(Parameter('theta'), 0)
    elif defect == 'layout':
        q._layout = object()
    elif defect == 'calibrations':
        q._calibrations = {'x': {((), ()): object()}}
    elif defect == 'nan':
        q.append(Instruction('nonfinite', 1, 0, [float('nan')]), [0])
    elif defect == 'object_array':
        q.append(Instruction('object', 1, 0, [np.array([object()], dtype=object)]), [0])
    elif defect == 'control_flow':
        with q.if_test((q.clbits[0], True)):
            q.x(0)
    else:
        g = Gate('cycle', 1, [])
        g.definition = QuantumCircuit(1)
        g.definition.append(g, [0], copy=False)
        q.append(g, [0], copy=False)
    with pytest.raises((ValueError, TypeError)) as old:
        canonical_qiskit_circuit_fingerprint(q)
    with pytest.raises(type(old.value), match=str(old.value)):
        streamed(q)


@pytest.mark.parametrize('change', ['phase', 'parameter', 'order', 'qubits',
                                    'measurement', 'definition', 'ctrl_state'])
def test_semantic_differences_cannot_alias(change):
    q = QuantumCircuit(3, 1); q.rz(.1, 0); q.cx(0, 1)
    q.append(RZZGate(.2).control(), [2, 0, 1])
    changed = q.copy()
    if change == 'phase': changed.global_phase = .01
    elif change == 'parameter': changed.data[0] = changed.data[0].replace(operation=RZGate(.2))
    elif change == 'order': changed.data[0], changed.data[1] = changed.data[1], changed.data[0]
    elif change == 'qubits': changed.data[0] = changed.data[0].replace(qubits=(changed.qubits[1],))
    elif change == 'measurement': changed.measure(0, 0)
    elif change == 'definition':
        d = QuantumCircuit(1); d.h(0)
        changed.data[0] = changed.data[0].replace(operation=d.to_gate())
    else:
        changed.data[2] = changed.data[2].replace(operation=RZZGate(.2).control(ctrl_state=0))
    same_bytes(q); same_bytes(changed)
    assert streamed(q) != streamed(changed)


def test_separate_calls_observe_mutation_without_stale_cache():
    q = QuantumCircuit(1); q.rz(.1, 0)
    before = streamed(q)
    q.rz(.2, 0)
    assert streamed(q) != before
    same_bytes(q)


def test_independent_synthetic_custom_gates_have_fixed_canonical_name():
    from benchmark_streamed_circuit_fingerprint import synthetic_circuit
    a = synthetic_circuit('one_large_definition', 2)
    b = synthetic_circuit('one_large_definition', 2)
    assert a.data[0].operation.name == b.data[0].operation.name == 'synthetic_fingerprint_body'
    same_bytes(a); same_bytes(b)
    assert streamed(a) == streamed(b)

"""Bounded canonical LRU, one retained evolution, and checked joint branches.

Only disposable process namespaces change. Scientific bytes and compiler
options remain frozen. Unsafe ancilla structures use the original two probes.
"""
from contextlib import contextmanager
from dataclasses import replace
import importlib
import math
import threading
import numpy as np
from . import h8_fast_cost_port_v1 as frozen
from .h8_hash_stream_reuse_v1 import StreamingFingerprintReuse, _OnePassWriter, ALIASES
from .h6_hash_reuse_v1 import _Guard, _Capture, _Writer, ENTRY_BYTES
from .ax2b_stream_fingerprint_v5 import _HashSink, canonical_qiskit_circuit_fingerprint
from .ax2b_diagnostics_v4 import fingerprint_stage
from .h8_control_reuse_v1 import validate_control as two_branch_control, hadamard_states_from_branches
from .h4_common_panel_port_v1 import bound_function


class LRUWriter(_OnePassWriter):
    def operation(self, op, circuit):
        key = self.guard.operation(op, circuit)
        payload = self.session.entries.get(key)
        if payload is not None:
            self.session.entries.move_to_end(key)
            self.session.stats['operation_byte_hits'] += 1
            self.sink.write(payload)
            return
        self.session.stats['operation_byte_misses'] += 1
        # The old reservation rule stopped capturing permanently near 32MiB.
        # Evict before reserving; nested capture remains bounded by the same cap.
        while self.session.entries and self.session.cached_bytes+self.session.reserved_bytes+ENTRY_BYTES > self.session.byte_cap:
            _, old = self.session.entries.popitem(last=False)
            self.session.cached_bytes -= len(old)
            self.session.stats['evictions'] += 1
        parent = self.sink
        capture = None
        if self.session.cached_bytes+self.session.reserved_bytes+ENTRY_BYTES <= self.session.byte_cap:
            capture = _Capture(parent, ENTRY_BYTES)
            self.sink = capture
            self.session.reserved_bytes += ENTRY_BYTES
            self.session._peak()
        try:
            _Writer.operation(self, op, circuit)
        finally:
            self.sink = parent
            if capture is not None:
                self.session.reserved_bytes -= ENTRY_BYTES
        if capture is not None and not capture.overflow:
            self.session.store(key, bytes(capture.buffer))


class LRUFingerprint(StreamingFingerprintReuse):
    def write_bytes(self, circuit, write):
        class Sink:
            def write(self, b): write(b)
        LRUWriter(Sink(), self, _Guard(self.stats)).circuit(circuit)

    def fingerprint(self, circuit):
        with fingerprint_stage(circuit):
            self.stats['hash_calls'] += 1
            sink = _HashSink()
            LRUWriter(sink, self, _Guard(self.stats)).circuit(circuit)
            sink.flush()
            return sink.hasher.hexdigest()


@contextmanager
def hashing():
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise ValueError('FULL_MAP_SINGLE_THREAD_HASH')
    modules = [importlib.import_module('.'+x, __package__) for x in ALIASES]
    if any(x.canonical_qiskit_circuit_fingerprint is not canonical_qiskit_circuit_fingerprint for x in modules):
        raise ValueError('FULL_MAP_HASH_ALIAS')
    cache = LRUFingerprint()
    try:
        for x in modules: x.canonical_qiskit_circuit_fingerprint = cache.fingerprint
        yield cache
    finally:
        for x in modules: x.canonical_qiskit_circuit_fingerprint = canonical_qiskit_circuit_fingerprint
        cache.clear()


def ancilla_diagonal(circuit, ancilla, *, limit=4000000):
    """Exact pinned gate types only; arbitrary names never establish a proof.

    Disjoint gates commute with ancilla Z. A ControlledGate preserves every
    control bit for all control states; diagonal pinned gates preserve every
    bit. Unknown, conditional, or target-ancilla gates fall back to two probes.
    """
    from qiskit.circuit import ControlledGate
    from qiskit.circuit.library import (IGate, ZGate, SGate, SdgGate, TGate,
        TdgGate, RZGate, PhaseGate, CZGate, CPhaseGate, RZZGate)
    diagonal = (IGate, ZGate, SGate, SdgGate, TGate, TdgGate, RZGate, PhaseGate,
                CZGate, CPhaseGate, RZZGate)
    if circuit.parameters or circuit.num_clbits or len(circuit.data) > limit:
        return False
    for item in circuit.data:
        indices = [circuit.find_bit(q).index for q in item.qubits]
        if ancilla not in indices:
            continue
        op = item.operation
        if getattr(op, 'condition', None) is not None or item.clbits:
            return False
        cls = getattr(op, 'base_class', type(op))
        # Qiskit 1.3 singleton subclasses have __module__=None.
        module = type(op).__module__ or getattr(cls,'__module__','') or ''
        if cls in diagonal and isinstance(op, cls) and module.startswith('qiskit.') and hasattr(op,'__array__'):
            continue
        if isinstance(op, ControlledGate) and indices.index(ancilla) < op.num_ctrl_qubits:
            # Use pinned Qiskit classes, not a user subclass or overridden array.
            if (not module.startswith('qiskit.circuit.library.standard_gates.')
                or not hasattr(op,'__array__') or '_define' in vars(op)):
                return False
            continue
        return False
    return True


def validate_control(evolution, qstate, *, calls, simulate, expectation,
                     expected_signal=None, previous_high=None):
    if not ancilla_diagonal(evolution.circuit, evolution.num_system_qubits):
        v, b = two_branch_control(evolution, qstate, calls=calls, simulate=simulate,
            expectation=expectation, expected_signal=expected_signal, previous_high=previous_high)
        v['joint_branch_structure_proved'] = False
        v['algorithm'] = 'two-branch fallback'
        return v, b
    n = evolution.num_system_qubits
    psi = np.asarray(qstate, dtype=complex)
    if psi.shape != (1 << n,) or not np.isfinite(psi).all() or abs(np.linalg.norm(psi)-1) > 1e-12:
        raise ValueError('JOINT_BRANCH_STATE')
    calls.take('control_probe', 2)
    evolved = simulate(evolution.circuit, np.concatenate((psi, psi))/math.sqrt(2.))
    if evolved.shape != (2*len(psi),) or not np.isfinite(evolved).all():
        raise ValueError('JOINT_BRANCH_RESULT')
    a = np.concatenate((math.sqrt(2.)*evolved[:len(psi)], np.zeros_like(psi)))
    b = np.concatenate((np.zeros_like(psi), math.sqrt(2.)*evolved[len(psi):]))
    low = np.concatenate((psi, np.zeros_like(psi)))
    z = complex(np.vdot(psi, b[len(psi):]))
    errors = [float(np.linalg.norm(a-low))]
    if expected_signal is not None:
        errors.append(abs(z-complex(expected_signal['real'], expected_signal['imag'])))
    if previous_high is not None:
        errors.append(float(np.linalg.norm(b-previous_high)))
    for axis, v in hadamard_states_from_branches(a, b, n).items():
        calls.take('control_probe')
        errors.append(abs(expectation(v, n)-(z.real if axis == 'cosine' else z.imag)))
    if not all(math.isfinite(x) for x in errors):
        raise ValueError('JOINT_BRANCH_NONFINITE')
    return dict(max_error=max(errors), signal=dict(real=z.real, imag=z.imag),
        logical_control_probes=4, full_evolution_simulations=1,
        expensive_full_wrapper_simulations=0, reused_axis_states=2,
        joint_branch_structure_proved=True, algorithm='ancilla-Z-preserving joint branch',
        branch_leakage='excluded by pinned gate structure; not a measured zero',
        normalization_policy='no renormalization', certified=False,
        random_signal_is_ensemble=False, validation_method_changed=True), b


def cost_port(root, manifest, task, writer, progress, trace):
    from .h48_preflight_contract_v1 import task_caps
    from copy import deepcopy
    local = deepcopy(manifest)
    local['plan']['caps_proposed'] = task_caps(task, manifest['plan']['caps_proposed'])
    retained = {}
    def factory(name, original):
        def build(*args, **kwargs):
            policy = kwargs['control_policy']
            key = name, policy
            if key not in retained:
                value = original(*args, **kwargs)
                # The original owner is consumed by cosine. A single retained
                # copy supplies sine without repeated native gate construction.
                retained[key] = replace(value, circuit=value.circuit.copy())
                return value
            value = retained.pop(key)
            return value
        return build
    worker = bound_function(frozen.cost_task, dict(validate_control=validate_control,
        build_deterministic_native=factory('deterministic', frozen.build_deterministic_native),
        partial_native_from_step_requests=factory('partial', frozen.partial_native_from_step_requests)))
    with hashing() as cache:
        try:
            worker(root, local, task, writer, progress, trace)
            writer.write('hash_reuse_receipt.json', dict(cache.snapshot(),
                native_evolution_builds_per_control=1, retained_evolutions_max=1,
                fallback_control_retained=True))
        finally:
            retained.clear()

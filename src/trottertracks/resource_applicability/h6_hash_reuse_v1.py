"""Bounded, content-checked reuse of unchanged canonical v2 bytes/digests.

Each call first walks the CURRENT graph. The guard hashes ordered structural
records and recursively memoizes shared nodes only within that call. Persistent
entries are keyed by content, never by a mutable object's ID. The guard is a
cache key, not the scientific circuit fingerprint. Final SHA256 still consumes
the exact v5 canonical bytes. Callers keep the graph immutable during a call.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import importlib
import json
import threading

from .ax2b_stream_fingerprint_v5 import _Writer, _HashSink
from .ax2b_diagnostics_v4 import fingerprint_stage
from trotterlib.rte_compiled_cost import _numeric_parameter_payload, _condition_payload

CACHE_BYTES = 32 * 2**20
ENTRY_BYTES = 2**20
CACHE_ENTRIES = 8192
GUARD_NODES = 8192
DIGEST_ENTRIES = 1024
_ENCODER = json.JSONEncoder(sort_keys=True, separators=(',', ':'))
_ALIASES = ('ax2b_native_df_v5', 'h6_matched_memory_v3', 'h6_matched_port_v1')


def _numeric(value):
    # The legacy array digest does not itself check finiteness. Fail closed
    # here, including arrays hidden inside parameter lists; valid bytes agree.
    import numpy as np
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object-array circuit parameters are not cacheable.')
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError('Circuit parameters must be finite.')
    elif isinstance(value, (list, tuple)):
        return [_numeric(x) for x in value]
    return _numeric_parameter_payload(value)


def _record(value):
    return _ENCODER.encode(value).encode()


class _Guard:
    """Per-call strong references prevent lazy-node ID recycling; bounded memo."""
    def __init__(self, stats):
        self.memo = {}; self.active = set(); self.stats = stats

    def _node(self, key, references, build):
        if key in self.active:
            raise ValueError('Recursive custom-gate definitions are not cacheable.')
        cached = self.memo.get(key)
        if cached is not None:
            assert all(a is b for a,b in zip(cached[0], references))
            self.stats['guard_memo_hits'] += 1
            return cached[1]
        self.active.add(key)
        try:
            result = build()
        finally:
            self.active.remove(key)
        if len(self.memo) < GUARD_NODES:
            self.memo[key] = (references, result)
        self.stats['guard_nodes_walked'] += 1
        return result

    def circuit(self, c):
        def build():
            if c.parameters:
                raise ValueError('Symbolic circuits are not cacheable; bind all parameters first.')
            if getattr(c, 'layout', None) is not None:
                raise ValueError('Circuits carrying transpiler layout metadata are not cacheable.')
            if getattr(c, 'calibrations', {}):
                raise ValueError('Circuits with pulse calibrations are not cacheable.')
            h = hashlib.sha256(b'h6_current_graph_guard_v1/circuit\0')
            h.update(_record([int(c.num_qubits), int(c.num_clbits), _numeric(c.global_phase)]))
            for item in c.data:
                h.update(b'\0instruction\0')
                h.update(_record([[c.find_bit(q).index for q in item.qubits],
                    [c.find_bit(b).index for b in item.clbits], self.operation(item.operation, c)]))
            return h.hexdigest()
        return self._node(('circuit', id(c)), (c,), build)

    def operation(self, op, c):
        def build():
            base = getattr(op, 'base_gate', None)
            blocks = getattr(op, 'blocks', ())
            definition = getattr(op, 'definition', None)
            fields = dict(type=f'{type(op).__module__}.{type(op).__qualname__}', name=str(op.name),
                num_qubits=int(op.num_qubits), num_clbits=int(op.num_clbits),
                params=[_numeric(x) for x in op.params],
                condition=_condition_payload(getattr(op, 'condition', None), c))
            for name in ('num_ctrl_qubits', 'ctrl_state'):
                if hasattr(op, name):fields[name] = int(getattr(op, name))
            if base is not None:fields['base_gate'] = self.operation(base, c)
            if blocks:fields['control_flow_blocks'] = [self.circuit(b) for b in blocks]
            if definition is not None:fields['definition'] = self.circuit(definition)
            return hashlib.sha256(b'h6_current_graph_guard_v1/operation\0'+_record(fields)).hexdigest()
        # Conditions refer to the containing circuit's bit indices.
        return self._node(('operation', id(op), id(c)), (op, c), build)


class _Capture:
    def __init__(self, parent, limit):
        self.parent = parent; self.limit = limit; self.buffer = bytearray(); self.overflow = False

    def write(self, chunk):
        self.parent.write(chunk)
        if not self.overflow:
            if len(self.buffer)+len(chunk) > self.limit:
                self.buffer.clear(); self.overflow = True
            else:self.buffer.extend(chunk)


class _ReuseWriter(_Writer):
    def __init__(self, sink, session, guard):
        super().__init__(sink); self.session = session; self.guard = guard

    def operation(self, op, circuit):
        key = self.guard.operation(op, circuit)
        payload = self.session.entries.get(key)
        if payload is not None:
            self.session.entries.move_to_end(key)
            self.session.stats['operation_byte_hits'] += 1
            self.sink.write(payload)
            return
        self.session.stats['operation_byte_misses'] += 1
        parent = self.sink; capture = None
        if self.session.cached_bytes+self.session.reserved_bytes+ENTRY_BYTES <= self.session.byte_cap:
            capture = _Capture(parent, ENTRY_BYTES); self.sink = capture
            self.session.reserved_bytes += ENTRY_BYTES
            self.session._peak()
        try:
            super().operation(op, circuit)
        finally:
            self.sink = parent
            if capture is not None:self.session.reserved_bytes -= ENTRY_BYTES
        if capture is not None and not capture.overflow:
            self.session.store(key, bytes(capture.buffer))


class FingerprintReuse:
    """One disposable worker/group; payloads and hash keys never cross runs."""
    def __init__(self, *, byte_cap=CACHE_BYTES):
        if type(byte_cap) is not int or not 0 <= byte_cap <= CACHE_BYTES:
            raise ValueError('HASH_CACHE_CAP')
        self.byte_cap = byte_cap; self.entries = OrderedDict(); self.digests = OrderedDict()
        self.cached_bytes = 0; self.reserved_bytes = 0
        self.stats = dict(hash_calls=0, final_digest_hits=0, operation_byte_hits=0,
            operation_byte_misses=0, guard_nodes_walked=0, guard_memo_hits=0,
            payload_peak_reserved_bytes=0, evictions=0)

    def _peak(self):
        self.stats['payload_peak_reserved_bytes'] = max(self.stats['payload_peak_reserved_bytes'],
            self.cached_bytes+self.reserved_bytes)

    def store(self, key, payload):
        if len(payload) > ENTRY_BYTES or len(payload)+self.reserved_bytes > self.byte_cap:return
        if key in self.entries:return
        while self.entries and (self.cached_bytes+self.reserved_bytes+len(payload) > self.byte_cap
                or len(self.entries) >= CACHE_ENTRIES):
            _, removed = self.entries.popitem(last=False)
            self.cached_bytes -= len(removed); self.stats['evictions'] += 1
        self.entries[key] = payload; self.cached_bytes += len(payload); self._peak()

    def fingerprint(self, circuit):
        with fingerprint_stage(circuit):
            self.stats['hash_calls'] += 1
            guard = _Guard(self.stats)
            key = guard.circuit(circuit)
            if key in self.digests:
                self.digests.move_to_end(key); self.stats['final_digest_hits'] += 1
                return self.digests[key]
            sink = _HashSink(); writer = _ReuseWriter(sink, self, guard)
            writer.circuit(circuit); sink.flush(); result = sink.hasher.hexdigest()
            if len(self.digests) >= DIGEST_ENTRIES:self.digests.popitem(last=False)
            self.digests[key] = result
            return result

    def snapshot(self):
        return dict(self.stats, payload_bytes=self.cached_bytes, payload_byte_cap=self.byte_cap,
            reserved_bytes=self.reserved_bytes, entries=len(self.entries), digest_entries=len(self.digests),
            persistent_object_identity_keys=False, canonical_policy='qiskit_recursive_numeric_circuit_v2')

    def clear(self):
        self.entries.clear(); self.digests.clear(); self.cached_bytes = 0


@contextmanager
def use_fingerprint_reuse():
    """Patch only frozen aliases in this single-threaded disposable process.

    Disk source and other processes are untouched; always restore on failure.
    Reject nested/concurrent use or a previously modified fingerprint alias.
    """
    from .ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as original
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise ValueError('HASH_REUSE_SINGLE_THREAD_REQUIRED')
    modules = [importlib.import_module('.'+name, __package__) for name in _ALIASES]
    if any(m.canonical_qiskit_circuit_fingerprint is not original for m in modules):
        raise ValueError('HASH_REUSE_ALIAS_ALREADY_CHANGED')
    session = FingerprintReuse()
    try:
        for m in modules:m.canonical_qiskit_circuit_fingerprint = session.fingerprint
        yield session
    finally:
        for m in modules:m.canonical_qiskit_circuit_fingerprint = original
        session.clear()

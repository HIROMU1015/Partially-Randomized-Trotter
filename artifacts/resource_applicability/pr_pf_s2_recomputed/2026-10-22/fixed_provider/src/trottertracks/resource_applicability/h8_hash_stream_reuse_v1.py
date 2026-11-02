"""One canonical pass with current-operation guards and bounded byte reuse.

No top-level prewalk or final-digest cache bypasses the current graph. Every
instruction/order/phase/control is inspected in the canonical streaming pass.
The frozen v2 byte schema and old content-guarded operation cache are reused.
Circuits must remain immutable during a call, as in the original contract.
"""
from contextlib import contextmanager
import importlib
import threading

from .h6_hash_reuse_v1 import FingerprintReuse, _Guard, _ReuseWriter, _numeric, _record
from .ax2b_stream_fingerprint_v5 import _HashSink, canonical_qiskit_circuit_fingerprint
from .ax2b_diagnostics_v4 import fingerprint_stage

WIRE_ENTRIES = 1024
WIRE_BYTES = 262144
ALIASES = ('ax2b_native_df_v5', 'h6_matched_memory_v3', 'h6_matched_port_v1')


class _OnePassWriter(_ReuseWriter):
    def __init__(self, sink, session, guard):
        super().__init__(sink, session, guard)
        self.wires = {}; self.wire_bytes = 0

    def positions(self, indices):
        key = tuple(indices)
        value = self.wires.get(key)
        if value is not None:
            self.session.stats['wire_payload_hits'] += 1
            return value
        value = _record(list(key))
        if len(self.wires) < WIRE_ENTRIES and self.wire_bytes+len(value) <= WIRE_BYTES:
            self.wires[key] = value; self.wire_bytes += len(value)
        return value

    def _circuit(self, circuit):
        if circuit.parameters:
            raise ValueError('Symbolic circuits are not cacheable; bind all parameters first.')
        if getattr(circuit, 'layout', None) is not None:
            raise ValueError('Circuits carrying transpiler layout metadata are not cacheable.')
        if getattr(circuit, 'calibrations', {}):
            raise ValueError('Circuits with pulse calibrations are not cacheable.')
        self.sink.write(b'{"fingerprint_policy":"qiskit_recursive_numeric_circuit_v2","global_phase":')
        self.scalar(_numeric(circuit.global_phase))
        self.sink.write(b',"instructions":[')
        for index, item in enumerate(circuit.data):
            if index:
                self.sink.write(b',')
            clbits = self.positions(circuit.find_bit(c).index for c in item.clbits)
            qubits = self.positions(circuit.find_bit(q).index for q in item.qubits)
            self.sink.write(b'{"clbits":'+clbits+b',"operation":')
            self.operation(item.operation, circuit)
            self.sink.write(b',"qubits":'+qubits+b'}')
            self.session.stats['canonical_instruction_visits'] += 1
        self.sink.write(b'],"num_clbits":'+_record(int(circuit.num_clbits)))
        self.sink.write(b',"num_qubits":'+_record(int(circuit.num_qubits))+b'}')


class StreamingFingerprintReuse(FingerprintReuse):
    def __init__(self, *, byte_cap=32*2**20):
        super().__init__(byte_cap=byte_cap)
        self.stats.update(top_level_prepasses=0, canonical_instruction_visits=0, wire_payload_hits=0)

    def write_bytes(self, circuit, write):
        class Sink:
            def write(self, payload):
                write(payload)
        writer = _OnePassWriter(Sink(), self, _Guard(self.stats))
        writer.circuit(circuit)

    def fingerprint(self, circuit):
        with fingerprint_stage(circuit):
            self.stats['hash_calls'] += 1
            sink = _HashSink()
            writer = _OnePassWriter(sink, self, _Guard(self.stats))
            writer.circuit(circuit)
            sink.flush()
            return sink.hasher.hexdigest()

    def snapshot(self):
        return dict(super().snapshot(), wire_byte_cap_per_call=WIRE_BYTES,
                    final_digest_reuse=False, current_graph_checked=True,
                    algorithm='canonical single-pass with operation content guards')


@contextmanager
def use_streaming_hash_reuse():
    """Only this single-threaded disposable process; restore aliases on all exits."""
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise ValueError('HASH_REUSE_SINGLE_THREAD_REQUIRED')
    modules = [importlib.import_module('.'+n, __package__) for n in ALIASES]
    if any(m.canonical_qiskit_circuit_fingerprint is not canonical_qiskit_circuit_fingerprint for m in modules):
        raise ValueError('HASH_REUSE_ALIAS_ALREADY_CHANGED')
    session = StreamingFingerprintReuse()
    try:
        for m in modules:
            m.canonical_qiskit_circuit_fingerprint = session.fingerprint
        yield session
    finally:
        for m in modules:
            m.canonical_qiskit_circuit_fingerprint = canonical_qiskit_circuit_fingerprint
        session.clear()

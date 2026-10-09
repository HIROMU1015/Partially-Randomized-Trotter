"""Stream the unchanged canonical v2 bytes with a call-local bounded cache.

The writer avoids a generator frame per JSON token. Only complete definition
bytes are reused, never definition hashes. The cache exists for one hash call;
fresh calls inspect current phase, parameters, controls and definitions again.
Circuits must remain immutable during each call, as in the legacy contract.
No whole top-level circuit payload/string/bytes is constructed.
"""
import hashlib
import json

from trotterlib.rte_compiled_cost import _numeric_parameter_payload, _condition_payload
from .ax2b_diagnostics_v4 import fingerprint_stage

HASH_BUFFER_BYTES = 65536
CACHE_BYTES = 262144
CACHE_ENTRY_BYTES = 65536
CACHE_ENTRIES = 256
_ENCODER = json.JSONEncoder(sort_keys=True, separators=(',', ':'))


class _HashSink:
    def __init__(self):
        self.hasher = hashlib.sha256()
        self.buffer = bytearray()

    def write(self, chunk):
        if len(chunk) >= HASH_BUFFER_BYTES:
            self.flush()
            self.hasher.update(chunk)
        else:
            if len(self.buffer) + len(chunk) > HASH_BUFFER_BYTES:
                self.flush()
            self.buffer.extend(chunk)

    def flush(self):
        if self.buffer:
            self.hasher.update(self.buffer)
            self.buffer.clear()


class _Capture:
    def __init__(self, parent):
        self.parent = parent
        self.buffer = bytearray()
        self.overflow = False

    def write(self, chunk):
        self.parent.write(chunk)
        if not self.overflow:
            if len(self.buffer) + len(chunk) > CACHE_ENTRY_BYTES:
                self.buffer.clear()
                self.overflow = True
            else:
                self.buffer.extend(chunk)


class _Writer:
    def __init__(self, sink):
        self.sink = sink
        self.active = set()
        self.cache = {}
        self.cache_bytes = 0
        self.capture_reserved = 0

    def scalar(self, value):
        self.sink.write(_ENCODER.encode(value).encode())

    def circuit(self, circuit, *, definition=False):
        identity = id(circuit)
        if definition:
            if identity in self.active:
                raise ValueError('Recursive custom-gate definitions are not cacheable.')
            cached = self.cache.get(identity)
            if cached is not None:
                # Keep a strong reference in each entry, so object IDs cannot
                # be recycled while lazy Qiskit definitions are traversed.
                assert cached[0] is circuit
                self.sink.write(cached[1])
                return
            self.active.add(identity)
        parent = self.sink
        capture = None
        if (definition and len(self.cache) < CACHE_ENTRIES
                and self.cache_bytes + self.capture_reserved + CACHE_ENTRY_BYTES <= CACHE_BYTES):
            capture = _Capture(parent)
            self.capture_reserved += CACHE_ENTRY_BYTES
            self.sink = capture
        try:
            self._circuit(circuit)
        finally:
            self.sink = parent
            if capture is not None:
                self.capture_reserved -= CACHE_ENTRY_BYTES
            if definition:
                self.active.remove(identity)
        if capture is not None and not capture.overflow and len(self.cache) < CACHE_ENTRIES:
            size = len(capture.buffer)
            # Nested entries may have filled the remaining budget. Drop this
            # capture rather than evicting or growing the byte allowance.
            if self.cache_bytes + self.capture_reserved + size <= CACHE_BYTES:
                payload = bytes(capture.buffer)
                self.cache[identity] = (circuit, payload)
                self.cache_bytes += size

    def _circuit(self, circuit):
        if circuit.parameters:
            raise ValueError('Symbolic circuits are not cacheable; bind all parameters first.')
        if getattr(circuit, 'layout', None) is not None:
            raise ValueError('Circuits carrying transpiler layout metadata are not cacheable.')
        if getattr(circuit, 'calibrations', {}):
            raise ValueError('Circuits with pulse calibrations are not cacheable.')
        self.sink.write(b'{"fingerprint_policy":"qiskit_recursive_numeric_circuit_v2","global_phase":')
        self.scalar(_numeric_parameter_payload(circuit.global_phase))
        self.sink.write(b',"instructions":[')
        for index, item in enumerate(circuit.data):
            if index:
                self.sink.write(b',')
            self.sink.write(b'{"clbits":')
            self.scalar([circuit.find_bit(c).index for c in item.clbits])
            self.sink.write(b',"operation":')
            self.operation(item.operation, circuit)
            self.sink.write(b',"qubits":')
            self.scalar([circuit.find_bit(q).index for q in item.qubits])
            self.sink.write(b'}')
        self.sink.write(b'],"num_clbits":')
        self.scalar(int(circuit.num_clbits))
        self.sink.write(b',"num_qubits":')
        self.scalar(int(circuit.num_qubits))
        self.sink.write(b'}')

    def operation(self, operation, circuit):
        # Emit precisely the sort_keys=True order, including every optional
        # legacy field. Numeric leaf and classical condition rules are shared.
        self.sink.write(b'{')
        base_gate = getattr(operation, 'base_gate', None)
        if base_gate is not None:
            self.sink.write(b'"base_gate":')
            self.operation(base_gate, circuit)
            self.sink.write(b',')
        self.sink.write(b'"condition":')
        self.scalar(_condition_payload(getattr(operation, 'condition', None), circuit))
        blocks = getattr(operation, 'blocks', ())
        if blocks:
            self.sink.write(b',"control_flow_blocks":[')
            for index, block in enumerate(blocks):
                if index:
                    self.sink.write(b',')
                self.circuit(block)
            self.sink.write(b']')
        if hasattr(operation, 'ctrl_state'):
            self.sink.write(b',"ctrl_state":')
            self.scalar(int(operation.ctrl_state))
        definition = getattr(operation, 'definition', None)
        if definition is not None:
            self.sink.write(b',"definition":')
            self.circuit(definition, definition=True)
        self.sink.write(b',"name":')
        self.scalar(str(operation.name))
        self.sink.write(b',"num_clbits":')
        self.scalar(int(operation.num_clbits))
        if hasattr(operation, 'num_ctrl_qubits'):
            self.sink.write(b',"num_ctrl_qubits":')
            self.scalar(int(operation.num_ctrl_qubits))
        self.sink.write(b',"num_qubits":')
        self.scalar(int(operation.num_qubits))
        self.sink.write(b',"params":[')
        for index, parameter in enumerate(operation.params):
            if index:
                self.sink.write(b',')
            self.scalar(_numeric_parameter_payload(parameter))
        self.sink.write(b'],"type":')
        self.scalar(f'{type(operation).__module__}.{type(operation).__qualname__}')
        self.sink.write(b'}')


def write_canonical_bytes(circuit, write):
    """Small-fixture inspection interface; production writes into a hash sink."""
    class Sink:
        def write(self, chunk):
            write(chunk)
    _Writer(Sink()).circuit(circuit)


def canonical_qiskit_circuit_fingerprint(circuit):
    with fingerprint_stage(circuit):
        sink = _HashSink()
        writer = _Writer(sink)
        try:
            writer.circuit(circuit)
            sink.flush()
            return sink.hasher.hexdigest()
        finally:
            # No bytes/circuits persist across hash calls, wrappers or groups.
            writer.cache.clear()

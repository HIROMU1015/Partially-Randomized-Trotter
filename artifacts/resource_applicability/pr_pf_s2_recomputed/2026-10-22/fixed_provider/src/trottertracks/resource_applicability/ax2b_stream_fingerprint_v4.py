"""Stream the SAME qiskit_recursive_numeric_circuit_v2 canonical JSON bytes.

No recursive payload, whole-circuit JSON string, or whole-circuit bytes are
materialized. Immutable fully numeric circuits are required, as in the legacy
fingerprinter. Extra memory scales with nesting, the largest leaf parameter
payload, and a 64 KiB hash buffer; gate-definition construction is separate.
"""
import hashlib
import json

from trotterlib.rte_compiled_cost import _numeric_parameter_payload, _condition_payload
from .ax2b_diagnostics_v4 import fingerprint_stage


def _scalar(value):
    yield json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _object(fields):
    yield b'{'
    for index, key in enumerate(sorted(fields)):
        if index:
            yield b','
        yield from _scalar(key)
        yield b':'
        yield from fields[key]()
    yield b'}'


def _array(items):
    yield b'['
    for index, chunks in enumerate(items):
        if index:
            yield b','
        yield from chunks
    yield b']'


def _definition(circuit, active):
    identifier = id(circuit)
    if identifier in active:
        raise ValueError('Recursive custom-gate definitions are not cacheable.')
    active.add(identifier)
    try:
        yield from _circuit(circuit, active)
    finally:
        active.remove(identifier)


def _operation(operation, circuit, active):
    fields = {
        'type': lambda: _scalar(f'{type(operation).__module__}.{type(operation).__qualname__}'),
        'name': lambda: _scalar(str(operation.name)),
        'num_qubits': lambda: _scalar(int(operation.num_qubits)),
        'num_clbits': lambda: _scalar(int(operation.num_clbits)),
        'params': lambda: _array(_scalar(_numeric_parameter_payload(p)) for p in operation.params),
        'condition': lambda: _scalar(_condition_payload(getattr(operation, 'condition', None), circuit)),
    }
    for name in ('num_ctrl_qubits', 'ctrl_state'):
        if hasattr(operation, name):
            fields[name] = lambda name=name: _scalar(int(getattr(operation, name)))
    base_gate = getattr(operation, 'base_gate', None)
    if base_gate is not None:
        fields['base_gate'] = lambda: _operation(base_gate, circuit, active)
    blocks = getattr(operation, 'blocks', ())
    if blocks:
        fields['control_flow_blocks'] = lambda: _array(_circuit(b, active) for b in blocks)
    definition = getattr(operation, 'definition', None)
    if definition is not None:
        fields['definition'] = lambda: _definition(definition, active)
    yield from _object(fields)


def _instruction(item, circuit, active):
    yield from _object({
        'operation': lambda: _operation(item.operation, circuit, active),
        'qubits': lambda: _scalar([circuit.find_bit(q).index for q in item.qubits]),
        'clbits': lambda: _scalar([circuit.find_bit(c).index for c in item.clbits]),
    })


def _circuit(circuit, active):
    if circuit.parameters:
        raise ValueError('Symbolic circuits are not cacheable; bind all parameters first.')
    if getattr(circuit, 'layout', None) is not None:
        raise ValueError('Circuits carrying transpiler layout metadata are not cacheable.')
    if getattr(circuit, 'calibrations', {}):
        raise ValueError('Circuits with pulse calibrations are not cacheable.')
    yield from _object({
        'num_qubits': lambda: _scalar(int(circuit.num_qubits)),
        'num_clbits': lambda: _scalar(int(circuit.num_clbits)),
        'global_phase': lambda: _scalar(_numeric_parameter_payload(circuit.global_phase)),
        'instructions': lambda: _array(_instruction(item, circuit, active) for item in circuit.data),
        'fingerprint_policy': lambda: _scalar('qiskit_recursive_numeric_circuit_v2'),
    })


def canonical_chunks(circuit):
    """Test/review interface: consume incrementally; do not join large circuits."""
    yield from _circuit(circuit, set())


def canonical_qiskit_circuit_fingerprint(circuit):
    with fingerprint_stage(circuit):
        hasher = hashlib.sha256()
        buffer = bytearray()
        for chunk in canonical_chunks(circuit):
            if len(chunk) >= 65536:
                if buffer:
                    hasher.update(buffer)
                    buffer.clear()
                hasher.update(chunk)
            else:
                if len(buffer) + len(chunk) > 65536:
                    hasher.update(buffer)
                    buffer.clear()
                buffer.extend(chunk)
        if buffer:
            hasher.update(buffer)
        return hasher.hexdigest()

"""Opt-in, byte-compatible streaming of the existing numeric circuit policy.

The default fingerprint and scientific builders are unchanged. Only one expanded
top-level instruction is retained at a time. A large individual gate definition,
Qiskit's lazy definition caches, and the input circuit still consume memory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

from qiskit import QuantumCircuit

from .rte_compiled_cost import _numeric_parameter_payload, _operation_payload


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def iter_canonical_qiskit_circuit_bytes(circuit: QuantumCircuit) -> Iterator[bytes]:
    """Yield exactly the legacy v2 JSON bytes without its whole-circuit payload.

    Reuse the existing recursive operation and numeric serializers. Preserve their
    acceptance rules; this does not extend support to symbolic/layout/calibrated
    circuits. Callers must consume the iterator completely and must not mutate the
    circuit during hashing. No persistent or identity-based operation cache is used.
    """
    if circuit.parameters:
        raise ValueError("Symbolic circuits are not cacheable; bind all parameters first.")
    if getattr(circuit, "layout", None) is not None:
        raise ValueError("Circuits carrying transpiler layout metadata are not cacheable.")
    if getattr(circuit, "calibrations", {}):
        raise ValueError("Circuits with pulse calibrations are not cacheable.")

    # Field order is the lexicographic order of json.dumps(sort_keys=True).
    yield b'{"fingerprint_policy":"qiskit_recursive_numeric_circuit_v2","global_phase":'
    yield _encoded(_numeric_parameter_payload(circuit.global_phase))
    yield b',"instructions":['
    active_definitions: set[int] = set()
    for index, item in enumerate(circuit.data):
        if index:
            yield b","
        yield _encoded({
            "operation": _operation_payload(
                item.operation, circuit, active_definitions=active_definitions,
            ),
            "qubits": [circuit.find_bit(bit).index for bit in item.qubits],
            "clbits": [circuit.find_bit(bit).index for bit in item.clbits],
        })
    yield b'],"num_clbits":' + _encoded(int(circuit.num_clbits))
    yield b',"num_qubits":' + _encoded(int(circuit.num_qubits)) + b"}"


def streamed_canonical_qiskit_circuit_fingerprint(circuit: QuantumCircuit) -> str:
    """Return the legacy SHA256, retaining at most one expanded instruction."""
    digest = hashlib.sha256()
    for chunk in iter_canonical_qiskit_circuit_bytes(circuit):
        digest.update(chunk)
    return digest.hexdigest()

"""Track B observation adapter; the frozen input identity policy is unchanged.

Shared source/API is untouched. This small no-backend adapter uses the shared
payload, metric and validation helpers, and mirrors the frozen cost assembly.
It evaluates the recursive payload ONCE, preserving the shared None fallback,
then saves its outcome before the transpiler or the caller's identity guard.
Do not substitute a new expected digest or use these diagnostics as a norm
certificate. Production calls still require the worker's separate launch gate.
"""
from __future__ import annotations

import hashlib
import json
import traceback

import numpy as np


MAX_ARRAY_BYTES = 128 * 128 * 16  # largest registered controlled r6 input


def array_record(value):
    """Lossless C-order bytes, plus original layout; never alter the operand."""
    value = np.asarray(value)
    if value.dtype.hasobject or value.nbytes > MAX_ARRAY_BYTES:
        raise ValueError('Diagnostic array type/size outside fixed local domain')
    raw = value.tobytes(order='C')
    return {'dtype': str(value.dtype), 'dtype_str': value.dtype.str,
            'shape': list(value.shape), 'strides': list(value.strides),
            'C_contiguous': bool(value.flags.c_contiguous),
            'F_contiguous': bool(value.flags.f_contiguous),
            'owns_data': bool(value.flags.owndata), 'writeable': bool(value.flags.writeable),
            'order': 'C', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'raw_bytes_hex': raw.hex()}


def payload_digest(payload):
    # EXACT shared canonical encoding, including default ensure_ascii=True.
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def exception_record(exc):
    frames = traceback.extract_tb(exc.__traceback__)
    return {'type': type(exc).__module__ + '.' + type(exc).__qualname__,
            'message': str(exc)[:4096], 'message_truncated': len(str(exc)) > 4096,
            'traceback': [{'file': f.filename, 'line': f.lineno, 'function': f.name}
                          for f in frames[-24:]], 'traceback_truncated': len(frames) > 24}


def _definition_index(payload):
    """Index ONLY the already captured payload; no second definition evaluation."""
    out = []
    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                child = path + '/' + key
                if key == 'definition':
                    out.append({'payload_path': child, 'sha256': payload_digest(value),
                                'num_qubits': value['num_qubits'],
                                'instruction_count': len(value['instructions'])})
                walk(value, child)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, path + '/' + str(i))
    walk(payload, '')
    return out


def diagnostic_transpile_and_measure_cost(circuit, compiler, *, expected,
                                         arrays, context, save):
    """One fingerprint attempt and one compile, same frozen no-backend settings.

    save(record) must synchronously persist each state (lossless chunk storage).
    A save failure aborts; there is no permissive fallback or diagnostic retry.
    TypeError/ValueError in fingerprint creation yield actual=None exactly as
    in the common helper. Other exceptions are recorded and re-raised.
    """
    from trotterlib import rte_compiled_cost as shared
    shared._validate_transpile_context(compiler, None)
    record = {'kind': 'FROZEN_WHOLE_INPUT_IDENTITY_DIAGNOSTIC',
              'context': context, 'expected_fingerprint': expected,
              'actual_fingerprint': None, 'fingerprint_attempts': 0,
              'fingerprint_exception': None, 'transpile_attempts': 0,
              'policy': 'qiskit_recursive_numeric_circuit_v2',
              'compiler_settings_hash': shared.compiler_settings_hash(compiler),
              'input_arrays': {name: array_record(value) for name, value in arrays.items()},
              'phase': 'BEFORE_FINGERPRINT', 'scientific_interpretation': None}
    save(record)
    record['fingerprint_attempts'] = 1
    try:
        payload = shared._circuit_payload(circuit)
        actual = payload_digest(payload)
    except BaseException as exc:
        record['fingerprint_exception'] = exception_record(exc)
        record['phase'] = 'FINGERPRINT_EXCEPTION'
        record['recursive_payload'] = None
        record['recursive_definition_index'] = None
        record['recursive_definition_status'] = 'INCOMPLETE_NOT_REEVALUATED'
        save(record)
        if not isinstance(exc, (TypeError, ValueError)):
            raise
        actual = None
    else:
        record['actual_fingerprint'] = actual
        record['recursive_payload'] = payload
        record['recursive_definition_index'] = _definition_index(payload)
        record['recursive_definition_status'] = 'COMPLETE_SAME_SINGLE_PAYLOAD'
        record['phase'] = 'FINGERPRINT_COMPLETE'
        save(record)
    backend_fingerprint = shared._canonical_backend_fingerprint_or_none(None)
    kwargs = {'basis_gates': list(compiler.basis_gates),
              'optimization_level': compiler.optimization_level,
              'seed_transpiler': compiler.transpiler_seed}
    if compiler.coupling_map is not None:
        kwargs['coupling_map'] = list(compiler.coupling_map)
    if compiler.layout_method is not None:
        kwargs['layout_method'] = compiler.layout_method
    if compiler.routing_method is not None:
        kwargs['routing_method'] = compiler.routing_method
    record['phase'] = 'BEFORE_TRANSPILE'
    record['transpile_attempts'] = 1
    save(record)
    try:
        transpiled = shared.transpile(circuit, **kwargs)
        counts = dict(shared._gate_counts(transpiled))
        cost = shared.TranspiledCircuitCost(
            pretranspile_gate_counts=shared._gate_counts(circuit),
            posttranspile_gate_counts=tuple(sorted(counts.items())),
            rz_count=counts.get('rz', 0), rz_depth=shared._filtered_depth(transpiled, 'rz'),
            cx_count=counts.get('cx', 0), cx_depth=shared._filtered_depth(transpiled, 'cx'),
            total_depth=int(transpiled.depth() or 0), circuit_size=int(transpiled.size()),
            qubit_count=transpiled.num_qubits, global_phase=float(transpiled.global_phase),
            compiler=compiler, compiler_settings_hash=shared.compiler_settings_hash(compiler),
            circuit_fingerprint=actual or 'uncacheable-circuit',
            actual_circuit_fingerprint=actual, backend_fingerprint=backend_fingerprint,
            transpiled_circuit=transpiled)
    except BaseException as exc:
        record['phase'] = 'TRANSPILE_OR_METRIC_EXCEPTION'
        record['execution_exception'] = exception_record(exc)
        save(record)
        raise
    record['phase'] = 'COST_RETURNED_BEFORE_IDENTITY_GUARD'
    record['actual_fingerprint'] = cost.actual_circuit_fingerprint
    record['matches_expected'] = cost.actual_circuit_fingerprint == expected
    record['posttranspile_instruction_count'] = len(transpiled.data)
    save(record)
    return cost

"""Layered acceptance. Historical representation hashes remain raw provenance.

No names are stripped and no reference hash is replaced. Target correctness is
checked by the evaluator's unchanged independent, phase-strict numeric guard.
"""
import hashlib
import json
from collections import Counter
from fractions import Fraction as F

import numpy as np

from ..df_correction_comparison import native as old_native
from ..df_correction_comparison.fingerprint_diagnostics import payload_digest


def ordered_native_ir(circuit):
    operations = []
    for position, item in enumerate(circuit.data):
        name = item.operation.name
        if name not in ('rz', 'sx', 'x', 'cx'):
            raise ArithmeticError('Unexpected native operation')
        wires = [circuit.find_bit(q).index for q in item.qubits]
        params = [str(F(float(p))) for p in item.operation.params]
        operations.append({'position': position, 'name': name,
                           'qubits': wires, 'params_binary64_exact': params})
    return {'num_qubits': circuit.num_qubits, 'operations': operations,
            'global_phase_binary64_exact': str(F(float(circuit.global_phase))),
            'basis': ['rz', 'sx', 'x', 'cx']}


def native_inventory(circuit, **kwargs):
    inventory = old_native.native_inventory(circuit, **kwargs)
    inventory['ordered_native_ir'] = ordered_native_ir(circuit)
    inventory['ordered_native_ir_sha256'] = payload_digest(inventory['ordered_native_ir'])
    validate_native_inventory(inventory)
    return inventory


def validate_native_inventory(inventory):
    ir = inventory['ordered_native_ir']
    if payload_digest(ir) != inventory['ordered_native_ir_sha256']:
        raise PermissionError('Native IR digest mismatch')
    counts = Counter()
    rz = []
    for i, item in enumerate(ir['operations']):
        name, wires, params = item['name'], item['qubits'], item['params_binary64_exact']
        if (item['position'] != i or name not in ir['basis']
            or len(wires) != (2 if name == 'cx' else 1)
            or len(set(wires)) != len(wires)
            or any(type(q) is not int or not 0 <= q < ir['num_qubits'] for q in wires)
            or len(params) != (1 if name == 'rz' else 0)):
            raise PermissionError('Native IR semantics/wires invalid')
        counts[name] += 1
        if name == 'rz':
            rz.append((i, params[0]))
    if (dict(counts) != inventory['gate_counts']
        or counts['rz'] != inventory['total_RZ'] or counts['cx'] != inventory['CX']
        or sum(v for k, v in counts.items() if k != 'cx') != inventory['1Q']
        or ir['global_phase_binary64_exact'] != inventory['global_phase_raw']
        or rz != [(x['position'], x['raw_binary64_radians']) for x in inventory['occurrences']]):
        raise PermissionError('Native IR cost/angle/phase inventory mismatch')
    return True


def bind_native_results(out, native):
    """Attach actual saved implementation identities to the frozen T inventory."""
    from ..df_correction_comparison.storage import artifact_identity
    bindings = []
    for r in (3, 6):
        name = 'E-r'+str(r)+'.json'
        identity = artifact_identity(out/name)
        identity['logical_path'] = name
        circuits = []
        for row in native[r]['native']:
            validate_native_inventory(row['inventory'])
            circuits.append({'arm': row['arm'], 'controlled': row['controlled'],
                             'label': row.get('label'),
                             'ordered_native_ir_sha256': row['inventory']['ordered_native_ir_sha256']})
        bindings.append({'result': identity, 'circuits': circuits})
    return bindings


def verify_native_bindings(out, inventory):
    from ..df_correction_comparison.storage import artifact_identity, load_json
    bindings = inventory['native_result_bindings']
    if len(bindings) != 2 or [b['result']['logical_path'] for b in bindings] != ['E-r3.json', 'E-r6.json']:
        raise PermissionError('Native result bindings missing/changed')
    for binding in bindings:
        name = binding['result']['logical_path']
        actual = artifact_identity(out/name)
        actual['logical_path'] = name
        if actual != binding['result']:
            raise PermissionError('Native result bytes changed after inventory freeze')
        saved = load_json(out/name)
        circuits = []
        for row in saved['native']:
            validate_native_inventory(row['inventory'])
            circuits.append({'arm': row['arm'], 'controlled': row['controlled'],
                             'label': row.get('label'),
                             'ordered_native_ir_sha256': row['inventory']['ordered_native_ir_sha256']})
        if circuits != binding['circuits']:
            raise PermissionError('Native circuit/inventory binding mismatch')
    return True


def validate_constructor(circuit, arrays):
    """A input matrix cannot be substituted, sign-flipped or phase-adjusted."""
    if 'input_target' not in arrays:
        return  # B fixed Pauli IR has its independent strict target guard.
    if len(circuit.data) != 1:
        raise PermissionError('Whole input must contain one unitary gate')
    item = circuit.data[0]
    from qiskit.circuit.library import UnitaryGate
    if not isinstance(item.operation, UnitaryGate):
        raise PermissionError('Whole input operation type changed')
    expected = np.asarray(arrays['input_target'])
    actual = np.asarray(item.operation.params[0])
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
        or actual.tobytes(order='C') != expected.tobytes(order='C')
        or [circuit.find_bit(q).index for q in item.qubits] != list(range(circuit.num_qubits))
        or float(circuit.global_phase) != 0):
        raise PermissionError('Whole input parameter/wire/phase identity mismatch')



def layered_transpile_and_measure_cost(circuit, compiler, *, expected,
                                         arrays, context, save):
    """One fingerprint attempt and one compile, same frozen no-backend settings.

    save(record) must synchronously persist each state (lossless chunk storage).
    Save/payload failure aborts before compile; no permissive fallback or retry.
    Raw recursive payload policy and encoding remain the shared policy.
    """
    validate_constructor(circuit, arrays)
    from ..df_correction_comparison.fingerprint_diagnostics import array_record, exception_record, _definition_index
    from trotterlib import rte_compiled_cost as shared
    shared._validate_transpile_context(compiler, None)
    record = {'kind': 'LAYERED_ACCEPTANCE_RAW_REPRESENTATION_PROVENANCE',
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
        raise  # Missing required raw provenance is a technical failure.
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
    record['historical_mismatch_rejects'] = False
    record['posttranspile_instruction_count'] = len(transpiled.data)
    save(record)
    return cost

"""Schema-only native DF connection checks. No circuits, sampling or costs.

Registry digests authenticate ordered DESCRIPTIONS, including producer-supplied
local matrix hashes. They do not certify those matrices against gate semantics.
The expected binding must come from independently pinned receipts, not a copy of
the candidate record. This module never supplies missing scientific metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from types import SimpleNamespace


def _digest(value, length=64):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{%d}' % length, value) is None:
        raise ValueError('Invalid immutable digest')


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer binding')


def _same(left, right):
    # Type-sensitive, finite JSON comparison; +0/-0 are deliberately distinct.
    encode = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return encode(left) == encode(right)


def runtime_basis_metadata(record, *, expected_basis_id, expected_basis_hash):
    """Return (name/params metadata, ordered qubits) for basis_from_runtime.

    Only the Gaussian gate set actually observed in the bounded H6/H8 audit is
    accepted. No Qiskit Gate/Operator or molecular operation is constructed.
    Registry hash and model operation_digest are intentionally different domains.
    """
    if record.get('phase_policy') != 'native_gate_list_only':
        raise ValueError('Missing/unsupported Gaussian phase convention')
    if record.get('control_scope') != 'uncontrolled':
        raise ValueError('Controlled Gaussian is outside the validated scope')
    n = record.get('num_system_qubits')
    _integer(n, 1)
    _digest(expected_basis_hash)
    if record.get('basis_hash') != expected_basis_hash:
        raise ValueError('Basis identity mismatch')
    if (not isinstance(expected_basis_id, str) or not expected_basis_id or
            record.get('basis_id') != expected_basis_id):
        raise ValueError('Missing/mismatched basis ID')
    ops = record.get('operations')
    if not isinstance(ops, list):
        raise ValueError('Missing ordered operation list')
    output = []
    for op in ops:
        if not isinstance(op, dict) or set(op) != {'name', 'qubits', 'parameters', 'matrix_sha256'}:
            raise ValueError('Incomplete operation description')
        name = op['name']
        if name not in ('rz', 'xx_plus_yy'):
            raise ValueError('Unsupported Gaussian gate; no opaque fallback')
        arity = 1 if name == 'rz' else 2
        qubits, params = op['qubits'], op['parameters']
        if not isinstance(qubits, list) or len(qubits) != arity:
            raise ValueError('Invalid ordered qubits')
        if any(type(q) is not int or not 0 <= q < n for q in qubits) or len(set(qubits)) != arity:
            raise ValueError('Invalid system qubit binding')
        if not isinstance(params, list) or len(params) != arity or any(type(p) is not str for p in params):
            raise ValueError('Expected producer string parameters')
        if any(not math.isfinite(float(p)) for p in params):
            raise ValueError('Nonfinite Gaussian parameter')
        _digest(op['matrix_sha256'])
        output.append((SimpleNamespace(name=name, params=tuple(params)), tuple(qubits)))
    payload = dict(num_system_qubits=n, operations=ops, hash_policy='ordered_local_basis_operations_v1')
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != expected_basis_hash:
        raise ValueError('Ordered operation/registry hash mismatch')
    return tuple(output)


_IDENTITY_FIELDS = (
    'execution_source', 'dataset_source', 'model_source', 'input_sha256',
    'hamiltonian_hash', 'df_hash', 'state_hash', 'profile_hash', 'compiler_hash',
    'candidate_id', 'family', 'formula', 'T', 'prefix', 'q', 'r', 'K', 'R', 'num_system_qubits',
    'ancilla', 'control_policy', 'control_state', 'axis', 'wrapper_scope', 'phase_receipt_sha256',
    'fragment_order_sha256', 'primitive_receipt_sha256', 'basis_registry_sha256',
    'sample_scope', 'normalization_receipt_sha256', 'shot_receipt_sha256',
    'numeric_allowance_receipt_sha256', 'epsilon', 'allowance_factor',
    'law_receipt_sha256', 'tau', 'tau_semantics', 'component_law_sha256',
    'model', 'basis_policy', 'rte_reuse_policy', 'occurrence_policy',
)


def validate_connection_identity(record, *, expected):
    """Check a complete declared connection envelope, never evaluate its law.

    This is not a launch gate, a numerical law validator, or evidence that a
    complete real H6/H8 envelope exists. Receipt contents/availability and local
    matrix semantics require separate source-bound verification.
    """
    for key in _IDENTITY_FIELDS:
        if key not in record or key not in expected or not _same(record[key], expected[key]):
            raise ValueError('Missing/mismatched connection field: ' + key)
    for key in ('execution_source', 'dataset_source', 'model_source'):
        _digest(record[key], 40)
    for key in ('input_sha256', 'hamiltonian_hash', 'df_hash', 'state_hash', 'profile_hash',
                'compiler_hash', 'phase_receipt_sha256', 'normalization_receipt_sha256',
                'shot_receipt_sha256', 'numeric_allowance_receipt_sha256',
                'fragment_order_sha256', 'primitive_receipt_sha256', 'basis_registry_sha256'):
        _digest(record[key])
    if record['family'] not in ('PF', 'PR') or record['model'] not in ('A', 'B'):
        raise ValueError('Unknown family/model')
    if record['formula'] not in ('2nd', '4th') or (record['family'] == 'PR' and record['formula'] != '2nd'):
        raise ValueError('Unsupported native formula binding')
    if not isinstance(record['candidate_id'], str) or not record['candidate_id']:
        raise ValueError('Missing candidate')
    for key in ('T', 'epsilon', 'allowance_factor'):
        value = record[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid finite scalar: ' + key)
    for key, minimum in (('prefix', 0), ('num_system_qubits', 1), ('q', 1), ('ancilla', 0)):
        _integer(record[key], minimum)
    if record['ancilla'] != record['num_system_qubits']:
        raise ValueError('Native ancilla ordering mismatch')
    if record['axis'] not in ('cosine', 'sine'):
        raise ValueError('Missing/unsupported axis')
    if record['control_policy'] not in ('ordinary', 'symmetric_directional'):
        raise ValueError('Unknown control convention')
    if type(record['control_state']) is not int or record['control_state'] != 1:
        raise ValueError('Unsupported native active control state')
    if record['wrapper_scope'] != 'measured_hadamard_state_preparation_excluded':
        raise ValueError('Wrapper scope mismatch')
    if record['sample_scope'] not in ('deterministic', 'exploration8', 'confirmation32'):
        raise ValueError('Sample scope mismatch')
    if record['basis_policy'] != 'registered_full_basis':
        raise ValueError('Support/history basis plans require a model extension')
    if record['occurrence_policy'] != 'closed_each_occurrence_and_prefix_half':
        raise ValueError('Cross-occurrence flattening is outside this model')
    if record['rte_reuse_policy'] not in ('disabled', 'raw_adjacent_equal_basis'):
        raise ValueError('Unsupported reuse policy')
    if record['family'] == 'PR':
        for key in ('r', 'R'):
            _integer(record[key], 1)
        _integer(record['K'])
        if record['K'] % 2 or record['R'] != record['q'] * record['r']:
            raise ValueError('Invalid finite Taylor/event dimensions')
        for key in ('law_receipt_sha256', 'component_law_sha256'):
            _digest(record[key])
        if (type(record['tau']) not in (int, float) or not math.isfinite(record['tau']) or
                record['tau_semantics'] != 'exact_rte_lambda_r*T/(q*r)'):
            raise ValueError('Tau/law semantics mismatch')
        if record['sample_scope'] == 'deterministic':
            raise ValueError('PR requires a separate sampled scope')
    elif any(record[key] is not None for key in
             ('r', 'R', 'K', 'law_receipt_sha256', 'component_law_sha256', 'tau', 'tau_semantics')):
        raise ValueError('PF must not invent an RTE law')
    elif record['sample_scope'] != 'deterministic':
        raise ValueError('PF sample scope mismatch')
    return {'status': 'SCHEMA_ONLY_VALIDATED', 'cost_evaluated': False,
            'receipt_contents_certified': False, 'runtime_authorization': False}

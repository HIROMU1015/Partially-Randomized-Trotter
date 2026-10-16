"""Validate frozen H8 descriptions; restoration needs an explicit permit.

No scientific imports. Gate factories and inspection are injected after the
runtime gate; metadata inspection never instantiates a gate or an operator.
"""
from copy import deepcopy
import math
from types import SimpleNamespace

from h8_strengthened_contract import digest


def real_hex(value):
    if type(value) is not str:
        raise ValueError('HEX_FLOAT_STRING_REQUIRED')
    result = float.fromhex(value)
    if not math.isfinite(result) or result.hex() != value:
        raise ValueError('NONCANONICAL_OR_NONFINITE_HEX')
    return result


def validate_block(block, index, n):
    if set(block) != {'block_id', 'basis_id', 'basis_hash', 'operations', 'diagonal_hex', 'lambda_hex'}:
        raise ValueError('EXPORTED_BLOCK_SCHEMA')
    bid = 'df-one-body-correction' if index == 0 else 'df-fragment-'+str(index-1)
    if block['block_id'] != bid or len(block['diagonal_hex']) != n:
        raise ValueError('BLOCK_ORDER_OR_DIAGONAL_DIMENSION')
    for value in block['diagonal_hex']:
        real_hex(value)
    if index == 0:
        if block['lambda_hex'] is not None:
            raise ValueError('ONE_BODY_LAMBDA')
    else:
        real_hex(block['lambda_hex'])
    if type(block['operations']) is not list or len(block['operations']) > 4096:
        raise ValueError('OPERATION_COUNT')
    for op in block['operations']:
        if set(op) != {'name', 'qubits', 'parameters', 'matrix_sha256'}:
            raise ValueError('OPERATION_SCHEMA')
        arity = {'rz':1, 'xx_plus_yy':2}.get(op['name'])
        if (arity is None or len(op['qubits']) != arity or len(op['parameters']) != arity or
                len(set(op['qubits'])) != arity or
                any(type(q) is not int or not 0 <= q < n for q in op['qubits']) or
                (arity == 2 and abs(op['qubits'][0]-op['qubits'][1]) != 1) or
                any(type(p) is not str or not math.isfinite(float(p)) for p in op['parameters']) or
                type(op['matrix_sha256']) is not str or len(op['matrix_sha256']) != 64 or
                any(c not in '0123456789abcdef' for c in op['matrix_sha256'])):
            raise ValueError('OPERATION_IDENTITY_OR_PARAMETERS')
    expected = digest(dict(num_system_qubits=n, operations=block['operations'],
                           hash_policy='ordered_local_basis_operations_v1'))
    if block['basis_hash'] != expected or block['basis_id'] != 'df-basis-'+expected:
        raise ValueError('BASIS_DESCRIPTION_HASH')


def validate_export(export, *, n=16, rank=28, old_prefixes=(7, 14, 21), expected_hashes=None):
    full = export['faithful:p'+str(rank)]
    fields = {'blocks', 'components', 'bases', 'constant_hex', 'extracted_hex',
              'lambda_r_hex', 'preparation_hash'}
    if (set(full) != fields or len(full['blocks']) != rank+1 or full['components'] or
            full['bases'] or real_hex(full['extracted_hex']) != 0 or
            real_hex(full['lambda_r_hex']) != 0):
        raise ValueError('FULL_PREPARATION_SCOPE')
    real_hex(full['constant_hex'])
    for i, block in enumerate(full['blocks']):
        validate_block(block, i, n)
    if expected_hashes is not None and full['preparation_hash'] != expected_hashes['faithful:p'+str(rank)]:
        raise ValueError('FULL_PREPARATION_IDENTITY')
    for p in old_prefixes:
        old = export['discard:p'+str(p)]
        if (set(old) != fields or old['blocks'] != full['blocks'][:p+1] or
                old['constant_hex'] != full['constant_hex'] or old['components'] or old['bases'] or
                real_hex(old['extracted_hex']) != 0 or real_hex(old['lambda_r_hex']) != 0 or
                (expected_hashes is not None and old['preparation_hash'] != expected_hashes['discard:p'+str(p)])):
            raise ValueError('SAVED_PREFIX_LINEAGE')
    return dict(status='PASS_METADATA_ONLY', n=n, rank=rank,
                prefix_lineage=list(old_prefixes), full_preparation_hash=full['preparation_hash'],
                block_description_sha256=digest(full['blocks']), scientific_objects_restored=False)


def restore_blocks(export, permit, *, gate_factory, inspect_operations, one_type, fragment_type, n=16, rank=28):
    permit.require('signal')  # Must precede every callback and live object.
    permit.take('full_preparation_decode')
    full = export['faithful:p'+str(rank)]
    blocks = []
    for i, row in enumerate(full['blocks']):
        validate_block(row, i, n)
        runtime = tuple((gate_factory(op['name'], tuple(float(p) for p in op['parameters'])),
                         tuple(op['qubits'])) for op in row['operations'])
        described, metadata = inspect_operations(runtime)
        if metadata != row['operations']:
            raise ValueError('RESTORED_RUNTIME_BASIS_DESCRIPTION_MISMATCH')
        common = dict(block_id=row['block_id'], basis_id=row['basis_id'], basis_hash=row['basis_hash'],
                      basis_change_operations=described, num_system_qubits=n, order_index=i,
                      runtime_basis_operations=runtime)
        diagonal = tuple(real_hex(x) for x in row['diagonal_hex'])
        if i == 0:
            value = one_type(**common, block_kind='one_body', original_fragment_index=None,
                             rank=None, diagonal_eigenvalues=diagonal)
        else:
            value = fragment_type(**common, block_kind='df_fragment', original_fragment_index=i-1,
                                  rank=i-1, diagonal_eta=diagonal, lam=real_hex(row['lambda_hex']))
        blocks.append(value)
    return tuple(blocks)


def prefix_view(blocks, full, prefix, *, export_sha256):
    if type(prefix) is not int or not 1 <= prefix < len(blocks):
        raise ValueError('PREFIX_VIEW_SCOPE')
    # This is a new view identity, never the old preparation_hash of a different
    # Hamiltonian. Fragment rank=index follows the fixed generation_partition.
    identity = digest(dict(policy='H8_FROZEN_FULL_PREFIX_VIEW_V1', prefix=prefix,
                           export_sha256=export_sha256,
                           full_preparation_hash=full['preparation_hash'],
                           blocks=[x.block_id for x in blocks[:prefix+1]]))
    return SimpleNamespace(deterministic_blocks=blocks[:prefix+1], preparation_hash=identity,
        constant_coefficient=real_hex(full['constant_hex']), extracted_identity_coefficient=0.,
        exact_rte_lambda_r=0., randomized_block_indices=(),
        deterministic_fragment_indices=tuple(range(prefix)),
        deterministic_block_order=tuple(x.block_id for x in blocks[:prefix+1]),
        num_system_qubits=blocks[0].num_system_qubits, ld=prefix)

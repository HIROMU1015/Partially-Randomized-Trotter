"""Read frozen preparation DESCRIPTIONS, without restoring scientific objects.

No NumPy/Qiskit imports, matrix evaluations, primitive production or gate costs.
The original full decoder is deliberately not called: it constructs live Gates.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


class FrozenMetadataGraph:
    """Bounded typed DAG reader; hash consistency is not a matrix certificate."""

    TYPES = frozenset(('DFPartialS2Preparation', 'DFDeterministicOneBodySpec',
        'DFDeterministicFragmentSpec', 'DFTailExtraction', 'DFDiagonalPauliComponent',
        'DFTailExtractionMetadata', 'SymbolicRTETail', 'BasisChangeMetadata',
        'DFRTEComponentCircuitSpec', 'DFRTEIdentityCircuitSpec', 'DFRTEEventPreparation',
        'BasisChangeOperation', 'RTEComponent', 'TailNormalizationMetadata'))

    def __init__(self, archive):
        if (type(archive) is not dict or
                set(archive) != {'schema', 'binding', 'codec', 'codec_digest', 'export_receipt'} or
                archive['schema'] != 'h6_frozen_df_preparations_v1'):
            raise ValueError('Unexpected frozen archive schema')
        codec = archive['codec']
        if (type(codec) is not dict or set(codec) != {'roots', 'nodes'} or
                digest(codec) != archive['codec_digest'] or
                type(codec['nodes']) is not dict or not 0 < len(codec['nodes']) <= 50000 or
                type(codec['roots']) is not dict or not 0 < len(codec['roots']) <= 8):
            raise ValueError('Invalid codec identity/domain')
        self.archive, self.nodes = archive, codec['nodes']
        self._checked = set()
        for key, node in self.nodes.items():
            if type(node) is not dict or digest(node) != key:
                raise ValueError('Frozen node hash mismatch')
        for key in self.nodes:
            self._validate({'ref': key}, set(), 0)
        for value in codec['roots'].values():
            self.fields(value, 'DFPartialS2Preparation')

    @classmethod
    def from_file(cls, path, *, expected_sha256):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 2**20:
            raise ValueError('Archive path/size rejected')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('Archive byte identity mismatch')
        def reject(value):
            raise ValueError('Nonfinite JSON constant: ' + value)
        archive = json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject)
        return cls(archive)

    def _validate(self, value, visiting, depth):
        if depth > 64:
            raise ValueError('Metadata depth cap')
        if value is None or type(value) in (str, int, bool):
            return
        if type(value) is not dict:
            raise ValueError('Only encoded scalar/reference values are supported')
        if set(value) == {'float_hex'}:
            self.scalar(value)
            return
        node = self.node(value)
        key = value['ref']
        if key in visiting:
            raise ValueError('Cyclic metadata reference')
        if key in self._checked:
            return
        visiting.add(key)
        kind = node.get('type')
        if kind == 'tuple' and set(node) == {'type', 'items'} and type(node['items']) is list:
            children = node['items']
        elif kind in self.TYPES and set(node) == {'type', 'fields'} and type(node['fields']) is dict:
            children = list(node['fields'].values())
        elif kind == 'gate' and set(node) == {'type', 'name', 'parameters', 'label', 'matrix_sha256'}:
            if node['name'] not in ('rz', 'xx_plus_yy') or type(node['parameters']) is not list:
                raise ValueError('Unsupported frozen gate description')
            if len(node['parameters']) != (1 if node['name'] == 'rz' else 2):
                raise ValueError('Gate parameter arity mismatch')
            if (type(node['matrix_sha256']) is not str or
                    re.fullmatch('[0-9a-f]{64}', node['matrix_sha256']) is None):
                raise ValueError('Invalid local matrix digest description')
            children = node['parameters']
            if any(type(self.scalar(v)) is not float for v in children):
                raise ValueError('Gate parameters must be stored numeric floats')
        elif kind == 'DFBasisRegistry' and set(node) == {'type', 'definitions'}:
            if type(node['definitions']) is not list:
                raise ValueError('Registry definitions must be ordered')
            seen, children = set(), []
            for definition in node['definitions']:
                if (type(definition) is not dict or set(definition) !=
                        {'basis_id', 'operations', 'num_system_qubits'} or
                        type(definition['basis_id']) is not str or not definition['basis_id'] or
                        type(definition['num_system_qubits']) is not int or
                        definition['num_system_qubits'] <= 0 or definition['basis_id'] in seen):
                    raise ValueError('Invalid/duplicate registry identity')
                seen.add(definition['basis_id'])
                children.append(definition['operations'])
        else:
            raise ValueError('Unsupported node type/fields')
        for child in children:
            self._validate(child, visiting, depth + 1)
        visiting.remove(key)
        self._checked.add(key)

    def node(self, value):
        if (type(value) is not dict or set(value) != {'ref'} or
                type(value['ref']) is not str or value['ref'] not in self.nodes):
            raise ValueError('Missing/invalid node reference')
        return self.nodes[value['ref']]

    def fields(self, value, expected_type):
        node = self.node(value)
        if node.get('type') != expected_type:
            raise ValueError('Unexpected metadata type: ' + expected_type)
        return node['fields']

    def sequence(self, value):
        node = self.node(value)
        if node.get('type') != 'tuple':
            raise ValueError('Expected encoded ordered tuple')
        return node['items']

    def scalar(self, value, depth=0):
        if depth > 64:
            raise ValueError('Scalar metadata depth cap')
        if value is None or type(value) in (str, int, bool):
            return value
        if type(value) is dict and set(value) == {'float_hex'}:
            if type(value['float_hex']) is not str:
                raise ValueError('Invalid hex float')
            result = float.fromhex(value['float_hex'])
            if not math.isfinite(result) or result.hex() != value['float_hex']:
                raise ValueError('Noncanonical/nonfinite hex float')
            return result
        return [self.scalar(x, depth + 1) for x in self.sequence(value)]

    def basis_description(self, operations, *, n, basis_id, basis_hash):
        """Validate the original registry description hash; instantiate nothing."""
        if type(n) is not int or n <= 0 or type(basis_id) is not str or not basis_id:
            raise ValueError('Missing basis identity/register')
        result = []
        for ref in self.sequence(operations):
            f = self.fields(ref, 'BasisChangeOperation')
            op = {k: self.scalar(v) for k, v in f.items()}
            if set(op) != {'name', 'qubits', 'parameters', 'matrix_sha256'}:
                raise ValueError('Incomplete basis description')
            arity = {'rz': 1, 'xx_plus_yy': 2}.get(op['name'])
            if (arity is None or len(op['qubits']) != arity or len(op['parameters']) != arity or
                    len(set(op['qubits'])) != arity or
                    any(type(q) is not int or not 0 <= q < n for q in op['qubits']) or
                    any(type(p) is not str or not math.isfinite(float(p)) for p in op['parameters'])):
                raise ValueError('Invalid Gaussian operation binding')
            result.append(op)
        if digest(dict(num_system_qubits=n, operations=result,
                       hash_policy='ordered_local_basis_operations_v1')) != basis_hash:
            raise ValueError('Basis registry description hash mismatch')
        return result

    def compare_runtime_descriptions(self, runtime, described):
        pairs = self.sequence(runtime)
        if len(pairs) != len(described):
            raise ValueError('Runtime/description operation count mismatch')
        for ref, op in zip(pairs, described):
            pair = self.sequence(ref)
            if len(pair) != 2:
                raise ValueError('Runtime operation pair schema')
            gate = self.node(pair[0])
            if (gate.get('type') != 'gate' or gate['name'] != op['name'] or
                    gate['matrix_sha256'] != op['matrix_sha256'] or
                    self.scalar(pair[1]) != op['qubits'] or
                    [self.scalar(p).hex() for p in gate['parameters']] !=
                    [float(p).hex() for p in op['parameters']]):
                raise ValueError('Runtime/description parameter, qubit or matrix claim mismatch')
        return True

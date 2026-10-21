"""Strict, bounded JSON coverage comparison; no numerical imports or work.

Only tuple/list representation is identified. Scalar types, binary64 values
(including signed zero), sequence order, keys and multiplicity remain exact.
The actual record and bounded differences are saved before rejection/action.
"""
from __future__ import annotations

import hashlib
import json
import math

MAX_DEPTH = 64
MAX_NODES = 200_000
MAX_BYTES = 4 * 2**20
MAX_DIFFERENCES = 24


def canonical_coverage(value):
    nodes = 0
    ancestors = set()

    def normalize(item, depth):
        nonlocal nodes
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise ValueError('COVERAGE_STRUCTURE_CAP')
        kind = type(item)
        if kind in (dict, list, tuple):
            identity = id(item)
            if identity in ancestors:
                raise ValueError('COVERAGE_CYCLE')
            ancestors.add(identity)
            try:
                if kind is dict:
                    if any(type(key) is not str for key in item):
                        raise ValueError('COVERAGE_KEY_TYPE')
                    return {key: normalize(part, depth + 1) for key, part in item.items()}
                return [normalize(part, depth + 1) for part in item]
            finally:
                ancestors.remove(identity)
        if kind is float:
            if not math.isfinite(item):
                raise ValueError('COVERAGE_NONFINITE')
        elif kind is str:
            if len(item.encode('utf-8')) > MAX_BYTES:
                raise ValueError('COVERAGE_BYTE_CAP')
        elif kind not in (int, bool, type(None)):
            raise ValueError('COVERAGE_VALUE_TYPE')
        return item

    normalized = normalize(value, 0)
    chunks, size = [], 0
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False,
                               separators=(',', ':'))
    for chunk in encoder.iterencode(normalized):
        part = chunk.encode('utf-8')
        size += len(part)
        if size > MAX_BYTES:
            raise ValueError('COVERAGE_BYTE_CAP')
        chunks.append(part)
    return b''.join(chunks)


def _differences(expected, actual):
    """Short paths/type labels only; full values live in the bounded records."""
    differences = []
    truncated = False

    def add(path, reason):
        nonlocal truncated
        if len(differences) >= MAX_DIFFERENCES:
            truncated = True
            return
        differences.append({'path': path[:256], 'reason': reason})

    def walk(left, right, path='$'):
        if truncated:
            return
        if type(left) is not type(right):
            add(path, 'SCALAR_OR_CONTAINER_TYPE')
        elif type(left) is dict:
            for key in sorted(set(left).union(right)):
                if key not in left or key not in right:
                    add(path + '.' + key, 'KEY_SET')
                else:
                    walk(left[key], right[key], path + '.' + key)
        elif type(left) is list:
            if len(left) != len(right):
                add(path, 'SEQUENCE_LENGTH')
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, path + '[' + str(index) + ']')
        elif type(left) is float:
            if left.hex() != right.hex():
                add(path, 'BINARY64_VALUE')
        elif left != right:
            add(path, 'VALUE')

    walk(expected, actual)
    return differences, truncated


def assert_coverage_binding(expected, actual, writer):
    """Writer enforces aggregate output/diagnostic caps and exclusive writes."""
    expected_bytes = canonical_coverage(expected)
    actual_bytes = canonical_coverage(actual)
    equal = expected_bytes == actual_bytes
    expected_json, actual_json = json.loads(expected_bytes), json.loads(actual_bytes)
    differences, truncated = _differences(expected_json, actual_json) if not equal else ([], False)
    receipt = {'schema': 'track_a_ax2b_coverage_comparison_v3',
               'comparison': 'exact JSON scalars/order; tuple/list representation only',
               'equal': equal, 'expected_sha256': hashlib.sha256(expected_bytes).hexdigest(),
               'actual_sha256': hashlib.sha256(actual_bytes).hexdigest(),
               'expected_canonical_bytes': len(expected_bytes), 'actual_canonical_bytes': len(actual_bytes),
               'differences': differences, 'differences_truncated': truncated,
               'representation_only_match_is_not_scientific_PASS': True,
               'numerical_allowance_certified': False, 'N': None, 'G': None,
               'H6_status': 'H6_NOT_AUTHORIZED', 'contract_status': 'DRAFT_NOT_AUTHORIZATION',
               'mandatory_stop': True, 'next_stage_authorized': False}
    writer.write('actual_coverage.json', actual_json, diagnostic=True)
    writer.write('coverage_comparison_v3.json', receipt, diagnostic=True)
    if not equal:
        raise ValueError('ACTUAL_COVERAGE_CHANGED')
    return receipt

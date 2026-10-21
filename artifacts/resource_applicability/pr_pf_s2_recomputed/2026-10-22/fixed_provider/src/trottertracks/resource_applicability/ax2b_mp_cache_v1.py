"""Cell/precision-local MP operator reuse. No persistent or native-side cache.

Heap accounting counts Python objects reachable from stored MP scalars; it is
an implementation budget, not a process RSS or numerical error certificate.
"""
from __future__ import annotations

import hashlib
import json
import sys


def oracle_identity(ham, basis, state, cell, *, T, scalar, lambda_r, extracted_identity, dps, backend):
    def exact(value):
        z = complex(value)
        return [z.real.hex(), z.imag.hex()]
    value = {'source_policy': 'ax2b_mp_cell_local_v1', 'backend': backend, 'dps': dps,
             'basis': [int(i) for i in basis], 'state': [exact(z) for z in state],
             'constant': exact(ham.constant), 'one': [[exact(z) for z in row] for row in ham.one_body],
             'weights': [exact(z) for z in ham.lambdas],
             'blocks': [[[exact(z) for z in row] for row in block] for block in ham.g_matrices],
             'cell': cell, 'T': exact(T), 'scalar': exact(scalar),
             'lambda_r': exact(lambda_r), 'extracted_identity': exact(extracted_identity)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def matrix_heap_bytes(matrix):
    seen = set()
    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        count = sys.getsizeof(value)
        if isinstance(value, (tuple, list)):
            count += sum(size(v) for v in value)
        elif hasattr(value, '_mpc_'):
            count += size(value._mpc_)
        elif hasattr(value, '_mpf_'):
            count += size(value._mpf_)
        return count
    # Include a conservative dict/key allowance for dense storage. Temporary
    # copies and factory workspaces are covered by the worker AS limit instead.
    return sys.getsizeof(matrix) + sum(256 + size(z) for z in matrix)


class OperatorCache:
    def __init__(self, identity, *, entries=128, heap_bytes=64*2**20,
                 exp_generations=64, polynomial_generations=4, lookups=4096, progress=None, mp_precision=None):
        limits = (entries, heap_bytes, exp_generations, polynomial_generations, lookups)
        if any(type(v) is not int or v < 1 for v in limits):
            raise ValueError('MP_CACHE_LIMITS')
        self.identity, self.progress = identity, progress
        self.mp_precision = mp_precision
        self.limits = dict(zip(('entries', 'heap_bytes', 'exp', 'polynomial', 'lookups'), limits))
        self.values = {}
        self.stats = {'hits': 0, 'misses': 0, 'lookups': 0, 'heap_bytes': 0, 'entries': 0,
                      'attempted': {'exp': 0, 'polynomial': 0}, 'completed': {'exp': 0, 'polynomial': 0}}

    def notify(self, point, key):
        if self.progress is not None:
            self.progress({'point': point, 'operator_key': repr(key), 'oracle_work': self.report()})

    def get(self, key, factory, *, kind='exp'):
        if self.mp_precision is not None:
            import mpmath as mp
            if mp.mp.prec != self.mp_precision:
                raise ValueError('MP_CACHE_PRECISION_CONTEXT')
        if kind not in ('exp', 'polynomial'):
            raise ValueError('MP_FACTORY_KIND')
        if self.stats['lookups'] >= self.limits['lookups']:
            raise RuntimeError('MP_LOOKUP_CAP')
        self.stats['lookups'] += 1
        key = (self.identity, key)
        if key in self.values:
            self.stats['hits'] += 1
            return self.values[key].copy()
        if self.stats['entries'] >= self.limits['entries']:
            raise RuntimeError('MP_CACHE_ENTRY_CAP')
        if self.stats['attempted'][kind] >= self.limits[kind]:
            raise RuntimeError('MP_GENERATION_CAP:'+kind)
        self.stats['misses'] += 1
        self.stats['attempted'][kind] += 1
        self.notify('operator_started', key)
        value = factory()
        self.stats['completed'][kind] += 1
        stored = value.copy()
        count = matrix_heap_bytes(stored)
        if self.stats['heap_bytes'] + count > self.limits['heap_bytes']:
            self.notify('operator_completed_cache_rejected', key)
            raise RuntimeError('MP_CACHE_HEAP_CAP')
        self.values[key] = stored
        self.stats['entries'] += 1
        self.stats['heap_bytes'] += count
        self.notify('operator_completed', key)
        return value.copy()

    def report(self):
        return json.loads(json.dumps(self.stats))

"""Bounded, value-keyed encoding reuse for small numeric matrix parameters.

This caches canonical text, never scientific outputs, circuits or compiler
results. Its lifetime is one candidate (including both measurement axes).
"""
from collections import OrderedDict
import sys

from .identity import EncodedValue, require
from .streaming import array_values


class ArrayEncodingMemo:
    def __init__(self, *, max_bytes=4*2**20, max_entries=1024):
        require(type(max_bytes) is int and 0 <= max_bytes <= 4*2**20,
                'matrix encoding cache byte budget')
        require(type(max_entries) is int and 0 <= max_entries <= 1024,
                'matrix encoding cache entry budget')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.retained_bytes = self.hits = self.misses = 0

    def encode(self, array):
        require(array.dtype.kind in 'fciub', 'matrix parameter dtype')
        # No large tobytes/encoded tree, even for an empty high-dimensional view.
        if (array.nbytes > 4096 or array.ndim > 8 or
                self.max_bytes == 0 or self.max_entries == 0):
            return None
        shape, dtype = tuple(array.shape), array.dtype.str
        raw = array.tobytes(order='C')
        key = (dtype, shape, raw)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0]
        self.misses += 1
        import numpy as np
        # Encode the immutable snapshot used for the key. Mutation, dtype,
        # signed zero and logical ordering cannot produce a stale cache hit.
        snapshot = np.frombuffer(raw, dtype=array.dtype).reshape(shape)
        encoded = EncodedValue(dict(array=array_values(snapshot), shape=list(shape), dtype=dtype))
        # Conservative per-entry allowance includes OrderedDict/value metadata.
        cost = (512 + sys.getsizeof(key) + sys.getsizeof(shape) +
                sum(sys.getsizeof(n) for n in shape) + sys.getsizeof(dtype) +
                sys.getsizeof(raw) + sys.getsizeof(encoded) + sys.getsizeof(encoded.text))
        if cost > self.max_bytes:
            return encoded
        while self.entries and (self.retained_bytes + cost > self.max_bytes or
                                len(self.entries) >= self.max_entries):
            _, (_, old_cost) = self.entries.popitem(last=False)
            self.retained_bytes -= old_cost
        self.entries[key] = (encoded, cost)
        self.retained_bytes += cost
        return encoded

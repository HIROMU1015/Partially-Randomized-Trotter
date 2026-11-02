"""Bounded value-keyed canonical fragments, local to one fingerprint walk.

Only small instruction records and small definitions qualify. Every key is
rebuilt from current values, including nested fragments; mutable operation or
definition identities are never cache keys. The public lazy serializer is
unchanged unless this memo is explicitly supplied.
"""
from collections import OrderedDict
import math
import sys

from .identity import EncodedValue, require


class _Uncacheable(Exception):
    pass


class InstructionEncodingMemo:
    MAX_KEY_BYTES = 16384
    MAX_NODES = 384
    MAX_DEFINITION_INSTRUCTIONS = 16

    def __init__(self, *, max_bytes=2*2**20, max_entries=2048):
        require(type(max_bytes) is int and 0 <= max_bytes <= 2*2**20,
                'instruction encoding byte budget')
        require(type(max_entries) is int and 0 <= max_entries <= 2048,
                'instruction encoding entry budget')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.retained_bytes = self.hits = self.misses = self.bypasses = 0

    @classmethod
    def _key(cls, value):
        remaining = [cls.MAX_NODES, cls.MAX_KEY_BYTES]

        def visit(item):
            remaining[0] -= 1
            if remaining[0] < 0:
                raise _Uncacheable
            if isinstance(item, EncodedValue):
                return ('encoded', visit(item.text))
            if item is None:
                return ('null',)
            if type(item) is bool:
                return ('bool', item)
            if type(item) is int:
                if item.bit_length() > 256:
                    raise _Uncacheable
                return ('int', item)
            if type(item) is float:
                require(math.isfinite(item), 'nonfinite instruction number')
                return ('float', item.hex())
            if type(item) is str:
                remaining[1] -= 4*len(item)  # conservative UTF-8 allowance
                if remaining[1] < 0:
                    raise _Uncacheable
                return ('str', item)
            if isinstance(item, (list, tuple)):
                if len(item) > remaining[0]:
                    raise _Uncacheable
                return ('list', tuple(visit(x) for x in item))
            if isinstance(item, dict):
                if len(item) > remaining[0] or not all(type(k) is str for k in item):
                    raise _Uncacheable
                return ('dict', tuple((visit(k), visit(v)) for k, v in sorted(item.items())))
            # LazyList/large matrix traversal is left to the streaming encoder.
            raise _Uncacheable

        return visit(value)

    @staticmethod
    def _key_size(value):
        size = sys.getsizeof(value)
        if isinstance(value, tuple):
            size += sum(InstructionEncodingMemo._key_size(x) for x in value)
        return size

    def encode(self, value):
        if not self.max_bytes or not self.max_entries:
            self.bypasses += 1
            return value
        try:
            key = self._key(value)
        except _Uncacheable:
            self.bypasses += 1
            return value
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0]
        self.misses += 1
        encoded = EncodedValue(value)  # the same exact, finite encoder
        cost = 512 + self._key_size(key) + sys.getsizeof(encoded) + sys.getsizeof(encoded.text)
        if cost > self.max_bytes:
            return encoded
        while self.entries and (self.retained_bytes + cost > self.max_bytes or
                                len(self.entries) >= self.max_entries):
            _, (_, old_cost) = self.entries.popitem(last=False)
            self.retained_bytes -= old_cost
        self.entries[key] = (encoded, cost)
        self.retained_bytes += cost
        return encoded

    def definition(self, record, instruction_count):
        if (not self.max_bytes or not self.max_entries or
                instruction_count > self.MAX_DEFINITION_INSTRUCTIONS):
            return record
        # At most 16 instruction values. Nested large definitions/arrays remain
        # lazy and cause encode() to bypass; never build a full circuit tree.
        entries = []
        for entry in record['instructions']:
            if not isinstance(entry, EncodedValue):
                return record
            entries.append(entry)
        return self.encode({**record, 'instructions': entries})

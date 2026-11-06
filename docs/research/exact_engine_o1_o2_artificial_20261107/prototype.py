"""Artificial-only O1/O2. No scientific data, RNG or gate builders."""
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction as F

from oracle import old


@dataclass(frozen=True, slots=True)
class PreparedMatrix:
    matrix: tuple
    rows: tuple
    diagonal: bool
    nnz: int


def prepare_matrix(matrix):
    d = len(matrix)
    if d < 1 or d & (d - 1) or any(len(row) != d for row in matrix):
        raise ValueError("local matrix must be square with power-of-two dimension")
    copied = []
    for row in matrix:
        cells = []
        for z in row:
            if len(z) != 2 or any(type(x) is not F for x in z):
                raise ValueError("exact real/imag Fraction pairs required")
            cells.append(tuple(z))
        copied.append(tuple(cells))
    frozen = tuple(copied)
    rows = tuple(tuple((j, z) for j, z in enumerate(row) if z != old.ZERO)
                 for row in frozen)
    return PreparedMatrix(frozen, rows,
                          all(all(i == j for j, _ in row) for i, row in enumerate(rows)),
                          sum(map(len, rows)))


def validate_support(dimension, qubits):
    qs = tuple(qubits)
    if type(dimension) is not int or dimension < 1 or dimension & (dimension - 1):
        raise ValueError("state dimension must be a positive power of two")
    n = dimension.bit_length() - 1
    if any(type(q) is not int or q < 0 or q >= n for q in qs) or len(set(qs)) != len(qs):
        raise ValueError("invalid gate support")
    return qs


def _validate(v, prepared, qubits):
    qs = validate_support(len(v), qubits)
    if len(prepared.matrix) != 1 << len(qs):
        raise ValueError("matrix/support size mismatch")
    if any(len(z) != 2 or any(type(x) is not F for x in z) for z in v):
        raise ValueError("exact real/imag Fraction state required")
    return qs


def _contract(v, prepared, blocks):
    out = list(v)  # Retain the original buffer/copy policy; O5 is out of scope.
    for indices in blocks:
        for i, at in enumerate(indices):
            if prepared.diagonal:
                out[at] = (old.multiply(prepared.rows[i][0][1], v[at])
                           if prepared.rows[i] else old.ZERO)
            else:
                value = old.ZERO
                for j, coefficient in prepared.rows[i]:
                    value = old.add(value, old.multiply(coefficient, v[indices[j]]))
                out[at] = value
    return out


def apply_o1(v, prepared, qubits):
    qs = _validate(v, prepared, qubits)
    d = 1 << len(qs)
    if prepared.nnz == d * d:
        return old.apply_exact(v, prepared.matrix, qs)
    mask = sum(1 << q for q in qs)
    # Exactly the original per-block index expression; no O2 reuse here.
    blocks = ([base | sum(((j >> k) & 1) << q for k, q in enumerate(qs))
               for j in range(d)] for base in range(len(v)) if not base & mask)
    return _contract(v, prepared, blocks)


@dataclass(frozen=True, slots=True)
class IndexPlan:
    dimension: int
    qubits: tuple
    blocks: tuple


def build_index_plan(dimension, qubits):
    qs = validate_support(dimension, qubits)
    d = 1 << len(qs)
    mask = sum(1 << q for q in qs)
    blocks = tuple(tuple(base | sum(((j >> k) & 1) << q for k, q in enumerate(qs))
                         for j in range(d))
                   for base in range(dimension) if not base & mask)
    return IndexPlan(dimension, qs, blocks)


class IndexCache:
    """Private bounded lazy LRU; only immutable plans are returned."""
    def __init__(self, max_entries=32):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("positive integer cache entry count required")
        self.max_entries = max_entries
        self._plans = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get(self, dimension, qubits):
        qs = validate_support(dimension, qubits)
        key = (dimension, qs)
        if key in self._plans:
            self.hits += 1
            self._plans.move_to_end(key)
            return self._plans[key]
        plan = build_index_plan(dimension, qs)
        self.misses += 1
        self._plans[key] = plan
        if len(self._plans) > self.max_entries:
            self._plans.popitem(last=False)
            self.evictions += 1
        return plan

    def info(self):
        return dict(entries=len(self._plans), hits=self.hits, misses=self.misses,
                    evictions=self.evictions,
                    stored_indices=sum(sum(map(len, p.blocks)) for p in self._plans.values()))


def apply_o1_o2(v, prepared, qubits, cache):
    qs = _validate(v, prepared, qubits)
    d = 1 << len(qs)
    if prepared.nnz == d * d:
        return old.apply_exact(v, prepared.matrix, qs)
    plan = cache.get(len(v), qs)
    return _contract(v, prepared, plan.blocks)

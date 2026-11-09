"""T0 resources of an explicit native word, without compilation or simulation.

The recurrence is adapted from C7-2 ``weighted_depth``; max-plus row updates
and composition extend the integer mock at the fixed a2b7099d design. Frozen
builders/counters are unchanged. Only the optional Qiskit reader imports Qiskit.
Neither this module nor its results qualify pre-P0 descriptors as P0 prices.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Integral, Real


DEPTHS = ("rz_depth", "cx_depth", "total_depth")
NATIVE_EXACT = "NATIVE_EXACT"
P0_UNQUALIFIED = "P0_UNQUALIFIED"
_ARITY = {"rz": (1, 0, 1), "sx": (1, 0, 0), "x": (1, 0, 0),
          "cx": (2, 0, 0), "measure": (1, 1, 0)}


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(label + " must be a nonnegative integer")
    return int(value)


def _real(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(label + " must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(label + " must be finite")
    return value


def _mapping(values, length, bound, label):
    values = tuple(_integer(v, label) for v in values)
    if len(values) != length or len(set(values)) != length or any(v >= bound for v in values):
        raise ValueError(label + " must be an injective in-range mapping of every local wire")
    return values


@dataclass(frozen=True)
class NativeInstruction:
    """Declared native instruction; qubit and classical indices are separate.

    RZ requires one finite bound angle. Zero angles still count: no optimizer
    or algebraic cancellation is performed. Conditions/directives are rejected.
    """
    name: str
    qubits: tuple[int, ...]
    clbits: tuple[int, ...] = ()
    params: tuple[float, ...] = ()
    condition: object = None

    def __post_init__(self):
        if not isinstance(self.name, str) or self.name not in _ARITY:
            raise ValueError("unsupported native gate: " + str(self.name))
        if self.condition is not None:
            raise ValueError("conditional instructions are outside the native word domain")
        q = tuple(_integer(v, "qubit") for v in self.qubits)
        c = tuple(_integer(v, "clbit") for v in self.clbits)
        p = tuple(_real(v, "parameter") for v in self.params)
        if (len(q), len(c), len(p)) != _ARITY[self.name] or len(set(q)) != len(q) or len(set(c)) != len(c):
            raise ValueError("native instruction arity or duplicate wire")
        object.__setattr__(self, "qubits", q)
        object.__setattr__(self, "clbits", c)
        object.__setattr__(self, "params", p)


def _arrivals(value, width):
    if value is None:
        return {key: [0] * width for key in DEPTHS}
    if set(value) != set(DEPTHS):
        raise ValueError("provide all three depth arrival vectors")
    out = {key: [_integer(v, "arrival") for v in value[key]] for key in DEPTHS}
    if any(len(row) != width for row in out.values()):
        raise ValueError("arrival vector width")
    return out


def _advance(rows, wires, weight):
    """Existing design-mock row recurrence, also usable with one-column rows."""
    row = []
    for column in zip(*(rows[i] for i in wires)):
        values = [v for v in column if v is not None]
        row.append(max(values) + weight if values else None)
    for i in wires:
        rows[i] = list(row)


def _identity(width):
    return [[0 if i == j else None for j in range(width)] for i in range(width)]


def _compose(later, earlier):
    """Max-plus product in chronological order: earlier, then later."""
    width = len(later)
    result = []
    for row in later:
        out = []
        for j in range(width):
            terms = [row[k] + earlier[k][j] for k in range(width)
                     if row[k] is not None and earlier[k][j] is not None]
            out.append(max(terms) if terms else None)
        result.append(tuple(out))
    return tuple(result)


def _result(nq, nc, counts, arrivals, phase):
    return dict(qualification=NATIVE_EXACT, p0_qualification=P0_UNQUALIFIED,
        metrics=dict(rz_count=counts.get("rz", 0), cx_count=counts.get("cx", 0),
            **{key: max(arrivals[key], default=0) for key in DEPTHS},
            circuit_size=sum(counts.values()), qubit_count=nq,
            measurement_count=counts.get("measure", 0), global_phase=phase),
        logical_width=nq, classical_width=nc, counts=dict(counts),
        wire_arrivals={key: tuple(arrivals[key]) for key in DEPTHS})


class NativeCounter:
    """One-pass O(L) count/depth, O(W) storage; optional O(W²) transfer capture.

    Nonuniform arrival vectors are integer dependency levels, one vector per
    metric, with quantum wires followed by classical wires. Full register width
    is declared explicitly and includes idle qubits, as in existing metrics.
    """
    def __init__(self, num_qubits, num_clbits=0, *, arrivals=None,
                 capture_transfer=False, global_phase=0.):
        self.num_qubits = _integer(num_qubits, "num_qubits")
        self.num_clbits = _integer(num_clbits, "num_clbits")
        if type(capture_transfer) is not bool:
            raise ValueError("capture_transfer must be bool")
        self.global_phase = _real(global_phase, "global_phase")
        self.width = self.num_qubits + self.num_clbits
        starts = _arrivals(arrivals, self.width)
        self._finish = {key: [[v] for v in starts[key]] for key in DEPTHS}
        self._transfers = {key: _identity(self.width) for key in DEPTHS} if capture_transfer else None
        self._counts = Counter()

    def feed(self, instruction):
        if not isinstance(instruction, NativeInstruction):
            raise TypeError("NativeInstruction required; no implicit lowering")
        if (any(q >= self.num_qubits for q in instruction.qubits)
                or any(c >= self.num_clbits for c in instruction.clbits)):
            raise ValueError("native wire outside declared register")
        wires = instruction.qubits + tuple(self.num_qubits + c for c in instruction.clbits)
        for key in DEPTHS:
            weight = int(key == "total_depth" or key == instruction.name + "_depth")
            _advance(self._finish[key], wires, weight)
            if self._transfers is not None:
                _advance(self._transfers[key], wires, weight)
        self._counts[instruction.name] += 1
        return self

    def extend(self, instructions):
        for instruction in instructions:
            self.feed(instruction)
        return self

    def result(self):
        arrivals = {key: [row[0] for row in self._finish[key]] for key in DEPTHS}
        return _result(self.num_qubits, self.num_clbits, self._counts, arrivals, self.global_phase)

    def template(self):
        """Universal transfer, independent of this counter's initial arrivals."""
        if self._transfers is None:
            raise ValueError("transfer capture was not enabled")
        return NativeTemplate(self.num_qubits, self.num_clbits,
            tuple(tuple(tuple(row) for row in self._transfers[key]) for key in DEPTHS),
            tuple(sorted(self._counts.items())), self.global_phase)


@dataclass(frozen=True)
class NativeTemplate:
    """Fixed-word counts and three immutable wire transfers; no inverse method.

    Use from_stream for an actual inverse word supplied by the emitter. Matrix
    inversion is neither implemented nor a substitute for actual A† lowering.
    """
    num_qubits: int
    num_clbits: int
    transfers: tuple
    counts: tuple
    global_phase: float = 0.

    def __post_init__(self):
        nq = _integer(self.num_qubits, "num_qubits")
        nc = _integer(self.num_clbits, "num_clbits")
        width = nq + nc
        matrices = tuple(tuple(tuple(None if v is None else _integer(v, "transfer weight")
                                    for v in row) for row in m) for m in self.transfers)
        if (len(matrices) != len(DEPTHS) or any(len(m) != width or
                any(len(row) != width or not any(v is not None for v in row) for row in m)
                for m in matrices)):
            raise ValueError("transfer matrix shape or missing dependency")
        counts = tuple((name, _integer(n, "gate count")) for name, n in self.counts)
        if any(name not in _ARITY for name, _ in counts) or len(dict(counts)) != len(counts):
            raise ValueError("native count keys")
        object.__setattr__(self, "num_qubits", nq)
        object.__setattr__(self, "num_clbits", nc)
        object.__setattr__(self, "transfers", matrices)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "global_phase", _real(self.global_phase, "global_phase"))

    @classmethod
    def from_stream(cls, instructions, num_qubits, num_clbits=0, *, global_phase=0.):
        return NativeCounter(num_qubits, num_clbits, capture_transfer=True,
                             global_phase=global_phase).extend(instructions).template()

    def apply(self, arrivals=None):
        starts = _arrivals(arrivals, self.num_qubits + self.num_clbits)
        return {key: tuple(max(v + starts[key][j] for j, v in enumerate(row) if v is not None)
                           for row in matrix) for key, matrix in zip(DEPTHS, self.transfers)}

    def result(self, arrivals=None):
        return _result(self.num_qubits, self.num_clbits, dict(self.counts),
                       self.apply(arrivals), self.global_phase)

    def then(self, following):
        if not isinstance(following, NativeTemplate):
            raise TypeError("NativeTemplate required")
        if (self.num_qubits, self.num_clbits) != (following.num_qubits, following.num_clbits):
            raise ValueError("embed templates into the same registers before composition")
        counts = Counter(dict(self.counts)) + Counter(dict(following.counts))
        return NativeTemplate(self.num_qubits, self.num_clbits,
            tuple(_compose(later, earlier) for earlier, later in zip(self.transfers, following.transfers)),
            tuple(sorted(counts.items())), math.fsum((self.global_phase, following.global_phase)))

    def embed(self, num_qubits, num_clbits=0, *, qubit_map, clbit_map=()):
        nq, nc = _integer(num_qubits, "num_qubits"), _integer(num_clbits, "num_clbits")
        q = _mapping(qubit_map, self.num_qubits, nq, "qubit_map")
        c = _mapping(clbit_map, self.num_clbits, nc, "clbit_map")
        mapping = q + tuple(nq + i for i in c)
        matrices = []
        for original in self.transfers:
            matrix = _identity(nq + nc)
            for i, target in enumerate(mapping):
                matrix[target] = [None] * (nq + nc)
                for j, source in enumerate(mapping):
                    matrix[target][source] = original[i][j]
            matrices.append(tuple(tuple(row) for row in matrix))
        return NativeTemplate(nq, nc, tuple(matrices), self.counts, self.global_phase)


def count_native(instructions, num_qubits, num_clbits=0, *, arrivals=None, global_phase=0.):
    return NativeCounter(num_qubits, num_clbits, arrivals=arrivals,
                         global_phase=global_phase).extend(instructions).result()


def remap_native(instructions, *, qubit_map, clbit_map=()):
    """Lazy injective remapping; the receiving counter checks target bounds."""
    q = tuple(_integer(i, "qubit_map") for i in qubit_map)
    c = tuple(_integer(i, "clbit_map") for i in clbit_map)
    if len(set(q)) != len(q) or len(set(c)) != len(c):
        raise ValueError("non-injective wire map")
    for op in instructions:
        if not isinstance(op, NativeInstruction):
            raise TypeError("NativeInstruction required")
        if any(i >= len(q) for i in op.qubits) or any(i >= len(c) for i in op.clbits):
            raise ValueError("local wire missing from mapping")
        yield NativeInstruction(op.name, tuple(q[i] for i in op.qubits),
                                tuple(c[i] for i in op.clbits), op.params)


def iter_qiskit_native(circuit):
    """Read an existing caller-owned native circuit, without copy or lowering.

    Optional boundary adapter for old emitters' .data. Raw CRZ/Gaussian/CSWAP,
    custom gates, directives and classical conditions are rejected. This does
    not construct a full circuit, compile it, or validate its unitary/phase.
    Pass circuit.global_phase explicitly to count_native when needed.
    """
    from qiskit.circuit import Measure
    from qiskit.circuit.library import RZGate, SXGate, XGate, CXGate
    allowed = dict(rz=RZGate, sx=SXGate, x=XGate, cx=CXGate, measure=Measure)
    if getattr(circuit, "num_vars", 0) or getattr(circuit, "num_captured_vars", 0):
        raise ValueError("classical variables are unsupported")
    for item in circuit.data:
        op = item.operation
        if (op.name not in allowed or op.base_class is not allowed[op.name]
                or getattr(op, "_condition", None) is not None
                or getattr(op, "_directive", False)):
            raise ValueError("bound unconditional standard native operation required")
        yield NativeInstruction(op.name, tuple(circuit.find_bit(q).index for q in item.qubits),
            tuple(circuit.find_bit(c).index for c in item.clbits), tuple(op.params))

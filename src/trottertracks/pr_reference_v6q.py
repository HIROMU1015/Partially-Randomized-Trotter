"""Opt-in exact-binary vacuum sum. No scientific runner is called on import.

Only phase arithmetic is new. F/G builders, the reference gadget, SOURCE
inverse and measurement wrapper remain in the preserved pilot/transfer code.
Scientific use requires its separately fixed execution scope.
"""
from fractions import Fraction
import math
import struct
import sys

_MIN_NORMAL = Fraction.from_float(sys.float_info.min)
_MAX_FINITE = Fraction.from_float(sys.float_info.max)
_SHAPES = {'rz': (1, 1), 'rzz': (2, 1), 'xx_plus_yy': (2, 2),
           'z': (1, 0), 'p': (1, 1), 'id': (1, 0)}


def _binary_input(value):
    """Preserve float storage; accept integers only with exact binary64 image."""
    if isinstance(value, float):
        # Bypass a subclass's __float__; obtain its actual binary64 storage.
        binary = float.__float__(value)
    elif type(value) is int:
        try:
            binary = float(value)
        except OverflowError as exc:
            raise ValueError('Integer is outside binary64 range') from exc
        if not math.isfinite(binary) or Fraction.from_float(binary) != value:
            raise ValueError('Integer conversion would round')
    else:
        raise TypeError('Unsupported parameter; no decimal/symbolic coercion')
    if not math.isfinite(binary):
        raise ValueError('Nonfinite binary input')
    return Fraction(*binary.as_integer_ratio())


def _lane_sum(qc):
    """Exact vacuum phase of trusted fixed-SOURCE gate families, in order."""
    total = _binary_input(qc.global_phase)
    for inst in qc.data:
        op = inst.operation
        if op.name not in _SHAPES:
            raise ValueError('Unqualified vacuum factor: ' + op.name)
        width, arity = _SHAPES[op.name]
        if (op.num_qubits != width or len(op.params) != arity
                or len(inst.qubits) != width or inst.clbits
                or getattr(op, 'condition', None) is not None):
            raise ValueError('Unsupported gate shape or conditional operation')
        params = tuple(_binary_input(x) for x in op.params)
        if op.name in ('rz', 'rzz'):
            total -= params[0] / 2  # Fraction division, no binary64 half-angle.
    return total


def _verify_nearest_even(total, candidate):
    """Exact midpoint check of a proposed finite RN64 result; no phase fit."""
    if type(candidate) is not float or not math.isfinite(candidate):
        raise ArithmeticError('Final conversion did not return finite binary64')
    center = Fraction.from_float(candidate)
    even = not (struct.unpack('>Q', struct.pack('>d', candidate))[0] & 1)
    for direction in (-math.inf, math.inf):
        adjacent = math.nextafter(candidate, direction)
        if not math.isfinite(adjacent):
            # total is already constrained to the closed finite-normal range.
            continue
        midpoint = (center + Fraction.from_float(adjacent)) / 2
        wrong_side = total < midpoint if direction < 0 else total > midpoint
        if wrong_side or (total == midpoint and not even):
            raise ArithmeticError('Final conversion is not round-to-nearest-even')


def _round_once(total):
    if (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.max_exp,
            struct.calcsize('d'), sys.float_info.rounds) != (2, 53, 1024, 8, 1):
        raise RuntimeError('Unsupported binary64 runtime model')
    if abs(total) > _MAX_FINITE:
        raise OverflowError('Exact phase exceeds the supported finite range')
    if total and abs(total) < _MIN_NORMAL:
        # Preserve the inherited normal-arithmetic error model; no new
        # subnormal absolute-error allowance is silently introduced.
        raise ArithmeticError('Nonzero subnormal exact phase is unqualified')
    candidate = float(total)  # The ONLY rounding of the combined exact sum.
    _verify_nearest_even(total, candidate)
    return candidate


def exact_binary_phase(f, g):
    """RN64(S_F+S_G), with a checked final conversion, no periodic reduction.

    Inputs are already-built lanes; no E0, ground vector or Hamiltonian is
    accepted. Gate semantics assume the genuine fixed Qiskit SOURCE families,
    not arbitrary custom gates that merely reuse their names.
    """
    if f.num_qubits != g.num_qubits:
        raise ValueError('Lane widths differ')
    return _round_once(_lane_sum(f) + _lane_sum(g))


def _legacy_module():
    from . import pr_reference_interference_pilot
    return pr_reference_interference_pilot


def phase_for_round(f, g, *, round_m=None, mode='legacy'):
    """Default reproduces transfer's old f+g callsite; exact opt-in is m17 only.

    The caller must bind round_m to the unchanged saved schedule. This helper
    does not infer the round from a circuit or qualify an m17 trajectory.
    """
    if mode == 'legacy':
        old = _legacy_module()
        return old.vacuum_phase(f) + old.vacuum_phase(g)
    if mode != 'exact_binary_m17' or type(round_m) is not int or round_m != 17:
        raise ValueError('Exact binary mode is opt-in for saved m17 only')
    return exact_binary_phase(f, g)


def reference_for_round(f, g, *, round_m=None, mode='legacy'):
    """Reuse the old gadget; only the passed ancilla relative P angle changes.

    Calling this on scientific lanes builds a circuit and needs separate
    scientific execution permission. Q1 tests use a mock gadget only.
    """
    phase = phase_for_round(f, g, round_m=round_m, mode=mode)
    return _legacy_module().reference_core(f, g, phase)

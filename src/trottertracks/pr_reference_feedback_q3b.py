"""Q3-B opt-in interval nearest-branch adapter; Q1 remains unchanged.

The proof is within the preserved 192-bit directed-interval model. Phase
width coverage for the two fixed saved schedules is supplied by B1, not by
the optional legacy float diagnostic. No scientific data is loaded on import.
"""
from __future__ import annotations
from fractions import Fraction as F
from . import pr_reference_feedback_q1 as q1

PHASE_WIDTH_LIMIT = F(1, 1 << 160)  # radians, accepted arg enclosure width
CENTER_WIDTH_LIMIT = F(1, 1 << 150)  # radians, base-phase enclosure width
KAPPA_LIMIT = F(1, 1 << 140)  # Ha, energy-center enclosure width
BASIC_SLACK = F(1, 1 << 160)  # radians, coarse 192-bit arithmetic bound
OMEGA_E = F(1, 1 << 170)  # Ha, final energy-hull restoration only
OMEGA_Q0 = F(1, 1 << 64)  # radians, coarse inverse/hull arithmetic bound
ARITHMETIC_ANGLE_LIMIT = F(1, 1 << 50)  # radians, B1 audit threshold


def bounds(value):
    q = q1._q0()
    try:
        return tuple(q.exact(t) for t in value._mpi_)
    except (AssertionError, TypeError, ValueError):
        raise q1.UnqualifiedPrefix('Nonfinite or invalid interval') from None


def width(value):
    lo, hi = bounds(value)
    return hi-lo


def count_phase(row):
    p = q1._count_phase(row)
    if width(p) > PHASE_WIDTH_LIMIT:
        raise q1.UnqualifiedPrefix('Arg enclosure exceeds the fixed B1 width contract')
    return p


def unique_integer_cell(value):
    """Strict nearest-integer cell; no float, point floor, or tie breaking."""
    lo, hi = bounds(value)
    left, right = lo+F(1,2), hi+F(1,2)
    j = left.numerator//left.denominator
    if (right.numerator//right.denominator != j
            or not (F(j)-F(1,2) < lo <= hi < F(j)+F(1,2))):
        raise q1.UnqualifiedPrefix('Nearest integer cell is not unique')
    return j


def prefix_from_counts(history, *, saved_arm=None, include_legacy=True):
    """Fixed-h m0..16 prefix; interval selection precedes any float diagnostic.

    Valid on the old prefix success event and fixed Gershgorin alias promise.
    A runtime acceptance does not observe that event or qualify native gates.
    """
    q = q1._q0()
    history = tuple(history)
    if len(history) != 17 or any(not isinstance(r,q1.RoundCounts) for r in history) or tuple(r.m for r in history) != tuple(range(17)):
        raise ValueError('Exactly one ordered m0..16 count history is required')
    if saved_arm is not None:
        ns = q1._saved_prefix_shots(saved_arm)
        if len(ns) != 17 or any((r.X.n,r.Y.n)!=(n,n) for r,n in zip(history,ns)):
            raise q1.UnqualifiedPrefix('Counts do not match the fixed saved arm schedule')
    phases = [count_phase(r) for r in history]
    center = phases[0]  # m0 canonical representative, including -pi cut rule.
    certificates = [dict(m=0,branch_unwrapped=0,center_width=str(width(center)),unique=True)]
    for m, phase in enumerate(phases[1:],1):
        power = 1 << m
        cell = (power*center-phase)/(2*q.mp.iv.pi)
        j = unique_integer_cell(cell)
        if abs(j) > power:
            raise q1.UnqualifiedPrefix('Branch exceeds the fixed-model arithmetic domain')
        center = (phase+2*j*q.mp.iv.pi)/power
        if width(center) > CENTER_WIDTH_LIMIT:
            raise q1.UnqualifiedPrefix('Base-phase enclosure exceeds the B0 width contract')
        lo,hi = bounds(cell)
        margin = min(lo-(F(j)-F(1,2)),F(j)+F(1,2)-hi)
        certificates.append(dict(m=m,branch_unwrapped=j,branch_mod_q=j%power,
            cell=q.pack(cell),cell_margin=str(margin),center_width=str(width(center)),unique=True))
    # Fixed H4 has |theta|<1/8. On the success event the m16 center is <1/4.
    # This coarse guard preserves that fixed alias promise, not a new prior.
    if bounds(abs(center))[1] >= 1:
        raise q1.UnqualifiedPrefix('History cannot support the fixed H4 linear alias branch')
    h = q1.REFERENCE_TIME
    ec = -center/q.iv(h)
    if width(ec) > KAPPA_LIMIT:
        raise q1.UnqualifiedPrefix('Energy-center enclosure exceeds the B0 width contract')
    w = q.iv(q1.CAP)/((1<<16)*q.iv(h))
    energy = q.I([q.endpoints(ec-w)[0],q.endpoints(ec+w)[1]])
    diagnostic = dict(status='NOT_REQUESTED',basis_for_selection=False)
    if include_legacy:
        # Diagnostic failure never replaces or rejects the proved interval cell.
        try:
            import math
            values = [math.atan2(float(r.Y.mean),float(r.X.mean)) for r in history]
            diagnostic = dict(status='AVAILABLE',basis_for_selection=False,
                              result=q1._legacy_decoder()(values,[1<<m for m in range(17)]))
        except Exception as exc:
            diagnostic = dict(status='UNAVAILABLE',basis_for_selection=False,error=repr(exc))
    return q1.Prefix(history,diagnostic,center,phases[-1],ec,energy,h,1<<16,
        tuple(certificates),saved_arm,status='INTERVAL_SELECTION_ON_FIXED_PREFIX_SUCCESS_EVENT')


# Same implementations, types, error accounting, and fixed n=340 as Q1.
feedback_from_prefix = q1.feedback_from_prefix
feedback_readout = q1.feedback_readout
decode_counts = q1.decode_counts

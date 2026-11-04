"""Partial B proof helpers; supplied artificial inputs only, no H4 entry point.

This append-only extension reuses the frozen finite-grid and SOURCE loaders.
No function supplies production a/c/d, certifies libm, or changes the decoder.
The RN bounds below are mathematical conditional lemmas, not runtime checks.
"""
from fractions import Fraction as Q
import math

import numpy as np
from flint import acb
from common import D, encode, real
from decoder_finite import feasible_pairs, represented_mean

UNIT_ROUNDOFF = Q(1, 2**53)
MAX_EXACT_MEAN_SHOTS = 2**52


def mean_bound_under_rn(n, style):
    """Coordinate bound, conditional on binary64 RN; see addendum proof.

    integer_sum also covers np.mean(int8 +/-1) and fsum outcomes followed by
    division when their documented accumulation paths and conditions hold.
    """
    D.integer(n)
    if n > MAX_EXACT_MEAN_SHOTS:
        D.stop('mean exact-integer domain not established')
    if style not in ('integer_sum', 'count_grid'):
        D.stop('measurement conversion path not fixed')
    return UNIT_ROUNDOFF * (1 if style == 'integer_sum' else 3)


def prewrap_bound_under_rn(q):
    """Uniform BEFORE candidate wrap: not c_m. Binary64 RN/gradual required.

    principal in [-pi64, pi64], branch in range(q), 2 <= q <= 2**17.
    All intermediate operations are the existing scalar SOURCE operations.
    The 2**17 boundary is the requested algorithm domain, not a resource cap.
    """
    D.integer(q)
    if q < 2 or q > 2**17 or q & (q-1):
        D.stop('prewrap lemma requires power-of-two q in [2,2**17]')
    if not abs(real(math.pi) - D.arb.pi()) < real(2 * UNIT_ROUNDOFF):
        D.stop('pi64 constant lemma not established at this precision')
    return 64 * UNIT_ROUNDOFF


def finite_input_envelope(row, style):
    """Cover EVERY exact strict-success count pair of one supplied round.

    Output includes mean conversion, scalar np.angle and the first SOURCE wrap.
    This finite-domain certificate needs no unproved libm accuracy assumption.
    It applies only to this scalar call shape/path on the evaluated runtime.
    SOURCE row centers must be exact rationals (Q.from_float for saved floats).
    No decoder histories, scientific data loaders, or automatic H4 application.
    """
    mean_bound_under_rn(row['N_cosine'], style)
    mean_bound_under_rn(row['N_sine'], style)
    pairs = feasible_pairs(row)
    unresolved = dict(decision='UNKNOWN/STOP', a_supplied_grid_upper=None,
                      H4_uniform_decoder_decision='UNKNOWN/STOP')
    if not pairs:
        return dict(unresolved, reason='empty supplied strict-success set')
    functions, _ = D.existing_functions()
    errors, coordinate_errors = [], [Q(), Q()]
    for k, l in pairs:
        exact = [Q(2*k, row['N_cosine'])-1, Q(2*l, row['N_sine'])-1]
        floats = [represented_mean(k, row['N_cosine'], style),
                  represented_mean(l, row['N_sine'], style)]
        if exact == [0, 0] or complex(*floats) == 0:
            return dict(unresolved, reason='zero signal in supplied set')
        if not all(math.isfinite(x) for x in floats):
            return dict(unresolved, reason='nonfinite mean')
        angle = float(np.angle(complex(*floats)))  # production scalar call shape
        if not math.isfinite(angle):
            return dict(unresolved, reason='nonfinite scalar angle')
        principal = functions['_wrap_phase'](angle)
        if not math.isfinite(principal):
            return dict(unresolved, reason='nonfinite SOURCE wrap')
        if not -math.pi <= principal <= math.pi:
            return dict(unresolved, reason='SOURCE principal outside candidate lemma domain')
        exact_angle = acb(*map(real, exact)).arg()
        errors.append(D.circle_distance(principal, exact_angle))
        for axis in range(2):
            coordinate_errors[axis] = max(coordinate_errors[axis],
                                         abs(Q.from_float(floats[axis])-exact[axis]))
    return dict(decision='EXHAUSTIVE_SUPPLIED_ROUND_INPUT_ONLY',
                successful_count_pairs=len(pairs), measurement_mean_path=style,
                angle_call_shape='scalar complex -> scalar np.angle',
                a_supplied_grid_upper=encode(max(errors)),
                coordinate_error_exact=[str(x) for x in coordinate_errors],
                H4_uniform_decoder_decision='UNKNOWN/STOP')

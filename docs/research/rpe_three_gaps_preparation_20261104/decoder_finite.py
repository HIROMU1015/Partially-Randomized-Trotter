"""Exhaustive checker for supplied small artificial finite prefixes.

No H4 loader. This is a feasibility demonstration, not a tractable uniform H4
verifier. It does not create full-domain libm error bounds from spot checks.
"""
from fractions import Fraction as Q
from itertools import product
import math
import numpy as np
from flint import acb, arb
from common import D, encode, real, upper


def represented_mean(k, n, style='count_grid'):
    D.integer(n)
    if type(k) is not int or not 0 <= k <= n:
        D.stop('invalid plus count')
    if style == 'count_grid':
        return float(2.0*np.int64(k)/n-1.0)  # old _axis_distribution path
    if style == 'integer_sum':
        return (2*k-n)/n  # exact sum of +/-1, one rounded division
    D.stop('measurement conversion path not fixed')


def feasible_pairs(row):
    """Exact open coordinate-success set, not a Float64 acceptance mask."""
    nc, ns = row['N_cosine'], row['N_sine']
    D.integer(nc); D.integer(ns)
    xc, xs = Q(row['mu_cosine']), Q(row['mu_sine'])
    ec, es = Q(row['eps_cosine']), Q(row['eps_sine'])
    if ec <= 0 or es <= 0:
        D.stop('positive coordinate epsilon required')
    return [(k, l) for k in range(nc+1) for l in range(ns+1)
            if abs(Q(2*k, nc)-1-xc) < ec and abs(Q(2*l, ns)-1-xs) < es]


def distance_interval(x, y):
    return abs(acb(0, real(x)-real(y)).exp().arg())


def finite_prefix_check(rows, phase_reference, style='count_grid', decoder=None):
    q = D.schedule([r['q'] for r in rows])
    functions, _ = D.existing_functions()
    decode = decoder or functions['reconstruct_rpe_phase']
    observations, angle_errors = [], []
    for row in rows:
        pairs = feasible_pairs(row)
        if not pairs:
            return dict(decision='UNKNOWN/STOP', reason='empty feasible set')
        values, errors = [], []
        for k, l in pairs:
            x, y = Q(2*k, row['N_cosine'])-1, Q(2*l, row['N_sine'])-1
            z = complex(represented_mean(k, row['N_cosine'], style),
                        represented_mean(l, row['N_sine'], style))
            if (x == 0 and y == 0) or z == 0:
                return dict(decision='UNKNOWN/STOP', reason='zero-signal realization')
            phase = float(np.angle(z))
            if not math.isfinite(phase):
                return dict(decision='UNKNOWN/STOP', reason='nonfinite angle')
            exact_angle = acb(real(x), real(y)).arg()
            errors.append(upper(distance_interval(phase, exact_angle)))
            values.append(((k, l), phase))
        observations.append(values)
        angle_errors.append(encode(max(errors)))
    traces = 0
    for trace in product(*observations):
        result = decode([t[1] for t in trace], q)
        for m, r in enumerate(result['rounds']):
            principal = real(r['measured_round_phase'])
            roots = [(principal+2*arb.pi()*k)/q[m] for k in range(q[m])]
            # cos(distance) is strictly decreasing on [0,pi]. This avoids an
            # interval Arg cut at an exactly antipodal WRONG root.
            scores = [(t-real(phase_reference)).cos() for t in roots]
            chosen = max(range(q[m]), key=lambda k: D.ball.dyadic(scores[k].lower()))
            if any(not scores[chosen].lower() > upper(scores[k])
                   for k in range(q[m]) if k != chosen):
                return dict(decision='UNKNOWN/STOP', reason='interval oracle tie')
            if r['selected_branch'] != chosen:
                return dict(decision='COUNTEREXAMPLE_ON_SUPPLIED_PREFIX', round=m,
                            counts=[t[0] for t in trace], expected=chosen,
                            actual=r['selected_branch'])
        traces += 1
    return dict(decision='EXHAUSTIVE_SUPPLIED_FINITE_PREFIX_ONLY', traces=traces,
                feasible_pairs_per_round=[len(x) for x in observations],
                angle_error_upper_on_supplied_grids=angle_errors,
                H4_uniform_decoder_decision='UNKNOWN/STOP')

"""Exact kappa intervals of the PR schedule (S2: one signed piece per round).

The mapper (B ``pr_rpe_rmse.paper_event_mapper``) sets, per round m,
    r* = kappa * lambda_R^2 * T_m^2,   R_m = max(J, ceil(ceil(r*) / q_m))  =  max(J, ceil(kappa * c_m / q_m)),
with c_m = lambda_R^2 T_m^2 and every quantity an exact rational (Fraction of the stored values).  R_m is a non-decreasing
step function of kappa, so the 13-round physical schedule changes only where kappa * c_m / q_m crosses an integer.  The
next distinct schedule above (below) kappa0 therefore starts right after (at) an exactly computable breakpoint, and no
schedule lies between two kappas with equal R vectors.  Generated configurations are still checked against the mapper.
"""
from fractions import Fraction
import math


def _frac(v):
    if isinstance(v, dict):
        return Fraction(int(v['numerator']), int(v['denominator']))
    return Fraction(v) if not isinstance(v, str) else Fraction(v)


def constants(config):
    """(c_m, q_m, J_m) per round from a canonical configuration (kappa-independent within an L_D / width family)."""
    lam = _frac(config['spec']['split']['lambda_R'])
    rows = []
    for r in sorted(config['rounds'], key=lambda r: r['m']):
        T = _frac(r['T']);J = len(r['structure'].get('signed_weights') or [1])
        rows.append(dict(m=r['m'], c=lam * lam * T * T, q=int(r['q_PF']), J=J))
    return rows


def R_vector(rows, kappa):
    kappa = Fraction(kappa)
    return [max(r['J'], math.ceil(Fraction(math.ceil(kappa * r['c']), r['q']))) for r in rows]


def simplest_between(a, b, closed_right=True):
    """Simplest rational x with a < x <= b (or a < x < b): smallest denominator, then smallest numerator."""
    a, b = Fraction(a), Fraction(b)
    if not a < b:
        raise ValueError('empty interval')
    q = 1
    while True:
        p = math.floor(a * q) + 1  # smallest p with p/q > a
        x = Fraction(p, q)
        if x < b or (closed_right and x == b):
            return x
        q += 1
        if q > 10 ** 7:
            return b if closed_right else (a + b) / 2


def next_distinct(rows, kappa0, direction, bounds=None):
    """The first schedule different from kappa0's in the given direction ('up' or 'down').

    Returns dict(kappa=representative (simplest rational inside the interval), interval=[lo, hi], changed_rounds,
    R=new R vector, R0) or None when no different schedule exists in that direction (within bounds)."""
    k0 = Fraction(kappa0);R0 = R_vector(rows, k0)
    if direction == 'up':
        # R_m rises above R0_m once kappa > R0_m * q_m / c_m
        t = [Fraction(R0[i] * r['q']) / r['c'] for i, r in enumerate(rows)]
        lo = min(t)
        R1 = R_vector(rows, lo + (min(x for x in t if x > lo) - lo) / 2 if any(x > lo for x in t) else lo * 2)
        R1 = [R0[i] + (1 if t[i] == lo else 0) for i in range(len(rows))]
        hi = min(Fraction(R1[i] * r['q']) / r['c'] for i, r in enumerate(rows))
        if bounds and lo >= Fraction(bounds[1]):
            return None
        top = min(hi, Fraction(bounds[1])) if bounds else hi
        rep = simplest_between(lo, top)
        return dict(kappa=rep, interval=[lo, hi], changed_rounds=[rows[i]['m'] for i in range(len(rows)) if t[i] == lo],
                    R0=R0, R=R_vector(rows, rep))
    # down: R_m falls below R0_m once kappa <= (R0_m - 1) * q_m / c_m (only rounds above their floor J)
    s = [Fraction((R0[i] - 1) * r['q']) / r['c'] for i, r in enumerate(rows) if R0[i] > r['J']]
    if not s:
        return None
    hi = max(s);idx = [i for i, r in enumerate(rows) if R0[i] > r['J'] and Fraction((R0[i] - 1) * r['q']) / r['c'] == hi]
    R1 = [R0[i] - (1 if i in idx else 0) for i in range(len(rows))]
    lows = [Fraction((R1[i] - 1) * r['q']) / r['c'] for i, r in enumerate(rows) if R1[i] > r['J']]
    lo = max(lows) if lows else Fraction(0)
    if bounds and hi < Fraction(bounds[0]):
        return None
    bottom = max(lo, Fraction(bounds[0])) if bounds else lo
    rep = simplest_between(bottom, hi) if bottom < hi else hi
    return dict(kappa=rep, interval=[lo, hi], changed_rounds=[rows[i]['m'] for i in idx], R0=R0, R=R_vector(rows, rep))


class Sched:
    """Family schedules for the neighbourhood plan: {L_D: constants rows (c as exact string, q, J)}."""

    def __init__(self, families, bounds):
        self.f = {int(k): [dict(m=r['m'], c=Fraction(r['c']), q=int(r['q']), J=int(r['J'])) for r in rows] for k, rows in families.items()}
        self.bounds = [Fraction(b) for b in bounds]

    def known(self, ld):
        return int(ld) in self.f

    def R(self, ld, k):
        return tuple(R_vector(self.f[int(ld)], k)) if self.known(ld) else None

    def next(self, ld, k, direction):
        if not self.known(ld):
            return None
        d = next_distinct(self.f[int(ld)], k, direction, self.bounds)
        if d is None or not (self.bounds[0] <= d['kappa'] <= self.bounds[1]):
            return None
        return d['kappa']

    def describe(self, ld, k, direction):
        d = next_distinct(self.f[int(ld)], k, direction, self.bounds) if self.known(ld) else None
        return None if d is None else dict(kappa=str(d['kappa']), interval=[str(x) for x in d['interval']],
                                           interval_float=[float(x) for x in d['interval']], changed_rounds=d['changed_rounds'],
                                           R_from=d['R0'], R_to=d['R'])


def encode_rows(rows):
    return [dict(m=r['m'], c=str(r['c']), q=r['q'], J=r['J']) for r in rows]

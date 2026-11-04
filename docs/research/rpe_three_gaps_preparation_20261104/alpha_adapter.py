"""Direct Hoeffding upper bounds. No saved-data loader or execution on import."""
from fractions import Fraction as Q
from common import D, ball, encode, real, upper


def failure_upper(n, epsilon, u_cat):
    D.integer(n)
    eps, u = D.positive(epsilon), D.nonnegative(u_cat)
    t = eps-u
    if not t > 0:
        return dict(decision='UNKNOWN/STOP', t=encode(t), probability_upper=None)
    # For positive t, using a lower t is conservative. Arb encloses the whole
    # expression, and the final dyadic endpoint is explicitly taken upward.
    probability = upper(2*(-real(n)*t*t/2).exp())
    return dict(decision='BOUND_AVAILABLE', t=encode(t), probability_upper=encode(probability))


def aggregate(axis_bounds, target='0.05'):
    limit = ball.rational(target)
    target = D.positive(limit)
    if not 0 < limit < 1 or not axis_bounds:
        D.stop('invalid direct-tail target or empty axis inventory')
    if any(x['probability_upper'] is None for x in axis_bounds):
        return dict(decision='UNKNOWN/STOP', failure_upper=None, success_lower=None)
    # Exact sum of the certified dyadic upper endpoints avoids an inward sum.
    endpoints = [Q(x['probability_upper']['upper']) for x in axis_bounds]
    if any(x < 0 for x in endpoints):
        D.stop('negative purported probability upper bound')
    total = sum(endpoints, Q())
    total_ball = real(total)
    # Exact rational <= handles equality without a floating tolerance.
    passed = total <= limit
    return dict(decision='DIRECT_HOEFFDING_95_SUFFICIENT' if passed else 'UNKNOWN/STOP',
                failure_upper_exact=str(total), failure_upper=encode(total_ball),
                success_lower_exact=str(max(Q(), 1-total)),
                target=encode(target), success_lower=encode(real(max(Q(), 1-total))))

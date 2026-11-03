"""Pure rational helpers for artificial bridge tests. No loaders or CLI."""
from fractions import Fraction as Q
import math


def exact(value):
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Finite binary64 required')
        return Q.from_float(value)
    if type(value) is int or isinstance(value, (Q, str)):
        return Q(value)
    raise ValueError('Exact rational or binary64 required')


def bernoulli_grid_mean(probability, bits=53):
    """E[+1 if k/2^bits < p else -1], independent uniform grid k."""
    probability = exact(probability)
    if not 0 <= probability <= 1 or type(bits) is not int or bits < 1:
        raise ValueError('Probability/grid outside domain')
    scaled = probability*(1 << bits)
    count = -(-scaled.numerator//scaled.denominator)
    return 2*Q(count, 1 << bits)-1


def postprocess_rounding_bound():
    """Accepted binary64 clipped t, round-to-nearest: <= u + 2u.

    fl(1+t) has absolute error <= u, division by two is exact here;
    converting p to the strict uniform53 Bernoulli mean adds < 2u.
    This bound contains no H4 input and does not establish guard acceptance.
    """
    return Q(3, 1 << 53)


def average_gain(forward_growth, reverse_growth, event_growth, order_mass, *, r, q):
    """Majorant, never the exact mean of a nonlinear floating-point program."""
    forward_growth, reverse_growth = exact(forward_growth), exact(reverse_growth)
    if (forward_growth < 1 or reverse_growth < 1 or type(r) is not int or r < 1
            or type(q) is not int or q < 1 or len(event_growth) != len(order_mass)):
        raise ValueError('Invalid growth/count contract')
    mass = [exact(p) for p in order_mass]
    growth = [exact(g) for g in event_growth]
    if not mass or sum(mass) != 1 or any(p < 0 for p in mass) or any(g < 1 for g in growth):
        raise ValueError('Normalized order mass and growth >= 1 required')
    event = sum(p*g for p, g in zip(mass, growth, strict=True))
    return (forward_growth*reverse_growth*event**r)**q


def combine_virtual_scope(logical_upper, arithmetic_upper, *, completion_proved):
    """No missing bridge or undefined abort domain is silently filled by zero."""
    if completion_proved is not True or logical_upper is None or arithmetic_upper is None:
        return {'U_virtual_IID': None, 'fixed_PCG64_IID_certified': False}
    values = [exact(logical_upper), exact(arithmetic_upper)]
    if any(x < 0 for x in values):
        raise ValueError('Negative error upper bound')
    return {'U_virtual_IID': sum(values)+postprocess_rounding_bound(), 'fixed_PCG64_IID_certified': False}


def strict_budget(upper, shot_upper, epsilon):
    if upper is None:
        return 'UNDETERMINED'
    upper, shot_upper, epsilon = map(exact, [upper, shot_upper, epsilon])
    if min(upper, shot_upper) < 0 or epsilon <= 0:
        raise ValueError('Invalid error/statistical budget')
    return 'SUFFICIENT_CONDITION_CONFIRMED' if upper+shot_upper < epsilon else 'NOT_CERTIFIED_BY_THIS_BOUND'

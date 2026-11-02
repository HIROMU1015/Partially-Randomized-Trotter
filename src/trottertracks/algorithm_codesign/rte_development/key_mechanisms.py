"""Saved-only linear T-price bookkeeping; no generator or numerical backend."""
from collections import Counter
from fractions import Fraction as F


def inventory_terms(gates):
    literal = sum(g[0] in ('T', 't') for g in gates)
    keys = Counter()
    for g in gates:
        if g[0] == 'R':
            if g[3] not in (-1, 1):
                raise ValueError('signed primitive required')
            keys[str(F(g[2]))] += 1
    return literal, keys


def linear_price(literal, multiplicity, prices):
    return F(literal) + sum((F(m) * prices[k] for k, m in multiplicity.items()), F(0))


def one_key_boundary(delta, coefficient, current):
    """A symbolic price substitution, never a new synthesis or budget."""
    d, c, t = F(delta), F(coefficient), F(current)
    if not c:
        return {'coefficient': c, 'current': t, 'tie_price': None,
                'nonnegative_tie': False, 'strict_sign_flip_integer': None}
    root = t - d / c
    flip = None
    if d < 0 and c > 0:
        flip = max(0, root.numerator // root.denominator + 1)
    elif d < 0 and c < 0 and root > 0:
        flip = 0
    elif d > 0 and c < 0:
        flip = max(0, root.numerator // root.denominator + 1)
    elif d > 0 and c > 0 and root > 0:
        flip = 0
    return {'coefficient': c, 'current': t, 'tie_price': root,
            'nonnegative_tie': root >= 0, 'strict_sign_flip_integer': flip,
            'P5_less_region': 't < tie' if c > 0 else 't > tie',
            'fixed_other_prices_law_bias_and_N': True}


def symmetric_root_split(q0, q1, price0, price1):
    q0, q1, p0, p1 = map(F, (q0, q1, price0, price1))
    price = (q0 + q1) * (p1 - p0) / 2
    probability = (p0 + p1) * (q1 - q0) / 2
    assert price + probability == q1 * p1 - q0 * p0
    return {'price': price, 'probability': probability,
            'total': price + probability, 'causal_share': False}

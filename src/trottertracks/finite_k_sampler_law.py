"""Categorical-map diagnostics and conditional TV bounds; no physical inputs.

Reuses the saved 53-bit CDF helper, not a new sampler. Its uniform-grid law
does not certify PCG64 independence or a fixed-seed trajectory distribution.
Only synthetic inputs are evaluated in this study. No import starts science.
"""
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np

CDF_SOURCE = '155529e20a8fac39e00786c40b57830b71203c4f'
CDF_PATH = 'scripts/h46_long_time_native.py'
GRID = 2**53


@lru_cache(maxsize=None)
def saved_cdf_helper(root):
    """Load just the existing pure function; never import its old launcher."""
    from .r_price_phase1_science import functions
    scope = dict(Fraction=Fraction)
    functions(root, CDF_SOURCE, CDF_PATH, ('cdf_law',), scope)
    return scope['cdf_law']


def _probabilities(values):
    values = tuple(Fraction(v) for v in values)
    if not values or any(v < 0 for v in values) or sum(values) <= 0:
        raise ValueError('Nonempty nonnegative weights with positive sum required')
    total = sum(values)
    return tuple(v / total for v in values)


def tv(p, q):
    """Exact TV for two explicitly normalized finite distributions."""
    p, q = tuple(map(Fraction, p)), tuple(map(Fraction, q))
    if len(p) != len(q) or not p or any(v < 0 for v in p + q) or sum(p) != 1 or sum(q) != 1:
        raise ValueError('Same support and normalized probabilities required')
    return sum(abs(a - b) for a, b in zip(p, q)) / 2


def categorical_audit(probabilities, *, cdf_law, ideal_weights=None):
    """Decompose normalization / CDF / grid error using exact rationals.

    P is the exact ratio of supplied binary64 reference weights, NOT a bound
    on analytic libm/physical input errors. Q normalizes supplied choice inputs
    in exact arithmetic; F is the continuous-uniform rounded-CDF law; G is the
    uniform-53-bit-grid law. The supplied helper must be the pinned cdf_law.
    Acceptance here is a diagnostic precondition, not a replacement for choice.
    """
    if np.__version__ != '1.26.4':
        raise ValueError('This audit describes NumPy 1.26.4')
    values = np.asarray(tuple(probabilities), dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Finite nonnegative one-dimensional probabilities required')
    if abs(math.fsum(values) - 1.0) > math.sqrt(np.finfo(float).eps):
        raise ValueError('Input outside the float64 choice normalization tolerance')
    q = _probabilities(map(float, values))
    p = _probabilities(map(float, values if ideal_weights is None else ideal_weights))
    if len(p) != len(q):
        raise ValueError('Reference support differs')
    cdf = np.cumsum(values)
    cdf /= cdf[-1]
    endpoints = (Fraction(0), *map(lambda v: Fraction(float(v)), cdf))
    f = tuple(b - a for a, b in zip(endpoints, endpoints[1:]))
    saved = cdf_law(values)
    g = tuple(Fraction(float(v)) for v in saved['probabilities'])
    # All grid masses are integers / 2**53, exactly representable in binary64.
    distances = dict(normalization=tv(p, q), cdf_rounding=tv(q, f), grid=tv(f, g), total=tv(p, g))
    counts = tuple(v * GRID for v in g)
    if any(v.denominator != 1 for v in counts):
        raise ValueError('Helper does not describe the 53-bit grid')
    return dict(P=p, Q=q, F=f, G=g, cdf=tuple(endpoints[1:]), grid_counts=tuple(map(int, counts)),
                TV=distances, triangle_bound=min(Fraction(1), sum(distances[k] for k in ('normalization', 'cdf_rounding', 'grid'))),
                map=saved['map'], PRNG_independence_proved=False,
                physical_law_distance=None, qualified_for_compiled_price=False)


def k2_event_bound(order_p, order_q, component_tv, *, iid_components=False):
    """Order maximal coupling, then 1 or 3 IID conditional component draws.

    Returns None without the conditional-IID model. Metadata/actions must also
    agree; this function bounds event labels only, not altered circuit words.
    """
    if not iid_components or component_tv is None:
        return None
    p, q = tuple(map(Fraction, order_p)), tuple(map(Fraction, order_q))
    delta = Fraction(component_tv)
    if len(p) != 2 or not 0 <= delta <= 1:
        raise ValueError('K2 orders (0,2) and component TV in [0,1] required')
    return min(Fraction(1), tv(p, q) + sum(min(a, b) * (1 - (1 - delta)**n)
                                         for a, b, n in zip(p, q, (1, 3))))


def trajectory_bound(event_tv, positions, *, uniform_conditional_bound=False):
    """Union coupling bound; marginal categorical TV alone is insufficient.

    The caller must justify a bound uniform over matched histories, e.g. an
    independent-uniform-grid model including every restarted piece stream.
    Work is independent of the number of event positions; no event generation.
    """
    if type(positions) is not int or positions < 0:
        raise ValueError('Nonnegative integer event count required')
    if event_tv is None or not uniform_conditional_bound:
        return None
    delta = Fraction(event_tv)
    if not 0 <= delta <= 1:
        raise ValueError('TV in [0,1] required')
    return min(Fraction(1), positions * delta)


def expectation_shift_bound(distance, oscillation):
    """Requires a proved full-support range max(f)-min(f), not sample SD/mean."""
    if distance is None or oscillation is None:
        return None
    distance, oscillation = Fraction(distance), Fraction(oscillation)
    if not 0 <= distance <= 1 or oscillation < 0:
        raise ValueError('TV in [0,1] and nonnegative oscillation required')
    return distance * oscillation

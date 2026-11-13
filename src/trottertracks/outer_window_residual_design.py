"""Passive outer-window design algebra; no science data, compiler, or fitting.

F_n is the supplied mean C-S of n target-law outer blocks with the SAME target
wrapper/scalar caps for every n. A truncated combination is not a price
certificate. All callers in this change use artificial numbers only.
"""
from fractions import Fraction
import math


def _count(value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError('Integer count outside design domain')
    return value


def _rational(value):
    if isinstance(value, bool) or isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Finite real design value required')
    return Fraction(value)


def connected_coefficients(span):
    """Stationary mean increment, with G_empty=0 after subtracting F_0."""
    span = _count(span, 1)
    return {1: 1, 0: -1} if span == 1 else {span: 1, span-1: -2, span-2: 1}


def truncation_coefficients(q, horizon):
    """Sum of weighted connected increments through horizon, including F_0."""
    q, horizon = _count(q, 1), _count(horizon, 1)
    if horizon > q:
        raise ValueError('Window horizon cannot exceed target q')
    return {k: v for k, v in ((horizon, q-horizon+1), (horizon-1, -(q-horizon))) if v}


def supplied_combination(q, horizon, window_means):
    """Conditional linear combination only; missing means and tail stay unknown."""
    weights = truncation_coefficients(q, horizon)
    value = None if any(window_means.get(k) is None for k in weights) else sum(
        (v*_rational(window_means[k]) for k, v in weights.items()), Fraction())
    return dict(quantity='SUPPLIED_OUTER_WINDOW_C_MINUS_S_COMBINATION', coefficients=weights,
        value=value, exact_algebra_at_horizon_q=horizon == q,
        compiler_influence_tail_bound=None, all_law_mean_compiled_price=None,
        qualified_for_candidate=False)


def paired_mean_variance(q, horizon, *, variance_hi, variance_lo, covariance, n):
    """w^T Cov(F_h,F_(h-1)) w/n. No independence/default-zero covariance."""
    weights = truncation_coefficients(q, horizon)
    n = _count(n, 1)
    if variance_hi is None:
        return None
    hi = _rational(variance_hi)
    if hi < 0:
        raise ValueError('Negative variance')
    a, b = weights.get(horizon, 0), weights.get(horizon-1, 0)
    if b == 0:
        return a*a*hi/n
    if variance_lo is None or covariance is None:
        return None
    lo, cross = _rational(variance_lo), _rational(covariance)
    if lo < 0 or cross*cross > hi*lo:
        raise ValueError('Invalid supplied covariance matrix')
    return (a*a*hi + b*b*lo + 2*a*b*cross)/n


def conditional_geometric_tail(q, horizon, *, amplitude, decay):
    """Conditional on |Delta_k| <= A*rho^(k-h-1), for EVERY h<k<=q.

    This function sums supplied bounds; it does not infer the premise from
    measured increments or certify locality of Qiskit or an H4 circuit.
    """
    truncation_coefficients(q, horizon)
    if amplitude is None or decay is None:
        return None
    amplitude, decay = _rational(amplitude), _rational(decay)
    if amplitude < 0 or not 0 <= decay < 1:
        raise ValueError('Require A>=0 and 0<=rho<1')
    remaining = q-horizon
    return amplitude * (remaining-(remaining+1)*decay+decay**(remaining+1))/(1-decay)**2

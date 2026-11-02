"""Independent saved-value bounds. Standard library only; no science imports.

The bounds concern one fixed finite ensemble and one Bernstein policy. They
neither construct an optimal proposal nor certify a physical shot lower bound.
"""
from fractions import Fraction as F
from math import factorial, isqrt


def sqrt_bounds(value, bits=256):
    value = F(value)
    if value < 0 or bits < 1:
        raise ValueError('nonnegative value and positive precision required')
    scale = 1 << bits
    k = isqrt(value.numerator * scale * scale // value.denominator)
    lo = F(k, scale)
    return lo, lo if lo * lo == value else F(k + 1, scale)


def exp_upper(z, terms=96):
    """Rational Taylor sum plus geometric tail, for a positive argument."""
    z = F(z)
    if z < 0 or terms + 2 <= z:
        raise ValueError('invalid geometric remainder')
    partial = sum(z ** n / factorial(n) for n in range(terms + 1))
    first = z ** (terms + 1) / factorial(terms + 1)
    return partial + first / (1 - z / (terms + 2))


def log_bounds(value, terms=96):
    """Range reduction and the positive atanh series, independently coded."""
    value = F(value)
    if value < 1:
        raise ValueError('only log(value>=1) is used')
    shift = 0
    while value >= 2:
        value /= 2
        shift += 1

    def series(q):
        z = (q - 1) / (q + 1)
        lo = 2 * sum(z ** (2*k+1) / (2*k+1) for k in range(terms))
        hi = lo + 2*z ** (2*terms+1) / ((2*terms+1)*(1-z*z))
        return lo, hi

    lo2, hi2 = series(F(2))
    lo, hi = series(value)
    return shift*lo2 + lo, shift*hi2 + hi


def affine_policy_lower(bindings, remaining, alpha_axis):
    """For every full-support q and h>=0: G(q;h)>=intercept+slope*h.

    m2*E(T+h)>=(sum alpha*sqrt(T+h))^2. For every pair Ti,Tj,
    sqrt((Ti+h)(Tj+h))>=sqrt(Ti*Tj)+h. Thus the square is at least
    (sum alpha*sqrt(T))^2+h*(sum alpha)^2. Zero-T events stay zero.
    Dropping the Bernstein range term and ceiling only lowers the policy cost.
    """
    s = F(remaining)
    alpha_axis = F(alpha_axis)
    if s <= 0 or not 0 < alpha_axis < 1 or not bindings:
        raise ValueError('positive margin, failure probability and support required')
    norm = F(0)
    weighted_root_lo = F(0)
    zero = 0
    for b in bindings:
        a, price = F(b['event']['coefficient']), b['cost']['T']
        if a <= 0 or type(price) is not int or price < 0:
            raise ValueError('positive coefficient and nonnegative integer price required')
        norm += a
        weighted_root_lo += a * sqrt_bounds(price)[0]
        zero += price == 0
    loglo, loghi = log_bounds(2 / alpha_axis)
    factor = 4 * loglo / s**2
    return {'intercept_lower': factor * weighted_root_lo**2,
            'prep_slope_lower': factor * norm**2,
            'coefficient_norm_rational': norm,
            'weighted_root_lower': weighted_root_lo,
            'log_lower': loglo, 'log_upper': loghi,
            'zero_T_events_retained': zero,
            'all_full_support_proposals': True,
            'all_common_nonnegative_prep_T': True,
            'attained_optimum_or_executable_law_claim': False}


def cts_coarse_check(cts, closed):
    """Independently verify the review's coarse affine inequality using raw data."""
    rotations = [b for b in cts['events'] if b['event']['rotation_sign']]
    real = [b for b in cts['events'] if not b['event']['rotation_sign']]
    if len(rotations) != 14 or len(real) != 10:
        raise ArithmeticError('registered literal CTS event set changed')
    if any(b['cost']['T'] != 0 for b in real):
        raise ArithmeticError('real correction zero-T structure changed')
    if any(b['cost']['T'] != 136 for b in rotations):
        raise ArithmeticError('rotation price changed')
    R = sum(F(b['event']['coefficient']) for b in rotations)
    s = F(cts['budget']['remaining'])
    alpha = F(cts['budget']['alpha_axis'])
    ell = F(34, 5)
    if not (R > F(7, 5) and s < F(1, 200)
            and exp_upper(ell) < 2/alpha):
        raise ArithmeticError('coarse analytic inequalities failed')
    slope = 4 * ell * F(7, 5)**2 * 200**2
    intercept = slope * 136
    actual_T = F(closed['two_axis_expected_native_cost']['T'])
    actual_K = F(closed['T_prep_readout_affine_coefficient'])
    if not (actual_T < 255860637 < intercept and actual_K == 1834258 < slope):
        raise ArithmeticError('closed-P5 saved affine upper failed')
    return {'rotation_events': len(rotations), 'zero_T_real_events': len(real),
            'rotation_coefficient_mass_rational': R,
            'rotation_T': 136, 'coarse_log_lower': ell,
            'coarse_intercept_strict_lower': intercept,
            'coarse_prep_slope_strict_lower': slope,
            'closed_P5_T_saved': actual_T, 'closed_P5_K_saved': actual_K,
            'all_common_h_ge_0_separated': True,
            'G9_original_classification_unchanged': True}


def serial(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value

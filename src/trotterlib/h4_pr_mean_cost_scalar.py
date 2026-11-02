"""Pure scalar arithmetic; no H4 input, evaluation or I/O on import.

H4 use requires the separate execution permission specified in the task.
The caller binds the frozen inputs. Compiler premises are never promoted here.
"""
from decimal import Context, Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F

from .h4_conditional_cng import trans_bounds

AXES = ('cosine', 'sine')
PRECISION = 96


def _positive_integer(value, name, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError('INVALID_INTEGER:' + name)
    return value


def _exact(value):
    if type(value) not in (int, str, F, Decimal):
        raise TypeError('EXACT_RATIONAL_REQUIRED')
    return F(value)


def _sqrt_endpoint(value, rounding):
    """One positive endpoint, one sqrt; exact zero needs no approximation."""
    if value == 0:
        return F(0)
    with localcontext(Context(prec=PRECISION)) as ctx:
        ctx.rounding = rounding
        operand = Decimal(value.numerator) / Decimal(value.denominator)
        if not operand.is_finite():
            raise ArithmeticError('NONFINITE_SQRT_OPERAND')
        root = operand.sqrt()
        if not root.is_finite():
            raise ArithmeticError('NONFINITE_SQRT_RESULT')
        endpoint = (max(Decimal(0), root.next_minus())
                    if rounding == ROUND_FLOOR else root.next_plus())
        return F(endpoint)


def sqrt_interval_bounds(lower, upper):
    """Outward sqrt at fixed precision, verified by exact rational squares."""
    lower, upper = _exact(lower), _exact(upper)
    if lower < 0 or upper < lower:
        raise ValueError('INVALID_NONNEGATIVE_INTERVAL')
    lo = _sqrt_endpoint(lower, ROUND_FLOOR)
    hi = _sqrt_endpoint(upper, ROUND_CEILING)
    if not (0 <= lo <= hi and lo * lo <= lower and hi * hi >= upper):
        raise ArithmeticError('SQRT_ENCLOSURE_NOT_VERIFIED')
    return lo, hi


def pauli_uniform_upper(*, n, L, q, r, K, axis):
    for name, value in (('n', n), ('q', q), ('r', r)):
        _positive_integer(value, name)
    _positive_integer(L, 'L', allow_zero=True)
    _positive_integer(K, 'K', allow_zero=True)
    if axis not in AXES or K % 2:
        raise ValueError('INVALID_AXIS_OR_TAYLOR_ORDER')
    return 12*q*L + q*r*((10*n+2)+7*n*K) + 7 + int(axis == 'sine')


def classify_separation(upper, lower_lo, lower_hi):
    lower_lo, lower_hi = _exact(lower_lo), _exact(lower_hi)
    if lower_lo < 0 or lower_hi < lower_lo:
        raise ValueError('INVALID_PRICE_LOWER_INTERVAL')
    if upper < lower_lo:
        return 'SUFFICIENT_INEQUALITY_SEPARATED'
    if upper >= lower_hi:
        return 'SUFFICIENT_BOUND_NOT_SEPARATED'
    return 'NUMERICALLY_INCONCLUSIVE'


def _interval(lo, hi):
    return dict(lower_exact=str(lo), upper_exact=str(hi))


def evaluate_scalar_batch(inputs):
    """One explicit batch. Tests supply artificial inputs; no saved-file reads.

    Both axes share exactly one ln-helper call and one two-endpoint sqrt.
    There is no precision escalation, sampling, compiler or shot accounting.
    """
    dimensions = {key: inputs[key] for key in ('n', 'L', 'q', 'r', 'K')}
    sample_count = _positive_integer(inputs['n_samples'], 'n_samples')
    minima = inputs['c_min']
    if set(minima) != set(AXES) or set(inputs['delta_axis']) != set(AXES):
        raise ValueError('BOTH_AXES_REQUIRED')
    for axis in AXES:
        _positive_integer(minima[axis], 'c_min', allow_zero=True)
        if type(inputs['delta_axis'][axis]) is not str:
            raise TypeError('DECIMAL_STRING_DELTA_REQUIRED')
    deltas = {a: F(inputs['delta_axis'][a]) for a in AXES}
    if (deltas['cosine'] != deltas['sine']
            or not 0 < deltas['cosine'] < 1
            or sum(deltas.values()) > _exact(inputs['delta_cost'])):
        raise ValueError('SHARED_LOG_OR_CONFIDENCE_ALLOCATION_INVALID')
    uppers = {a: pauli_uniform_upper(**dimensions, axis=a) for a in AXES}
    with localcontext(Context(prec=PRECISION)):
        log_lo, log_hi = trans_bounds('ln', F(2) / deltas['cosine'])
    log_lo, log_hi = _exact(log_lo), _exact(log_hi)
    if not 0 <= log_lo <= log_hi:
        raise ArithmeticError('INVALID_LOG_ENCLOSURE')
    denominator = 2 * sample_count
    e_lo, e_hi = sqrt_interval_bounds(log_lo/denominator, log_hi/denominator)
    axes = {}
    for a in AXES:
        lo = minima[a] * max(F(0), 1-e_hi)
        hi = minima[a] * max(F(0), 1-e_lo)
        axes[a] = dict(Pauli_uniform_upper=uppers[a], DF_mean_lower=_interval(lo, hi),
                       numerical_separation_status=classify_separation(uppers[a], lo, hi))
    statuses = [axes[a]['numerical_separation_status'] for a in AXES]
    both = all(s == 'SUFFICIENT_INEQUALITY_SEPARATED' for s in statuses)
    status = ('BOTH_AXES_SEPARATED' if both else 'NUMERICALLY_INCONCLUSIVE'
              if 'NUMERICALLY_INCONCLUSIVE' in statuses else 'NOT_SEPARATED_BY_THIS_BOUND')
    return dict(axes=axes, common_ln=_interval(log_lo, log_hi), common_e=_interval(e_lo, e_hi),
                decimal_precision=PRECISION, numerical_separation_status=status,
                count_premise_status='SOURCE_COUNT_BOUND_SUPPORTED_ON_NORMAL_RETURN; TOTALITY_AND_WHEEL_CORRESPONDENCE_UNPROVED',
                statistical_model_status='EXPLICIT_IID_ASSUMPTION; UNION_BOUND; AXES_NOT_ASSUMED_INDEPENDENT',
                population_claim_scope=('CONDITIONAL_NAMED_MEAN_PRICE_ORDER_ONLY' if both
                                        else 'NO_POPULATION_ORDER_FROM_THIS_SUFFICIENT_TEST'),
                unconditional_full_law_guarantee=False, formal_certificate_accepted=False,
                successful_batch_operations=dict(batches=1, upper_evaluations=2, ln_helper_calls=1,
                    sqrt_endpoint_budget=2, lower_reductions=2, separation_tests=2),
                mandatory_STOP=True)

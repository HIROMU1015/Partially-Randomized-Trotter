"""Approved B: exact arithmetic on forty saved scalar rows, never N/G.

All input floats are interpreted as their exact binary64 values. Fractions
give the algebraic budget; a separate Decimal path checks the arithmetic.
Nothing here evaluates a Hamiltonian, state, signal or certificate.
"""
from decimal import Decimal, localcontext, ROUND_FLOOR
from fractions import Fraction
import math

CASES = ('pauli_jw_PF','pauli_jw_PR','df_native_PF','df_native_PR')
EPSILONS = (.05,.01,.005,.001,.0005)
AXES = ('cosine','sine')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact(value):
    require(type(value) in (int,float) and math.isfinite(value), 'FINITE_SCALAR_REQUIRED')
    return Fraction(value)


def decimal_text(value, *, floor=False):
    if value is None:
        return None
    with localcontext() as ctx:
        ctx.prec = 80
        if floor:
            ctx.rounding = ROUND_FLOOR
        return str(Decimal(value.numerator)/Decimal(value.denominator))


def number(value):
    return None if value is None else dict(exact=str(value),decimal=decimal_text(value))


def independent_decimal_check(epsilon, bias, rho, expected):
    # Separate conversion and expression: solve u/rho + u = epsilon - |b|.
    # This path does not obtain its result from the Fraction budget expression.
    with localcontext() as ctx:
        ctx.prec = 100
        e = Decimal.from_float(float(epsilon))
        b = abs(Decimal.from_float(float(bias)))
        r = Decimal.from_float(float(rho))
        got = (e-b)/(Decimal(1)/r+Decimal(1))
        want = Decimal(expected.numerator)/Decimal(expected.denominator)
        error = abs(got-want)
        require(error <= Decimal('1e-90')*max(Decimal(1),abs(want)), 'INDEPENDENT_DECIMAL_MISMATCH')
        return dict(status='PASS',precision_digits=100,absolute_difference=str(error),
                    method='Decimal.from_float, solve u/rho+u=epsilon-abs(bias)')


def diagnose_axis(row, *, rho=.01):
    require(row['case'] in CASES and row['epsilon_sig'] in EPSILONS and row['axis'] in AXES, 'AXIS_IDENTITY')
    require(row['saved_status']=='UNDETERMINED' and row['saved_shots'] is None and
            row['saved_N_total'] is None and row['saved_G'] is None, 'ORIGINAL_STATE_MUST_REMAIN_UNDETERMINED_NULL')
    e,b,u,r = exact(row['epsilon_axis']),abs(exact(row['signed_bias'])),exact(row['saved_u']),exact(rho)
    require(e>0 and u>=0 and r>0, 'NONNEGATIVE_PRECISION_BUDGET')
    require(b==exact(row['absolute_saved_bias']), 'SIGNED_ABSOLUTE_BIAS_MISMATCH')
    slack=e-b
    h=slack-u
    # Preserve both the stored binary64 subtraction and exact-input arithmetic.
    old_h = row['saved_headroom']
    require(old_h == float(row['epsilon_axis'])-abs(float(row['signed_bias']))-float(row['saved_u']), 'SAVED_BINARY_HEADROOM_MISMATCH')
    group='R' if old_h>0 else 'H'
    require(row['group']==group, 'SAVED_GROUP_MISMATCH')
    float_ratio=row['saved_u']/old_h if old_h>0 else None
    require(group!='R' or float_ratio>rho, 'R_GROUP_MUST_EXCEED_SOURCE_RATIO_GUARD')
    budget=r*slack/(1+r) if slack>0 else None
    decimal_guard=slack/Fraction(101) if slack>0 else None
    conservative=min(budget,decimal_guard) if budget is not None else None
    reduction=max(Fraction(0),u-conservative) if conservative is not None else None
    reduction_rate=reduction/u if reduction is not None and u>0 else (Fraction(0) if reduction is not None else None)
    factor=u/conservative if conservative is not None and conservative>0 else None
    ratio=u/h if h>0 else None
    independent=independent_decimal_check(row['epsilon_axis'],row['signed_bias'],rho,budget) if budget is not None else dict(status='NOT_APPLICABLE_BIAS_EXHAUSTS_BUDGET')
    if budget is not None:
        require(budget/(slack-budget)==r, 'CLOSED_GUARD_BOUNDARY_IDENTITY')
        lower=Fraction(Decimal(decimal_text(conservative,floor=True)))
        require(lower<=conservative and lower>=0, 'DISPLAY_MUST_ROUND_INWARD')
    else:
        lower=None
    return dict(**row,epsilon_axis_exact=number(e),bias_magnitude_exact=number(b),
        exact_headroom_from_binary_inputs=number(h),stored_headroom_rounding_difference=number(exact(old_h)-h),
        current_guard_ratio_exact=number(ratio),source_binary64_guard_ratio=float_ratio,
        current_guard_diagnostic='RATIO_EXCEEDS_LIMIT' if group=='R' else 'NONPOSITIVE_SAVED_HEADROOM',
        headroom_positive_requires=dict(strict_u_less_than=number(slack),possible_for_nonnegative_u=slack>0),
        u_max_source_binary_rho=number(budget),u_max_ideal_decimal_rho=number(decimal_guard),
        u_max_conservative=number(conservative),u_budget_inward_display=decimal_text(lower),
        required_u_reduction=number(reduction),required_u_reduction_fraction=number(reduction_rate),
        required_u_reduction_percent=number(100*reduction_rate if reduction_rate is not None else None),
        required_u_improvement_factor=number(factor),fixed_point_bias_makes_guard_impossible=slack<=0,
        independent_arithmetic=independent,bound_is_budget_not_certificate=True,
        saved_bias_uncertainty_certified=False,official_status='UNDETERMINED',official_N=None,official_G=None,
        boundary_policy='Closed exact algebra; do not accept rounded equality as a new certificate or official eligibility.',
        C_design_target_only=number(conservative/2 if conservative is not None else None))


def validate_registry(rows):
    expected={(c,e,a) for c in CASES for e in EPSILONS for a in AXES}
    actual=[(x['case'],x['epsilon_sig'],x['axis']) for x in rows]
    require(len(actual)==40 and len(set(actual))==40 and set(actual)==expected, 'EXACT_FORTY_AXIS_REGISTRY_REQUIRED')
    require(sum(x['group']=='R' for x in rows)==22 and sum(x['group']=='H' for x in rows)==18, 'FIXED_R22_H18_COHORT')


def summarize(results):
    output={}
    for group in ('R','H'):
        rows=[x for x in results if x['group']==group]
        entry=dict(axes=len(rows),fixed_bias_impossible=sum(x['fixed_point_bias_makes_guard_impossible'] for x in rows))
        for field in ['u_max_conservative','required_u_reduction','required_u_reduction_percent','required_u_improvement_factor']:
            values=[(Fraction(x[field]['exact']),x['case'],x['epsilon_sig'],x['axis']) for x in rows if x[field] is not None]
            lo=min(values);hi=max(values)
            entry[field]=dict(minimum=number(lo[0]),minimum_axis=list(lo[1:]),maximum=number(hi[0]),maximum_axis=list(hi[1:]))
        output[group]=entry
    return dict(groups=output,axes=40,official_rows_unchanged=20,official_status='UNDETERMINED',
        official_N_G_remain_null=True,necessary_budget_is_not_achievability_or_certificate=True,
        original_native_completion_verified=False)

"""Conditional shot adapter, dormant and unqualified by default.

No mathematical or runtime qualification is issued here. Reuses the existing
interval endpoint/ceil primitives, without the literal-price score policy.
"""
from fractions import Fraction


def require_adopted_statistics(statistics):
    if statistics.get('status')!='USER_ADOPTED':
        raise PermissionError('Statistical numerical policy is not adopted')
    eta=Fraction(statistics['eta_stat']); alpha=Fraction(statistics['alpha'])
    allowance=Fraction(statistics['finite_bit_allowance'])
    if not 0<=allowance<eta or not 0<alpha<1:
        raise ValueError('Positive remaining statistical budget required')
    scope=statistics.get('alpha_scope')
    if scope not in ('PER_ARM_COMPLEX_SIGNAL','SIMULTANEOUS_FOUR_ARMS'):
        raise PermissionError('Failure-probability scope must be adopted')
    required=('fresh_iid_per_hadamard_shot','independent_bounded_outcomes',
              'range_bound_justified','finite_bit_allowance_justified')
    qualification=statistics.get('sampling_qualification',{})
    if qualification.get('status')!='EXTERNALLY_APPROVED_CONDITIONS' or not all(
            qualification.get(k) is True for k in required) or not qualification.get('evidence'):
        raise PermissionError('IID/range/finite-bit qualifications are unresolved')
    return eta-allowance,alpha/(4 if scope=='SIMULTANEOUS_FOUR_ARMS' else 1)


def qualified_shots(range_bound,statistics):
    """Only a qualified outward interval and a resolved integer ceil can give N."""
    eta,alpha=require_adopted_statistics(statistics)
    provider=statistics.get('interval_qualification',{})
    if provider.get('status')!='QUALIFIED_FOR_REGISTERED_SCALAR_FUNCTIONS':
        raise PermissionError('SHOT_COUNT_UNRESOLVED: interval provider unqualified')
    # Import after all eligibility checks; never evaluates in preparation tests.
    from .intervals import scalar_context,rational_interval,endpoints
    from .gates import shot_count_from_enclosure
    bound=Fraction(range_bound)
    if bound<=0: raise ValueError('Positive corrected-outcome range required')
    ctx=scalar_context(); b=rational_interval(ctx,bound)
    expr=4*b*b/rational_interval(ctx,eta)**2*ctx.ln(rational_interval(ctx,4/alpha))
    lower,upper=endpoints(expr)
    n=shot_count_from_enclosure(lower,upper)
    return dict(N_X=n,N_Y=n,range_bound=str(bound),eta_sampling=str(eta),
        alpha_per_arm=str(alpha),N_expression_bounds=[str(lower),str(upper)],
        status='CONDITIONAL_FINITE_TARGET_POLICY',quantum_shots_drawn=0)

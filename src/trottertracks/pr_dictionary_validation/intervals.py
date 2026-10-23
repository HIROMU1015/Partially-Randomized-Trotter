"""Dormant scalar interval score provider, pinned to mpmath 1.3.0 internals.

Uses existing outward-rounded library operations; this is not a new proof.
Runtime enclosure/ceil qualification remains a B prerequisite, not an I result.
"""
from __future__ import annotations
from fractions import Fraction
import math
from .gates import shot_count_from_enclosure
from .pricing import ScoreInterval,UnpricedError,expected_literal_price


def binary_mpf_fraction(record):
    sign,man,exponent,bits=record
    if bits<0: raise UnpricedError('Nonfinite interval endpoint')
    value=Fraction(int(man))*(Fraction(2)**int(exponent))
    return -value if sign else value


def endpoints(interval):
    low,high=interval._mpi_
    return binary_mpf_fraction(low),binary_mpf_fraction(high)


def scalar_context():
    from mpmath.ctx_iv import MPIntervalContext
    ctx=MPIntervalContext(); ctx.dps=90
    return ctx


def rational_interval(ctx,value):
    x=Fraction(value)
    return ctx.mpf(x.numerator)/ctx.mpf(x.denominator)


def score_enclosure(atoms,prices,context,*,fixed_x,fixed_y,provider_identity,
                    qualification):
    if qualification!='QUALIFIED_FOR_REGISTERED_SCALAR_FUNCTIONS':
        raise UnpricedError('Scalar interval provider is not runtime-qualified')
    ctx=scalar_context()
    exact_lambda=sum((abs(a.coefficient) for a in atoms),Fraction(0))
    if exact_lambda<=0: raise UnpricedError('No randomized lambda')
    tau=rational_interval(ctx,exact_lambda*context.micro_time)
    weights={n:abs(tau)**n/math.factorial(n)*ctx.sqrt(1+(tau/(n+1))**2)
             for n in range(0,context.cutoff+1,2)}
    b=sum(weights.values(),ctx.mpf(0)); total_b=b**(context.q*context.r)
    probs={a.atom_id:rational_interval(ctx,abs(a.coefficient)/exact_lambda) for a in atoms}
    orders={n:w/b for n,w in weights.items()}
    # Common P0 complex-signal Hoeffding policy, eta=1/40, alpha=1/20.
    n_expr=4*total_b**2/rational_interval(ctx,Fraction(1,40))**2*ctx.ln(80)
    shots=shot_count_from_enclosure(*endpoints(n_expr))
    phis=[]
    for fixed in (fixed_x,fixed_y):
        mass,phi=expected_literal_price(prices,probs,orders,q=context.q,r=context.r,
                                       fixed_price=fixed,scalar_reserves=1)
        low,high=endpoints(mass)
        if not low<=1<=high: raise UnpricedError('Interval probability mass misses1')
        phis.append(phi)
    score=shots*(phis[0]+phis[1]); lower,upper=endpoints(score)
    result=ScoreInterval(lower,upper,provider_identity,True); result.validate()
    return result,dict(shots_X=shots,shots_Y=shots,Btot_bounds=tuple(map(str,endpoints(total_b))),
        Phi_X_bounds=tuple(map(str,endpoints(phis[0]))),Phi_Y_bounds=tuple(map(str,endpoints(phis[1]))),
        score_bounds=(str(lower),str(upper)))

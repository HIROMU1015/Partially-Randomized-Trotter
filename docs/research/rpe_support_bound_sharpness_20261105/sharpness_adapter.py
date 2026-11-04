"""Generic support-bound improvements. No real input loader or import-time evaluation."""
from __future__ import annotations

from fractions import Fraction as Q
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'docs/research/rpe_categorical_circuit_connection_20261104'),
               str(ROOT/'docs/research/rpe_controlled_connection_runner_20261104')]
import connection_adapter as old
from integration_audit import require


def nonnegative(value):
    require(value is not None,'missing bound/probability')
    q = old.exact(value)
    require(q>=0,'negative bound/probability')
    return q


def unit_interval(value):
    q = nonnegative(value)
    require(q<=1,'unit interval required')
    return q


def projector_bound(full, selected, support):
    """Parity of a rank-k subspace, k=1/2: min(2,sqrt(2k)||Pi-Pi'||F).

    Both MatrixBalls must enclose QUALIFIED unitary one-particle actions.
    Thin centers and both input radii are retained. This does not re-synthesize
    a basis or use an unverified float zero residual as an identity proof.
    """
    from flint import acb_mat
    b,_ = old.modules()
    n,k = full.center.nrows(),len(support)
    require(full.center.ncols()==n and selected.center.nrows()==n
            and selected.center.ncols()==n and k in (1,2)
            and len(set(support))==k
            and all(type(i) is int and 0<=i<n for i in support),'projector support/dimension')
    require(full.norm==1 and selected.norm==1,'known unitary target norms required')
    require(full.radius.is_finite() and selected.radius.is_finite()
            and full.radius>=0 and selected.radius>=0,'finite nonnegative radii required')
    if k==n:
        return Q(0)  # Entire mode-space parity is basis-independent.
    a = acb_mat([[full.center[i,j] for j in support] for i in range(n)])
    c = acb_mat([[selected.center[i,j] for j in support] for i in range(n)])
    center_difference = a*b.dagger(a)-c*b.dagger(c)
    radius = b.real(k).sqrt()*(full.radius*(2+full.radius)+selected.radius*(2+selected.radius))
    frobenius = b.upper(b.frobenius(center_difference)+radius)
    return min(Q(2),b.dyadic(b.upper(b.real(2*k).sqrt()*frobenius)))


def grouped_moments(probabilities, errors, colors):
    """d_j=sum p_l d_l(1-h_color)^j, j=0,1,2. Exact normalized law."""
    require(len(probabilities)==len(errors)==len(colors)>0,'component coverage')
    p,d = list(map(unit_interval,probabilities)),list(map(nonnegative,errors))
    require(sum(p,Q(0))==1,'normalized exact component law required')
    require(all(x<=2 for x in d),'component unitary-gap upper must be <=2')
    h = {}
    for probability,color in zip(p,colors,strict=True):
        h[color] = h.get(color,Q(0))+probability
    moments = [sum((probability*error*(1-h[color])**j
                    for probability,error,color in zip(p,d,colors,strict=True)),Q(0))
               for j in range(3)]
    return tuple(moments)


def singleton_mean_tail(r, order_masses, sine_uppers, moments):
    """Expected additive bound for the actual run_le_1 policy in ONE tail.

    IID components/orders, no identity applications, K2; circuit mean is NOT
    factored. r=1 endpoints and the single-application n=0 event are explicit.
    """
    old.positive_integer(r)
    require(len(order_masses)==len(sine_uppers)==2 and len(moments)==3,'K2 evidence required')
    w0,w2 = map(unit_interval,order_masses)
    s0,s2 = map(unit_interval,sine_uppers)
    d0,d1,d2 = map(nonnegative,moments)
    require(w0+w2==1 and d2<=d1<=d0,'normalized law/monotone moments required')
    if r==1:
        value = w0*s0*d0+w2*((1+s2)*d1+d2)
    else:
        ends = 2*w0*s0+w2*(1+s2)
        inside = (r-2)*w0*s0+w2*((2*r-1)+(r-1)*s2)
        value = ends*d1+inside*d2
    return min(Q(2),value)


def unrestricted_mean_tail(r, order_masses, sine_uppers, mean_error):
    """Safe fallback: discard the word-dependent support indicator."""
    old.positive_integer(r)
    require(len(order_masses)==len(sine_uppers)==2,'K2 evidence required')
    w0,w2 = map(unit_interval,order_masses)
    s0,s2 = map(unit_interval,sine_uppers)
    require(w0+w2==1,'normalized exact order law required')
    return min(Q(2),r*(w0*s0+w2*(2+s2))*nonnegative(mean_error))


def singleton_word_tail(r, basis_error_uppers, sine_uppers):
    """Max additive bound over all K2 words; DP has <=3B end-of-event states.

    A pending contribution survives only if its source-basis run stays singleton.
    This changes the coefficient, not the generic linear q dependence.
    """
    old.positive_integer(r)
    require(bool(basis_error_uppers) and len(sine_uppers)==2,'basis/K2 coverage')
    require(None not in basis_error_uppers,'None is reserved for the initial DP state')
    bounds = {c:nonnegative(d) for c,d in basis_error_uppers.items()}
    require(all(x<=2 for x in bounds.values()),'component gap upper must be <=2')
    s0,s2 = map(unit_interval,sine_uppers)
    states = {(None,Q(0)):Q(0)}
    for _ in range(r):
        ends = {}
        for coefficients in ((s0,),(Q(1),Q(1),s2)):
            current = states
            for coefficient in coefficients:
                following = {}
                for (last,pending),value in current.items():
                    for color,d in bounds.items():
                        key = (color,Q(0) if color==last else coefficient*d)
                        candidate = value if color==last else value+pending
                        following[key] = max(following.get(key,Q(0)),candidate)
                current = following
            for key,value in current.items():
                ends[key] = max(ends.get(key,Q(0)),value)
        states = ends
    return min(Q(2),max(value+pending for (_,pending),value in states.items()))


def sine_upper(theta_hex):
    """96-bit caller context required by the future scientific plan."""
    b,_ = old.modules()
    value = b.upper(abs(b.real(theta_hex).sin()))
    return min(Q(1),b.dyadic(value))


def mean_extra(q, mean_tail, boundary, lowering, previous_uniform):
    """Whole-trajectory phases retained; existing I_cat already contains pi64."""
    old.positive_integer(q)
    if any(x is None for x in (mean_tail,boundary,lowering,previous_uniform)):
        return None
    tail,phase,gate,previous = map(nonnegative,(mean_tail,boundary,lowering,previous_uniform))
    return min(Q(2),previous,phase+q*tail+gate)

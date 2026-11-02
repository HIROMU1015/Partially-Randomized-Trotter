"""Finite-bit executed law; cost, range and moment use the same exact masses."""
from fractions import Fraction as F
from .intervals import iv, upper, bounds, midpoint


def largest_remainder(probabilities, bits=48):
    p = {k: F(v) for k, v in probabilities.items()}
    if not p or any(v < 0 for v in p.values()) or sum(p.values()) != 1:
        raise ArithmeticError('Invalid complete computational law')
    scale = 1 << bits
    integers = {k: (v*scale).numerator // (v*scale).denominator for k, v in p.items()}
    order = sorted(p, key=lambda k: (-(p[k]*scale-integers[k]), k))
    for k in order[:scale-sum(integers.values())]:
        integers[k] += 1
    return {k: F(v, scale) for k, v in integers.items()}


def enclose_law(probabilities):
    """Result-prior amendment: exact rational midpoint law, certified distance.

    LRM acts on this rational law, not an uncertain ordering of ideal remainders.
    All labels retained; zero computational mass is charged to L1 distance.
    """
    q = {k: midpoint(v) for k, v in probabilities.items()}
    if any(v < 0 for v in q.values()) or sum(q.values()) <= 0:
        raise ArithmeticError('Invalid interval probability midpoint')
    total = sum(q.values())
    q = {k: v/total for k, v in q.items()}
    distance = sum((max(abs(q[k]-a), abs(q[k]-b))
                    for k, v in probabilities.items() for a, b in [bounds(v)]), F(0))
    return q, distance


def rounded_law(intervals, bits=48):
    q, radius = enclose_law(intervals)
    rounded = largest_remainder(q, bits)
    radius += sum(abs(rounded[k]-q[k]) for k in q)
    return rounded, radius


def corrector_events(eta, tuples, paulis=None, *, bits=48, proved_zero_tuples=()):
    """Coin exact; conditional masses rounded. Return complete single-slot law.

    tuples: exact c/A. paulis: tau -> exact computational beta table.
    A unit weight; B dyadic implemented weight enclosed from sqrt(1+(u ell)^2)
    is supplied later. Identity stays an event and PR base cost remains separate.
    """
    eta = F(eta)
    if not 0 <= eta <= 1:
        raise ArithmeticError('Invalid exact outer coin')
    q = largest_remainder(tuples, bits)
    qerr = sum(abs(q[k]-F(tuples[k])) for k in q)
    rows = [{'label': 'identity', 'p': 1-eta, 'tau': None, 'pauli': None}]
    error = eta*qerr
    for tau, qt in q.items():
        beta = None if paulis is None else paulis[tau]
        ell = sum(abs(F(v)) for v in beta.values()) if beta is not None else None
        if tau in proved_zero_tuples or (beta is not None and ell == 0):
            rows.append({'label': tau+':zero', 'p': eta*qt, 'tau': tau, 'pauli': None})
        elif beta is None:
            rows.append({'label': tau, 'p': eta*qt, 'tau': tau, 'pauli': 'whole'})
        else:
            p = {k: abs(F(v))/ell for k, v in beta.items()}
            pt = largest_remainder(p, bits)
            error += eta*qt*sum(abs(pt[k]-p[k]) for k in p)
            for label in sorted(pt):
                rows.append({'label': tau+':'+label, 'p': eta*qt*pt[label],
                             'tau': tau, 'pauli': label, 'beta': F(beta[label]), 'ell': ell})
    assert sum(r['p'] for r in rows) == 1
    return rows, error


def account(rows):
    """No independence factorization between weights and per-event costs."""
    if sum(F(r['p']) for r in rows) != 1:
        raise ArithmeticError('Incomplete executed law')
    expected = sum((F(r['p'])*F(r.get('cost', 0)) for r in rows), F(0))
    moment = sum((F(r['p'])*F(r['weight'])**2 for r in rows), F(0))
    width = max((abs(F(r['weight'])) for r in rows if F(r['p']) > 0), default=F(0))
    weighted_error = sum((F(r['p'])*abs(F(r['weight']))*F(r.get('error', 0)) for r in rows), F(0))
    return {'expected_cost': expected, 'second_moment': moment, 'range': width,
            'weighted_native_bias': 2*weighted_error}


def weight_point(u, ell):
    import mpmath as mp
    x = mp.iv.sqrt(1+(iv(u)*iv(ell))**2)
    # Fixed binary grid, no stochastic or precision retry.
    lo, hi = bounds(x)
    scale = 1 << 80
    q = ((lo+hi)*scale/2).numerator // ((lo+hi)*scale/2).denominator
    value = F(q, scale)
    return value, max(abs(value-lo), abs(value-hi))


def sufficient_shots(nu, width, statistical_width, alpha):
    import mpmath as mp
    from .intervals import ceil_upper
    s = F(statistical_width)
    if s <= 0 or F(nu) < 0 or F(width) < 0:
        raise ArithmeticError('Infeasible confidence budget')
    return ceil_upper(mp.iv.ln(2/iv(alpha))*(2*iv(nu)/iv(s)**2+4*iv(width)/(3*iv(s))))

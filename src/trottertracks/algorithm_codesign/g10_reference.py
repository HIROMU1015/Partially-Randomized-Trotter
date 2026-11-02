"""Reference-only finite Pauli collection and small-support event traversal."""
from fractions import Fraction as F
from itertools import product
from math import factorial
from .return_aggregation import Interval, root_interval, dyadic_distribution
from .g9_native import A, provider_polynomials, poly_product
from .g10_generator import ClosedP5Tail


def reference_events(g):
    from .g9_p5 import P5Closed
    from .g7_reference import reference_events as old_reference
    if isinstance(g, P5Closed):
        yield from g.reference_events()
    elif isinstance(g, ClosedP5Tail):
        for e in g.prefix.reference_events():
            word, child = e['word'], e['child']
            mode = 'root' if not word else 'two' if len(word) == 2 else 'four'
            index = next(i for i, z in enumerate(g.prefix.groups)
                         if z[0] == mode and (mode == 'root' or z[1] == word[0])
                         and (mode != 'two' or z[2] == word[1]))
            yield g.event(index, word, child)
        for j, ti in enumerate(g.tail_indices):
            for word in product(range(len(g.p)), repeat=g.tail.groups[ti].degree):
                for child in range(len(g.p)):
                    yield g.event(len(g.prefix.groups)+j, word, child)
    else:
        yield from old_reference(g)


def exact_target(p, x, m):
    if type(m) is not int or m < 1 or m % 2 != 1:
        raise ValueError('positive odd degree required')
    R = {}
    for pi, q in zip(p, provider_polynomials()):
        for k, v in q.items():
            R[k] = R.get(k, A()) + F(pi)*v
    power = {('III', 0): A(1)}
    target = power.copy()
    for n in range(1, m+1):
        power = poly_product(R, power)
        phase, factor = (-n) % 4, F(x)**n/factorial(n)
        for (axis, j), v in power.items():
            ph = (phase+j) % 4
            key = axis, ph % 2
            target[key] = target.get(key, A()) + (-1 if ph >= 2 else 1)*factor*v
    return {k: v for k, v in target.items() if v}


def cts_events(p, x, m, H=160, K=256, eta=F(1, 10**12), rho=F(1, 10**12)):
    """Literal CTS, including identity real correction, specialized to P_m."""
    target = exact_target(p, x, m)
    real, odd, error = [], [], F(0)
    for (axis, phase), v in sorted(target.items()):
        if phase == 0:
            v = v - A(axis == 'III')
            if not v:
                continue
        z = v.interval()
        if z.lo <= 0 <= z.hi:
            raise ArithmeticError('CTS sign unresolved')
        sign = 1 if z.lo > 0 else -1
        az = Interval(z.lo, z.hi) if sign > 0 else Interval(-z.hi, -z.lo)
        error += (az.hi-az.lo)/2
        (real if phase == 0 else odd).append((axis, sign, az.midpoint))
    if any(axis == 'III' for axis, _, _ in odd):
        raise ArithmeticError('unsupported scalar imaginary event')
    L = sum(v for _, _, v in odd)
    norm = root_interval(1+L*L, K)
    if norm.hi-norm.lo > 2*rho*norm.lo:
        raise ArithmeticError('CTS norm precision')
    error += rho*(1+L)
    if error > 8*rho:
        raise ArithmeticError('common coefficient mean bias exceeded')
    events = [{'pauli': axis, 'coefficient': c, 'phase_i_power': 0 if sign > 0 else 2,
               'ratio': F(0), 'rotation_sign': 0} for axis, sign, c in real]
    events += [{'pauli': axis, 'coefficient': norm.midpoint*c/L, 'phase_i_power': 0,
                'ratio': L, 'rotation_sign': -sign} for axis, sign, c in odd]
    B = sum(e['coefficient'] for e in events)
    law = dyadic_distribution(tuple(Interval(e['coefficient']/B, e['coefficient']/B)
                                   for e in events), H, eta)
    for e, q in zip(events, law):
        e.update(proposal=q, weight=e['coefficient']/q)
    return events, {'degree': m, 'literal_finite_Theorem1_specialization': True,
                    'first_operator_moment_not_channel': True,
                    'identity_even_correction_kept_separate': True,
                    'target_exact_Qsqrt2': {k[0]+':'+str(k[1]): v.json() for k, v in target.items()},
                    'rational_rotation_tangent': str(L),
                    'coefficient_mean_error_upper': str(error),
                    'normalizer_rational': str(B),
                    'Pauli_access': 'explicit cheap I1; no DF acquisition advantage claimed'}


def matrix_target(p, x, m):
    """Independent 8x8 oracle, used only in focused off-domain tests/the run."""
    import numpy as np
    from .g9_matrix import Q_matrices
    R = sum(float(pi)*q for pi, q in zip(p, Q_matrices()))
    total = np.eye(8, dtype=complex)
    term = total.copy()
    for n in range(1, m+1):
        term = term@(-1j*float(x)*R)/n
        total += term
    return total

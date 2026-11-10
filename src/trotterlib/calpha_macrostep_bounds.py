"""Fixed-law C-alpha bound inputs; no chemistry or signals are run on import.

The multi-Strang and defect formulas implement the supplied 2026-11-10
limited review, equations (3.3)-(3.5). Floating norms remain diagnostics.
Exact rational inertia certifies only the explicitly supplied real matrix.
"""
from fractions import Fraction
import math


def macrostep_diagnostics(head_outer_to_inner, rtheta, gamma, delta):
    """Wrap the central dissipative exponential, from inner head to outer.

    The input head order is the existing circuit append order (one-body,
    generation fragments 0,1,2). No old head/split bound is added here.
    """
    import numpy as np
    from scipy.linalg import expm
    heads = [np.asarray(x, dtype=complex) for x in head_outer_to_inner]
    rtheta, gamma = np.asarray(rtheta), np.asarray(gamma)
    if not heads or not math.isfinite(delta) or delta <= 0:
        raise ValueError('POSITIVE_TIME_AND_HEAD_REQUIRED')
    shape = rtheta.shape
    if len(shape) != 2 or shape[0] != shape[1] or gamma.shape != shape:
        raise ValueError('COMMON_SQUARE_SHAPE_REQUIRED')
    if any(x.shape != shape for x in heads):
        raise ValueError('COMMON_SQUARE_SHAPE_REQUIRED')
    if any(not np.isfinite(x).all() for x in [*heads, rtheta, gamma]):
        raise ValueError('FINITE_MATRICES_REQUIRED')
    norm = lambda x: float(np.linalg.norm(x, 2))
    comm = lambda x, y: x @ y - y @ x
    generator = -1j * rtheta - gamma
    step = expm(delta * generator)
    terms = []
    for index in reversed(range(len(heads))):
        a = -1j * heads[index]
        c = comm(a, generator)
        aa, gg = norm(comm(a, c)), norm(comm(generator, c))
        terms.append(dict(outer_to_inner_index=index,
                          double_A_norm=aa, double_G_norm=gg,
                          contribution=aa / 24 + gg / 12))
        half = expm(delta * a / 2)
        step = half @ step @ half
        generator = generator + a
    defect = norm(step - expm(delta * generator))
    constant = math.fsum(x['contribution'] for x in terms)
    return dict(C_multi=constant, eta_macro=defect,
                local_commutator_bound=delta**3 * constant,
                stage_terms_central_to_outer=terms,
                gamma_min_eigenvalue=float(np.linalg.eigvalsh(gamma)[0]),
                qualification='binary64_diagnostic_NOT_CERTIFIED'), step, generator


def _rational(value):
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    z = complex(value)
    if z.imag != 0 or not math.isfinite(z.real):
        raise ValueError('FINITE_REAL_EXACT_INPUT_REQUIRED')
    return Fraction.from_float(z.real)


def rational_inertia(matrix, shift):
    """Exact real LDL congruence, refusing unresolved zero pivots.

    Each float is interpreted as its exact dyadic value. No eigenvectors or
    ground state are computed. A zero leading pivot is not classified by a
    floating tolerance and does not produce a certificate.
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError('SQUARE_MATRIX_REQUIRED')
    a = [[_rational(x) for x in row] for row in matrix]
    if any(a[i][j] != a[j][i] for i in range(n) for j in range(i)):
        raise ValueError('EXACT_REAL_SYMMETRY_REQUIRED')
    shift = _rational(shift)
    for i in range(n):
        a[i][i] -= shift
    negative = positive = 0
    for k in range(n):
        pivot = a[k][k]
        if pivot == 0:
            raise ValueError('ZERO_EXACT_PIVOT_INERTIA_UNDETERMINED')
        negative += pivot < 0
        positive += pivot > 0
        for i in range(k + 1, n):
            multiplier = a[i][k] / pivot
            for j in range(i, n):
                a[j][i] -= multiplier * a[j][k]
                a[i][j] = a[j][i]
    return dict(negative=negative, positive=positive, zero=0)


def certify_gap_from_brackets(matrix, ground_bracket, excited_bracket):
    """Certify the lowest two eigenvalue brackets by four exact counts."""
    bounds = [_rational(x) for x in (*ground_bracket, *excited_bracket)]
    if not bounds[0] < bounds[1] < bounds[2] < bounds[3]:
        raise ValueError('DISJOINT_ORDERED_BRACKETS_REQUIRED')
    counts = [rational_inertia(matrix, x) for x in bounds]
    if [x['negative'] for x in counts] != [0, 1, 1, 2]:
        raise ValueError('LOWEST_TWO_EIGENVALUE_COUNTS_NOT_CERTIFIED')
    return dict(method='exact_rational_LDL_inertia',
                matrix_semantics='exact dyadic entries of supplied real symmetric matrix',
                ground_interval=[str(x) for x in bounds[:2]],
                first_excited_interval=[str(x) for x in bounds[2:]],
                gap_lower_exact=str(bounds[2] - bounds[1]),
                counts=counts, certified_for_supplied_matrix=True)

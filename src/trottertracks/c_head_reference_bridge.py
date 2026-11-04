"""Phase-sensitive head-target transport; dormant on real molecular data.

Reuses CAR Arithmetic/exponentials, the existing chronological scheduler,
Arb norms, and compact outward scalar endpoints. No head is replaced.
"""
from fractions import Fraction as F
from . import c_per_word_population_range as literal

PRECISION = 384
TERM = 'full_CAR_head_to_saved_sector_head'


def identity(full_heads, saved, indices, *, candidate, expected_candidate):
    import numpy as np
    if candidate != expected_candidate:
        raise ValueError('HEAD_CANDIDATE_IDENTITY')
    ix = np.asarray(indices)
    if (ix.ndim != 1 or ix.dtype.kind not in 'iu' or len(set(map(int, ix))) != len(ix)
            or not len(ix) or len(full_heads) != 3 or len(saved) != 3):
        raise ValueError('HEAD_BASIS_SCHEMA')
    d = len(full_heads[0]); complement = np.setdiff1d(np.arange(d), ix)
    if np.any(ix < 0) or np.any(ix >= d):
        raise ValueError('HEAD_BASIS_RANGE')
    for A, B in zip(full_heads, saved):
        if (A.shape != (d, d) or B.shape != (len(ix), len(ix))
                or A.dtype.kind not in 'fc' or B.dtype.kind not in 'fc'
                or not np.isfinite(A).all() or not np.isfinite(B).all()):
            raise ValueError('FINITE_HEAD_SHAPE')
        if np.any(A[np.ix_(ix, complement)] != 0) or np.any(A[np.ix_(complement, ix)] != 0):
            raise ValueError('EXACT_SECTOR_INVARIANCE_REQUIRED')
    return [A[np.ix_(ix, ix)] for A in full_heads]


def half_pair(task):
    """Both exact dyadic generators, exp(-i h G/2), and norm-growth bounds."""
    from flint import acb, ctx
    from .c_pr_aware_df_fixed_car import Arithmetic, Matrix, pack
    from . import c_pr_aware_df_zsxx_enclosure as z
    A, B, h = task
    with z._BALL_LOCK, ctx.workprec(PRECISION):
        mats = [z._matrix(G) for G in (A, B)]
        values = [Arithmetic.exponential(F(h)/2, Matrix(G)).ball for G in mats]
        norms = []
        for G in mats:
            anti = z._scale(G - G.conjugate().transpose(), z._real(F(1, 2)))
            k = F(z._norm(anti)['upper_exact'])
            norms.append(str(z._fraction((z._real(abs(F(h))/2)*z._real(k)).exp().upper())))
        return dict(full=pack(values[0]), sector=pack(values[1]), dimension=len(A),
                    half_norms=norms, generator_difference=z._norm(mats[0]-mats[1]),
                    half_difference=z._norm(values[0]-values[1]))


def products(pairs):
    from flint import ctx
    from .c_pr_aware_df_fixed_car import Matrix, Arithmetic, unpack
    from .c1_stage3_reference import _ordered_exponentials
    from . import c_pr_aware_df_zsxx_enclosure as z
    if len(pairs) != 3 or len({p['dimension'] for p in pairs}) != 1:
        raise ValueError('THREE_MATCHING_HEAD_PAIRS')
    with z._BALL_LOCK, ctx.workprec(PRECISION):
        values = {}
        for side in ('full', 'sector'):
            factors = [Matrix(unpack(p[side], p['dimension'])) for p in pairs]
            schedule = tuple((i, F(1, 20)) for i in range(3))
            cache = {(i, F(1, 20)): value for i, value in enumerate(factors)}
            values[side] = [_ordered_exponentials(factors, order, cache, arithmetic=Arithmetic()).ball
                            for order in (schedule, tuple(reversed(schedule)))]
        norm = [F(1), F(1)]
        for p in pairs:
            for i in range(2): norm[i] *= F(p['half_norms'][i])
        return dict(left=z._norm(values['full'][0]-values['sector'][0]),
                    right=z._norm(values['full'][1]-values['sector'][1]),
                    full_half_product_norm=str(norm[0]), sector_half_product_norm=str(norm[1]),
                    phase_fit=False, rounding_enclosed=True, scope='physical_sector_head_products')


def kernel_bridge(heads, population):
    """One shared tail map, so no atom/reference arithmetic is counted twice."""
    from flint import ctx
    from . import c_pr_aware_df_zsxx_enclosure as z
    ids = population['support']['positive']
    if (set(ids) != set(population['events']) or population['word_length'] != 32
            or population['accepted_P0_C_norm_exact'] != '1'
            or F(population['d_range_upper_exact']) != 1+F(population['D_norm_upper_exact'])):
        raise ValueError('BOTH_P_AND_Q_FULL_TAIL_SUPPORT_REQUIRED')
    event = max(F(population['events'][key]['event_norm_upper_exact']) for key in ids)
    with z._BALL_LOCK, ctx.workprec(PRECISION):
        tail = z._real(event)**32
        L = z._real(F(heads['full_half_product_norm']))
        B = z._real(F(heads['sector_half_product_norm']))
        err = tail*(z._real(F(heads['right']['upper_exact']))*L
                    + B*z._real(F(heads['left']['upper_exact'])))
        return dict(upper_exact=str(z._fraction(err.upper())),
                    sector_atom_norm_upper_exact=str(z._fraction((B*B*tail).upper())),
                    full_atom_norm_upper_exact=population['D_norm_upper_exact'],
                    target_id=population['target_id'], included_error_terms=[TERM],
                    scope='same p or q tail expectation;physical_sector36',
                    phase='same extracted identity phase,absolute value1;no fitted phase',
                    rounding_enclosed=True)


def round_bridge(kernel, m):
    from flint import ctx
    from . import c_pr_aware_df_zsxx_enclosure as z
    if type(m) is not int or not 0 <= m <= 12:
        raise ValueError('THIRTEEN_ROUNDS')
    n = 1 << m
    # Exact telescoping: ||A^n-B^n|| <= n e max(||A||,||B||)^(n-1).
    with z._BALL_LOCK, ctx.workprec(PRECISION):
        growth = max(F(kernel['full_atom_norm_upper_exact']), F(kernel['sector_atom_norm_upper_exact']))
        endpoint = literal.upper_record(n*z._real(F(kernel['upper_exact']))*z._real(growth)**(n-1))
    return dict(m=m, bound=endpoint['upper_exact'], enclosure=endpoint, rounding_enclosed=True,
                included_error_terms=[TERM], norm='raw_complex_signal', scope='sector36_reference',
                target_id=kernel['target_id'], phase_fit=False, actual_CA_achieved=None)


def connect_b2(rows, b2, *, target_id):
    """New head leg + the two ORIGINAL B2 legs, exactly once, no total reuse."""
    if ([x['m'] for x in rows] != list(range(13)) or len(b2['rounds']) != 13
            or any(x['target_id'] != target_id or x['included_error_terms'] != [TERM] for x in rows)):
        raise ValueError('HEAD_BRIDGE_IDENTITY_OR_DUPLICATE_TERM')
    result = []
    for head, old in zip(rows, b2['rounds']):
        if head['m'] != old['m']:
            raise ValueError('B2_ROUND_IDENTITY')
        B, U = F(old['bridge']['upper_exact']), F(old['reference_uncertainty']['upper_exact'])
        if F(old['total_upper_exact']) != B+U:
            raise ValueError('B2_ORIGINAL_PARTITION')
        total = None if head['bound'] is None else str(F(head['bound'])+B+U)
        result.append(dict(m=head['m'], head_bridge=head['bound'],
                           B2_atom_to_CAR=old['bridge']['upper_exact'],
                           B2_reference_uncertainty=old['reference_uncertainty']['upper_exact'],
                           total_upper_exact=total,
                           included_error_terms=[TERM, 'atom_to_mathematical_CAR',
                                                 'mathematical_CAR_to_binary64_reference']))
    return dict(status='BOUND_CONNECTION_ONLY', rows=result, target_id=target_id,
                Stage_C_ready=False, actual_CA_achieved=None, categorical_p_to_q_included=False,
                whole_law_synthesis_included=False, physical_input_error_included=False)

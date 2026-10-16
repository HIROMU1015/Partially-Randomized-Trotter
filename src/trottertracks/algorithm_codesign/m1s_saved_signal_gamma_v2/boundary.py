"""Pure scalar comparison of candidate sufficient designs, not H4 entrypoint."""
from .bounds import interval_text, rational_text


def compare(a, u, c, v, B, pi, trig=None):
    """All arguments are exact a,c=A_lower and u,v=b_upper, not true values.

    No unqualified arctan formula: check domain, dominance and strict tradeoff.
    A reverse tradeoff uses the same reviewed theorem with the labels swapped.
    """
    if not (0 < a <= 1 and 0 < c <= 1 and u >= 0 and v >= 0):
        raise ValueError('boundary inputs outside bounded-signal domain')
    L = max(u, v)
    result = {'status': None, 'formula_applicability': None,
              'common_domain': {'lower_open_rad': rational_text(L),
                                'upper_open': 'pi/2'},
              'boundary_interval_rad': None, 'boundary_location_status': 'NOT_APPLICABLE',
              'continuous_relation_before': None, 'continuous_relation_after': None,
              'strict_integer_saving': 'NOT_INFERRED_FROM_CONTINUOUS_ORDER',
              'integer_condition': 'ceil(K/M_sh(gamma)^2)<ceil(K/M_cmp(gamma)^2)',
              'scientific_GO': None}
    if L >= pi[1]/2:
        result.update(status='NO_COMMON_PHASE_DOMAIN', formula_applicability='DOMAIN_EMPTY')
        return result
    if L >= pi[0]/2:
        result.update(status='ARITHMETIC_UNRESOLVED', formula_applicability='DOMAIN_NOT_RESOLVED')
        return result
    if u == v:
        relation = 'EQUAL' if a == c else ('SHARED_HIGHER' if a > c else 'SHARED_LOWER')
        result.update(status='ALL_COMMON_GAMMA_' + relation,
                      formula_applicability='NOT_APPLICABLE_EQUAL_BIAS',
                      continuous_relation_before=relation)
        return result
    if a >= c and u <= v:
        result.update(status='ALL_COMMON_GAMMA_SHARED_HIGHER',
                      formula_applicability='NOT_APPLICABLE_SHARED_DOMINANCE',
                      continuous_relation_before='SHARED_HIGHER')
        return result
    if a <= c and u >= v:
        result.update(status='ALL_COMMON_GAMMA_SHARED_LOWER',
                      formula_applicability='NOT_APPLICABLE_COMPARATOR_DOMINANCE',
                      continuous_relation_before='SHARED_LOWER')
        return result
    swapped = u < v
    if swapped:
        a, u, c, v = c, v, a, u
    # Exactly the reviewed assumptions a>c>0 and 0<=v<u<pi/2.
    if not (a > c > 0 and 0 <= v < u and u < pi[0]/2):
        raise ValueError('strict tradeoff assumptions not established')
    if trig is None:
        su, cu = B.sin_positive(u), B.cos_positive(u)
        sv, cv = B.sin_positive(v), B.cos_positive(v)
    else:
        su, cu = trig[u]
        sv, cv = trig[v]
    X = B.pack((a*cu[0]-c*cv[1], a*cu[1]-c*cv[0]))
    Y = B.pack((a*su[0]-c*sv[1], a*su[1]-c*sv[0]))
    low = 'SHARED_HIGHER' if swapped else 'SHARED_LOWER'
    high = 'SHARED_LOWER' if swapped else 'SHARED_HIGHER'
    result.update(formula_applicability='REVIEW_TRADEOFF_SWAPPED' if swapped else 'REVIEW_TRADEOFF_DIRECT',
                  coefficient_intervals={'X': interval_text(X), 'Y': interval_text(Y)},
                  oriented_equation='X*sin(gamma)-Y*cos(gamma)',
                  equation_orientation='comparator-minus-shared' if swapped else 'shared-minus-comparator')
    if X[1] <= 0:
        result.update(status='ALL_COMMON_GAMMA_' + low,
                      continuous_relation_before=low,
                      boundary_location_status='NO_INTERIOR_ROOT_X_NONPOSITIVE')
    elif X[0] <= 0 or Y[0] <= 0:
        result.update(status='ARITHMETIC_UNRESOLVED',
                      boundary_location_status='COEFFICIENT_SIGN_UNRESOLVED')
    else:
        root = B.atan_interval((Y[0]/X[1], Y[1]/X[0]), pi)
        result.update(status='UNIQUE_TRADEOFF_BOUNDARY',
                      boundary_interval_rad=interval_text(root),
                      boundary_location_status=('ENCLOSED_INTERIOR' if root[0] > L and root[1] < pi[0]/2
                                                else 'ROOT_EXISTS_LOCATION_UNRESOLVED'),
                      continuous_relation_before=low, continuous_relation_after=high,
                      relation_rule='before root lower endpoint / after root upper endpoint; enclosure band unresolved',
                      unique_root_existence='THEOREM_UNDER_VERIFIED_TRADEOFF_ASSUMPTIONS')
    return result

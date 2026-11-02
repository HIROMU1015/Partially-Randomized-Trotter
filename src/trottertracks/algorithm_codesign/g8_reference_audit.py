"""Independent small-support DIAGNOSTIC only; never imported by production path."""
from fractions import Fraction as F
from .return_aggregation import Interval, root_interval, dyadic_distribution
from .g7_reference import reference_events
from .g8_bounds import log_upper, log_interval, ceil, p5_root_formula, accepted_cap


def saved_factorizations(old):
    output = []
    for full in (r for r in old['rows'] if r['arm'] == 'full_return'):
        ft = full['total_conditional_resource']
        for base in (r for r in old['rows'] if r['input'] == full['input'] and r['arm'] != 'full_return'):
            bt = base['total_conditional_resource']
            k_ratio = F(ft['expected_quantum_calls_two_axes']) / F(bt['expected_quantum_calls_two_axes'])
            per_call_ratio = (F(ft['T_Rz']) / F(ft['expected_quantum_calls_two_axes'])) / \
                             (F(bt['T_Rz']) / F(bt['expected_quantum_calls_two_axes']))
            ratio = F(ft['T_Rz']) / F(bt['T_Rz'])
            if k_ratio * per_call_ratio != ratio: raise ArithmeticError('saved factorization mismatch')
            output.append({'input': full['input'], 'baseline': base['arm'], 'K_ratio': k_ratio,
                'Rz_per_accepted_ratio': per_call_ratio, 'T_Rz_ratio': ratio,
                'delta_T_Rz': F(ft['T_Rz']) - F(bt['T_Rz']),
                'delta_provider_coefficients': [F(a)-F(b) for a,b in zip(ft['T_provider_coefficients'], bt['T_provider_coefficients'])],
                'G7_reclassified': False})
    return output


def independent_review_cap(generator, old_row):
    a, s = p5_root_formula(generator.p, generator.x)
    parent = generator.kernel.parent(())
    if (a, s) != (parent.a, parent.s): raise ArithmeticError('independent P5 root formula mismatch')
    M = old_row['total_conditional_resource']['quantum_call_hard_cap_two_axes']
    cap = accepted_cap(M, F(108, 125), t=7)
    return {'independent_p5_root_a': a, 'independent_p5_root_s': s,
            'coarse_z_upper': F(108, 125), 'M_original_G7': M,
            'review_t7_accepted_cap': cap, 'G7_hard_cap_unchanged': M,
            'reallocates_G7_failure': False, 'is_G7_registered_result': False}


def oracle_IS_diagnostic(generator, saved_cache, plan, probability_bits=160, root_bits=256):
    """One fixed finite-bit oracle q~alpha/sqrt(C_Rz), no grid, synthesis or RNG.

    It keeps each G7 rational coefficient/angle/event, but uses an enumerated
    sampling diagnostic. It is not the implemented local law or a native claim.
    """
    events = list(reference_events(generator))
    costs = [F(2 * saved_cache[str(e['ratio'])]['T_count']) for e in events]
    if min(costs) <= 0: raise ArithmeticError('registered Rz oracle diagnostic requires positive costs')
    unnormalized = [root_interval(e['coefficient']**2 / cost, root_bits) for e,cost in zip(events,costs)]
    lower, upper = sum(v.lo for v in unnormalized), sum(v.hi for v in unnormalized)
    law = dyadic_distribution(tuple(Interval(v.lo/upper,v.hi/lower) for v in unnormalized),
                             probability_bits, generator.eta)
    alphas = [e['coefficient'] for e in events]
    moment = sum(a*a/q for a,q in zip(alphas,law))
    W = max(a/q for a,q in zip(alphas,law))
    cost = sum(q*c for q,c in zip(law,costs))
    s = plan['remaining']
    N = ceil(log_upper(2/plan['alpha_axis']) * (2*moment/s**2+4*W/(3*s)))
    sqrt_terms = [root_interval(a*a*c,root_bits) for a,c in zip(alphas,costs)]
    leading_lower = 4 * log_interval(2/plan['alpha_axis']).lo * sum(v.lo for v in sqrt_terms)**2 / s**2
    return {'events': len(events), 'information_access': 'enumerated G7 coefficient+angle+saved Rz-cost table',
            'proposal': 'one q proportional to alpha_tilde/sqrt(C_Rz), dyadic160; no zero mass',
            'all_coefficients_preserved_exactly': all(q*(a/q)==a for q,a in zip(law,alphas)),
            'weight_second_moment': moment, 'range': W, 'expected_T_Rz_per_trial': cost,
            'N_per_axis': N, 'T_Rz_two_axes': 2*N*cost,
            'Cauchy_leading_objective_root_interval': Interval(sum(v.lo for v in sqrt_terms),sum(v.hi for v in sqrt_terms)),
            'fixed_Bernstein_policy_T_Rz_lower_for_all_positive_proposals': leading_lower,
            'native_provider_T_not_priced': True, 'runtime_acquisition_cost_not_free': True,
            'quantum_sampling_or_synthesis_calls': 0, 'production_law_changed': False,
            'G7_primary_classification_changed': False}

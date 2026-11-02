"""Exact sufficient-resource accounting, with no fixture or matrix access."""
from fractions import Fraction as F
from math import factorial

from ..g8_bounds import accepted_cap, ceil, log_upper, acceptance_upper


def validate_contract(c):
    if c['kind'] != 'RTE_DEVELOPMENT_D1_FIXED_CONTRACT_V1':
        raise ValueError('fixed D1 contract required')
    if c['steps'] != [1, 2] or c['degree'] != 5 or c['rows'] != 12:
        raise ValueError('fixed two-step-count comparison required')
    if c['input']['p'] != ['1/5', '3/10', '1/2'] or c['input']['tau'] != '5/7':
        raise ValueError('input changed')
    if c['input']['state'] != '000' or c['input']['sigma'] != 1:
        raise ValueError('common state and sign required')
    if c['arms'] != ['ordinary', 'partial_return_tail', 'closed_P3_tail',
                     'full_return', 'closed_P5_full', 'matched_CTS']:
        raise ValueError('same six comparators required')
    if c['primary_pair'] != ['partial_return_tail', 'closed_P5_full']:
        raise ValueError('primary comparator changed')
    if (24 * F(c['alpha_axis']) + 12 * F(c['resource_failure_per_row'])
            != F(c['familywise_alpha'])):
        raise ValueError('familywise accounting mismatch')
    if c['runs'] != 1 or c['retries'] != 0 or c['mandatory_STOP'] is not True:
        raise ValueError('one-shot STOP required')
    if c['synthesizer_options'].get('up_to_phase') is not False:
        raise ValueError('strict phase required')
    if c['caps']['total_wall_seconds'] is not None:
        raise ValueError('no total wall cap in this new contract')
    if c['cross_step_fusion'] or c['SP_enabled'] or c['proposal_search']:
        raise ValueError('comparison scope expanded')
    return c


def error_budget(x, steps, c):
    """exp(x)<3 for 0<=x<=1; each step's finite mean need not be unitary."""
    x, rho, primitive = F(x), F(c['rho']), F(c['primitive_error'])
    if not 0 <= x <= 1 or type(steps) is not int or steps < 1:
        raise ValueError('short independent steps required')
    epsilon = 3 * x**6 / factorial(6)
    e = 8 * rho + 8 * (1 + rho) * 2 * primitive
    truncation = (1 + epsilon)**steps - 1
    implementation = (1 + epsilon + e)**steps - (1 + epsilon)**steps
    bias = 2 * (truncation + implementation)
    remaining = F(c['epsilon_axis']) - bias
    if remaining <= 0:
        raise ArithmeticError('bias budget exhausted; no precision or r rescue')
    return dict(Taylor_step_upper=epsilon, rounded_native_step_upper=e,
                Taylor_product_upper=truncation,
                implementation_product_upper=implementation,
                axis_bias_upper=bias, remaining=remaining,
                uses_signal=False, probability_rounding_counted_once=True)


def step_bounds(generator=None, cts=None, c=None):
    if (generator is None) == (cts is None):
        raise ValueError('exactly one production representation required')
    rho, eta = F(c['rho']), F(c['eta'])
    B = generator.B.hi if generator is not None else sum(e['coefficient'] for e in cts)
    if not 0 < B < 8:
        raise ArithmeticError('common norm envelope violated')
    local = generator is not None and generator.arm == 'full_return'
    U = generator.U.hi if local else B
    kappa = (1 + rho)**2 / (1 - eta)**7
    return dict(M2_upper=kappa * B * U,
                range_upper=(1 + rho) * B / (1 - eta)**7,
                acceptance_upper=acceptance_upper(generator) if local else F(1),
                norm_or_envelope_upper=B, U_upper=U,
                budget_uses_reference_full_normalizer=False)


def whole_budget(bound, x, steps, c):
    err = error_budget(x, steps, c)
    moment, maximum = bound['M2_upper']**steps, bound['range_upper']**steps
    s = err['remaining']
    n = ceil(log_upper(2 / F(c['alpha_axis'])) *
             (2 * moment / s**2 + 4 * maximum / (3 * s)))
    if n > c['caps']['shot_cap_per_axis']:
        raise ArithmeticError('fixed shot cap exceeded; no clipping')
    z = bound['acceptance_upper']**steps
    return dict(**err, M2_upper=moment, range_upper=maximum,
                acceptance_upper=z, N_per_axis=n,
                alpha_axis=F(c['alpha_axis']),
                resource_failure_per_row=F(c['resource_failure_per_row']),
                accepted_tail_calls=accepted_cap(2*n, z, c['resource_tail_t']),
                hard_attempts=2*n, policy='independent whole trials including zeros',
                quantum_shots_executed=0)


def validate_events(events, bounds):
    if not events:
        raise ArithmeticError('empty positive support')
    q = moment = F(0)
    maximum = F(0)
    for e in events:
        a, p, w = map(F, (e['coefficient'], e['proposal'], e['weight']))
        if a <= 0 or p <= 0 or w != a/p:
            raise ArithmeticError('coefficient/proposal/weight mismatch')
        q += p; moment += p*w*w; maximum = max(maximum, w)
    if (not 0 < q <= 1 or q > bounds['acceptance_upper']
            or moment > bounds['M2_upper'] or maximum > bounds['range_upper']):
        raise ArithmeticError('digital law exceeds its production certificate')
    return dict(acceptance=q, M2=moment, range=maximum,
                budget_oracle=False, zero_mass=1-q)


def whole_cost(acceptance, per_trial, steps, budget, worst):
    z = F(acceptance)
    if not 0 < z <= 1 or steps < 1:
        raise ValueError('nonzero whole-path law required')
    factor = steps * z**(steps-1)
    p = {k: factor*F(v) for k, v in per_trial.items()}
    n = budget['N_per_axis']
    return dict(per_whole_trial_native=p,
                expected_native_intercept={k: 2*n*v for k,v in p.items()},
                prep_readout_slope=2*n*z**steps,
                expected_quantum_calls=2*n*z**steps,
                accepted_tail_native={k: budget['accepted_tail_calls']*steps*v for k,v in worst.items()},
                hard_native={k: budget['hard_attempts']*steps*v for k,v in worst.items()},
                cross_step_fusion=False, classical_zero_trial_cost_is_not_zero=True)


def affine_relation(left, right):
    """All h>=0 relation; no materiality threshold or scalarized resource."""
    d0 = F(left['intercept']) - F(right['intercept'])
    d1 = F(left['slope']) - F(right['slope'])
    if d0 == 0 and d1 == 0: kind = 'EQUAL_UNDER_FIXED_POLICY'
    elif d0 <= 0 and d1 <= 0: kind = 'LEFT_NONWORSE_ALL_NONNEGATIVE_H'
    elif d0 >= 0 and d1 >= 0: kind = 'RIGHT_NONWORSE_ALL_NONNEGATIVE_H'
    else: kind = 'RESOURCE_TRADEOFF'
    return dict(classification=kind, difference_intercept=d0, difference_slope=d1,
                crossover=(-d0/d1 if d1 and -d0/d1 >= 0 else None),
                method_adoption=False, necessary_shot_lower_bound_claim=False)

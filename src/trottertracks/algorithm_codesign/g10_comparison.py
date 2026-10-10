"""Fixed finite-confidence policy and native accounting for G10."""
from copy import deepcopy
from fractions import Fraction as F
from .g8_bounds import log_upper, ceil, accepted_cap, acceptance_upper
from .g9_native import native_ir, cost


def finish_plan(m2, W, B, z, c):
    rho, eps = F(c['rho']), F(c['primitive_error'])
    bias = 2*(8*rho+8*(1+rho)*2*eps)
    s, alpha = F(c['epsilon_axis'])-bias, F(c['alpha_axis'])
    if s <= 0 or not 0 < B < 8:
        raise ArithmeticError('common margin/norm envelope violated')
    N = ceil(log_upper(2/alpha)*(2*m2/s**2+4*W/(3*s)))
    if N > c['caps']['shot_cap_per_axis']:
        raise RuntimeError('registered shot cap')
    return {'N_per_axis': N, 'm2_upper': m2, 'range_upper': W,
            'common_bias_upper': bias, 'remaining': s,
            'alpha_axis': alpha, 'estimation_failure_34_axes': 34*alpha,
            'resource_failure_per_row': F(c['resource_failure_per_row']),
            'resource_failure_17_rows': 17*F(c['resource_failure_per_row']),
            'total_failure_upper': F(1, 20), 'exact_provider_delta': '0',
            'accepted_call_cap_two_axes': accepted_cap(2*N, z, c['resource_tail_t']),
            'acceptance_upper': z, 'hard_attempt_cap_two_axes': 2*N,
            'uses_signal_or_full_normalizer_for_local_budget': False}


def plan(c, m, g=None, cts=None):
    rho, eta = F(c['rho']), F(c['eta'])
    B = g.B.hi if g is not None else sum(e['coefficient'] for e in cts)
    local = g is not None and g.arm == 'full_return'
    kappa = (1+rho)**2/(1-eta)**(m+2)
    m2 = kappa*B*(g.U.hi if local else B)
    W = (1+rho)*B/(1-eta)**(m+2)
    return finish_plan(m2, W, B, acceptance_upper(g) if local else F(1), c)


def rebudget_anchor(saved, c):
    """Reuse G9's entire saved native/phase accounting, with the common 17-row N.

    No matrix, circuit, generator, synthesis, or error-guard evaluation here.
    Original G9 rows/status remain immutable and separately identified.
    """
    row = deepcopy(saved)
    b = saved['budget']
    # Old N policy also used this certified range; no full-normalizer knowledge is added.
    new = finish_plan(F(b['m2_upper']), F(b['range_upper']),
                      F(b['range_upper']), F(b['acceptance_upper']), c)
    N = new['N_per_axis']
    q = F(saved['reference_acceptance'])
    per = {k: F(v) for k, v in saved['per_trial_native_cost'].items()}
    row.update(degree=5, acquisition='G9_saved_anchor_rebudget_only',
               original_G9_budget=deepcopy(b), budget=new,
               two_axis_expected_native_cost={k: 2*N*v for k, v in per.items()},
               expected_accepted_calls=2*N*q, T_prep_readout_affine_coefficient=2*N*q,
               common_1Q_outer_prep_readout_two_axes=5*N*q,
               accepted_tail_T_upper=new['accepted_call_cap_two_axes']*saved['registered_worst_event_T'],
               hard_attempt_T_upper=2*N*saved['registered_worst_event_T'])
    return row


def row(m, arm, events, b, cache, guard=None):
    import numpy as np
    from .g9_matrix import circuit_error, event_operator
    from .g10_reference import matrix_target
    q = sum(e['proposal'] for e in events)
    m2 = sum(e['proposal']*e['weight']**2 for e in events)
    W = max(e['weight'] for e in events)
    if not 0 < q <= 1 or m2 > b['m2_upper'] or W > b['range_upper']:
        raise ArithmeticError('reference/policy bound mismatch')
    mean = sum(float(e['coefficient'])*event_operator(e) for e in events)
    residual = float(np.linalg.norm(mean-matrix_target(tuple(map(F, ('1/5','3/10','1/2'))), F(5,7), m), 2))
    if residual > 1e-10:
        raise ArithmeticError('same finite P_m first operator moment mismatch')
    bindings, per = [], {k: F(0) for k in ('T', 'CX', '1Q')}
    ideal_max = real_max = 0.
    maxT = 0
    for e in events:
        if guard is not None:
            guard.check()
        gates = native_ir(e)
        price = cost(gates, cache)
        ideal, realized = circuit_error(e, gates, None), circuit_error(e, gates, cache)
        if ideal > 1e-10 or realized > float(price['strict_error_upper'])+1e-10:
            raise ArithmeticError('strict controlled phase/order/composition failure')
        ideal_max, real_max = max(ideal_max, ideal), max(real_max, realized)
        maxT = max(maxT, price['T'])
        for k in per:
            per[k] += e['proposal']*price[k]
        bindings.append({'event': e, 'native_ir': gates, 'cost': price,
                         'ideal_phase_reference_error': ideal, 'realized_reference_error': realized})
    N, C = b['N_per_axis'], b['accepted_call_cap_two_axes']
    return {'degree': m, 'arm': arm, 'implementation': 'direct_primary', 'primary': True,
            'acquisition': 'G10_registered_degree', 'workspace_beyond_system': 1,
            'events': bindings, 'budget': b, 'reference_acceptance': q,
            'reference_m2': m2, 'reference_range': W,
            'ideal_mean_matrix_residual_diagnostic': residual,
            'max_ideal_native_phase_error_diagnostic': ideal_max,
            'max_realized_operator_error_diagnostic': real_max,
            'event_strict_error_analytic_bound': '2/1000000',
            'matrix_tolerance_is_diagnostic_not_bias_budget': True,
            'per_trial_native_cost': per,
            'two_axis_expected_native_cost': {k: 2*N*v for k, v in per.items()},
            'expected_accepted_calls': 2*N*q, 'T_prep_readout_affine_coefficient': 2*N*q,
            'common_1Q_outer_prep_readout_two_axes': 5*N*q,
            'registered_worst_event_T': maxT, 'accepted_tail_T_upper': C*maxT,
            'hard_attempt_T_upper': 2*N*maxT,
            'all_event_reference_not_injected_into_local_generator': True,
            'whole_circuit_global_optimization': False, 'physical_quantum_shots_executed': 0}

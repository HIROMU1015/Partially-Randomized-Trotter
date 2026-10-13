"""Rational accounting; accepts explicit data, never opens registered inputs."""
from fractions import Fraction as F
from math import factorial

from ..g8_bounds import ceil, log_upper, accepted_cap
from ..rte_development.boundary import native_price, gate

METRICS = ('T', 'CX', '1Q')


def error_budget(x, r, policy, c):
    x = F(x)
    if not 0 <= x <= 1 or type(r) is not int or r not in (1, 2):
        raise ValueError('fixed short steps')
    if policy not in ('G0', 'G1'):
        raise ValueError('unknown guarantee')
    a = c['error_accounting']
    if F(a['primitive_error_account']) != F(1, 10**6):
        raise ValueError('native account must remain 1e-6 in all ledgers')
    rho, primitive = F(a['rho']), F(a['primitive_error_account'])
    delta = (3 if policy == 'G0' else 1) * x**6 / factorial(6)
    e = 8*rho + 16*(1+rho)*primitive
    truncation = (1+delta)**r - 1
    implementation = (1+delta+e)**r - (1+delta)**r
    bias = 2*(truncation+implementation)
    remaining = F(a['epsilon_axis'])-bias
    if remaining <= 0:
        raise ArithmeticError('bias budget exhausted; no rescue')
    return dict(Taylor_step_upper=delta, rounded_native_step_upper=e,
                Taylor_product_upper=truncation, implementation_product_upper=implementation,
                axis_bias_upper=bias, remaining=remaining, uses_signal=False,
                probability_rounding_counted_once=True)


def whole_budget(bounds, x, r, policy, c, log_constant=None):
    err = error_budget(x, r, policy, c)
    m, w = F(bounds['M2_upper'])**r, F(bounds['range_upper'])**r
    z = F(bounds['acceptance_upper'])**r
    if m <= 0 or w <= 0 or not 0 < z <= 1 or F(bounds['norm_or_envelope_upper']) >= 8:
        raise ArithmeticError('saved production bounds invalid')
    a = c['error_accounting']
    log = log_upper(2/F(a['alpha_axis'])) if log_constant is None else log_constant
    s = err['remaining']
    n = ceil(log*(2*m/s**2 + 4*w/(3*s)))
    if n > c['caps']['shot_cap_per_axis']:
        raise ArithmeticError('shot cap; no clipping')
    return dict(**err, M2_upper=m, range_upper=w, acceptance_upper=z,
                N_per_axis=n, alpha_axis=F(a['alpha_axis']),
                resource_failure_per_row=F(a['resource_failure_per_row']),
                accepted_tail_calls=accepted_cap(2*n, z, a['resource_tail_t']),
                hard_attempts=2*n, policy='independent whole trials including zeros',
                quantum_shots_executed=0)


def equal_budget(actual, saved):
    if set(actual) != set(saved):
        raise ArithmeticError('legacy budget fields differ')
    for k, v in actual.items():
        other = F(saved[k]) if isinstance(v, F) else saved[k]
        if type(v) is bool and type(other) is not bool:
            raise ArithmeticError('legacy boolean changed')
        if v != other:
            raise ArithmeticError('legacy budget mismatch: '+k)
    return True


def event_prices(bindings, caches, check=lambda: None):
    """Both price conditions in one saved-event pass. No simplify or pairs."""
    per = {p: {k: F(0) for k in METRICS} for p in ('P0', 'P1')}
    worst = {p: {k: 0 for k in METRICS} for p in per}
    z = F(0)
    for b in bindings:
        check()
        e = b['event']
        q, a, weight = map(F, (e['proposal'], e['coefficient'], e['weight']))
        if q <= 0 or a <= 0 or weight != a/q:
            raise ArithmeticError('saved finite-bit coefficient/proposal/weight')
        ir = [gate(g) for g in b['native_ir']]
        for p in per:
            cost = native_price(ir, caches[p])
            if p == 'P0' and any(cost[k] != b['native_cost'][k] for k in METRICS):
                raise ArithmeticError('legacy event price mismatch')
            for k in METRICS:
                per[p][k] += q*cost[k]
                worst[p][k] = max(worst[p][k], cost[k])
        z += q
    if not 0 < z <= 1:
        raise ArithmeticError('saved acceptance invalid')
    return z, per, worst


def row_account(row, budget, step_prices, worst, z, saved_boundary=None):
    r, n = row['steps'], budget['N_per_axis']
    z = F(z)
    additive = {k: r*z**(r-1)*F(step_prices[k]) for k in METRICS}
    savings = {k: F(0) for k in METRICS}
    if r == 2:
        if saved_boundary is None or F(saved_boundary['acceptance_squared']) != z*z:
            raise ArithmeticError('fixed D2 boundary anchor required')
        savings = {k: F(saved_boundary['cancellation_saving_per_whole_trial'][k]) for k in METRICS}
    elif saved_boundary is not None:
        raise ValueError('r1 cannot have a boundary')
    prices = {k: additive[k]-savings[k] for k in METRICS}
    if any(savings[k] < 0 or prices[k] < 0 for k in METRICS):
        raise ArithmeticError('literal saving invalid')
    return dict(steps=r, arm=row['arm'], degree=5, x=F(row['x']),
                budget=budget, production_step_bounds=row['production_step_bounds'],
                step_acceptance=z, acceptance_path=z**r, per_step_trial_native=step_prices,
                literal_boundary_saving_per_whole_trial=savings,
                per_whole_trial_native=prices,
                expected_native_intercept={k: 2*n*prices[k] for k in METRICS},
                prep_readout_slope=2*n*z**r,
                expected_quantum_calls=2*n*z**r,
                worst_single_step_event_native=worst,
                accepted_tail_native={k: budget['accepted_tail_calls']*r*worst[k] for k in METRICS},
                hard_native={k: 2*n*r*worst[k] for k in METRICS},
                workspace_beyond_system=row['workspace_beyond_system'],
                reference_law_diagnostic_only={'acceptance': z},
                error_guarantee_is_state_uniform=True,
                native_error_account=F(1,10**6),
                classical_online_per_trial_cost='NOT_ACQUIRED; never zero',
                circuit_builds=0, registered_pair_evaluations=0)


def assert_legacy_cost(value, row, boundary):
    target = boundary['new_expected_native_intercept'] if boundary else row['expected_native_intercept']
    if any(value['expected_native_intercept'][k] != F(target[k]) for k in METRICS):
        raise ArithmeticError('G0/P0 cost anchor mismatch')
    if value['prep_readout_slope'] != F(row['prep_readout_slope']):
        raise ArithmeticError('G0/P0 slope anchor mismatch')
    for name in ('accepted_tail_native', 'hard_native'):
        if any(value[name][k] != F(row[name][k]) for k in METRICS):
            raise ArithmeticError('G0/P0 conservative resource cap anchor mismatch')
    return True


def factorial_effects(rows):
    by = {p: {v['row_index']: v for v in values} for p, values in rows.items()}
    result = []
    for i in sorted(by['G0_P0']):
        a,b,d,e = (by[p][i] for p in ('G0_P0','G1_P0','G0_P1','G1_P1'))
        for field in ('expected_native_intercept', 'prep_readout_slope'):
            coordinates = METRICS if field == 'expected_native_intercept' else ('common',)
            for k in coordinates:
                def get(v):
                    return F(v[field][k] if field == 'expected_native_intercept' else v[field])
                v0,v1,p0,p1 = map(get,(a,b,d,e))
                result.append(dict(row_index=i, field=field, coordinate=k,
                                   guarantee_only=v1-v0, price_only_G0=p0-v0,
                                   price_only_G1=p1-v1, interaction=(p1-v1)-(p0-v0)))
        if (a['budget'] != d['budget'] or b['budget'] != e['budget']):
            raise ArithmeticError('price axis altered error budget')
    return result

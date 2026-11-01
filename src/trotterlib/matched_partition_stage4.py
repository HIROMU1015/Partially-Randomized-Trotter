"""Saved-result conditional scalar accounting; no simulator/compiler imports."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import math

from .fixed_signal_accounting import AXES, fixed_signal_shots
from .h4_conditional_cng import trans_bounds

EPSILONS = ('0.05', '0.01', '0.005', '0.001', '0.0005')
METRICS = ('RZ', 'CX', 'depth')
AXIS_FIELDS = dict(cosine='real', sine='imag')


def q(value):
    if isinstance(value, bool) or isinstance(value, float) and not math.isfinite(value):
        raise ValueError('FINITE_SCALAR_REQUIRED')
    return F(value)


def outward(value, *, upper):
    value = q(value)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('BINARY64_RANGE')
    if (upper and F(result) < value) or (not upper and F(result) > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    return result


def epsilon_axis(epsilon):
    epsilon = q(epsilon)
    with localcontext() as ctx:
        ctx.prec = 96
        root_upper = F(Decimal(2).sqrt().next_plus())
    result = outward(epsilon / root_upper, upper=False)
    assert 2 * F(result)**2 <= epsilon**2
    return result


def bias_inputs(stage2, arm):
    nominal = stage2['signals']['nom_' + arm]
    reference = stage2['reference']
    fields = ('u_num', 'u_bind', 'u_ref_transfer')
    if any(nominal.get(key) is None for key in fields) or reference.get('transfer_uncertainty') is None:
        return dict(status='MISSING_NUMERICAL_ALLOWANCE', axes=None)
    terms = {key: q(nominal[key]) for key in fields}
    transfer = q(reference['transfer_uncertainty'])
    if min(*terms.values(), transfer) < 0:
        raise ValueError('NEGATIVE_NUMERICAL_ALLOWANCE')
    result = {}
    for axis, field in AXIS_FIELDS.items():
        if nominal.get(field) is None:
            return dict(status='MISSING_NOMINAL_POINT', axes=None)
        z = q(nominal[field])
        interval = reference['physical_interval'][field]
        lower, upper = q(interval['lower']), q(interval['upper'])
        if lower > upper:
            raise ValueError('REFERENCE_INTERVAL_ORDER')
        distance = max(abs(z-lower), abs(z-upper))
        signed = z - (lower+upper)/2
        total = distance + sum(terms.values()) + transfer
        result[axis] = dict(nominal_point_exact=str(z), signed_bias_midpoint_exact=str(signed),
            reference_interval_exact=[str(lower), str(upper)], endpoint_distance_exact=str(distance),
            numerical_terms_exact={key: str(value) for key, value in terms.items()},
            transfer_exact=str(transfer), bias_upper_exact=str(total),
            bias_upper_operand=outward(total, upper=True),
            split_uncertainty_added=False, reference_width_added_separately=False)
    return dict(status='ENGINEERING_CONDITIONAL', axes=result)


def normalization_input(saved):
    if any(saved.get(key) is None for key in ('B', 'B_uncertainty')) or saved.get('decimal50', {}).get('B') is None:
        return dict(status='MISSING_B_UPPER_CANDIDATE')
    anchor, uncertainty, check50 = q(saved['B']), q(saved['B_uncertainty']), q(saved['decimal50']['B'])
    if anchor < 1 or uncertainty < 0 or check50 < 1:
        raise ValueError('NORMALIZATION_DOMAIN')
    engineering_upper = max(anchor + uncertainty, check50)
    if engineering_upper == 1:
        log_upper, exp_lower, exp_upper = 0., F(1), F(1)
    else:
        _, log_hi = trans_bounds('ln', engineering_upper)
        log_upper = outward(log_hi, upper=True)
        exp_lower, exp_upper = trans_bounds('exp', F(log_upper))
        if exp_lower < engineering_upper:
            raise ValueError('LOG_B_OUTWARD_CHECK')
    live_B = math.exp(log_upper)
    effective_upper = max(engineering_upper, exp_upper, F(live_B))
    return dict(status='ENGINEERING_UPPER_ASSUMED', B_saved_exact=str(anchor),
        B_uncertainty_exact=str(uncertainty), decimal50_B_exact=str(check50),
        B_engineering_upper_exact=str(engineering_upper), log_B_upper_operand=log_upper,
        live_exp_log_B_exact=str(F(live_B)), B_scalar_upper_exact=str(effective_upper),
        nominal_signal_rescaled=False, normalization_certificate=False)


def price_moments(prices, summary, *, n=32):
    expected = {f'{arm}_{i:02d}/F/{axis}' for arm in ('P', 'DF') for i in range(n) for axis in AXES}
    if set(prices) != expected or summary['old_pilot_pooled'] or summary['cross_arm_paired']:
        raise ValueError('FIXED_INDEPENDENT_PRICE_COHORT')
    results = {}
    for arm in ('P', 'DF'):
        if summary['arms'][arm]['n'] != n or n < 2:
            raise ValueError('PRICE_SAMPLE_SIZE')
        results[arm] = {}
        for metric in METRICS:
            columns = {axis: [q(prices[f'{arm}_{i:02d}/F/{axis}'][metric]) for i in range(n)] for axis in AXES}
            if any(v < 0 or v.denominator != 1 for col in columns.values() for v in col):
                raise ValueError('INTEGER_PRICE_REQUIRED')
            means = {axis: sum(col)/n for axis,col in columns.items()}
            variances = {axis: sum((v-means[axis])**2 for v in col)/(n-1) for axis,col in columns.items()}
            covariance = sum((x-means['cosine'])*(y-means['sine'])
                             for x,y in zip(columns['cosine'], columns['sine']))/(n-1)
            for axis in AXES:
                saved = summary['arms'][arm]['axes'][axis][metric]
                if q(saved['mean_exact']) != means[axis] or not math.isclose(
                        saved['unbiased_sample_variance'], float(variances[axis]), rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError('SAVED_PRICE_STATISTICS_MISMATCH')
            if q(summary['arms'][arm]['cosine_sine_sample_covariance_exact'][metric]) != covariance:
                raise ValueError('SAVED_PRICE_COVARIANCE_MISMATCH')
            results[arm][metric] = dict(n=n, means_exact={k:str(v) for k,v in means.items()},
                sample_variances_exact={k:str(v) for k,v in variances.items()},
                cosine_sine_covariance_exact=str(covariance))
    return results


def weighted_cost(prices, arm, metric, moments, shots):
    if any(shots.get(axis) is None for axis in AXES):
        return None
    n = moments['n']
    means = {a:q(v) for a,v in moments['means_exact'].items()}
    variances = {a:q(v) for a,v in moments['sample_variances_exact'].items()}
    covariance = q(moments['cosine_sine_covariance_exact'])
    nc, ns = shots['cosine'], shots['sine']
    mean = sum(shots[a]*means[a] for a in AXES)
    variance_mean = (nc*nc*variances['cosine'] + ns*ns*variances['sine'] + 2*nc*ns*covariance)/n
    values = [sum(shots[a]*q(prices[f'{arm}_{i:02d}/F/{a}'][metric]) for a in AXES) for i in range(n)]
    if mean != sum(values)/n or variance_mean != sum((x-mean)**2 for x in values)/(n-1)/n:
        raise ValueError('WEIGHTED_COVARIANCE_IDENTITY')
    if variance_mean < 0:
        raise ValueError('NEGATIVE_COST_VARIANCE')
    return dict(G_exact=str(mean), G=float(mean), variance_of_mean_exact=str(variance_mean),
        price_standard_error=math.sqrt(float(variance_mean)),
        cosine_sine_covariance_included=True, direct_weighted_sample_check=True,
        SE_scope='conditional on fixed shots; price sample uncertainty only')


def account_row(arm, epsilon, bias, normalization, moments, prices):
    result = dict(arm=arm, epsilon_sig=epsilon, evidence_kind='EMPIRICAL',
        conditional_assumptions=dict(finite_bit_gap=0, compiled_rho=0, B_engineering_upper_assumed=True),
        numerical_certificate=False, N_total=None, costs={m:None for m in METRICS})
    if bias.get('axes') is None or normalization['status'] != 'ENGINEERING_UPPER_ASSUMED':
        return dict(result, status='UNRESOLVED_MISSING_INPUT', axes={}, legacy=None)
    eps = epsilon_axis(epsilon)
    alpha = outward(F(1,40), upper=False)
    upper_bias = {a:bias['axes'][a]['bias_upper_operand'] for a in AXES}
    legacy = fixed_signal_shots(epsilon_sig=float(epsilon), bias=upper_bias,
        allowance=dict.fromkeys(AXES, 0.), log_B_upper=normalization['log_B_upper_operand'],
        evidence_kind='EMPIRICAL', evidence_ref='P3_STAGE4_SAVED_NOMINAL_ENGINEERING_CONDITIONAL',
        epsilon_axes=dict.fromkeys(AXES, eps), alpha_axes=dict.fromkeys(AXES, alpha))
    _, ln_upper = trans_bounds('ln', F(2)/F(alpha))
    scale = 2*q(normalization['B_scalar_upper_exact'])**2*ln_upper
    axes = {}
    for axis in AXES:
        h_exact = F(eps)-F(upper_bias[axis])
        live_h = legacy['axes'][axis]['headroom']
        h = min(h_exact, F(live_h))
        row = dict(epsilon_operand=eps, epsilon_operand_exact=str(F(eps)),
            alpha_operand=alpha, alpha_operand_exact=str(F(alpha)),
            bias_upper_operand=upper_bias[axis], headroom_exact=str(h), u_remaining=0,
            legacy_shots=legacy['axes'][axis]['shots'], shots=None,
            status='UNRESOLVED_NO_POSITIVE_HEADROOM')
        if h > 0:
            upper = scale/h**2
            conservative = max(1, math.ceil(upper))
            candidate = legacy['axes'][axis]['shots']
            if candidate is not None and max(candidate, conservative) < 2**52:
                n = max(candidate, conservative)
                assert F(n)*h*h >= scale
                row.update(status='ELIGIBLE_CONDITIONAL', shots=n,
                    scalar_bound_upper_exact=str(upper), ceil_adjustment=n-candidate,
                    sufficient_scalar_inequality_verified=True,
                    ceil_adjustment_reason='max of legacy return and rational scalar upper ceiling')
            else:
                row.update(status='UNRESOLVED_SAFE_INTEGER_LIMIT', scalar_bound_upper_exact=str(upper))
        axes[axis] = row
    result.update(axes=axes, legacy=legacy, status='UNRESOLVED',
                  population_mean_upper=None, implemented_accuracy_guaranteed=False)
    if all(row['shots'] is not None for row in axes.values()):
        shots = {a:axes[a]['shots'] for a in AXES}
        result.update(status='ELIGIBLE_CONDITIONAL', N_total=sum(shots.values()),
            costs={m:weighted_cost(prices, arm, m, moments[m], shots) for m in METRICS})
    return result


def comparisons(rows):
    result = []
    for epsilon in EPSILONS:
        p, df = (next(r for r in rows if r['arm']==arm and r['epsilon_sig']==epsilon) for arm in ('P','DF'))
        metrics = {}
        for metric in METRICS:
            left, right = p['costs'][metric], df['costs'][metric]
            value = None
            if left is not None and right is not None:
                delta = q(left['G_exact'])-q(right['G_exact'])
                variance = q(left['variance_of_mean_exact'])+q(right['variance_of_mean_exact'])
                ratio = q(left['G_exact'])/q(right['G_exact']) if q(right['G_exact']) else None
                value = dict(P_minus_DF_exact=str(delta), P_minus_DF=float(delta),
                    independent_price_standard_error=math.sqrt(float(variance)),
                    P_over_DF_exact=str(ratio) if ratio is not None else None,
                    P_over_DF=float(ratio) if ratio is not None else None,
                    cross_arm_paired=False, population_order_guaranteed=False)
            metrics[metric] = value
        result.append(dict(epsilon_sig=epsilon, P_status=p['status'], DF_status=df['status'], metrics=metrics))
    return result


def evaluate(stage2, prices, summary, checkpoint=None):
    biases = {a:bias_inputs(stage2, a) for a in ('P','DF')}
    normalizations = {a:normalization_input(stage2['normalization'][a]) for a in ('P','DF')}
    moments = price_moments(prices, summary)
    rows = []
    for epsilon in EPSILONS:
        for arm in ('P','DF'):
            rows.append(account_row(arm, epsilon, biases[arm], normalizations[arm], moments[arm], prices))
            if checkpoint: checkpoint(rows)
    return dict(bias_inputs=biases, normalization_inputs=normalizations, price_moments=moments,
        rows=rows, comparisons=comparisons(rows), analysis='NOMINAL_CONDITIONAL_ONLY',
        numerical_certificate=False, finite_bit_gap_measured=False, compiled_rho_measured=False,
        population_tail_bound=None, RPE_all_round_cost_evaluated=False)

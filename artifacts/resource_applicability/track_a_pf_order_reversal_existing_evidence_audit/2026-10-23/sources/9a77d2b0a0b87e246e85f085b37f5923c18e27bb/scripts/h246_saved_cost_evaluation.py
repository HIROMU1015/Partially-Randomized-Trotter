"""Saved integer circuit prices and conditional empirical shot accounting only."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import statistics
import d0_frozen_shot_rule_v1 as frozen

EPS = (.05, .01, .005, .001, .0005)
MODES = ('USUAL_SIGNED', 'NONCANCELLATION')
FORMULAS = ('2nd', '4th', '4th(m5_best)')
STRATA = ('S0', 'S2_single', 'S2_multi', 'S4')
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
NATIVE = 'M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def close(a, b, code):
    require(math.isfinite(a) and math.isfinite(b) and
            abs(a-b) <= 256*math.ulp(1.)*max(1., abs(a), abs(b)), code)


def rational(x):
    return dict(numerator=x.numerator, denominator=x.denominator)


def frac(value):
    return Fraction(value['numerator'], value['denominator'])


def moments(values):
    """Exact integer sufficient statistics; independent stdlib variance check."""
    require(values and all(type(x) is int and x >= 0 for x in values), 'INTEGER_PRICES')
    n = len(values)
    mean = Fraction(sum(values), n)
    variance = (Fraction(sum(x*x for x in values), 1)-n*mean*mean)/(n-1) if n > 1 else None
    close(float(mean), statistics.mean(values), 'INDEPENDENT_MEAN')
    if variance is not None:
        close(float(variance), statistics.variance(values), 'INDEPENDENT_VARIANCE')
    return dict(n=n, mean=float(mean), mean_exact=rational(mean),
                variance=None if variance is None else float(variance),
                variance_exact=None if variance is None else rational(variance),
                SE=None if variance is None else math.sqrt(float(variance/n)),
                minimum=min(values), maximum=max(values))


def paired(prices):
    x, y = zip(*prices)
    a, b = moments(x), moments(y)
    n = len(x)
    mx, my = frac(a['mean_exact']), frac(b['mean_exact'])
    cov = (Fraction(sum(u*v for u, v in prices))-n*mx*my)/(n-1) if n > 1 else None
    if cov is not None:
        ox, oy = x[0], y[0]
        dx, dy = math.fsum(u-ox for u in x)/n, math.fsum(v-oy for v in y)/n
        independent = math.fsum(((u-ox)-dx)*((v-oy)-dy) for u, v in prices)/(n-1)
        close(float(cov), independent, 'INDEPENDENT_CENTERED_COVARIANCE')
    return dict(cosine=a, sine=b, covariance=None if cov is None else float(cov),
                covariance_exact=None if cov is None else rational(cov))


def load_price(result, task, binding, preparation, log_B, source, plan):
    require(result['task'] == task and result['SOURCE'] == source and
            result['plan_sha256'] == plan, 'TRAJECTORY_SOURCE_IDENTITY')
    require(result['stage'] == task['stage'] and result['seed_cohort'] == 'EXPLORATION'
            and task['stage'].endswith('_PR_EXPLORATION') and result['paired_trajectory'],
            'EXPLORATION_ONLY_PAIRED')
    require(result['status'] == 'UNIT_COST_PASS_NOT_ACCOUNTING' and
            result['native_scientific_PASS'] and result['byte_restoration_PASS'] and
            not any(result[k] for k in ('conditional_sampling', 'cohort_pooling',
                                       'shot_accounting', 'cost_ranking')), 'ACCEPTED_UNIT_COST')
    require(result['input_binding'] == binding and result['preparation_hash'] == preparation,
            'INPUT_PREPARATION_MATCH')
    close(result['log_B'], log_B, 'SAVED_NORMALIZATION_MATCH')
    checks = result['checks']
    require(len(checks) == 8 and all(v['PASS'] and not v['phase_alignment'] and
            v['strictly_below'] == (1e-7 if v['check'].startswith('compiled_vector_') else 1e-9)
            and 0 <= v['error'] < v['strictly_below'] for v in checks), 'NATIVE_GATES')
    axes = result['axis_costs']
    require([a['axis'] for a in axes] == ['cosine', 'sine'], 'AXIS_PAIR')
    require(all(a['compiler_settings_hash'] == COMPILER and a['native_policy'] == NATIVE
                and a['preparation_hash'] == preparation and
                a['event_digest'] == result['event_digest'] for a in axes), 'COMPILER_NATIVE_PAIR')
    event = result['events']
    require(all(type(event[k]) is int and event[k] >= 0 for k in ('N2', 'N4')),
            'EVENT_COUNTS')
    stratum = ('S4' if event['N4'] else 'S2_multi' if event['N2'] >= 2 else
               'S2_single' if event['N2'] else 'S0')
    require(event['stratum'] == stratum and event['N2'] + event['N4'] <= 32*task['candidate']['q'],
            'SAVED_STRATUM_LABEL')
    values = tuple(a['metrics']['rz_count'] for a in axes)
    require(all(type(v) is int and v >= 0 for v in values), 'DIRECT_INTEGER_RZ')
    return values, stratum


def signal_bias(record, mode):
    s = record['signal']
    for axis in ('real', 'imag'):
        total = s['total_signed_bias'][axis]
        require(math.isfinite(total), 'FINITE_BIAS')
        close(total, s['corrected'][axis]-s['reference'][axis], 'TOTAL_BIAS_CLOSURE')
        close(s['finite_RTE_bias'][axis]+s['outer_PF_or_discard_bias'][axis],
              total, 'COMPONENT_BIAS_CLOSURE')
    if mode == 'USUAL_SIGNED':
        return {a: abs(s['total_signed_bias'][a]) for a in ('real', 'imag')}, None
    if record['candidate']['method'] == 'PR':
        return {a: abs(s['outer_PF_or_discard_bias'][a])+abs(s['finite_RTE_bias'][a])
                for a in ('real', 'imag')}, None
    if record['candidate']['method'] == 'FULL':
        return {a: abs(s['total_signed_bias'][a]) for a in ('real', 'imag')}, None
    # The saved combined PF/discard error cannot be separated after the fact.
    return None, 'MISSING_SEPARATE_DISCARD_AND_PF_COMPONENTS'


def independent_shots(epsilon, bias, u, log_B):
    """Decimal primitive arithmetic, independently rounded to binary64 per op.

    sqrt/exp/log inputs use the same libm values as the frozen contract; exact
    integer ceiling and primitive arithmetic are checked on a separate path.
    This is numerical crosschecking, not a second independent scientific review.
    """
    D = Decimal.from_float
    with localcontext() as context:
        context.prec = 90
        eps_axis = float(D(epsilon)/D(math.sqrt(2)))
        h = float(D(float(D(eps_axis)-D(bias)))-D(u))
        if h <= 0:
            return h, None
        log_n = math.log(2)+2*log_B-2*math.log(h)+math.log(math.log(2/.025))
        if log_n >= math.log(2**52):
            return h, None
        B = math.exp(log_B)
        b2 = float(D(B)*D(B))
        h2 = float(D(h)*D(h))
        numerator = float(Decimal(2)*D(b2))
        quotient = float(D(numerator)/D(h2))
        bound = float(D(quotient)*D(math.log(2/.025)))
        return h, max(1, int(Decimal.from_float(bound).to_integral_value(rounding='ROUND_CEILING')))


def shot_record(record, epsilon, mode):
    biases, missing = signal_bias(record, mode)
    if missing:
        return dict(status='MISSING_COMPONENTS', missing_reason=missing, axes=None,
                    N_cos=None, N_sin=None, N_total=None)
    empirical = record['empirical_eligibility']
    u, margin = empirical['u'], empirical['log_B_margin']
    require(math.isfinite(u) and u >= 0 and math.isfinite(margin) and margin >= 0,
            'SAVED_EMPIRICAL_ALLOWANCE')
    L = record['signal']['log_B'] + margin
    result = frozen.u_aware_shots(biases, dict(real=u, imag=u), epsilon=epsilon,
        log_B_upper=L, alpha_axis=.025, evidence_kind='EMPIRICAL',
        evidence_ref='SAVED_SIGNAL_EMPIRICAL_REFERENCE_CONDITIONAL')
    axis_rows = result['axes']
    for axis, row in axis_rows.items():
        h, independent = independent_shots(epsilon, biases[axis], u, L)
        require(h == row['headroom'], 'INDEPENDENT_HEADROOM_EXACT')
        if row['shots'] is not None:
            require(independent == row['shots'], 'INDEPENDENT_SHOT_INTEGER_EXACT')
        if row['status'] == 'ELIGIBLE' and row['u_over_headroom'] > .01:
            row['status'] = 'UNDETERMINED'
            row['shots'] = None
            row['guard_reason'] = 'U_OVER_HEADROOM'
        if mode == 'USUAL_SIGNED':
            saved = next(x for x in empirical['rows'] if x['epsilon'] == epsilon and x['axis'] == axis)
            require(saved['u'] == u and saved['log_B'] == record['signal']['log_B'] and
                    saved['log_B_margin'] == margin and saved['bias'] == record['signal']['total_signed_bias'][axis],
                    'SAVED_PRECISION_ROW_IDENTITY')
            require(saved['status_under_proposed_empirical_policy'] == row['status'],
                    'SAVED_ELIGIBILITY_REPRODUCTION')
    statuses = {x['status'] for x in axis_rows.values()}
    status = ('INELIGIBLE' if 'INELIGIBLE' in statuses else
              'UNDETERMINED' if statuses != {'ELIGIBLE'} else
              'LOG_DOMAIN_ONLY' if any(x['shots'] is None for x in axis_rows.values()) else 'ELIGIBLE')
    nc, ns = (axis_rows['real']['shots'], axis_rows['imag']['shots']) if status == 'ELIGIBLE' else (None, None)
    return dict(status=status, missing_reason=None, axes=axis_rows,
                N_cos=nc, N_sin=ns, N_total=None if nc is None else nc+ns,
                bias_used=biases, saved_signed_bias=record['signal']['total_signed_bias'],
                u=u, log_B=record['signal']['log_B'], log_B_margin=margin,
                log_B_for_shots=L, strict_bias_certificate=None)


def weighted(prices, stats, nc, ns):
    values = [nc*x+ns*y for x, y in prices]
    value = moments(values)
    exact_mean = nc*frac(stats['cosine']['mean_exact'])+ns*frac(stats['sine']['mean_exact'])
    require(frac(value['mean_exact']) == exact_mean, 'PAIRED_COST_EXACT_MEAN')
    if len(values) > 1:
        exact_var = (nc*nc*frac(stats['cosine']['variance_exact'])+
                     ns*ns*frac(stats['sine']['variance_exact'])+
                     2*nc*ns*frac(stats['covariance_exact']))
        require(frac(value['variance_exact']) == exact_var, 'PAIRED_COST_COVARIANCE_IDENTITY')
    # Independent centered two-pass path, distinct from integer sufficient sums.
    anchor = values[0]
    offset_mean = math.fsum(x-anchor for x in values)/len(values)
    mean = anchor+offset_mean
    variance = math.fsum(((x-anchor)-offset_mean)**2 for x in values)/(len(values)-1) if len(values) > 1 else None
    close(mean, value['mean'], 'INDEPENDENT_WEIGHTED_MEAN')
    if variance is not None:
        close(variance, value['variance'], 'INDEPENDENT_WEIGHTED_VARIANCE')
    return value


def evaluate_candidate(record, prices, strata=None):
    c = record['candidate']
    stats = paired(prices) if prices else None
    rows = []
    for mode in MODES:
        for epsilon in EPS:
            shot = shot_record(record, epsilon, mode)
            row = dict(system=c['system'], candidate_id=c['id'], formula=c['formula'],
                method=c['method'], prefix=c['prefix'], q=c['q'], mode=mode,
                epsilon=epsilon, n=len(prices) if c['method'] == 'PR' else None,
                shot=shot, status=shot['status'], total_RZ=None, total_RZ_exact=None,
                total_RZ_variance=None, total_RZ_SE=None, effective_unit_RZ=None,
                mean_cos_RZ=None if not stats else stats['cosine']['mean'],
                mean_sin_RZ=None if not stats else stats['sine']['mean'],
                unit_cost_kind='DIRECT_MEASURED_SAMPLE' if c['method']=='PR' else
                               'DIRECT_MEASURED_NEW_OR_SAVED_REUSE' if prices else 'MISSING',
                whole_strengthened_PF_set_best=False, population_guarantee=False,
                independent_confirmation=False)
            if shot['status'] == 'ELIGIBLE' and not prices:
                row['status'] = 'MISSING_COMPILED_COST'
            if shot['status'] == 'ELIGIBLE' and prices:
                nc, ns = shot['N_cos'], shot['N_sin']
                g = weighted(prices, stats, nc, ns)
                row.update(total_RZ=g['mean'], total_RZ_exact=g['mean_exact'],
                    total_RZ_variance=g['variance'] if c['method']=='PR' else None,
                    total_RZ_SE=g['SE'] if c['method']=='PR' else None,
                    cosine_RZ_contribution=float(nc*frac(stats['cosine']['mean_exact'])),
                    sine_RZ_contribution=float(ns*frac(stats['sine']['mean_exact'])),
                    effective_unit_RZ=float(frac(g['mean_exact'])/(nc+ns)))
            rows.append(row)
    stratified = None
    if strata is not None:
        stratified = {}
        for s in STRATA:
            selected = [p for p, label in zip(prices, strata) if label == s]
            stratified[s] = dict(observed_count=len(selected),
                conditional_statistics=paired(selected) if selected else None,
                population_cost_bound=None, population_variance_bound=None,
                observation_is_population_guarantee=False)
    return dict(candidate=c, unit_statistics=stats, strata=stratified, rows=rows)


def select(rows):
    """Unknown admissibility/cost prevents a complete-set optimum claim."""
    if any(r['status'] not in ('ELIGIBLE', 'INELIGIBLE') for r in rows):
        return dict(status='UNRESOLVED_FIXED_SET', best=None, tied_candidate_ids=[])
    usable = [r for r in rows if r['status'] == 'ELIGIBLE' and r['total_RZ_exact'] is not None]
    if not usable:
        return dict(status='NO_ELIGIBLE_CANDIDATE', best=None, tied_candidate_ids=[])
    best = min(usable, key=lambda r: (frac(r['total_RZ_exact']), r['q'], r['candidate_id']))
    ties = sorted(r['candidate_id'] for r in usable if frac(r['total_RZ_exact']) == frac(best['total_RZ_exact']))
    return dict(status='EXPLORATORY_POINT_BEST_FIXED_SET', best=best, tied_candidate_ids=ties)


def comparison(base, other):
    if base is None or other is None:
        return None
    a, b = frac(base['total_RZ_exact']), frac(other['total_RZ_exact'])
    n0, n1 = base['shot']['N_total'], other['shot']['N_total']
    ratio = b/a
    shot_ratio = Fraction(n1, n0)
    unit_ratio = (b/n1)/(a/n0)
    require(ratio == shot_ratio*unit_ratio, 'EXACT_SHOT_EFFECTIVE_UNIT_DECOMPOSITION')
    return dict(total_cost_ratio=float(ratio), total_cost_ratio_exact=rational(ratio),
                improvement_percent=float(100*(1-ratio)), shot_total_ratio=float(shot_ratio),
                effective_unit_ratio=float(unit_ratio),
                unpaired_difference_SE=math.hypot(base['total_RZ_SE'] or 0., other['total_RZ_SE'] or 0.),
                independent_candidate_cohorts_assumed=True,
                ratio_SE_or_population_CI=None,
                selection_effect_corrected=False, population_guarantee=False)


def summarize(evaluations):
    rows = [row for value in evaluations for row in value['rows']]
    summary = []
    for system in ('H2', 'H4', 'H6'):
        for mode in MODES:
            for epsilon in EPS:
                subset = [r for r in rows if (r['system'],r['mode'],r['epsilon']) == (system,mode,epsilon)]
                families = {formula: select([r for r in subset if r['method']=='PR' and r['formula']==formula])
                            for formula in FORMULAS}
                det = [r for r in subset if r['method'] != 'PR']
                det_result = select(det) if det else dict(status='NO_IDENTITY_BOUND_DETERMINISTIC_SIGNAL_SET',best=None)
                s2 = families['2nd']['best']
                summary.append(dict(system=system, mode=mode, epsilon=epsilon,
                    families=families, PR_best=select([r for r in subset if r['method']=='PR']),
                    higher_PF_vs_S2={f:comparison(s2, families[f]['best']) for f in FORMULAS[1:]},
                    deterministic_m5_subset=det_result,
                    deterministic_m5_subset_only=True, strengthened_S2_S4_best=None,
                    PR_vs_m5_subset={f:comparison(det_result['best'], families[f]['best']) if det_result['best'] else None
                                     for f in FORMULAS},
                    old_PF_controls_replaced=False, AD03='DEFERRED'))
    return summary

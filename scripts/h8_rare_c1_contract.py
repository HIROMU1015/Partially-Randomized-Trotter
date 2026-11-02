"""Contract arithmetic only: no molecular inputs, sampler, provider or launch CLI."""
import math

BASE = 'cb9c994be608233fccc49ff2e44007bbb32e2fb4'
ETA = 0.5
ALPHA = 0.1


def chi2_cdf(x, df):
    """Regularized lower-gamma positive series, binary64 planning arithmetic."""
    if not (df >= 1 and x >= 0):
        raise ValueError('CHI2_DOMAIN')
    if x == 0:
        return 0.0
    a, z = df / 2, x / 2
    term = total = 1 / a
    k = 0
    while True:
        k += 1
        term *= z / (a + k)
        total += term
        if term <= total * 2e-16:
            break
    return min(1.0, total * math.exp(a * math.log(z) - z - math.lgamma(a)))


def independent_cdf(x, df):
    """Separate half-integer recurrence from exp/erf; no gamma-series call."""
    z = x / 2
    if df % 2 == 0:
        a = 1.0
        p = -math.expm1(-z)
        density_term = math.exp(-z) * z
    else:
        a = 0.5
        p = math.erf(math.sqrt(z))
        density_term = 2 * math.sqrt(z / math.pi) * math.exp(-z)
    while a < df / 2:
        p -= density_term
        a += 1
        density_term *= z / a
    return p


def quantile(p, df, cdf=chi2_cdf):
    if not 0 < p < 1:
        raise ValueError('QUANTILE_DOMAIN')
    low, high = 0., float(df)
    while cdf(high, df) < p:
        high *= 2
    while high - low > 1e-12 * max(1., high):
        middle = (high + low) / 2
        if cdf(middle, df) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def sample_design():
    """Smallest n: both equal-tail 90% variance-CI ends within s²*(1±eta)."""
    rows = []
    n = 2
    while True:
        df = n - 1
        # Equivalent quantile inequalities; avoids repeated inverse solves.
        a = chi2_cdf(df / (1 + ETA), df)
        b = chi2_cdf(df / (1 - ETA), df)
        ia = independent_cdf(df / (1 + ETA), df)
        ib = independent_cdf(df / (1 - ETA), df)
        if max(abs(a - ia), abs(b - ib)) > 5e-13:
            raise ValueError('INDEPENDENT_DESIGN_DISAGREEMENT')
        passed = a <= ALPHA / 2 and b >= 1 - ALPHA / 2
        if passed != (ia <= ALPHA / 2 and ib >= 1 - ALPHA / 2):
            raise ValueError('INDEPENDENT_DESIGN_DECISION')
        rows.append(dict(n=n, df=df, lower_threshold_CDF=a, upper_threshold_CDF=b,
                         independent_lower_CDF=ia, independent_upper_CDF=ib, passed=passed))
        if passed:
            break
        n += 1
    endpoints = []
    for count in (n - 1, n):
        df = count - 1
        low = quantile(ALPHA / 2, df)
        high = quantile(1 - ALPHA / 2, df)
        endpoints.append(dict(n=count, df=df, chi2_05=low, chi2_95=high,
                              CI_lower_over_sample_variance=df / high,
                              CI_upper_over_sample_variance=df / low))
    return dict(n_per_condition=n, conditions=2, trajectories_total=2*n, eta='1/2',
                planning_confidence='0.90 per condition, not simultaneous',
                criterion='[df/chi2(.95,df),df/chi2(.05,df)] subset [0.5,1.5]',
                assumption='iid Normal conditional paired cost, positive finite variance; unverified planning model',
                all_smaller_integer_n_checked=True, rows=rows, endpoints=endpoints,
                H8_scientific_analysis=False, population_guarantee=False)


def budget(q, n):
    if q not in (2, 16) or type(n) is not int or n < 2:
        raise ValueError('FIXED_PILOT_SCOPE')
    return dict(saved_preparation_decode=1, prefix_reconstruction=1, law_formation=1,
                law_compatibility=1, count_law_setup=1, component_table_setup=1,
                trajectory=n, occurrence=q*n, RTE_event=4*q*n, native_evolution=n,
                wrapper_build=2*n, wrapper_compile=2*n, control_probe=4,
                event_oracle=1, validation_wrapper_build=2, pilot_accounting=1,
                independent_numeric_validation=1)


def task_plan(candidate, q, n):
    if (candidate, q) not in (('B2_p8_q2_2nd_r4_K4', 2), ('B2_p7_q16_2nd_r4_K4', 16)):
        raise ValueError('FIXED_CANDIDATE')
    return [dict(id=candidate+'_S2_SINGLE_pilot_'+str(i).zfill(3), candidate=candidate,
                 q=q, r=4, K=4, stratum='S2_SINGLE', phase='variance_load_pilot', replica=i)
            for i in range(n)]


def check_contract(c, n):
    """Contract-only invariant gate. Never grants, loads, samples or compiles."""
    if c['approved'] or c['runtime_authorization'] or c['C1_scientific_execution_authorized']:
        raise ValueError('CONTRACT_ONLY_NO_EXECUTION')
    if c['strata'] != ['S2_SINGLE'] or c['n_by_stratum'] != dict(S2_SINGLE=n, S2_MULTI=None, S4_PRESENT=None):
        raise ValueError('PILOT_STRATUM_OR_N')
    if c['frozen_scientific_budget'] != budget(c['q'], n):
        raise ValueError('CALL_BUDGET')
    tasks = task_plan(c['candidate'], c['q'], n)
    if c['validation_task_ids'] != [tasks[0]['id']]:
        raise ValueError('VALIDATION_PRESELECTED_NOT_OUTCOME_SELECTED')
    if c['production_authorized'] or c['cost_cache_enabled'] or c['valid_grant_issued']:
        raise ValueError('UNAUTHORIZED_EXTENSION')
    if c['numeric_model_approved'] or c['live_session_library_verified'] or c['fork_safety_verified']:
        raise ValueError('PENDING_GATE_MUST_NOT_BE_FORGED')
    if c['original_binding']['rank'] != 28 or c['existing_S0_n'] != 48:
        raise ValueError('FIXED_H8_ANCHOR')
    return True

"""Explanation-only saved-value analysis; imports have no scientific I/O.

The original mechanism core is unchanged. This module never reconstructs G,
averages trajectories, ranks candidates or computes statistical uncertainty.
"""
import copy
import math
import sys
import mechanism_analysis as original

SCHEMA = 'TRACK_A_MECHANISM_EXPLANATION_V2'
AXES = original.AXES
require = original.require
AnalysisError = original.AnalysisError
artificial_permit = original.artificial_permit
_authorized_production_permit = original._authorized_production_permit
finite = original.finite
exp_safe = original.exp_safe


def identity(row, method):
    return {k: row[k] for k in ('contrast_id', 'system', 'H4_stage', 'epsilon',
            'factor', 'bias_mode', 'PR_role', 'PR_cohort', 'PR_id', 'PF_id',
            'binding')} | dict(method=method,
                candidate_id=row[method + '_id'], axes=list(AXES))


def validate_identity(row):
    require(row['H4_stage'] == ('H4_PRE_D_P_VALIDATION' if row['system'] == 'H4'
            else 'NOT_APPLICABLE'), 'H4_FUTURE_DATASET_REQUIRES_NEW_CONTRACT')
    require(row['bias_mode'] == 'USUAL_SIGNED' and row['factor'] == '1',
            'FIXED_PRIMARY32_ACCOUNTING')
    role = row['PR_role']
    require(role in ('OLD_PR', 'NEW_PR', 'CONFIRMATION_N48'), 'PR_ROLE')
    cohort = ('FIXED_ANCHOR_SELECTION_AND_ESTIMATION' if row['system'] == 'H4'
            and role == 'OLD_PR' else 'CONFIRMATION_UNION_POST_SELECTION'
            if role == 'OLD_PR' else 'EXPLORATION_N8' if role == 'NEW_PR'
            else 'INDEPENDENT_FIXED_CONFIRMATION_N48')
    require(row['PR_cohort'] == cohort and row['cohorts_pooled'] is False,
            'COHORT_CLASSIFICATION_OR_MIXING')
    require(row['binding']['system'] == row['system'], 'SCIENCE_BINDING')
    for method in ('PR', 'PF'):
        require(row['sides'][method]['identity'] == identity(row, method),
                'EXPLANATION_IDENTITY_BINDING')


def normalization(data):
    """Stage identity and the original conditional shot-input policy.

    step_b is never passed as cumulative/upper normalization. The margin is
    empirical; its use does not establish an operator or population guarantee.
    """
    nom = finite(data['log_B_nominal'])
    margin = finite(data['log_B_margin'])
    require(nom >= 0 and margin >= 0, 'NORMALIZATION_LOG_RANGE')
    step_b = None if data.get('step_b') is None else finite(data['step_b'], positive=True)
    steps = data['step_count']
    if data['law'] == 'FINITE_RTE':
        require(type(steps) is int and steps > 0 and step_b is not None
                and step_b >= 1, 'FINITE_RTE_STEP_SCHEMA')
        require(abs(steps * math.log(step_b) - nom) <= 1e-12,
                'STEP_CUMULATIVE_LOG_MISMATCH')
    else:
        require(data['law'] == 'DETERMINISTIC_PF' and
                (steps is None or (type(steps) is int and steps == 0)) and
                (step_b is None or step_b == 1) and nom == 0,
                'DETERMINISTIC_NORMALIZATION_SCHEMA')
        # These optional RTE-only fields stay null on PF overlays. No stored b=1
        # is invented; the recorded log_B=0 and deterministic SOURCE define the law.
    # factor1 is the fixed accounting endpoint; no scientific value is filled in.
    require(data['factor'] == '1', 'NORMALIZATION_FACTOR_BINDING')
    upper = nom + 1.0 * margin
    require(math.isfinite(upper), 'NORMALIZATION_UPPER_LOG_NONFINITE')
    route = data['upper_route']
    if route == 'EXP_OF_LOG_UPPER':
        require(data.get('normalization_upper_saved') is None,
                'UNEXPECTED_SAVED_UPPER_ON_EXP_ROUTE')
        B = math.exp(upper) if upper <= math.log(sys.float_info.max) else None
    else:
        require(route == 'SAVED_UPPER_BINARY64', 'NORMALIZATION_UPPER_ROUTE')
        B = finite(data['normalization_upper_saved'], positive=True)
        require(B >= 1 and math.log(B) == upper, 'SAVED_UPPER_LOG_BINDING')
    return dict(step_b=step_b, step_count=steps, log_B_nominal=nom,
                log_B_margin=margin, log_B_upper=upper, B_upper=B,
                upper_route=route, certification='EMPIRICAL_CONDITIONAL_ONLY')


def explain_side(side, epsilon):
    eligible = side.get('eligible')
    require(eligible is None or type(eligible) is bool, 'ELIGIBILITY_TYPE')
    if eligible is None:
        return dict(status='UNRESOLVED', reason='ELIGIBILITY_UNDETERMINED')
    if eligible is False:
        return dict(status='INELIGIBLE')
    fields = ('normalization', 'signed_bias', 'allowance', 'headroom',
              'alpha_axis', 'log_shot_bound', 'integer_status', 'shots')
    missing = [k for k in fields if side.get(k) is None]
    if missing:
        return dict(status='EXPLANATORY_INPUT_MISSING', missing_fields=missing)
    for key in ('signed_bias', 'allowance', 'headroom', 'log_shot_bound',
                'integer_status', 'shots'):
        require(set(side[key]) == set(AXES), 'EXPLANATION_AXIS_SCHEMA')
        if key != 'shots' and any(side[key][a] is None for a in AXES):
            return dict(status='EXPLANATORY_INPUT_MISSING', missing_fields=[key])
    norm = normalization(side['normalization'])
    alpha = finite(side['alpha_axis'], positive=True)
    require(alpha < 1, 'ALPHA_RANGE')
    eps = finite(epsilon, positive=True)
    result = dict(status='EXPLANATION_COMPLETE', normalization=norm, axes={})
    for axis in AXES:
        signed = finite(side['signed_bias'][axis])
        allowance = finite(side['allowance'][axis])
        require(allowance >= 0, 'NEGATIVE_ALLOWANCE')
        h = finite(side['headroom'][axis], positive=True)
        require(abs(h - (eps / math.sqrt(2) - abs(signed) - allowance)) <= 1e-12,
                'HEADROOM_MISMATCH')
        log_n = (math.log(2) + 2 * norm['log_B_upper'] - 2 * math.log(h)
                 + math.log(math.log(2 / alpha)))
        require(math.isfinite(log_n), 'LOG_BOUND_NONFINITE')
        require(abs(log_n - finite(side['log_shot_bound'][axis])) <= 1e-12,
                'LOG_BOUND_MISMATCH')
        saved_status = side['integer_status'][axis]
        log_only = log_n >= math.log(2**52)
        require(saved_status == ('LOG_DOMAIN_ONLY' if log_only else
                                 'BINARY64_DIAGNOSTIC'), 'INTEGER_ROUTE_MISMATCH')
        saved_n = side['shots'][axis]
        bound = kappa = log_kappa = None
        if log_only:
            require(saved_n is None, 'LOG_ONLY_INTEGER_NOT_SAVED')
        else:
            original.integer(saved_n)
            try:
                bound = 2 * norm['B_upper']**2 / h**2 * math.log(2 / alpha)
            except (TypeError, OverflowError, ZeroDivisionError):
                bound = math.inf
            if not math.isfinite(bound) or bound == 0:
                bound = math.exp(log_n)
            require(math.isfinite(bound) and bound > 0, 'BOUND_OVERFLOW_UNDERFLOW')
            # A consistency check only: saved_n is never replaced or reoptimized.
            require(saved_n == max(1, math.ceil(bound)), 'SAVED_ROUNDING_CONSISTENCY')
            kappa = saved_n / bound
            log_kappa = math.log(saved_n) - log_n
        result['axes'][axis] = dict(headroom=h, signed_bias=signed,
            allowance=allowance, alpha_axis=alpha, log_Nstar=log_n,
            Nstar=bound, saved_shots=saved_n, kappa=kappa, log_kappa=log_kappa,
            integer_status=saved_status)
    return result


def explain(row):
    sides = {}
    for method in ('PR', 'PF'):
        try:
            sides[method] = explain_side(row['sides'][method], row['epsilon'])
        except (AnalysisError, OverflowError) as exc:
            sides[method] = dict(status='UNRESOLVED', reason=str(exc))
    states = [s['status'] for s in sides.values()]
    # A missing side must not hide a numeric mismatch on the other side.
    status = ('UNRESOLVED' if 'UNRESOLVED' in states else 'INELIGIBLE'
              if 'INELIGIBLE' in states else 'EXPLANATORY_INPUT_MISSING'
              if 'EXPLANATORY_INPUT_MISSING' in states else 'EXPLANATION_COMPLETE')
    ratios = {}
    if status == 'EXPLANATION_COMPLETE':
        pnorm, dnorm = [sides[m]['normalization'] for m in ('PR', 'PF')]
        for axis in AXES:
            p, d = [sides[m]['axes'][axis] for m in ('PR', 'PF')]
            logs = dict(nominal_normalization_squared=2*(pnorm['log_B_nominal']-dnorm['log_B_nominal']),
                normalization_margin_squared=2*(pnorm['log_B_margin']-dnorm['log_B_margin']),
                normalization_upper_squared=2*(pnorm['log_B_upper']-dnorm['log_B_upper']),
                headroom_squared=2*(math.log(d['headroom'])-math.log(p['headroom'])),
                alpha_log=math.log(math.log(2/p['alpha_axis']))-math.log(math.log(2/d['alpha_axis'])),
                continuous_shot=p['log_Nstar']-d['log_Nstar'])
            if p['log_kappa'] is not None and d['log_kappa'] is not None:
                logs['rounding_factor'] = p['log_kappa']-d['log_kappa']
            ratios[axis] = dict(log_factors=logs,
                factors={k: exp_safe(v) for k, v in logs.items()})
    return dict(status=status, sides=sides, ratios=ratios)


def analyze_bundle(bundle, saved_core, permit):
    require(bundle['schema'] == SCHEMA and bundle['kind'] == permit.kind,
            'SCHEMA_OR_DATA_KIND')
    rows = bundle['comparisons']
    ids = [r['contrast_id'] for r in rows]
    require(ids == bundle['target_ids'] == [r['contrast_id'] for r in saved_core]
            and len(ids) == len(set(ids)), 'FIXED_CORE_TARGET_JOIN')
    output = []
    for row, core in zip(rows, saved_core):
        original.check_permission(row, permit)
        validate_identity(row)
        require(core['core_status'] == 'CORE_COMPLETE' and
                core['saved_G_exact_match'] is True and
                core['exact_decompositions_match'] is True and
                core['system'] == row['system'] and core['H4_stage'] == row['H4_stage'],
                'CORE_REVIEW_REQUIRED')
        output.append(dict(contrast_id=row['contrast_id'], system=row['system'],
            core_reused_without_recalculation=True, explanation=explain(row)))
    return output

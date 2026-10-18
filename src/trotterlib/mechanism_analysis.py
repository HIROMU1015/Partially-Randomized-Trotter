"""Exact, permission-gated saved-value mechanism analysis; stdlib only.

No scientific I/O or numerical execution at import. Preparation must not invoke
analyze on SAVED_REAL_VALUES. Statistics are copied, never recomputed.
"""
from fractions import Fraction
import math

AXES = ('cosine', 'sine')
SCHEMA = 'TRACK_A_MECHANISM_V1'
_REAL_TOKEN = object()

class AnalysisError(ValueError):
    pass

def require(ok, code):
    if not ok:
        raise AnalysisError(code)

class Permit:
    def __init__(self, kind, token=None):
        require(kind == 'ARTIFICIAL' or (kind == 'SAVED_REAL_VALUES' and token is _REAL_TOKEN),
                'NUMERICAL_AUTHORIZATION_REQUIRED')
        self.kind = kind

def artificial_permit():
    return Permit('ARTIFICIAL')

def _authorized_production_permit():
    # Called only by the launcher after all external authorization gates.
    return Permit('SAVED_REAL_VALUES', _REAL_TOKEN)

def exact(value):
    require(type(value) in (str, int), 'EXACT_INTEGER_OR_FRACTION_REQUIRED')
    try:
        out = Fraction(value)
    except (ValueError, ZeroDivisionError, OverflowError):
        raise AnalysisError('INVALID_EXACT_VALUE') from None
    require(out > 0, 'POSITIVE_EXACT_VALUE_REQUIRED')
    return out

def integer(value):
    require(type(value) is int and value > 0, 'SAVED_POSITIVE_INTEGER_SHOT_REQUIRED')
    return value

def check_permission(row, permit):
    require(isinstance(permit, Permit), 'NUMERICAL_AUTHORIZATION_REQUIRED')
    require((permit.kind == 'ARTIFICIAL') == (row.get('artificial') is True), 'DATA_KIND_PERMIT_MISMATCH')

def validate_identity(row):
    require(row['PR']['binding'] == row['PF']['binding'] == row['binding'], 'SOURCE_OR_BINDING_MISMATCH')
    require(row['system'] == row['binding']['system'], 'SOURCE_OR_BINDING_MISMATCH')
    require(row['PR']['candidate_id'] == row['PR_id'] and row['PF']['candidate_id'] == row['PF_id'],
            'CANDIDATE_BINDING_MISMATCH')
    require(row['bias_mode'] == 'USUAL_SIGNED' and row['factor'] == '1', 'PRIMARY_32_ACCOUNTING_ONLY')
    require(row['H4_stage'] == ('H4_PRE_D_P_VALIDATION' if row['system'] == 'H4' else 'NOT_APPLICABLE'),
            'H4_FUTURE_DATASET_REQUIRES_NEW_CONTRACT')
    expected = ('FIXED_ANCHOR_SELECTION_AND_ESTIMATION' if row['system'] == 'H4' and row['PR_role'] == 'OLD_PR'
                else 'CONFIRMATION_UNION_POST_SELECTION' if row['PR_role'] == 'OLD_PR'
                else 'EXPLORATION_N8' if row['PR_role'] == 'NEW_PR'
                else 'INDEPENDENT_FIXED_CONFIRMATION_N48')
    require(row['PR']['cohort'] == expected, 'COHORT_CLASSIFICATION_MISMATCH')
    n = 32 if row['PR_role'] == 'OLD_PR' else 8 if row['PR_role'] == 'NEW_PR' else 48
    require(row['PR']['n'] == n and row.get('cohorts_pooled') is False, 'COHORT_MIXING_FORBIDDEN')
    require(row['PF']['cost_class'] in ('DIRECT_MEASURED', 'VALIDATED_MODEL_ESTIMATE'), 'PF_COST_PROVENANCE')
    require(row['PF'].get('source_evidence'), 'PF_SOURCE_EVIDENCE_MISSING')

def paired_costs(side):
    pairs = side.get('trajectories')
    if pairs is None:
        return None
    n = side['n']
    require(len(pairs) == n, 'EXACT_COHORT_COVERAGE')
    seen, axes_values = set(), {a: [] for a in AXES}
    for pair in pairs:
        replica = pair['replica']
        require(type(replica) is int and 0 <= replica < n and replica not in seen, 'DUPLICATE_OR_INVALID_REPLICA')
        seen.add(replica)
        require(pair['cohort'] == side['cohort'] and pair['candidate_id'] == side['candidate_id'], 'COHORT_MIXING_FORBIDDEN')
        require(set(pair['axes']) == set(AXES), 'PAIRED_AXES_REQUIRED')
        for axis in AXES:
            record = pair['axes'][axis]
            require(record['pair_id'] == pair['pair_id'], 'PAIRED_TRAJECTORY_IDENTITY')
            require(record.get('seed') == pair.get('seed') and record.get('event_digest') == pair.get('event_digest'),
                    'PAIRED_SEED_EVENT_IDENTITY')
            price=exact(record['RZ']);require(price.denominator==1,'INTEGER_MEASURED_RZ_REQUIRED')
            axes_values[axis].append(price)
    require(seen == set(range(n)), 'EXACT_COHORT_COVERAGE')
    return {a: sum(axes_values[a], Fraction()) / n for a in AXES}

def finite(value, *, positive=False):
    require(not isinstance(value, bool), 'FINITE_BINARY64_REQUIRED')
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise AnalysisError('FINITE_BINARY64_REQUIRED') from None
    require(math.isfinite(result) and (not positive or result > 0), 'FINITE_BINARY64_REQUIRED')
    return result

def exp_safe(value):
    try:
        out=math.exp(value)
    except OverflowError:
        return None
    return out if math.isfinite(out) and out > 0 else None

def explanation_side(side, epsilon):
    data = side.get('explanation')
    needed = ('log_B', 'signed_bias', 'allowance', 'headroom', 'alpha_axis')
    if not data or any(data.get(k) is None for k in needed):
        return dict(status='EXPLANATORY_INPUT_MISSING')
    if any(any(data[k].get(a) is None for a in AXES) for k in ('signed_bias', 'allowance', 'headroom')):
        return dict(status='EXPLANATORY_INPUT_MISSING')
    log_b = finite(data['log_B']); require(log_b >= 0, 'NORMALIZATION_LOG_RANGE')
    alpha = finite(data['alpha_axis'], positive=True); require(alpha < 1, 'ALPHA_RANGE')
    B = finite(data['B'], positive=True) if data.get('B') is not None else None
    if B is not None:
        require(B >= 1 and math.log(B) == log_b, 'NORMALIZATION_LOG_BINDING')
    else:
        try:
            B = math.exp(log_b)
        except OverflowError:
            B = None
    output = dict(status='EXPLANATION_COMPLETE', log_B=log_b, B=B,
                  B_evidence='SAVED_BINARY64' if data.get('B') is not None else 'DERIVED_FROM_SAVED_LOG_NO_BIT_IDENTITY_CLAIM', axes={})
    for axis in AXES:
        signed = finite(data['signed_bias'][axis]); allowance = finite(data['allowance'][axis])
        h = finite(data['headroom'][axis], positive=True)
        require(allowance >= 0, 'NEGATIVE_ALLOWANCE')
        predicted = epsilon / math.sqrt(2) - abs(signed) - allowance
        require(abs(h - predicted) <= 1e-12, 'HEADROOM_MISMATCH')
        log_n = math.log(2) + 2 * log_b - 2 * math.log(h) + math.log(math.log(2 / alpha))
        require(math.isfinite(log_n),'LOG_BOUND_NONFINITE')
        saved_log = (data.get('log_shot_bound') or {}).get(axis)
        if saved_log is not None:
            require(abs(log_n - finite(saved_log)) <= 1e-12, 'LOG_BOUND_MISMATCH')
        integer_status = (data.get('integer_status') or {}).get(axis)
        log_only = log_n >= math.log(2**52) or integer_status == 'LOG_DOMAIN_ONLY'
        bound = None
        if not log_only:
            try:
                bound = 2 * B**2 / h**2 * math.log(2 / alpha)
            except (TypeError, OverflowError, ZeroDivisionError):
                bound = math.inf
            if not math.isfinite(bound) or bound == 0:
                try:
                    bound = math.exp(log_n)
                except OverflowError:
                    bound = None
            require(bound is not None and math.isfinite(bound) and bound > 0, 'BOUND_OVERFLOW_UNDERFLOW')
        saved_n = integer(side['shots'][axis])
        if bound is not None:
            require((bound <= saved_n < bound + 1) or (bound < 1 and saved_n == 1), 'SAVED_ROUNDING_CONSISTENCY')
        output['axes'][axis] = dict(signed_bias=signed, allowance=allowance, headroom=h,
            alpha_axis=alpha, log_Nstar=log_n, Nstar=bound,
            kappa=None if bound is None else saved_n / bound,
            log_kappa=math.log(saved_n)-log_n,
            integer_status='LOG_DOMAIN_ONLY' if log_only else 'SAVED_INTEGER_UNCHANGED',
            B_binary64_hex=None if B is None else B.hex())
    return output

def explain(row):
    result = {}
    for method in ('PR', 'PF'):
        try:
            result[method] = explanation_side(row[method], finite(row['epsilon'], positive=True))
        except AnalysisError as exc:
            result[method] = dict(status='UNRESOLVED', reason=str(exc))
    if any(x['status'] == 'EXPLANATORY_INPUT_MISSING' for x in result.values()):
        status = 'EXPLANATORY_INPUT_MISSING'
    elif any(x['status'] != 'EXPLANATION_COMPLETE' for x in result.values()):
        status = 'UNRESOLVED'
    else:
        status = 'EXPLANATION_COMPLETE'
        for axis in AXES:
            p, d = result['PR']['axes'][axis], result['PF']['axes'][axis]
            delta = p['log_Nstar'] - d['log_Nstar']
            result.setdefault('ratios', {})[axis] = dict(log_continuous_shot_ratio=delta,
                log_rounding_factor_ratio=p['log_kappa']-d['log_kappa'])
            try:
                result['ratios'][axis].update(continuous_shot_ratio=exp_safe(delta),
                    rounding_factor_ratio=exp_safe(p['log_kappa']-d['log_kappa']),
                    normalization_ratio_squared=exp_safe(2*(result['PR']['log_B']-result['PF']['log_B'])),
                    headroom_ratio_squared=exp_safe(2*(math.log(d['headroom'])-math.log(p['headroom']))),
                    alpha_log_ratio=math.log(2/p['alpha_axis'])/math.log(2/d['alpha_axis']))
            except OverflowError:
                result['ratios'][axis].update(continuous_shot_ratio=None, rounding_factor_ratio=None)
    return dict(status=status, sides=result)

def analyze_contrast(row, permit):
    check_permission(row, permit)
    validate_identity(row)
    output = dict(contrast_id=row['contrast_id'], system=row['system'], H4_stage=row['H4_stage'],
        core_status=None, explanation_status='EXPLANATORY_INPUT_MISSING', saved_SE=row.get('saved_SE'),
        new_statistics_computed=False, population_conclusion='UNRESOLVED', PF_cost_class=row['PF']['cost_class'])
    require(all(row[m].get('eligible') is None or type(row[m].get('eligible')) is bool for m in ('PR','PF')), 'SAVED_ELIGIBILITY_TYPE')
    if any(row[m].get('eligible') is False for m in ('PR', 'PF')):
        output['core_status']='INELIGIBLE'; return output
    if any(row[m].get('eligible') is None for m in ('PR', 'PF')):
        output['core_status']='UNRESOLVED'; return output
    if any(row[m].get('saved_G_exact') is None for m in ('PR', 'PF')):
        output['core_status']='DISPLAY_ONLY_SOURCE' if any(row[m].get('saved_G_display') is not None for m in ('PR','PF')) else 'CORE_INPUT_MISSING'
        return output
    if any(row[m].get('shots') is None or any(row[m]['shots'].get(a) is None for a in AXES) for m in ('PR','PF')):
        output['core_status']='CORE_INPUT_MISSING'; return output
    cp = paired_costs(row['PR'])
    cd_raw = row['PF'].get('axis_costs')
    if cp is None or cd_raw is None or any(cd_raw.get(a) is None for a in AXES):
        output['core_status']='CORE_INPUT_MISSING'; return output
    cd = {a: exact(cd_raw[a]) for a in AXES}
    shots = {m: {a: integer(row[m]['shots'][a]) for a in AXES} for m in ('PR','PF')}
    totals = {m: sum(shots[m].values()) for m in shots}
    for m in shots:
        require(row[m].get('saved_N_total') in (None, totals[m]), 'SAVED_SHOT_TOTAL_MISMATCH')
    gp=sum(shots['PR'][a]*cp[a] for a in AXES); gd=sum(shots['PF'][a]*cd[a] for a in AXES)
    require(gp == exact(row['PR']['saved_G_exact']) and gd == exact(row['PF']['saved_G_exact']), 'SAVED_EXACT_G_MISMATCH')
    rg=gp/gd; rn=Fraction(totals['PR'],totals['PF']); cepr=gp/totals['PR']; cepf=gd/totals['PF']; rc=cepr/cepf
    axis_rows=[]
    for a in AXES:
        weight=shots['PF'][a]*cd[a]/gd; nratio=Fraction(shots['PR'][a],shots['PF'][a]); cratio=cp[a]/cd[a]
        axis_rows.append(dict(axis=a,C_PR=str(cp[a]),C_PF=str(cd[a]),N_PR=shots['PR'][a],N_PF=shots['PF'][a],
                             R_N_axis=str(nratio),R_C_axis=str(cratio),w_PF=str(weight),weighted_term=str(weight*nratio*cratio)))
    require(rg == rn*rc == sum(Fraction(a['weighted_term']) for a in axis_rows), 'EXACT_DECOMPOSITION_MISMATCH')
    reduction=100*(1-rg)
    if row.get('saved_reduction_percent_exact') is not None:
        require(reduction == Fraction(row['saved_reduction_percent_exact']), 'SAVED_REDUCTION_MISMATCH')
    explanation=explain(row)
    output.update(core_status='CORE_COMPLETE',G_PR=str(gp),G_PF=str(gd),N_total_PR=totals['PR'],N_total_PF=totals['PF'],
        C_eff_PR=str(cepr),C_eff_PF=str(cepf),R_G=str(rg),R_N=str(rn),R_Ceff=str(rc),
        reduction_percent_exact=str(reduction),saved_G_exact_match=True,exact_decompositions_match=True,
        axes=axis_rows,explanation_status=explanation['status'],explanation=explanation)
    return output

def analyze_bundle(bundle, permit):
    require(bundle.get('schema') == SCHEMA and bundle.get('kind') == permit.kind, 'SCHEMA_OR_DATA_KIND')
    rows=bundle['comparisons']; ids=[r['contrast_id'] for r in rows]
    require(len(ids)==len(set(ids)) and ids==bundle['target_ids'], 'EXACT_REGISTERED_TARGETS')
    return [analyze_contrast(row,permit) for row in rows]

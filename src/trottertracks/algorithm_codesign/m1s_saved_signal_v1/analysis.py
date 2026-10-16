"""Future V1 analysis; never imported by candidate/status/static entrypoints."""
from fractions import Fraction as F
from .bounds import Bounds, ceil_fraction, interval_text, rational_text
from .spec import EVIDENCE, FORECAST, INPUT, keys, contrast_keys


def saved_binary64(value):
    # JSON's round-trip float, not the rational value of the decimal spelling.
    if type(value) is int:
        return F(value)
    if type(value) is not float:
        raise ValueError('saved numeric field missing or invalid')
    return F(*value.as_integer_ratio())  # rejects NaN and infinity


def row_key(row):
    return row['split'], row['R'], row['delta'], row['T'], row['method']


def skeleton():
    return [dict(zip(['split', 'R', 'delta', 'T', 'method'], k),
                 status='NOT_EVALUATED', forecast=None) for k in keys()]


def evaluate(raw, contract, permit, result, checkpoint):
    from .control import require_permit
    require_permit(permit)
    if raw.get('schema') != INPUT['schema'] or raw.get('status') != INPUT['status']:
        raise ValueError('saved result schema/status')
    source_rows = {row_key(r): r for r in raw['rows']}
    if len(raw['rows']) != 108 or set(source_rows) != set(keys()):
        raise ValueError('saved fixed grid duplicates/missing/extra')
    science = contract['science']
    B = Bounds(science['arithmetic_bits'], contract['resources'])
    gamma, alpha = F(science['gamma_rad']), F(science['alpha_total'])
    if not 0 < alpha < 1 or gamma <= 0:
        raise ValueError('scientific parameter domain')
    pi = B.pi()
    if gamma >= pi[0]/2:
        raise ValueError('gamma<pi/2 not established at adopted arithmetic resolution')
    alpha_axis = alpha / (144 if science['confidence_scope'] == 'FAMILY_ALL_72_METHOD_ROWS' else 2)
    log = B.log_above_one(2/alpha_axis)
    result['arithmetic'] = {'pi_interval': interval_text(pi),
                            'log_2_over_alpha_axis_interval': interval_text(log),
                            'alpha_axis': rational_text(alpha_axis),
                            'bits': science['arithmetic_bits']}
    for out in result['rows']:
        saved = source_rows[row_key(out)]
        out['saved_status'] = saved['status']
        out['saved_phase_status'] = saved['phase_status']
        out['saved_evidence'] = saved['evidence']
        if out['T'] == '0':
            out['status'] = 'SAVED_ZERO_CONTROL_NOT_FORECAST'
            checkpoint(result)
            continue
        if (saved['phase_status'] != 'RESOLVED_PRINCIPAL_PHASE'
                or saved['phase_interpretation'] != 'TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H'
                or saved['evidence'] != EVIDENCE):
            out['status'] = 'TASK_OR_EVIDENCE_UNRESOLVED'
            checkpoint(result)
            continue
        fields = ['visibility', 'visibility_error_radius', 'phase_bias_rad', 'phase_error_radius_rad']
        if any(saved.get(k) is None for k in fields):
            out['status'] = 'TASK_OR_EVIDENCE_UNRESOLVED'
            checkpoint(result)
            continue
        v, rv, b, rb = [saved_binary64(saved[k]) for k in fields]
        if rv < 0 or rb < 0:
            raise ValueError('negative saved radius')
        A = max(F(0), v-rv)
        if A > 1:
            raise ValueError('signal lower bound exceeds bounded measurement domain')
        unclip = abs(b) + rb
        bias = (min(pi[0], unclip), min(pi[1], unclip))
        q = (gamma-bias[1], gamma-bias[0])
        out['bounds'] = {'visibility_lower': rational_text(A),
                         'bias_upper_interval_rad': interval_text(bias),
                         'statistical_slack_interval_rad': interval_text(q),
                         'model_drift_added_again': False}
        if q[1] <= 0:
            out['status'] = 'PHASE_BUDGET_NOT_CERTIFIED_UNDER_SELECTED_ROUTE'
        elif q[0] <= 0:
            out['status'] = 'ARITHMETIC_RESOLUTION_UNDETERMINED'
        elif A <= 0:
            out['status'] = 'TASK_OR_EVIDENCE_UNRESOLVED'
        else:
            # sin is increasing on (0,pi/2); q is bounded by gamma.
            s_lo = B.sin_positive(q[0])[0]
            s_hi = B.sin_positive(q[1])[1]
            if s_lo <= 0:
                out['status'] = 'ARITHMETIC_RESOLUTION_UNDETERMINED'
            else:
                denominator = (B.check(A*A*s_lo*s_lo), B.check(A*A*s_hi*s_hi))
                formula = (B.check(4*log[0]/denominator[1]), B.check(4*log[1]/denominator[0]))
                n = ceil_fraction(formula[1])
                if n.bit_length() > contract['resources']['max_shot_integer_bits']:
                    raise ValueError('shot integer resource cap')
                out['status'] = 'CONDITIONAL_SUFFICIENT_DESIGN_AVAILABLE'
                out['forecast'] = {'kind': FORECAST, 'N_X': n, 'N_Y': n,
                                   'N_total': 2*n, 'formula_interval': interval_text(formula),
                                   'integerized_formula_interval': [ceil_fraction(x) for x in formula],
                                   'effective_margin_interval': interval_text((A*s_lo, A*s_hi)),
                                   'rounding': 'CEILING_OF_RATIONAL_UPPER_ENCLOSURE',
                                   'minimal_required_shots': None}
        result['arithmetic']['bound_steps'] = B.steps
        checkpoint(result)

    old = {}
    for c in raw['comparisons']:
        k = (c['split'], c['R'], c['delta'], c['T'], c['comparator'])
        if k in old:
            raise ValueError('duplicate saved contrast')
        old[k] = c
    expected = set(contrast_keys())
    if set(old) != expected:
        raise ValueError('saved contrast grid')
    new = {row_key(r): r for r in result['rows']}
    for i, k in enumerate(contrast_keys()):
        split, R, delta, T, comparator = k
        shared, other = [new[(split, R, delta, T, m)] for m in ['shared_pm', comparator]]
        contrast = dict(zip(['split', 'R', 'delta', 'T', 'comparator'], k),
                        saved_visibility_contrast_status=old[k]['contrast_status'],
                        status='NOT_COMPARABLE_UNDER_SELECTED_ROUTE',
                        sufficient_design_ratio=None, symbolic_cost_break_even=None,
                        scientific_GO=None, native_costs=None,
                        scientific_gain_claim='NOT_ASSESSED')
        if shared['forecast'] is not None and other['forecast'] is not None:
            ns, nc = [r['forecast']['N_total'] for r in [shared, other]]
            contrast['status'] = 'COMMON_SUFFICIENT_DESIGNS_ONLY'
            contrast['sufficient_design_relation'] = 'SHARED_SMALLER' if ns < nc else ('EQUAL' if ns == nc else 'SHARED_LARGER')
            contrast['sufficient_design_ratio'] = rational_text(F(ns, nc))
            contrast['symbolic_cost_break_even'] = {
                'equal_axis_cost_no_fixed_cost_only': '(C_sh+P_sh)/(C_cmp+P_cmp) < threshold',
                'threshold': rational_text(F(nc, ns)),
                'not_optimal_or_native_resource_claim': True}
        if old[k]['contrast_status'] == 'UNRESOLVED_ABSOLUTE_CONTRAST':
            contrast['scientific_gain_claim'] = 'ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE'
        result['comparisons'][i] = contrast
        checkpoint(result)
    result['status'] = 'CONDITIONAL_DESIGNS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT'
    return result

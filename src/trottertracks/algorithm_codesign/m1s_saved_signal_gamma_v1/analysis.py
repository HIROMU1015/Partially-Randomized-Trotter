"""Future saved-science analysis. V0 tests use only separate pure scalar kernels."""
from fractions import Fraction as F
from .bounds import Bounds, interval_text, rational_text
from .boundary import compare
from .spec import EVIDENCE, INPUT, keys, contrast_keys


def saved_binary64(value):
    if type(value) is int:
        return F(value)
    if type(value) is not float:
        raise ValueError('saved numeric field missing or invalid')
    return F(*value.as_integer_ratio())


def row_key(row):
    return row['split'], row['R'], row['delta'], row['T'], row['method']


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
    alpha = F(science['alpha_total'])
    if not 0 < alpha < 1:
        raise ValueError('alpha outside (0,1)')
    pi = B.pi()
    alpha_axis = alpha/(144 if science['confidence_scope'] == 'FAMILY_ALL_72_METHOD_ROWS' else 2)
    log = B.log_above_one(2/alpha_axis)
    result['arithmetic'] = {'pi_interval': interval_text(pi), 'K_interval': interval_text((4*log[0],4*log[1])),
                            'alpha_axis': rational_text(alpha_axis), 'bits': science['arithmetic_bits'],
                            'K_use': 'symbolic N_axis(gamma)=ceil(K/M_m(gamma)^2); no fixed gamma evaluated'}
    usable = {}
    trig = {}
    for out in result['rows']:
        saved = source_rows[row_key(out)]
        out.update(saved_status=saved['status'], saved_phase_status=saved['phase_status'],
                   saved_evidence=saved['evidence'])
        if out['T'] == '0':
            out['status'] = 'SAVED_ZERO_CONTROL_NOT_REANALYZED'
        elif (saved['phase_status'] != 'RESOLVED_PRINCIPAL_PHASE'
              or saved['phase_interpretation'] != 'TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H'
              or saved['evidence'] != EVIDENCE):
            out['status'] = 'TASK_OR_EVIDENCE_UNRESOLVED'
        else:
            fields = ['visibility', 'visibility_error_radius', 'phase_bias_rad', 'phase_error_radius_rad']
            if any(saved.get(k) is None for k in fields):
                out['status'] = 'TASK_OR_EVIDENCE_UNRESOLVED'
            else:
                v, rv, b, rb = [saved_binary64(saved[k]) for k in fields]
                if rv < 0 or rb < 0:
                    raise ValueError('negative saved radius')
                A, unclip = max(F(0),v-rv), abs(b)+rb
                if A > 1:
                    raise ValueError('visibility lower bound exceeds bounded measurement domain')
                out['bounds'] = {'A_lower': rational_text(A), 'unclipped_bias_rad': rational_text(unclip),
                                 'bias_definition': 'min(pi,unclipped_bias_rad)',
                                 'model_drift_added_again': False}
                if unclip >= pi[1]/2:
                    out['status'] = 'NO_POSITIVE_PHASE_DOMAIN'
                elif unclip >= pi[0]/2:
                    out['status'] = 'ARITHMETIC_DOMAIN_UNRESOLVED'
                elif A <= 0:
                    out['status'] = 'TASK_OR_EVIDENCE_UNRESOLVED'
                else:
                    out['status'] = 'READY_FOR_GAMMA_DOMAIN'
                    out['domain'] = {'lower_open_rad': rational_text(unclip), 'upper_open': 'pi/2',
                                     'positive_margin': 'gamma>b_upper and A_lower>0'}
                    usable[row_key(out)] = A, unclip
                    if unclip not in trig:
                        trig[unclip] = B.sin_positive(unclip), B.cos_positive(unclip)
        result['arithmetic']['bound_steps'] = B.steps
        checkpoint(result)
    old = {}
    for c in raw['comparisons']:
        k = c['split'], c['R'], c['delta'], c['T'], c['comparator']
        if k in old:
            raise ValueError('duplicate saved contrast')
        old[k] = c
    if set(old) != set(contrast_keys()):
        raise ValueError('saved comparison grid missing/extra')
    rows = {row_key(r):r for r in result['rows']}
    for i,k in enumerate(contrast_keys()):
        split,R,delta,T,comparator = k
        shared_key,other_key = [(split,R,delta,T,m) for m in ['shared_pm',comparator]]
        item = result['comparisons'][i]
        item.update(saved_visibility_contrast_status=old[k]['contrast_status'],
                    scientific_gain_claim=('ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE'
                                           if old[k]['contrast_status']=='UNRESOLVED_ABSOLUTE_CONTRAST' else 'NOT_ASSESSED'))
        if shared_key not in usable or other_key not in usable:
            statuses = {rows[x]['status'] for x in [shared_key,other_key]}
            item['status'] = ('NO_COMMON_PHASE_DOMAIN' if 'NO_POSITIVE_PHASE_DOMAIN' in statuses else
                              'ARITHMETIC_UNRESOLVED' if 'ARITHMETIC_DOMAIN_UNRESOLVED' in statuses else
                              'EVIDENCE_UNRESOLVED_NOT_COMPARABLE')
        else:
            a,u = usable[shared_key]
            c,v = usable[other_key]
            item.update(compare(a,u,c,v,B,pi,trig))
        result['arithmetic']['bound_steps'] = B.steps
        checkpoint(result)
    result['status'] = 'GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT'
    return result

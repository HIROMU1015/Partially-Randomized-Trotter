"""Replace six saved unit prices; never calculate bias, shots, PR means or SE.

All arithmetic uses Fraction. The old normal/noncancellation stages are inputs,
not callable dependencies. A fresh runtime permit or explicit synthetic permit
is required. This module imports only the Python standard library.
"""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json

EPSILONS = (.05, .01, .005, .001, .0005)
FACTORS = (1., 10.)
MODES = ('USUAL_SIGNED_SAVED', 'EMPIRICAL_NONCANCELLATION')
AXES = ('cosine', 'sine')
CHANNEL = dict(cosine='real', sine='imag')
REPLACEMENTS = {'B0I_p15_q4_4th': 668636, 'B0I_p11_q1_2nd': 42268,
                'B0_p15_q4_2nd': 223940}
MODEL = '8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
SIX_SOURCE = '41e466e13f32e9224bb37790cc16f0ff6621c399'
OLD_SIX_SOURCE = '352581103b7dbd4fa58e28af7a6ca65570d36e8e'
SIX_RESULT = 'ce6cd8ead5c0311ec72165c04869e7df4050b75b'
USUAL_SOURCE = '931755c4c84aa0751dbde7ddf84cb5187b5172ae'
NONCANCEL_SOURCE = 'ac932974a0f250be0df5720e1b6dad06f9ff9535'
_PERMIT = object()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class Permit:
    def __init__(self, *, synthetic=False, _token=None):
        require(synthetic is True or _token is _PERMIT, 'NEW_ONE_SHOT_GRANT_REQUIRED')
        self.synthetic = synthetic


def index(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, 'DUPLICATE_KEY:' + str(k))
        result[k] = row
    return result


def row_key(row):
    # The frozen B0+I accounting stores factors as "1.0"/"10.0";
    # the completed cost stages store numbers. This is an ID normalization,
    # never a recomputation or modification of the saved accounting value.
    return row['cell_id'], row['epsilon'], float(row['factor'])


def minimum(rows):
    """Exact old-stage tie rule, with unresolved and empty domains distinct."""
    unresolved = sorted(r['cell_id'] for r in rows if r['eligible'] is None)
    if unresolved:
        return dict(status='UNRESOLVED_FULL_DOMAIN', best=None, unresolved=unresolved)
    valid = [r for r in rows if r['eligible'] is True]
    require(all(r.get('G_exact') is not None for r in valid), 'ELIGIBLE_COST_MISSING')
    if not valid:
        return dict(status='NO_ELIGIBLE_CANDIDATE', best=None, exact_ties=[])
    best = min(valid, key=lambda r: (F(r['G_exact']), r['cell_id']))
    return dict(status='CONDITIONAL_POINT_MINIMUM', best=best,
                exact_ties=sorted(r['cell_id'] for r in valid if F(r['G_exact']) == F(best['G_exact'])))


def reduction(pr, det):
    if det['best'] is None or pr['best'] is None:
        return None
    g = F(det['best']['G_exact'])
    require(g > 0, 'POSITIVE_DETERMINISTIC_COST')
    return str(100 * (1 - F(pr['best']['G_exact']) / g))


def same_minimum(computed, saved):
    require(computed['status'] == saved['status'], 'OLD_MINIMUM_STATUS')
    if computed['best'] is not None:
        for key in ('cell_id', 'G_exact'):
            require(computed['best'][key] == saved['best'][key], 'OLD_MINIMUM_' + key)
        require(computed['exact_ties'] == saved['exact_ties'], 'OLD_EXACT_TIES')
    else:
        require(saved['best'] is None, 'OLD_UNRESOLVED_MINIMUM')


def saved_pr_gate(rows, saved):
    """Check the saved selection without calculating or selecting any PR cost."""
    best = saved['best']
    require(best is not None, 'SAVED_PR_WINNER_REQUIRED')
    by = index(rows, lambda r: r['cell_id'])
    row = by[best['cell_id']]
    require(row['eligible'] is True and row['G_exact'] == best['G_exact'] and
            row['paired_samples'] == 32, 'SAVED_PR_WINNER_BINDING')
    se = row.get('saved_paired_SE', row.get('G_standard_error'))
    require(se == best.get('saved_paired_SE', best.get('G_standard_error')) and
            se is not None, 'SAVED_PR_SE_BINDING')
    ties = []
    for candidate in rows:
        require(candidate['eligible'] is not None, 'UNRESOLVED_PR_DOMAIN')
        if candidate['eligible'] is True:
            require(candidate['paired_samples'] == 32 and candidate['G_exact'] is not None,
                    'SAVED_PR_N32_COST_REQUIRED')
            require((F(candidate['G_exact']), candidate['cell_id']) >=
                    (F(row['G_exact']), row['cell_id']), 'SAVED_PR_SELECTION_INCONSISTENT')
            if F(candidate['G_exact']) == F(row['G_exact']):
                ties.append(candidate['cell_id'])
    require(sorted(ties) == saved['exact_ties'], 'SAVED_PR_TIES')


def shots_from_accounting(accounting, eligible):
    if eligible is not True:
        return None
    require(accounting is not None and accounting['cost_acquisition_eligible'] is True and
            accounting['eligibility_under_declared_allowance'] == 'ELIGIBLE', 'SAVED_ELIGIBILITY')
    shots = {a: accounting['axes'][a]['shots'] for a in ('real', 'imag')}
    require(all(type(n) is int and n > 0 for n in shots.values()) and
            accounting['N_total'] == sum(shots.values()), 'SAVED_POSITIVE_INTEGER_SHOTS')
    return shots


def replace_row(saved, accounting, measured):
    """Only the six RZ unit prices and their consequent G/provenance may change."""
    row = deepcopy(saved)
    shots = shots_from_accounting(accounting, saved['eligible'])
    if saved['eligible'] is not True:
        require(saved['G_exact'] is None, 'NONELIGIBLE_COST_MUST_BE_NULL')
        return row, shots
    axes = saved['axis_costs']
    old_g = sum(shots[CHANNEL[a]] * F(axes[a]['RZ']) for a in AXES)
    require(old_g == F(saved['G_exact']), 'SAVED_SHOTS_UNIT_COST_CLOSURE')
    if saved['cell_id'] not in measured:
        return row, shots
    new_axes = {}
    for axis in AXES:
        proof = measured[saved['cell_id']][axis]
        old = axes[axis]
        if old['measured']:
            prior = old.get('provenance', old)
            require(F(old['RZ']) == proof['measured_RZ'] and
                    prior.get('origin_SOURCE', prior.get('SOURCE')) == proof['SOURCE'] and
                    prior.get('profile_sha256') == proof['profile_sha256'] and
                    prior.get('compiler_hash') == proof['compiler_hash'], 'OVERLAPPING_MEASUREMENT_CONFLICT')
        else:
            require(old['model_sha256'] == MODEL and
                    F(old['RZ']) == F(proof['model_check']['exact_prediction']), 'UNCHANGED_FIXED_MODEL_REQUIRED')
        new_axes[axis] = dict(RZ=str(proof['measured_RZ']), measured=True,
            status='SELECTED_SIX_DIRECT_MEASURED_AXIS', SOURCE=proof['SOURCE'],
            compiler_hash=proof['compiler_hash'], profile_sha256=proof['profile_sha256'],
            result_sha256=proof['result_sha256'], model_sha256=MODEL,
            previous_unit_evidence=deepcopy(old), model_check=deepcopy(proof['model_check']))
    row['axis_costs'] = new_axes
    row['G_exact'] = str(sum(shots[CHANNEL[a]] * F(new_axes[a]['RZ']) for a in AXES))
    row['cost_status'] = 'SELECTED_SIX_DIRECT_MEASURED_POINT'
    return row, shots


def counts(rows):
    return dict(registered=len(rows), eligible=sum(r['eligible'] is True for r in rows),
        ineligible=sum(r['eligible'] is False for r in rows),
        unresolved=sum(r['eligible'] is None for r in rows),
        eligible_missing_cost=sum(r['eligible'] is True and r['G_exact'] is None for r in rows))


def reaccount(data, registry, measured, permit):
    """One pass over both fixed saved-accounting modes; no old runner calls."""
    require(isinstance(permit, Permit) and permit.synthetic == (data['kind'] == 'ARTIFICIAL'), 'DATA_SCOPE_PERMIT')
    old_ids = {c['id'] for c in registry['old']}
    det_ids = old_ids | {c['id'] for c in registry['additional']}
    pr_ids = set(registry['confirmation_ids'])
    require(len(old_ids) == 32 and len(det_ids) == 184 and len(pr_ids) == 6 and
            not (det_ids & pr_ids), 'FIXED_32_184_UNION6_DOMAIN')
    require(set(measured) == set(REPLACEMENTS) and not (set(measured) & old_ids) and
            set(measured) <= det_ids, 'EXACT_SIX_OUTSIDE_OLD32')
    original_digest = digest(data)
    results, all_rows, changed = [], [], []
    for mode in MODES:
        bundle = data['modes'][mode]
        saved = index(bundle['rows'], row_key)
        expected = {(c, e, f) for c in det_ids | pr_ids for e in EPSILONS for f in FACTORS}
        require(set(saved) == expected, 'COMPLETE_1900_SAVED_ROWS:' + mode)
        accounting = index(bundle['accounting'], row_key)
        require(set(accounting) == expected, 'COMPLETE_SAVED_ACCOUNTING_JOIN')
        cmps = index(bundle['comparisons'], lambda r: (r['epsilon'], r['factor']))
        require(set(cmps) == {(e, f) for e in EPSILONS for f in FACTORS}, 'COMPLETE_TEN_SAVED_COMPARISONS')
        for e in EPSILONS:
            for f in FACTORS:
                before, after = {}, {}
                for cid in sorted(det_ids | pr_ids):
                    src = saved[cid, e, f]
                    if cid in pr_ids:
                        new, shots = deepcopy(src), None
                    else:
                        ac = accounting[(cid, e, f)]['accounting']
                        new, shots = replace_row(src, ac, measured)
                    before[cid], after[cid] = src, new
                    all_rows.append(dict(bias_kind=mode, saved=deepcopy(src), updated=new,
                                         saved_shots=shots))
                    if cid in measured:
                        changed.append(dict(bias_kind=mode, cell_id=cid, epsilon=e, factor=f,
                            eligible=src['eligible'], saved_shots=shots, G_before=src['G_exact'],
                            G_after=new['G_exact'], previous_axes=src.get('axis_costs'),
                            updated_axes=new.get('axis_costs'), selected_before=[], selected_after=[]))
                for scope, ids in (('old32', old_ids), ('strengthened184', det_ids)):
                    old_min = minimum([before[cid] for cid in ids])
                    new_min = minimum([after[cid] for cid in ids])
                    prior = cmps[e, f]['scopes'][scope]
                    same_minimum(old_min, prior['deterministic'])
                    pr = deepcopy(prior['confirmed_PR'])
                    saved_pr_gate([before[cid] for cid in pr_ids], pr)
                    require(all(after[cid] == before[cid] for cid in pr_ids), 'PR_ROWS_AND_SE_UNCHANGED')
                    old_r, new_r = reduction(pr, old_min), reduction(pr, new_min)
                    require(old_r == prior['PR_reduction_percent_exact'], 'SAVED_REDUCTION_EXACT_MATCH')
                    if scope == 'old32':
                        require(all(after[cid] == before[cid] for cid in old_ids) and
                                new_min == old_min and new_r == old_r, 'OLD32_EXACT_INVARIANCE')
                    results.append(dict(bias_kind=mode, epsilon=e, factor=f, scope=scope,
                        before=old_min, after=new_min, saved_PR=pr,
                        reduction_before_exact=old_r, reduction_after_exact=new_r,
                        change_percentage_points_exact=None if new_r is None or old_r is None else str(F(new_r)-F(old_r)),
                        winner_changed=(old_min['best'] or {}).get('cell_id') != (new_min['best'] or {}).get('cell_id'),
                        counts=counts([after[cid] for cid in ids]), PR_counts=counts([after[cid] for cid in pr_ids])))
                    for change in changed:
                        if (change['bias_kind'], change['epsilon'], change['factor']) == (mode, e, f):
                            if change['cell_id'] in old_min.get('exact_ties', []):
                                change['selected_before'].append(scope)
                            if change['cell_id'] in new_min.get('exact_ties', []):
                                change['selected_after'].append(scope)
    require(digest(data) == original_digest, 'INPUT_MUTATION')
    by = {(r['bias_kind'], r['epsilon'], r['factor'], r['scope']): r for r in results}
    differences = []
    for e in EPSILONS:
        for f in FACTORS:
            for scope in ('old32', 'strengthened184'):
                a, b = [by[m, e, f, scope] for m in MODES]
                differences.append(dict(epsilon=e, factor=f, scope=scope,
                    noncancellation_minus_usual_before_pp_exact=str(F(b['reduction_before_exact'])-F(a['reduction_before_exact'])),
                    noncancellation_minus_usual_after_pp_exact=str(F(b['reduction_after_exact'])-F(a['reduction_after_exact']))))
    return dict(status='COMPLETE_MANDATORY_STOP', comparisons=results, accounting_rows=all_rows,
        measured_candidate_changes=changed, mode_differences=differences,
        scientific_attempts=1, reaccounting_passes=1, saved_rows=3800, comparison_count=40,
        PR_recalculation=0, shot_recalculation=0, model_prediction=0, compile=0, fit=0,
        signal=0, sampling=0, retry=0, next_stage_authorized=False)

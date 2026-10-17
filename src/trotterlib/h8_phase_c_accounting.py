"""Saved H8 scalar arithmetic only. Real callers must pass the execution gate.

No signal, shot rule, circuit, fit, compiler, sampler or molecular imports.
S4 model use is conditional on all four registered B2 axes and final gates.
"""
from fractions import Fraction as F
import hashlib
import json
import math

EPSILONS = (.05, .01, .005, .001, .0005)
FACTORS = (1., 10.)
AXES = ('cosine', 'sine')
CHANNEL = dict(cosine='real', sine='imag')
B1_RESULT = '83a3c4a6688e66576f85471bf41a45e29cbfb705'
MODEL_SHA = '8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0'
B2_SOURCE = '244c45e752fa703e286d40ece9ad641430ea9c6e'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
B2_AXES = [(cid, a) for cid in ('B0_p15_q1_4th', 'B0I_p21_q8_4th') for a in AXES]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def finite(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, name)
    return value


def unique(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, 'DUPLICATE:' + str(k))
        result[k] = row
    return result


def b2_gate(proof):
    """Accept completed native B2 evidence only; do not redo its residuals."""
    terminal, cleanup, final = (proof[k] for k in ('terminal', 'cleanup', 'final'))
    require(terminal['status'] == cleanup['status'] == 'COMPLETE_MANDATORY_STOP' and
            terminal['SOURCE'] == B2_SOURCE and final['SOURCE'] == B2_SOURCE and
            final['status'] == 'PASS_MANDATORY_STOP' and final['profile_sha256'] == PROFILE,
            'ALL_B2_COMPLETION_GATES_REQUIRED')
    require(cleanup['native_exit_code'] == 0 and cleanup['all_reaped'] is True and
            cleanup['cleanup_errors'] == [] and cleanup['first_stop'] is None and
            0 < finite(cleanup['max_monitor_gap_seconds'], 'B2_MONITOR_GAP') <= 5,
            'B2_NATIVE_EXIT_CLEANUP_MONITOR')
    require(proof['launcher_native_exit']['native_exit_code'] == 0 and
            proof['launcher_native_exit']['wait_completed'] is True, 'B2_LAUNCHER_NATIVE_WAIT')
    require(terminal['fit_performed'] is False and terminal['next_stage_authorized'] is False and
            terminal['validation']['status'] == 'VALIDATED_ONLY_ON_FOUR_REGISTERED_S4_AXES', 'B2_SCOPE')
    rows = terminal['axes']
    require([(r['case_id'], r['axis']) for r in rows] == B2_AXES, 'EXACT_B2_FOUR_AXES')
    for row in rows:
        require(row['model_sha256'] == MODEL_SHA and row['profile_sha256'] == PROFILE and
                row['compiler_hash'] == COMPILER and row['native_exit_code'] == 0 and
                row['reaped'] is True and row['fingerprint_before'] == row['fingerprint_after'] and
                type(row['measured_RZ']) is int and row['measured_RZ'] > 0 and
                0 <= finite(row['max_control_error'], 'B2_CONTROL') < 1e-8 and
                row['model_check']['PASS'] is True and
                0 <= F(row['model_check']['exact_relative_error']) < F(1, 50), 'B2_AXIS_NOT_PASS')
    return rows


def eligibility(a):
    if a is None:
        return None
    require(a['evidence_kind'] == 'EMPIRICAL' and
            a['allowance_certification_claimed_by_caller'] is False, 'EMPIRICAL_ACCOUNTING_ONLY')
    if a['cost_acquisition_eligible'] is True:
        require(a['eligibility_under_declared_allowance'] == 'ELIGIBLE', 'ELIGIBILITY_CONTRADICTION')
        shots = []
        for axis in ('real', 'imag'):
            row = a['axes'][axis]; n = row['shots']
            require(type(n) is int and 0 < n < 2**52 and
                    finite(row['headroom'], 'HEADROOM') > 0 and
                    finite(row['u_over_headroom'], 'U_HEADROOM') <= .01, 'SAVED_SHOT_GUARD')
            shots.append(n)
        require(a['N_total'] == sum(shots), 'SAVED_SHOT_TOTAL')
        return True
    require(a['cost_acquisition_eligible'] is False, 'ELIGIBILITY_BOOLEAN')
    return False if a['eligibility_under_declared_allowance'] == 'INELIGIBLE' else None


def minimum(rows):
    missing = [r['cell_id'] for r in rows if r['eligible'] is None]
    if missing:
        return dict(status='UNRESOLVED_FULL_DOMAIN', best=None, missing=missing)
    valid = [r for r in rows if r['eligible'] is True]
    if not valid:
        return dict(status='NO_ELIGIBLE_CANDIDATE', best=None)
    require(all(r['G_exact'] is not None for r in valid), 'ELIGIBLE_COST_MISSING')
    best = min(valid, key=lambda r: (F(r['G_exact']), r['cell_id']))
    return dict(status='CONDITIONAL_POINT_MINIMUM', best=best,
                exact_ties=sorted(r['cell_id'] for r in valid if F(r['G_exact']) == F(best['G_exact'])))


def evaluate(data, registry, *, take=lambda key: None, emit=lambda kind, value: None, guard=lambda: None):
    """One indexed pass; measured overrides and fixed-model estimates stay labeled."""
    validated_b2 = b2_gate(data['B2'])  # Before any accounting/model prediction/selection.
    require(data['B1_terminal']['status'] == 'COMPLETE_MANDATORY_STOP' and
            data['B1_terminal']['model_sha256'] == MODEL_SHA and
            data['B1_result_commit'] == B1_RESULT and digest(data['model']) == MODEL_SHA,
            'FIXED_B1_MODEL_AND_RESULT')
    old, extra = registry['old'], registry['additional']
    cells = unique(old + extra, lambda c: c['id'])
    require(len(old) == 32 and len(extra) == 152 and len(cells) == 184, 'FIXED_DETERMINISTIC_DOMAIN')
    confirmed = registry['confirmation_ids']
    require(len(confirmed) == len(set(confirmed)) == 6, 'FIXED_CONFIRMATION_UNION6')
    saved = unique(data['accounting'], lambda r: (r['cell_id'], r['epsilon'], float(r['factor'])))
    expected = {(cid, e, f) for cid in [*cells, *confirmed] for e in EPSILONS for f in FACTORS}
    require(set(saved) == expected and len(saved) == 1900, 'EXACT_SAVED_1900_SHOT_ROWS')
    strata = unique(data['model']['models'], lambda r: (r['method'], r['axis']))
    require(set(strata) == {(m, a) for m in ('B0', 'B0I') for a in AXES}, 'FOUR_MODEL_STRATA')
    measured = unique(data['measured_axes'], lambda r: (r['cell_id'], r['axis']))
    for key, row in measured.items():
        require(key[0] in cells and key[1] in AXES and row['compiler_hash'] == COMPILER and
                type(row['RZ']) is int and row['RZ'] > 0 and row['provenance_verified'] is True,
                'MEASURED_AXIS_LINEAGE')
        require(row['kind'] in ('B1_REGISTERED_ANCHOR', 'LEGACY_FULL_PF_DECLARED_LIMITS'),
                'NO_UNRESOLVED_OLD_B0_COST_REUSE')
        require((cells[key[0]]['method'] == 'B1') == (row['kind'] == 'LEGACY_FULL_PF_DECLARED_LIMITS'),
                'MEASURED_COST_METHOD')
    for row in validated_b2:
        measured[(row['case_id'], row['axis'])] = dict(cell_id=row['case_id'], axis=row['axis'],
            RZ=row['measured_RZ'], kind='B2_REGISTERED_S4_MEASUREMENT', compiler_hash=COMPILER,
            provenance_verified=True, origin_SOURCE=B2_SOURCE, profile_sha256=PROFILE)
    take('reaccounting_pass')
    axis_cache = {}

    def axis_cost(cell, axis):
        key = cell['id'], axis
        if key in axis_cache:
            return axis_cache[key]
        if key in measured:
            row = measured[key]
            value = dict(RZ=str(row['RZ']), status='SAVED_MEASURED_AXIS', measured=True,
                         origin=row['kind'], provenance=row)
        else:
            require(cell['method'] in ('B0', 'B0I'), 'ELIGIBLE_FULL_PF_COST_MISSING')
            model = strata[cell['method'], axis]
            require(model['status'] == 'S2_ANCHORS_AND_REGISTERED_HOLDOUT_PASS', 'B1_STRATUM_STATUS')
            a, b, d = map(F, model['coefficients'])
            pieces = 1 if cell['formula'] == '2nd' else 3
            require(cell['formula'] in ('2nd', '4th'), 'REGISTERED_FORMULA')
            value_rz = a + pieces * cell['q'] * (b + d * cell['prefix'])
            require(value_rz > 0, 'NONPOSITIVE_MODEL_COST')
            take('model_axis_estimate')
            value = dict(RZ=str(value_rz), status='FIXED_MODEL_ESTIMATE', measured=False,
                         model_sha256=MODEL_SHA, S4_validated_on_four_axes_only=True,
                         prefix_outside_B1_fit_range=not 14 <= cell['prefix'] <= 21,
                         full_grid_error_guaranteed=False)
        axis_cache[key] = value
        return value

    all_rows = []
    comparisons = []
    for epsilon in EPSILONS:
        for factor in FACTORS:
            current = {}
            for cid in [*cells, *confirmed]:
                guard(); take('saved_row')
                source = saved[cid, epsilon, factor]
                status = eligibility(source['accounting'])
                row = dict(cell_id=cid, epsilon=epsilon, factor=factor, eligible=status,
                           G_exact=None, cost_status='INELIGIBLE' if status is False else 'MISSING_OR_UNRESOLVED',
                           P=0, rigorous_bias_guaranteed=False, population_winner_certified=False)
                if status is True:
                    if cid in cells:
                        axes = {a: axis_cost(cells[cid], a) for a in AXES}
                        value = sum(F(axes[a]['RZ']) * source['accounting']['axes'][CHANNEL[a]]['shots'] for a in AXES)
                        row.update(axis_costs=axes, cost_status='MODEL_CONDITIONAL_POINT' if
                                   any(not x['measured'] for x in axes.values()) else 'SAVED_MEASURED_POINT')
                    else:
                        metric = source['PR_metric']
                        require(source['sample_phase'] == 'confirmation' and source['sample_count'] == 32 and
                                metric['paired_samples'] == 32 and metric['population_winner_certified'] is False,
                                'CONFIRMATION_N32_ONLY')
                        exact = metric['G_exact']
                        value = F(int(exact['numerator']), int(exact['denominator']))
                        finite(metric['G_standard_error'], 'SAVED_PAIRED_SE')
                        row.update(cost_status='SAVED_PR_FINITE_SAMPLE_MEAN_N32', paired_samples=32,
                                   saved_paired_SE=metric['G_standard_error'], uncertainty_scope='SAMPLE_SE_NOT_CI')
                    require(value > 0, 'NONPOSITIVE_WEIGHTED_COST')
                    row['G_exact'] = str(value)
                current[cid] = row; all_rows.append(row); emit('accounting_row', row)
            pr = minimum([current[cid] for cid in confirmed])
            scopes = {}
            for scope, candidates in (('old32', old), ('strengthened184', old + extra)):
                take('comparison')
                det = minimum([current[c['id']] for c in candidates])
                ratio = (F(pr['best']['G_exact']) / F(det['best']['G_exact'])
                         if pr['best'] is not None and det['best'] is not None else None)
                scopes[scope] = dict(deterministic=det, confirmed_PR=pr,
                    PR_over_deterministic_exact=str(ratio) if ratio is not None else None,
                    PR_reduction_percent_exact=str(100 * (1-ratio)) if ratio is not None else None,
                    baseline_policy='SAME_CURRENT_MEASURED_FIRST_FIXED_MODEL_COST_POLICY',
                    full_candidate_measured_minimum_claimed=False)
            before, after = (scopes[s]['PR_reduction_percent_exact'] for s in ('old32', 'strengthened184'))
            result = dict(epsilon=epsilon, factor=factor, scopes=scopes,
                          reduction_change_percentage_points_exact=str(F(after)-F(before))
                          if before is not None and after is not None else None)
            comparisons.append(result); emit('comparison', result)
    return dict(status='COMPLETE_MANDATORY_STOP', accounting_rows=len(all_rows), comparisons=comparisons,
                model_sha256=MODEL_SHA, state_preparation_P=0, all_candidates_measured=False,
                rigorous_bias_or_PR_population_guaranteed=False, original_B0_run02_status='STOP',
                next_stage_authorized=False, automatic_retry=False, refit=False)

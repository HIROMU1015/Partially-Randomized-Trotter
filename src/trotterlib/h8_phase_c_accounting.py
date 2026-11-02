"""Saved H8 scalar arithmetic only. Real callers must pass the execution gate.

No signal, shot rule, circuit, fit, compiler, sampler or molecular imports.
S4 model use is conditional on the final B2 four-axis acceptance contract.
Only saved usual signed-bias shots enter; noncancellation diagnostics never do.
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
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
B2_RESULT = 'fef2db7a79886ab682b652da7266b463c98c7d3a'
B2_STATUS = 'PASS_REGISTERED_FOUR_AXES'
B2_FINAL_SOURCE = 'f58da9590597f42ff7260c22d771734096a706bb'
# Registered order: (case, axis, execution SOURCE of that axis, frozen B1 prediction).
B2_AXES = [('B0_p15_q1_4th', 'cosine', '244c45e752fa703e286d40ece9ad641430ea9c6e', '1175933/7'),
           ('B0_p15_q1_4th', 'sine', '244c45e752fa703e286d40ece9ad641430ea9c6e', '1175933/7'),
           ('B0I_p21_q8_4th', 'cosine', '2e604e4a3a04eee5090e03d9bb83818d7afd3c87', '1852640'),
           ('B0I_p21_q8_4th', 'sine', B2_FINAL_SOURCE, '1852640')]
USUAL_BIAS = 'USUAL_SIGNED_SAVED'


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
    """Accept only the final registered-four-axis B2 acceptance contract.

    Three saved axes and one newly completed axis carry their own execution SOURCE.
    The binding and its terminal-receipt projection must agree exactly; residuals are
    checked against the frozen predictions, never recomputed from a new model.
    """
    b, r = proof['binding'], proof['receipts']
    require(b['schema'] == 'H8_B2_REGISTERED_FOUR_AXES_PASS_BINDING_V1' and b['status'] == B2_STATUS and
            b['final_execution_SOURCE'] == B2_FINAL_SOURCE and b['source_per_axis'] is True and
            (b['registered_axes'], b['saved_axes_accepted'], b['new_axes_completed']) == (4, 3, 1) and
            b['model_sha256'] == MODEL_SHA and b['compiler_sha256'] == COMPILER and b['profile_sha256'] == PROFILE and
            b['relative_error'] == 'abs(predicted-measured)/measured' and b['STOP_threshold'] == '1/50' and
            b['comparison'] == 'strictly_less_than', 'B2_FINAL_FOUR_AXIS_BINDING_REQUIRED')
    require(b['final_gate'] == dict(status='PASS_MANDATORY_STOP', SOURCE=B2_FINAL_SOURCE, profile_sha256=PROFILE,
                                    next_stage_authorized=False), 'B2_FINAL_GATE_PASS_REQUIRED')
    require(b['original_runs_retroactively_changed'] is False and b['old_STOPs_and_unsettled_reservations_preserved'] is True and
            b['single_original_attempt_completed'] is False and b['full_grid_guaranteed'] is False and
            b['Phase_C_authorized'] is False and b['mandatory_STOP'] is True, 'B2_HISTORY_AND_SCOPE_FLAGS')
    rows = b['axes']
    require([(x['case_id'], x['axis'], x['SOURCE']) for x in rows] == [a[:3] for a in B2_AXES], 'EXACT_B2_FOUR_AXES_AND_SOURCES')
    measured = {}
    for row, (cid, axis, source, prediction) in zip(rows, B2_AXES):
        check = row['model_check']
        m = row['measured_RZ']
        require(type(m) is int and m > 0 and check['measured_RZ'] == m and check['exact_prediction'] == prediction and
                check['PASS'] is True and F(check['exact_relative_error']) == abs(F(prediction)-m)/m < F(1, 50) and
                row['model_sha256'] == MODEL_SHA and row['profile_sha256'] == PROFILE and row['compiler_hash'] == COMPILER and
                row['native_exit_code'] == 0 and row['reaped'] is True and row['fingerprint_before'] == row['fingerprint_after'] and
                0 <= finite(row['max_control_error'], 'B2_CONTROL') < 1e-8, 'B2_AXIS_NOT_PASS:' + cid + ':' + axis)
        measured[cid, axis] = dict(RZ=m, SOURCE=source, acceptance='NEW_AXIS_COMPLETED' if source == B2_FINAL_SOURCE
                                   else 'SAVED_AXIS_ACCEPTED_BY_FINAL_CONTRACT', result_sha256=row['result_sha256'])
    t, lt, w, adm = r['raw_driver_terminal'], r['raw_launcher_terminal'], r['native_launcher_wait'], r['admission_terminal']
    require(t['status'] == 'COMPLETE_MANDATORY_STOP' and t['SOURCE'] == B2_FINAL_SOURCE and t['fit_performed'] is False and
            t['retry'] is False and t['next_stage_authorized'] is False and t['axes'] == rows[3:] and
            t['saved_B0_axes'] == rows[:2] and t['saved_cosine_axis'] == rows[2] and
            t['validation']['C_authorized'] is False and t['validation']['full_grid_guaranteed'] is False,
            'B2_DRIVER_TERMINAL_MATCHES_BINDING')
    require(lt['status'] == 'COMPLETE_MANDATORY_STOP' and lt['native_exit_code'] == 0 and lt['all_reaped'] is True and
            lt['cleanup_errors'] == [] and lt['first_stop'] is None and not lt.get('process_thread_violation'),
            'B2_NATIVE_EXIT_CLEANUP')
    require(w['status'] == 'NATIVE_LAUNCHER_WAIT_RECORDED' and w['native_exit_code'] == 0 and w['wait_completed'] is True and
            w['witness_stop_requested'] is False, 'B2_LAUNCHER_NATIVE_WAIT')
    require(adm['status'] == 'COMPLETE_MANDATORY_STOP' and adm['SOURCE'] == B2_FINAL_SOURCE and adm['reported_exit_code'] == 0 and
            r['final_gate'] == b['final_gate'] and r['new_compile_calls'] == 1 and r['extra_unregistered_compile_calls'] == 0 and
            r['Phase_C_authorized'] is False and r['fit_performed'] is False and r['scientific_retry_after_mismatch'] is False,
            'B2_ADMISSION_FINAL_GATE_AND_CALLS')
    return measured


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
    require(all(r.get('S4_status') == 'UNVALIDATED_SCHEDULE_COMPOSITION_HYPOTHESIS' and r.get('full_grid_guaranteed') is False
                for r in strata.values()), 'B1_S4_HYPOTHESIS_FROZEN')
    measured = unique(data['measured_axes'], lambda r: (r['cell_id'], r['axis']))
    for key, row in measured.items():
        require(key[0] in cells and key[1] in AXES and row['compiler_hash'] == COMPILER and
                type(row['RZ']) is int and row['RZ'] > 0 and row['provenance_verified'] is True,
                'MEASURED_AXIS_LINEAGE')
        require(row['kind'] in ('B1_REGISTERED_ANCHOR', 'LEGACY_FULL_PF_DECLARED_LIMITS'),
                'NO_UNRESOLVED_OLD_B0_COST_REUSE')
        require((cells[key[0]]['method'] == 'B1') == (row['kind'] == 'LEGACY_FULL_PF_DECLARED_LIMITS'),
                'MEASURED_COST_METHOD')
    for (cid, axis), row in validated_b2.items():
        require(cid in cells and cells[cid]['formula'] == '4th', 'B2_AXIS_OUTSIDE_REGISTERED_DOMAIN')
        measured[cid, axis] = dict(cell_id=cid, axis=axis, RZ=row['RZ'], kind='B2_REGISTERED_S4_MEASUREMENT',
            compiler_hash=COMPILER, provenance_verified=True, origin_SOURCE=row['SOURCE'], profile_sha256=PROFILE,
            B2_result_commit=B2_RESULT, B2_acceptance=row['acceptance'], result_sha256=row['result_sha256'])
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
                         model_sha256=MODEL_SHA, model_formula='S2' if pieces == 1 else 'S4',
                         model_validation='B1_S2_ANCHORS_HOLDOUT_OR_TRAINING_RESIDUAL' if pieces == 1 else
                         'S4_COMPOSITION_HYPOTHESIS_PASSED_ONLY_ON_FOUR_REGISTERED_B2_AXES',
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
                require(source.get('bias_kind') == USUAL_BIAS, 'ONLY_SAVED_USUAL_SIGNED_BIAS_ROWS')
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
                bias_kind=USUAL_BIAS, noncancellation_diagnostic_used=False,
                B2_binding=dict(status=B2_STATUS, result_commit=B2_RESULT, final_SOURCE=B2_FINAL_SOURCE,
                    axes=[dict(cell_id=k[0], axis=k[1], **v) for k, v in validated_b2.items()],
                    single_original_attempt_completed=False),
                S4_model_status='COMPOSITION_HYPOTHESIS_PASSED_ON_FOUR_REGISTERED_B2_AXES_ONLY_NOT_FULL_GRID',
                rigorous_bias_or_PR_population_guaranteed=False, original_B0_run02_status='STOP',
                next_stage_authorized=False, automatic_retry=False, refit=False)

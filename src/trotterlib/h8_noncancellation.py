"""Empirical per-axis noncancellation diagnostic over validated saved signed parts.

Usual signed bias: abs(total_axis). Empirical noncancellation: abs(discard)+abs(PF)
for B0/B0I, abs(PF) for full PF and abs(outer_PF)+abs(finite_RTE) for PR. Neither
is a rigorous bias bound (always None). Shots call the frozen D0 accounting text;
only the declared nonnegative bias magnitude changes. Costs follow measured-first,
fixed-B1-model, PR finite-sample labels. No signal, compile, sampling, fit,
SCF/DF/state or molecular import. Saved H8 values require an approved permit.
"""
from fractions import Fraction as F
import hashlib
import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = 'H8_EMPIRICAL_NONCANCELLATION_DIAGNOSTIC_V1'
STAGES = ('NONCANCELLATION_SHOTS', 'COST_COMPARISON')
EPSILONS = (.05, .01, .005, .001, .0005)
FACTORS = (1., 10.)
ALPHA_AXIS = .025
MAXIMUM_U_OVER_HEADROOM = .01
AXES = ('real', 'imag')
CHANNEL = dict(cosine='real', sine='imag')
TOLERANCES = dict(rel_tol=1e-12, abs_tol=1e-12)
PARTS = dict(B0=('discard', 'PF'), B0I=('discard', 'PF'), B1=('PF',), B2=('outer_PF', 'finite_RTE'))
FROZEN_D0 = {'scripts/d0_saved_value_reanalysis_v1.py':
             '1ded644585bfabe4f8f28be26b92e8c09f00c4e8f9712b4cf45e0233cfc8737e',
             'scripts/d0_frozen_shot_rule_v1.py':
             'ed3f57937a231719583c2893677b7e82f934dd98bda35cd37fc0f17f83d4279f'}
SIGNED = dict(SOURCE='64ec6cbba7e48ac7c04a6ad0b38c3e76f0f14526',
              plan_sha256='7c00b31b35cf890716fdf94a90ce9ec5561098372c986cbcdc8dde139ba12a09',
              grant_sha256='3a5a930d928927ebd0d8b6bd8dc87c4bd24286015886476e2660f10c063dc94c')
MODEL_SHA = '8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
ANCHOR_SOURCE = dict(B0='db559a518214fd27fa96e2f1451911df05c4f4a8',
                     B0I='54ed98c5fa5b7190f77255271f4e3bec8db15630')
B1_FIT_PREFIXES = (14, 21)
B2 = dict(SOURCE='2e604e4a3a04eee5090e03d9bb83818d7afd3c87',
          scope='H8_B2_SAVED_B0_ACCEPTANCE_B0I_TWO_AXES_PARALLEL_PREPARATION_V1',
          validation='TWO_SAVED_B0_PLUS_TWO_NEW_B0I_REGISTERED_AXES_ONLY',
          axes=(('B0_p15_q1_4th', 'cosine', '1175933/7'), ('B0_p15_q1_4th', 'sine', '1175933/7'),
                ('B0I_p21_q8_4th', 'cosine', '1852640'), ('B0I_p21_q8_4th', 'sine', '1852640')))
CONFIRMATION_N = 32
# Corrupt numbers STOP; absent or failed component evidence stays UNRESOLVED.
NONFINITE = ('NONFINITE_NUMBER', 'NONFINITE_NONCANCELLATION_ENVELOPE')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _frozen_d0():
    for rel, sha in FROZEN_D0.items():
        require(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == sha, 'FROZEN_D0_SOURCE_CHANGED:' + rel)
    spec = importlib.util.spec_from_file_location('h8_noncancellation_frozen_d0',
                                                  ROOT/'scripts/d0_saved_value_reanalysis_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d0 = _frozen_d0()


def finite(value):
    require(type(value) in (float, int), 'FINITE_REAL_NUMBER_REQUIRED')
    try:
        require(math.isfinite(value), 'NONFINITE_NUMBER')
    except OverflowError as exc:
        raise ValueError('NONFINITE_NUMBER') from exc
    return value


def envelope(method, parts, total):
    """One axis: usual and empirical noncancellation magnitudes; never a bound."""
    require(method in PARTS and type(parts) is dict and set(parts) == set(PARTS[method]),
            'SIGNED_COMPONENT_SET')
    values = [finite(parts[k]) for k in PARTS[method]]
    total = finite(total)
    # Same closure rule as the signed validation; a different total is not repaired.
    require(math.isclose(sum(values), total, **TOLERANCES), 'COMPONENTS_DO_NOT_CLOSE_TO_SAVED_TOTAL')
    diagnostic = sum(abs(v) for v in values)
    require(math.isfinite(diagnostic), 'NONFINITE_NONCANCELLATION_ENVELOPE')
    return dict(usual_signed=abs(total), empirical_noncancellation=diagnostic,
                signed_parts={k: parts[k] for k in PARTS[method]}, rigorous_bias_bound=None)


def frozen_accounting(record, epsilon, factor, magnitude=None):
    """Frozen D0 accounting text. None reproduces the usual saved row; otherwise
    abs(total_signed) is replaced by the declared nonnegative axis magnitude only.
    The view is never reported as a signed value or written back."""
    require(epsilon in EPSILONS and factor in FACTORS, 'REGISTERED_PRECISION_AND_FACTOR')
    if magnitude is None:
        return d0.accounting(record, epsilon, factor)
    require(type(magnitude) is dict and set(magnitude) == set(AXES) and
            all(finite(magnitude[a]) >= 0 for a in AXES), 'DIAGNOSTIC_MAGNITUDE')
    view = dict(record, error_decomposition=dict(total_signed=dict(magnitude)))
    return d0.accounting(view, epsilon, factor)


def eligibility(a):
    """True/False/None; None is unresolved (UNDETERMINED or numerical guard)."""
    require(a['evidence_kind'] == 'EMPIRICAL' and
            a['allowance_certification_claimed_by_caller'] is False and
            a['certificate_verified_by_this_function'] is False, 'EMPIRICAL_ACCOUNTING_ONLY')
    if a['cost_acquisition_eligible'] is True:
        require(a['eligibility_under_declared_allowance'] == 'ELIGIBLE', 'ELIGIBILITY_CONTRADICTION')
        shots = {}
        for axis in AXES:
            row = a['axes'][axis]
            n = row['shots']
            require(type(n) is int and 0 < n < 2**52 and finite(row['headroom']) > 0 and
                    finite(row['u_over_headroom']) <= MAXIMUM_U_OVER_HEADROOM, 'SHOT_GUARD')
            shots[axis] = n
        require(a['N_total'] == sum(shots.values()), 'SHOT_TOTAL')
        return True, shots
    require(a['cost_acquisition_eligible'] is False, 'ELIGIBILITY_BOOLEAN')
    return (False if a['eligibility_under_declared_allowance'] == 'INELIGIBLE' else None), None


def label(eligible):
    return {True: 'ELIGIBLE', False: 'INELIGIBLE', None: 'UNRESOLVED'}[eligible]


def bind_domain(data, registry):
    old, extra = registry['old'], registry['additional']
    require(len(old) == 32 and len(extra) == 152, 'FIXED_REGISTRY_COUNTS')
    cells = {c['id']: c for c in old + extra}
    require(len(cells) == 184 and all(c['method'] in ('B0', 'B0I', 'B1') for c in cells.values()),
            'FIXED_REGISTRY_UNIQUE')
    union = list(data['confirmation']['selected_ids'])
    require(len(union) == len(set(union)) == 6 and not set(union) & set(cells), 'FIXED_CONFIRMATION_UNION6')
    records = {}
    for row in data['records']:
        cid = row['cell']['id']
        require(cid not in records, 'DUPLICATE_RECORD:' + cid)
        records[cid] = row
    require(set(records) == set(cells) | set(union), 'EXACT_184_PLUS_UNION6_RECORDS')
    for cid, cell in cells.items():
        require(all(records[cid]['cell'][k] == cell[k] for k in ('id', 'method', 'prefix', 'q', 'formula')),
                'CELL_BINDING:' + cid)
    for cid in union:
        cell = records[cid]['cell']
        require(cell['method'] == 'B2' and type(cell['q']) is int and type(cell['r']) is int and
                cell['R'] == cell['q']*cell['r'], 'PR_CELL_BINDING:' + cid)
    return cells, union, records


def signed_index(signed, cells):
    """Validated run01 components; failed or absent axes stay unresolved."""
    require(signed['schema'] == 'H8_SIGNED_COMPONENT_VALIDATION_RESULT_V1' and
            all(signed[k] == v for k, v in SIGNED.items()) and signed['tolerances'] == TOLERANCES and
            signed['rigorous_bias_guarantee'] is None, 'VALIDATED_SIGNED_RESULT_BINDING')
    index = {}
    for row in signed['rows']:
        key = row['cell']['id'], row['axis']
        require(key not in index and key[0] in cells and key[1] in AXES, 'SIGNED_AXIS_KEY')
        require(all(row['cell'][k] == cells[key[0]][k] for k in ('method', 'prefix', 'q', 'formula')),
                'SIGNED_CELL_BINDING')
        index[key] = row
    unresolved = {}
    for cid in cells:
        for axis in AXES:
            row = index.get((cid, axis))
            if row is None:
                unresolved[cid, axis] = 'MISSING_VALIDATED_SIGNED_AXIS'
            elif row['status'] != 'PASS_EMPIRICAL_NUMERICAL_CONSISTENCY' or row['first_failure'] is not None:
                unresolved[cid, axis] = 'SIGNED_VALIDATION_NOT_PASS'
    return index, unresolved


def usual_index(saved, ids):
    index = {}
    for row in saved:
        key = row['cell_id'], float(row['epsilon']), float(row['factor'])
        require(key not in index, 'DUPLICATE_SAVED_USUAL_ROW')
        index[key] = row['accounting']
    expected = {(cid, e, f) for cid in ids for e in EPSILONS for f in FACTORS}
    require(set(index) == expected and len(index) == 10*len(ids), 'EXACT_SAVED_USUAL_ROWS')
    return index


def validate_PR(record, reference):
    """Fixed per-axis endpoint/closure/forward checks; failures are retained."""
    checks, failures = [], []

    def check(name, observed, wanted):
        observed, wanted = finite(observed), finite(wanted)
        ok = math.isclose(observed, wanted, **TOLERANCES)
        checks.append(dict(name=name, passed=ok, signed_difference=finite(observed-wanted)))
        if not ok:
            failures.append(name)

    parts = {}
    try:
        saved, s, f = record['error_decomposition'], record['signals'], record['independent_forward_signals']
        require(set(saved['signed_parts']) == set(PARTS['B2']), 'PR_SIGNED_COMPONENT_SET')
        for axis in AXES:
            corrected, tail = finite(s['corrected'][axis]), finite(s['exact_tail'][axis])
            ref = finite(record['reference'][axis])
            expected = dict(outer_PF=finite(tail-ref), finite_RTE=finite(corrected-tail))
            check(axis + ':full_H_reference_unrotated', ref, reference[axis])
            check(axis + ':saved_total_matches_endpoints', saved['total_signed'][axis], finite(corrected-ref))
            for name in PARTS['B2']:
                check(axis + ':saved_' + name + '_matches_endpoints', saved['signed_parts'][name][axis],
                      expected[name])
            check(axis + ':saved_component_closure',
                  sum(finite(saved['signed_parts'][k][axis]) for k in PARTS['B2']), saved['total_signed'][axis])
            check(axis + ':independent_forward_corrected', f['corrected'][axis], corrected)
            check(axis + ':independent_forward_exact_tail', f['exact_tail'][axis], tail)
            parts[axis] = {k: saved['signed_parts'][k][axis] for k in PARTS['B2']}
    except (KeyError, TypeError, ValueError) as exc:
        if str(exc) in NONFINITE:
            raise
        failures.append(type(exc).__name__ + ':' + str(exc))
    return dict(cell_id=record['cell']['id'], status='FAIL_OR_MISSING' if failures else 'PASS_EMPIRICAL_PR_COMPONENTS',
                first_failure=failures[0] if failures else None, failures=failures, checks=checks,
                signed_parts=None if failures else parts, rigorous_bias_guarantee=None)


def PR_normalization(record):
    """Frozen D0 H8 normalization binding; exact equality, no tolerance rescue."""
    b = finite(record['b'])
    require(b > 0 and record['numerical_allowance_certified'] is False, 'PR_NORMALIZATION_OR_U_EVIDENCE')
    require(record['log_B'] == record['cell']['R']*math.log(b), 'PR_SAVED_NORMALIZATION_LOG_BINDING')
    require(record['allowance']['kind'] == 'EMPIRICAL' and record['allowance']['certified'] is False,
            'PR_EMPIRICAL_U_REQUIRED')


def b2_gate(proof, saved_B0_registry):
    """Accept only the completed sealed B2 recovery with all four axes PASS."""
    t, lt, fg, lg, w = (proof[k] for k in ('driver_terminal', 'launcher_terminal', 'final_gate',
                                           'launcher_log_terminal', 'witness_native_exit'))
    require(t['status'] == 'COMPLETE_MANDATORY_STOP' and t['scope'] == B2['scope'] and
            t['SOURCE'] == B2['SOURCE'] and t['fit_performed'] is False and t['retry'] is False and
            t['next_stage_authorized'] is False and t['validation']['status'] == B2['validation'],
            'B2_DRIVER_TERMINAL_COMPLETE_REQUIRED')
    require(lt['status'] == 'COMPLETE_MANDATORY_STOP' and lt['native_exit_code'] == 0 and
            lt['first_stop'] is None and lt['cleanup_errors'] == [] and lt['all_reaped'] is True and
            not lt.get('process_thread_violation') and
            0 < finite(lt['max_monitor_gap_seconds']) <= 5, 'B2_DRIVER_NATIVE_EXIT_CLEANUP_MONITOR')
    require(fg['status'] == 'PASS_MANDATORY_STOP' and fg['SOURCE'] == B2['SOURCE'] and
            fg['profile_sha256'] == PROFILE, 'B2_FINAL_GATE_PASS_REQUIRED')
    require(lg['status'] == 'COMPLETE_MANDATORY_STOP' and lg['SOURCE'] == B2['SOURCE'] and
            lg['reported_exit_code'] == 0, 'B2_LAUNCHER_TERMINAL_COMPLETE')
    require(w['status'] == 'NATIVE_LAUNCHER_WAIT_RECORDED' and w['native_exit_code'] == 0 and
            w['wait_completed'] is True and w['witness_stop_requested'] is False, 'B2_LAUNCHER_NATIVE_WAIT')
    accepted = proof['saved_B0_acceptance']
    require(t['saved_B0_axes'] == saved_B0_registry['axes'] == accepted['axes'] and
            accepted['acceptance_scope'] == B2['scope'] and accepted['original_B2_status'] == 'STOP' and
            accepted['accepted_by_new_grant'] is True and
            saved_B0_registry['original_execution_status'] == 'STOP' and
            saved_B0_registry['model_sha256'] == MODEL_SHA, 'B2_SAVED_B0_ACCEPTANCE_BINDING')
    rows = list(t['saved_B0_axes']) + list(t['axes'])
    require([(r['case_id'], r['axis']) for r in rows] == [x[:2] for x in B2['axes']], 'EXACT_B2_FOUR_AXES')
    measured = {}
    for row, (cid, axis, prediction) in zip(rows, B2['axes']):
        check = row['model_check']
        require(row['model_sha256'] == MODEL_SHA and row['profile_sha256'] == PROFILE and
                row['compiler_hash'] == COMPILER and row['native_exit_code'] == 0 and row['reaped'] is True and
                row['fingerprint_before'] == row['fingerprint_after'] and
                type(row['measured_RZ']) is int and row['measured_RZ'] > 0 and
                check['measured_RZ'] == row['measured_RZ'] and check['exact_prediction'] == prediction and
                0 <= finite(row['max_control_error']) < 1e-8 and check['PASS'] is True and
                abs(F(prediction)-row['measured_RZ'])/row['measured_RZ'] == F(check['exact_relative_error']) <
                F(1, 50), 'B2_AXIS_NOT_PASS:' + cid + ':' + axis)
        measured[cid, axis] = row['measured_RZ']
    return measured


def model_strata(model, terminal):
    require(d0.digest(model) == MODEL_SHA and model['S2'] == 'a+b*q+d*prefix*q' and
            model['S4'] == 'a+3*b*q+3*d*prefix*q' and model['profile_sha256'] == PROFILE and
            model['compiler_hash'] == COMPILER and model['old_B0_used'] is False and
            model['S4_measured_validation'] is False, 'FIXED_B1_MODEL')
    require(terminal['status'] == 'COMPLETE_MANDATORY_STOP' and terminal['model_sha256'] == MODEL_SHA,
            'B1_TERMINAL_COMPLETE')
    strata = {}
    for row in model['models']:
        key = row['method'], row['axis']
        require(key not in strata and row['status'] == 'S2_ANCHORS_AND_REGISTERED_HOLDOUT_PASS' and
                row['S4_status'] == 'UNVALIDATED_SCHEDULE_COMPOSITION_HYPOTHESIS' and
                row['full_grid_guaranteed'] is False and len(row['coefficients']) == 3, 'B1_STRATUM')
        strata[key] = [F(x) for x in row['coefficients']]
    require(set(strata) == {(m, a) for m in ('B0', 'B0I') for a in CHANNEL}, 'FOUR_B1_STRATA')
    return strata


def anchor_index(anchors):
    rows = anchors['rows']
    expected = {(m, p, q, a) for m in ('B0', 'B0I') for p in B1_FIT_PREFIXES
                for q in ((1, 8) if m == 'B0' else (1, 2, 4, 8)) for a in CHANNEL}
    keys = [(r['method'], r['prefix'], r['q'], r['axis']) for r in rows]
    require(len(rows) == 24 and len(set(keys)) == 24 and set(keys) == expected, 'EXACT_B1_24_MEASURED_ANCHORS')
    index = {}
    for r in rows:
        require(r['runtime_profile_sha256'] == PROFILE and r['compiler_hash'] == COMPILER and
                r['control'] == 'symmetric_directional' and r['formula'] == '2nd' and
                r['origin_SOURCE'] == ANCHOR_SOURCE[r['method']] and type(r['saved_RZ']) is int and
                r['saved_RZ'] > 0 and r['evidence'] and all(x is True for x in r['evidence'].values()),
                'B1_ANCHOR_LINEAGE')
        index[f"{r['method']}_p{r['prefix']}_q{r['q']}_2nd", r['axis']] = r['saved_RZ']
    return index


def wrapper_pairs(wrappers, cid, phase, allowed_sources):
    rows = [r for r in wrappers if r['phase'] == phase and r['cell_id'] == cid]
    require(all(r['source'] in allowed_sources for r in rows), 'WRAPPER_SOURCE_BINDING')
    return d0.paired_rows(rows, cell_id=cid, phase=phase, compiler_hash=COMPILER)


def full_PF_units(wrappers, cells, allowed_sources, saved_axis_means):
    """Legacy saved N=1 measurement; old runtime profile kept unknown, never backfilled."""
    units = {}
    for cid, cell in cells.items():
        if cell['method'] != 'B1':
            continue
        pairs = wrapper_pairs(wrappers, cid, 'exploration', allowed_sources)
        require(len(pairs) == 1 and pairs[0]['cosine']['replica'] == 0, 'LEGACY_FULL_PF_N1_PAIR')
        for axis in CHANNEL:
            value = pairs[0][axis]['RZ']
            require(type(value) is int and value > 0, 'LEGACY_FULL_PF_RZ')
            for means in saved_axis_means.get(cid, []):
                require(means[axis] == value, 'LEGACY_FULL_PF_SAVED_MEAN_MISMATCH')
            units[cid, axis] = value
    return units


def PR_pairs(wrappers, union, registrations, allowed_sources):
    """Confirmation n32 only. Exploration n8 rows are never pooled."""
    seeds = {}
    for item in registrations:
        task = item['task']
        key = task['cell_id'], task['replica']
        require(key not in seeds and task['phase'] == 'confirmation' and task['cell_id'] in union,
                'CONFIRMATION_REGISTRATION')
        seeds[key] = task['seed']
    require(set(seeds) == {(cid, i) for cid in union for i in range(CONFIRMATION_N)},
            'CONFIRMATION_REGISTRATION_COVERAGE')
    result = {}
    for cid in union:
        pairs = wrapper_pairs(wrappers, cid, 'confirmation', allowed_sources)
        if not pairs:
            result[cid] = None
            continue
        require(len(pairs) == CONFIRMATION_N and
                [p['cosine']['replica'] for p in pairs] == list(range(CONFIRMATION_N)),
                'CONFIRMATION_N32_PAIRS_REQUIRED')
        require(all(p['cosine']['seed'] == seeds[cid, p['cosine']['replica']] for p in pairs),
                'CONFIRMATION_SEED_BINDING')
        result[cid] = pairs
    return result


def minimum(rows):
    unresolved = sorted(r['cell_id'] for r in rows if r['eligible'] is None)
    if unresolved:
        return dict(status='UNRESOLVED_FULL_DOMAIN', best=None, unresolved=unresolved)
    valid = [r for r in rows if r['eligible'] is True]
    if not valid:
        return dict(status='NO_ELIGIBLE_CANDIDATE', best=None)
    missing = sorted(r['cell_id'] for r in valid if r['G_exact'] is None)
    if missing:
        return dict(status='UNRESOLVED_ELIGIBLE_COST', best=None, missing=missing)
    best = min(valid, key=lambda r: (F(r['G_exact']), r['cell_id']))
    return dict(status='CONDITIONAL_POINT_MINIMUM',
                best={k: best[k] for k in ('cell_id', 'G_exact', 'cost_status', 'axis_cost_status',
                                           'G_standard_error') if k in best},
                exact_ties=sorted(r['cell_id'] for r in valid if F(r['G_exact']) == F(best['G_exact'])))


def rare_layers(histograms):
    out = {}
    for phase, histogram in histograms.items():
        out[phase] = [dict(order=order, observed_count=histogram.get(str(order), 0),
                           status='UNOBSERVED' if histogram.get(str(order), 0) == 0 else 'OBSERVED_FINITE_SAMPLE',
                           population_layer_cost=None, mean_cost_upper_bound_proved=False)
                      for order in (0, 2, 4)]
    return out


def evaluate(data, registry, permit, *, stages, guard=lambda: None):
    """One indexed pass; gates STOP, unresolved components stay explicit nulls."""
    permit.require(data.get('kind'))
    stages = tuple(stages)
    require(stages in (STAGES[:1], STAGES), 'REGISTERED_STAGE_SET')
    cells, union, records = bind_domain(data, registry)
    old_ids = [c['id'] for c in registry['old']]
    all_ids = old_ids + [c['id'] for c in registry['additional']]
    ordered = all_ids + union
    measured_B2 = None
    if 'COST_COMPARISON' in stages:
        # Before any accounting, model prediction or candidate selection.
        measured_B2 = b2_gate(data['B2'], data['B2_saved_B0_registry'])
    signed, unresolved = signed_index(data['signed'], cells)
    reference = data['full_reference']
    usual = usual_index(data['saved_usual'], ordered)
    for cid in ordered:
        guard()
        for e in EPSILONS:
            for f in FACTORS:
                require(frozen_accounting(records[cid], e, f) == usual[cid, e, f],
                        'SAVED_USUAL_ACCOUNTING_NOT_REPRODUCED:' + cid + ':' + str(e) + ':' + str(f))
    PR_checks = {}
    for cid in union:
        PR_normalization(records[cid])
        PR_checks[cid] = validate_PR(records[cid], reference)
    axis_rows, envelopes = [], {}
    for cid in ordered:
        guard()
        method = records[cid]['cell']['method']
        for axis in AXES:
            reason, parts = None, None
            if method == 'B2':
                if PR_checks[cid]['first_failure'] is not None:
                    reason = 'PR_COMPONENT_VALIDATION_NOT_PASS:' + PR_checks[cid]['first_failure']
                else:
                    parts = PR_checks[cid]['signed_parts'][axis]
            elif (cid, axis) in unresolved:
                reason = unresolved[cid, axis]
            else:
                parts = signed[cid, axis]['signed_parts']
            value = None
            if reason is None:
                try:
                    value = envelope(method, parts, records[cid]['error_decomposition']['total_signed'][axis])
                except ValueError as exc:
                    if str(exc) in NONFINITE:
                        raise
                    reason = 'COMPONENT_BINDING:' + str(exc)
            envelopes[cid, axis] = value
            axis_rows.append(dict(cell_id=cid, method=method, axis=axis,
                status='EMPIRICAL_NONCANCELLATION_ENVELOPE' if value else 'UNRESOLVED_COMPONENTS',
                unresolved_reason=reason, usual_signed_bias=value and value['usual_signed'],
                empirical_noncancellation_bias=value and value['empirical_noncancellation'],
                reconstructed_from_saved_endpoints=method == 'B0I' and bool(
                    signed.get((cid, axis), {}).get('reconstructed')),
                rigorous_bias_bound=None))
    rows = []
    for cid in ordered:
        guard()
        values = [envelopes[cid, a] for a in AXES]
        for e in EPSILONS:
            for f in FACTORS:
                saved = usual[cid, e, f]
                usual_ok, usual_shots = eligibility(saved)
                row = dict(cell_id=cid, method=records[cid]['cell']['method'], epsilon=e, factor=f,
                           in_old32=cid in old_ids, in_strengthened184=cid in cells, PR_confirmation=cid in union,
                           bias_kind='EMPIRICAL_NONCANCELLATION_DIAGNOSTIC',
                           usual_saved=dict(eligibility=label(usual_ok), N_total=saved['N_total'],
                                            source='SAVED_USUAL_SIGNED_BIAS_ACCOUNTING_REPRODUCED'),
                           rigorous_bias_guarantee=None, G_exact=None, G_standard_error=None)
                if None in values:
                    row.update(eligible=None, eligibility='UNRESOLVED_COMPONENTS', accounting=None,
                               shots=None, N_total=None, shot_ratio_to_usual=None)
                else:
                    a = frozen_accounting(records[cid], e, f,
                                          {ax: v['empirical_noncancellation'] for ax, v in zip(AXES, values)})
                    ok, shots = eligibility(a)
                    row.update(eligible=ok, eligibility=label(ok), accounting=a, shots=shots,
                               N_total=a['N_total'] if ok else None,
                               shot_ratio_to_usual=str(F(a['N_total'], saved['N_total']))
                               if ok and usual_ok else None)
                rows.append(row)
    result = dict(schema='H8_NONCANCELLATION_DIAGNOSTIC_RESULT_V1', scope=SCOPE, stages_executed=list(stages),
                  gates=dict(saved_usual_rows_reproduced=len(usual), PR_normalization_bound=len(union),
                             PR_component_checks=PR_checks, B2_accepted=measured_B2 is not None),
                  axis_envelopes=axis_rows, accounting_rows=rows,
                  unresolved_axes=sum(1 for r in axis_rows if r['status'] != 'EMPIRICAL_NONCANCELLATION_ENVELOPE'),
                  rare_layers=rare_layers(data['Taylor_orders']), lineage_limitations=data['lineage_limitations'],
                  rigorous_bias_guarantee=None, new_signal_compile_sampling_SCF_DF_state_fit=0, PR_sampling=0,
                  next_stage_authorized=False, automatic_retry=False, mandatory_STOP=True)
    if measured_B2 is None:
        result.update(status='COMPLETE_NONCANCELLATION_SHOTS_ONLY_MANDATORY_STOP',
                      cost_stage=dict(status='NOT_EXECUTED', reason='COST_STAGE_NOT_AUTHORIZED_OR_B2_NOT_BOUND',
                                      comparisons=None, PR_reduction_percent=None))
        return result
    result['cost_stage'] = costs(data, cells, union, records, rows, usual, measured_B2, old_ids, all_ids, guard)
    result['status'] = 'COMPLETE_NONCANCELLATION_SHOTS_AND_COST_COMPARISON_MANDATORY_STOP'
    return result


def costs(data, cells, union, records, rows, usual, measured_B2, old_ids, all_ids, guard):
    strata = model_strata(data['B1_model'], data['B1_terminal'])
    anchors = anchor_index(data['B1_anchors'])
    allowed = data['wrapper_binding']['allowed_sources']
    require(data['wrapper_binding']['compiler'] == COMPILER, 'WRAPPER_COMPILER_BINDING')
    legacy = full_PF_units(data['wrappers'], cells, allowed, data['full_PF_saved_axis_means'])
    pairs = PR_pairs(data['wrappers'], union, data['confirmation']['registrations'], allowed)
    saved_metrics = {}
    for item in data['PR_saved_metrics']:
        key = item['cell_id'], float(item['epsilon']), float(item['factor'])
        require(key not in saved_metrics and item['sample_phase'] == 'confirmation' and
                item['sample_count'] == CONFIRMATION_N, 'PR_SAVED_METRIC_PHASE_N32')
        saved_metrics[key] = item
    require(set(saved_metrics) == {(c, e, f) for c in union for e in EPSILONS for f in FACTORS},
            'PR_SAVED_METRIC_COVERAGE')
    for (cid, e, f), item in saved_metrics.items():
        guard()
        ok, shots = eligibility(usual[cid, e, f])
        if item['status'] != 'DERIVED_SAVED_SCALAR':
            # A saved MISSING cost must agree with absent pairs; it is never a zero cost.
            require(ok is not True or (item['status'] == 'MISSING' and pairs[cid] is None),
                    'PR_SAVED_ELIGIBLE_COST_MISSING')
            continue
        require(ok is True and pairs[cid] is not None, 'PR_SAVED_METRIC_WITHOUT_PAIRS')
        again = d0.weighted_metric(pairs[cid], {a: shots[CHANNEL[a]] for a in CHANNEL}, 'RZ',
                                   point_rule='sum_axis_products')
        require(again == item['metrics']['RZ'], 'PR_SAVED_PAIRED_COST_NOT_REPRODUCED:' + cid)
    cache = {}

    def unit(cell, axis):
        key = cell['id'], axis
        if key not in cache:
            if key in measured_B2:
                value = dict(RZ=str(measured_B2[key]), status='B2_MEASURED_REGISTERED_S4_AXIS', measured=True)
            elif key in anchors:
                value = dict(RZ=str(anchors[key]), status='B1_REGISTERED_ANCHOR_MEASURED', measured=True)
            elif cell['method'] == 'B1':
                require(key in legacy, 'ELIGIBLE_FULL_PF_COST_MISSING')
                value = dict(RZ=str(legacy[key]), status='LEGACY_FULL_PF_SAVED_N1_MEASURED', measured=True,
                             legacy_runtime_profile='ORIGINAL_OR_UNKNOWN_NOT_BACKFILLED')
            else:
                a, b, d = strata[cell['method'], axis]
                require(cell['formula'] in ('2nd', '4th'), 'REGISTERED_FORMULA')
                pieces = 1 if cell['formula'] == '2nd' else 3
                rz = a + pieces*cell['q']*(b + d*cell['prefix'])
                require(rz > 0, 'NONPOSITIVE_MODEL_COST')
                value = dict(RZ=str(rz), measured=False, model_sha256=MODEL_SHA,
                             status='FIXED_B1_S2_MODEL_ESTIMATE' if pieces == 1 else
                             'FIXED_B1_S4_MODEL_ESTIMATE_B2_FOUR_AXES_ONLY',
                             prefix_outside_B1_fit_range=not B1_FIT_PREFIXES[0] <= cell['prefix'] <= B1_FIT_PREFIXES[1],
                             full_grid_error_guaranteed=False)
            cache[key] = value
        return cache[key]

    current = {}
    for row in rows:
        guard()
        current[row['cell_id'], row['epsilon'], row['factor']] = row
        if row['eligible'] is not True:
            row['cost_status'] = 'NOT_ELIGIBLE_OR_UNRESOLVED_NO_COST'
            continue
        shots = row['shots']
        if row['method'] == 'B2':
            if pairs[row['cell_id']] is None:
                row.update(cost_status='PR_PAIRED_COST_MISSING', G_exact=None, G_standard_error=None)
                continue
            metric = d0.weighted_metric(pairs[row['cell_id']], {a: shots[CHANNEL[a]] for a in CHANNEL}, 'RZ',
                                        point_rule='sum_axis_products')
            value = F(int(metric['G_exact']['numerator']), int(metric['G_exact']['denominator']))
            row.update(cost_status='PR_FINITE_SAMPLE_MEAN_CONFIRMATION_N32', paired_samples=metric['paired_samples'],
                       G_standard_error=metric['G_standard_error'], uncertainty_scope='PAIRED_SAMPLE_SE_NOT_CI',
                       population_mean_guaranteed=False, axis_cost_status=None)
        else:
            axes = {a: unit(cells[row['cell_id']], a) for a in CHANNEL}
            value = sum(F(axes[a]['RZ'])*shots[CHANNEL[a]] for a in CHANNEL)
            row.update(axis_costs=axes, axis_cost_status=sorted({x['status'] for x in axes.values()}),
                       cost_status='SAVED_MEASURED_POINT' if all(x['measured'] for x in axes.values())
                       else 'MODEL_CONDITIONAL_POINT')
        require(value > 0, 'NONPOSITIVE_WEIGHTED_COST')
        row['G_exact'] = str(value)
    comparisons = []
    for e in EPSILONS:
        for f in FACTORS:
            guard()
            pr = minimum([current[c, e, f] for c in union])
            scopes = {}
            for name, ids in (('old32', old_ids), ('strengthened184', all_ids)):
                det = minimum([current[c, e, f] for c in ids])
                ratio = (F(pr['best']['G_exact'])/F(det['best']['G_exact'])
                         if pr['best'] is not None and det['best'] is not None else None)
                scopes[name] = dict(deterministic=det, confirmed_PR=pr,
                    PR_over_deterministic_exact=str(ratio) if ratio is not None else None,
                    PR_reduction_percent_exact=str(100*(1-ratio)) if ratio is not None else None,
                    baseline_policy='SAME_MEASURED_FIRST_FIXED_B1_MODEL_UNIT_POLICY',
                    all_candidate_measured_minimum_claimed=False)
            before, after = (scopes[s]['PR_reduction_percent_exact'] for s in ('old32', 'strengthened184'))
            comparisons.append(dict(epsilon=e, factor=f, P=0, bias_kind='EMPIRICAL_NONCANCELLATION_DIAGNOSTIC',
                scopes=scopes, reduction_change_percentage_points_exact=str(F(after)-F(before))
                if before is not None and after is not None else None))
    return dict(status='COMPLETE', comparisons=comparisons, unit_costs=[dict(cell_id=k[0], axis=k[1], **v)
                for k, v in sorted(cache.items())], model_sha256=MODEL_SHA, state_preparation_P=0,
                B2_measured_axes=[dict(cell_id=k[0], axis=k[1], RZ=v) for k, v in sorted(measured_B2.items())],
                PR_policy='CONFIRMATION_UNION6_N32_ONLY_EXPLORATION_N8_NOT_POOLED',
                refit=False, rigorous_bias_or_PR_population_guaranteed=False)

"""Cost-only adapter over completed shots; no bias/shot/signal evaluation.

The frozen noncancellation cost kernel supplies measured-first exact arithmetic,
confirmation-pair reproduction and newly weighted sample SE. The ordinary Phase C
result is read as saved evidence, never recomputed or overwritten.
"""
from copy import deepcopy
import csv
from fractions import Fraction as F
import io
import math
import h8_noncancellation as n
import h8_phase_c_accounting as phase_c

require = n.require
digest = phase_c.digest
MODEL_SHA = phase_c.MODEL_SHA
B2_RESULT = phase_c.B2_RESULT
SCOPE = 'H8_EMPIRICAL_NONCANCELLATION_COST_COMPARISON_V3'
BIAS = 'EMPIRICAL_NONCANCELLATION'
SHOT_SOURCE = 'cbfd7cacad8a5828c0dbfd284a99ef0871860e66'
SHOT_PLAN = '002f9edbf35506fca256431df641552d7c319581f306f13a316b736e07897e6d'
SHOT_GRANT = '020b4ca8bc6a456d452d0fe070881d79fcc29d3c6a5147bca43c452ae82a9322'
SHOT_INPUT = 'b3ce74d5e2c40ce7c08164caf96818f7832fb6cffe0b661b7fec6d87bc9cbbbe'
SHOT_RESULT = '3f902229cef9da26cb6feeff09eb35860b6115a6'
USUAL_SOURCE = '931755c4c84aa0751dbde7ddf84cb5187b5172ae'
USUAL_RESULT = '6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086'
SHOT_STATUS = 'COMPLETE_NONCANCELLATION_SHOTS_ONLY_MANDATORY_STOP'


def csv_rows(raw):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8')))
    require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'CSV_DUPLICATE_COLUMNS')
    rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), 'CSV_ROW_SHAPE')
    numeric_fields = ('epsilon', 'allowance_factor', 'shots_real', 'shots_imag',
        'noncancellation_N_total', 'headroom_real', 'headroom_imag', 'u_over_headroom_real',
        'u_over_headroom_imag', 'usual_signed_bias_abs_total', 'empirical_noncancellation_bias', 'signed_difference')
    require(all(math.isfinite(float(row[k])) for row in rows for k in numeric_fields if row.get(k) not in ('', None)),
            'NONFINITE_CSV')
    return rows


def index(rows, key):
    return phase_c.unique(rows, key)


def row_key(row):
    return row['cell_id'], float(row['epsilon']), float(row['factor'])


def completed_shots(proof, input_manifest, registry):
    """Compare saved projections/receipts only. Never call a shot rule."""
    b, summary, raw, terminal, native = (proof[k] for k in ('binding', 'summary', 'raw', 'terminal', 'native'))
    require(b['SOURCE'] == summary['SOURCE'] == raw['SOURCE'] == terminal['SOURCE'] == SHOT_SOURCE and
            b['plan_digest'] == summary['plan_digest'] == raw['plan_sha256'] == SHOT_PLAN and
            b['grant']['digest'] == summary['grant_digest'] == raw['grant_sha256'] == SHOT_GRANT and
            b['input_manifest_sha256'] == SHOT_INPUT, 'COMPLETED_SHOT_SOURCE_PLAN_GRANT_INPUT')
    require(b['post_execution_binding_verified'] is raw['post_execution_binding_verified'] is True and
            b['one_shot_lock_retained'] is True and b['grant']['authorized_stages'] == ['NONCANCELLATION_SHOTS'] and
            b['grant']['cost_stage_authorized'] is False and b['grant']['one_shot'] is True and
            b['grant']['automatic_retry'] is False, 'SHOT_APPROVAL_SCOPE')
    require(summary['status'] == raw['status'] == terminal['status'] == SHOT_STATUS and
            summary['stages_executed'] == raw['stages_executed'] == terminal['stages_executed'] == ['NONCANCELLATION_SHOTS'] and
            terminal['first_stop'] is None and terminal['mandatory_STOP'] is True and
            terminal['automatic_retry'] is False and terminal['next_stage_authorized'] is False and
            native['native_exit_code'] == 0 and native['reaped'] is True and native['pid_absent_after_wait'] is True and
            terminal['all_children_reaped'] is True and native['children_started_by_runner'] == 0,
            'COMPLETED_SHOT_NATIVE_TERMINAL')
    require(summary['rows'] == 1900 and summary['axes'] == 380 and
            summary['unresolved_axes'] == raw['unresolved_axes'] == 0 and
            summary['gates']['saved_usual_rows_reproduced'] == raw['gates']['saved_usual_rows_reproduced'] == 1900 and
            raw['cost_stage']['status'] == summary['cost_stage']['status'] == 'NOT_EXECUTED' and
            raw['rigorous_bias_guarantee'] is None, 'SHOT_COMPLETENESS_AND_NO_COST_STAGE')
    expected_receipts = {r['role']: (r['bytes'], r['sha256']) for r in input_manifest['files']}
    require(len(expected_receipts) == len(input_manifest['files']) == 22, 'SHOT_INPUT_ROLES')
    for receipts in (b['input_receipts'], raw['input_receipts']):
        require(len(receipts) == 22 and {r['role']: (r['bytes'], r['sha256']) for r in receipts} == expected_receipts,
                'SHOT_ORIGINAL_INPUT_RECEIPTS')
    cells = {c['id']: c for c in registry['old'] + registry['additional']}
    union = set(registry['confirmation_ids'])
    ids = set(cells) | union
    rows = index(raw['accounting_rows'], row_key)
    require(len(rows) == 1900 and set(rows) == {(cid, e, f) for cid in ids for e in n.EPSILONS for f in n.FACTORS},
            'EXACT_SAVED_1900_NONCANCELLATION_ROWS')
    projected = csv_rows(proof['csv'])
    csv_index = index(projected, lambda x: (x['candidate_id'], float(x['epsilon']), float(x['allowance_factor'])))
    require(set(csv_index) == set(rows), 'SHOT_CSV_EXACT_DOMAIN')
    for key, row in rows.items():
        saved = csv_index[key]
        require(row['bias_kind'] == saved['bias_kind'] == 'EMPIRICAL_NONCANCELLATION_DIAGNOSTIC' and
                row['rigorous_bias_guarantee'] is None and saved['rigorous_bias_guarantee'] == '' and
                row['G_exact'] is None and row['G_standard_error'] is None, 'SHOT_NAMESPACE_NOT_COST_RESULT')
        method = cells[key[0]]['method'] if key[0] in cells else 'B2'
        group = 'old32' if key[0] in {c['id'] for c in registry['old']} else 'additional152' if key[0] in cells else 'PR_confirmation_union6'
        require(row['method'] == saved['method'] == method and saved['set'] == group and
                saved['noncancellation_eligibility'] == row['eligibility'], 'SHOT_CSV_METHOD_SCOPE')
        require(type(row['eligible']) is bool and row['eligible'] == (row['eligibility'] == 'ELIGIBLE'),
                'SHOT_UNRESOLVED_OR_CONTRADICTORY')
        if row['eligible']:
            require(row['usual_saved']['eligibility'] == 'ELIGIBLE', 'NONCANCELLATION_NOT_USUAL_ELIGIBLE')
            shots = row['shots']
            require(set(shots) == {'real', 'imag'} and all(type(v) is int and 0 < v < 2**52 for v in shots.values()) and
                    row['N_total'] == sum(shots.values()), 'SAVED_SHOT_INTEGERS')
            for axis in ('real', 'imag'):
                a = row['accounting']['axes'][axis]
                require(a['shots'] == shots[axis] and str(shots[axis]) == saved['shots_'+axis] and
                        math.isfinite(a['headroom']) and a['headroom'] > 0 and
                        math.isfinite(a['u_over_headroom']) and 0 <= a['u_over_headroom'] <= .01 and
                        float(saved['headroom_'+axis]) == a['headroom'] and
                        float(saved['u_over_headroom_'+axis]) == a['u_over_headroom'], 'SAVED_SHOT_AXIS_GUARD')
            require(saved['noncancellation_N_total'] == str(row['N_total']), 'SHOT_CSV_TOTAL')
        else:
            require(row['shots'] is None and row['N_total'] is None and
                    saved['noncancellation_N_total'] == '',
                    'INELIGIBLE_COST_NOT_ZERO')
            # A jointly ineligible row can retain one individually eligible axis.
            # Preserve that saved projection without treating it as a circuit cost.
            for axis in ('real', 'imag'):
                axis_shots = row['accounting']['axes'][axis]['shots']
                require(saved['shots_'+axis] == (str(axis_shots) if axis_shots is not None else ''),
                        'INELIGIBLE_PARTIAL_AXIS_PROJECTION')
    axes = index(raw['axis_envelopes'], lambda x: (x['cell_id'], x['axis']))
    projected_axes = index(csv_rows(proof['axis_csv']), lambda x: (x['candidate_id'], x['axis']))
    require(set(axes) == set(projected_axes) == {(cid, a) for cid in ids for a in n.AXES}, 'EXACT_380_SAVED_AXES')
    for key, row in axes.items():
        require(row['status'] == 'EMPIRICAL_NONCANCELLATION_ENVELOPE' and row['rigorous_bias_bound'] is None and
                projected_axes[key]['status'] == row['status'] and
                float(projected_axes[key]['usual_signed_bias_abs_total']) == row['usual_signed_bias'] and
                float(projected_axes[key]['empirical_noncancellation_bias']) == row['empirical_noncancellation_bias'] and
                all(math.isfinite(row[k]) for k in ('usual_signed_bias', 'empirical_noncancellation_bias')),
                'SAVED_AXIS_PROJECTION')
    checks = csv_rows(proof['PR_csv'])
    expected = {(cid, x['name']): (x['passed'], x['signed_difference'])
                for cid, group in raw['gates']['PR_component_checks'].items() for x in group['checks']}
    actual = index(checks, lambda x: (x['candidate_id'], x['check']))
    require(len(checks) == len(expected) == 84 and set(actual) == set(expected) and
            all(actual[k]['passed'] == 'True' and expected[k][0] is True and
                float(actual[k]['signed_difference']) == expected[k][1] and
                math.isfinite(float(actual[k]['signed_difference'])) for k in expected), 'SAVED_PR_COMPONENT_CHECKS')
    return [deepcopy(row) for row in rows.values()]


def evaluate(data, registry, permit, *, take=lambda key: None, emit=lambda kind, value: None, guard=lambda: None):
    permit.require(data['kind'])
    measured = phase_c.b2_gate(data['B2'])
    cells, union, records = n.bind_domain(data, registry)
    require(set(union) == set(registry['confirmation_ids']), 'CONFIRMATION_UNION_CHANGED')
    rows = completed_shots(data['completed_shots'], data['shot_manifest'], registry)
    usual = n.usual_index(data['saved_usual'], [*cells, *union])
    for row in rows:
        ok, _ = n.eligibility(usual[row_key(row)])
        require(n.label(ok) == row['usual_saved']['eligibility'], 'ORIGINAL_USUAL_ELIGIBILITY_JOIN')
    original = data['usual_result']
    require(original['status'] == 'COMPLETE_MANDATORY_STOP' and original['bias_kind'] == phase_c.USUAL_BIAS and
            original['noncancellation_diagnostic_used'] is False and original['model_sha256'] == MODEL_SHA and
            original['B2_binding']['status'] == phase_c.B2_STATUS, 'SAVED_USUAL_PHASE_C_RESULT')
    old_ids = [x['id'] for x in registry['old']]
    take('reaccounting_pass')
    # Reuse the existing cost-only kernel. In particular do NOT call n.evaluate,
    # envelope, frozen_accounting, or d0.accounting here.
    cost = n.costs(data, cells, union, records, rows, usual,
                   {k: v['RZ'] for k, v in measured.items()}, old_ids, list(cells), guard)
    saved_rows = index(data['usual_rows'], row_key)
    require(set(saved_rows) == {row_key(x) for x in rows}, 'SAVED_PHASE_C_1900_DOMAIN')
    for row in rows:
        guard(); take('saved_row')
        key = row_key(row); normal = saved_rows[key]
        if row['eligible']:
            require(normal['eligible'] is True and row['G_exact'] is not None, 'ELIGIBLE_COST_OR_USUAL_ROW_MISSING')
            if row['method'] != 'B2':
                for axis, unit in row['axis_costs'].items():
                    prior = normal['axis_costs'][axis]
                    require(F(unit['RZ']) == F(prior['RZ']) and unit['measured'] == prior['measured'],
                            'UNIT_COST_DIFFERS_FROM_FIXED_PHASE_C_POLICY')
                    unit['Phase_C_saved_unit_evidence'] = deepcopy(prior)
                    if (row['cell_id'], axis) in measured:
                        unit['final_B2_axis_evidence'] = deepcopy(measured[row['cell_id'], axis])
            else:
                require(row['paired_samples'] == 32 and math.isfinite(row['G_standard_error']), 'PR_N32_NEW_WEIGHTED_SE')
                take('PR_weighted_cost_and_SE')
        row.update(bias_kind=BIAS, rigorous_bias_guarantee=None, population_mean_guaranteed=False,
                   state_preparation_P=0, shots_reused_without_recalculation=True)
        emit('accounting_row', row)
    baseline = index(original['comparisons'], lambda x: (x['epsilon'], x['factor']))
    require(set(baseline) == {(e, f) for e in n.EPSILONS for f in n.FACTORS}, 'USUAL_PHASE_C_EXACT_TEN_CONDITIONS')
    for comparison in cost['comparisons']:
        saved = baseline[comparison['epsilon'], comparison['factor']]
        comparison.update(bias_kind=BIAS, usual_bias_kind=phase_c.USUAL_BIAS, usual_result_commit=USUAL_RESULT)
        differences = {}
        for scope, result in comparison['scopes'].items():
            take('comparison')
            old = saved['scopes'][scope]
            a, b = result['PR_reduction_percent_exact'], old['PR_reduction_percent_exact']
            differences[scope] = dict(usual_saved=deepcopy(old),
                reduction_change_percentage_points_exact=str(F(a)-F(b)) if a is not None and b is not None else None,
                deterministic_winner_changed=(result['deterministic'].get('best') or {}).get('cell_id') !=
                    (old['deterministic'].get('best') or {}).get('cell_id'),
                PR_winner_changed=(result['confirmed_PR'].get('best') or {}).get('cell_id') !=
                    (old['confirmed_PR'].get('best') or {}).get('cell_id'))
        comparison['difference_from_saved_usual'] = differences
    require(all(not x[k]['status'].startswith('UNRESOLVED') for row in cost['comparisons']
                for x in row['scopes'].values() for k in ('deterministic', 'confirmed_PR')),
            'UNRESOLVED_FULL_DOMAIN_NO_CONFIRMED_REDUCTION')
    for unit in cost['unit_costs']:
        if not unit['measured']:take('model_axis_estimate')
    limitations=deepcopy(data['lineage_limitations'])
    limitations['prior_saved_S4_model_validation_scope']=limitations.get('S4_model_validation_scope')
    limitations['S4_model_validation_scope']='FOUR_REGISTERED_B2_AXES_ONLY_FINAL_PASS_BOUND_NOT_FULL_GRID'
    return dict(status='COMPLETE_MANDATORY_STOP', scope=SCOPE, stages_executed=['COST_COMPARISON'],
        bias_kind=BIAS, usual_bias_kind=phase_c.USUAL_BIAS, rigorous_bias_guarantee=None,
        comparisons=cost['comparisons'], gates=cost['gates'], accounting_rows=len(rows),
        unit_costs=cost['unit_costs'], model_sha256=MODEL_SHA, state_preparation_P=0,
        B2_result_commit=B2_RESULT, B2_status=phase_c.B2_STATUS,
        completed_shots_result_commit=SHOT_RESULT, completed_shots_SOURCE=SHOT_SOURCE,
        usual_result_commit=USUAL_RESULT, usual_SOURCE=USUAL_SOURCE,
        shots_recalculated=0, ordinary_Phase_C_recalculated=0, PR_SE_reweighted=True,
        rare_layers=deepcopy(data['completed_shots']['raw']['rare_layers']),
        lineage_limitations=limitations,
        S4_model_status='FOUR_REGISTERED_AXES_PASS_ONLY_UNMEASURED_GRID_MODEL_ESTIMATES',
        all_candidate_costs_measured=False, full_grid_model_error_guaranteed=False,
        rigorous_bias_or_PR_population_guaranteed=False, new_signal_compile_sampling_SCF_DF_state_fit=0,
        automatic_retry=False, next_stage_authorized=False, mandatory_STOP=True)

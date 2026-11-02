"""Independent H8 -> frozen D0 adapter, with an artificial-only analysis entry.

No molecular arrays, native imports, IO runner, launch or grant. Real-data
equality/reaccounting needs a future separately approved bounded execution.
"""
from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path


D0_SOURCE = 'fa75c7997e135da7b7a0c4e23949e47252bd7a2e'
FROZEN_FILES = {
    'd0_saved_value_reanalysis_v1.py': '1ded644585bfabe4f8f28be26b92e8c09f00c4e8f9712b4cf45e0233cfc8737e',
    'd0_frozen_shot_rule_v1.py': 'ed3f57937a231719583c2893677b7e82f934dd98bda35cd37fc0f17f83d4279f',
}
for _name, _sha in FROZEN_FILES.items():
    if hashlib.sha256(Path(__file__).with_name(_name).read_bytes()).hexdigest() != _sha:
        raise ValueError('FROZEN_D0_SOURCE_CHANGED:'+_name)
_spec = importlib.util.spec_from_file_location('pre_review_frozen_d0',
    Path(__file__).with_name('d0_saved_value_reanalysis_v1.py'))
d0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d0)
ORIGINAL_EPSILONS = (.05, .01, .005, .001)
EPSILONS = (*ORIGINAL_EPSILONS, .0005)
FACTORS = (1.0, 10.0)
SEED_NAMESPACE = 'H8_CORE_RESOURCE_MAP_V1_20261013'


def task_seed(binding, cell_id, phase, replica):
    key = f"{SEED_NAMESPACE}|{binding['seed_source']}|{binding['input']}|{cell_id}|{phase}|{replica}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def registry():
    result = []
    def add(method, prefix, q, formula='2nd', r=None, K=None):
        cid = f'{method}_p{prefix}_q{q}_{formula}' + (f'_r{r}_K{K}' if r else '')
        result.append(dict(id=cid, method=method, prefix=prefix, q=q,
            formula=formula, r=r, R=q*r if r else None, K=K))
    for q in (1, 2, 4, 8):
        for p in (7, 14, 21):
            add('B0', p, q)
        for formula in ('2nd', '4th'):
            add('B1', 28, q, formula)
        for p in (7, 14, 21):
            for r in (1, 2, 4):
                for K in (2, 4):
                    add('B2', p, q, r=r, K=K)
    return result


def validate_inputs(signals, wrappers, selection, binding):
    """Structural validation only: no shots or costs are calculated here."""
    if binding.get('analysis_source') != D0_SOURCE:
        raise ValueError('ANALYSIS_SOURCE_BINDING')
    for name, length in [('source', 40), ('seed_source', 40), ('result', 40),
            ('input', 64), ('compiler', 64)]:
        value = binding.get(name)
        if (not isinstance(value, str) or len(value) != length or
                any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('INPUT_BINDING:'+name)
    if d0.digest(selection) != binding.get('selection_digest'):
        raise ValueError('SELECTION_BINDING')
    if selection.get('signal_digest') != d0.digest(signals):
        raise ValueError('SIGNAL_SELECTION_BINDING')
    if (selection.get('fixed_before_confirmation') is not True or
            selection.get('exploration_confirmation_pooled') is not False):
        raise ValueError('INDEPENDENT_SELECTION_REQUIRED')
    expected = {c['id']: c for c in registry()}
    if len(signals) != 92 or {s['cell']['id'] for s in signals} != set(expected):
        raise ValueError('H8_REGISTRY92')
    for s in signals:
        if s['cell'] != expected[s['cell']['id']]:
            raise ValueError('H8_CELL_DEFINITION')
        if set(s['precision_rows']) != {str(e) for e in ORIGINAL_EPSILONS}:
            raise ValueError('OLD_PRECISION_COVERAGE')
        if any(set(v) != {'1.0', '10.0'} for v in s['precision_rows'].values()):
            raise ValueError('OLD_FACTOR_COVERAGE')
        if s['allowance'].get('kind') != 'EMPIRICAL' or s['allowance'].get('certified') is not False:
            raise ValueError('EMPIRICAL_U_REQUIRED')
        for z in [s['reference'], s['signals']['corrected']]:
            for a in ('real', 'imag'):
                d0.finite(z[a], a)
        d0.finite(s['log_B'], 'log_B', minimum=0)
    selected = selection.get('selected_ids', [])
    if (len(selected) != 6 or len(set(selected)) != 6 or
            any(x not in expected or expected[x]['method'] != 'B2' for x in selected)):
        raise ValueError('H8_CONFIRMATION_UNION6')
    by_signal = {s['cell']['id']: s for s in signals}
    registrations = {}
    for item in selection['registrations']:
        task = item['task']
        key = (task['cell_id'], task['replica'])
        if (key in registrations or task['cell_id'] not in selected or
                task['cell'] != expected[task['cell_id']] or task['phase'] != 'confirmation' or
                type(task['replica']) is not int or not 0 <= task['replica'] < 32 or
                type(task['seed']) is not int or task['seed'] < 0 or
                task['seed'] != task_seed(binding, task['cell_id'], 'confirmation', task['replica']) or
                task.get('expected_signal') != by_signal[task['cell_id']]['signals']['corrected']):
            raise ValueError('CONFIRMATION_REGISTRATION')
        registrations[key] = task['seed']
    if set(registrations) != {(cid, i) for cid in selected for i in range(32)}:
        raise ValueError('CONFIRMATION_REGISTRATION_COVERAGE')
    seen = set()
    for row in wrappers:
        if (row['cell_id'] not in expected or row['phase'] not in ('exploration', 'confirmation') or
                row['source'] not in binding['allowed_wrapper_sources'] or
                row['compiler_hash'] != binding['compiler']):
            raise ValueError('WRAPPER_SOURCE_PHASE_BINDING')
        if (row['axis'] not in d0.AXIS_CHANNEL or row['control'] not in ('ordinary', 'symmetric_directional') or
                any(type(row[k]) is not int or row[k] < 0 for k in ('replica', 'seed', *d0.METRICS))):
            raise ValueError('WRAPPER_SCHEMA')
        key = (row['phase'], row['cell_id'], row['replica'], row['axis'], row['control'])
        if key in seen:
            raise ValueError('DUPLICATE_WRAPPER')
        seen.add(key)
        if row['phase'] == 'confirmation':
            if (row['control'] != 'symmetric_directional' or
                    registrations.get((row['cell_id'], row['replica'])) != row['seed']):
                raise ValueError('CONFIRMATION_SEED_BINDING')
        elif type(row['replica']) is not int or not 0 <= row['replica'] < (
                8 if expected[row['cell_id']]['method'] == 'B2' else 1):
            raise ValueError('EXPLORATION_REPLICA')
        elif row['seed'] != task_seed(binding, row['cell_id'], 'exploration', row['replica']):
            raise ValueError('EXPLORATION_SEED_BINDING')
    return dict(registry_count=92, confirmation_union=6, confirmation_scope_count=26,
        source_input_selection_bound=True, real_old_equality_executed=False)


def analyze_artificial(signals, wrappers, selection, binding, old_maps):
    """Exercise the proposed order using fabricated scalar fixtures only."""
    if binding.get('fixture_kind') != 'ARTIFICIAL' or any(
            s.get('fixture_kind') != 'ARTIFICIAL' for s in signals):
        raise ValueError('REAL_REACCOUNT_NOT_AUTHORIZED_BY_THIS_PROTOTYPE')
    validate_inputs(signals, wrappers, selection, binding)
    exploration = [r for r in wrappers if r['phase'] == 'exploration']
    confirmation = [r for r in wrappers if r['phase'] == 'confirmation']
    specs = {
        'H8_EXPLORATION92': dict(primary_phase='exploration', compiler_hashes=[binding['compiler']],
            point_rule='sum_axis_products'),
        'H8_CONFIRMATION_UNION6': dict(primary_phase='confirmation', compiler_hashes=[binding['compiler']],
            point_rule='sum_axis_products', confirmation_ids=selection['selected_ids']),
    }
    original = {}
    equality = {}
    # Both original scopes must PASS before any .0005/metric/P extension.
    for scope, spec in specs.items():
        rows = exploration if spec['primary_phase'] == 'exploration' else confirmation
        original[scope] = d0.analyze_scope(signals, rows, spec, ORIGINAL_EPSILONS, FACTORS, exploration)
        expected_keys = {(r['cell_id'], r['epsilon'], r['factor'], m)
            for r in original[scope] if r['status'] == 'DERIVED_SAVED_SCALAR' for m in d0.METRICS}
        actual_keys = [(r['cell_id'], r['epsilon'], r['allowance_factor'], r['metric'])
            for r in old_maps[scope]['rows']]
        if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
            raise ValueError('FULL_ORIGINAL_MAP_COVERAGE:'+scope)
        equality[scope] = d0.verify_original(signals, original[scope], old_maps[scope])
    result = {}
    for scope, spec in specs.items():
        rows = exploration if spec['primary_phase'] == 'exploration' else confirmation
        records = d0.analyze_scope(signals, rows, spec, EPSILONS, FACTORS, exploration)
        selections, curves = {}, {}
        for eps in EPSILONS:
            for factor in FACTORS:
                subset = [r for r in records if r['epsilon'] == eps and r['factor'] == str(factor)]
                for metric in d0.METRICS:
                    key = f'{eps}/{factor}/{metric}'
                    selections[key] = d0.family_selection(subset, metric)
                    lines = [dict(cell_id=r['cell_id'], intercept=r['metrics'][metric]['G_point'],
                        slope=r['N_total']) for r in subset if r['status'] == 'DERIVED_SAVED_SCALAR']
                    families = selections[key]
                    pairs = []
                    for baseline in ('B0', 'B1'):
                        ids = [families[x]['RZ_selected_cell'] for x in ('B2', baseline)]
                        if None not in ids:
                            a, b = [next(l for l in lines if l['cell_id'] == cid) for cid in ids]
                            pairs.append(dict(B2=ids[0], baseline=ids[1], crossing=d0.fixed_crossing(a, b)))
                    curves[key] = dict(fixed_RZ_selected_pairs=pairs, lower_envelope=d0.lower_envelope(lines))
        result[scope] = dict(records=records, selections=selections, P=curves)
    coverage = {}
    for key, families in result['H8_EXPLORATION92']['selections'].items():
        cid = families['B2']['metric_selected_cell']
        coverage[key] = dict(cell_id=cid, in_original_union6=cid in selection['selected_ids'],
            status='NO_ELIGIBLE_CANDIDATE' if cid is None else
                'COVERED' if cid in selection['selected_ids'] else 'OUTSIDE_ORIGINAL_UNION6')
    return dict(artificial_only=True, original_equality=equality, scopes=result,
        confirmation_coverage=coverage, population_winner_certified=False)


def unobserved_layer(order, observed_count, *, saved_probability=None):
    if type(order) is not int or order < 0 or type(observed_count) is not int or observed_count < 0:
        raise ValueError('RARE_LAYER_SCHEMA')
    if saved_probability is not None:
        p = d0.finite(saved_probability, 'saved_probability', minimum=0)
        if p > 1:
            raise ValueError('PROBABILITY_RANGE')
    return dict(order=order, observed_count=observed_count,
        status='UNOBSERVED' if observed_count == 0 else 'OBSERVED_FINITE_SAMPLE',
        saved_probability=saved_probability, population_layer_cost=None,
        mean_cost_upper_bound_proved=False)

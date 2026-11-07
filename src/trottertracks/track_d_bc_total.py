"""Fixed A/F/S scalar accounting; no sampler, circuit, fit or shot solver."""
import copy
import math

import numpy as np

from . import track_d_b6b_bc as bc

B6B = '7ffe8b281b76cef3a6fd06f43413904d6766d0ec'
C74_DIR = 'artifacts/track_d_c7_4_models/2026-11-07/run02/'
SCENARIOS = {'A': 'affine', 'F': 'frozen_C6a', 'S': 'sqrt_lane'}
AXES = ('X', 'Y')
OUTPUT_LABELS = [(method, g) for method in (*bc.METHODS, 'paired_delta') for g in bc.METRICS]


def premises():
    return {k: True for k in ('same_V', 'actual_adjoint', 'vacuum_eigenray_with_chi',
        'same_initial_system_state', 'same_whole_path_law', 'same_XY_convention',
        'fresh_independent_axis_trials', 'same_target_decoder', 'law_is_B4_ideal_product')}


def finite(value, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('NONFINITE_NUMERIC_INPUT')
    if nonnegative and value < 0:
        raise ValueError('NEGATIVE_ABSOLUTE_PRICE')
    return float(value)


def close(a, b):
    if not math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-7):
        raise ValueError('PAIRED_ACCOUNTING_IDENTITY')


def check_model(model, name):
    expected = ['1', 'q'] if name == 'affine' else ['1', 'q', 'sqrt(q)']
    if model['design'] != expected or model['coefficient_order'] != 'design column, then quantity':
        raise ValueError('MODEL_DESIGN_OR_ORDER')
    labels = [tuple(x) for x in model['quantity_labels']]
    required = {(m, a, g) for m in (*bc.METHODS, 'paired_delta') for a in AXES for g in bc.METRICS}
    if len(labels) != 36 or set(labels) != required:
        raise ValueError('MODEL_QUANTITY_COVERAGE')
    coefficients = np.asarray(model['coefficients'], float)
    if coefficients.shape != (len(expected), 36) or not np.isfinite(coefficients).all():
        raise ValueError('MODEL_COEFFICIENT_SCHEMA')
    check_covariance(model['coefficient_covariance'], len(expected)*36)


def check_covariance(matrix, size):
    a = np.asarray(matrix, float)
    if a.shape != (size, size) or not np.isfinite(a).all() or not np.allclose(a, a.T, rtol=1e-12, atol=1e-9):
        raise ValueError('SAVED_COVARIANCE_SCHEMA')
    return a


def load_inputs(root):
    """Read/validate saved scalars. No model prediction or H4 total here."""
    import json
    bindings = {}
    def get(commit, path):
        return json.loads(bc.blob(root, commit, path, bindings))
    handoff, b4_metadata = bc.read_b4(root, bindings)
    catalog = bc.price_catalog(root, bindings)
    b6dir = 'artifacts/track_d_b6b_bc/2026-11-08/run01/'
    old_common = get(B6B, b6dir+'common_N_conditions.json')
    for leaf in ('SOURCE.json', 'STOP.json', 'coverage13.json'):
        get(B6B, b6dir+leaf)
    if not old_common['ideal']['common_N'] or not bc.common_N(premises())['common_N']:
        raise ValueError('B6B_IDEAL_CONDITIONS')
    c74 = {leaf: get(bc.C74, C74_DIR+leaf) for leaf in
        ('SOURCE.json', 'result.json', 'STOP.json', 'input_bindings.json', 'integrated_statistics.json',
         'model_affine.json', 'model_sqrt_lane.json', 'model_frozen_C6a.json', 'frozen_C6a_future.json')}
    if c74['SOURCE.json']['execution_source'] != 'dd31e1e2878479eabe2e7cc27c21b26c8c7dc794':
        raise ValueError('C74_EXECUTION_SOURCE')
    if not c74['STOP.json']['mandatory_stop'] or c74['result.json']['status'] != 'C7_4_ANALYSIS_COMPLETE_MANDATORY_STOP':
        raise ValueError('C74_COMPLETION')
    for q, n, commit, directory in bc.PRICE_SETS:
        path = 'artifacts/'+directory+'/prices.json'
        used = c74['input_bindings.json'][path]
        if used['commit'] != commit or used['sha256'] != bindings[commit+':'+path]['sha256']:
            raise ValueError('C74_DIRECT_PRICE_BINDING')
    direct = {}
    for q, n, commit, directory in bc.PRICE_SETS:
        stats = get(commit, 'artifacts/'+directory+'/paired_statistics.json')
        entry = stats['by_q'][str(q)] if 'by_q' in stats else stats
        values = {}
        for row in entry['rows']:
            axis, g = row['axis'], row['metric']
            for method, field in zip(bc.METHODS, ('symmetric', 'reference')):
                if row[field]['n'] != n:
                    raise ValueError('DIRECT_SAMPLE_COUNT')
                key = (method, axis, g)
                if key in values:
                    raise ValueError('DUPLICATE_DIRECT_MEAN')
                values[key] = finite(row[field]['mean'], nonnegative=True)
            values[('paired_delta', axis, g)] = finite(row['paired_delta']['mean'])
            close(values[('reference', axis, g)]-values[('symmetric_directional', axis, g)], values[('paired_delta', axis, g)])
        direct[q] = dict(n=n, values=values, source=dict(commit=commit, path='artifacts/'+directory+'/paired_statistics.json'))
    q1 = catalog[0]
    q1dir = q1['price_path'].rsplit('/', 1)[0]+'/'
    summary = get(q1['commit'], q1dir+'finite_cohort_summary.json')
    source = get(q1['commit'], q1dir+'SOURCE.json')
    inputs = source['original_saved_inputs']
    if (inputs['artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz'] != bc.PACKET or
        inputs['artifacts/research_c_near_practical/2026-11-03/run01/PR/endpoint.npz'] != bc.CDF or
        source['extra_saved_law']['sha256'] != bc.LAW):
        raise ValueError('Q1_FIXED_IDENTITY')
    values = {}
    for axis in AXES:
        for g in bc.METRICS:
            row = summary['axes'][axis][g]
            if row['n_paths'] != 32:
                raise ValueError('Q1_COHORT_SIZE')
            for m, f in [('symmetric_directional', 'symmetric_mean'), ('reference', 'reference_mean'), ('paired_delta', 'delta_mean')]:
                values[(m, axis, g)] = finite(row[f], nonnegative=m != 'paired_delta')
            close(values[('reference', axis, g)]-values[('symmetric_directional', axis, g)], values[('paired_delta', axis, g)])
    direct[1] = dict(n=32, values=values, source=dict(commit=q1['commit'], path=q1dir+'finite_cohort_summary.json'))
    integrated = c74['integrated_statistics.json']
    if set(r['q'] for r in integrated['by_q']) != {2, 4, 8, 16, 32}:
        raise ValueError('SAVED_DIRECT_COVARIANCE_COVERAGE')
    for row in integrated['by_q']:
        q = row['q']
        if row['n'] != direct[q]['n']:
            raise ValueError('SAVED_COVARIANCE_N')
        check_covariance(row['sample_covariance'], 36)
        for v in row['quantities']:
            if v['mean'] != direct[q]['values'][tuple(v['quantity'])]:
                raise ValueError('ORIGINAL_AND_C74_SAVED_MEAN_MISMATCH')
    for name in ('affine', 'sqrt_lane'):
        check_model(c74['model_'+name+'.json'], name)
    frozen_path = 'artifacts/track_d_c6a_paired_price/2026-11-06/run01/exploratory_affine.json'
    frozen = get(bc.PRICE_SETS[0][2], frozen_path)
    if c74['input_bindings.json'][frozen_path]['sha256'] != bindings[bc.PRICE_SETS[0][2]+':'+frozen_path]['sha256']:
        raise ValueError('FROZEN_COEFFICIENT_BINDING')
    fcoeff = {}
    for method in bc.METHODS:
        for row in frozen['models'][method]:
            axis = {'cosine': 'X', 'sine': 'Y'}[row['axis']]
            fcoeff[(method, axis, row['metric'])] = (finite(row['intercept']), finite(row['slope']))
            for saved in row['calibration_residuals']:
                if saved['observed_mean'] != direct[saved['q_m']]['values'][(method, axis, row['metric'])]:
                    raise ValueError('FROZEN_SHARED_CALIBRATION_BINDING')
    if set(fcoeff) != {(m, a, g) for m in bc.METHODS for a in AXES for g in bc.METRICS}:
        raise ValueError('FROZEN_COEFFICIENT_COVERAGE')
    check_covariance(c74['frozen_C6a_future.json']['saved_parameter_covariance']['matrix'], 48)
    return dict(handoff=handoff, b4_metadata=b4_metadata, catalog=catalog, direct=direct,
        c74=c74, frozen_coefficients=fcoeff, bindings=bindings, premises=premises())


def model_value(inputs, scenario, method, axis, g, q):
    if scenario not in SCENARIOS or q not in (64, 128, 256, 512, 1024, 2048, 4096):
        raise ValueError('FIXED_SCENARIO_OR_LONG_Q')
    if scenario == 'F':
        a, b = inputs['frozen_coefficients'][(method, axis, g)]
        return finite(a+b*q, nonnegative=True)
    model = inputs['c74']['model_'+SCENARIOS[scenario]+'.json']
    col = [tuple(x) for x in model['quantity_labels']].index((method, axis, g))
    x = np.array([1., q] if scenario == 'A' else [1., q, math.sqrt(q)])
    return finite(float(x @ np.asarray(model['coefficients'])[:, col]), nonnegative=True)


def price_records(inputs, scenario):
    if scenario not in SCENARIOS:
        raise ValueError('FIXED_SCENARIO')
    records = []
    for method in bc.METHODS:
        for shot in inputs['handoff']['rows']:
            m, q = shot['m'], shot['q']
            for axis in AXES:
                direct = q <= 32
                source = inputs['direct'][q]['source'] if direct else dict(commit=bc.C74,
                    path=C74_DIR+('frozen_C6a_future.json' if scenario == 'F' else 'model_'+SCENARIOS[scenario]+'.json'))
                records.append(dict(m=m, q=q, axis=axis, method=method, scenario=scenario,
                    mean={g: inputs['direct'][q]['values'][(method, axis, g)] if direct else
                        model_value(inputs, scenario, method, axis, g, q) for g in bc.METRICS},
                    binding=bc.binding(), scope=bc.SCOPE, compiler=bc.P0, source=source,
                    estimate_unit='whole_path_one_interrogation', expectation_basis='explicit_conditional_model',
                    model_assumptions=['saved finite-cohort mean adopted as exploratory price proxy' if direct else
                        'unchanged saved coefficients extrapolate beyond observed q2..32; bias and law unknown'],
                    price_classification='DIRECT_FINITE_SAMPLE_PRICE_INPUT' if direct else 'EXPLORATORY_LONG_Q_PROXY',
                    cut=dict(kind='complete_event_within_step', after=16) if q == 1 else dict(kind='complete_step', after=q//2),
                    n_whole_paths=inputs['direct'][q]['n'] if direct else None,
                    peak_qubits=17 if method == 'reference' else 9,
                    price_uncertainty_qualified=False, attenuation_price_multiplier=False))
    validate_records(inputs, records, scenario)
    return records


def validate_records(inputs, records, scenario):
    expected = {(m, axis, method) for m in range(13) for axis in AXES for method in bc.METHODS}
    keys = [(r['m'], r['axis'], r['method']) for r in records]
    if len(keys) != 52 or set(keys) != expected:
        raise ValueError('EXACT_52_PRICE_RECORDS')
    for r in records:
        q = r['q']
        if q != 2**r['m'] or r['scenario'] != scenario:
            raise ValueError('ROUND_SCENARIO')
        cut = dict(kind='complete_event_within_step', after=16) if q == 1 else dict(kind='complete_step', after=q//2)
        if r['cut'] != cut or r.get('attenuation_price_multiplier') is not False:
            raise ValueError('CUT_OR_ATTENUATION')
        if r['source'] != (inputs['direct'][q]['source'] if q <= 32 else dict(commit=bc.C74,
            path=C74_DIR+('frozen_C6a_future.json' if scenario == 'F' else 'model_'+SCENARIOS[scenario]+'.json'))):
            raise ValueError('UNADOPTED_PRICE_SOURCE')
        if set(r['mean']) != set(bc.METRICS):
            raise ValueError('SIX_ADDITIVE_METRICS_NO_WIDTH')
        for v in r['mean'].values():
            finite(v, nonnegative=True)
        if r['binding'] != bc.binding() or r['compiler'] != bc.P0 or r['scope'] != bc.SCOPE:
            raise ValueError('IDENTITY_P0_SCOPE')


def covariance_component(matrix, transform, label, assumptions):
    """Push forward one available joint block; never fill missing cross-blocks."""
    c = check_covariance(matrix, len(matrix))
    out = transform @ c @ transform.T
    out = (out+out.T)/2
    diag = np.diag(out)
    rounding = 64*np.finfo(float).eps*np.diag(np.abs(transform) @ np.abs(c) @ np.abs(transform.T))
    if np.any(diag < -np.maximum(rounding, 1e-14)):
        raise ValueError('NEGATIVE_PROPAGATED_VARIANCE')
    return dict(component=label, output_labels=OUTPUT_LABELS, covariance=out.tolist(),
        nominal_component_SD=np.sqrt(np.maximum(0., diag)).tolist(), assumptions=assumptions,
        negative_variance_roundoff=[dict(index=int(i), value=float(diag[i]), allowance=float(rounding[i]))
            for i in np.flatnonzero(diag < 0)],
        full_total_uncertainty=False, confidence_interval=None)


def uncertainty(inputs, scenario):
    shots = inputs['handoff']['rows']
    components = []
    integrated = inputs['c74']['integrated_statistics.json']
    labels = [tuple(x) for x in integrated['quantity_labels']]
    for row in integrated['by_q']:
        q = row['q'];shot = next(s for s in shots if s['q'] == q)
        transform = np.zeros((18, 36))
        for i, (method, g) in enumerate(OUTPUT_LABELS):
            for axis in AXES:
                transform[i, labels.index((method, axis, g))] = shot['N_'+axis]
        components.append(covariance_component(np.asarray(row['sample_covariance'])/row['n'], transform,
            'direct_q'+str(q), ['saved C7-4 within-q sample covariance divided by n',
                'nominal mean-price variation under saved pseudo-IID interpretation; no joint-IID proof']))
    if scenario in ('A', 'S'):
        model = inputs['c74']['model_'+SCENARIOS[scenario]+'.json']
        p = model['parameters'];labels = [tuple(x) for x in model['quantity_labels']]
        transform = np.zeros((18, p*36))
        for shot in shots[6:]:
            x = [1., shot['q']] if p == 2 else [1., shot['q'], math.sqrt(shot['q'])]
            for i, (method, g) in enumerate(OUTPUT_LABELS):
                for axis in AXES:
                    col = labels.index((method, axis, g))
                    for j, v in enumerate(x):
                        transform[i, j*36+col] += shot['N_'+axis]*v
        components.append(covariance_component(model['coefficient_covariance'], transform, 'long_q_shared_model',
            [model['bootstrap_scope'], 'unchanged saved coefficient covariance; all seven future rounds share coefficients']))
    else:
        saved = inputs['c74']['frozen_C6a_future.json']['saved_parameter_covariance']
        labels = [tuple(x) for x in saved['labels']]
        def ftransform(qs):
            transform = np.zeros((18, 48))
            for shot in shots:
                if shot['q'] not in qs:
                    continue
                for i, (method, g) in enumerate(OUTPUT_LABELS):
                    methods = [(method, 1)] if method != 'paired_delta' else [('reference', 1), ('symmetric_directional', -1)]
                    for a in AXES:
                        for m, sign in methods:
                            for param, v in [('slope', shot['q']), ('intercept', 1.)]:
                                transform[i, labels.index((m, a, g, param))] += sign*shot['N_'+a]*v
            return transform
        long_q = [s['q'] for s in shots[6:]]
        for label, qs in [('long_q_shared_model', long_q), ('linked_q2_q4_and_long_q', [2, 4]+long_q)]:
            components.append(covariance_component(saved['matrix'], ftransform(qs), label,
                [saved['formula'], 'saved frozen beta is an invertible linear transform of the same q2/q4 means',
                 'linked component replaces overlapping direct_q2/direct_q4/long components; never add them twice']))
    return dict(status='UNCERTAINTY_PARTIALLY_QUALIFIED', components=components,
        total_covariance=None, total_SD=None, total_confidence_interval=None,
        missing=['q1 full joint mean-price covariance', 'cross-q direct cohort covariance',
                 'direct/model-fit cross-covariance for A/S; F linkage only covers q2/q4 and long-q',
                 'cross-scenario coefficient/direct joint covariance', 'population/model bias and finite-bit law',
                 'independent q>=64 extrapolation uncertainty'],
        components_are_not_independent_blocks=True, missing_covariance_is_zero=False,
        model_scenario_spread_is_CI=False, actual_success_guaranteed=False)


def evaluate(inputs, on_scenario=None):
    """One campaign, all three pre-adopted scenarios, using saved scalars only."""
    results = {}
    for scenario in SCENARIOS:
        records = price_records(inputs, scenario)
        totals = bc.aggregate_registered(inputs['handoff'], records, inputs['premises'])
        contributions = []
        for r in records:
            n = inputs['handoff']['rows'][r['m']]['N_'+r['axis']]
            for g in bc.METRICS:
                v = r['mean'][g]
                contributions.append(dict(scenario=scenario, m=r['m'], q=r['q'], method=r['method'],
                    axis=r['axis'], metric=g, price=v, N=n, shot_times_price=n*v,
                    price_classification=r['price_classification']))
        grouped = {}
        for group, select in [('q1', lambda q:q == 1), ('q2_32', lambda q:2 <= q <= 32), ('q64_4096', lambda q:q >= 64)]:
            grouped[group] = {}
            for g in bc.METRICS:
                values = {method: math.fsum(r['shot_times_price'] for r in contributions if
                    r['method'] == method and r['metric'] == g and select(r['q'])) for method in bc.METHODS}
                values['paired_delta'] = values['reference']-values['symmetric_directional']
                grouped[group][g] = values
        summary = {}
        paired_rounds = []
        for g in bc.METRICS:
            sym = totals['methods']['symmetric_directional']['totals'][g]
            ref = totals['methods']['reference']['totals'][g]
            delta = totals['paired_delta'][g]
            paired = math.fsum(inputs['handoff']['rows'][r['m']]['N_'+r['axis']] *
                (r['mean'][g]-next(s for s in records if s['m'] == r['m'] and s['axis'] == r['axis'] and
                    s['method'] == 'symmetric_directional')['mean'][g]) for r in records if r['method'] == 'reference')
            close(delta, paired)
            for method, value in [('symmetric_directional', sym), ('reference', ref), ('paired_delta', delta)]:
                close(value, math.fsum(grouped[k][g][method] for k in grouped))
            summary[g] = dict(symmetric=sym, reference=ref, paired_delta=delta,
                reduction_fraction=(sym-ref)/sym if sym > 0 else None,
                reference_direction='decrease' if delta < 0 else 'increase' if delta > 0 else 'equal',
                long_q_fraction_symmetric=grouped['q64_4096'][g]['symmetric_directional']/sym if sym > 0 else None,
                long_q_fraction_reference=grouped['q64_4096'][g]['reference']/ref if ref > 0 else None,
                long_q_fraction_signed_delta=grouped['q64_4096'][g]['paired_delta']/delta if delta else None)
        for shot in inputs['handoff']['rows']:
            for axis in AXES:
                pair = {r['method']:r for r in records if r['m'] == shot['m'] and r['axis'] == axis}
                for g in bc.METRICS:
                    delta = pair['reference']['mean'][g]-pair['symmetric_directional']['mean'][g]
                    paired_rounds.append(dict(scenario=scenario, m=shot['m'], q=shot['q'], axis=axis, metric=g,
                        N=shot['N_'+axis], paired_price_delta=delta, paired_shot_times_price=shot['N_'+axis]*delta))
        results[scenario] = dict(records=records, totals=summary, contributions=contributions,
            paired_rounds=paired_rounds, grouped=grouped, uncertainty=uncertainty(inputs, scenario))
        if on_scenario is not None:
            on_scenario(scenario, copy.deepcopy(results[scenario]))
    sensitivity = {g: {s: results[s]['totals'][g] for s in SCENARIOS} for g in bc.METRICS}
    return dict(status='CONDITIONAL_IDEAL_RPE_TOTAL_ESTIMATE', primary_scenario='A',
        scenarios=results, sensitivity=sensitivity,
        stored_shots=sum(r['N_X']+r['N_Y'] for r in inputs['handoff']['rows']), new_shot_solver_calls=0,
        peak_qubits=dict(symmetric_directional=9, reference=17, additive=False),
        uncertainty_status='UNCERTAINTY_PARTIALLY_QUALIFIED', actual_status='ACTUAL_RPE_UNQUALIFIED',
        actual_success_guaranteed=False, confirmatory_resource_advantage=False,
        model_sensitivity_is_CI=False, H4_sampling=0, H4_build=0, H4_compile=0, new_model_fits=0,
        mandatory_stop=True, next_stage_authorized=False)

"""S1 only: corrections to frozen V1 scalars, without circuits or sampling."""
import ast
from collections import defaultdict
import subprocess

import numpy as np

CARRIER = 'eec5bdf9cc391fe49b55b5673f3e385deeb20d42'
FIT_PATH = 'src/trottertracks/track_d_c7_4_models.py'
AXES = ('cosine', 'sine')
V1_MODEL = 'DF10_SOURCE_NATIVE_BASIS_CANCEL_ZFOLD_V1'


def fixed_fit(root):
    """Reuse the exact generic fit body, avoiding unrelated D launcher imports."""
    raw = subprocess.check_output(['git', 'show', CARRIER+':'+FIT_PATH], cwd=root)
    node = next(n for n in ast.parse(raw).body
                if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    scope = {'np': np}
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, CARRIER+':'+FIT_PATH, 'exec'), scope)
    return scope['fit']


def pair_rows(rows):
    """Pair by scientific identity, never by input position or price equality."""
    groups = defaultdict(dict)
    keys = ('method', 'cohort', 'candidate_id', 'q', 'T', 'path_index')
    for row in rows:
        key = tuple(row[k] for k in keys)
        axis = row['axis']
        if axis not in AXES or axis in groups[key]:
            raise ValueError('DUPLICATE_OR_UNKNOWN_AXIS')
        groups[key][axis] = row
    pairs = []
    for key, axes in sorted(groups.items()):
        if set(axes) != set(AXES):
            raise ValueError('MISSING_PAIRED_AXIS')
        pair = dict(zip(keys, key))
        if pair['method'] not in ('PF', 'PR') or pair['q'] <= 0:
            raise ValueError('METHOD_OR_Q_BINDING')
        features = [axes[a]['prediction']['features'] for a in AXES]
        if any(axes[a]['prediction']['model'] != V1_MODEL for a in AXES):
            raise ValueError('V1_MODEL_BINDING')
        if pair['method'] == 'PR':
            for field in ('tail_basis_groups', 'event_order_2'):
                values = [f.get(field, 0) for f in features]
                if values[0] != values[1] or values[0] < 0:
                    raise ValueError('PAIRED_FEATURE_BINDING')
            if any('tail_basis_groups' not in f for f in features):
                raise ValueError('MISSING_BASIS_FEATURE')
            pair['basis_groups'] = features[0]['tail_basis_groups']
            pair['order_2_count'] = features[0].get('event_order_2', 0)
        else:
            pair['basis_groups'] = pair['order_2_count'] = None
        pair['actual'] = [axes[a]['compiled_rz'] for a in AXES]
        pair['baseline'] = [axes[a]['prediction']['structural_rz'] for a in AXES]
        if not np.isfinite([pair['actual'], pair['baseline']]).all():
            raise ValueError('NONFINITE_SAVED_PRICE')
        if min(pair['actual']) < 0:
            raise ValueError('NEGATIVE_SAVED_PRICE')
        pairs.append(pair)
    return pairs


def design(pairs, model):
    q = np.asarray([p['q'] for p in pairs], float)
    if model == 'Mq':
        return q[:, None]
    if model == 'MqB':
        return np.column_stack((q, [p['basis_groups'] for p in pairs]))
    raise ValueError('UNKNOWN_CORRECTION_MODEL')


def calibrate(pairs, model, fit):
    x = design(pairs, model)
    y = np.asarray([p['actual'] for p in pairs], float)
    baseline = np.asarray([p['baseline'] for p in pairs], float)
    try:
        beta = fit(x, y-baseline, np.asarray([p['q'] for p in pairs]))
    except ValueError as error:
        return dict(model=model, status='EVALUATION_UNAVAILABLE', reason=str(error),
                    rank=int(np.linalg.matrix_rank(x)), coefficients=None)
    if not np.isfinite(beta).all():
        return dict(model=model, status='EVALUATION_UNAVAILABLE',
                    reason='NONFINITE_COEFFICIENT', rank=x.shape[1], coefficients=None)
    return dict(model=model, status='FITTED', rank=x.shape[1],
                features=['q'] if model == 'Mq' else ['q', 'tail_basis_groups'],
                response_axes=list(AXES), training_paths=len(pairs),
                coefficients=beta.tolist(), coefficient_uncertainty=None)


def evaluate(pairs, role, model, calibration=None):
    if calibration is not None and calibration['status'] != 'FITTED':
        return []
    baseline = np.asarray([p['baseline'] for p in pairs], float)
    correction = (np.zeros_like(baseline) if model == 'M0' else
                  design(pairs, model) @ np.asarray(calibration['coefficients']))
    prediction = baseline+correction
    rows = []
    for i, pair in enumerate(pairs):
        meta = {k: v for k, v in pair.items() if k not in ('actual', 'baseline')}
        for j, axis in enumerate(AXES):
            actual, pred = pair['actual'][j], float(prediction[i, j])
            finite = bool(np.isfinite(pred))
            error = actual-pred if finite else None
            rows.append(dict(**meta, role=role, model=model, axis=axis,
                compiled_rz=actual, baseline_rz=pair['baseline'][j],
                correction_rz=float(correction[i, j]) if finite else None,
                prediction_rz=pred if finite else None, error_rz=error,
                absolute_error_rz=abs(error) if finite else None,
                relative_error=error/actual if finite and actual != 0 else None,
                diagnostic=('NONFINITE_PREDICTION' if not finite else
                            'NEGATIVE_PRICE' if pred < 0 else 'FINITE')))
    return rows


def summarize(rows, stratum=None):
    keys = ('role', 'method', 'cohort', 'model', 'q', 'axis')
    if stratum is not None:
        keys += (stratum,)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        entry = dict(zip(keys, key))
        entry.update(n_paths=len(group),
            negative_predictions=sum(r['diagnostic'] == 'NEGATIVE_PRICE' for r in group),
            nonfinite_predictions=sum(r['diagnostic'] == 'NONFINITE_PREDICTION' for r in group))
        # Do not silently summarize only a finite subset as a successful model.
        if entry['nonfinite_predictions']:
            entry['metrics'] = None
        else:
            e = np.asarray([r['error_rz'] for r in group])
            rel = np.asarray([r['relative_error'] for r in group
                              if r['relative_error'] is not None])
            sd = float(np.std(e, ddof=1)) if len(group) > 1 and key[1] == 'PR' else None
            entry['metrics'] = dict(
                observed_mean=float(np.mean([r['compiled_rz'] for r in group])),
                prediction_mean=float(np.mean([r['prediction_rz'] for r in group])),
                bias=float(e.mean()), mean_absolute_error=float(abs(e).mean()),
                RMSE=float(np.sqrt(np.mean(e**2))), min_error=float(e.min()),
                max_error=float(e.max()), max_absolute_error=float(abs(e).max()),
                residual_SD=sd, exact_float_zero_errors=int(np.count_nonzero(e == 0)),
                relative_error_n=int(len(rel)),
                relative_bias=float(rel.mean()) if len(rel) else None,
                relative_RMSE=float(np.sqrt(np.mean(rel**2))) if len(rel) else None,
                min_relative_error=float(rel.min()) if len(rel) else None,
                max_relative_error=float(rel.max()) if len(rel) else None,
                max_absolute_relative_error=float(abs(rel).max()) if len(rel) else None,
                descriptive_mean_error_SE_under_IID_given_fit=(
                    sd/len(group)**.5 if sd is not None and key[0] == 'KNOWN_V1_USED' else None),
                total_prediction_CI=None)
        summaries.append(entry)
    return summaries


def campaign(training, known, fit):
    """Exactly three development-only fits; no adaptive model or feature search."""
    calibrations, rows = [], []
    for method in ('PF', 'PR'):
        train = [p for p in training if p['method'] == method]
        check = [p for p in known if p['method'] == method]
        for role, pairs in [('TRAINING', train), ('KNOWN_V1_USED', check)]:
            rows.extend(evaluate(pairs, role, 'M0'))
        for model in (('Mq',) if method == 'PF' else ('Mq', 'MqB')):
            calibration = dict(method=method, **calibrate(train, model, fit))
            calibrations.append(calibration)
            for role, pairs in [('TRAINING', train), ('KNOWN_V1_USED', check)]:
                rows.extend(evaluate(pairs, role, model, calibration))
    pr = [row for row in rows if row['method'] == 'PR']
    return dict(calibrations=calibrations, summaries=summarize(rows),
                basis_strata=summarize(pr, 'basis_groups'),
                order_2_strata=summarize(pr, 'order_2_count')), rows

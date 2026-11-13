"""Exact calibration / independent-holdout checks for SUPPLIED compiled prices.

No compile, sampling, search over model families or automatic refit. A record
may describe a non-configuration cell (e.g. one Full-H group or a Full PF m4/m5
template point), so it carries its own SOURCE-bound feature values and route.
The feature design and tolerance are fixed by the caller before holdouts are
read; a failed holdout keeps the model unvalidated (no refit, no new features).
Validation status never sets qualified_for_candidate: adoption stays external.
"""
from copy import deepcopy
from fractions import Fraction

from . import rpe_mean_price_model as P

MODULE = 'src/trottertracks/price_model_validation.py'
RECORD = 'PRICE_CALIBRATION_RECORD_V1'
ROLES = ('CALIBRATION', 'HOLDOUT')
KINDS = ('DIRECT_COMPILED_DETERMINISTIC', 'DIRECT_COMPILED_FINITE_SAMPLE')


def _exact(value, name):
    if isinstance(value, bool):
        raise ValueError(name+': exact number required')
    if isinstance(value, dict):
        value = Fraction(value['numerator'], value['denominator'])
    result = Fraction(value)
    if result < 0:
        raise ValueError(name+': nonnegative value required')
    return result


def _enc(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def record(*, cell_id, role, route, axis, features, rz_count, kind, source, implementation, profile,
           quantity=P.QUANTITY, n_paths=None, law=None):
    """Normalize one supplied record; reject missing SOURCE/route/profile."""
    if role not in ROLES or axis not in ('X', 'Y') or kind not in KINDS:
        raise ValueError('Explicit role, X/Y axis and direct compiled kind required')
    if not cell_id or not source or not implementation or not implementation.get('lowering_id') or not profile:
        raise ValueError('Cell id, price SOURCE, implementation route and compiler profile required')
    if quantity != P.QUANTITY:
        raise ValueError('Full measured wrapper mean compiled price quantity required')
    if kind == 'DIRECT_COMPILED_FINITE_SAMPLE' and (type(n_paths) is not int or n_paths < 1 or not law):
        raise ValueError('Finite-sample price needs n_paths and its sampling law')
    if not features or any(not isinstance(k, str) for k in features):
        raise ValueError('Named feature values required')
    return dict(record=RECORD, cell_id=cell_id, role=role, route=route, axis=axis,
        features={k: _enc(_exact(v, k)) for k, v in sorted(features.items())},
        rz_count=_enc(_exact(rz_count, 'rz_count')), kind=kind, n_paths=n_paths, law=deepcopy(law),
        source=deepcopy(source), implementation=deepcopy(implementation), profile=deepcopy(profile),
        quantity=quantity, finite_sample_mean_is_law_expectation=False)


def _domain(records):
    keys = ('route', 'axis', 'implementation', 'profile', 'quantity', 'kind', 'law')
    first = {k: records[0][k] for k in keys}
    if any({k: r[k] for k in keys} != first for r in records):
        raise ValueError('One route/axis/implementation/profile/kind/law per model')
    return first


def _solve(a, b):
    """Exact Gauss-Jordan on a square Fraction system; singular -> ValueError."""
    n = len(a); m = [row[:]+[rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
        if pivot is None:
            raise ValueError('Unidentified model design (singular); add a distinct calibration point')
        m[col], m[pivot] = m[pivot], m[col]
        m[col] = [v/m[col][col] for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                m[r] = [x-m[r][col]*y for x, y in zip(m[r], m[col])]
    return [row[-1] for row in m]


def fit(records, *, features, design_source):
    """Exact solve (square) or exact normal-equation least squares (overdetermined)."""
    if not design_source or not features or len(set(features)) != len(features):
        raise ValueError('Pre-declared unique feature design with SOURCE required')
    calibration = [r for r in records if r['role'] == 'CALIBRATION']
    if len(calibration) != len(records) or len({r['cell_id'] for r in records}) != len(records):
        raise ValueError('Distinct CALIBRATION records only')
    if len(calibration) < len(features):
        raise ValueError('Fewer calibration points than coefficients')
    domain = _domain(calibration)
    x = [[Fraction(r['features'][f]['numerator'], r['features'][f]['denominator']) if f in r['features']
          else None for f in features] for r in calibration]
    if any(v is None for row in x for v in row):
        raise ValueError('Every calibration record must supply every design feature')
    y = [Fraction(r['rz_count']['numerator'], r['rz_count']['denominator']) for r in calibration]
    if len(calibration) == len(features):
        coef = _solve(x, y); method = 'EXACT_SQUARE_SOLVE'
    else:
        xt = list(zip(*x))
        coef = _solve([[sum(a*b for a, b in zip(ci, cj)) for cj in xt] for ci in xt],
                      [sum(a*b for a, b in zip(ci, y)) for ci in xt]); method = 'EXACT_NORMAL_EQUATION_LEAST_SQUARES'
    residuals = [yi-sum(c*v for c, v in zip(coef, row)) for row, yi in zip(x, y)]
    return dict(model='PRICE_LINEAR_FEATURE_MODEL_V1', status='CALIBRATED_UNVALIDATED', method=method,
        features=list(features), coefficients={f: _enc(c) for f, c in zip(features, coef)}, domain=domain,
        calibration=[dict(cell_id=r['cell_id'], source=r['source']) for r in calibration],
        calibration_residuals=[_enc(abs(v)) for v in residuals], design_source=deepcopy(design_source),
        qualified_for_candidate=False, extrapolation_scope=None, source=dict(module=MODULE))


def predict(model, features):
    total = Fraction(0)
    for f in model['features']:
        if f not in features:
            raise ValueError('Missing feature for prediction: '+f)
        c = model['coefficients'][f]
        total += Fraction(c['numerator'], c['denominator'])*_exact(features[f], f)
    return total


def validate(model, holdouts, *, tolerance):
    """Independent holdout check against a PRE-DECLARED tolerance; no refit."""
    if model.get('status') != 'CALIBRATED_UNVALIDATED':
        raise ValueError('Validate a freshly calibrated model once')
    if (tolerance.get('kind') not in ('RELATIVE', 'ABSOLUTE') or not tolerance.get('source')
            or _exact(tolerance.get('value'), 'tolerance') <= 0):
        raise ValueError('Pre-declared positive RELATIVE/ABSOLUTE tolerance with SOURCE required')
    if not holdouts or any(r['role'] != 'HOLDOUT' for r in holdouts):
        raise ValueError('HOLDOUT records only')
    used = {c['cell_id'] for c in model['calibration']}
    if any(r['cell_id'] in used for r in holdouts):
        raise ValueError('Holdout overlaps a calibration cell')
    if _domain(holdouts) != model['domain']:
        raise ValueError('Holdout route/axis/implementation/profile/kind/law differs from calibration')
    limit = _exact(tolerance['value'], 'tolerance'); rows = []
    for r in holdouts:
        observed = Fraction(r['rz_count']['numerator'], r['rz_count']['denominator'])
        predicted = predict(model, {k: v for k, v in r['features'].items()})
        error = abs(predicted-observed); scale = observed if tolerance['kind'] == 'RELATIVE' else Fraction(1)
        if scale == 0:
            raise ValueError('Relative tolerance with zero observed price')
        rows.append(dict(cell_id=r['cell_id'], observed=_enc(observed), predicted=_enc(predicted),
                         error=_enc(error), within=error <= limit*scale, source=r['source']))
    ok = all(v['within'] for v in rows); out = deepcopy(model)
    out.update(status='HOLDOUT_WITHIN_DECLARED_TOLERANCE' if ok else 'HOLDOUT_FAILED_NO_REFIT',
               holdout=rows, tolerance=deepcopy(tolerance), qualified_for_candidate=False,
               extrapolation_scope='HOLDOUT_FEATURE_RANGE_ONLY' if ok else None)
    return out


def quote_terms(model, covers):
    """Map coefficients to rpe_mean_price_model.quote_model ADDITIVE terms (no new evaluator)."""
    if set(covers) != set(model['features']):
        raise ValueError('Every coefficient needs exactly one declared coverage')
    return [dict(feature=f, coefficient=deepcopy(model['coefficients'][f]), covers=list(covers[f]))
            for f in model['features']]

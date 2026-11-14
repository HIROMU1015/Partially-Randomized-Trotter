"""Plan-time identifiability and admission for finite compiled-price calibration designs.

Design work only: no compile, sampling, price or fit on real data. C's
`price_model_validation.fit` (c628e9e8, executed unchanged from its blob) already
refuses a singular design, but only after prices exist. This module checks a
declared plan BEFORE acquisition and admits records to C's fit only when they are
exactly the plan's calibration cells:

* exact rank of the planned design matrix (Fractions), with the null direction and a
  readable reason when it is deficient (e.g. q=1 only, fixed R, collinear (q-1, s));
* one model domain (route, law, lowering/native, scalar, compiler profile, quantity,
  kind) for calibration AND holdout cells; calibration and holdout are disjoint;
* single-path wiring cohorts (e.g. the Full-H P-1 path) are never admitted unless a
  plan declares them explicitly, and a record never claims a law expectation;
* an artificial covariance model (per-event noise sigma_e^2 * s per path, optional
  boundary noise, shared group paths for paired compositions) to compare designs in
  units of sigma_e. It is a design aid, not an estimate of real price variance.
"""
from copy import deepcopy
from fractions import Fraction
import subprocess
import sys
import types
from pathlib import Path

MODULE = 'src/trottertracks/finite_price_design.py'
C_MODEL = dict(commit='c628e9e8447c11c984064b922cbce6c52dc72d48', path='src/trottertracks/price_model_validation.py')
C_MODEL_DEPENDENCY = 'src/trottertracks/rpe_mean_price_model.py'
P1_WIRING_COHORT = 'D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1'
B0_WIRING_COHORT = 'D_PAULI_LD1_M0_WIRING_V1'
DOMAIN_KEYS = ('route', 'law', 'implementation', 'profile', 'quantity', 'kind')
ROLES = ('CALIBRATION', 'HOLDOUT')

# Feature maps: name -> (ordered feature names, function(cell) -> values).
FEATURE_MAPS = {
    # C's PRICE_VALIDATION_PLAN Full-H model E[C] = a + b*s + c*(q-1).
    'FULL_H_GROUP_V1': (('one', 'event_count', 'outer_boundary_count'),
                        lambda c: (1, c['q'] * c['R'], c['q'] - 1)),
    # PR with a deterministic head per PF step: E[C] = a + h*q + b*s (s = q*R tail events).
    'PR_STEP_V1': (('one', 'pf_step_count', 'event_count'),
                   lambda c: (1, c['q'], c['q'] * c['R'])),
}


def features(cell, feature_map):
    names, f = FEATURE_MAPS[feature_map]
    if type(cell.get('q')) is not int or type(cell.get('R')) is not int or cell['q'] < 1 or cell['R'] < 1:
        raise ValueError('Integer q>=1 and R>=1 required: ' + str(cell.get('cell_id')))
    return dict(zip(names, (Fraction(v) for v in f(cell))))


def design_matrix(cells, feature_map):
    names = FEATURE_MAPS[feature_map][0]
    return [[features(c, feature_map)[n] for n in names] for c in cells]


def rank_and_null(matrix):
    """Exact rank by Gauss-Jordan on Fractions; one null vector if rank-deficient."""
    if not matrix:
        return 0, None
    m = [row[:] for row in matrix]; rows, cols = len(m), len(m[0]); pivots = []; r = 0
    for col in range(cols):
        p = next((i for i in range(r, rows) if m[i][col] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        m[r] = [v / m[r][col] for v in m[r]]
        for i in range(rows):
            if i != r and m[i][col] != 0:
                m[i] = [x - m[i][col] * y for x, y in zip(m[i], m[r])]
        pivots.append(col); r += 1
        if r == rows:
            break
    if r == cols:
        return r, None
    free = next(c for c in range(cols) if c not in pivots)
    null = [Fraction(0)] * cols; null[free] = Fraction(1)
    for i, pc in enumerate(pivots):
        null[pc] = -m[i][free]
    return r, null


def identifiability(cells, feature_map):
    """Full column rank of the calibration design, with a reason when it fails."""
    names = FEATURE_MAPS[feature_map][0]
    x = design_matrix(cells, feature_map)
    rank, null = rank_and_null(x)
    reason = None
    if rank < len(names):
        qs = {c['q'] for c in cells}; Rs = {c['R'] for c in cells}
        if len(cells) < len(names):
            reason = 'FEWER_DISTINCT_CELLS_THAN_COEFFICIENTS'
        elif qs == {1}:
            reason = 'ALL_Q_EQUAL_ONE_BOUNDARY_OR_STEP_COLUMN_NOT_SEPARABLE'
        elif len(qs) == 1:
            reason = 'SINGLE_Q_VALUE_COLLINEAR_WITH_INTERCEPT'
        elif len(Rs) == 1:
            reason = 'FIXED_R_EVENT_COUNT_IS_AFFINE_IN_Q'
        else:
            reason = 'CELLS_COLLINEAR_IN_FEATURE_PLANE'
    return dict(feature_map=feature_map, features=list(names), rank=rank, full_column_rank=rank == len(names),
                null_direction=None if null is None else [str(v) for v in null], reason=reason,
                cells=[c['cell_id'] for c in cells])


def _domain(cell):
    return {k: deepcopy(cell.get(k)) for k in DOMAIN_KEYS}


def check_plan(plan):
    """Validate a declared finite design before any acquisition; returns a summary or raises."""
    fm = plan['feature_map']
    if fm not in FEATURE_MAPS or not plan.get('design_source'):
        raise ValueError('Known feature map and design SOURCE required')
    cells = plan['cells']
    ids = [c['cell_id'] for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate cell id')
    if any(c.get('role') not in ROLES for c in cells):
        raise ValueError('Every cell needs role CALIBRATION or HOLDOUT')
    cal = [c for c in cells if c['role'] == 'CALIBRATION']; hold = [c for c in cells if c['role'] == 'HOLDOUT']
    key = lambda c: (c['q'], c['R'], str(c.get('T')), c.get('cohort'))
    if {key(c) for c in cal} & {key(c) for c in hold}:
        raise ValueError('A cell cannot be both calibration and holdout')
    shared = {g for c in hold for g in c.get('composed_of', ())} & {g for c in cal for g in _paths(c)}
    if shared:
        raise ValueError('Holdout composed from calibration group paths (not independent): ' + str(sorted(shared)))
    fresh = {t: c for c in cal if not c.get('composed_of') for t in _paths(c)}
    for c in cells:
        if c.get('composed_of'):
            toks = c['composed_of']
            if (c['n_paths'] != 1 or len(toks) != c['q'] or len(set(toks)) != len(toks)
                    or any(t not in fresh or fresh[t]['q'] != 1 or fresh[t]['R'] != c['R'] for t in toks)):
                raise ValueError('A paired composition is one path of q distinct planned q=1 group paths with the same R: '
                                 + c['cell_id'])
    first = _domain(cells[0])
    for c in cells:
        if _domain(c) != first:
            raise ValueError('One route/law/implementation(lowering,native,scalar)/profile/quantity/kind per model: '
                             + c['cell_id'])
        if c.get('cohort') in plan.get('excluded_cohorts', (P1_WIRING_COHORT, B0_WIRING_COHORT)) and not c.get('declared_reuse'):
            raise ValueError('Single-path wiring cohort is not a planned price cell: ' + c['cell_id'])
        if type(c.get('n_paths')) is not int or c['n_paths'] < 1:
            raise ValueError('Integer n_paths>=1 per cell required: ' + c['cell_id'])
        if c.get('expectation') not in (None, 'FINITE_COHORT_ONLY'):
            raise ValueError('A finite cell never declares a law expectation: ' + c['cell_id'])
    ident = identifiability(cal, fm)
    if not ident['full_column_rank']:
        raise ValueError('Unidentified calibration design: ' + str(ident['reason']) + ' null=' + str(ident['null_direction']))
    axes = plan.get('axes', ('X', 'Y'))
    new = sum(c['n_paths'] for c in cells if not c.get('declared_reuse')) * len(axes)
    reused = sum(c['n_paths'] for c in cells if c.get('declared_reuse')) * len(axes)
    return dict(status='DESIGN_IDENTIFIED_NOT_EXECUTED', identifiability=ident, axes=list(axes),
                calibration=[c['cell_id'] for c in cal], holdout=[c['cell_id'] for c in hold],
                new_compiles=new, reused_compiles=reused, domain=first, design_source=deepcopy(plan['design_source']))


def _paths(cell):
    """Group-path tokens of a cell: fresh tokens per path, or the declared composition."""
    if cell.get('composed_of'):
        return list(cell['composed_of'])
    return ['%s#p%d#g%d' % (cell['cell_id'], i, g) for i in range(cell['n_paths']) for g in range(cell['q'])]


def classify_saved_price(record):
    """Evidence class of a saved B-kind price record for planning (never upgrades its meaning)."""
    kind, expectation, n = record.get('classification'), record.get('expectation'), record.get('n_paths')
    if kind == 'DIRECT_COMPILED_FINITE_SAMPLE' and expectation == 'FINITE_COHORT_ONLY':
        if record.get('law_expectation') not in (False, None) or (record.get('diagnostic') or {}).get('law_expectation'):
            raise ValueError('A finite-sample record cannot claim a law expectation')
        if n == 1:
            return dict(evidence='SINGLE_PATH_DIRECT_FINITE_SAMPLE', law_expectation=False, calibration_record=False,
                        allowed=['wiring/implementation evidence', 'one path of a PRE-DECLARED cohort of the same cell'],
                        forbidden=['E[C]', 'model calibration point without a declared cell', 'holdout', 'G'])
        return dict(evidence='FINITE_COHORT_MEAN', law_expectation=False, calibration_record=False,
                    allowed=['finite cohort observation of its own cell'], forbidden=['E[C] without an interval contract'])
    if kind == 'DIRECT_COMPILED_DETERMINISTIC' and expectation == 'DETERMINISTIC':
        return dict(evidence='DETERMINISTIC_DIRECT', law_expectation=None, calibration_record=False, allowed=['its own cell'],
                    forbidden=[])
    raise ValueError('Unknown or non-B price kind: %s/%s' % (kind, expectation))


# ---------------------------------------------------------------- admission to C's existing fit

def c_model(root):
    """C's price_model_validation, executed unchanged from its c628e9e8 blob in this tree's package."""
    name = 'trottertracks._price_model_validation_c628e9e8'
    if name not in sys.modules:
        root = Path(root)
        if str(root / 'src') not in sys.path:
            sys.path.insert(0, str(root / 'src'))
        import trottertracks  # noqa: F401
        blob = lambda c, p: subprocess.check_output(['git', 'show', c + ':' + p], cwd=root)
        if blob(C_MODEL['commit'], C_MODEL_DEPENDENCY) != (root / C_MODEL_DEPENDENCY).read_bytes():
            raise ValueError("C model dependency differs from this tree's rpe_mean_price_model")
        module = types.ModuleType(name); module.__package__ = 'trottertracks'
        module.__file__ = C_MODEL['commit'] + ':' + C_MODEL['path']; sys.modules[name] = module
        exec(compile(blob(C_MODEL['commit'], C_MODEL['path']), module.__file__, 'exec'), module.__dict__)
    return sys.modules[name]


def fit_identified(root, records, *, plan, axis):
    """Admit C records to C's fit only if they are exactly the plan's calibration cells for one axis."""
    summary = check_plan(plan)
    M = c_model(root)
    names = FEATURE_MAPS[plan['feature_map']][0]
    cal = {c['cell_id']: c for c in plan['cells'] if c['role'] == 'CALIBRATION'}
    if any(r.get('record') != M.RECORD for r in records):
        raise ValueError('C PRICE_CALIBRATION_RECORD_V1 records required')
    if any(r['role'] != 'CALIBRATION' for r in records):
        raise ValueError('Holdout or unknown-role record offered to fit')
    if sorted(r['cell_id'] for r in records) != sorted(cal) or any(r['axis'] != axis for r in records):
        raise ValueError('Records must be exactly the planned calibration cells of one axis')
    for r in records:
        c = cal[r['cell_id']]
        want = {k: {'numerator': v.numerator, 'denominator': v.denominator} for k, v in features(c, plan['feature_map']).items()}
        if r['features'] != want:
            raise ValueError('Record features differ from the planned cell: ' + r['cell_id'])
        if r.get('finite_sample_mean_is_law_expectation') is not False or r['kind'] != c['kind']:
            raise ValueError('Record kind/expectation meaning differs from the plan: ' + r['cell_id'])
        if r['kind'] == 'DIRECT_COMPILED_FINITE_SAMPLE' and r.get('n_paths') != c['n_paths']:
            raise ValueError('Record n_paths differs from the planned cohort size: ' + r['cell_id'])
        if (r['route'], r['law'], r['implementation'], r['profile']) != (c['route'], c['law'], c['implementation'], c['profile']):
            raise ValueError('Record route/law/implementation/profile differs from the plan: ' + r['cell_id'])
    model = M.fit(records, features=list(names), design_source=plan['design_source'])
    return dict(model=model, plan=summary)


# ---------------------------------------------------------------- artificial design comparison

def coefficient_covariance(cells, feature_map, *, sigma_event=1.0, sigma_boundary=0.0):
    """Cov of C's OLS coefficients from cell means, in units of sigma_event^2 (artificial model).

    Path price noise = sum over its group paths of event noise (variance sigma_event^2 * R per
    group) + boundary noise sigma_boundary^2 * (q-1); paths of different cells are independent
    unless a composed cell names the same group-path tokens.
    """
    import numpy as np
    x = np.array([[float(v) for v in row] for row in design_matrix(cells, feature_map)])
    tok_R = {}
    weights = []
    for c in cells:
        w = {}
        if c.get('composed_of'):
            for t in c['composed_of']:
                w[t] = w.get(t, 0.0) + 1.0 / c['n_paths']
        else:
            for t in _paths(c):
                w[t] = w.get(t, 0.0) + 1.0 / c['n_paths']
                tok_R[t] = c['R']
        weights.append(w)
    for c in cells:
        for t in c.get('composed_of') or ():
            if t not in tok_R:
                raise ValueError('Composed token without a group path of known R: ' + t)
    n = len(cells); cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = sum(wi * weights[j].get(t, 0.0) * sigma_event ** 2 * tok_R[t] for t, wi in weights[i].items())
        cov[i, i] += sigma_boundary ** 2 * (cells[i]['q'] - 1) / cells[i]['n_paths']
    a = np.linalg.solve(x.T @ x, x.T)
    return a @ cov @ a.T


def prediction_se(cov, feature_values):
    import numpy as np
    v = np.array([float(x) for x in feature_values])
    return float(np.sqrt(v @ cov @ v))

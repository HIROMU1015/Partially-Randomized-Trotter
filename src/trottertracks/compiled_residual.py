"""Full-wrapper compiled residual D = C - S: SOURCE inventory, paired records, E[C] = E[S] + E[D].

C is the compiled RZ count of the controlled evolution with the measured full
Hadamard wrapper; S is the C1 structural RZ count of the SAME path and axis over
the SAME full-wrapper word (head/tail/outer/scalar/X-Y suffix/terminal). D is
only defined for such a pair. Nothing here samples, builds, compiles, recounts S,
fits, or reanalyses saved values: the inventory copies saved metadata/summaries,
and the remaining functions act on records supplied by a caller (artificial in
the tests). Unknown values stay None; no residual is filled with zero.
"""
from copy import deepcopy
from fractions import Fraction
import json
import math

from . import pf_pr_rpe_pipeline as p
from .mean_compiled_price import combine_mean

FULL = 'FULL_WRAPPER'
PATH_S = 'FULL_WRAPPER_STRUCTURAL_RZ_C1_ACCOUNTING'
PATH_C = 'FULL_WRAPPER_COMPILED_RZ'
MEAN_S = 'MEAN_STRUCTURAL_RZ_COUNT_NOT_COMPILED_C'          # B envelope for C's K1 (full-wrapper E[S])
MEAN_D = 'FULL_WRAPPER_MEAN_C_MINUS_S'
MEAN_C = 'FULL_WRAPPER_MEAN_COMPILED_PRICE'                 # the only quantity B prices
D_MODES = ('PAIRED_DIRECT_FINITE_COHORT', 'C1_AFFINE', 'LOCAL_C_MINUS_S')
ROLES = ('DEVELOPMENT', 'CALIBRATION', 'HOLDOUT')
AXIS_LABEL = {'X': 'cosine', 'Y': 'sine'}
FEATURES = ('intercept', 'q', 'R', 'qR')

C1_RESULTS = '2e6a0cdf5a647058cc7afeeb5741bfafa24c974b'     # C1 calibration/validation/rare (frozen 7eeabf0a)
C1_SCIENCE = '1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7'     # r_price_phase1_science.structural / Backend.price_requests
C1_DIR = 'docs/research/r_price_phase1_results_20261111/run01/'
S0A = ('a11efd130e9dfe7086b88386570594ebe54d46af', 'docs/research/s0a_saved_structure_results_20261113/run02/RESULTS.json')
TARGET = ('d458b5e4f596d628d923263f3b2f1678960f2ffa', 'docs/research/h4_s0_mean_price_plan_20261113/PRICE_REQUESTS.json')


def _read(store, commit, path):
    return store.read(dict(commit=commit, path=path))


# ----------------------------------------------------------------------------- D-1 inventory
def saved_paired_inventory(store):
    """Saved paired S/C metadata of C1 (and S0a parity), copied; nothing recomputed.

    store is pf_pr_rpe_pipeline.SavedArtifacts (pinned Git reads, SHA recorded).
    """
    plan = _read(store, C1_RESULTS, C1_DIR + 'PLAN.json')
    calibration = _read(store, C1_RESULTS, C1_DIR + 'CALIBRATION.json')
    validation = _read(store, C1_RESULTS, C1_DIR + 'VALIDATION.json')
    evaluations = {e['cell_id']: e for e in _read(store, C1_RESULTS, C1_DIR + 'CALIBRATION_EVALUATION.json')['evaluations']}
    evaluations.update({e['cell_id']: e for e in validation['evaluations']})
    frozen = _read(store, C1_RESULTS, C1_DIR + 'FROZEN_MODELS.json')
    rare = _read(store, C1_RESULTS, C1_DIR + 'RARE_DIAGNOSTICS.json')
    paths = dict(calibration, **validation['data'])
    cells = []
    for cell in plan['cells']:
        c = cell['request']['identity']['candidate']; rows = paths.get(cell['id'], [])
        ev = evaluations.get(cell['id']) or {}
        complete = [r for r in rows if all(type(v) is int for v in r['C'] + r['S']) and len(r['C']) == len(r['S']) == 2]
        cells.append(dict(cell_id=cell['id'], system=cell['system'], role=cell['role'], cohort=cell['cohort'],
            candidate_id=c['id'], q_PF=c['q'], R_outer=c['R_outer'], T=c['T'], delta=c['delta'], prefix=c['prefix'],
            K=c['K'], formula=c['formula'], allocation=cell['request']['identity']['structure']['allocation'],
            binding=dict(hamiltonian=cell['request']['identity']['binding']['hamiltonian'],
                         state=cell['request']['identity']['binding']['state'],
                         snapshot=cell['request']['identity']['binding']['snapshot'],
                         series=cell['request']['identity']['binding']['series']),
            law_source=cell['request'].get('law_binding', {}).get('definition', {}).get('source'),
            planned_n=cell['n'], saved_paths=len(rows), paired_complete_paths=len(complete),
            conditional_paths=sum(bool(r['conditional']) for r in rows),
            event_digests=sum(bool(r.get('event_digest')) for r in rows),
            fingerprints=sum(len(r.get('fingerprints') or []) == 2 for r in rows),
            compilers=sorted({r['compiler'] for r in rows}),
            path_sources=sorted({json.dumps(r['source'], sort_keys=True) for r in rows}),
            seeds_frozen_in_plan=bool(cell.get('seeds')),
            saved_summary=dict((k, deepcopy(ev[k])) for k in ('observed_cohort_mean', 'paired_price_covariance',
                               'paired_residual_covariance', 'nominal_mean_residual_covariance_given_frozen_fit',
                               'price_unit', 'validation') if k in ev),
            residual_mean_saved=False,
            note='paired_residual_covariance is Cov(C - C_hat); within a cell C_hat - S is constant per axis, '
                 'so it equals the within-cell Cov(D). Cell means of D are derivable from saved paths but are not '
                 'saved summaries and were not computed here.'))
    models = {s: dict(features=m['features'], calibration_cells=m['calibration_cells'], point_roles=m['point_roles'],
                      coefficients=deepcopy(m['coefficients']), rank=m['rank'],
                      cell_level_lack_of_fit_DOF=m['cell_level_lack_of_fit_DOF'],
                      unobserved_q_R_qualified=m['unobserved_q_R_qualified'], qualified_for_candidate=m['qualified_for_candidate'],
                      total_CI=m['total_CI'], R_q_systematic=m['R_q_systematic'])
              for s, m in frozen['models'].items()}
    forced = [dict(system=v['system'], pair=v['pair'], unit=v['unit'], paths=[dict(conditional=x['conditional'], C=x['C'], S=x['S'],
              event_digest=x['event_digest']) for x in v['paths']]) for v in rare['pairs']]
    s0a = _read(store, *S0A)
    return dict(status='SAVED_PAIRED_S_C_METADATA_COPIED_NOT_REANALYSED',
        sources=dict(results=dict(commit=C1_RESULTS, dir=C1_DIR), science=C1_SCIENCE, frozen_model_commit=validation['frozen_model_commit'],
                     s0a=dict(commit=S0A[0], path=S0A[1])),
        unit=dict(S=PATH_S, C=PATH_C, pairing='SAME_EVENTS_SAME_AXIS_FULL_WRAPPER (Backend.price_requests)',
                  axes=['cosine', 'sine'], xy_are_paired_responses_not_independent_samples=True),
        cells=cells, frozen_models=models, forced_rare_pairs=forced,
        s0a_parity=[dict(cell_id=v['cell_id'], index=v['index'], event_digest=v['event_digest'], parity=v['parity'],
                         unit=v['unit']) for v in s0a['paths']],
        new_residual_statistics=0, compile=0, sampling=0)


def target_rounds(store):
    """Fixed H4 S0 13-round physical metadata (C d458b5e4); saved price coverage is copied."""
    plan = _read(store, *TARGET)
    return [dict(m=r['m'], physical_candidate_id=r['physical_candidate_id'], B_candidate_id=r.get('B_candidate_id'),
                 T=r['T'], q_PF=r['q_PF'], R_outer=r['R_outer'], event_count=r['event_count'],
                 saved_direct=r['saved_direct_metadata_present'], saved_model=r['saved_model_metadata_present'],
                 price=r['price']) for r in plan['round_price_coverage']]


# ----------------------------------------------------------------------------- paired records
def paired_residuals(result, *, context):
    """One C1-style same-path result (C/S per axis) -> D records per axis.

    context: candidate_id, role, compiler (profile), S_scope, C_scope, law_binding,
    structural_source, compiled_source. Forced/conditional paths are their own stratum.
    """
    if context['S_scope'] != FULL or context['C_scope'] != FULL:
        raise ValueError('S and C must both cover the full wrapper; internal/local S or C is not paired D')
    if context['role'] not in ROLES:
        raise ValueError('Explicit DEVELOPMENT/CALIBRATION/HOLDOUT role required')
    if result['candidate_id'] != context['candidate_id']:
        raise ValueError('Path candidate differs from the requested candidate')
    if result['compiler'] != context['compiler']:
        raise ValueError('Compiled C compiler/profile differs from the requested profile')
    if not result.get('event_digest') or len(result['C']) != 2 or len(result['S']) != 2:
        raise ValueError('Same-path pairing needs the event digest and both axes')
    if result.get('fingerprints') is not None and len(result['fingerprints']) != 2:
        raise ValueError('One compiled wrapper fingerprint per axis required')
    key = dict(candidate_id=result['candidate_id'], cohort=result['cohort'], index=result['index'], event_digest=result['event_digest'])
    out = []
    for i, axis in enumerate(p.AXES):
        c, s = result['C'][i], result['S'][i]
        ok = all(type(v) is int and v >= 0 for v in (c, s))
        out.append(dict(path=deepcopy(key), axis=axis, C=c, S=s, D=(c - s) if ok else None,
            stratum='FORCED_CONDITIONAL' if result['conditional'] else 'NORMAL', role=context['role'],
            compiler=result['compiler'], fingerprint=(result.get('fingerprints') or [None, None])[i],
            law_binding=deepcopy(context['law_binding']), source=dict(structural=context['structural_source'],
            compiled=context['compiled_source'], path=deepcopy(result.get('source')))))
    return out


def cohort_summary(records):
    """Finite-cohort mean of D per axis over ONE candidate/cohort/role/normal stratum.

    X and Y are paired responses of the same paths: n counts paths, not axes.
    Exact rationals; descriptive variance only. Missing D keeps the mean None.
    """
    if not records:
        raise ValueError('Empty cohort')
    first = records[0]['path']
    if any(r['path']['candidate_id'] != first['candidate_id'] or r['path']['cohort'] != first['cohort'] for r in records):
        raise ValueError('One candidate and cohort per summary; no pooling')
    if len({r['role'] for r in records}) != 1 or any(r['stratum'] != 'NORMAL' for r in records):
        raise ValueError('Do not pool calibration/holdout roles or forced strata')
    paths = {}
    for r in records:
        k = (r['path']['index'], r['path']['event_digest'])
        if r['axis'] in paths.setdefault(k, {}):
            raise ValueError('Duplicate path/axis')
        paths[k][r['axis']] = r
    if any(set(v) != set(p.AXES) for v in paths.values()):
        raise ValueError('Each path needs its paired X and Y record')
    n = len(paths); out = dict(candidate_id=first['candidate_id'], cohort=first['cohort'], role=records[0]['role'],
                               n_paths=n, expectation='FINITE_COHORT_ONLY', all_law_expectation=None)
    values = {a: [v[a]['D'] for v in paths.values()] for a in p.AXES}
    for a in p.AXES:
        if any(v is None for v in values[a]):
            out[a] = dict(mean_D=None, status='INCOMPLETE_PAIRS', missing=sum(v is None for v in values[a]))
            continue
        mean = Fraction(sum(values[a]), n)
        var = None if n < 2 else sum((Fraction(v) - mean) ** 2 for v in values[a]) / (n - 1)
        out[a] = dict(mean_D=mean, sample_variance_D=var, status='FINITE_COHORT_MEAN')
    if all(out[a]['mean_D'] is not None for a in p.AXES) and n >= 2:
        mx, my = out['X']['mean_D'], out['Y']['mean_D']
        out['XY_covariance_D'] = sum((Fraction(x) - mx) * (Fraction(y) - my) for x, y in zip(values['X'], values['Y'])) / (n - 1)
    else:
        out['XY_covariance_D'] = None
    out['uncertainty'] = dict(rare_order2=None, law_distance=None, long_q_R_systematic=None, total_CI=None)
    return out


# ----------------------------------------------------------------------------- legacy domain / identifiability
def legacy_domain(model, target):
    """Reasons why a saved coefficient set cannot be used for target. Empty list = evaluated point only."""
    reasons = []
    for key in ('system', 'series', 'prefix', 'delta', 'K', 'formula', 'control', 'compiler'):
        if model.get(key) != target.get(key):
            reasons.append('DOMAIN_' + key.upper() + '_DIFFERS')
    if model.get('unit') != FULL or target.get('unit') != FULL:
        reasons.append('UNIT_NOT_CONTROLLED_FULL_WRAPPER')
    if model.get('quantity') != MEAN_D:
        reasons.append('MODEL_IS_NOT_A_C_MINUS_S_RESIDUAL')
    if (target.get('q_PF'), target.get('R_outer')) not in {tuple(v) for v in model.get('evaluated_points', [])}:
        reasons.append('UNOBSERVED_Q_R_POINT')
    if model.get('qualified_for_candidate') is not True:
        reasons.append('MODEL_NOT_QUALIFIED')
    return reasons


def _row(point, features):
    q, R = point
    return [dict(intercept=1, q=q, R=R, qR=q * R)[f] for f in features]


def _rank(rows):
    m = [[Fraction(v) for v in r] for r in rows]; rank = 0; cols = len(m[0]) if m else 0
    for c in range(cols):
        pivot = next((i for i in range(rank, len(m)) if m[i][c] != 0), None)
        if pivot is None:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][c] != 0:
                f = m[i][c] / m[rank][c]; m[i] = [a - f * b for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank


def identifiability(calibration, features, holdout=()):
    """Rank of a linear residual design over DISTINCT (q, R) points; X/Y add no rows."""
    if not features or any(f not in FEATURES for f in features):
        raise ValueError('Features from ' + ','.join(FEATURES))
    cal = sorted({tuple(v) for v in calibration}); hold = sorted({tuple(v) for v in holdout})
    if set(cal) & set(hold):
        raise ValueError('Calibration and holdout points must be disjoint')
    rank = _rank([_row(v, features) for v in cal]) if cal else 0
    ratios = {Fraction(R, q) for q, R in cal}
    def kind(point):
        q, R = point
        same_q = any(c[0] == q for c in cal); same_R = any(c[1] == R for c in cal)
        return ('CROSS_OF_SEEN_LEVELS' if same_q and same_R else 'R_ONLY' if same_q
                else 'Q_ONLY' if same_R else 'JOINT_NEW')
    return dict(points=cal, features=list(features), rank=rank, parameters=len(features),
                identifiable=rank == len(features), lack_of_fit_dof=len(cal) - rank,
                R_over_q_constant=len(ratios) == 1 and len(cal) > 1,
                R_over_q_range=None if not cal else [str(min(ratios)), str(max(ratios))],
                holdout=[dict(point=v, transfer=kind(v)) for v in hold],
                note='A constant R/q makes q and R effects inseparable even when rank suffices along that line')


# ----------------------------------------------------------------------------- E[C] = E[S] + E[D]
def _same(a, b):
    return p.same(a, b)


def combine(structural, residual, *, config):
    """E[C] for one round/axis from B-accepted E[S] (C K1 envelope row) and ONE residual record.

    Never stacks residual models, never adds raw kappa(C) to S, never clamps,
    and never certifies the sum: the output is an unqualified component record.
    """
    if isinstance(residual, (list, tuple)):
        if len(residual) != 1:
            raise ValueError('Choose one residual model; do not stack C1 affine and local/paired D')
        residual = residual[0]
    if residual.get('quantity') != MEAN_D or residual.get('target') != 'C_MINUS_S':
        raise ValueError('Residual must be paired C-S; raw compiled clusters would double count S')
    if residual.get('mode') not in D_MODES:
        raise ValueError('Explicit residual mode required')
    row = config['rounds'][structural['m']]
    for key, value in dict(identity=row['identity'], law_binding=row['law_binding'],
                           physical_candidate_id=row['physical_candidate']['id'], profile=config['spec']['profile']).items():
        if not _same(structural.get(key), value) or not _same(residual.get(key), value):
            raise ValueError('E[S]/E[D] identity, law, physical candidate or profile mismatch: ' + key)
    if structural['axis'] != residual['axis'] or structural['m'] != residual['m']:
        raise ValueError('E[S] and E[D] must refer to the same round and axis')
    if structural.get('quantity', MEAN_S) != MEAN_S:
        raise ValueError('Structural side must be the full-wrapper mean S envelope')
    S, D = structural.get('mean_S'), residual.get('mean_D')
    reasons = []
    if residual.get('coverage') != FULL:
        reasons.append('RESIDUAL_NOT_FULL_WRAPPER')
    if S is None:
        reasons.append('MEAN_S_MISSING')
    if D is None:
        reasons.append('MEAN_D_MISSING')
    s_law, d_law = (structural.get('law_context') or {}).get('law'), (residual.get('law_context') or {}).get('law')
    if not s_law or s_law != d_law:
        reasons.append('S_AND_D_LAW_DIFFER_OR_UNDECLARED')
    if residual['mode'] == 'PAIRED_DIRECT_FINITE_COHORT':
        if residual.get('expectation') != 'FINITE_COHORT_ONLY' or type(residual.get('n_paths')) is not int or residual['n_paths'] < 1:
            raise ValueError('Paired direct D is a finite cohort with n_paths')
        if 'RESIDUAL_NOT_FULL_WRAPPER' in reasons or S is None or D is None:
            value = None
        else:
            value = Fraction(S) + Fraction(D)
            if value < 0:
                raise ValueError('Negative mean price; no clamping')
    else:   # existing C rule set for model residuals
        term = dict(target='C_MINUS_S', mode=residual['mode'], mean=None if D is None else float(D), coverage=residual.get('coverage'))
        c = combine_mean(None if S is None else float(S), residual_terms=[term])
        value = c['mean']
    expectation = ('LAW_EXPECTATION_S_PLUS_FINITE_COHORT_D' if residual['mode'] == 'PAIRED_DIRECT_FINITE_COHORT'
                   else 'LAW_EXPECTATION_S_PLUS_MODEL_D')
    return dict(m=structural['m'], axis=structural['axis'], quantity=MEAN_C, mean=value,
        status='MISSING_COMPONENT' if value is None else 'UNQUALIFIED_COMPONENT_SUM', reasons=reasons,
        expectation=expectation, qualified_for_candidate=False,
        components=dict(structural=dict(mean_S=S, source=deepcopy(structural.get('source')), checks=deepcopy(structural.get('checks'))),
                        residual=dict(mean_D=D, mode=residual['mode'], role=residual.get('role'), n_paths=residual.get('n_paths'),
                                      source=deepcopy(residual.get('source')))),
        uncertainty=dict(structural=deepcopy(structural.get('uncertainty')), residual=deepcopy(residual.get('uncertainty')),
                         S_D_cross_covariance=None, rare_order2=None, law_distance=None, long_q_R_systematic=None, total_CI=None))


def b_price_record(combined, *, config, cohort, source):
    """B/C price record of a component sum. Never claims QUALIFIED_EXPECTATION."""
    row = config['rounds'][combined['m']]; axis = combined['axis']
    missing = combined['mean'] is None
    return dict(**p.measurement(row, axis, saved_sampler=True), identity=deepcopy(row['identity']),
        law_binding=deepcopy(row['law_binding']), axis=axis, quantity=MEAN_C, cohort=cohort,
        classification='UNAVAILABLE' if missing else 'MODEL_PREDICTION',
        expectation='UNAVAILABLE' if missing else 'SUPPLIED_MODEL_EXPECTATION',
        price_status='MISSING' if missing else 'UNQUALIFIED_MODEL',
        mean=dict(rz_count=None if missing else float(combined['mean'])), n_paths=None,
        model=None if missing else dict(kind='E_S_PLUS_E_C_MINUS_S', qualified_for_candidate=False,
            domain=deepcopy(row['candidate']), binding=deepcopy(config['spec']['binding']),
            compiler=config['spec']['profile']['compiler'], components=deepcopy(combined['components'])),
        source=deepcopy(source), normalization_is_price_multiplier=False,
        uncertainty=dict(joint_covariance=None, R_q_systematic=None, total_CI=None, all_law_expectation=None),
        reasons=list(combined['reasons']))

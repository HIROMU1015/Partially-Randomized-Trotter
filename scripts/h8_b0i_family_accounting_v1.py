"""Fixed family metadata/and saved-cost arithmetic; no science imports."""
from copy import deepcopy
import math

REGISTRY = tuple((p, q) for p in (7, 14, 21) for q in (1, 2, 4, 8))
REUSED = ()
NEW = tuple(x for x in REGISTRY if x not in REUSED)
EPS = (.05, .01, .005, .001, .0005)
FACTORS = (1., 10.)
METRICS = ('RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size')

LIMITS = dict(snapshot_load=1,prefix_preparation=3,tail_scalar=12,analytic_overlay=12,
    native_evolution_build=12,measured_wrapper_build=24,probe_wrapper_copy=24,
    control_probe=48,compile=24)


def screen(rows, g):
    if (len(rows) != 10 or {(r['epsilon'], r['factor']) for r in rows} !=
            {(e, f) for e in EPS for f in FACTORS}):
        raise g.Stop('ALL_TEN_PRECISION_CONDITIONS_REQUIRED')
    needed = False
    for row in rows:
        a = row['accounting']
        if a.get('cost_acquisition_eligible') is True:
            if any(type(a['axes'][k].get('shots')) is not int or a['axes'][k]['shots'] <= 0
                    for k in ('real', 'imag')):
                raise g.Stop('ELIGIBLE_INTEGER_SHOTS_REQUIRED')
            needed = True
        elif (a.get('cost_acquisition_eligible') is not False or
                a.get('eligibility_under_declared_allowance') != 'INELIGIBLE'):
            raise g.Stop('UNRESOLVED_SCREEN_CANNOT_SKIP_COMPILE')
    return needed


def cost_rows(cases, g):
    output = []
    if len(cases) != 12 or {(c['prefix'], c['q']) for c in cases} != set(REGISTRY):
        raise g.Stop('FULL_FAMILY_TWELVE_CASE_COVERAGE')
    for case in cases:
        screen(case['accounting'], g)
        for row in case['accounting']:
            a = row['accounting']; eligible = a['cost_acquisition_eligible']
            axes = case['axes']; metrics = {}
            if eligible:
                if axes is None or set(axes) != {'cosine', 'sine'}:
                    raise g.Stop('ELIGIBLE_B0I_COST_UNMEASURED')
                for metric in METRICS:
                    point = sum(a['axes'][channel]['shots'] * axes[axis]['metrics'][metric]
                        for axis, channel in g.d0.AXIS_CHANNEL.items())
                    metrics[metric] = dict(G_point=point, G_standard_error=None, paired_samples=1)
            output.append(dict(cell_id=f"B0I_p{case['prefix']}_q{case['q']}_2nd", method='B0I',
                cell=dict(prefix=case['prefix'], q=case['q'], formula='2nd'),
                epsilon=row['epsilon'], factor=str(float(row['factor'])),
                status='DERIVED_SAVED_SCALAR' if eligible else 'INELIGIBLE',
                cost_status='ACQUIRED' if axes is not None else 'UNMEASURED',
                accounting=a, N_total=a.get('N_total'), metrics=metrics,
                origin_SOURCE=case['origin_SOURCE'], acquisition=case['acquisition'],
                matched_B0_delta=None, population_winner_certified=False))
    return output


def compare(b0i_rows, saved, contract, g, guard=lambda: None):
    """Frozen D0 baselines, confirmation n32 only; no signal reaccounting here."""
    old = saved['H8_CONFIRMATION_UNION6']
    expected = set(contract['comparison_policy']['candidate_ids'])
    if (len(expected) != 26 or len(old) != 260 or
            {(r['cell_id'], r['epsilon'], float(r['factor'])) for r in old} !=
            {(cid, e, f) for cid in expected for e in EPS for f in FACTORS}):
        raise g.Stop('D0_CONFIRMATION_REGISTERED_DOMAIN_COVERAGE')
    for row in old:
        method = row['method']; cell = row['cell']
        if (method not in ('B0', 'B1', 'B2') or
                (method == 'B0' and (cell['prefix'], cell['q']) not in REGISTRY) or
                (method == 'B1' and (cell['prefix'] != 28 or cell['formula'] not in ('2nd', '4th'))) or
                (method == 'B2' and (row.get('sample_phase') != 'confirmation' or row.get('sample_count') != 32))):
            # Unacquired INELIGIBLE PR rows still require their fixed confirmation provenance.
            raise g.Stop('D0_METHOD_DOMAIN_OR_CONFIRMATION_NOT_N32')
    all_rows = old + b0i_rows; comparisons = []; sensitivity = []
    for epsilon in EPS:
        for factor in FACTORS:
            subset = [r for r in all_rows if r['epsilon'] == epsilon and float(r['factor']) == factor]
            for metric in METRICS:
                guard(); good = []
                for row in subset:
                    if row['status'] == 'INELIGIBLE': continue
                    if row['status'] != 'DERIVED_SAVED_SCALAR' or metric not in row['metrics']:
                        raise g.Stop('ELIGIBLE_COST_MISSING_OR_UNRESOLVED')
                    value = row['metrics'][metric].get('G_point')
                    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0 or
                            type(row.get('N_total')) is not int or row['N_total'] <= 0):
                        raise g.Stop('FINITE_COST_INTEGER_SHOTS_REQUIRED')
                    good.append(row)
                def best(rows):
                    if not rows: return None
                    row = min(rows, key=lambda r: (r['metrics'][metric]['G_point'], r['cell_id']))
                    return dict(cell_id=row['cell_id'], G_point=row['metrics'][metric]['G_point'])
                groups = dict(B0=[r for r in good if r['method'] == 'B0'],
                    B0I=[r for r in good if r['method'] == 'B0I'],
                    full_S2=[r for r in good if r['method'] == 'B1' and r['cell']['formula'] == '2nd'],
                    full_S4=[r for r in good if r['method'] == 'B1' and r['cell']['formula'] == '4th'])
                det = best([r for r in good if r['method'] in ('B0', 'B0I', 'B1')])
                pr = best([r for r in good if r['method'] == 'B2'])
                comparisons.append(dict(epsilon=epsilon, factor=factor, metric=metric,
                    family_minima={k: best(v) for k, v in groups.items()},
                    best_registered_deterministic=det, confirmed_PR_union8=pr,
                    PR_over_deterministic=(pr['G_point'] / det['G_point'] if pr and det and det['G_point'] > 0 else None),
                    point_only=True, matched_B0_delta=None, population_winner_certified=False))
                lines = [dict(cell_id=r['cell_id'], slope=r['N_total'], intercept=r['metrics'][metric]['G_point'])
                    for r in good]
                sensitivity.append(dict(epsilon=epsilon, factor=factor, metric=metric,
                    units=metric+' per shot', P_model='COMMON_HYPOTHETICAL_NOT_MEASURED',
                    confirmation_scope='B2_UNION6_N32_ONLY', lower_envelope=g.d0.lower_envelope(lines)))
    return dict(comparisons=comparisons, P_sensitivity=sensitivity,
        registered_domain_cost_coverage_complete=True, population_winner_certified=False,
        scientific_decision=None, next_stage_authorized=False)

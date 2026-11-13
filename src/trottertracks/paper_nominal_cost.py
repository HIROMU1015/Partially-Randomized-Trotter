"""Level 1 (paper-type) nominal shots and guarded G_paper accounting.

N_paper per axis is B's existing pr_rpe_rmse.nominal_axis_shots, i.e.
ceil(B_m^2 [11 + 4(M - m)]), with the method's OWN finite log B taken from the
signal packet (deterministic PF: B = 1). It is a nominal paper schedule, not a
sufficient / MSE-qualified N, so it never enters B.assess/B.cost/B.select.
G_paper is computed only after the C implementation guard has admitted every
valued price; missing prices stay UNKNOWN (never zero) and model / finite-cohort
prices downgrade the reported status. No saved data, compiler or provider here.
"""
from copy import deepcopy
from fractions import Fraction
import sys

from . import price_implementation_guard as G
from . import rpe_optimization_pipeline as I

MODULE = 'src/trottertracks/paper_nominal_cost.py'
LEVEL = 'PAPER_NOMINAL_LEVEL1'
EVIDENCE = 'PAPER_NOMINAL'
N_STATUS = 'PAPER_NOMINAL_NOT_SUFFICIENT'
DETERMINISTIC = ('FULL_PF', 'TRUNCATED_PF')
# Weakest tier present determines the G_paper label (left = strongest).
TIERS = ('DETERMINISTIC_DIRECT', 'QUALIFIED_EXPECTATION', 'QUALIFIED_MODEL',
         'FINITE_COHORT_SCENARIO', 'UNQUALIFIED_MODEL', 'UNPAIRED_XY_UNQUALIFIED')
UNAVAILABLE = ('MISSING', 'NOT_EVALUATED', 'OUTSIDE_DOMAIN', 'FAILED', 'INFEASIBLE',
               'NUMERICAL_UNSTABLE', 'SELECTION_REQUIRED')


def _api(root):
    api = I.providers(str(root))
    return api, sys.modules[api.B.__package__].pr_rpe_rmse


def nominal_packet(root, config, signal_packet, *, source, normalization_rule='finite_B_diagnostic'):
    """Paper nominal N for every round/axis; identity bound to the consumed signal."""
    api, R = _api(root); B = api.B; B._validate(config)
    if normalization_rule != 'finite_B_diagnostic':
        raise ValueError('Only the finite_B_diagnostic rule is connected; no paper kappa envelope substitution')
    if not source or signal_packet.get('configuration_id') != config['id']:
        raise ValueError('Explicit SOURCE and the signal packet of this configuration required')
    method = B.comparison_method(config['spec']); M = config['spec']['M']
    means = {v['m']: v for v in signal_packet['means']}
    if len(means) != len(signal_packet['means']):
        raise ValueError('Duplicate signal rounds')
    rows = []
    for row in config['rounds']:
        mean = means.get(row['m']); out = dict(m=row['m'], identity=deepcopy(row['identity']),
            law_binding=deepcopy(row['law_binding']), log_B_used=None, nominal_N_X=None, nominal_N_Y=None,
            nominal_status='NOT_EVALUATED_MISSING_NORMALIZATION', sufficient_N_X=None, sufficient_N_Y=None)
        log_B = None if mean is None else (mean.get('normalization') or {}).get('log_B')
        if mean is not None and (not api.C.p.same(mean.get('identity'), row['identity'])
                                 or not api.C.p.same(mean.get('law_binding'), row['law_binding'])):
            raise ValueError('Signal mean identity/law differs from configuration round')
        if log_B is not None:
            if method in DETERMINISTIC and log_B != 0:
                raise ValueError('Deterministic PF normalization must be B = 1')
            n = R.nominal_axis_shots(M, row['m'], log_B)
            out.update(log_B_used=log_B, nominal_N_X=n['nominal_N_X'], nominal_N_Y=n['nominal_N_Y'],
                       nominal_status=n['status'], ceiling_precision_stable=n.get('ceiling_precision_stable'))
        rows.append(out)
    complete = all(type(r['nominal_N_X']) is int and type(r['nominal_N_Y']) is int for r in rows)
    return dict(kind=LEVEL, evidence_kind=EVIDENCE, status=N_STATUS if complete else 'NOT_EVALUATED_MISSING_NORMALIZATION',
        configuration_id=config['id'], comparison_method=method, rounds=rows,
        rule=dict(formula='ceil(B_m^2 [11 + 4(M - m)]) per axis', normalization_rule=normalization_rule,
                  function='pr_rpe_rmse.nominal_axis_shots', B_source=I.B_SOURCE),
        paper_applicability='NOT_EVALUATED_NO_HORIZON_OR_BUDGET_CHECK',
        signal_input=I.signal_input(signal_packet), source=deepcopy(source),
        sufficient_N=None, rmse_qualified=False, global_RMSE_bound=None)


def _tier(B, price, method):
    value = (price.get('mean') or {}).get('rz_count')
    if value is None or price.get('price_status') in UNAVAILABLE:
        return None
    if B.a.exact_scalar(value) < 0:
        raise ValueError('Negative compiled price; no clamp')
    kind = price.get('classification')
    if kind not in B.a.PRICE_KINDS:
        raise ValueError('Unknown compiled price kind')
    if kind == 'MODEL_PREDICTION':
        model = price.get('model') or {}
        return ('QUALIFIED_MODEL' if model.get('qualified_for_candidate') is True and model.get('validation_source')
                else 'UNQUALIFIED_MODEL')
    if kind == 'DIRECT_COMPILED_DETERMINISTIC':
        if method not in DETERMINISTIC or price.get('expectation') != 'DETERMINISTIC':
            raise ValueError('Deterministic price outside deterministic PF')
        return 'DETERMINISTIC_DIRECT'
    law = price.get('law_context') or {}
    if price.get('expectation') == 'QUALIFIED_EXPECTATION' and law.get('ideal_sampler_distance_qualified') is True:
        return 'QUALIFIED_EXPECTATION'
    return 'FINITE_COHORT_SCENARIO'


def paper_cost(root, config, packet, prices, *, cohorts, price_bindings, comparison_policy):
    """Guard first, then exact sum_m,a N_paper * E[C]; never B.cost, never zero-fill."""
    if comparison_policy is None:
        raise ValueError('Level-1 G_paper also requires an explicit implementation comparison policy')
    api, _ = _api(root); B = api.B; B._validate(config)
    check = G.check(root, config, prices, cohorts=cohorts, bindings=price_bindings, policy=comparison_policy)
    if packet.get('kind') != LEVEL or packet.get('evidence_kind') != EVIDENCE or packet.get('configuration_id') != config['id']:
        raise ValueError('Paper nominal packet of this configuration required')
    method = B.comparison_method(config['spec']); rows = {r['m']: r for r in packet['rounds']}
    total = 0; lower = 0; upper = 0; intervals = True; missing = []; cells = []; tiers = set(); terms = []
    for row in config['rounds']:
        n = rows.get(row['m'])
        if n is None or not api.C.p.same(n['identity'], row['identity']):
            raise ValueError('Nominal N round identity differs')
        pair = []
        for axis in B.p.AXES:
            cohort = cohorts.get((row['candidate']['id'], axis))
            match = [v for v in prices if v.get('axis') == axis and v.get('cohort') == cohort
                     and api.C.p.same(v.get('identity'), row['identity'])]
            N = n['nominal_N_' + axis]; price = match[0] if len(match) == 1 else None
            tier = None if price is None else _tier(B, price, method)
            cell = dict(m=row['m'], axis=axis, nominal_N=N, tier=tier, cohort=cohort,
                        source=None if price is None else deepcopy(price.get('source')))
            if tier is None or type(N) is not int:
                missing.append(dict(m=row['m'], axis=axis, reason='MISSING_PRICE' if tier is None else 'MISSING_NOMINAL_N'))
            else:
                value = B.a.exact_scalar(price['mean']['rz_count']); term = N*value; total += term
                terms.append(dict(m=row['m'], axis=axis, N=N, price=B.a.exact_value(value), term=B.a.exact_value(term)))
                tiers.add(tier); pair.append(price)
                side = [s for s in price_bindings if s.get('axis') == axis and s.get('cohort') == cohort
                        and api.C.p.same(s.get('identity'), row['identity'])]
                box = (side[0].get('mean_interval') or {}) if len(side) == 1 else {}
                if box.get('lower') is None or box.get('upper') is None:
                    intervals = False
                else:
                    lo, hi = B.a.exact_scalar(box['lower']), B.a.exact_scalar(box['upper'])
                    if not lo <= value <= hi:
                        raise ValueError('Supplied price interval excludes its point value')
                    lower += N*lo; upper += N*hi
            cells.append(cell)
        if len(pair) == 2 and any(pair[0].get(k) != pair[1].get(k) for k in ('cohort', 'n_paths', 'classification', 'expectation')):
            tiers.add('UNPAIRED_XY_UNQUALIFIED')
            for cell in cells[-2:]:
                cell['tier'] = 'UNPAIRED_XY_UNQUALIFIED'
    weakest = max(tiers, key=TIERS.index) if tiers else None
    status = ('NOT_EVALUATED_MISSING_PRICE_OR_N' if missing else 'PAPER_NOMINAL_G_' + weakest)
    return dict(level=LEVEL, configuration_id=config['id'], comparison_method=method,
        route=G.route(B, config['spec']), status=status, G_paper=None if missing else B.a.exact_value(total),
        G_paper_interval=None if missing or not intervals else [B.a.exact_value(lower), B.a.exact_value(upper)],
        weakest_price_tier=None if missing else weakest, missing=missing, cells=cells, terms=terms if not missing else [],
        nominal_N_source=deepcopy(packet['source']), signal_input=deepcopy(packet['signal_input']),
        price_implementation_check=check, comparison_policy=deepcopy(comparison_policy),
        task={k: deepcopy(config['spec'][k]) for k in ('binding', 'h', 'M', 'profile')},
        G_qualified=None, sufficient_N=None, mse_qualified=False, normalization_is_price_multiplier=False,
        numeric_encoding='exact_rational', source=dict(module=MODULE))


def paper_compare(results):
    """Order level-1 results under ONE policy and common task; no qualified winner."""
    if not results or any(r.get('level') != LEVEL for r in results):
        raise ValueError('Level-1 paper results only')
    policy = results[0]['comparison_policy']
    if any(not I.C.p.same(r['comparison_policy'], policy) for r in results):
        raise ValueError('One common SOURCE-bound implementation comparison policy required')
    if any(not I.C.p.same(r['task'], results[0]['task']) for r in results):
        raise ValueError('Common H/state/h/M/price profile required')
    if len({r['configuration_id'] for r in results}) != len(results):
        raise ValueError('Unique configurations required')
    def exact(v):
        return Fraction(v['numerator'], v['denominator']) if isinstance(v, dict) else Fraction(v)
    valued = sorted((r for r in results if r['G_paper'] is not None), key=lambda r: exact(r['G_paper']))
    return dict(level=LEVEL, policy_id=policy.get('id'),
        ranking=[dict(configuration_id=r['configuration_id'], route=r['route'], G_paper=r['G_paper'],
                      status=r['status'], G_paper_interval=r['G_paper_interval']) for r in valued],
        unknown=[dict(configuration_id=r['configuration_id'], status=r['status']) for r in results if r['G_paper'] is None],
        point_minimum=None if not valued else valued[0]['configuration_id'],
        interval_separated=None, qualified_winner=None, G_qualified=None,
        scope='PAPER_NOMINAL_FINITE_MENU_NOT_MSE_QUALIFIED')

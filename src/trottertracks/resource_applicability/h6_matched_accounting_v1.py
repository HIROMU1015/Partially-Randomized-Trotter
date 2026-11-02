"""Conditional empirical accounting and fixed, task-bound cost selection."""
from __future__ import annotations
import hashlib
import math
import statistics
import sys
from .ax2b_numerical_accounting import u_aware_shots


def seed(cell_id, phase, replica):
    if phase not in ('exploration', 'confirmation') or type(replica) is not int or replica < 0:
        raise ValueError('COST_SEED_IDENTITY')
    s = f'H6_MATCHED_V1_FRESH_COST/{cell_id}/{phase}/{replica}'
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], 'big')


def empirical_allowance(*, reference_error, path_errors, normalization_error,
                        raw_closure, maximum_norm, work, settings):
    values = [reference_error, normalization_error, raw_closure, maximum_norm, *path_errors]
    if (any(not math.isfinite(x) or x < 0 for x in values) or
            type(work) is not int or work < 0):
        raise ValueError('NONFINITE_ALLOWANCE_INPUT')
    rounding = settings['roundoff_multiplier']*sys.float_info.epsilon*(1+work)*max(1.,maximum_norm)
    agreement = reference_error + max(path_errors, default=0.) + raw_closure
    u = settings['safety_factor']*max(settings['floor'], rounding, agreement)
    du = settings['safety_factor']*max(settings['floor'], rounding, normalization_error)
    if not math.isfinite(u+du):
        raise ValueError('ALLOWANCE_OVERFLOW')
    return dict(real=u, imag=u, log_B_margin=du, rounding_proxy=rounding,
                agreement=agreement, kind='EMPIRICAL', certified=False,
                scope=settings['scope'], local_probe_discrepancy_is_global_bound=False)


def precision_rows(record, plan):
    rows = {}
    b = record['error_decomposition']['total_signed']
    allowance = record['allowance']
    for eps in plan['epsilons']:
        variants = {}
        for factor in plan['numerical']['sensitivity_factors']:
            r = u_aware_shots({a:abs(b[a]) for a in ('real','imag')},
                    {a:factor*allowance[a] for a in ('real','imag')}, epsilon=eps,
                    log_B_upper=record['log_B']+factor*allowance['log_B_margin'],
                    evidence_kind='EMPIRICAL', evidence_ref=record['cell']['id']+'_signal.json',
                    alpha_axis=plan['shot_rule']['alpha_axis'])
            ratios = [v['u_over_headroom'] for v in r['axes'].values()]
            usable = (r['eligibility_under_declared_allowance']=='ELIGIBLE' and r['N_total'] is not None
                      and all(x is not None and x <= plan['numerical']['maximum_u_over_headroom'] for x in ratios))
            r['cost_acquisition_eligible'] = usable
            r['boundary_guard'] = 'PASS' if usable else 'INELIGIBLE_OR_NUMERICALLY_UNRESOLVED'
            variants[str(factor)] = r
        rows[str(eps)] = variants
    return rows


def eligible(record):
    return any(v['1.0']['cost_acquisition_eligible'] for v in record['precision_rows'].values())


def metric_summary(rows, metric):
    v = [r['metrics'][metric] for r in rows]
    if not v or any(type(x) is not int or x < 0 for x in v):
        raise ValueError('COST_STATISTICS_INPUT')
    sd = statistics.stdev(v) if len(v)>1 else None
    return dict(n=len(v), mean=statistics.mean(v), min=min(v), max=max(v),
                sample_sd=sd, standard_error=sd/math.sqrt(len(v)) if sd is not None else None,
                population_mean_certified=False, interval_is_confidence_bound=False)


def resource_row(signal, cost_rows, epsilon, *, metric='RZ', factor='1.0'):
    accounting = signal['precision_rows'][str(epsilon)][factor]
    if not accounting['cost_acquisition_eligible']:
        return None
    summaries = {}
    for axis, channel in (('cosine','real'), ('sine','imag')):
        rows = [r for r in cost_rows if r['axis']==axis and r['control']=='symmetric_directional']
        if not rows:
            return None
        summaries[axis] = metric_summary(rows, metric)
        summaries[axis]['shots'] = accounting['axes'][channel]['shots']
    value = sum(s['shots']*s['mean'] for s in summaries.values())
    # Pairing across axes is retained when estimating variability of G.
    paired = {}
    for r in cost_rows:
        if r['control']=='symmetric_directional':
            paired.setdefault(r['replica'], {})[r['axis']] = r['metrics'][metric]
    if any(set(x) != {'cosine','sine'} for x in paired.values()):
        raise ValueError('INCOMPLETE_PAIRED_AXES')
    totals = [sum(summaries[a]['shots']*v[a] for a in ('cosine','sine')) for v in paired.values()]
    se = statistics.stdev(totals)/math.sqrt(len(totals)) if len(totals)>1 else None
    return dict(cell_id=signal['cell']['id'], method=signal['cell']['method'], epsilon=epsilon,
                metric=metric, axes=summaries, G_point=value, G_standard_error=se,
                N_total=accounting['N_total'], preparation_cost_P=0.,
                common_preparation_slope=accounting['N_total'], formal_winner_certified=False)


def confirmation_ids(signals, cost_rows, plan):
    ids = set()
    for eps in plan['epsilons']:
        rows = []
        for s in signals:
            if s['cell']['method'] != 'B2':
                continue
            r = resource_row(s, [c for c in cost_rows if c['cell_id']==s['cell']['id']], eps)
            if r is not None:
                rows.append(r)
        rows.sort(key=lambda r:(r['G_point'], r['cell_id']))
        ids.update(r['cell_id'] for r in rows[:plan['cost_rule']['confirmation_top_per_epsilon']])
    if len(ids)>plan['cost_rule']['confirmation_union_max']:
        raise ValueError('CONFIRMATION_SELECTION_CAP')
    return sorted(ids)


def tasks(signals, plan, phase, selected=None):
    result = []
    for s in signals:
        c = s['cell']
        if not eligible(s) or (phase=='confirmation' and c['id'] not in (selected or [])):
            continue
        n = (plan['cost_rule']['confirmation_random_trajectories'] if phase=='confirmation'
             else plan['cost_rule']['exploratory_random_trajectories'] if c['method']=='B2' else 1)
        for i in range(n):
            result.append(dict(cell=c, cell_id=c['id'], phase=phase, replica=i,
                seed=seed(c['id'], phase, i), validate=(phase=='exploration' and i==0),
                ordinary=(phase=='exploration' and i==0 and c['id'] in plan['sensitivity_cells'])))
    return result


def order_diagnostic(distribution, observed_orders, *, R, trajectories):
    """Known order-law masses; no implication for an unobserved cost mean."""
    probabilities = dict(zip(distribution['orders'],distribution['order_probabilities'],strict=True))
    seen = set(observed_orders)
    unseen_mass = sum(p for k,p in probabilities.items() if k not in seen)
    return dict(orders=probabilities,observed_counts={k:observed_orders.count(k) for k in probabilities},
        R=R,trajectories=trajectories,all_zero_order_trajectory_probability=probabilities[0]**R,
        probability_trajectory_contains_unseen_order=1-(1-unseen_mass)**R,
        probability_no_nonzero_order_in_sample=probabilities[0]**(R*trajectories),
        cost_mean_tail_bound=None,instruction_cap_is_population_bound=False,
        component_sequence_support_fully_covered=False,population_mean_certified=False)

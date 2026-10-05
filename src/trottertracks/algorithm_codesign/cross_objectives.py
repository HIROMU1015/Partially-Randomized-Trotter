"""Prespecified post-search objective attribution on the existing union only."""
from __future__ import annotations

ARMS = ('O', 'L', 'F')


def cross_score(evaluator, search_points, fixed_points, on_row=None):
    """No new candidates, optimizer calls, ideal signals or finite cells.

    Every arm must have completed its 32 evaluations. Primary finite
    rescoring must already have populated all required cells. Missing
    cells stop the run instead of acquiring data from a diagnostic path.
    """
    if set(search_points) != set(ARMS) or any(
            len(search_points[a]) != 32 or len({p.identity for p in search_points[a]}) != 32 for a in ARMS):
        raise RuntimeError('CROSS_SCORE_REQUIRES_THREE_COMPLETED_32_POINT_SEARCHES')
    if len(fixed_points) != 4:
        raise RuntimeError('CROSS_SCORE_REQUIRES_FOUR_FIXED_REFERENCES')
    union, origins = {}, {}
    for arm in ARMS:
        for point in search_points[arm]:
            union[point.identity] = point
            origins.setdefault(point.identity, set()).add(arm)
    fixed_labels = {}
    for point in fixed_points:
        union[point.identity] = point
        origins.setdefault(point.identity, set()).add('fixed')
        fixed_labels.setdefault(point.identity, []).append(point.label)
    before = dict(ideal=len(evaluator.ideal), finite=len(evaluator.finite))
    rows = []
    with evaluator.cached_diagnostics():
        for identity, point in sorted(union.items()):
            evaluator.limits.check()
            row = dict(coefficient=identity, origins=sorted(origins[identity]),
                       fixed_labels=sorted(fixed_labels.get(identity, [])),
                       scores={a: evaluator.objective_record(point, a) for a in ARMS})
            rows.append(row)
    after = dict(ideal=len(evaluator.ideal), finite=len(evaluator.finite))
    if after != before:
        raise RuntimeError('CROSS_SCORE_CHANGED_PHYSICAL_CELL_COUNTS')
    minima, own_best = {}, {}
    for arm in ARMS:
        values = sorted(r['scores'][arm]['value'] for r in rows if r['scores'][arm]['feasible'])
        minimum = values[0] if values else None
        minima[arm] = dict(value=minimum, coefficients=[r['coefficient'] for r in rows
                          if minimum is not None and r['scores'][arm]['value'] == minimum])
        own = [r for r in rows if arm in r['origins'] and r['scores'][arm]['feasible']]
        own_minimum = min((r['scores'][arm]['value'] for r in own), default=None)
        own_best[arm] = dict(value=own_minimum, coefficients=[r['coefficient'] for r in own
                            if r['scores'][arm]['value'] == own_minimum])
        for row in rows:
            score = row['scores'][arm]
            value = score['value']
            # NumPy comparisons can promote the count to int64; JSON needs int.
            score['union_rank'] = int(1+sum(v < value for v in values)) if score['feasible'] else None
            score['relative_regret_to_union_minimum'] = value/minimum-1 if score['feasible'] else None
    if on_row:
        for row in rows:
            on_row(dict(kind='cross_objective', **row))
    return dict(schema='bf1_postsearch_cross_objective_v1', phase='POST_SEARCH_PRIMARY_RESCORE',
                epsilon=.01, rows=rows, union_minima=minima, own_search_minima=own_best,
                union_coefficient_count=len(rows), logical_cross_scores=3*len(rows),
                added_search_evaluations=0, added_ideal_cells=0, added_finite_cells=0,
                cell_counts_before=before, cell_counts_after=after, primary_route_changed=False)


def attribute_F_winner(cross, decision):
    """Descriptive point-estimate attribution; it cannot upgrade BF-C."""
    result = dict(diagnostic_only=True, primary_outcome=decision['outcome'],
                  affects_search=False, affects_primary_classification=False,
                  establishes_design_principle=False)
    winner = decision.get('best_F')
    if winner is None:
        return dict(**result, interpretation='NO_ELIGIBLE_F_WINNER')
    row = next(r for r in cross['rows'] if r['coefficient'] == winner['coefficient'])
    L_score, F_score = row['scores']['L'], row['scores']['F']
    L_minimum = cross['own_search_minima']['L']['value']
    if L_minimum is None:
        interpretation = 'NO_FEASIBLE_L_DESIGN_REFERENCE'
    elif L_score['feasible'] and L_score['value'] <= L_minimum:
        interpretation = 'SEARCH_REACHABILITY_OR_BUDGET_EXPLANATION_NOT_EXCLUDED'
    elif F_score['feasible']:
        interpretation = 'L_DID_NOT_PREFER_F_WINNER_POINT_ESTIMATE'
    else:
        interpretation = 'UNRESOLVED_OBJECTIVE_ATTRIBUTION'
    return dict(**result, interpretation=interpretation, F_winner_coefficient=row['coefficient'],
                F_winner_cross_scores=row['scores'], L_own_search_minimum=L_minimum,
                explanation='Compare each objective only within itself; point-estimate ordering is not a numerical certification of the mechanism.')

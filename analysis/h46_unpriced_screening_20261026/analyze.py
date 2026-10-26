"""Saved-data screening only. No scientific builder, sampler or compiler calls.

Run from any directory: python3 analysis/h46_unpriced_screening_20261026/analyze.py
Original study outputs and baselines are read-only. All writes stay beside this file.
"""
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import h46_long_time_signal as study
import h46_long_time_report as report
from h46_long_time_native import cost_proxy

BASE = '155529e20a8fac39e00786c40b57830b71203c4f'


def write_csv(name, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode()
    (HERE / name).write_bytes(gzip.compress(data, mtime=0) if name.endswith('.gz') else data)


def family(c):
    return 'PR' if c['method'] == 'PR' else 'PF'


def key(c, fields):
    return tuple(c.get(k) for k in fields)


STRUCTURE = ('system', 'method', 'formula', 'prefix', 'q', 'R_outer')
QSCALE = ('system', 'method', 'formula', 'prefix', 'R_outer')
RSCALE = ('system', 'method', 'formula', 'prefix')
FALLBACK = ('system', 'method', 'formula')


class Neighbors:
    """Transparent measured analogues, with proxy-only fallbacks kept separate."""

    def __init__(self, anchors):
        self.index = {}
        for fields in (STRUCTURE, QSCALE, RSCALE, FALLBACK):
            ix = defaultdict(list)
            for a in anchors:
                ix[key(a, fields)].append(a)
            self.index[fields] = ix

    def predict(self, c, holdout=None):
        def get(fields):
            return [a for a in self.index[fields].get(key(c, fields), [])
                    if holdout is None or not holdout(a)]
        anchors = get(STRUCTURE)
        tier = 'EMPIRICAL_SAME_STRUCTURE_OTHER_T_OR_K'
        scale = lambda a: 1.
        if not anchors:
            anchors = get(QSCALE)
            tier = 'EMPIRICAL_Q_PROPORTIONAL'
            scale = lambda a: c['q'] / a['q']
        # Different R or prefix is deliberately only a calibrated work proxy.
        # No unvalidated fit or interpolation is promoted to a measured price.
        if not anchors:
            anchors = get(RSCALE) or get(FALLBACK)
            tier = 'PROXY_ONLY_DIFFERENT_R_OR_PREFIX'
            if anchors:
                distance = lambda a: (abs(a['prefix'] - c['prefix']),
                                      abs((a.get('R_outer') or 1) - (c.get('R_outer') or 1)))
                nearest = min(map(distance, anchors))
                anchors = [a for a in anchors if distance(a) == nearest]
            scale = lambda a: cost_proxy(c) / cost_proxy(a)
        if not anchors:
            return dict(estimate_kind='INSUFFICIENT_INFORMATION', C_cos_est=None, C_sin_est=None,
                        analogue_ids='', analogue_count=0)
        result = dict(estimate_kind=tier, analogue_ids=';'.join(sorted(a['id'] for a in anchors)),
                      analogue_count=len(anchors),
                      analogue_max_unit_SE=max((a['unit_SE'] or 0) * scale(a) for a in anchors))
        for axis in ('cos', 'sin'):
            values = [a['C_' + axis] * scale(a) for a in anchors]
            result['C_' + axis + '_est'] = statistics.median(values)
            result['C_' + axis + '_analogue_min'] = min(values)
            result['C_' + axis + '_analogue_max'] = max(values)
        return result

    def r_secant(self, c):
        """Reference proxy from two measured R groups; never a bound or price."""
        if c['method'] != 'PR':
            return {}
        groups = defaultdict(list)
        for a in self.index[RSCALE].get(key(c, RSCALE), []):
            groups[a['R_outer']].append(a)
        rs = sorted(groups)
        if len(rs) < 2 or c['R_outer'] in rs:
            return {}
        r = c['R_outer']
        if r < rs[0]:
            lo, hi = rs[:2]
        elif r > rs[-1]:
            lo, hi = rs[-2:]
        else:
            lo, hi = max(x for x in rs if x < r), min(x for x in rs if x > r)
        result = dict(R_secant_proxy_kind='INTERPOLATION' if lo < r < hi else 'EXTRAPOLATION',
                      R_secant_proxy_R_low=lo, R_secant_proxy_R_high=hi)
        for axis in ('cos', 'sin'):
            a = statistics.median(x['C_' + axis] / x['q'] for x in groups[lo])
            b = statistics.median(x['C_' + axis] / x['q'] for x in groups[hi])
            result['R_secant_proxy_C_' + axis] = c['q'] * (a + (b - a) * (r - lo) / (hi - lo))
        result['R_secant_proxy_analogue_ids'] = ';'.join(sorted(a['id'] for rr in (lo, hi) for a in groups[rr]))
        return result


def main():
    endpoints, prices = report.endpoints(), report.costs()
    candidates = {e['candidate']['id']: e['candidate'] for e in endpoints}
    assert len(candidates) == 11632 and len(endpoints) == 58160 and len(prices) == 642
    assert len({(e['candidate']['id'], e['epsilon']) for e in endpoints}) == len(endpoints)
    assert all(study.cid(c) == c['id'] for c in candidates.values())
    source_paths = set()
    def load(path):
        source_paths.add(path)
        return study.read(path)
    public = study.PUBLIC
    best_records = load(public / 'best_configurations.json')
    best = {(b['system'], b['T'], b['epsilon'], b['family']): b['best']
            for b in best_records if b['formula_group'] == 'ALL'}
    assert len(best) == 80
    anchors = []
    for identity, price in prices.items():
        c = candidates[identity]
        if 'candidate' in price:
            assert price['candidate']['id'] == identity
            assert all(price['candidate'].get(k) == c.get(k) for k in STRUCTURE + ('T', 'K'))
        unit = report.cost_for(dict(candidate=c, N_cos=1, N_sin=1), price)
        anchors.append(dict(c, C_cos=unit['C_cos'], C_sin=unit['C_sin'],
                            unit_SE=None if unit['G_SE'] is None else unit['G_SE'] / 2,
                            n=unit['n']))
    model = Neighbors(anchors)
    predictions = {i: model.predict(c) for i, c in candidates.items() if i not in prices}
    r_proxies = {i: model.r_secant(c) for i, c in candidates.items() if i not in prices}

    # Retrospective diagnostics, not confidence bounds or independent validation.
    diagnostics = []
    for a in anchors:
        for label, fields in [('SAME_STRUCTURE', None), ('Q_TRANSFER', STRUCTURE)]:
            holdout = ((lambda x, i=a['id']: x['id'] == i) if fields is None else
                       (lambda x, a=a: key(x, STRUCTURE) == key(a, STRUCTURE)))
            pred = model.predict(a, holdout)
            intended = ('EMPIRICAL_SAME_STRUCTURE_OTHER_T_OR_K' if fields is None else
                        'EMPIRICAL_Q_PROPORTIONAL')
            if pred['estimate_kind'] != intended:
                continue
            actual = (a['C_cos'] + a['C_sin']) / 2
            expected = (pred['C_cos_est'] + pred['C_sin_est']) / 2
            diagnostics.append(dict(id=a['id'], system=a['system'], family=family(a),
                formula=a['formula'], diagnostic=label, actual_C=actual, predicted_C=expected,
                actual_over_prediction=actual / expected,
                absolute_relative_error=abs(expected / actual - 1),
                analogue_ids=pred['analogue_ids']))
    write_csv('analogue_diagnostics.csv', diagnostics)

    # Reuse the original accepted loader and ID mapping. Never invoke inventory()
    # or account(), which write into the original study.
    old_prices = study.saved_assets()[2]
    old_match = set(old_prices) & set(candidates)
    assert len(old_match) == 203 and not (old_match - set(prices))
    for i in old_match:
        ep = dict(candidate=candidates[i], N_cos=1, N_sin=1)
        assert report.cost_for(ep, old_prices[i])['G_exact'] == report.cost_for(ep, prices[i])['G_exact']
    sources = ('reused_compiled_prices.json.gz', 'exploration_costs.json.gz',
               'extension_exploration_costs.json.gz', 'extension_parallel_costs.json.gz',
               'fixed_delta_exploration_costs.json.gz')
    price_sources = defaultdict(list)
    for name in sources:
        if not (public / name).exists():
            continue
        for i in load(public / name):
            price_sources[i].append(name)
    confirmation = load(public / 'confirmation_costs.json.gz')
    old_confirmation = load(public / 'reused_confirmation_costs.json.gz')
    assert not ((set(confirmation) | set(old_confirmation)) - set(prices))
    for s in study.PREFIX:
        for name in ('PF_endpoint_costs.json.gz', 'old_cost_identity_audit.json'):
            source_paths.add(ROOT / f'artifacts/resource_applicability/track_a_pf_common_{s}_results/2026-10-22/{name}')
    source_paths.add(ROOT / 'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/evaluation.json')
    write_csv('price_reuse_audit.csv', [dict(
        id=i, **{k: c[k] for k in ('system', 'T', 'method', 'formula', 'prefix', 'q', 'R_outer', 'K')},
        original_saved_price=i in old_match, acquired_sources=';'.join(price_sources[i]),
        saved_independent_confirmation=i in confirmation or i in old_confirmation,
        compatible_price_already_in_baseline=True) for i, c in sorted(candidates.items()) if i in prices])

    groups = defaultdict(list)
    mapping, screened = [], []
    reproduced = defaultdict(list)
    for e in endpoints:
        c = e['candidate']; identity = c['id']; eligible = e['status'] == 'ELIGIBLE'
        priced = identity in prices
        state = ('ELIGIBLE_PRICED' if priced else 'ELIGIBLE_UNPRICED') if eligible else e['status']
        cell = (c['system'], c['T'], e['epsilon'])
        groups[(*cell, c['method'], c['formula'])].append((identity, state, priced))
        mapping.append(dict(c, epsilon=e['epsilon'], signal_status=e['status'],
            N_cos=e['N_cos'], N_sin=e['N_sin'], price_status='COMPATIBLE_ACQUIRED' if priced else 'NOT_ACQUIRED',
            category=state, price_sources=';'.join(price_sources[identity])))
        if eligible and priced:
            cost = report.cost_for(e, prices[identity])
            reproduced[(*cell, family(c))].append((Fraction(cost['G_exact']), identity))
        if not eligible or priced:
            continue
        pf, pr = best[(*cell, 'PF')], best[(*cell, 'PR')]
        winner = min((pf, pr), key=lambda b: Fraction(b['G_exact']))
        own, opposite = (pr, pf) if family(c) == 'PR' else (pf, pr)
        g = Fraction(winner['G_exact']); nc, ns = e['N_cos'], e['N_sin']; n = nc + ns
        pred = predictions[identity]
        out = dict(c, epsilon=e['epsilon'], family=family(c), N_cos=nc, N_sin=ns, N_total=n,
            current_winner_family=family(winner), current_best_id=winner['id'],
            current_best_G=float(g), current_best_G_exact=str(g),
            current_best_PF_id=pf['id'], current_best_PF_G=pf['G'],
            current_best_PR_id=pr['id'], current_best_PR_G=pr['G'],
            current_best_PR_G_SE=pr.get('G_SE'),
            own_family_best_G=own['G'], opposite_family_best_G=opposite['G'],
            own_family_equal_axis_break_even_C=own['G'] / n,
            equal_axis_break_even_C=float(g / n), equal_axis_break_even_exact=str(g / n),
            opposite_family_equal_axis_break_even_C=opposite['G'] / n,
            C_cos_axis_intercept=float(g / nc), C_sin_axis_intercept=float(g / ns),
            C_sin_intercept=float(g / ns), C_sin_slope=-nc / ns,
            would_reverse_family_ranking=family(c) != family(winner),
            rigorous_lower_bound_C_cos=0, rigorous_lower_bound_C_sin=0,
            excluded_by_rigorous_bound=False, work_proxy=cost_proxy(c),
            uncertainty='analogue spread is descriptive, not CI/bound; rare-event population cost unknown',
            **pred)
        if pred['C_cos_est'] is not None:
            estimate = nc * pred['C_cos_est'] + ns * pred['C_sin_est']
            out.update(estimated_G=estimate, estimated_G_over_best=estimate / float(g),
                estimated_G_over_own_family_best=estimate / own['G'],
                estimated_G_over_opposite_family_best=estimate / opposite['G'],
                analogue_min_G_over_best=(nc * pred['C_cos_analogue_min'] + ns * pred['C_sin_analogue_min']) / float(g),
                analogue_max_G_over_best=(nc * pred['C_cos_analogue_max'] + ns * pred['C_sin_analogue_max']) / float(g))
            out['assessment'] = ('PROXY_PRIORITY_ONLY' if pred['estimate_kind'].startswith('PROXY') else
                'EMPIRICAL_POINT_BELOW_BEST' if estimate < float(g) else
                'EMPIRICAL_ANALOGUE_RANGE_CROSSES_BEST' if out['analogue_min_G_over_best'] < 1 else
                'EMPIRICALLY_LESS_PROMISING_NOT_EXCLUDED')
        else:
            out['assessment'] = 'INSUFFICIENT_INFORMATION'
        out.update(r_proxies[identity])
        if 'R_secant_proxy_C_cos' in out:
            out['R_secant_proxy_G_over_best'] = (nc * out['R_secant_proxy_C_cos'] + ns * out['R_secant_proxy_C_sin']) / float(g)
        screened.append(out)
    for k, vals in reproduced.items():
        assert min(vals) == (Fraction(best[k]['G_exact']), best[k]['id'])
    assert sum(best[(*k, 'PR')]['G'] < best[(*k, 'PF')]['G'] for k in {k[:3] for k in best}) == 33
    write_csv('candidate_endpoint_coverage.csv.gz', mapping)
    write_csv('all_unpriced_screening.csv.gz', screened)
    coverage = []
    for (s, t, e, method, f), vals in sorted(groups.items()):
        counts = Counter(v[1] for v in vals)
        coverage.append(dict(system=s, T=t, epsilon=e, family=family(dict(method=method)), method=method,
            formula=f, evaluated_endpoints=len(vals), distinct_candidates_within_this_epsilon=len({x[0] for x in vals}),
            eligible_priced=counts['ELIGIBLE_PRICED'], eligible_unpriced=counts['ELIGIBLE_UNPRICED'],
            ineligible=counts['INELIGIBLE'], undetermined=counts['UNDETERMINED'],
            acquired_regardless_of_eligibility=sum(x[2] for x in vals)))
    write_csv('coverage.csv', coverage)
    cell_rows = []
    for cell in sorted({k[:3] for k in best}):
        domain = [x for x in screened if (x['system'], x['T'], x['epsilon']) == cell]
        row = dict(system=cell[0], T=cell[1], epsilon=cell[2],
            current_best_PF_G=best[(*cell, 'PF')]['G'], current_best_PR_G=best[(*cell, 'PR')]['G'],
            current_winner='PR' if best[(*cell, 'PR')]['G'] < best[(*cell, 'PF')]['G'] else 'PF',
            rigorous_exclusion_count=0, global_minimum_certified=False)
        for fam in ('PF', 'PR'):
            sub = [x for x in domain if x['family'] == fam]
            empirical = [x for x in sub if x['estimate_kind'].startswith('EMPIRICAL')]
            for label, test in [('unpriced', lambda x: True),
                ('empirical_point_below_best', lambda x: x['assessment'] == 'EMPIRICAL_POINT_BELOW_BEST'),
                ('analogue_range_below_best', lambda x: x.get('analogue_min_G_over_best', 2) < 1 and x['estimate_kind'].startswith('EMPIRICAL')),
                ('proxy_only', lambda x: x['assessment'] == 'PROXY_PRIORITY_ONLY')]:
                row[fam + '_' + label] = sum(test(x) for x in sub)
            if empirical:
                leader = min(empirical, key=lambda x: x['estimated_G_over_best'])
                row[fam + '_empirical_leader_id'] = leader['id']
                row[fam + '_minimum_estimated_G_over_best'] = leader['estimated_G_over_best']
                row[fam + '_minimum_estimated_G_over_own_family_best'] = min(x['estimated_G_over_own_family_best'] for x in empirical)
            proxies = [x for x in sub if x['assessment'] == 'PROXY_PRIORITY_ONLY']
            if proxies:
                lead = min(proxies, key=lambda x: x['estimated_G_over_best'])
                row[fam + '_proxy_leader_id'] = lead['id']
                row[fam + '_proxy_leader_G_over_best'] = lead['estimated_G_over_best']
            secants = [x for x in sub if 'R_secant_proxy_G_over_best' in x]
            if secants:
                lead = min(secants, key=lambda x: x['R_secant_proxy_G_over_best'])
                row[fam + '_R_secant_proxy_leader_id'] = lead['id']
                row[fam + '_R_secant_proxy_leader_G_over_best'] = lead['R_secant_proxy_G_over_best']
        cell_rows.append(row)
    write_csv('cell_summary.csv', cell_rows)
    # Save the complete empirical frontier; final execution recommendations are
    # independently documented and deduplicated by physical ID in REPORT.md.
    frontier = [x for x in screened if x['assessment'] in
                ('EMPIRICAL_POINT_BELOW_BEST', 'EMPIRICAL_ANALOGUE_RANGE_CROSSES_BEST')]
    write_csv('empirical_frontier.csv', sorted(frontier, key=lambda x: x['estimated_G_over_best']))
    # Scientific questions determine the size; there is no top-N cap. Keep one
    # K representative per structure in the empirical frontier, retaining every
    # K alternative in empirical_frontier.csv and the full endpoint table.
    selected = {}
    front_groups = defaultdict(list)
    for x in frontier:
        front_groups[key(x, STRUCTURE + ('T',))].append(x)
    for values in front_groups.values():
        lead = min(values, key=lambda x: (x['estimated_G_over_best'], x['K'] or 0, x['id']))
        point = lead['estimated_G_over_best'] < 1
        selected[lead['id']] = dict(priority=2 if point else 4,
            recommendation='INITIAL' if point else 'FOLLOWUP',
            reason='EMPIRICAL_POINT_IMPROVEMENT' if point else 'ANALOGUE_RANGE_NEAR_CURRENT_MINIMUM',
            representative=lead)
    # R-only scenarios that cross the current best: one K per physical
    # structure. These remain explicitly proxy-based recommendations.
    proxy_groups = defaultdict(list)
    for x in screened:
        if x.get('R_secant_proxy_G_over_best', 2) < 1:
            proxy_groups[key(x, STRUCTURE + ('T',))].append(x)
    for values in proxy_groups.values():
        lead = min(values, key=lambda x: (x['R_secant_proxy_G_over_best'], x['K'] or 0, x['id']))
        selected[lead['id']] = dict(priority=3, recommendation='INITIAL',
            reason='R_SECANT_PROXY_IMPROVEMENT_AND_UNPRICED_R_REPRESENTATIVE', representative=lead)
    # Closest family reversal and a separate R control at that very same cell;
    # H6 q1 is a secondary reversal sentinel because only q-scaled prices exist.
    explicit = {
        'd748ee85d02193720b34': (1, 'INITIAL', 'CLOSEST_PR_CHALLENGE_TO_PF; required decrease comparable to analogue MCSE'),
        '205f25b8cb63949b4b18': (1, 'INITIAL', 'SAME_REVERSAL_CELL_R8_CONTROL; R_secant_reference_within_0.4_percent_of_threshold'),
        '7626c328e694ff47a23e': (5, 'FOLLOWUP', 'H6_PF_WINNER_CHALLENGE; q1_missing_for_prefix5_R4'),
        '906ba462ca8cef95cac2': (5, 'FOLLOWUP', 'H6_SAME_CELL_R8_CONTROL; proxy_only'),
        'af293c9c7685199eeb11': (6, 'OPTIONAL_PF_SENTINEL', 'CLOSEST_UNPRICED_PF_TO_PR_WINNER; empirical_price_requires_20_percent_decrease'),
        '2d83443094660669545a': (6, 'OPTIONAL_R_SENTINEL', 'H6_m5_prefix5_R16_unpriced; only_R32_price_anchors'),
        '2df1a570a6e782f04ea6': (6, 'OPTIONAL_R_SENTINEL', 'H6_m5_prefix5_R64_unpriced; original_proxy_near_best_but_unvalidated'),
    }
    for identity, (priority, recommendation, reason) in explicit.items():
        values = [x for x in screened if x['id'] == identity]
        assert values, identity
        lead = min(values, key=lambda x: x['estimated_G_over_best'])
        selected[identity] = dict(priority=priority, recommendation=recommendation,
                                 reason=reason, representative=lead)
    recommendations = []
    for identity, info in selected.items():
        lead = info['representative']
        values = [x for x in screened if x['id'] == identity]
        affected = [x for x in values if x['assessment'] in
            ('EMPIRICAL_POINT_BELOW_BEST', 'EMPIRICAL_ANALOGUE_RANGE_CROSSES_BEST')
            or x.get('R_secant_proxy_G_over_best', 2) < 1]
        recommendations.append(dict(lead, priority=info['priority'], recommendation=info['recommendation'],
            selection_reason=info['reason'], eligible_eps=';'.join(str(x['epsilon']) for x in values),
            sensitive_eps=';'.join(str(x['epsilon']) for x in affected),
            needed_axes='cosine;sine', price_reused_across_epsilon=True,
            acquisition_authorized=False, sample_count_fixed=False))
    write_csv('promising_candidates.csv', sorted(recommendations,
              key=lambda x: (x['priority'], x['estimated_G_over_best'], x['id'])))
    for name in ('signal_endpoints.json.gz', 'extension_signal_endpoints.json.gz',
                 'candidate_precision_accounting.csv', 'accounting_summary.json',
                 'scientific_provenance.json', 'completion_provenance.json', 'independent_comparisons.csv'):
        source_paths.add(public / name)
    for name in ('h46_long_time_signal.py', 'h46_long_time_report.py', 'h46_long_time_native.py'):
        source_paths.add(ROOT / 'scripts' / name)
    eligible_ids = {e['candidate']['id'] for e in endpoints if e['status'] == 'ELIGIBLE'}
    summary = dict(base_commit=BASE, endpoint_count=len(endpoints), distinct_candidates=len(candidates),
        signal_status_endpoints=dict(Counter(e['status'] for e in endpoints)),
        eligible_distinct_candidates=len(eligible_ids), compatible_price_candidates=len(prices),
        priced_eligible_endpoints=sum(x['category'] == 'ELIGIBLE_PRICED' for x in mapping),
        priced_eligible_distinct_candidates=len(eligible_ids & set(prices)),
        unpriced_eligible_endpoints=len(screened), unpriced_eligible_distinct_candidates=len(eligible_ids - set(prices)),
        unpriced_distinct_by_method=dict(Counter(candidates[i]['method'] for i in eligible_ids - set(prices))),
        old_price_candidates_in_registered_domain=len(old_match), omitted_compatible_old_prices=0,
        current_PR_lower_cost_cells=33, total_cells=40, rigorously_excluded_candidates=0,
        empirical_frontier_endpoints=len(frontier), empirical_frontier_distinct_candidates=len({x['id'] for x in frontier}),
        recommended_distinct_candidates=len(selected),
        initial_recommended_distinct_candidates=sum(x['recommendation'] == 'INITIAL' for x in recommendations),
        estimate_kind_distinct=dict(Counter(predictions[i]['estimate_kind'] for i in eligible_ids - set(prices))),
        PF_empirical_point_below_own_family_best_endpoints=sum(x['family'] == 'PF' and x.get('estimated_G_over_own_family_best', 2) < 1 for x in screened),
        empirical_point_cells=[list(c) for c in sorted({(x['system'], x['T'], x['epsilon']) for x in frontier if x['estimated_G_over_best'] < 1})],
        empirical_range_cells=[list(c) for c in sorted({(x['system'], x['T'], x['epsilon']) for x in frontier})],
        R_secant_proxy_below_best_cells=[list(c) for c in sorted({(x['system'], x['T'], x['epsilon']) for x in screened if x.get('R_secant_proxy_G_over_best', 2) < 1})],
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)},
        new_builds=0, new_compiles=0, new_sampling=0, new_signal_or_reference_calculations=0,
        mandatory_stop=True, new_cost_acquisition_authorized=False)
    (HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_hashes'}, indent=2))


if __name__ == '__main__':
    main()

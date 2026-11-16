"""Exact saved-price accounting and sensitivity; no fit or scientific producer.

The input is already fixed by a caller: configuration, allocation, per-axis
prices, evidence tier and optional linear coefficient features. This module
does not choose candidates, infer evidence coverage or estimate an error law.
"""
from collections import defaultdict
from fractions import Fraction

from .saved_rpe_cost_catalog import enc, require


TIERS = ('DIRECT', 'SUPPORTED_MODEL_RANGE', 'EXTRAPOLATION')


def number(value):
    if isinstance(value, dict):
        return Fraction(int(value['numerator']), int(value['denominator']))
    return Fraction(value)  # binary64 inputs retain their exact representation


def analyze(point):
    """Join 2*(M+1) axes, reproduce a saved total and expose correlated shifts.

    Component ids identify shared parameters, not independent error samples.
    A direct price can have no model components: moving the fitted coefficients
    then affects only model applications, leaving direct observations fixed.
    """
    rows = point['axes']; M = point['M']
    require(type(M) is int and M >= 0, 'Invalid M')
    expected = {(m, a) for m in range(M+1) for a in ('X', 'Y')}
    require(len(rows) == len(expected) and {(r['m'], r['axis']) for r in rows} == expected,
            'Missing or duplicate price axis')
    shots = {(r['m'], a): r['N_'+a] for r in point['allocation'] for a in ('X', 'Y')}
    require(set(shots) == expected and len(point['allocation']) == M+1, 'Allocation coverage')
    total = Fraction(); tiers = defaultdict(Fraction); tier_axes = defaultdict(int)
    params = {}; rounds = defaultdict(lambda: dict(cost=Fraction(), N=0, axes=[]))
    normalized = []
    for row in sorted(rows, key=lambda r: (r['m'], r['axis'])):
        key = row['m'], row['axis']; N = shots[key]
        require(type(N) is int and N >= 0 and row['N'] == N, 'Price/allocation N differs')
        require(row['configuration_id'] == point['configuration_id'], 'Configuration differs')
        require(row.get('source') and row.get('physical_id'), 'Price source/physical identity missing')
        require(row['tier'] in TIERS, 'Unknown evidence tier')
        require(row.get('unit_price_RZ') is not None, 'Missing per-axis price')
        price = number(row['unit_price_RZ']); require(price >= 0, 'Negative price')
        if row.get('saved_N_times_price_RZ') is not None:
            require(number(row['saved_N_times_price_RZ']) == N*price, 'Saved axis total differs')
        components = row.get('components', [])
        if components:
            require(row['tier'] != 'DIRECT', 'Model shifts must not change direct observations')
            require(sum((number(c['value'])*number(c['feature']) for c in components), Fraction()) == price,
                    'Coefficient features do not reproduce unit price')
        for component in components:
            ident = component['id']; value = number(component['value'])
            if ident not in params:
                params[ident] = dict(value=value, sensitivity=Fraction(), axes=[])
            require(params[ident]['value'] == value, 'Shared coefficient has inconsistent values')
            params[ident]['sensitivity'] += N*number(component['feature'])
            params[ident]['axes'].append(dict(m=row['m'], axis=row['axis']))
        cost = N*price; total += cost
        tiers[row['tier']] += cost; tier_axes[row['tier']] += 1
        rr = rounds[row['m']]; rr['cost'] += cost; rr['N'] += N; rr['axes'].append(row['axis'])
        normalized.append(dict(row, unit_price_RZ=enc(price), cost_RZ=enc(cost),
                               d_total_per_unit_RZ=N, one_percent_price_shift_RZ=enc(cost/100)))
    require(total == number(point['saved_total_RZ']), 'Reproduced total differs from selected point')
    return dict(point_id=point['point_id'], configuration_id=point['configuration_id'],
        total_RZ=enc(total), axes=normalized,
        evidence_totals={t:dict(axes=tier_axes[t], cost_RZ=enc(tiers[t]),
                               fraction=enc(tiers[t]/total)) for t in TIERS},
        rounds=[dict(m=m, cost_RZ=enc(r['cost']), fraction=enc(r['cost']/total),
                     d_total_per_same_shift_on_both_axes=r['N'],
                     one_percent_both_axis_price_shift_RZ=enc(r['cost']/100))
                for m, r in sorted(rounds.items())],
        shared_coefficients={k:dict(value=enc(v['value']), sensitivity=enc(v['sensitivity']),
            cost_component_RZ=enc(v['value']*v['sensitivity']), affected_axes=v['axes'])
            for k,v in params.items()},
        uncertainty_bound=None, CI=None,
        qualification='SAVED_PRICE_MODEL_SENSITIVITY_NOT_EXPECTATION_OR_RANK_GUARANTEE')


def break_even(lower, higher):
    """Fixed-shot counterfactuals; direct prices stay fixed in the main scenarios."""
    low, high = number(lower['total_RZ']), number(higher['total_RZ']); gap = high-low
    require(gap > 0, 'Expected a strictly positive fixed-point gap')
    lmodel = sum((number(v['cost_RZ']) for t,v in lower['evidence_totals'].items() if t != 'DIRECT'), Fraction())
    hmodel = sum((number(v['cost_RZ']) for t,v in higher['evidence_totals'].items() if t != 'DIRECT'), Fraction())
    lextra = number(lower['evidence_totals']['EXTRAPOLATION']['cost_RZ'])
    require(lmodel > 0 and hmodel > 0 and lextra > 0, 'Model exposure needed')
    return dict(gap_RZ=enc(gap),
        equation='Delta_higher_cost - Delta_lower_cost = -gap',
        direct_prices_fixed=dict(lower_model_RZ=enc(lmodel), higher_model_RZ=enc(hmodel),
            lower_only_model_price_increase=enc(gap/lmodel),
            higher_only_model_price_decrease=enc(gap/hmodel),
            equal_adverse_relative_changes=enc(gap/(lmodel+hmodel))),
        lower_extrapolation_only=dict(cost_RZ=enc(lextra),price_increase=enc(gap/lextra),
            equal_adverse_with_higher_model=enc(gap/(lextra+hmodel))),
        all_prices_algebra_only=dict(lower_increase=enc(gap/low),higher_decrease=enc(gap/high),
            equal_adverse=enc(gap/(low+high)),
            caveat='Moving observed deterministic prices conflicts with their fixed receipts'),
        observed_error_bound=None, rank_probability=None, allocation_reoptimization=False)

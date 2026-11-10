"""Passive supplied-input cost bridge; saved-input preparation never sums N*C.

Historical D1/connection SOURCE remains unchanged. Real-data calls to
account_subset require the separately approved numerical scope.
"""
from copy import deepcopy
import math

from . import h4_df10_rpe_connection as connection

AXIS_MAP = {'X': 'cosine', 'Y': 'sine'}
OBSERVABLES = {'X': 'Re(z_raw)', 'Y': 'Im(z_raw)'}
UNITS = {g: ('gate_count' if g.endswith('_count') else
             'instruction_count' if g == 'circuit_size' else 'circuit_layers')
         for g in connection.existing.METRICS}


def validate_inputs(plans, prices, *, selected_rounds, expected_rounds, compiler,
                    metrics=('rz_count',)):
    """Identity/coverage checks only: no shot solving, resource products or B math.

    A finite-cohort price is an explicit conditional proxy, not a population
    expectation. The declared law includes the ideal/sampler qualification gap.
    This checks declared bindings, not physical equivalence of two laws.
    """
    selected, expected = tuple(selected_rounds), tuple(expected_rounds)
    if (not expected or expected != tuple(range(len(expected))) or not selected
            or selected != tuple(sorted(set(selected)))
            or any(type(m) is not int or m not in expected for m in selected)):
        raise ValueError('Explicit consecutive full horizon and selected subset required')
    if (not metrics or len(set(metrics)) != len(metrics)
            or any(g not in UNITS for g in metrics)):
        raise ValueError('Explicit additive metrics only; width is separate')
    ps = tuple(plans); cs = tuple(prices)
    if len({p['m'] for p in ps}) != len(ps):
        raise ValueError('Duplicate plan round')
    if {p['m'] for p in ps} != set(selected):
        return 'INCOMPLETE_SELECTED_ROUNDS'
    if len({p['target_id'] for p in ps}) != 1 or len({p['base_time'] for p in ps}) != 1:
        raise ValueError('One common target/base time required')
    common = {k: v for k, v in ps[0]['binding'].items() if k != 'T'}
    if any({k: v for k, v in p['binding'].items() if k != 'T'} != common for p in ps):
        raise ValueError('All rounds must use the same H/state/control scientific binding')
    lookup = {p['m']: p for p in ps}
    keys = [(c['m'], c['axis']) for c in cs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate axis or pooled cohorts; select one cohort per case')
    if set(keys) != {(m, a) for m in selected for a in AXIS_MAP}:
        return 'INCOMPLETE_PRICE_COVERAGE'
    for p in ps:
        if not all(p.get('binding', {}).get(k) for k in ('hamiltonian', 'state', 'snapshot', 'rank_policy')):
            return 'MISSING_SCIENTIFIC_BINDING'
        if (p['status'] != 'CONDITIONAL_ROUND_PASS'
                or any(type(p.get('N_'+a)) is not int or p['N_'+a] <= 0 for a in AXIS_MAP)):
            return 'UNQUALIFIED_SHOTS'
        if (p['q_RPE'] != 1 << p['m'] or p['q_PF'] != p['candidate']['q']
                or p['candidate_id'] != p['candidate']['id']
                or p['candidate']['T'] != math.ldexp(p['base_time'], p['m'])
                or ('T' in p['binding'] and p['binding']['T'] != p['candidate']['T'])):
            raise ValueError('Candidate, q_PF, q_RPE or time mismatch')
    for m in selected:
        cohorts = {c.get('cohort') for c in cs if c['m'] == m}
        if len(cohorts) != 1 or not next(iter(cohorts)):
            raise ValueError('Keep a single declared cohort per round and case')
    for c in cs:
        p = lookup[c['m']]; a = c['axis']
        if (c['candidate'] != p['candidate'] or c['binding'] != p['binding']
                or c['compiler'] != compiler or c['scope'] != connection.SCOPE):
            raise ValueError('Candidate/H/state/control/compiler/full-wrapper mismatch')
        if (not p.get('law_context') or c.get('law_context') != p['law_context']
                or c.get('observable') != OBSERVABLES[a]
                or c.get('measurement_axis') != AXIS_MAP[a]
                or c.get('outcome_convention') != 'P0_minus_P1'):
            raise ValueError('Law or signed observable/wrapper mismatch')
        if any(c.get('units', {}).get(g) != UNITS[g] for g in metrics):
            raise ValueError('Resource units mismatch')
        if not c.get('source'):
            return 'MISSING_PRICE_PROVENANCE'
        if c.get('classification') not in connection.PRICE_KINDS:
            raise ValueError('Explicit direct/model classification required')
        if (c.get('expectation_basis') != 'explicit_conditional_price_proxy'
                or not c.get('price_assumptions')):
            return 'UNQUALIFIED_PRICE_INTERPRETATION'
        if any(c.get('mean', {}).get(g) is None for g in metrics):
            return 'MISSING_PRICE_METRIC'
        for g in metrics:
            connection._finite(c['mean'][g], 'one-shot resource')
        if c.get('peak_qubits') is not None:
            if type(c['peak_qubits']) is not int or c['peak_qubits'] <= 0:
                raise ValueError('Positive integer width required')
        if c['classification'] == 'MODEL_PREDICTION':
            model = c.get('model', {})
            if not model.get('source') or not model.get('coverage'):
                return 'UNQUALIFIED_MODEL_CONTEXT'
            var = model.get('q_variable')
            if (var not in ('q_PF', 'q_RPE') or model.get('q') != p[var]
                    or model.get('binding') != p['binding'] or model.get('compiler') != compiler):
                raise ValueError('Model q/scientific/compiler context mismatch')
    return 'INPUTS_READY_FOR_CONDITIONAL_PROXY_ACCOUNTING'


def account_subset(plans, prices, *, selected_rounds, expected_rounds, compiler,
                   alpha_total, metrics=('rz_count',)):
    """Explicit numerical entry. Never call on real data during preparation.

    Complete schedules delegate to the existing axis accountant. Partial sums
    keep their actual round indices; missing rounds never become zero entries.
    """
    ps, cs = tuple(plans), tuple(prices)
    selected, expected = tuple(selected_rounds), tuple(expected_rounds)
    status = validate_inputs(ps, cs, selected_rounds=selected,
        expected_rounds=expected, compiler=compiler, metrics=metrics)
    unavailable = dict(status=status, subset_resources=None, full_rpe_resources=None)
    if status != 'INPUTS_READY_FOR_CONDITIONAL_PROXY_ACCOUNTING':
        return unavailable
    if not connection.failure_budget(ps, alpha_total=alpha_total)['within_budget']:
        return dict(unavailable, status='FAILURE_BUDGET_EXCEEDED')
    if selected == expected:
        old = connection.aggregate_supplied_axes(ps, cs, expected_rounds=expected,
            alpha_total=alpha_total, compiler=compiler, metrics=metrics)
        if old['totals'] is None:
            return dict(unavailable, status=old['status'])
        values, width = old['totals'], old['peak_qubits']
    else:
        lookup = {p['m']: p for p in ps}
        values = {g: math.fsum(lookup[c['m']]['N_'+c['axis']]*c['mean'][g]
                              for c in cs) for g in metrics}
        widths = [c.get('peak_qubits') for c in cs]
        width = max(widths) if all(v is not None for v in widths) else None
    return dict(status='CONDITIONAL_FULL_HORIZON_PRICE_SUM' if selected == expected
                else 'CONDITIONAL_SELECTED_ROUND_SUBTOTAL',
        covered_rounds=list(selected), missing_rounds=[m for m in expected if m not in selected],
        subset_resources=values, full_rpe_resources=values if selected == expected else None,
        peak_qubits=width, units={g: UNITS[g] for g in metrics},
        total_covariance=None, total_confidence_interval=None,
        classical_n_and_normalization_are_price_multipliers=False,
        full_rpe_energy_success=False, actual_success_qualified=False,
        population_expected_price_qualified=False)


def prepare_saved_cases(saved_result, saved_points):
    """Copy saved N/C and validate joins. Does not call account_subset or N solver."""
    cases = []
    for row in saved_result['rows']:
        point = next(p for p in saved_points['points']
                     if p['candidate']['id'] == row['candidate_id'])
        connection.inspect_saved_point(point)
        plan = deepcopy(row['ideal_numerical_pilot'])
        if plan['candidate'] != point['candidate'] or plan['binding'] != point['binding']:
            raise ValueError('Saved D1/point identity differs')
        if plan['axes']['X']['law_model'] != plan['axes']['Y']['law_model']:
            raise ValueError('Saved axis laws differ')
        plan['law_context'] = dict(ideal_shot_law=plan['axes']['X']['law_model'],
            circuit_law='deterministic' if plan['candidate']['method'] != 'PR'
                        else 'paired_unweighted_finite_RTE',
            candidate=deepcopy(plan['candidate']),
            price_sampling='deterministic' if plan['candidate']['method'] != 'PR'
                           else 'saved_PCG64_NumPy_choice_float64_CDF',
            actual_law_distance_qualified=False)
        for original in point['prices']:
            prices = []
            for a, axis in AXIS_MAP.items():
                prices.append(dict(m=plan['m'], axis=a, measurement_axis=axis,
                    observable=OBSERVABLES[a], outcome_convention='P0_minus_P1',
                    candidate=deepcopy(original['candidate']), binding=deepcopy(original['binding']),
                    law_context=deepcopy(plan['law_context']), compiler=deepcopy(original['compiler']),
                    scope=original['scope'], classification=original['classification'],
                    source=deepcopy(original['metrics_source']), cohort=original['cohort'],
                    n=original['n'], mean=deepcopy(original['mean_metrics'][axis]),
                    peak_qubits=original['peak_qubits'], units=deepcopy(UNITS),
                    expectation_basis='explicit_conditional_price_proxy',
                    price_assumptions=['Saved finite-cohort means used as conditional point prices; no population guarantee',
                        'Ideal-shot-law versus implemented sampling/native error gap remains unqualified']))
            status = validate_inputs([plan], prices, selected_rounds=[plan['m']],
                expected_rounds=range(saved_result['configuration']['horizon']['round_count_M_plus_one']),
                compiler=point['compiler'], metrics=tuple(UNITS))
            cases.append(dict(candidate_id=row['candidate_id'], cohort=original['cohort'],
                plan=deepcopy(plan), prices=prices, input_status=status,
                resource_products_evaluated=False, subset_resources=None, full_rpe_resources=None))
    return cases

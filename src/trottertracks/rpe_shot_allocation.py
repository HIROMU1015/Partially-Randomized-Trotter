"""Finite whole-shot allocation adapter for ONE unchanged V2 configuration.

prepare() is metadata-only; diagnose() explicitly runs the existing simulator.
compare() uses supplied N evidence and existing assess/cost/select only. Real
data evaluation needs its own scientific scope. No risk policy is adopted here.
"""
from copy import deepcopy

from . import pr_rpe_rmse_controller as B
from . import rpe_decoder_statistics as S

p, a = B.p, B.a


def _contract(contract):
    if not contract.get('source') or not contract.get('id') or not contract.get('claim_level'):
        raise ValueError('Explicit statistical contract ID/SOURCE/claim required')


def _menu(config, allocations):
    B._validate(config)
    ids = [v.get('allocation_id') for v in allocations]
    if not ids or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError('Nonempty unique allocation IDs required')
    for allocation in allocations:
        if not allocation.get('source') or allocation.get('shot_policy') not in (
                'NOMINAL', 'USER_SUPPLIED', 'SUFFICIENT_EXTERNALLY_QUALIFIED'):
            raise ValueError('Independent allocation SOURCE and shot policy required')
        shots = allocation['shots']
        if [v['m'] for v in shots] != list(range(config['spec']['M'] + 1)):
            raise ValueError('Complete ordered whole-round shot vector required')
        for row in shots:
            S._integer(row['m'], 'm')
            for axis in p.AXES:
                S._integer(row['N_' + axis], 'N_' + axis, 1)


def prepare(config, means, allocations, *, target, statistical_contract):
    """Reuse canonical mean validation for each exact supplied whole vector."""
    _menu(config, allocations); _contract(statistical_contract)
    return [dict(configuration_id=config['id'], allocation=deepcopy(v),
        statistical_contract=deepcopy(statistical_contract),
        prepared=S.prepare(config, means, v['shots'], target=target,
            shot_policy=dict(meaning=v['shot_policy'], source=v['source']))) for v in allocations]


def diagnose(entry, *, repetitions, seed, source, batch_decoder=None):
    """Explicit existing MC/summary path; NEVER produces qualified N evidence.

    The contract supplies simulation_policy, allow_conditional_unqualified and
    summary_options (including unresolved_loss). No loss/confidence defaults
    are selected by this adapter. The caller retains all trials and SOURCE.
    """
    contract = entry['statistical_contract']; _contract(contract)
    if not source:
        raise ValueError('Diagnostic result SOURCE required')
    prepared = entry['prepared']
    if (prepared['configuration_id'] != entry['configuration_id']
            or not p.same([v['shots'] for v in prepared['rows']], entry['allocation']['shots'])
            or not p.same(prepared['shot_policy'], dict(meaning=entry['allocation']['shot_policy'],
                                                       source=entry['allocation']['source']))):
        raise ValueError('Prepared allocation/configuration/SOURCE mismatch')
    conditional = contract['allow_conditional_unqualified']
    if type(conditional) is not bool:
        raise ValueError('Explicit conditional-scope boolean required')
    simulation = S.simulate_prepared(prepared, repetitions=repetitions, seed=seed,
        policy=contract['simulation_policy'], allow_conditional_unqualified=conditional,
        batch_decoder=batch_decoder)
    summary = None if simulation.get('trials') is None else S.summarize(simulation['trials'],
        target_energy_ha=prepared['target']['energy_ha'], **contract['summary_options'])
    return dict(configuration_id=entry['configuration_id'], allocation=deepcopy(entry['allocation']),
        source=deepcopy(source), statistical_contract=deepcopy(contract),
        diagnostic=dict(simulation=simulation, summary=summary), n_packet=None,
        sufficient_N=None, global_RMSE_bound=None, G=None)


def _packet_binding(config, allocation, evidence, contract):
    if (evidence.get('configuration_id') != config['id'] or not evidence.get('source')
            or not p.same(evidence.get('allocation'), allocation)
            or not p.same(evidence.get('statistical_contract'), contract)):
        raise ValueError('Evaluation configuration/allocation/SOURCE/contract mismatch')
    packet = evidence.get('n_packet')
    if packet is not None:
        if not p.same(packet.get('source'), evidence['source']):
            raise ValueError('N packet SOURCE differs from evaluation SOURCE')
        if not p.same((packet.get('rmse_evidence') or {}).get('statistical_contract'), contract):
            raise ValueError('N risk evidence statistical contract missing or different')
        expected = {v['m']: v for v in allocation['shots']}
        for row in packet.get('rounds', []):
            for axis in p.AXES:
                n = row.get('N_' + axis)
                if n is not None and (row['m'] not in expected
                        or not p.same(n, expected[row['m']]['N_' + axis])):
                    raise ValueError('N evidence belongs to a different whole-shot vector')
    return packet


def _price_metadata(config, prices, cohorts, bindings, extension, interval_contract):
    """Bind D's profile extension/interval sidecars to the original mean SOURCE.

    A qualified flag records an external claim; this checks its domain and
    provenance, and does not prove its scientific validity or adopt confidence.
    """
    extension = extension or {}
    extension_complete = bool(extension.get('source') and extension.get('lowering_id'))
    recorded = config['spec']['profile'].get('lowering_id')
    if extension_complete and recorded is not None and recorded != extension['lowering_id']:
        raise ValueError('Configuration and price lowering differ')
    simultaneous = bool(interval_contract and interval_contract.get('source')
        and interval_contract.get('configuration_id') == config['id']
        and interval_contract.get('simultaneous') is True
        and interval_contract.get('scope') == 'ALL_SELECTED_ROUNDS_AXES')
    issues = [] if extension_complete else ['MISSING_PRICE_PROFILE_EXTENSION']
    cells = []; bounds = []
    for row in config['rounds']:
        for axis in p.AXES:
            cohort = cohorts.get((row['candidate']['id'], axis))
            matches = [v for v in prices if v.get('axis') == axis and v.get('cohort') == cohort
                and v.get('identity', {}).get('candidate', {}).get('id') == row['candidate']['id']]
            selected = [v for v in bindings if v.get('axis') == axis and v.get('cohort') == cohort
                and v.get('identity', {}).get('candidate', {}).get('id') == row['candidate']['id']]
            reasons = []; interval_reasons = []; lower = upper = None
            price = matches[0] if len(matches) == 1 else None
            sidecar = selected[0] if len(selected) == 1 else None
            if price is not None:
                if not extension_complete:
                    reasons.append('MISSING_PRICE_PROFILE_EXTENSION')
                if sidecar is None:
                    reasons.append('MISSING_OR_AMBIGUOUS_PRICE_PROFILE_BINDING')
                elif (not sidecar.get('source')
                        or not p.same(sidecar.get('identity'), row['identity'])
                        or not p.same(sidecar.get('law_binding'), row['law_binding'])
                        or not p.same(sidecar.get('price_source'), price.get('source'))
                        or not p.same(sidecar.get('profile_extension'), extension)):
                    reasons.append('PRICE_PROFILE_LAW_OR_SOURCE_MISMATCH')
            interval = (sidecar or {}).get('mean_interval')
            if reasons or price is None or not interval:
                interval_reasons.append('MISSING_OR_UNBOUND_MEAN_INTERVAL')
            elif (not simultaneous or not p.same(interval.get('coverage'), interval_contract)
                    or interval.get('quantity') != 'FULL_WRAPPER_MEAN_COMPILED_RZ_INTERVAL'
                    or interval.get('qualification') != 'QUALIFIED' or not interval.get('source')
                    or interval.get('evidence_kind') not in ('DERIVED_PATHWISE_BOUND', 'SIMULTANEOUS_MEAN_INTERVAL')
                    or not p.same(interval.get('domain'), row['candidate'])):
                interval_reasons.append('UNQUALIFIED_INTERVAL_DOMAIN_SOURCE_OR_SIMULTANEOUS_COVERAGE')
            else:
                try:
                    lower = a.exact_scalar(interval['lower'])
                    upper = None if interval.get('upper') is None else a.exact_scalar(interval['upper'])
                    if lower < 0 or (upper is not None and upper < lower):
                        raise ValueError('Inconsistent price interval')
                except (ValueError, TypeError, KeyError):
                    lower = upper = None
                    interval_reasons.append('INVALID_OR_INCONSISTENT_PRICE_INTERVAL')
            issues.extend(reasons)
            bounds.append((lower, upper))
            cells.append(dict(m=row['m'], axis=axis, profile_reasons=reasons,
                interval_reasons=interval_reasons, binding=deepcopy(sidecar),
                lower=None if lower is None else a.exact_value(lower),
                upper=None if upper is None else a.exact_value(upper)))
    return list(dict.fromkeys(issues)), cells, bounds


def _linear_bounds(coefficients, bounds):
    # A simultaneous box needs no independence/covariance assumption. Equal N
    # cancels the SAME shared price even when its endpoints are unknown.
    lower = upper = 0
    for coefficient, (lo, hi) in zip(coefficients, bounds):
        if not coefficient:
            continue
        left, right = (lo, hi) if coefficient > 0 else (hi, lo)
        lower = None if lower is None or left is None else lower + coefficient * left
        upper = None if upper is None or right is None else upper + coefficient * right
    return [None if v is None else a.exact_value(v) for v in (lower, upper)]


def compare(config, allocations, evaluations, prices, *, cohorts, statistical_contract,
            price_extension=None, price_bindings=(), interval_contract=None,
            price_policy='qualified_expectation', unexplored_scope='OUTSIDE_EXPLICIT_FINITE_ALLOCATION_MENU',
            root=None, comparison_policy=None):
    """Exact finite-menu point comparison of qualified, externally supplied N.

    Missing diagnostics/N/C stay unknown. A point minimum is distinct from
    strict interval dominance. An incomplete inner menu has no eligible outer
    representative. Interval-only prices are preserved but never midpoint-filled.
    An explicit comparison_policy runs C's pinned implementation guard before
    any price metadata/assess/cost/select; non-synthetic N without one stays pending.
    """
    _menu(config, allocations); _contract(statistical_contract)
    lookup = {v['allocation']['allocation_id']: v for v in evaluations}
    if len(lookup) != len(evaluations) or not set(lookup) <= {v['allocation_id'] for v in allocations}:
        raise ValueError('Duplicate or foreign allocation evaluation')
    implementation = None
    if comparison_policy is not None:
        if root is None:
            raise ValueError('Repository root required for the SOURCE-bound implementation check')
        from . import rpe_price_implementation_guard as G
        implementation = G.check(root, config, prices, cohorts=cohorts, bindings=price_bindings,
                                 policy=comparison_policy)
    issues, price_coverage, bounds = _price_metadata(config, prices, cohorts,
        price_bindings, price_extension, interval_contract)
    if implementation is None and any(v['n_packet'].get('evidence_kind') != 'SYNTHETIC_ONLY'
                                      for v in evaluations if v.get('n_packet') is not None):
        issues.append('MISSING_PRICE_COMPARISON_IMPLEMENTATION_POLICY')
    results = []; usable = []; pending = []; kinds = set()
    vectors = {v['allocation_id']: [r['N_' + axis] for r in v['shots'] for axis in p.AXES] for v in allocations}
    for allocation in allocations:
        key = allocation['allocation_id']; evidence = lookup.get(key)
        packet = None if evidence is None else _packet_binding(config, allocation, evidence, statistical_contract)
        if packet is not None and packet.get('evidence_kind') in ('SYNTHETIC_ONLY', 'SCIENTIFIC'):
            kinds.add(packet['evidence_kind'])
            if len(kinds) > 1:
                raise ValueError('Do not compare synthetic and scientific allocation evidence')
        assessment = B.assess(config, packet, prices, cohorts=cohorts, price_policy=price_policy)
        # All extra price-context checks precede the existing arithmetic entry.
        local_issues = issues + (['NOMINAL_ALLOCATION_IS_NOT_QUALIFIED_N']
                                 if allocation['shot_policy'] == 'NOMINAL' else [])
        if local_issues:
            assessment.update(eligible=False, cost=None)
            assessment['reasons'] = list(dict.fromkeys(assessment['reasons'] + local_issues))
            if assessment['status'] != 'INFEASIBLE':
                assessment['status'] = 'UNQUALIFIED_ALLOCATION_OR_PRICE_PROFILE'
        if assessment['eligible']:
            assessment = B.cost(config, packet, prices, cohorts=cohorts, price_policy=price_policy)
            usable.append((a.exact_scalar(assessment['cost']), key))
        if assessment['status'] not in ('ELIGIBLE_SUPPLIED_RECORDS', 'INFEASIBLE'):
            pending.append(key)
        results.append(dict(allocation=deepcopy(allocation), configuration_id=config['id'],
            source=None if evidence is None else deepcopy(evidence['source']),
            n_packet=deepcopy(packet),
            diagnostic=None if evidence is None else deepcopy(evidence.get('diagnostic')),
            evaluation=assessment, G=assessment['cost'],
            G_status=('FINITE_COHORT_CONDITIONAL_SCENARIO' if price_policy == 'finite_cohort_scenario'
                else 'AVAILABLE_QUALIFIED_POINT') if assessment['cost'] is not None else 'UNKNOWN',
            G_interval=None if assessment['cost'] is None else _linear_bounds(vectors[key], bounds)))
    best = min(usable)[0] if usable else None
    ties = sorted(key for cost, key in usable if cost == best)
    point = dict(allocation_id=ties[0] if ties else None,
        cost=None if best is None else a.exact_value(best), ties=ties)
    differences = []; winners = []
    complete = not pending
    if complete and price_policy == 'qualified_expectation':
        for _, key in usable:
            dominates = True
            for _, rival in usable:
                if key == rival:
                    continue
                interval = _linear_bounds([x-y for x, y in zip(vectors[key], vectors[rival])], bounds)
                differences.append(dict(left=key, right=rival, interval=interval))
                dominates &= interval[1] is not None and a.exact_scalar(interval[1]) < 0
            if dominates:
                winners.append(key)
    representative = None
    outer = None
    if complete and ties:
        selected = next(v for v in results if v['allocation']['allocation_id'] == ties[0])
        representative = deepcopy(selected['evaluation'])
        representative.update(allocation_id=ties[0], allocation_menu_complete=True,
            allocation=deepcopy(selected['allocation']), statistical_contract=deepcopy(statistical_contract),
            unexplored_allocation_scope=deepcopy(unexplored_scope),
            price_implementation_check=deepcopy(implementation))
        outer = B.select([config], [representative], unexplored_scope=deepcopy(unexplored_scope))
    return dict(configuration_id=config['id'], statistical_contract=deepcopy(statistical_contract),
        allocations=results, price_coverage=price_coverage, price_interval_contract=deepcopy(interval_contract),
        point_minimum=point, strict_interval_winner=winners[0] if len(winners) == 1 else None,
        pairwise_shared_price_difference_intervals=differences,
        menu_complete=complete, pending_allocation_ids=pending,
        selection_scope='EXPLICIT_FINITE_ALLOCATION_MENU' if complete else 'AVAILABLE_QUALIFIED_SUBSET',
        unexplored_scope=deepcopy(unexplored_scope), outer_evaluation=representative, outer_selection=outer,
        N_evidence_kind=next(iter(kinds)) if kinds else None, price_policy=price_policy,
        price_implementation_check=implementation,
        search_ranges_scientifically_adopted=False, scientific_claim_adopted=False,
        covariance=None, missing_covariance_is_zero=False,
        normalization_is_price_multiplier=False, classical_samples_are_shot_multipliers=False)

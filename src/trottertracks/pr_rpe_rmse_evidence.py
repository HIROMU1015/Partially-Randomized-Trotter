"""Finite, supplied V2 menu -> missing-evidence report. No provider or G calls.

Configurations are supplied, never chosen/generated here. Existing assess and
select with cost=None perform qualification and comparison-context checks only.
"""
from copy import deepcopy

from . import pr_rpe_rmse_controller as B


def _source_constraints(entry, config):
    expected = {}
    for item in entry.get('source_requirements', []):
        key = item['m'], item['axis']
        if key in expected or key[0] not in range(config['spec']['M'] + 1) or key[1] not in B.p.AXES:
            raise ValueError('Duplicate or foreign round/axis source requirement')
        expected[key] = item
    return expected


def _foreign_prices(config, prices):
    """Report supplied old/foreign records without relabeling or numeric reuse."""
    rows = config['rounds']; out = []
    for record in prices:
        if any(B.p.same(record.get('identity'), r['identity']) and
               B.p.same(record.get('law_binding'), r['law_binding']) for r in rows):
            continue
        out.append(dict(candidate_id=record.get('identity', {}).get('candidate', {}).get('id'),
            axis=record.get('axis'), cohort=record.get('cohort'), source=deepcopy(record.get('source')),
            reason='FULL_V2_IDENTITY_OR_LAW_MISMATCH', reusable_as_N_C=False))
    return out


def plan(entries, *, price_policy, required_methods=B.COMPARISON_METHODS,
         formula_menu=None, unexplored_scope='OUTSIDE_USER_SUPPLIED_FINITE_MENU'):
    """Assess a finite menu without filling N/C, summing G or choosing a winner.

    entry = {configuration, N_packet?, prices?, cohorts?: [{candidate_id,axis,
    cohort}], source_requirements?: [{m,axis,N?,C?}], diagnostic_references?: []}.
    Optional N/C source constraints are exact record-source matches. Unspecified
    sources remain unspecified; mandatory physics/policy/law/SOURCE checks are
    always delegated to assess. Diagnostic references cannot supply N or C.
    """
    if not entries or not required_methods or len(set(required_methods)) != len(required_methods):
        raise ValueError('Nonempty explicit menu and unique required methods required')
    if any(v not in B.COMPARISON_METHODS for v in required_methods):
        raise ValueError('Use the four explicit comparison method names')
    configs = [v['configuration'] for v in entries]
    if any(c.get('schema') != B.SCHEMA for c in configs):
        raise ValueError('Supply complete PR_RPE_RMSE_V2 configurations; historical inventories are not schedules')
    reports = []; assessments = []
    for entry, config in zip(entries, configs):
        prices = entry.get('prices', []); n_packet = entry.get('N_packet')
        cohorts = {}
        for value in entry.get('cohorts', []):
            key = value['candidate_id'], value['axis']
            if (key in cohorts or key[1] not in B.p.AXES or
                    key[0] not in {r['candidate']['id'] for r in config['rounds']}):
                raise ValueError('Duplicate or foreign candidate/axis cohort selection')
            cohorts[key] = value['cohort']
        assessment = B.assess(config, n_packet, prices, cohorts=cohorts, price_policy=price_policy)
        assessments.append(assessment)
        required = _source_constraints(entry, config)
        cells = []; source_issues = []; s = config['spec']
        n_rows = {} if n_packet is None else {r['m']: r for r in n_packet.get('rounds', [])}
        for cell in assessment['coverage']:
            request = config['rounds'][cell['m']]; supplied = required.get((cell['m'], cell['axis']), {})
            reasons = []; components = {}
            for component in ('N', 'C'):
                current = deepcopy(cell[component]); expected_source = supplied.get(component)
                source_status = ('NOT_SPECIFIED' if expected_source is None else 'MISSING' if
                    current['source'] is None else 'MATCHED' if
                    B.p.same(expected_source, current['source']) else 'MISMATCH')
                if source_status in ('MISSING', 'MISMATCH'):
                    reason = component + '_REQUIRED_SOURCE_' + source_status
                    reasons.append(reason); source_issues.append(reason)
                current.update(required_source=deepcopy(expected_source), source_constraint=source_status)
                components[component] = current
            n_row = n_rows.get(cell['m'], {})
            if ('source_candidate_id' in n_row and
                    n_row['source_candidate_id'] != request['candidate'].get('source_candidate_id')):
                reasons.append('N_SOURCE_CANDIDATE_ID_MISMATCH')
                source_issues.append('N_SOURCE_CANDIDATE_ID_MISMATCH')
            ev = {} if n_packet is None else (n_packet.get('rmse_evidence') or {})
            cells.append(dict(configuration_id=config['id'], m=cell['m'], axis=cell['axis'],
                physical_candidate=deepcopy(request.get('physical_candidate', request['candidate'])),
                physical_candidate_origin='A_SOURCE_CANDIDATE' if 'physical_candidate' in request else 'B_CANONICAL_REQUEST',
                source_candidate_id=request['candidate'].get('source_candidate_id'),
                physical_schedule_mapping=deepcopy(request['mapping']),
                identity=deepcopy(cell['identity']), law_binding=deepcopy(request['law_binding']),
                **components, source_issues=reasons,
                normalization=dict(requirement='NONE_DETERMINISTIC' if s['method'] == 'PF' else
                    'A_SOURCE_BOUND_NORMALIZATION_WITHIN_GLOBAL_RMSE_CONTRACT',
                    independent_qualification='NOT_ASSESSED_BY_B', finite_K_status=ev.get('finite_K_status'),
                    signed_piece_kappa_status=ev.get('signed_piece_kappa_status')),
                error_contract=dict(policy=deepcopy(s['rmse_policy']), source=deepcopy(ev.get('source')),
                    status=ev.get('status', 'NOT_EVALUATED'),
                    all_rounds=ev.get('covers_state_target_alias_prior_decoder_all_rounds') is True,
                    bound_present=ev.get('rmse_bound_ha') is not None),
                reuse_conditions=['EXACT_FULL_V2_IDENTITY_AND_WHOLE_CONFIGURATION', 'SAME_LAW_AND_SOURCE',
                    'SAME_COMPILER_CONTROL_MEASUREMENT_FULL_WRAPPER_UNITS', 'MATCHED_POLICY_AND_GLOBAL_RMSE_QUALIFICATION',
                    'EXPLICIT_COHORT_AND_MODEL_DOMAIN_QUALIFICATION'],
                cost_evaluation_possible=False, G=None))
        infeasible = assessment['status'] == 'INFEASIBLE' and not any(v.startswith('N_') for v in source_issues)
        ready = assessment['eligible'] and not source_issues
        for cell in cells:
            # Even valid local cells cannot bypass the WHOLE-schedule contract.
            cell['cost_evaluation_possible'] = ready
            cell['whole_schedule_blockers'] = assessment['reasons'] + list(dict.fromkeys(source_issues))
            cell['next_work'] = []
            if not infeasible:
                for component, owner, action in (
                        ('N', 'Codex A', 'CHECK_SAVED_MEAN_REFERENCE_THEN_MISSING_N_AND_GLOBAL_RMSE_EVIDENCE'),
                        ('C', 'Codex C', 'CHECK_SAVED_FULL_WRAPPER_C_THEN_MISSING_PRICE_LAW_OR_DOMAIN_EVIDENCE')):
                    if (cell[component]['qualification'] != 'AVAILABLE' or
                            cell[component]['source_constraint'] in ('MISSING', 'MISMATCH') or
                            (component == 'N' and 'N_SOURCE_CANDIDATE_ID_MISMATCH' in cell['source_issues'])):
                        cell['next_work'].append(dict(component=component, owner=owner, action=action,
                            automatic_execution=False))
        reports.append(dict(configuration_id=config['id'], method=B.comparison_method(s),
            assessment_status=assessment['status'], reasons=assessment['reasons'] + list(dict.fromkeys(source_issues)),
            required_physics_source=deepcopy(s['binding']['source']),
            required_law_source=deepcopy(s['law']['source']),
            required_RMSE_policy_source=deepcopy(s['rmse_policy']['source']),
            M=s['M'], common_kappa=deepcopy(s['kappa']), schedule_times=[deepcopy(r['T']) for r in config['rounds']],
            N_evidence_kind=assessment['N_evidence_kind'], evidence_ready=ready,
            infeasibility_usable=infeasible, rows=cells,
            unmatched_prices=_foreign_prices(config, prices),
            diagnostic_references=deepcopy(entry.get('diagnostic_references', [])),
            diagnostics_promoted_to_N_C=False, G=None))
    # Every assessment has cost=None. Existing select validates common H/state,
    # h/prior/policy/profile, formula menus, kinds and whole-schedule identities;
    # no supplied cost is passed and no numeric minimum can be computed.
    B.select(configs, assessments, required_methods=required_methods,
             formula_menu=formula_menu, unexplored_scope=unexplored_scope)
    ready_ids = [r['configuration_id'] for r in reports if r['evidence_ready']]
    pending = [r['configuration_id'] for r in reports if not r['evidence_ready'] and not r['infeasibility_usable']]
    methods = {method: dict(present=any(r['method'] == method for r in reports),
        has_ready_evidence=any(r['method'] == method and r['evidence_ready'] for r in reports)) for method in required_methods}
    return dict(status='FINITE_MENU_EVIDENCE_PLAN_NO_G', configurations=reports,
        price_policy=price_policy, required_methods=list(required_methods), formula_menu=deepcopy(formula_menu),
        method_coverage=methods, evidence_menu_resolved=not pending,
        requested_methods_have_ready_evidence=all(v['has_ready_evidence'] for v in methods.values()),
        ready_configuration_ids=ready_ids, pending_configuration_ids=pending,
        evidence_scope='EXPLICIT_FINITE_MENU' if not pending else 'AVAILABLE_QUALIFIED_SUBSET',
        cost_selection_scope='NOT_EVALUATED', unexplored_scope=deepcopy(unexplored_scope),
        comparison_complete=False, G=None, winner=None, covariance=None, total_CI=None,
        scientific_conditions_adopted_by_planner=False, scientific_execution_performed=False,
        automatic_science_calls=0, mandatory_stop=True)

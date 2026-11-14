"""Supplied-record RMSE controller. No signal/N/C providers or scientific runners.

Schedule construction is metadata. assess() never sums G. cost()/select() are
explicit numerical entries; real-data use requires a separate scientific scope.
The draft paper mapper does not establish finite-K or signed-piece guarantees.
"""
from copy import deepcopy
import hashlib
import json

from . import pf_pr_rpe_pipeline as p

a = p.accounting
SCHEMA = 'PR_RPE_RMSE_V2'
METHODS = ('PF', 'PR', 'FULL_H_RTE')
COMPARISON_METHODS = ('FULL_PF', 'TRUNCATED_PF', 'PR', 'FULL_H_RTE')
DRAFT_MAPPER = 'PAPER_TOTAL_UNIFORM_OUTER_DRAFT'
EXPLICIT_LAYOUT = 'FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1'


def _explicit_preparation(s):
    # Reuse D's source-bound physical descriptor validator; no preparation runs.
    from .full_baseline_v2_requests import check_full_h_preparation
    prep = s['split'].get('preparation') or {}
    check_full_h_preparation(prep, s['binding'])
    if (not p.same(s['split']['source'], prep['source'])
            or not p.same(s['split'].get('lambda_R'), prep['lambda_R'])
            or not p.same(s['split'].get('lambda_source'), prep['lambda_source'])
            or not p.same(s['law'].get('preparation_source'), prep['source'])
            or not s['law'].get('sampling')):
        raise ValueError('Explicit Full-H preparation/lambda/law SOURCE mismatch')


def _explicit_layout(mapping, s):
    layout = mapping.get('physical_layout') or {}
    q, outer = layout.get('q'), layout.get('R_outer')
    if (type(q) is not int or q < 1 or type(outer) is not int or outer < 1
            or layout.get('formula') != '2nd' or layout.get('pieces_per_outer') != 1
            or layout.get('allocation') != [outer] or layout.get('allocation_policy') != 'ABS_TIME_PER_OUTER'
            or layout.get('event_positions') != q * outer or mapping['event_count'] != q * outer
            or layout.get('taylor_K') != s['finite_K'] or layout.get('signed_times') != 'ALL_POSITIVE_S2'
            or layout.get('event_tau') != 'lambda_R*(T/q)/R_outer'
            or layout.get('scalar') != 'ONCE_AT_TOTAL_T' or layout.get('deterministic_blocks') != 0
            or layout.get('priced_equivalence_claim') is not False or not layout.get('compose_source')):
        raise ValueError('Explicit Full-H q/R layout, K, scalar or price-equivalence mismatch')


def _id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _positive(value):
    value = a.exact_scalar(value)
    if value <= 0:
        raise ValueError('Positive finite/exact scalar required')
    return value


def _ceil(value):
    return -(-value.numerator // value.denominator)


def comparison_method(spec):
    """PF remains the provider arm; its retained-H scope is a separate identity."""
    if spec['method'] != 'PF':
        return spec['method']
    return 'FULL_PF' if spec['split'].get('deterministic_scope') == 'FULL_H' else 'TRUNCATED_PF'


def _coverage(record, has_value, reasons):
    """Separate record existence, value presence and qualification; no arithmetic."""
    declared = (record or {}).get('price_status')
    status = ('OUTSIDE_DOMAIN' if declared == 'OUTSIDE_DOMAIN' else
              'MISSING' if record is None else
              'NOT_EVALUATED' if not has_value and record.get('status') == 'NOT_EVALUATED' else
              'UNQUALIFIED' if reasons else 'AVAILABLE')
    return dict(availability='MISSING' if record is None else 'AVAILABLE',
        value_status='AVAILABLE' if has_value else 'MISSING', qualification=status,
        reasons=list(dict.fromkeys(reasons)), source=deepcopy((record or {}).get('source')))


def paper_event_mapper(spec, row):
    """Shared A/B integer proposal, not a finite-K normalization/RMSE theorem."""
    target = _ceil(_positive(spec['kappa']['value']) *
                   (_positive(spec['split']['lambda_R']) * _positive(row['T'])) ** 2)
    outer = None if spec['method'] == 'FULL_H_RTE' else max(row['piece_count'], _ceil(a.exact_scalar(target) / row['q_PF']))
    return dict(kind=DRAFT_MAPPER, source=deepcopy(spec['event_mapping']['source']),
                kappa=deepcopy(spec['kappa']), R_outer=outer,
                event_count=target if outer is None else row['q_PF'] * outer,
                paper_total_target=target, finite_K_signed_guarantee=False)


def generate(spec, *, event_mapper=None):
    """Generate one WHOLE schedule for (L_D, delta_pf, M, common kappa)."""
    if spec.get('schedule_adapter') == 'A_FIXED_GRID_510FBACD_V1':
        if event_mapper is not None:
            raise ValueError('A fixed-grid schedule uses the pinned A mapper only')
        from .pr_rpe_rmse_integration import generate_from_a_spec
        return generate_from_a_spec(spec)
    if spec.get('schedule_adapter') == 'A_FIXED_GRID_510FBACD_TOTAL_DECODER_VIEW_V1':
        if event_mapper is not None:
            raise ValueError('Decoder-only fixed grid does not accept an external mapper')
        from .rpe_total_binding import generate_fixed_grid_view
        return generate_fixed_grid_view(spec)
    if spec.get('schedule_adapter') is not None:
        raise ValueError('Unsupported schedule adapter; no implicit generic-grid conversion')
    s = deepcopy(spec)
    json.dumps(s, allow_nan=False)
    if s['method'] not in METHODS:
        raise ValueError('Use PF/PR/FULL_H_RTE; DF B3 is not full-H pure RTE')
    if type(s['M']) is not int or s['M'] < 0 or type(s['L_D']) is not int or s['L_D'] < 0:
        raise ValueError('Nonnegative integer M and L_D required')
    for key in ('hamiltonian', 'state', 'snapshot', 'decomposition', 'source', 'system'):
        if not s['binding'].get(key):
            raise ValueError('Explicit H/state/snapshot/decomposition/SOURCE required')
    if (s['split'].get('L_D') != s['L_D'] or not s['split'].get('source')
            or not s['law'].get('source') or not s['event_mapping'].get('source')):
        raise ValueError('Source-bound split/law/event mapping required')
    profile = s['profile']
    if (not profile.get('compiler') or profile.get('scope') != p.SCOPE
            or profile.get('wrapper') != 'full_Hadamard_with_measurement'
            or profile.get('P') != 0 or not profile.get('control')):
        raise ValueError('Explicit full measurement wrapper/compiler/control/P0 required')
    policy = s['rmse_policy']
    if not all(policy.get(k) for k in ('family', 'id', 'source', 'error_contract', 'scientific_status')):
        raise ValueError('Explicit policy definition/contract/source required')
    s['h'] = a.exact_value(_positive(s['h']))
    policy['target_rmse_ha'] = a.exact_value(_positive(policy['target_rmse_ha']))
    method = s['method']
    explicit_layout = s['event_mapping']['kind'] == EXPLICIT_LAYOUT
    if explicit_layout and (method != 'FULL_H_RTE' or s['kappa'] is not None):
        raise ValueError('Explicit layout mapping requires Full-H RTE with kappa=None')
    if method == 'FULL_H_RTE':
        if (s['L_D'] != 0 or s['delta_pf'] is not None or s['formula'] is not None
                or s['split'].get('randomized_scope') != 'FULL_H_INCLUDING_ONE_BODY'
                or s['split'].get('one_body_policy') != 'RANDOMIZED'
                or s['split'].get('deterministic_scope') not in (None, 'NONE')
                or s['split'].get('original_PF_method') == 'B3'):
            raise ValueError('Full-H pure RTE must randomize one-body; no PF steps/B3 alias')
    else:
        s['delta_pf'] = a.exact_value(_positive(s['delta_pf']))
        if s['formula'] not in ('2nd', '4th(m5_best)') and not (method == 'PF' and s['formula'] == '4th'):
            raise ValueError('Explicit supported fixed formula required; no formula search')
    if method == 'PF':
        if (s['kappa'] is not None or s['finite_K'] is not None
                or s['event_mapping']['kind'] != 'NONE'
                or s['split'].get('deterministic_scope') not in ('FULL_H', 'TRUNCATED_H')
                or s['split'].get('randomized_scope') != 'NONE'
                or s['law'].get('model') != 'deterministic'):
            raise ValueError('PF requires explicit full/truncated retained H without kappa/randomization')
        if s['split']['deterministic_scope'] == 'TRUNCATED_H':
            retained = s['split'].get('retained_hamiltonian') or {}
            if (not retained.get('source') or not retained.get('construction')
                    or retained['construction'] in ('FULL', 'FULL_H')
                    or retained.get('prefix') != s['L_D']
                    or not p.same(retained.get('target_hamiltonian'), s['binding']['hamiltonian'])):
                raise ValueError('Truncated PF requires a sourced retained-H descriptor and original target H')
        elif s['split'].get('retained_hamiltonian') is not None:
            retained = s['split']['retained_hamiltonian']
            if (retained.get('construction') not in ('FULL', 'FULL_H')
                    or not p.same(retained.get('target_hamiltonian'), s['binding']['hamiltonian'])):
                raise ValueError('Full PF cannot carry a truncated/different retained Hamiltonian')
    else:
        if method == 'PR' and (s['L_D'] == 0 or s['split'].get('randomized_scope') != 'TAIL_AFTER_PREFIX'):
            raise ValueError('Explicit PR split required; full-H RTE is a different method')
        if type(s['finite_K']) is not int or s['finite_K'] < 0 or s['finite_K'] % 2:
            raise ValueError('Current finite-K implementation requires an even nonnegative K')
        if explicit_layout:
            _explicit_preparation(s)
        else:
            if not s['kappa'] or not s['kappa'].get('kind') or not s['kappa'].get('definition_source'):
                raise ValueError('Named/source-bound common kappa definition required')
            s['kappa']['value'] = a.exact_value(_positive(s['kappa']['value']))
        if s['law'].get('model') != 'ideal_finite_RTE_law':
            raise ValueError('Explicit finite-RTE target law required')
        if s['event_mapping']['kind'] == DRAFT_MAPPER:
            if s['kappa']['kind'] != 'PAPER_TOTAL_ROTATIONS' or not s['split'].get('lambda_source'):
                raise ValueError('Paper kappa and supplied lambda SOURCE required')
            if event_mapper is not None and event_mapper is not paper_event_mapper:
                raise ValueError('Do not replace the named shared paper mapper')
            event_mapper = paper_event_mapper
    rounds = []
    for m in range(s['M'] + 1):
        T = _positive(s['h']) * (1 << m)
        q = None if method == 'FULL_H_RTE' else _ceil(T / _positive(s['delta_pf']))
        proto = dict(m=m, T=a.exact_value(T), q_PF=q, piece_count=1)
        if method != 'FULL_H_RTE':
            proto['piece_count'] = len(p.structure(dict(formula=s['formula'], method='FULL', q=q, R_outer=None))['signed_weights'])
        mapping = None if method == 'PF' or event_mapper is None else event_mapper(deepcopy(s), deepcopy(proto))
        if mapping is not None:
            if (mapping.get('kind') != s['event_mapping']['kind'] or not p.same(mapping.get('source'), s['event_mapping']['source'])
                    or not p.same(mapping.get('kappa'), s['kappa'])):
                raise ValueError('Mapper kind/SOURCE/common kappa mismatch')
            events = mapping['event_count']; outer = mapping['R_outer']
            if type(events) is not int or events < 1:
                raise ValueError('Positive integer event count required')
            if method == 'PR' and (type(outer) is not int or outer < proto['piece_count'] or events != q * outer):
                raise ValueError('Whole-round event count must match uniform outer allocation')
            if method == 'FULL_H_RTE' and outer is not None:
                raise ValueError('Pure RTE has no PF outer count')
            if explicit_layout:
                _explicit_layout(mapping, s)
        outer = None if mapping is None else mapping['R_outer']
        if method == 'FULL_H_RTE':
            structure = dict(formula_id='NO_PF_FULL_H_RTE', signed_weights=[1.], allocation=None if mapping is None else [mapping['event_count']],
                event_count=None if mapping is None else mapping['event_count'], scalar_policy='scalar_identity_once_at_total_T',
                allocation_policy='FULL_H_ONE_RANDOMIZED_BLOCK', law_model=s['law']['model'])
        elif method == 'PR' and mapping is None:
            structure = p.structure(dict(formula=s['formula'], method='FULL', q=q, R_outer=None))
            structure.update(allocation=None, event_count=None, law_model=s['law']['model'])
        else:
            structure = p.structure(dict(formula=s['formula'], method='PR' if method == 'PR' else 'FULL', q=q, R_outer=outer))
        rounds.append(dict(m=m, q_RPE=1 << m, q_PF=q, T=a.exact_value(T),
            actual_delta_pf=None if q is None else a.exact_value(T / q), structure=structure,
            mapping=deepcopy(mapping), resolution='GENERATED' if method == 'PF' or mapping is not None else 'NOT_EVALUATED', R_outer=outer))
    return _assemble(s, rounds)


def _assemble(s, rounds):
    """Common identity construction for native and source-bound A schedules."""
    method = s['method']; policy = s['rmse_policy']
    cid = _id(dict(schema=SCHEMA, spec=s, rounds=rounds))
    config = dict(schema=SCHEMA, id=cid, spec=s, rounds=rounds, scientific_definition_adopted=False)
    task = dict(binding=s['binding'], profile=s['profile'], configuration=dict(schema=SCHEMA, configuration_id=cid, spec=s),
                contract=dict(split=s['split'], rmse_policy=policy, law=s['law']))
    for row in rounds:
        row['candidate'] = dict(id='RMSE:' + _id([cid, row['m']]), method=method, system=s['binding']['system'],
            prefix=s['L_D'], formula=s['formula'], T=row['T'], q=row['q_PF'], delta=row['actual_delta_pf'],
            K=s['finite_K'], R_outer=row['R_outer'], event_count=row['structure']['event_count'])
        if method == 'PF':
            row['candidate'].update(comparison_method=comparison_method(s),
                target_hamiltonian=deepcopy(s['binding']['hamiltonian']),
                retained_hamiltonian=deepcopy(s['split'].get('retained_hamiltonian') or
                    dict(construction='FULL_H', target_hamiltonian=s['binding']['hamiltonian'], source=s['split']['source'])))
        if 'physical_candidate' in row:
            physical = row['physical_candidate']
            row['candidate'].update(prefix=physical['prefix'], K=physical['K'],
                physical_method=physical['method'], source_candidate_id=physical['id'])
        row['identity'] = p.identity(task, row)
        row['law_binding'] = dict(definition=deepcopy(row.get('round_law_definition', s['law'])), configuration_id=cid, candidate_id=row['candidate']['id'],
                                  kappa=deepcopy(s['kappa']), event_mapping=deepcopy(s['event_mapping']))
    return config


def _validate(config):
    if config['spec'].get('schedule_adapter') in (
            'A_FIXED_GRID_510FBACD_V1', 'A_FIXED_GRID_510FBACD_TOTAL_DECODER_VIEW_V1'):
        if not p.same(config, generate(config['spec'])):
            raise ValueError('A configuration/schedule/identity mutated after generation')
        return
    # An explicit external mapper's results are validated by regeneration via
    # the recorded round mappings; no callback/provider is invoked at assessment.
    def supplied(s, row):
        return deepcopy(config['rounds'][row['m']]['mapping'])
    mapper = None if config['spec']['event_mapping']['kind'] == DRAFT_MAPPER else supplied
    expected = generate(config['spec'], event_mapper=mapper if any(r['mapping'] for r in config['rounds']) else None)
    if not p.same(config, expected):
        raise ValueError('Configuration/schedule/identity mutated after generation')


def assess(config, n_packet, prices, *, cohorts, price_policy='qualified_expectation'):
    """Qualifications/availability only. No N/C calculation, G sum or selection."""
    _validate(config)
    if price_policy not in ('qualified_expectation', 'finite_cohort_scenario'):
        raise ValueError('Explicit supported price policy required')
    reasons = []; chosen = []; coverage = []; s = config['spec']; infeasible = False; unstable = False
    if any(row['resolution'] != 'GENERATED' for row in config['rounds']):
        reasons.append('UNRESOLVED_PHYSICAL_LAYOUT' if s['event_mapping']['kind'] == EXPLICIT_LAYOUT else 'UNRESOLVED_KAPPA_MAPPING')
    if s['rmse_policy']['family'] != 'PR_PAPER_RMSE':
        reasons.append('HIGH_PROBABILITY_POLICY_IS_NOT_RMSE')
    if n_packet is None:
        reasons.append('MISSING_N_PACKET')
    else:
        if n_packet.get('configuration_id') != config['id'] or not p.same(n_packet.get('policy'), s['rmse_policy']):
            reasons.append('N_CONFIGURATION_OR_POLICY_MISMATCH')
        if not n_packet.get('source') or n_packet.get('status') != 'RMSE_SCHEDULE_EVALUATED':
            reasons.append('UNQUALIFIED_N_SOURCE_OR_STATUS')
        if n_packet.get('evidence_kind') not in ('SYNTHETIC_ONLY', 'SCIENTIFIC'):
            reasons.append('UNKNOWN_N_EVIDENCE_KIND')
        if n_packet.get('evidence_kind') != 'SYNTHETIC_ONLY' and s['rmse_policy']['scientific_status'] != 'EXTERNALLY_QUALIFIED':
            reasons.append('RMSE_ERROR_CONTRACT_NOT_SCIENTIFICALLY_ADOPTED')
        if n_packet.get('evidence_kind') == 'SCIENTIFIC' and (
                any(not s['binding'].get(k) for k in ('geometry', 'basis'))
                or any(s['rmse_policy']['error_contract'].get(k) is None
                       for k in ('state_assumption', 'energy_prior_ha', 'decoder'))):
            reasons.append('MISSING_SCIENTIFIC_COMPARISON_CONTEXT')
        ev = n_packet.get('rmse_evidence') or {}
        if comparison_method(s) == 'TRUNCATED_PF' and (
                ev.get('truncation_status') != 'QUALIFIED' or ev.get('covers_truncation_error') is not True
                or not p.same(ev.get('target_hamiltonian'), s['binding']['hamiltonian'])
                or not p.same(ev.get('retained_hamiltonian'), s['split'].get('retained_hamiltonian'))):
            reasons.append('UNQUALIFIED_TRUNCATION_ERROR_AGAINST_ORIGINAL_H')
        unstable = ev.get('status') == 'NUMERICAL_UNSTABLE'
        infeasible = ev.get('status') == 'INFEASIBLE'
        if (not ev.get('source') or ev.get('status') not in ('QUALIFIED', 'INFEASIBLE')
                or not p.same(ev.get('error_contract'), s['rmse_policy']['error_contract'])
                or ev.get('covers_state_target_alias_prior_decoder_all_rounds') is not True):
            reasons.append('UNQUALIFIED_GLOBAL_RMSE_ERROR_CONTRACT')
        if infeasible:
            reasons.append('RMSE_CONSTRAINT_INFEASIBLE')
        if s['method'] != 'PF' and (ev.get('finite_K_status') != 'QUALIFIED' or ev.get('signed_piece_kappa_status') != 'QUALIFIED'):
            reasons.append('UNQUALIFIED_FINITE_K_SIGNED_KAPPA_CONDITIONS')
        try:
            bound = a.exact_scalar(ev.get('rmse_bound_ha'))
            if bound < 0:
                raise ValueError('Negative bound')
            if bound > _positive(s['rmse_policy']['target_rmse_ha']):
                infeasible = True; reasons.append('RMSE_TARGET_EXCEEDED')
        except (ValueError, TypeError):
            reasons.append('MISSING_OR_UNSTABLE_RMSE_BOUND')
    nr = {} if n_packet is None else {v['m']: v for v in n_packet.get('rounds', [])}
    if n_packet is not None and (len(nr) != len(n_packet.get('rounds', [])) or set(nr) != set(range(s['M'] + 1))):
        reasons.append('MISSING_DUPLICATE_OR_FOREIGN_N_ROUNDS')
    n_global_reasons = list(reasons)
    for row in config['rounds']:
        n_start = len(reasons)
        n = nr.get(row['m'])
        if n is None:
            reasons.append('MISSING_N_ROUND')
        elif (not p.same(n.get('identity'), row['identity']) or not p.same(n.get('law_binding'), row['law_binding'])
                or any(type(n.get('N_' + axis)) is not int or n['N_' + axis] < 1 for axis in p.AXES)):
            reasons.append('N_IDENTITY_LAW_OR_INTEGER_SHOTS_MISMATCH')
        n_reasons = n_global_reasons + reasons[n_start:]
        pair = []
        cells = []
        for axis in p.AXES:
            c_start = len(reasons)
            cohort = cohorts.get((row['candidate']['id'], axis))
            matches = [v for v in prices if v.get('axis') == axis and v.get('cohort') == cohort
                       and v.get('identity', {}).get('candidate', {}).get('id') == row['candidate']['id']]
            if cohort is None or len(matches) != 1:
                reasons.append('MISSING_OR_AMBIGUOUS_SELECTED_C')
                value = matches[0] if len(matches) == 1 else None
                cells.append(dict(m=row['m'], axis=axis, identity=deepcopy(row['identity']),
                    N=_coverage(n, n is not None and type(n.get('N_' + axis)) is int, n_reasons),
                    C=_coverage(value, False, reasons[c_start:])))
                continue
            value = matches[0]; pair.append(value)
            if value.get('price_status') in ('OUTSIDE_DOMAIN', 'FAILED', 'MISSING', 'NOT_EVALUATED',
                                            'UNQUALIFIED', 'UNQUALIFIED_MODEL', 'INFEASIBLE', 'NUMERICAL_UNSTABLE'):
                reasons.append('C_' + value['price_status'])
            if value.get('quantity', 'FULL_WRAPPER_MEAN_COMPILED_PRICE') != 'FULL_WRAPPER_MEAN_COMPILED_PRICE':
                reasons.append('C_STRUCTURAL_OR_LOCAL_QUANTITY_IS_NOT_MEAN_COMPILED_PRICE')
            if (not p.same(value.get('identity'), row['identity']) or not p.same(value.get('law_binding'), row['law_binding'])
                    or not value.get('source')):
                reasons.append('C_IDENTITY_LAW_OR_SOURCE_MISMATCH')
            expected = p.measurement(row, axis)
            if any(not p.same(value.get(k), expected[k]) for k in ('measurement_axis', 'observable', 'outcome_convention', 'units')):
                reasons.append('C_MEASUREMENT_OR_UNIT_MISMATCH')
            try:
                if a.exact_scalar(value.get('mean', {}).get('rz_count')) < 0:
                    raise ValueError('Negative price')
            except (ValueError, TypeError):
                reasons.append('MISSING_OR_UNSTABLE_C_VALUE')
                unstable |= value.get('mean', {}).get('rz_count') is not None
            classification = value.get('classification')
            if classification not in a.PRICE_KINDS:
                reasons.append('UNKNOWN_C_KIND')
            expectation = value.get('expectation')
            allowed = {'QUALIFIED_EXPECTATION'} | ({'DETERMINISTIC'} if s['method'] == 'PF' else set())
            if price_policy == 'finite_cohort_scenario':
                allowed.add('FINITE_COHORT_ONLY')
            if expectation not in allowed or value.get('forced_cohort'):
                reasons.append('UNQUALIFIED_C_EXPECTATION_OR_FORCED_COHORT')
            if (value.get('normalization_is_price_multiplier') is True
                    or value.get('classical_samples_are_shot_multipliers') is True):
                reasons.append('FORBIDDEN_PRICE_OR_SAMPLE_MULTIPLIER')
            if s['method'] != 'PF' and classification == 'DIRECT_COMPILED_DETERMINISTIC':
                reasons.append('DETERMINISTIC_C_IS_NOT_RANDOMIZED_EXPECTATION')
            law = value.get('law_context') or {}
            if law.get('N_law') != s['law']['model']:
                reasons.append('C_N_LAW_MISMATCH')
            if expectation == 'FINITE_COHORT_ONLY':
                if type(value.get('n_paths')) is not int or value['n_paths'] < 1 or not law.get('price_law'):
                    reasons.append('UNQUALIFIED_FINITE_COHORT_METADATA')
            elif law.get('price_law') != law.get('N_law') or law.get('ideal_sampler_distance_qualified') is not True:
                reasons.append('PRICE_TARGET_LAW_DISTANCE_UNQUALIFIED')
            if classification == 'MODEL_PREDICTION':
                model = value.get('model') or {}; qvar = 'event_count' if s['method'] == 'FULL_H_RTE' else 'q_PF'
                qvalue = row['structure']['event_count'] if qvar == 'event_count' else row['q_PF']
                if (model.get('qualified_for_candidate') is not True or model.get('q_variable') != qvar or model.get('q') != qvalue
                        or not all(model.get(k) for k in ('source', 'validation_source', 'coverage'))
                        or not p.same(model.get('domain'), row['candidate']) or not p.same(model.get('binding'), s['binding'])
                        or model.get('compiler') != s['profile']['compiler']):
                    reasons.append('UNQUALIFIED_MODEL_DOMAIN_SOURCE_OR_COVERAGE')
            cells.append(dict(m=row['m'], axis=axis, identity=deepcopy(row['identity']),
                N=_coverage(n, n is not None and type(n.get('N_' + axis)) is int, n_reasons),
                C=dict(_coverage(value, value.get('mean', {}).get('rz_count') is not None, reasons[c_start:]),
                    classification=classification, expectation=expectation, cohort=value.get('cohort'),
                    n_paths=value.get('n_paths'), law_context=deepcopy(law),
                    model_qualified_for_candidate=(value.get('model') or {}).get('qualified_for_candidate'),
                    uncertainty=deepcopy(value.get('uncertainty')))))
        if len(pair) == 2:
            if (any(pair[0].get(k) != pair[1].get(k) for k in ('cohort', 'n_paths', 'classification', 'expectation'))
                    or pair[0].get('law_context', {}).get('price_law') != pair[1].get('law_context', {}).get('price_law')):
                reasons.append('UNPAIRED_XY_COHORT')
                for cell in cells:
                    cell['C']['qualification'] = 'UNQUALIFIED'
                    cell['C']['reasons'].append('UNPAIRED_XY_COHORT')
            chosen.extend(pair)
        coverage.extend(cells)
        for cell in cells:
            if cell['N']['source'] is None and n_packet is not None:
                cell['N']['source'] = deepcopy(n_packet.get('source'))
    reasons = list(dict.fromkeys(reasons))
    constraint_blockers = ('N_', 'MISSING_N', 'MISSING_DUPLICATE_OR_FOREIGN_N_ROUNDS', 'UNQUALIFIED_N', 'UNQUALIFIED_A_', 'UNKNOWN_N', 'UNRESOLVED_', 'HIGH_PROBABILITY_',
                           'RMSE_ERROR_', 'UNQUALIFIED_GLOBAL_', 'UNQUALIFIED_FINITE_', 'UNQUALIFIED_TRUNCATION_', 'MISSING_SCIENTIFIC_', 'MISSING_OR_UNSTABLE_RMSE_')
    trusted_infeasible = infeasible and not any(v.startswith(constraint_blockers) for v in reasons)
    status = ('NUMERICAL_UNSTABLE' if unstable else 'INFEASIBLE' if trusted_infeasible else
              'ELIGIBLE_SUPPLIED_RECORDS' if not reasons else
              'NOT_EVALUATED' if any(v.startswith(('MISSING_', 'UNRESOLVED_')) for v in reasons) else 'UNQUALIFIED')
    if trusted_infeasible:
        for cell in coverage:
            cell['N']['qualification'] = 'INFEASIBLE'
    return dict(configuration_id=config['id'], status=status, eligible=not reasons and not unstable and not infeasible, reasons=reasons,
        price_policy=price_policy, selected_price_sources=[deepcopy(v['source']) for v in chosen if v.get('source')],
        N_evidence_kind=None if n_packet is None else n_packet.get('evidence_kind'),
        price_uncertainty=[deepcopy(v.get('uncertainty')) for v in chosen],
        coverage=coverage, comparison_method=comparison_method(s),
        cost=None, scientific_definition_adopted=False, full_RPE_qualified=False,
        normalization_is_price_multiplier=False, classical_samples_are_shot_multipliers=False)


def cost(config, n_packet, prices, *, cohorts, price_policy='qualified_expectation'):
    """Explicit supplied-value exact arithmetic. No scientific provider fallback."""
    out = assess(config, n_packet, prices, cohorts=cohorts, price_policy=price_policy)
    if not out['eligible']:
        return out
    ns = {v['m']: v for v in n_packet['rounds']}; total = 0; terms = []
    for row in config['rounds']:
        for axis in p.AXES:
            value = next(v for v in prices if v['axis'] == axis and v['cohort'] == cohorts[row['candidate']['id'], axis]
                         and p.same(v['identity'], row['identity']))
            term = ns[row['m']]['N_' + axis] * a.exact_scalar(value['mean']['rz_count'])
            total += term
            terms.append(dict(m=row['m'], axis=axis, N=ns[row['m']]['N_' + axis], cost=a.exact_value(term)))
    out.update(cost=a.exact_value(total), terms=terms, numeric_encoding='exact_rational',
        cost_scope='FINITE_COHORT_CONDITIONAL_SCENARIO' if price_policy == 'finite_cohort_scenario' else 'SUPPLIED_QUALIFIED_EXPECTATION',
        uncertainty=dict(cross_configuration_covariance=None, joint_covariance=None, total_CI=None,
                         selection_induced_uncertainty_qualified=False))
    return out


def select(configs, evaluations, *, unexplored_scope='OUTSIDE_EXPLICIT_FINITE_MENU', required_methods=COMPARISON_METHODS,
           formula_menu=None):
    """Whole-configuration minima, including inner M/kappa and outer choices."""
    if not configs or len({v['id'] for v in configs}) != len(configs):
        raise ValueError('Explicit nonempty unique configuration menu required')
    for config in configs:
        _validate(config)
    # h/formula/profile and RMSE target/contract are not implicit search variables.
    def common(c):
        s = c['spec']
        return {k: s[k] for k in ('binding', 'h', 'rmse_policy', 'profile')}
    if any(not p.same(common(v), common(configs[0])) for v in configs):
        raise ValueError('Common H/state/h/RMSE/price profile required for fair comparison')
    pf_formulas = {v['spec']['formula'] for v in configs if v['spec']['method'] != 'FULL_H_RTE'}
    if formula_menu is not None:
        if not formula_menu.get('source'):
            raise ValueError('Explicit formula menu SOURCE required; this does not adopt a scientific search range')
        if any(v['spec']['formula'] not in formula_menu.get('by_method', {}).get(comparison_method(v['spec']), [])
               for v in configs if v['spec']['method'] != 'FULL_H_RTE'):
            raise ValueError('Formula outside the explicitly supplied method menu')
    elif len(pf_formulas) > 1:
        raise ValueError('PF formula is fixed, not an implicit search variable')
    randomized = [v['spec'] for v in configs if v['spec']['method'] != 'PF']
    def fixed_randomization(s):
        kappa = s['kappa'] or {}
        return dict(finite_K=s['finite_K'], kappa_kind=kappa.get('kind'),
                    definition=kappa.get('definition_source'), event_mapping=s['event_mapping'])
    for method in ('PR', 'FULL_H_RTE'):
        for explicit in (False, True):
            family = [v for v in randomized if v['method'] == method and (v['event_mapping']['kind'] == EXPLICIT_LAYOUT) == explicit]
            if family and any(not p.same(fixed_randomization(v), fixed_randomization(family[0])) for v in family):
                raise ValueError('finite-K/kappa definition/mapper are fixed within a method, not implicit search variables')
    lookup = {v['configuration_id']: v for v in evaluations}
    if len(lookup) != len(evaluations) or not set(lookup) <= {v['id'] for v in configs}:
        raise ValueError('Duplicate or foreign evaluation')
    policies = {v['price_policy'] for v in evaluations}
    if len(policies) > 1:
        raise ValueError('One common price policy required')
    kinds = {v.get('N_evidence_kind') for v in evaluations if v.get('eligible')}
    if len(kinds) > 1:
        raise ValueError('Do not compare synthetic evidence with scientific results')
    resolved = ('ELIGIBLE_SUPPLIED_RECORDS', 'INFEASIBLE')
    pending = [c['id'] for c in configs if c['id'] not in lookup or lookup[c['id']]['status'] not in resolved
               or (lookup[c['id']]['status'] == 'ELIGIBLE_SUPPLIED_RECORDS'
                   and (lookup[c['id']].get('cost') is None or lookup[c['id']].get('eligible') is not True))]
    groups = {}; usable = []; splits = {}
    for c in configs:
        ev = lookup.get(c['id']); s = c['spec']
        group_parts = [comparison_method(s), s['formula'], s['L_D'], s['delta_pf'], s['split'].get('retained_hamiltonian')]
        if s['event_mapping']['kind'] == EXPLICIT_LAYOUT:
            group_parts.append(EXPLICIT_LAYOUT)  # Not an inner common-kappa search.
        group = json.dumps(group_parts, sort_keys=True)
        if group in splits and not p.same(splits[group], s['split']):
            raise ValueError('Split/SOURCE must stay fixed inside one M/kappa group')
        splits[group] = s['split']
        groups.setdefault(group, [])
        if ev is not None and ev.get('eligible') is True and ev.get('cost') is not None:
            if ev.get('status') != 'ELIGIBLE_SUPPLIED_RECORDS' or a.exact_scalar(ev['cost']) < 0:
                raise ValueError('Eligible nonnegative supplied cost evaluation required')
            usable.append((a.exact_scalar(ev['cost']), c['id']))
            groups[group].append((a.exact_scalar(ev['cost']), c['id']))
    def minimum(options):
        if not options:
            return dict(configuration_id=None, cost=None, ties=[])
        best = min(v[0] for v in options); ties = sorted(v[1] for v in options if v[0] == best)
        return dict(configuration_id=ties[0], cost=a.exact_value(best), ties=ties)
    def matches(c, method):
        return c['spec']['method'] == method if method in METHODS else comparison_method(c['spec']) == method
    methods = {method: dict(present=any(matches(c, method) for c in configs),
        has_eligible=any(matches(c, method) and any(cid == c['id'] for _, cid in usable) for c in configs)) for method in required_methods}
    return dict(status='SUPPLIED_WHOLE_SCHEDULE_POINT_SELECTION', winner=minimum(usable),
        inner_M_kappa_minima={key: minimum(value) for key, value in groups.items()},
        menu_complete=not pending, pending_configuration_ids=pending,
        selection_scope='EXPLICIT_FINITE_MENU' if not pending else 'AVAILABLE_QUALIFIED_SUBSET',
        method_coverage=methods, fair_method_comparison_complete=not pending and all(v['present'] and v['has_eligible'] for v in methods.values()),
        formula_menu=deepcopy(formula_menu), search_ranges_scientifically_adopted=False,
        schedule_menu=[dict(configuration_id=c['id'], method=comparison_method(c['spec']), M=c['spec']['M'],
            parameterization='EXPLICIT_LAYOUT' if c['spec']['event_mapping']['kind'] == EXPLICIT_LAYOUT else
                'DETERMINISTIC_PF' if c['spec']['method'] == 'PF' else 'COMMON_KAPPA',
            times=[deepcopy(r['T']) for r in c['rounds']]) for c in configs],
        unexplored_scope=deepcopy(unexplored_scope), analytic_kappa_is_compiled_optimum=False,
        evidence_kind=next(iter(kinds)) if kinds else None,
        cross_configuration_covariance=None, total_CI=None, strict_cost_dominance_qualified=False,
        PF_baseline_optimality=None, cross_method_advantage_qualified=False,
        scientific_optimum_adopted=False, full_RPE_qualified=False)

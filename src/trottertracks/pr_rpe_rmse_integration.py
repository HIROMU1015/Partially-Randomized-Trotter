"""Pinned A schedule/N metadata and C passive lookup -> final B controller.

No nominal-to-sufficient-N conversion; no mean-price kernel, fitting or compiler
is called by the adapters. Only synthetic tests invoke A.evaluate on mock means.
"""
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

from . import pr_rpe_rmse as A
from . import pr_rpe_rmse_controller as B

p = B.p
A_RELEASE = '510fbacd01a2d855202de4c045d8ba1972e5ff3d'
C_RELEASE = '2a882fc9d17ed5e32f5581d0f220514daad0235a'
C_SOURCE = '09489eedbbd61c7396138634a896b8d8ecc75161'
ADAPTER = 'A_FIXED_GRID_510FBACD_V1'


def _exact(value):
    return B.a.exact_value(Fraction(value))


def _schedule(inputs):
    return A.schedule(**deepcopy(inputs))


def a_spec(inputs, method='PR'):
    """Metadata only. PF has no paper-kappa dependence; PR keeps the common kappa."""
    if method not in ('PF', 'PR'):
        raise ValueError('Pinned A implements PF/PR only; FULL_H_RTE is not B3')
    args = deepcopy(inputs)
    if method == 'PF':
        args['kappa_paper'] = '1'  # unused by PF, neutral canonical schedule context
    schedule = _schedule(args); d = schedule['design']; policy = schedule['task']['configuration']['policy']
    if (not d['profile'].get('compiler') or d['profile'].get('scope') != p.SCOPE
            or d['profile'].get('wrapper') != 'full_Hadamard_with_measurement'
            or d['profile'].get('P') != 0 or not d['profile'].get('control')):
        raise ValueError('Final B requires an explicit compiler/control/full-wrapper/P0 profile')
    binding = deepcopy(d['binding'])
    binding.update(source=deepcopy(d['source']), decomposition=dict(snapshot=binding['snapshot'],
        contract=deepcopy(d['contract']), source=deepcopy(d['source'])))
    kappa = None if method == 'PF' else dict(kind=A.KAPPA_DEFINITION,
        value=_exact(args['kappa_paper']), definition_source=deepcopy(A.PAPER))
    return dict(schedule_adapter=ADAPTER, A_input=args, A_release=A_RELEASE,
        binding=binding, method=method, L_D=d['prefix'] if method == 'PR' else d['PF']['prefix'],
        split=dict(L_D=d['prefix'] if method == 'PR' else d['PF']['prefix'], source=deepcopy(d['source']),
            randomized_scope='TAIL_AFTER_PREFIX' if method == 'PR' else 'NONE',
            deterministic_scope='FULL_H' if d['PF']['method'] == 'FULL' else 'A_SOURCE_BOUND_PF_BASELINE',
            original_PF_method=d['PF']['method'],
            lambda_R=_exact(args['tail_norm']['lambda_R']) if method == 'PR' else None,
            lambda_source=deepcopy(args['tail_norm']['source']) if method == 'PR' else None),
        delta_pf=_exact(d['delta']), formula=d['formula'], h=_exact(d['base_time']), M=args['M'], kappa=kappa,
        event_mapping=dict(kind='A_SAVED_EARLY_FIXED_GRID_V1', source=dict(commit=A_RELEASE, path='src/trottertracks/pr_rpe_rmse.py'),
            pf_grid='EXACT_FIXED_DELTA_WITH_EXPLICIT_SAVED_EARLY'),
        finite_K=d['K'] if method == 'PR' else None,
        law=dict(model='ideal_finite_RTE_law' if method == 'PR' else 'deterministic', source=deepcopy(d['source'])),
        rmse_policy=dict(family=policy['family'], id=policy['policy_id'], source=deepcopy(policy['source']),
            target_rmse_ha=_exact(policy['target_rmse']), error_contract=dict(A_policy=deepcopy(policy),
                state_assumption=d['state_assumption'], energy_prior_ha=deepcopy(d['energy_prior_ha']), decoder=d['decoder']),
            scientific_status='PROPOSED_NOT_ADOPTED'), profile=deepcopy(d['profile']))


def generate_from_a_spec(spec):
    """Use A's exact grid/early rules, rather than B's generic ceil grid."""
    if spec.get('A_release') != A_RELEASE or not p.same(spec, a_spec(spec['A_input'], spec['method'])):
        raise ValueError('Source-bound A specification changed')
    schedule = _schedule(spec['A_input']); rows = []
    for request in schedule['rows']:
        original = request['identity']; c = original['candidate']
        if p.arm(c) != spec['method']:
            continue
        mapping = deepcopy(request['mapping'])
        if spec['method'] == 'PF':
            mapping['requested_kappa'] = None
        rows.append(dict(m=original['m'], q_RPE=original['q_RPE'], q_PF=original['q_PF'], T=_exact(c['T']),
            actual_delta_pf=_exact(c['delta']), structure=deepcopy(original['structure']), mapping=mapping,
            resolution='GENERATED', R_outer=c['R_outer'], physical_candidate=deepcopy(c),
            round_law_definition=dict(model=original['structure']['law_model'],
                source=deepcopy(request['law_binding']['source']), A_declaration=deepcopy(request['law_binding']))))
    return B._assemble(deepcopy(spec), rows)


def n_from_a(config, packet, *, inputs):
    """Translate schema only. A nominal values and null N/bound stay unchanged."""
    B._validate(config)
    if not p.same(config['spec'], a_spec(inputs, config['spec']['method'])):
        raise ValueError('A inputs differ from B whole schedule')
    schedule = _schedule(inputs)
    # With no means, evaluate validates identities and returns NOT_EVALUATED.
    # It does not call a signal/normalization/N numerical leaf.
    empty = A.b_handoff(schedule, A.evaluate(schedule), packet['N_packet']['source'])
    expected = next(v for v in empty['packets'] if v['spec']['method'] == config['spec']['method'])
    if packet['configuration_id'] != expected['configuration_id'] or not p.same(packet['spec'], expected['spec']):
        raise ValueError('A packet configuration/SOURCE/policy mismatch')
    original = packet['N_packet']
    if (original['configuration_id'] != packet['configuration_id'] or not p.same(original['policy'], packet['spec']['rmse_policy'])
            or not original.get('source') or len(original['rounds']) != len(config['rounds'])):
        raise ValueError('A N packet policy/source/round mismatch')
    if any(original['rmse_evidence'].get(k) != v for k, v in
           dict(supplied_rmse_bound_ha=None, schedule_error_contract_qualified=False, finite_K_signed_mapping_qualified=False).items()):
        raise ValueError('Pinned A has no qualified sufficient N or global RMSE bound')
    rounds = []
    for row, supplied, raw_expected in zip(config['rounds'], original['rounds'], expected['N_packet']['rounds']):
        if (any(not p.same(supplied.get(k), raw_expected[k]) for k in ('m', 'identity', 'law_binding', 'source_request_id'))
                or supplied.get('N_X') is not None or supplied.get('N_Y') is not None):
            raise ValueError('A row identity/law mismatch or nominal N promoted to sufficient N')
        rounds.append(dict(m=row['m'], identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
            N_X=None, N_Y=None, nominal_N_X=supplied.get('nominal_N_X'), nominal_N_Y=supplied.get('nominal_N_Y'),
            model_status=supplied.get('model_status'), source_request_id=supplied['source_request_id']))
    return dict(configuration_id=config['id'], policy=deepcopy(config['spec']['rmse_policy']), source=deepcopy(original['source']),
        status=original['status'], evidence_kind='SYNTHETIC_ONLY' if original.get('evidence_domain') == 'SYNTHETIC_ONLY' else 'SCIENTIFIC',
        rounds=rounds, rmse_evidence=dict(source=deepcopy(original['source']), status='UNQUALIFIED', rmse_bound_ha=None,
            error_contract=deepcopy(config['spec']['rmse_policy']['error_contract']),
            covers_state_target_alias_prior_decoder_all_rounds=False, finite_K_status='UNQUALIFIED', signed_piece_kappa_status='UNQUALIFIED'),
        A_original_packet=deepcopy(packet), technical_identity_verified=True, nominal_promoted_to_sufficient=False)


class MeanPriceLookup:
    """Final B identity -> C's physical lookup view -> original B law/domain.

    Physical numeric fields come directly from A, never from a rounded model.
    Record prices, source, cohort, expectation and qualification are unchanged.
    PF and H6 early/other out-of-domain rows retain C's OUTSIDE_DOMAIN response.
    """
    def __init__(self, config, records=(), *, selections=None):
        from .mean_compiled_price import MeanPriceProvider
        B._validate(config)
        if config['spec'].get('schedule_adapter') != ADAPTER:
            raise ValueError('Supply the final A-source-bound B schedule')
        self.config = deepcopy(config); self.originals = {}; views = []; converted = []
        for row in config['rounds']:
            view = dict(m=row['m'], q_RPE=row['q_RPE'], q_PF=row['q_PF'],
                candidate=deepcopy(row['physical_candidate']), structure=deepcopy(row['structure']))
            view['candidate']['id'] = row['candidate']['id']; views.append(view)
        byid = {r['candidate']['id']: r for r in config['rounds']}
        view_byid = {r['candidate']['id']: r for r in views}
        for record in records:
            cid = record.get('identity', {}).get('candidate', {}).get('id')
            if cid not in byid:
                continue  # other schedule records are never relabelled
            row = byid[cid]
            if not p.same(record['identity'], row['identity']) or not p.same(record.get('law_binding'), row['law_binding']):
                raise ValueError('C record does not match final B identity/law')
            key = cid, record['axis'], record['cohort']
            if key in self.originals:
                raise ValueError('Duplicate C identity/axis/cohort')
            self.originals[key] = deepcopy(record); copy = deepcopy(record)
            physical = view_byid[cid]['candidate']
            copy['law_binding'].update(R_outer=physical['R_outer'], K=physical['K'], q_PF=physical['q'])
            if copy.get('model'):
                if not p.same(copy['model'].get('domain'), row['candidate']):
                    raise ValueError('Foreign final B model domain')
                copy['model']['domain'] = deepcopy(physical)
            converted.append(copy)
        self.task = dict(candidates=views, binding=deepcopy(config['spec']['binding']), profile=deepcopy(config['spec']['profile']))
        def validate_task(task):
            if not p.same(task, self.task):
                raise ValueError('Foreign C lookup task')
            B._validate(self.config)
        def identity(task, row):
            return deepcopy(byid[row['candidate']['id']]['identity'])
        api = SimpleNamespace(validate_task=validate_task, same=p.same, identity=identity, arm=p.arm,
                              AXES=p.AXES, measurement=p.measurement)
        context = dict(source=dict(commit=A_RELEASE, path='src/trottertracks/pr_rpe_rmse.py'),
            schedule_id=config['id'], M=config['spec']['M'], kappa_schedule=deepcopy(config['spec']['kappa']))
        self.provider = MeanPriceProvider(converted, pipeline=api, selections=selections, schedule_context=context)
        # C 09489eed's constructor stores its module p, ignoring the supplied
        # pipeline argument. Set only this owned instance; do not patch C SOURCE.
        self.provider.pipeline = api

    def __call__(self):
        records = self.provider(self.task, self.task['candidates'])
        for record in records:
            cid = record['identity']['candidate']['id']; key = cid, record['axis'], record['cohort']
            if key in self.originals:
                original = self.originals[key]
                record['law_binding'] = deepcopy(original['law_binding'])
                record['model'] = deepcopy(original.get('model'))
                for field in ('identity', 'source', 'mean', 'classification', 'expectation', 'cohort', 'n_paths'):
                    if not p.same(record.get(field), original.get(field)):
                        raise ValueError('C passive provider changed supplied price/source/cohort')
            record['B_lookup_view_only'] = True
        return records

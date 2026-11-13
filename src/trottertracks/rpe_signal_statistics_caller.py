"""Passive saved-A signal binding and explicit-target whole-RPE diagnostics.

No signal producer, N qualifier, price evaluator or optimizer. Real-data
diagnose() calls require their own finite scientific execution scope. Imports,
bind_bundle(), accept_saved() and prepare() never run a simulation.
"""
from copy import deepcopy
import inspect

from . import rpe_shot_allocation as W

B, S, p = W.B, W.S, W.p


def bind_bundle(config, bundle, records, *, bundle_source, request, integration):
    """Add a source-bound C view; retain every original mean/target field.

    integration is C's existing rpe_optimization_pipeline module (price_context
    and signal_input). request is its metadata-only plan(). records maps m to
    the immutable raw record named by that mean's SOURCE. No gate is rerun.
    """
    B._validate(config)
    rounds = config['rounds']; expected = list(range(config['spec']['M'] + 1))
    if (not bundle_source or not bundle.get('configuration_source')
            or not bundle.get('numerical_source') or not bundle.get('raw_result_carrier')
            or bundle.get('configuration_id') != config['id']
            or not p.same(request.get('configuration'), config)
            or request.get('provider') != 'A_PR'
            or not p.same(request.get('target'), bundle['target'])
            or not p.same(request.get('signal_source'), bundle['numerical_source'])):
        raise ValueError('Bundle/configuration/request/target/SOURCE mismatch')
    means = bundle['means']; jobs = request['pending']['jobs']
    if ([v['m'] for v in means] != expected or [v['m'] for v in jobs] != expected
            or set(records) != set(expected)):
        raise ValueError('Complete ordered mean/carrier/request rounds required')
    # Reuse the original canonical identity/law/target acceptance, without shots.
    S.prepare(config, means, [], target=bundle['target'],
        shot_policy=dict(meaning='USER_SUPPLIED', source=bundle_source))
    accepted = []
    for row, mean, job in zip(rounds, means, jobs):
        m = row['m']; rec = records[m]; source = mean['source']
        context = integration.price_context(request, m)
        if (source.get('commit') != bundle['raw_result_carrier'] or not source.get('path')
                or not p.same(mean.get('implementation_source'), bundle['numerical_source'])
                or not p.same(rec.get('source'), bundle['numerical_source'])
                or rec.get('m') != m or rec.get('B_configuration_id') != config['id']
                or rec.get('B_candidate_id') != row['candidate']['id']
                or not p.same(rec.get('B_identity'), row['identity'])
                or not p.same(rec.get('B_law_binding'), row['law_binding'])
                or not p.same(job.get('B_identity'), row['identity'])
                or not p.same(job.get('B_law_binding'), row['law_binding'])
                or not p.same(rec.get('candidate'), row['physical_candidate'])
                or any(not p.same(context.get(k), rec.get('physical_context', {}).get(k)) for k in context)
                or not p.same(rec.get('physical_context'), job.get('physical_context'))
                or not rec.get('input_source') or not rec.get('reference_source')):
            raise ValueError('Physical candidate/law/carrier/numerical SOURCE mismatch at m=' + str(m))
        signal = rec['signal']
        for key in ('raw', 'corrected', 'reference', 'exact_tail'):
            if key not in mean or not p.same(mean[key], signal.get(key)):
                raise ValueError('Bundle/carrier signal mismatch: ' + key)
        for key in ('log_B', 'B', 'attenuation'):
            if key not in mean['normalization'] or not p.same(mean['normalization'][key], signal.get(key)):
                raise ValueError('Bundle/carrier normalization mismatch: ' + key)
        if (not p.same(mean['normalization'].get('source'), source)
                or not p.same(mean.get('reference_source'), rec['reference_source'])
                or not p.same(mean.get('state_evidence'), rec.get('state_evidence'))
                or not p.same((mean.get('uncertainty') or {}).get('empirical_allowance'), rec.get('allowance'))
                or not p.same((mean.get('uncertainty') or {}).get('numerical_observations'), rec.get('numerical'))
                or any(not p.same(value, signal.get(key)) for key, value in mean.get('bias', {}).items())):
            raise ValueError('Bundle/carrier uncertainty/reference/bias mismatch')
        view = deepcopy(mean)
        view['provider_binding'] = dict(physical_context=deepcopy(context),
            # C compares this historical field to mean.source: it is the raw
            # carrier reference, NOT the implementation commit.
            numerical_source=deepcopy(source), input_source=deepcopy(rec['input_source']),
            reference_source=deepcopy(rec['reference_source']),
            implementation_source=deepcopy(bundle['numerical_source']),
            carrier_qualification=deepcopy(rec.get('qualification')))
        accepted.append(view)
    packet = deepcopy(bundle)
    packet.update(source=deepcopy(bundle['numerical_source']), means=accepted,
        bundle_source=deepcopy(bundle_source),
        binding_status='SOURCE_BOUND_VIEW_NOT_PHYSICAL_QUALIFICATION')
    return packet


def accept_saved(root, *, bundle_source, integration):
    """Read pinned Git JSON; reuse C.plan, without produce_signals/MC/compile."""
    store = p.SavedArtifacts(root)
    bundle = store.read(bundle_source)
    config = store.read(bundle['configuration_source'])
    request = integration.plan(root, config, target=bundle['target'],
        signal_source=bundle['numerical_source'],
        pr_physical={r['m']: r['physical_candidate'] for r in config['rounds']})
    records = {v['m']: store.read(v['source']) for v in bundle['means']}
    packet = bind_bundle(config, bundle, records, bundle_source=bundle_source,
        request=request, integration=integration)
    return dict(request=request, signal_packet=packet,
        signal_input=integration.signal_input(packet), statistics_executed=False)


def _target(value, original):
    if (not isinstance(value, dict) or not value.get('kind')
            or not value.get('source') or not value.get('qualification_scope')
            or not p.same(value.get('binding'), original['binding'])
            or not p.same(value.get('original_target_source'), original['source'])):
        raise ValueError('Explicit evaluation_target value/meaning/SOURCE/scope and original target binding required')
    S._finite(value.get('energy_ha'), 'evaluation_target.energy_ha')


def prepare(config, signal_packet, allocations, *, evaluation_target=None,
            statistical_contract, integration):
    """Metadata only. evaluation_target is separate from A's original target."""
    _target(evaluation_target, signal_packet['target'])
    if signal_packet.get('configuration_id') != config['id'] or not signal_packet.get('source'):
        raise ValueError('Signal packet configuration/SOURCE required')
    sources = signal_packet.get('implementation_sources')
    if sources is not None:
        if (signal_packet.get('per_round_source_is_authoritative') is not True
                or set(sources) != {str(r['m']) for r in config['rounds']}
                or not all(sources.values())):
            raise ValueError('Complete authoritative per-round implementation SOURCE required')
    for mean in signal_packet['means']:
        binding = mean.get('provider_binding') or {}
        implementation = signal_packet['source'] if sources is None else sources.get(str(mean['m']))
        if (not binding.get('physical_context')
                or not p.same(binding.get('numerical_source'), mean.get('source'))
                or not p.same(mean.get('implementation_source'), implementation)
                or (sources is not None and not p.same(binding.get('implementation_source'), implementation))):
            raise ValueError('Source-bound C provider_binding required')
    entries = W.prepare(config, signal_packet['means'], allocations,
        target=signal_packet['target'], statistical_contract=statistical_contract)
    consumed = integration.signal_input(signal_packet)
    for entry in entries:
        entry.update(signal_input=deepcopy(consumed), evaluation_target=deepcopy(evaluation_target))
    return entries


def _summary_options(options, target):
    """Reject malformed summary contracts before any random draws.

    Actual trial/loss/CI arithmetic remains exclusively S.summarize's.
    """
    inspect.signature(S.summarize).bind([], target_energy_ha=target, **options)
    loss = options['unresolved_loss']; kind = loss.get('kind')
    if not loss.get('source') or kind not in ('UNDEFINED', 'BOUNDED_INTERVAL'):
        raise ValueError('Explicit unresolved-trial loss policy/SOURCE required')
    if kind == 'BOUNDED_INTERVAL':
        lo = S._finite(loss['lower'], 'loss lower'); hi = S._finite(loss['upper'], 'loss upper')
        if not 0 <= lo <= hi:
            raise ValueError('Nonnegative ordered failure loss interval required')
    confidence = options.get('confidence')
    if confidence is not None:
        S._finite(confidence, 'confidence')
        if options.get('iid_repetitions') is not True or not 0 < confidence < 1:
            raise ValueError('CI requires explicit IID repetitions/confidence')
        support = options.get('loss_support')
        if kind == 'BOUNDED_INTERVAL' and support is not None:
            upper = S._finite(support['upper'], 'loss support upper')
            if not support.get('source') or upper <= 0 or loss['upper'] > upper:
                raise ValueError('Source-bound uniform loss support required')


def diagnose(entry, *, repetitions, seed, source, batch_decoder=None, risk_options=None):
    """Explicit MC + existing summary, never a scientific N/G producer.

    No scalar-target fallback, loss, seed, confidence or acceleration default is
    adopted. Pass R3 decode_sampled_counts explicitly to select acceleration.
    Optional risk_options adds fixed-family intervals without changing the
    original diagnostic or issuing N evidence.
    """
    prepared = entry['prepared']; consumed = entry['signal_input']
    target = entry.get('evaluation_target'); _target(target, prepared['target'])
    contract = entry['statistical_contract']; W._contract(contract)
    if (not source or entry['configuration_id'] != prepared['configuration_id']
            or consumed.get('configuration_id') != prepared['configuration_id']
            or not consumed.get('source')
            or not p.same(consumed.get('target'), prepared['target'])
            or not p.same(consumed.get('means'), [r['mean'] for r in prepared['rows']])
            or not p.same([r['shots'] for r in prepared['rows']], entry['allocation']['shots'])
            or not p.same(prepared['shot_policy'], dict(meaning=entry['allocation']['shot_policy'],
                                                       source=entry['allocation']['source']))):
        raise ValueError('Prepared allocation/configuration/consumed signal/SOURCE mismatch')
    conditional = contract['allow_conditional_unqualified']
    if type(conditional) is not bool:
        raise ValueError('Explicit conditional-scope boolean required')
    S._integer(repetitions, 'repetitions', 1); S._integer(seed, 'seed')
    S.iid_policy(contract['simulation_policy'])
    _summary_options(contract['summary_options'], target['energy_ha'])
    if risk_options is not None:
        from . import rpe_risk_intervals as R
        R.validate_request(entry, repetitions=repetitions, seed=seed, source=source, **risk_options)
    simulation = S.simulate_prepared(prepared, repetitions=repetitions, seed=seed,
        policy=contract['simulation_policy'], allow_conditional_unqualified=conditional,
        batch_decoder=batch_decoder)
    summary = None if simulation.get('trials') is None else S.summarize(simulation['trials'],
        target_energy_ha=target['energy_ha'], **contract['summary_options'])
    result = dict(configuration_id=entry['configuration_id'], allocation=deepcopy(entry['allocation']),
        source=deepcopy(source), statistical_contract=deepcopy(contract),
        evaluation_target=deepcopy(target), signal_input=deepcopy(consumed),
        diagnostic=dict(simulation=simulation, summary=summary), n_packet=None,
        sufficient_N=None, global_RMSE_bound=None, G=None, winner=None)
    if risk_options is not None:
        result['risk_intervals'] = R.summarize(result, **risk_options)
    return result

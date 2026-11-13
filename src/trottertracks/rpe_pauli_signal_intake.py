"""Passive Pauli-tail intake: pinned records -> existing statistics/C input.

No probability, normalization, N, risk or price arithmetic. The assembly SOURCE
is distinct from each round's numerical implementation and raw carrier.
"""
import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import subprocess
from types import SimpleNamespace as NS

from . import rpe_signal_statistics_caller as K

B, S, p = K.B, K.S, K.p
A = '714247619afb4714fd663eb15c0470ef32bdc7af'
HANDOFF = dict(commit=A, path='docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json')
DF = '91349e703e0d09ef702ccd108745dfaf1acc29f7'
P0 = '4220f3006d0d8d413e195ed948e250578513b557'
P0DIR = 'docs/research/df_head_pauli_tail_p0_results_20261114/'
KIND = 'GLOBAL_COLLECTED_JW_PAULI_OF_SAVED_DF_TAIL'


@lru_cache(maxsize=None)
def _metadata_helpers(root):
    # Reuse A's unmodified pure descriptor/job functions, excluding all imports
    # of numerical providers. This is the same pinned-function pattern as the
    # existing B/A integration. Neither function evaluates a signal.
    def functions(path, names, scope):
        raw = subprocess.check_output(['git', 'show', A + ':' + path], cwd=root)
        tree = ast.parse(raw)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise ValueError('Pinned A metadata functions unavailable')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), A + ':' + path, 'exec'), scope)
        return NS(**{name: scope[name] for name in names})
    recipe = functions('src/trottertracks/pr_rpe_signal_provider.py', ('numerical_recipe',),
        dict(deepcopy=deepcopy, RECIPE='DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1'))
    return functions('src/trottertracks/df_head_pauli_tail_p1.py', ('dictionary_identity', 'job'),
        dict(deepcopy=deepcopy, KIND=KIND, P0=P0, P0DIR=P0DIR, provider=recipe))


def _same(actual, expected, label):
    if not p.same(actual, expected):
        raise ValueError('Pauli intake mismatch: ' + label)


def bind_packet(config, packet, records, dictionary, *, target, target_source,
                assembly_source, helpers):
    """Source-bound copy only; records are selected from the named Git carriers.

    helpers is A's pinned metadata-only dictionary_identity/job interface.
    Passing a different helper is intended only for artificial fixtures.
    """
    B._validate(config)
    _same(packet['configuration'], config, 'canonical configuration')
    s = config['spec']; descriptor = packet['dictionary_identity']
    if (s['method'] != 'PR' or descriptor.get('kind') != KIND
            or s['event_mapping']['kind'] != B.DRAFT_MAPPER
            or not assembly_source or not target_source
            or packet.get('per_round_source_is_authoritative') is not True):
        raise ValueError('Common-kappa Pauli packet and explicit SOURCE required')
    _same(s['split']['tail_dictionary'], descriptor, 'split dictionary')
    _same(s['law']['tail_dictionary'], descriptor, 'law dictionary')
    raw = json.dumps(dictionary, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    item = dict(payload=dictionary, dictionary_sha256=hashlib.sha256(raw).hexdigest(),
                source=descriptor['source'])
    _same(helpers.dictionary_identity(item), descriptor, 'dictionary/hash/bit order')
    _same(dictionary['binding'], s['binding'], 'dictionary H/state/snapshot')
    _same(dictionary['prefix'], s['L_D'], 'dictionary prefix')
    if B.a.exact_scalar(dictionary['lambda_P']) != B.a.exact_scalar(s['split']['lambda_R']):
        raise ValueError('Pauli intake mismatch: Pauli lambda')
    _same(dictionary['saved_DF_scalar'], s['split']['scalar'], 'scalar once')
    _same(s['split']['lambda_source'], dict(descriptor['source'], selector=['lambda_P']), 'lambda SOURCE')
    expected = list(range(s['M'] + 1))
    _same([r['m'] for r in packet['rounds']], expected, 'ordered all rounds')
    _same(sorted(records), expected, 'complete carrier records')
    _same(packet['evaluated_rounds'], expected, 'evaluated rounds')
    if (packet.get('failed_rounds') or packet.get('signal_coverage_complete') is not True
            or packet.get('actual_sampler_qualified') is not False
            or packet.get('state_ground_certified') is not False):
        raise ValueError('Complete means-only unqualified packet required')
    means = []; implementation_sources = {}; provenance = []
    api = NS(validate=B._validate, exact_scalar=B.a.exact_scalar)
    for canonical, row in zip(config['rounds'], packet['rounds']):
        m = row['m']; rec = records[m]; ref = row['mean_source']
        _same(row['mean_record'], rec, 'carrier payload m=' + str(m))
        for key, value in dict(candidate_id=canonical['candidate']['id'], identity=canonical['identity'],
                               law_binding=canonical['law_binding']).items():
            _same(row[key], value, key)
        job = helpers.job(config, m, rec['source'], api)
        for key in ('m', 'B_configuration_id', 'B_candidate_id', 'B_identity', 'B_law_binding', 'physical_context'):
            _same(rec[key], job[key], 'record ' + key)
        _same(rec['candidate'], job['physical_context']['physical_candidate'], 'physical candidate')
        for key in ('signed_weights', 'allocation', 'event_count'):
            _same(rec[key], canonical['structure'][key], key)
        _same(ref.get('scientific_source'), rec['source'].get('commit'), 'per-round numerical SOURCE')
        if not rec.get('input_source') or not rec.get('reference_source') or rec.get('scalar_once') is not True:
            raise ValueError('Missing input/reference SOURCE or scalar convention')
        for key in ('N_X', 'N_Y', 'C', 'G', 'global_RMSE_bound', 'scientific_PASS_FAIL'):
            _same(rec.get(key), None, 'means-only ' + key)
        for key in ('N_X', 'N_Y', 'nominal_N_X', 'nominal_N_Y', 'sufficient_N_X', 'sufficient_N_Y',
                    'global_RMSE_bound', 'C', 'G', 'scientific_PASS_FAIL'):
            _same(row.get(key), None, 'row means-only ' + key)
        signal = rec['signal']; source = {k: deepcopy(ref[k]) for k in ('commit', 'path', 'selector')}
        view = dict(m=m, identity=deepcopy(canonical['identity']), law_binding=deepcopy(canonical['law_binding']),
            source=source, implementation_source=deepcopy(rec['source']), target=deepcopy(target),
            qualification='UNQUALIFIED', source_status=rec['status'],
            normalization=dict(kind='FINITE_LOG_B_SUPPLIED_BY_A', source=deepcopy(source),
                **{k: deepcopy(signal.get(k)) for k in ('log_B', 'B', 'attenuation')}),
            uncertainty=dict(empirical_allowance=deepcopy(rec['allowance']),
                numerical_observations=deepcopy(rec['numerical']),
                numerical_failure_reasons=deepcopy(rec['numerical_failure_reasons']),
                carrier_qualification=deepcopy(rec['qualification'])),
            bias={k: deepcopy(signal.get(k)) for k in ('total_signed_bias', 'outer_PF_or_discard_bias', 'finite_RTE_bias')},
            **{k: deepcopy(signal.get(k)) for k in ('raw', 'corrected', 'reference', 'exact_tail')})
        view['provider_binding'] = dict(physical_context=deepcopy(rec['physical_context']),
            numerical_source=deepcopy(source), implementation_source=deepcopy(rec['source']),
            input_source=deepcopy(rec['input_source']), reference_source=deepcopy(rec['reference_source']),
            carrier_qualification=deepcopy(rec['qualification']))
        view.update(reference_source=deepcopy(rec['reference_source']), state_evidence=deepcopy(rec['state_evidence']))
        means.append(view); implementation_sources[str(m)] = deepcopy(rec['source'])
        provenance.append(dict(m=m, candidate_id=canonical['candidate']['id'], mean_source=deepcopy(ref),
            reused_without_reanalysis=row['reused_without_reanalysis']))
    # Reuse B's full V2 identity/law/target checks with no shot vector.
    prepared = S.prepare(config, means, [], target=target,
        shot_policy=dict(meaning='USER_SUPPLIED', source=assembly_source))
    view = dict(configuration_id=config['id'], source=deepcopy(assembly_source),
        target=deepcopy(target), target_view_source=deepcopy(target_source), means=means,
        dictionary_identity=deepcopy(descriptor), implementation_sources=implementation_sources,
        per_round_source_is_authoritative=True, provenance=provenance,
        binding_status='SOURCE_BOUND_VIEW_NOT_PHYSICAL_QUALIFICATION', statistics_executed=False,
        sufficient_N=None, global_RMSE_bound=None, G=None, winner=None)
    return dict(configuration=deepcopy(config), signal_packet=view, readiness=prepared,
        C_plan_price_bridge='UNKNOWN_PAULI_PHYSICAL_ID_ADAPTER_REQUIRED',
        probability_mapping='X=Re(raw), Y=Im(raw); EXACT_AFFINE_OF_BINARY64_RAW only on explicit scientific execution')


def accept_saved(root, *, integration):
    """Read immutable JSON only. C.signal_input is a passive input projection.

    C.plan is intentionally not called: its legacy physical-ID checker does
    not support this Pauli packet. No IDs are rewritten to bypass that check.
    """
    store = p.SavedArtifacts(root); handoff = store.read(HANDOFF)
    _same(handoff['status'], 'ALL_26_PAULI_MEANS_AVAILABLE_NOT_RPE_QUALIFIED', 'A assembly status')
    _same([v['prefix'] for v in handoff['packets']], [1, 2], 'LD1/LD2')
    helpers = _metadata_helpers(str(root)); results = []
    for packet in handoff['packets']:
        prefix = packet['prefix']; config = packet['configuration']
        target_source = dict(commit=DF,
            path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD' + str(prefix) + '.json', selector=['target'])
        target = store.read(target_source)['target']
        records = {}
        for row in packet['rounds']:
            rec = store.read(row['mean_source'])
            for key in row['mean_source']['selector']:
                rec = rec[key]
            records[row['m']] = rec
        result = bind_packet(config, packet, records, store.read(packet['dictionary_identity']['source']),
            target=target, target_source=target_source, assembly_source=HANDOFF, helpers=helpers)
        result['signal_input'] = integration.signal_input(result['signal_packet'])
        _same(result['signal_input'], {k: result['signal_packet'][k] for k in ('configuration_id', 'target', 'source', 'means')},
              'C consumed signal input')
        if hasattr(integration, 'provider_binding_check'):
            view = result['signal_packet']
            request = integration.plan(root, config, target=view['target'], signal_source=view['source'],
                                       implementation_sources=view['implementation_sources'])
            for mean in view['means']:
                integration.provider_binding_check(request, mean)
            result.update(request=request, C_plan_price_bridge='SOURCE_BOUND_PAULI_PHYSICAL_BRIDGE_METADATA_ONLY')
        results.append(result)
    return dict(status='SOURCE_BOUND_ALL_26_MEANS_NOT_RPE_QUALIFIED', results=results,
        source=deepcopy(HANDOFF), source_blobs=deepcopy(store.sources), science_calls=0,
        sufficient_N=None, global_RMSE_bound=None, G=None, winner=None, mandatory_stop=True)

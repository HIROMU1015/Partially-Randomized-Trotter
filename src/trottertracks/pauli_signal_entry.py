"""PR-Pauli signal entry: any B canonical PR-Pauli configuration -> 13 round mean packet.

Input: a B canonical configuration (B2's optimizer, a stage-1 configure run, or a saved packet's
configuration), given as a file, a Git locator, a MEAN_HANDOFF packet, a B2 NEEDS candidate or a
B spec that B ``generate`` turns into the canonical configuration.

For each round, the physical conditions that determine the finite-K mean are compared with every
saved Pauli mean record (``physical_key``):

    H/state/snapshot binding (compile profile excluded), Pauli dictionary content (sha256, kind,
    mapping, coefficient/scalar policy, event space), L_D and split scopes, head PF method,
    lambda_P, saved s_R, one-body policy, law model/event model/scalar, formula, signed weights,
    per-piece allocation and policy, T, q, actual step width, K, R_outer, event count, and the
    numerical recipe kind.

Locators, configuration/candidate IDs, kappa and the compile profile are not compared: they do
not enter the mean. Saved records are indexed only when their provider kernel files are byte-equal
to the fixed P1 kernel (e8641b84f).

    REUSE        same physical key: the saved record, unchanged when its identity is this
                 configuration's, otherwise re-bound to this configuration's canonical identity
                 with provenance (signal/numerics/SOURCE untouched);
    REQUEST      no saved match and the existing provider supports the round: a concrete job;
    UNSUPPORTED  the existing provider cannot evaluate it (reason codes).

Each round also carries the provider support class: SUPPORTED_VALIDATED_FOR_PAULI (2nd, K=2: the
LD1/LD2/LD3/LD5 runs), SUPPORTED_NOT_VALIDATED_FOR_PAULI (provider scope, never run on a Pauli
tail) or UNSUPPORTED.

Requests run on the unchanged LD3 stage-2 path (S3.load_context, checked against P1
``load_contexts`` on LD2; P1 ``job``; saved H4 reference; fixed provider ``evaluate``). The packet
is the LD1-LD5 MEAN_HANDOFF layout. No sampling, build/compile, price, N, Q3 or shot choice.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import subprocess

from . import df_head_pauli_tail_ld3_signals as S3

L, p0, p1, provider = S3.L, S3.p0, S3.p1, S3.provider
common = provider.common
MODULE = 'src/trottertracks/pauli_signal_entry.py'
KIND = 'GLOBAL_COLLECTED_JW_PAULI_OF_SAVED_DF_TAIL'
EVENT_SPACE = 'FULL_2_TO_N_NOT_PROJECTED_PAULI'
SCALAR_POLICY = 'KEEP_SAVED_sR_AND_c_ONCE'
DRAFT_MAPPER = 'PAPER_TOTAL_UNIFORM_OUTER_DRAFT'
LOCAL = S3.LOCAL
PROVIDER_FORMULAS = ('2nd', '4th(m5_best)')        # provider.evaluate scope
PROVIDER_K = (2, 4, 6)                               # provider.evaluate scope
VALIDATED = dict(formula='2nd', K=2)                 # every saved Pauli-tail run (LD1/LD2/LD3/LD5)
KERNEL_FILES = (provider.MODULE, 'src/trottertracks/pf_power_stability.py',
                'src/trottertracks/h4_four_method_screen_mean.py', 'src/trotterlib/rte.py')
HELPER = dict(commit='8c8ce542e9fc25ead709fe6b0eb71b87ba7434ca', path=S3.MODULE, function='b_intake_dictionary_identity')
SAVED_PACKETS = (
    dict(commit='714247619afb4714fd663eb15c0470ef32bdc7af',
         path='docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json'),
    dict(commit='cdbce383f9944b7f9da1cbbd7851406dbbaae88a',
         path='docs/research/df_head_pauli_tail_ld3_signals_20261116/MEAN_HANDOFF.json'),
    dict(commit='ac46d100e74f3f56b4a3ff4d95c15fdc7fe8ca98',
         path='docs/research/df_head_pauli_tail_ld5_signals_20261116/MEAN_HANDOFF.json'),
)
BINDING_NOT_PHYSICAL = ('compiler', 'profile', 'profile_is_reference_only')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_ref(root, ref):
    value = L.read(root, ref['commit'], ref['path'])
    for key in ref.get('selector', []):
        value = value[key]
    return value


# ---------------------------------------------------------------- configuration input
def load_configuration(root, ref, *, configuration_id=None):
    """(B canonical configuration, Git locator or None) from any accepted input form."""
    b = p1.b_api(root); locator = None
    if isinstance(ref, str):
        if ':' in ref and not ref.startswith('/') and len(ref.split(':', 1)[0]) >= 7:
            commit, path = ref.split(':', 1); ref = dict(commit=commit, path=path)
        else:
            with open(ref) as f:
                ref = json.load(f)
    if isinstance(ref, dict) and set(ref) <= {'commit', 'path', 'selector'} and 'commit' in ref and 'path' in ref:
        locator = {k: deepcopy(ref[k]) for k in ('commit', 'path', 'selector') if k in ref}
        locator['commit'] = subprocess.check_output(['git', 'rev-parse', '--verify', ref['commit'] + '^{commit}'],
                                                    cwd=root).decode().strip()
        value = read_ref(root, ref)
    else:
        value = deepcopy(ref)
    if 'method' in value and 'needs' in value:                        # B2 NEEDS candidate
        if value.get('configuration_id') is None:
            raise ValueError('CONFIGURATION_AND_SCHEDULE_REQUIRED_FIRST: run the stage-1 P0/configure route '
                             '(scripts/run_df_head_pauli_tail_split_stage1.py) for L_D=%s'
                             % value.get('configuration', {}).get('L_D'))
        return load_configuration(root, value['signal_source'], configuration_id=configuration_id or value['configuration_id'])
    if 'packets' in value:                                           # MEAN_HANDOFF
        hits = [i for i, pk in enumerate(value['packets']) if configuration_id in (None, pk['configuration']['id'])]
        if len(hits) != 1:
            raise ValueError('ONE_PACKET_REQUIRED: give configuration_id')
        value = value['packets'][hits[0]]['configuration']
        if locator is not None:
            locator['selector'] = list(locator.get('selector', [])) + ['packets', hits[0], 'configuration']
    elif 'configuration' in value and 'rounds' in value and 'spec' not in value:   # one packet
        value = value['configuration']
        if locator is not None:
            locator['selector'] = list(locator.get('selector', [])) + ['configuration']
    if 'spec' not in value and 'split' in value and 'L_D' in value:  # a B spec
        value = b.generate(value); locator = None
    b.validate(value)
    if configuration_id is not None and value['id'] != configuration_id:
        raise ValueError('CONFIGURATION_ID_DIFFERS')
    return value, locator


# ---------------------------------------------------------------- physical key and saved index
def physical_key(context):
    """Physical conditions of one finite-K mean, from a job's or a record's physical_context."""
    split, law, recipe, td = (context['split'], context['law'], context['numerical_recipe'],
                              context['split']['tail_dictionary'])
    value = dict(
        binding={k: v for k, v in context['binding'].items() if k not in BINDING_NOT_PHYSICAL},
        candidate={k: v for k, v in context['physical_candidate'].items() if k != 'id'},
        structure=context['structure'],
        dictionary={k: td.get(k) for k in ('kind', 'dictionary_sha256', 'prefix', 'mode_mapping', 'coefficient_policy',
                                           'scalar_policy', 'event_space')},
        split={k: split.get(k) for k in ('L_D', 'randomized_scope', 'deterministic_scope', 'original_PF_method',
                                         'lambda_R', 'scalar', 'one_body_policy')},
        law={k: law.get(k) for k in ('model', 'event_model', 'scalar', 'actual_sampler_qualified')},
        recipe={k: recipe[k] for k in ('deterministic_PF', 'finite_RTE', 'sampling')})
    return hashlib.sha256(canonical(value).encode()).hexdigest(), value


def kernel_fingerprint(root, commit):
    return {path: hashlib.sha256(p0.blob(root, commit, path)).hexdigest() for path in KERNEL_FILES}


def saved_index(root, packets=SAVED_PACKETS):
    """Saved LOCAL, gate-passing Pauli mean records of the listed handoffs, by physical key.

    The record must equal its carrier at mean_source, and its numerical SOURCE must carry the fixed
    P1 kernel files byte-for-byte.
    """
    index = {}; kernel = kernel_fingerprint(root, S3.P1_KERNEL); prints = {}
    for ref in packets:
        handoff = read_ref(root, ref)
        for pi, packet in enumerate(handoff['packets']):
            for row in packet['rounds']:
                rec, src = row.get('mean_record'), row.get('mean_source')
                if rec is None or src is None or rec.get('status') != LOCAL or rec.get('numerical', {}).get('gate_pass') is not True:
                    continue
                commit = rec['source']['commit']
                if commit not in prints:
                    prints[commit] = kernel_fingerprint(root, commit)
                if prints[commit] != kernel or read_ref(root, src) != rec or src.get('scientific_source') != commit:
                    continue
                key, _ = physical_key(rec['physical_context'])
                index.setdefault(key, dict(
                    record=rec, mean_source=deepcopy(src), handoff=deepcopy(ref), packet_index=pi, m=row['m'],
                    configuration_id=rec['B_configuration_id'], candidate_id=rec['B_candidate_id']))
    return index


# ---------------------------------------------------------------- support
def dictionary_item(root, config):
    """This configuration's P0 dictionary from its own descriptor locators (no prefix-specific code)."""
    td = config['spec']['split']['tail_dictionary']
    src, res = td['source'], td['P0_operator_residual_source']
    result = L.read(root, res['commit'], res['path'])
    prefix = config['spec']['L_D']
    if (not result['status'].endswith('_COMPLETE_NOT_P1_OR_RPE_QUALIFIED') or not result['checks']
            or not all(c['PASS'] for c in result['checks'])
            or result.get('control', dict(PASS=True)).get('PASS') is not True):
        raise ValueError('P0_RESULT_NOT_COMPLETE')
    row = next(r for r in result['prefixes'] if r['prefix'] == prefix)
    obj = L.read(root, src['commit'], src['path'])
    if (L.canonical_sha(obj) != td['dictionary_sha256'] or row['dictionary_sha256'] != td['dictionary_sha256']
            or obj['prefix'] != prefix or 'GATES_PASS' not in row['status']):
        raise ValueError('P0_DICTIONARY_DIFFERS_FROM_DESCRIPTOR')
    item = dict(payload=obj, dictionary_sha256=td['dictionary_sha256'], source=deepcopy(src),
                P0_metrics=row['identity_metrics'], P0_result_source=deepcopy(res))
    if L.dictionary_identity(item) != td:
        raise ValueError('DESCRIPTOR_NOT_REPRODUCED_FROM_ITS_DICTIONARY')
    return item


def configuration_support(root, config):
    """(reasons, dictionary item, saved P0 metadata): configuration-level provider coverage."""
    s = config['spec']; reasons = []
    td = s['split'].get('tail_dictionary') or {}
    if s.get('method') != 'PR':
        reasons.append('NOT_PR')
    if td.get('kind') != KIND or td.get('event_space') != EVENT_SPACE:
        reasons.append('NOT_PAULI_DICTIONARY')
    elif td.get('scalar_policy') != SCALAR_POLICY:
        reasons.append('SCALAR_POLICY_NOT_PROVIDER')
    if s.get('event_mapping', {}).get('kind') != DRAFT_MAPPER:
        reasons.append('EVENT_MAPPING_NOT_PAPER_MAPPER')
    if reasons:
        return reasons, None, None
    try:
        item = dictionary_item(root, config)
        located = S3.operators_locator(item)
        p0.blob(root, located['commit'], located['path'])
        saved = L.p0_metadata(root, (2, s['L_D']))
    except Exception as exc:
        return ['DICTIONARY_OPERATORS_OR_DESCRIPTOR_UNAVAILABLE: %s: %s' % (type(exc).__name__, str(exc)[:200])], None, None
    if s['binding'] != saved['binding'] or item['payload']['binding'] != saved['binding']:
        reasons.append('BINDING_NOT_SAVED_H4')
    if not (s['split'].get('scalar') == item['payload']['saved_DF_scalar']
            == saved['saved_tail_descriptors'][str(s['L_D'])]['scalar']):
        reasons.append('SCALAR_NOT_SAVED_sR')
    if Fraction(s['split']['lambda_R']['numerator'], s['split']['lambda_R']['denominator']) != Fraction(item['payload']['lambda_P']):
        reasons.append('LAMBDA_NOT_DICTIONARY_LAMBDA_P')
    if s['split'].get('lambda_source') != dict(item['source'], selector=['lambda_P']):
        reasons.append('LAMBDA_SOURCE_NOT_DICTIONARY')
    return reasons, item, saved


def round_support(root, job, *, store=None):
    """(reasons, class) with the provider.evaluate pre-call conditions and the saved H4 reference."""
    ctx = job['physical_context']; c = ctx['physical_candidate']; st = ctx['structure']; reasons = []
    if c['formula'] not in PROVIDER_FORMULAS:
        reasons.append('FORMULA_NOT_IN_PROVIDER')
    if c['K'] not in PROVIDER_K:
        reasons.append('K_NOT_IN_PROVIDER')
    if not reasons and not common.same(st, common.structure(c)):
        reasons.append('STRUCTURE_NOT_PROVIDER_STRUCTURE')
    if (c['prefix'] != ctx['split']['L_D'] or ctx['law']['model'] != 'ideal_finite_RTE_law'
            or ctx['split'].get('one_body_policy') not in (None, 'DETERMINISTIC')
            or ctx['split']['randomized_scope'] != 'TAIL_AFTER_PREFIX' or ctx['split']['deterministic_scope'] != 'PREFIX'):
        reasons.append('SPLIT_OR_LAW_NOT_PROVIDER')
    try:
        provider._reference(job, provider.saved_h4_reference(root, job, store=store))
    except Exception:
        reasons.append('NO_SAVED_REFERENCE_AT_THIS_ROUND_T')
    if reasons:
        return reasons, 'UNSUPPORTED'
    validated = c['formula'] == VALIDATED['formula'] and c['K'] == VALIDATED['K']
    return [], 'SUPPORTED_VALIDATED_FOR_PAULI' if validated else 'SUPPORTED_NOT_VALIDATED_FOR_PAULI'


def numerical(source):
    return dict(commit=source, path=provider.MODULE, adapter=MODULE, kernel_source=S3.P1_KERNEL, recipe=provider.RECIPE)


def changed_fields(left, right):
    """Dotted paths whose values differ (two levels)."""
    out = []
    for k in sorted(set(left) | set(right)):
        a, b = left.get(k), right.get(k)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            out += [k + '.' + x for x in sorted(set(a) | set(b)) if a.get(x) != b.get(x)]
        else:
            out.append(k)
    return out


# ---------------------------------------------------------------- plan
def plan(root, config, *, packets=SAVED_PACKETS, index=None, configuration_source=None, source='PLANNED'):
    """Per-round REUSE / REQUEST / UNSUPPORTED (metadata only: no array, no mean)."""
    b = p1.b_api(root); b.validate(config)
    reasons, item, saved = configuration_support(root, config)
    index = saved_index(root, packets) if index is None else index
    store = common.SavedArtifacts(root)
    by_m = {}
    for value in index.values():
        by_m.setdefault(value['m'], []).append(value)
    rows = []
    for row_c in config['rounds']:
        m = row_c['m']
        if reasons:
            rows.append(dict(m=m, decision='UNSUPPORTED', provider_support='UNSUPPORTED', reasons=list(reasons),
                             candidate_id=row_c['candidate']['id']))
            continue
        job = p1.job(config, m, numerical(source), b)
        key, value = physical_key(job['physical_context'])
        c = job['physical_context']['physical_candidate']; st = job['physical_context']['structure']
        bad, support = round_support(root, job, store=store)
        row = dict(m=m, candidate_id=job['B_candidate_id'], physical_key=key, provider_support=support, provider_reasons=bad,
                   job=dict(q_RPE=job['q_RPE'], q_PF=job['q_PF'], T=c['T'], delta=c['delta'], formula=c['formula'], K=c['K'],
                            R_outer=c['R_outer'], event_count=c['event_count'], signed_weights=st['signed_weights'],
                            allocation=st['allocation']))
        hit = index.get(key)
        if hit is not None:
            identical = hit['configuration_id'] == config['id'] and hit['candidate_id'] == job['B_candidate_id'] and hit['m'] == m
            row.update(decision='REUSE', reuse='IDENTICAL_IDENTITY' if identical else 'REBOUND_SAME_PHYSICAL_KEY',
                       saved=dict(mean_source=deepcopy(hit['mean_source']), handoff=deepcopy(hit['handoff']),
                                  configuration_id=hit['configuration_id'], candidate_id=hit['candidate_id'], m=hit['m']))
        elif bad:
            row.update(decision='UNSUPPORTED', reasons=bad)
        else:
            row.update(decision='REQUEST', request=dict(
                provider=provider.MODULE, kernel=S3.P1_KERNEL, recipe=provider.RECIPE, space='sector36',
                context=dict(builder=S3.MODULE + ':load_context', common_operators=S3.COMMON,
                             tail_operators=dict(S3.operators_locator(item), field='Pauli_nonidentity')),
                reference='provider.saved_h4_reference: PHYSICAL_INPUTS rounds[%d] (T=%r)' % (m, c['T']),
                route='scripts/run_pauli_signal_entry.py --execute --approve-new-means'))
        if row['decision'] != 'REUSE':
            same_m = [v for v in by_m.get(m, []) if v['record']['physical_context']['split']['tail_dictionary']
                      .get('dictionary_sha256') == job['physical_context']['split']['tail_dictionary']['dictionary_sha256']]
            if same_m:
                row['changed_from_saved_round'] = dict(
                    mean_source=same_m[0]['mean_source'],
                    fields=changed_fields(physical_key(same_m[0]['record']['physical_context'])[1], value))
        rows.append(row)
    counts = {}
    for r in rows:
        counts[r['decision']] = counts.get(r['decision'], 0) + 1
    return dict(schema='A_PAULI_SIGNAL_ENTRY_PLAN_V1', configuration_id=config['id'], configuration_source=configuration_source,
                prefix=config['spec'].get('L_D'), configuration_reasons=reasons, counts=counts,
                reuse_rounds=[r['m'] for r in rows if r['decision'] == 'REUSE'],
                rebind_rounds=[r['m'] for r in rows if r.get('reuse') == 'REBOUND_SAME_PHYSICAL_KEY'],
                request_rounds=[r['m'] for r in rows if r['decision'] == 'REQUEST'],
                unsupported_rounds=[r['m'] for r in rows if r['decision'] == 'UNSUPPORTED'],
                rounds=rows,
                dictionary=None if item is None else dict(source=item['source'], sha256=item['dictionary_sha256'],
                                                           lambda_P=item['payload']['lambda_P'],
                                                           operators=S3.operators_locator(item)),
                saved_packets=[deepcopy(p) for p in packets],
                provider=dict(module=provider.MODULE, kernel=S3.P1_KERNEL, formulas=list(PROVIDER_FORMULAS), K=list(PROVIDER_K),
                              validated_for_pauli=VALIDATED),
                new_mean_calls=0, compile=0, Q3=0)


# ---------------------------------------------------------------- carriers, execution, packet
def rebind(record, config, m, b, *, provenance):
    """A saved record with the same physical key, under this configuration's canonical identity."""
    job = p1.job(config, m, record['source'], b)
    if physical_key(job['physical_context'])[0] != physical_key(record['physical_context'])[0]:
        raise ValueError('PHYSICAL_KEY_DIFFERS_NO_REBIND')
    out = deepcopy(record)
    out.update(physical_context=deepcopy(job['physical_context']),
               candidate=deepcopy(job['physical_context']['physical_candidate']),
               B_identity=deepcopy(job['B_identity']), B_law_binding=deepcopy(job['B_law_binding']),
               B_configuration_id=job['B_configuration_id'], B_candidate_id=job['B_candidate_id'], m=m,
               reuse=dict(deepcopy(provenance), physical_key=physical_key(record['physical_context'])[0],
                          kind='SAME_PHYSICAL_KEY_IDENTITY_REBOUND_SIGNAL_UNCHANGED'))
    return out


def carrier_name(prefix, m):
    return 'LD%d_m%d.json' % (prefix, m)


def write_carriers(root, config, plan_value, index, save, result):
    """Re-bound REUSE records as carriers of this run (metadata only)."""
    b = p1.b_api(root); prefix = config['spec']['L_D']
    for row in plan_value['rounds']:
        if row.get('reuse') != 'REBOUND_SAME_PHYSICAL_KEY':
            continue
        hit = index[row['physical_key']]
        rec = rebind(hit['record'], config, row['m'], b, provenance=dict(
            mean_source=hit['mean_source'], configuration_id=hit['configuration_id'], candidate_id=hit['candidate_id'],
            m=hit['m'], handoff=hit['handoff']))
        item = dict(prefix=prefix, m=row['m'], space='sector36', kind='REUSED_SAME_PHYSICAL_KEY_REBOUND', record=rec)
        save(carrier_name(prefix, row['m']), item)
        result['rebound'].append(row['m'])


def run_requests(config, plan_value, jobs, evaluate, save, result):
    """REQUEST rounds only, fixed increasing m; the first FAIL or identity mismatch stops all calls."""
    prefix = config['spec']['L_D']; requested = plan_value['request_rounds']
    if sorted(jobs) != requested:
        raise ValueError('JOBS_DIFFER_FROM_PLAN_REQUESTS')
    for m in requested:
        if len(result['records']) >= len(requested):
            raise ValueError('SCIENTIFIC_CALL_SCOPE_EXHAUSTED')
        job = jobs[m]; record = evaluate(m, job)
        item = dict(prefix=prefix, m=m, space='sector36', kind='COMMON_KAPPA', record=record)
        result['records'].append(item)
        save(carrier_name(prefix, m), item); save('CHECKPOINT.json', result)
        if (record.get('B_candidate_id') != job['B_candidate_id'] or record.get('B_configuration_id') != job['B_configuration_id']
                or record.get('candidate', {}).get('id') != job['B_candidate_id'] or record.get('m') != m
                or record.get('B_identity') != job['B_identity'] or record.get('B_law_binding') != job['B_law_binding']
                or record.get('source') != job['physical_context']['numerical_recipe']['source']):
            result.update(status='STOP_RETURNED_IDENTITY_SOURCE_OR_LAW_MISMATCH', error=dict(m=m))
            return False
        if record['status'] != LOCAL or record.get('numerical', {}).get('gate_pass') is not True:
            result.update(status='STOP_NUMERICAL_FAILURE', error=dict(m=m, reason=record.get('numerical_failure_reasons')))
            return False
    return True


def check_record(config, m, record, b):
    """S3.check_record with the record's own numerical SOURCE (per-round SOURCE is authoritative)."""
    job = p1.job(config, m, record['source'], b)
    S3.check_record(config, m, record, job, science_source=record['source']['commit'])


def packet(config, rows, *, raw_result_source=None, configuration_source=None):
    """MEAN_HANDOFF in the LD1-LD5 layout. rows: {m: (record, mean_source, reused_without_reanalysis)}."""
    prefix = config['spec']['L_D']; out_rows, evaluated, failed, reused, new = [], [], [], [], []
    sources = set(); rounds = list(range(config['spec']['M'] + 1))
    for row_c in config['rounds']:
        m = row_c['m']; hit = rows.get(m)
        row = dict(m=m, candidate_id=row_c['candidate']['id'], identity=deepcopy(row_c['identity']),
                   law_binding=deepcopy(row_c['law_binding']), status='NOT_EVALUATED', mean_record=None, mean_source=None)
        if hit is not None:
            record, source, reuse = hit
            row.update(status=record['status'], mean_record=deepcopy(record), mean_source=deepcopy(source))
            sources.add(record['source']['commit'])
            (reused if reuse else new).append(m)
            (evaluated if record['status'] == LOCAL and record['numerical'].get('gate_pass') is True else failed).append(m)
        row.update(N_X=None, N_Y=None, nominal_N_X=None, nominal_N_Y=None, sufficient_N_X=None, sufficient_N_Y=None,
                   global_RMSE_bound=None, C=None, G=None, scientific_PASS_FAIL=None,
                   reused_without_reanalysis=None if hit is None else hit[2])
        out_rows.append(row)
    complete = evaluated == rounds
    value = dict(schema=config['schema'], kind='A_PAULI_ALL_ROUND_MEANS_ONLY_NOT_N_PACKET' if complete
                 else 'A_PAULI_PARTIAL_MEANS_ONLY_NOT_N_PACKET', prefix=prefix, configuration=deepcopy(config), B_source=p1.B,
                 scientific_source=next(iter(sources)) if len(sources) == 1 else 'PER_ROUND_MEAN_SOURCE',
                 dictionary_identity=deepcopy(config['spec']['split']['tail_dictionary']), rounds=out_rows,
                 evaluated_rounds=evaluated, sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None,
                 actual_sampler_qualified=False, state_ground_certified=False, mandatory_stop=True,
                 per_round_source_is_authoritative=True, reused_rounds=reused, new_rounds=new, failed_rounds=failed,
                 signal_coverage_complete=complete)
    n = len(rounds)
    return dict(kind=('A_PAULI_LD%d_ALL_ROUND_SIGNAL_HANDOFF' if complete else 'A_PAULI_LD%d_PARTIAL_SIGNAL_HANDOFF') % prefix,
                status='ALL_%d_PAULI_LD%d_MEANS_AVAILABLE_NOT_RPE_QUALIFIED' % (n, prefix) if complete
                else 'INCOMPLETE_NUMERICAL_OR_INPUT_STOP', packets=[value], raw_result_source=deepcopy(raw_result_source),
                configuration_source=deepcopy(configuration_source),
                dictionary_source=deepcopy(config['spec']['split']['tail_dictionary']['source']),
                B_intake_dictionary_identity=deepcopy(HELPER), assembly=MODULE,
                sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None, mandatory_stop=True)


def assemble_rows(root, config, plan_value, *, carriers, raw_commit, directory):
    """{m: (record, mean_source, reused)}: saved carriers for identical REUSE, this run's carriers otherwise.

    carriers(m) returns this run's committed carrier item for round m.
    """
    b = p1.b_api(root); prefix = config['spec']['L_D']; rows = {}
    for row in plan_value['rounds']:
        m = row['m']
        if row['decision'] == 'UNSUPPORTED':
            raise ValueError('UNSUPPORTED_ROUND_IN_PLAN: m=%d' % m)
        if row.get('reuse') == 'IDENTICAL_IDENTITY':
            src = deepcopy(row['saved']['mean_source']); rec = read_ref(root, src); reused = True
        else:
            item = carriers(m)
            if item is None:
                continue
            if item['m'] != m or item['prefix'] != prefix:
                raise ValueError('CARRIER_ROUND_DIFFERS: m=%d' % m)
            rec = item['record']; reused = row['decision'] == 'REUSE'
            if reused and (item['kind'] != 'REUSED_SAME_PHYSICAL_KEY_REBOUND'
                           or rec['reuse']['mean_source'] != row['saved']['mean_source']):
                raise ValueError('REBOUND_CARRIER_PROVENANCE_DIFFERS: m=%d' % m)
            if not reused and item['kind'] != 'COMMON_KAPPA':
                raise ValueError('NEW_CARRIER_KIND_DIFFERS: m=%d' % m)
            src = dict(commit=raw_commit, path=directory + carrier_name(prefix, m), selector=['record'],
                       scientific_source=rec['source']['commit'])
        if src['scientific_source'] != rec['source']['commit'] or physical_key(rec['physical_context'])[0] != row['physical_key']:
            raise ValueError('RECORD_SOURCE_OR_PHYSICAL_KEY_DIFFERS: m=%d' % m)
        if rec['status'] == LOCAL and rec.get('numerical', {}).get('gate_pass') is True:
            check_record(config, m, rec, b)
        rows[m] = (rec, src, reused)
    return rows


def b_intake_check(config, handoff, records, dictionary, b):
    """B's bind_packet conditions (rpe_pauli_signal_intake b6d4dea49), applied before handing over."""
    pkt = handoff['packets'][0]; s = config['spec']; d = pkt['dictionary_identity']; problems = []
    def need(ok, label):
        if not ok:
            problems.append(label)
    b.validate(config)
    rounds = list(range(s['M'] + 1))
    need(pkt['configuration'] == config, 'configuration')
    need(s['method'] == 'PR' and d.get('kind') == KIND and s['event_mapping']['kind'] == DRAFT_MAPPER
         and pkt.get('per_round_source_is_authoritative') is True, 'pauli_packet_and_source')
    need(s['split']['tail_dictionary'] == d and s['law']['tail_dictionary'] == d, 'split_law_dictionary')
    item = dict(payload=dictionary, dictionary_sha256=hashlib.sha256(canonical(dictionary).encode()).hexdigest(), source=d['source'])
    need(S3.b_intake_dictionary_identity(item) == d, 'dictionary_identity_helper')
    need(dictionary['binding'] == s['binding'] and dictionary['prefix'] == s['L_D'], 'dictionary_binding_prefix')
    need(Fraction(dictionary['lambda_P']) == Fraction(s['split']['lambda_R']['numerator'], s['split']['lambda_R']['denominator']), 'lambda')
    need(dictionary['saved_DF_scalar'] == s['split']['scalar'], 'scalar')
    need(s['split']['lambda_source'] == dict(d['source'], selector=['lambda_P']), 'lambda_source')
    need([r['m'] for r in pkt['rounds']] == rounds and sorted(records) == rounds and pkt['evaluated_rounds'] == rounds, 'all_rounds')
    need(not pkt.get('failed_rounds') and pkt.get('signal_coverage_complete') is True
         and pkt.get('actual_sampler_qualified') is False and pkt.get('state_ground_certified') is False, 'complete_unqualified')
    for row_c, row in zip(config['rounds'], pkt['rounds']):
        m = row['m']; rec = records.get(m)
        if rec is None:
            problems.append('record_m%d' % m); continue
        need(row['mean_record'] == rec, 'carrier_payload_m%d' % m)
        need(row['candidate_id'] == row_c['candidate']['id'] and row['identity'] == row_c['identity']
             and row['law_binding'] == row_c['law_binding'], 'canonical_m%d' % m)
        job = p1.job(config, m, rec['source'], b)
        need(all(rec[k] == job[k] for k in ('m', 'B_configuration_id', 'B_candidate_id', 'B_identity', 'B_law_binding',
                                             'physical_context')) and rec['candidate'] == job['physical_context']['physical_candidate'],
             'record_job_m%d' % m)
        need(all(rec[k] == row_c['structure'][k] for k in ('signed_weights', 'allocation', 'event_count')), 'structure_m%d' % m)
        need(row['mean_source'].get('scientific_source') == rec['source'].get('commit'), 'per_round_source_m%d' % m)
        need(bool(rec.get('input_source')) and bool(rec.get('reference_source')) and rec.get('scalar_once') is True,
             'input_reference_scalar_m%d' % m)
        need(all(rec.get(k) is None for k in ('N_X', 'N_Y', 'C', 'G', 'global_RMSE_bound', 'scientific_PASS_FAIL'))
             and all(row.get(k) is None for k in ('N_X', 'N_Y', 'nominal_N_X', 'nominal_N_Y', 'sufficient_N_X', 'sufficient_N_Y',
                                                  'global_RMSE_bound', 'C', 'G', 'scientific_PASS_FAIL')), 'means_only_m%d' % m)
    return problems


def git_head(root):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()

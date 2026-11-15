"""PR-Pauli stage 2 for any saved split: the 13 round means of a fixed configuration.

The LD3 stage-2 path (8c8ce542e, ``df_head_pauli_tail_ld3_signals``) is reused unchanged:
its sector36 context builder (checked against P1 ``load_contexts``), record checks against
P1 ``job``, B intake helper and summary columns. Only the fixed inputs (prefix, P0
dictionary carrier, stage-1 configuration) and output names become parameters. LD3 is kept
as a spec so the generalization can be checked against the published LD3 packet.
No sampling, build/compile, price, N, Q3 or shot allocation.
"""
from copy import deepcopy
from fractions import Fraction

from . import df_head_pauli_tail_ld3_signals as S3

L, p0, p1, provider = S3.L, S3.p0, S3.p1, S3.provider
MODULE = 'src/trottertracks/df_head_pauli_tail_split_signals.py'
ROUNDS = S3.ROUNDS
LOCAL = S3.LOCAL
SPECS = {
    3: dict(prefix=3, stage1_code='2e1949a0b66dd8fe086ac7cd3104eb89b4553e38', p0_status=L.LD3_P0_SUCCESS,
            p0=dict(commit='50640b74c65c36d381d6058824745f6e38ab3c2d',
                    directory='docs/research/df_head_pauli_tail_ld3_stage1_20261116/p0_run01/'),
            config=dict(commit='421258d400a86fffd7c4955d5cf1bd24ac5729cb',
                        path='docs/research/df_head_pauli_tail_ld3_stage1_20261116/config_run01/CONFIGURATION_LD3.json'),
            config_id='dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847',
            config_status='LD3_CONFIGURATION_AND_SCHEDULE_COMPLETE_NOT_SIGNAL_OR_PRICE',
            directory='docs/research/df_head_pauli_tail_ld3_signals_20261116/'),
    5: dict(prefix=5, stage1_code='d6385593e86577ce63deebb7e7316ab152ac969b',
            p0_status='P0_LD5_DICTIONARY_COMPLETE_NOT_P1_OR_RPE_QUALIFIED',
            p0=dict(commit='6255fbc422121fd777cad52763ef5a684abd46a3',
                    directory='docs/research/df_head_pauli_tail_ld5_stage1_20261116/p0_run01/'),
            config=dict(commit='39a1fc7dfe9b093f0e246be595e2f7938191368b',
                        path='docs/research/df_head_pauli_tail_ld5_stage1_20261116/config_run01/CONFIGURATION_LD5.json'),
            config_id='4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919eb',
            config_status='LD5_CONFIGURATION_AND_SCHEDULE_COMPLETE_NOT_SIGNAL_OR_PRICE',
            directory='docs/research/df_head_pauli_tail_ld5_signals_20261116/')}


def success(spec):
    return 'ALL_13_PAULI_LD%d_MEANS_AVAILABLE_NOT_RPE_QUALIFIED' % spec['prefix']


def dictionary_item(root, spec, meta):
    """L.dictionary_item checks with the stage-1 P0 status of this prefix."""
    folder = spec['p0']['directory']; prefix = spec['prefix']
    result = L.read(root, spec['p0']['commit'], folder+'result.json')
    if (result['status'] != spec['p0_status'] or not result['checks'] or not all(c['PASS'] for c in result['checks'])
            or result['binding'] != meta['binding'] or result['control']['PASS'] is not True
            or result['scientific_source'] != spec['stage1_code']):
        raise ValueError('STAGE1_P0_NOT_COMPLETE')
    row = next(r for r in result['prefixes'] if r['prefix'] == prefix)
    path = folder+'LD'+str(prefix)+'_DICTIONARY.json'
    obj = L.read(root, spec['p0']['commit'], path)
    if (L.canonical_sha(obj) != row['dictionary_sha256'] or obj['binding'] != meta['binding']
            or obj['prefix'] != prefix or 'GATES_PASS' not in row['status']
            or obj['saved_DF_scalar'] != meta['saved_tail_descriptors'][str(prefix)]['scalar']):
        raise ValueError('P0_DICTIONARY_BINDING_OR_SCALAR')
    return dict(payload=obj, dictionary_sha256=row['dictionary_sha256'], source=dict(commit=spec['p0']['commit'], path=path),
                P0_metrics=row['identity_metrics'], P0_result_source=dict(commit=spec['p0']['commit'], path=folder+'result.json'))


def load_fixed(root, spec):
    """Saved configuration, dictionary and bindings only (no arrays, no mean); as S3.load_fixed."""
    prefix = spec['prefix']
    saved = L.p0_metadata(root, (2, prefix))
    stage1 = L.read(root, spec['config']['commit'], spec['config']['path'].rsplit('/', 1)[0]+'/result.json')
    if stage1['status'] != spec['config_status']:
        raise ValueError('STAGE1_CONFIGURATION_NOT_COMPLETE')
    item = dictionary_item(root, spec, saved)
    config = L.read(root, spec['config']['commit'], spec['config']['path'])
    b = p1.b_api(root); b.validate(config)
    s = config['spec']
    if (config['id'] != spec['config_id'] or s['L_D'] != prefix or s['M'] != 12 or s['formula'] != '2nd'
            or s['finite_K'] != 2 or provider._scalar(s['h']) != Fraction(.1)
            or provider._scalar(s['delta_pf']) != Fraction(.1)
            or provider._scalar(s['kappa']['value']) != Fraction(saved['original_exact_kappa'])
            or s['binding'] != saved['binding']
            or s['split']['tail_dictionary'] != L.dictionary_identity(item)
            or s['split']['tail_dictionary'] != S3.b_intake_dictionary_identity(item)
            or s['split']['scalar'] != saved['saved_tail_descriptors'][str(prefix)]['scalar']
            or s['split']['tail_dictionary']['scalar_policy'] != 'KEEP_SAVED_sR_AND_c_ONCE'
            or [r['m'] for r in config['rounds']] != list(ROUNDS)):
        raise ValueError('CANONICAL_FIXED_CONDITIONS_DIFFER')
    return saved, item, config, b


def run(jobs, evaluate, save, result, spec):
    """S3.run with this prefix: fixed m0..m12 order; the first FAIL or mismatch stops all calls."""
    prefix = spec['prefix']
    if set(jobs) != set(ROUNDS):
        raise ValueError('ONLY_THE_13_ROUNDS')
    for m in ROUNDS:
        if len(result['records']) >= len(ROUNDS):
            raise ValueError('SCIENTIFIC_CALL_SCOPE_EXHAUSTED')
        job = jobs[m]; record = evaluate(m, job)
        item = dict(prefix=prefix, m=m, space='sector36', kind='COMMON_KAPPA', record=record)
        result['records'].append(item)
        save('LD%d_m%d.json' % (prefix, m), item); save('CHECKPOINT.json', result)
        if (record.get('B_candidate_id') != job['B_candidate_id']
                or record.get('B_configuration_id') != job['B_configuration_id']
                or record.get('candidate', {}).get('id') != job['B_candidate_id']
                or record.get('m') != m or record.get('B_identity') != job['B_identity']
                or record.get('B_law_binding') != job['B_law_binding']
                or record.get('source') != job['physical_context']['numerical_recipe']['source']):
            result.update(status='STOP_RETURNED_IDENTITY_SOURCE_OR_LAW_MISMATCH',
                          error=dict(m=m, reason='Returned record differs from fixed request'))
            return False
        if record['status'] != LOCAL or record.get('numerical', {}).get('gate_pass') is not True:
            result.update(status='STOP_NUMERICAL_FAILURE', error=dict(m=m, reason=record.get('numerical_failure_reasons')))
            return False
    result['status'] = success(spec)
    return True


def packet(config, records, jobs, *, raw_commit, science_source, spec, helper_source):
    """S3.packet layout for this prefix; stored values are copied, never recomputed."""
    prefix = spec['prefix']; folder = spec['directory']
    rows, evaluated, failed = [], [], []
    for canonical in config['rounds']:
        m = canonical['m']; record = records.get(m)
        row = dict(m=m, candidate_id=canonical['candidate']['id'], identity=deepcopy(canonical['identity']),
                   law_binding=deepcopy(canonical['law_binding']), status='NOT_EVALUATED', mean_record=None, mean_source=None)
        if record is not None:
            S3.check_record(config, m, record, jobs[m], science_source=science_source)
            row.update(status=record['status'], mean_record=deepcopy(record),
                       mean_source=dict(commit=raw_commit, path=folder+'run01/LD%d_m%d.json' % (prefix, m),
                                        selector=['record'], scientific_source=science_source))
            (evaluated if record['status'] == LOCAL and record['numerical'].get('gate_pass') is True else failed).append(m)
        row.update(N_X=None, N_Y=None, nominal_N_X=None, nominal_N_Y=None, sufficient_N_X=None, sufficient_N_Y=None,
                   global_RMSE_bound=None, C=None, G=None, scientific_PASS_FAIL=None,
                   reused_without_reanalysis=False if record is not None else None)
        rows.append(row)
    complete = evaluated == list(ROUNDS)
    value = dict(schema=config['schema'], kind='A_PAULI_ALL_ROUND_MEANS_ONLY_NOT_N_PACKET' if complete
                 else 'A_PAULI_PARTIAL_MEANS_ONLY_NOT_N_PACKET', prefix=prefix, configuration=deepcopy(config),
                 B_source=p1.B, scientific_source=science_source,
                 dictionary_identity=deepcopy(config['spec']['split']['tail_dictionary']),
                 rounds=rows, evaluated_rounds=evaluated, sufficient_N=None, global_RMSE_bound=None, C=None, G=None,
                 winner=None, actual_sampler_qualified=False, state_ground_certified=False, mandatory_stop=True,
                 per_round_source_is_authoritative=True, reused_rounds=[], new_rounds=sorted(records),
                 failed_rounds=failed, signal_coverage_complete=complete)
    return dict(kind='A_PAULI_LD%d_ALL_ROUND_SIGNAL_HANDOFF' % prefix if complete else 'A_PAULI_LD%d_PARTIAL_SIGNAL_HANDOFF' % prefix,
                status=success(spec) if complete else 'INCOMPLETE_NUMERICAL_OR_INPUT_STOP', packets=[value],
                raw_result_source=dict(commit=raw_commit, path=folder+'run01/result.json'),
                configuration_source=deepcopy(spec['config']),
                dictionary_source=deepcopy(config['spec']['split']['tail_dictionary']['source']),
                B_intake_dictionary_identity=deepcopy(helper_source),
                sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None, mandatory_stop=True)


def summary_row(record, *, carrier, path, science_source, prefix):
    """S3.summary_row with this prefix's label and provenance."""
    row = S3.summary_row(record, carrier=carrier, path=path, science_source=science_source)
    row.update(label='LD%d_m%d' % (prefix, record['m']), provenance='NEW_AUTHORIZED_LD%d_13_ROUND_BATCH' % prefix)
    return row

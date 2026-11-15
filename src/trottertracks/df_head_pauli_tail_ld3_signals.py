"""PR-Pauli LD3 stage 2: the 13 round means of configuration dd5b881d.

The P1 all-round path is reused: P1 ``job``, the saved H4 reference, the
fixed provider ``evaluate`` and a sector36 context built as P1
``load_contexts`` builds it (common operators from 4220f3006, the LD3
Pauli_nonidentity from 50640b74c, scalar KEEP_SAVED_sR_AND_c_ONCE). The
builder is checked against P1 ``load_contexts`` on LD2. No sampling,
build/compile, price, N, Q3 or shot allocation.
"""
from copy import deepcopy
from fractions import Fraction
import io
import json
import math
from pathlib import PurePosixPath

from . import df_head_pauli_tail_ld3 as L

p0, p1 = L.p0, L.p1
provider = p1.provider
MODULE = 'src/trottertracks/df_head_pauli_tail_ld3_signals.py'
P1_KERNEL = 'e8641b84fd8a3e98a49b65ecaa8fbd371a9d8ab7'
STAGE1_CODE = '2e1949a0b66dd8fe086ac7cd3104eb89b4553e38'
STAGE1_DIR = 'docs/research/df_head_pauli_tail_ld3_stage1_20261116/'
P0_LD3 = dict(commit='50640b74c65c36d381d6058824745f6e38ab3c2d', directory=STAGE1_DIR+'p0_run01/')
CONFIG = dict(commit='421258d400a86fffd7c4955d5cf1bd24ac5729cb', path=STAGE1_DIR+'config_run01/CONFIGURATION_LD3.json')
CONFIG_ID = 'dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847'
COMMON = dict(commit=p1.P0, path=p1.P0DIR+'run01/COMMON_OPERATORS.npz')
DIRECTORY = 'docs/research/df_head_pauli_tail_ld3_signals_20261116/'
PREFIX = 3
ROUNDS = tuple(range(13))
LOCAL = 'LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED'
SUCCESS = 'ALL_13_PAULI_LD3_MEANS_AVAILABLE_NOT_RPE_QUALIFIED'


def operators_locator(item):
    """The P0 operator archive written next to the dictionary in the same run."""
    path = PurePosixPath(item['source']['path'])
    return dict(commit=item['source']['commit'], path=str(path.parent/('LD'+str(item['payload']['prefix'])+'_OPERATORS.npz')))


def b_intake_dictionary_identity(item):
    """Metadata helper for B intake: residual locator = sibling result.json.

    Identical to P1 ``dictionary_identity`` for the saved LD1/LD2 dictionaries
    and to the stage-1 LD3 descriptor; ``item`` is B's (payload, sha, source).
    """
    path = PurePosixPath(item['source']['path'])
    return L.dictionary_identity(dict(item, P0_result_source=dict(commit=item['source']['commit'],
        path=str(path.parent/'result.json'))))


def git_archive(root):
    def archive(locator):
        import numpy as np
        with np.load(io.BytesIO(p0.blob(root, locator['commit'], locator['path'])), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return archive


def load_fixed(root):
    """Saved configuration, dictionary and bindings only (no arrays, no mean)."""
    saved = L.p0_metadata(root, (2, PREFIX))
    p0_result = L.read(root, P0_LD3['commit'], P0_LD3['directory']+'result.json')
    if (p0_result['status'] != L.LD3_P0_SUCCESS or p0_result['control']['PASS'] is not True
            or p0_result['scientific_source'] != STAGE1_CODE):
        raise ValueError('STAGE1_P0_NOT_COMPLETE')
    stage1 = L.read(root, CONFIG['commit'], STAGE1_DIR+'config_run01/result.json')
    if stage1['status'] != 'LD3_CONFIGURATION_AND_SCHEDULE_COMPLETE_NOT_SIGNAL_OR_PRICE':
        raise ValueError('STAGE1_CONFIGURATION_NOT_COMPLETE')
    item = L.dictionary_item(root, P0_LD3['commit'], P0_LD3['directory'], PREFIX, saved)
    config = L.read(root, CONFIG['commit'], CONFIG['path'])
    b = p1.b_api(root); b.validate(config)
    s = config['spec']
    if (config['id'] != CONFIG_ID or s['L_D'] != PREFIX or s['M'] != 12 or s['formula'] != '2nd'
            or s['finite_K'] != 2 or provider._scalar(s['h']) != Fraction(.1)
            or provider._scalar(s['delta_pf']) != Fraction(.1)
            or provider._scalar(s['kappa']['value']) != Fraction(saved['original_exact_kappa'])
            or s['binding'] != saved['binding']
            or s['split']['tail_dictionary'] != L.dictionary_identity(item)
            or s['split']['tail_dictionary'] != b_intake_dictionary_identity(item)
            or s['split']['scalar'] != saved['saved_tail_descriptors'][str(PREFIX)]['scalar']
            or s['split']['tail_dictionary']['scalar_policy'] != 'KEEP_SAVED_sR_AND_c_ONCE'
            or [r['m'] for r in config['rounds']] != list(ROUNDS)):
        raise ValueError('LD3_CANONICAL_FIXED_CONDITIONS_DIFFER')
    return saved, item, config, b


def load_context(root, saved, item, normalizers, *, budget, archive=None, loaded=None):
    """P1 ``load_contexts`` sector36 construction for one saved dictionary."""
    import numpy as np
    archive = archive or git_archive(root)
    ham, sector, full_state, state, observed = loaded or p0.load_saved(root, saved)
    common_ops = archive(COMMON); basis = sector.basis_indices
    if not np.array_equal(basis, common_ops['sector_basis_indices']):
        raise ValueError('P0_SECTOR_BASIS_DIFFERENT')
    prefix = item['payload']['prefix']; obj = item['payload']; located = operators_locator(item)
    A = archive(located)['Pauli_nonidentity']
    tails = {prefix: dict(scalar=obj['saved_DF_scalar'], lambda_R=obj['lambda_P'],
        source=dict(item['source'], selector=['lambda_P']))}
    ctx = provider.make_context([a[np.ix_(basis, basis)] for a in common_ops['blocks']], state,
        ham.constant, tails, binding=saved['binding'], source=deepcopy(COMMON))
    matrix = np.array(A[np.ix_(basis, basis)], copy=True); matrix.setflags(write=False)
    ctx.tails[prefix].update(matrix=matrix, dictionary_identity=L.dictionary_identity(item),
        matrix_source=dict(located, field='Pauli_nonidentity'), P0_operator_metrics=deepcopy(item['P0_metrics']))
    ctx.normalizers = normalizers; ctx.normalizer_budget = budget
    ctx.state_metadata = dict(saved_loader=observed, ground_state_certified=False,
        actual_law_error=None, state_not_regenerated=True)
    return ctx


def same_context(left, right):
    """Exact equality of every array and descriptor a mean call reads."""
    import numpy as np
    eq = lambda a, b: a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    if (len(left.matrices) != len(right.matrices) or not all(eq(a, b) for a, b in zip(left.matrices, right.matrices))
            or not eq(left.state, right.state) or left.constant != right.constant
            or left.binding != right.binding or left.source != right.source
            or set(left.tails) != set(right.tails) or left.state_metadata != right.state_metadata):
        return False
    for k in left.tails:
        a, b = left.tails[k], right.tails[k]
        if set(a) != set(b) or not eq(a['matrix'], b['matrix']):
            return False
        if any(a[x] != b[x] for x in a if x != 'matrix'):
            return False
    return True


def run(jobs, evaluate, save, result):
    """Fixed m0..m12 order; the first FAIL or identity mismatch stops all calls."""
    if set(jobs) != set(ROUNDS):
        raise ValueError('ONLY_THE_13_LD3_ROUNDS')
    for m in ROUNDS:
        if len(result['records']) >= len(ROUNDS):
            raise ValueError('SCIENTIFIC_CALL_SCOPE_EXHAUSTED')
        job = jobs[m]; record = evaluate(m, job)
        item = dict(prefix=PREFIX, m=m, space='sector36', kind='COMMON_KAPPA', record=record)
        result['records'].append(item)
        save('LD3_m'+str(m)+'.json', item); save('CHECKPOINT.json', result)
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
    result['status'] = SUCCESS
    return True


def check_record(config, m, record, job, *, science_source):
    """B intake's per-round conditions, checked before a row enters the packet."""
    canonical = config['rounds'][m]
    if (record['B_configuration_id'] != config['id'] or record['B_candidate_id'] != canonical['candidate']['id']
            or record['B_identity'] != canonical['identity'] or record['B_law_binding'] != canonical['law_binding']
            or record['candidate']['id'] != canonical['candidate']['id'] or record['source']['commit'] != science_source
            or any(record[k] != job[k] for k in ('m', 'B_configuration_id', 'B_candidate_id', 'B_identity',
                'B_law_binding', 'physical_context'))
            or record['candidate'] != job['physical_context']['physical_candidate']
            or any(record[k] != canonical['structure'][k] for k in ('signed_weights', 'allocation', 'event_count'))
            or not record.get('input_source') or not record.get('reference_source') or record.get('scalar_once') is not True):
        raise ValueError('LD3_RECORD_CANONICAL_ID_SOURCE_OR_LAW_MISMATCH: m='+str(m))
    for k in ('N_X', 'N_Y', 'C', 'G', 'global_RMSE_bound', 'scientific_PASS_FAIL'):
        if record.get(k) is not None:
            raise ValueError('UNAUTHORIZED_QUALIFICATION_OR_COST')


def packet(config, records, jobs, *, raw_commit, science_source):
    """LD1/LD2 MEAN_HANDOFF packet layout; stored values are copied, never recomputed."""
    rows = []; evaluated = []; failed = []
    for canonical in config['rounds']:
        m = canonical['m']; record = records.get(m)
        row = dict(m=m, candidate_id=canonical['candidate']['id'], identity=deepcopy(canonical['identity']),
            law_binding=deepcopy(canonical['law_binding']), status='NOT_EVALUATED', mean_record=None, mean_source=None)
        if record is not None:
            check_record(config, m, record, jobs[m], science_source=science_source)
            row.update(status=record['status'], mean_record=deepcopy(record),
                mean_source=dict(commit=raw_commit, path=DIRECTORY+'run01/LD3_m'+str(m)+'.json',
                    selector=['record'], scientific_source=science_source))
            (evaluated if record['status'] == LOCAL and record['numerical'].get('gate_pass') is True else failed).append(m)
        row.update(N_X=None, N_Y=None, nominal_N_X=None, nominal_N_Y=None, sufficient_N_X=None, sufficient_N_Y=None,
            global_RMSE_bound=None, C=None, G=None, scientific_PASS_FAIL=None,
            reused_without_reanalysis=False if record is not None else None)
        rows.append(row)
    complete = evaluated == list(ROUNDS)
    value = dict(schema=config['schema'], kind='A_PAULI_ALL_ROUND_MEANS_ONLY_NOT_N_PACKET' if complete
            else 'A_PAULI_PARTIAL_MEANS_ONLY_NOT_N_PACKET', prefix=PREFIX, configuration=deepcopy(config),
        B_source=p1.B, scientific_source=science_source, dictionary_identity=deepcopy(config['spec']['split']['tail_dictionary']),
        rounds=rows, evaluated_rounds=evaluated, sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None,
        actual_sampler_qualified=False, state_ground_certified=False, mandatory_stop=True,
        per_round_source_is_authoritative=True, reused_rounds=[], new_rounds=sorted(records),
        failed_rounds=failed, signal_coverage_complete=complete)
    return dict(kind='A_PAULI_LD3_ALL_ROUND_SIGNAL_HANDOFF' if complete else 'A_PAULI_LD3_PARTIAL_SIGNAL_HANDOFF',
        status=SUCCESS if complete else 'INCOMPLETE_NUMERICAL_OR_INPUT_STOP', packets=[value],
        raw_result_source=dict(commit=raw_commit, path=DIRECTORY+'run01/result.json'),
        configuration_source=deepcopy(CONFIG), dictionary_source=deepcopy(config['spec']['split']['tail_dictionary']['source']),
        B_intake_dictionary_identity=dict(commit=science_source, path=MODULE, function='b_intake_dictionary_identity'),
        sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None, mandatory_stop=True)


def summary_row(record, *, carrier, path, science_source):
    """ROUND_SUMMARY.csv columns of the LD1/LD2 all-round publication."""
    s = record['signal']; c = record['candidate']; n = record['numerical']; a = record['allowance']
    z = lambda k: complex(s[k]['real'], s[k]['imag'])
    return dict(label='LD3_m'+str(record['m']), kind='COMMON_KAPPA', space='sector36', prefix=c['prefix'],
        m=record['m'], T=c['T'], qPF=c['q'], R_outer=c['R_outer'], event_count=c['event_count'],
        B_configuration_id=record['B_configuration_id'], B_candidate_id=record['B_candidate_id'],
        raw_real=s['raw']['real'], raw_imag=s['raw']['imag'], corrected_real=s['corrected']['real'],
        corrected_imag=s['corrected']['imag'], exact_tail_real=s['exact_tail']['real'], exact_tail_imag=s['exact_tail']['imag'],
        reference_real=s['reference']['real'], reference_imag=s['reference']['imag'], log_B=s['log_B'], B=s['B'],
        attenuation=s['attenuation'], corrected_reference_abs=abs(z('corrected')-z('reference')),
        finite_RTE_bias_abs=abs(z('finite_RTE_bias')), closure=n['closure'], power_unitarity_error=n['power_unitarity_error'],
        empirical_u=a['u'], empirical_log_B_margin=a['log_B_margin'], status=record['status'],
        provenance='NEW_AUTHORIZED_LD3_13_ROUND_BATCH', carrier_commit=carrier, record_path=path,
        scientific_source=science_source)

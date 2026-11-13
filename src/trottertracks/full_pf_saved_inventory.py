"""Full PF saved-evidence inventory: signal, reference and compiled price coverage.

Reads pinned Git carriers (checked against the study's recorded SHA-256) and,
optionally, read-only private context files. Saved values are copied, never
recomputed, aggregated, extrapolated or ranked. Signal, reference, numerical
evidence and X/Y prices are judged separately. No Hamiltonian/state load,
time evolution, normalization, N/RMSE, build/compile, fit or G.
"""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from . import full_baseline_v2_requests as D

p = D.p
CARRIER = '155529e20a8fac39e00786c40b57830b71203c4f'      # long-time public carriers (SHA in provenance)
LONG = 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
PF_COMMON = {'H4': ('0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c', 'artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/'),
             'H6': ('90f1cd55a0d963953696bd0a27d9d041ff939bb7', 'artifacts/resource_applicability/track_a_pf_common_H6_results/2026-10-22/')}
PRIVATE = Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025')
CARRIER_FILES = ('candidate_inventory.json.gz', 'signal_records.json.gz', 'signal_endpoints.json.gz',
                 'extension_signal_records.json.gz', 'extension_signal_endpoints.json.gz', 'exploration_costs.json.gz',
                 'extension_parallel_costs.json.gz', 'confirmation_costs.json.gz', 'fixed_delta_exploration_costs.json.gz',
                 'reused_compiled_prices.json.gz')
FORMULAS = ('2nd', '4th', '4th(m5_best)')
SIGNAL_FIELDS = ('reference', 'corrected', 'raw', 'total_signed_bias', 'allowance_evidence')
COST_POLICY = 'FULL_COSINE_SINE_HADAMARD_WRAPPER_MEASUREMENT_INCLUDED_P0'
AXES = {'X': 'cosine', 'Y': 'sine'}
CANDIDATE_FIELDS = ('K', 'R_outer', 'T', 'delta', 'formula', 'id', 'method', 'prefix', 'q', 'system')
EXACT, CONDITIONAL, MISMATCH = 'EXACT_METADATA_MATCH', 'CONDITIONAL_REUSE', 'MISMATCH'
MISSING, UNAVAILABLE, FAILED = 'MISSING_IN_CHECKED_SCOPE', 'SOURCE_UNAVAILABLE', 'FAILED_OR_UNQUALIFIED'


def _json(raw, name):
    return json.loads(gzip.decompress(raw) if name.endswith('.gz') else raw)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_carriers(*, private=PRIVATE):
    """Pinned carriers + provenance; private context is optional and read-only."""
    provenance = json.loads(D._blob(D.BASE, LONG + 'completion_provenance.json'))
    science = json.loads(D._blob(D.BASE, LONG + 'scientific_provenance.json'))
    out = dict(files={}, sha={}, provenance=dict(commit=D.BASE, path=LONG + 'completion_provenance.json'),
               science=science, pf_common={}, context={}, context_sha={})
    for name in CARRIER_FILES:
        raw = D._blob(CARRIER, LONG + name); recorded = provenance['result_sha256'].get(name)
        out['files'][name] = _json(raw, name)
        out['sha'][name] = dict(sha256=_sha(raw), recorded=recorded,
                                status='MATCH' if recorded == _sha(raw) else 'NOT_RECORDED' if recorded is None else 'DIFF')
    for system, (commit, path) in PF_COMMON.items():
        out['pf_common'][system] = dict(commit=commit, path=path + 'source_input_identity.json',
            binding=json.loads(D._blob(commit, path + 'source_input_identity.json'))['binding'])
    for system in ('H4', 'H6'):
        f = Path(private) / (system + '_context.json')
        if f.exists():
            raw = f.read_bytes(); out['context'][system] = json.loads(raw); out['context_sha'][system] = _sha(raw)
    return out


def expected_binding(k, system):
    return k['science']['inputs'][system]['binding']


def full_candidates(k):
    """Every saved FULL row (base grid + extension); other methods are excluded."""
    rows = {r['id']: {f: r[f] for f in CANDIDATE_FIELDS} for r in k['files']['candidate_inventory.json.gz'] if r['method'] == 'FULL'}
    for v in k['files']['extension_signal_records.json.gz'].values():
        if v['candidate']['method'] == 'FULL':
            rows.setdefault(v['candidate']['id'], {f: v['candidate'][f] for f in CANDIDATE_FIELDS})
    return [rows[i] for i in sorted(rows, key=lambda i: (rows[i]['system'], rows[i]['formula'], rows[i]['T'], rows[i]['q']))]


def check_physical(c, k):
    """FULL, full rank, known formula, canonical cid and the DF10/DF19 study binding."""
    reasons = []
    rank = D.saved_grid()['rank'][c['system']]
    if c['method'] != 'FULL':
        reasons.append('NOT_FULL_PF_METHOD_' + str(c['method']))
    if c['prefix'] != rank:
        reasons.append('PREFIX_NOT_FULL_RANK')
    if c['formula'] not in FORMULAS:
        reasons.append('FORMULA_OUT_OF_SCOPE')
    if c.get('R_outer') is not None or c.get('K') is not None:
        reasons.append('PF_CARRIES_R_OR_K')
    if c['id'] != p.candidate_id(c):
        reasons.append('CANDIDATE_ID_NOT_CANONICAL')
    if c['delta'] != c['T'] / c['q']:
        reasons.append('WIDTH_NOT_T_OVER_Q')
    binding = expected_binding(k, c['system'])
    if binding.get('rank') != rank or binding.get('rank_policy') != 'tol_only_1e-8':
        reasons.append('STUDY_BINDING_NOT_TOL_ONLY_DF' + str(rank))
    return reasons


def _signal(c, k):
    sig = (k['files']['signal_records.json.gz'].get(c['id']) or k['files']['extension_signal_records.json.gz'].get(c['id']))
    if sig is None:
        return dict(status=MISSING, carrier=None), None
    carrier = dict(commit=CARRIER, path=LONG + ('signal_records.json.gz' if c['id'] in k['files']['signal_records.json.gz']
                                                else 'extension_signal_records.json.gz'), key=c['id'])
    if not p.same(sig['candidate'], c):
        return dict(status=MISMATCH, carrier=carrier, reason='CARRIER_CANDIDATE_DIFFERS'), sig
    fields = [f for f in SIGNAL_FIELDS if sig['signal'].get(f) is not None]
    origin = sig['origin']; reasons = []
    if origin == 'NEW_SAVED_INPUT_FINITE_MEAN':
        kind = 'LONG_TIME_STUDY_EVOLVE_BATCH'
    elif 'track_a_pf_common_' in origin:
        kind = 'REUSED_PF_COMMON_T0.8'
        pc = k['pf_common'][c['system']]['binding']
        if any(not p.same(pc.get(f), expected_binding(k, c['system']).get(f)) for f in ('hamiltonian', 'state', 'snapshot', 'rank_policy')):
            reasons.append('PF_COMMON_BINDING_DIFFERS')
    elif 'fixed_delta' in origin:
        kind = 'REUSED_H6_FIXED_DELTA_SUMMARY'
    else:
        kind = 'UNKNOWN_ORIGIN'; reasons.append('UNKNOWN_ORIGIN')
    if sig.get('saved_allowance_only'):
        reasons.append('SAVED_ALLOWANCE_ONLY_NO_PROPAGATION_EVIDENCE')
    if sig.get('raw_signal_missing'):
        reasons.append('RAW_SIGNAL_MISSING')
    missing = [f for f in SIGNAL_FIELDS if f not in fields]
    status = MISMATCH if 'PF_COMMON_BINDING_DIFFERS' in reasons else CONDITIONAL if reasons or missing else EXACT
    return dict(status=status, carrier=carrier, origin=origin, kind=kind, fields_present=fields, fields_missing=missing,
                reasons=reasons, numerical_SOURCE=k['science'].get('signal_SOURCE') or 'SEE_signal_provenance.json'), sig


def _reference(c, k, sig):
    if sig is None or sig['signal'].get('reference') is None:
        return dict(status=MISSING, value_present=False)
    ctx = k['context'].get(c['system'])
    out = dict(value_present=True, target='ORIGINAL_DF_H_FULL_MATRIX_PLUS_CONSTANT_ON_SAVED_SECTOR_STATE',
               carrier=dict(commit=CARRIER, path=LONG + 'signal_records.json.gz', key=c['id'], field='signal.reference'))
    if 'track_a_pf_common_' in sig['origin'] or 'fixed_delta' in sig['origin']:
        out.update(kind='REUSED_WITH_SAVED_SIGNAL_RECORD', independent_pair=None,
                   independent_pair_status=MISSING, status=CONDITIONAL)
        return out
    if ctx is None:
        out.update(kind=None, status=UNAVAILABLE, reason='PRIVATE_CONTEXT_NOT_READ', independent_pair_status=UNAVAILABLE)
        return out
    T = c['T']
    kind = ('SAVED_REFERENCE_REUSED' if T in ctx['reused_reference_times'] else
            'NEW_SPECTRAL_FULL_H_REFERENCE' if T in ctx['new_reference_times'] else 'UNLISTED_TIME')
    pair = dict(method='scipy.expm_multiply_full_H', T=2.0, error=ctx['reference_expm_multiply_check'],
                source=dict(path=str(PRIVATE / (c['system'] + '_context.json')), sha256=k['context_sha'][c['system']])) if T == 2.0 else None
    out.update(kind=kind, independent_pair=pair, independent_pair_status=EXACT if pair else MISSING,
               status=EXACT if kind != 'UNLISTED_TIME' else CONDITIONAL,
               context=dict(path=str(PRIVATE / (c['system'] + '_context.json')), sha256=k['context_sha'][c['system']]))
    return out


def _numerical(c, k):
    eps = [e for e in k['files']['signal_endpoints.json.gz'] + k['files']['extension_signal_endpoints.json.gz'] if e['candidate']['id'] == c['id']]
    return dict(status='ORIGINAL_CONTRACT_RECORDED' if eps else MISSING, endpoints=len(eps),
                statuses=dict(Counter(e['status'] for e in eps)), by_epsilon={str(e['epsilon']): e['status'] for e in eps},
                policy='OLD_EMPIRICAL_SIGNAL_ELIGIBILITY_alpha_axis_0.025_u_floor_1e-9', rmse_policy_qualified=False,
                note='Saved PASS/FAIL is the original contract result, not a new RMSE result')


def _prices(c, k):
    """All saved price sources for this candidate, each axis judged separately."""
    sources = []; compiler = k['science']['compiler']['hash']; policy = k['science'].get('cost_policy')
    for name in ('exploration_costs.json.gz', 'extension_parallel_costs.json.gz', 'confirmation_costs.json.gz',
                 'fixed_delta_exploration_costs.json.gz'):
        e = k['files'][name].get(c['id'])
        if e is None or 'n' not in e:
            continue
        axes = {}
        for axis, label in AXES.items():
            metric = (e.get('mean_metrics') or {}).get(label) or {}
            value = metric.get('rz_count')
            reasons = []
            if not p.same(e['candidate'], c):
                reasons.append('CANDIDATE_DIFFERS')
            if compiler != p.PROFILE['compiler'] or policy != COST_POLICY:
                reasons.append('COMPILER_OR_COST_POLICY_DIFFERS')
            axes[axis] = dict(rz_count=value, status=MISSING if value is None else MISMATCH if reasons else EXACT, reasons=reasons)
        sources.append(dict(kind='DIRECT_COMPILED_DETERMINISTIC', cohort=e['cohort'], n=e['n'], carrier=dict(commit=CARRIER, path=LONG + name, key=c['id']),
            compiler=compiler, cost_policy=policy, observed_maximum_compiled_error=e.get('observed_maximum_compiled_error'),
            trial_files=e.get('trial_files'), private_trials_present=[Path(f).exists() for f in e.get('trial_files') or []], axes=axes))
    reused = k['files']['reused_compiled_prices.json.gz'].get(c['id'])
    if reused is not None:
        axes = {}; receipts = {}
        for proof in reused.get('proof') or []:
            if 'receipt' in proof:      # paired receipt with candidate fields
                receipts[proof['axis']] = dict(proof['receipt'], format='CANDIDATE_RECEIPT')
            for item in proof.get('axes') or []:    # per-axis compiled record without candidate fields
                receipts[item['axis']] = dict(item, rz_count=item['metrics']['rz_count'], format='AXIS_RECORD',
                                              native_policy=item.get('native_policy'))
        for axis, label in AXES.items():
            value = reused.get(label); reasons = []
            receipt = receipts.get(label)
            if receipt is None:
                reasons.append('NO_RECEIPT_IN_CARRIER_CHECK_RESULT_SHA_IN_PF_COMMON')
            elif (receipt.get('compiler_settings_hash') != p.PROFILE['compiler'] or receipt.get('rz_count') != value
                  or (receipt['format'] == 'CANDIDATE_RECEIPT' and (
                      receipt.get('method') != 'FULL' or receipt.get('prefix') != c['prefix'] or receipt.get('formula') != c['formula']
                      or receipt.get('q') != c['q'] or receipt.get('system') != c['system']))):
                reasons.append('RECEIPT_DIFFERS')
            elif receipt['format'] == 'AXIS_RECORD':
                reasons.append('AXIS_RECEIPT_WITHOUT_CANDIDATE_FIELDS')
            axes[axis] = dict(rz_count=value, status=(MISSING if value is None else MISMATCH if 'RECEIPT_DIFFERS' in reasons
                                                     else CONDITIONAL if reasons else EXACT), reasons=reasons)
        sources.append(dict(kind='DIRECT_COMPILED_DETERMINISTIC_REUSED_' + reused['kind'], cohort='SAVED_T0.8_COMMON',
            n=1, carrier=dict(commit=CARRIER, path=LONG + 'reused_compiled_prices.json.gz', key=c['id']),
            reuse_provenance=reused.get('reuse_provenance'), result_sha256=reused.get('result_sha256'), axes=axes))
    return sources


def _best(sources, axis):
    order = [EXACT, CONDITIONAL, MISMATCH, FAILED, MISSING]
    statuses = [s['axes'][axis]['status'] for s in sources]
    return min(statuses, key=order.index) if statuses else MISSING


def classify(c, k):
    reasons = check_physical(c, k)
    fact = D.FORMULAS.get(c['formula'], {})
    row = dict(system=c['system'], formula=c['formula'], formula_id=fact.get('formula_id'),
               physical_formula_label=fact.get('physical_formula_label', fact.get('formula_id')),
               T=c['T'], q_PF=c['q'], actual_width=c['delta'], prefix=c['prefix'], scope='FULL_H_ONE_BODY_PLUS_ALL_FRAGMENTS',
               physical_candidate_id=c['id'], physical_candidate=deepcopy(c), physical_reasons=reasons,
               binding=dict(status=EXACT if not reasons else MISMATCH, source=dict(commit=D.BASE, path=LONG + 'scientific_provenance.json')))
    if reasons:
        row.update(signal=dict(status=MISMATCH), reference=dict(status=MISMATCH), numerical=dict(status=MISMATCH),
                   prices=[], X=MISMATCH, Y=MISMATCH, group='PHYSICAL_MISMATCH', missing=['FULL_PF_PHYSICAL_MATCH'])
        return row
    signal, sig = _signal(c, k)
    prices = _prices(c, k)
    X, Y = _best(prices, 'X'), _best(prices, 'Y')
    priced = X in (EXACT, CONDITIONAL) and Y in (EXACT, CONDITIONAL)
    has_signal = signal['status'] in (EXACT, CONDITIONAL)
    row.update(signal=signal, reference=_reference(c, k, sig), numerical=_numerical(c, k), prices=prices, X=X, Y=Y,
        group=('SIGNAL_AND_XY_PRICE' if has_signal and priced else 'SIGNAL_ONLY_PRICE_MISSING' if has_signal
               else 'PRICE_ONLY_SIGNAL_MISSING' if priced else 'ALL_MISSING'),
        missing=(['RMSE_SUFFICIENT_N_AND_GLOBAL_CONTRACT'] + ([] if has_signal else ['SIGNAL'])
                 + ([] if X in (EXACT, CONDITIONAL) else ['X_PRICE']) + ([] if Y in (EXACT, CONDITIONAL) else ['Y_PRICE'])))
    return row


def other_saved_series(read_json=None):
    """H6 Full m5_best delta 1.6 prices of the fixed-R price study (direct and Affine model)."""
    from . import h6_df19_price_provider as h6
    provider = h6.H6AffinePriceProvider(read_json or p.SavedArtifacts(D.ROOT).read)
    rows = []
    for r in provider._records:
        c = r['candidate']
        if c['method'] != 'FULL':
            continue
        is_model = r['kind'] == 'model_prediction'
        rows.append(dict(system=c['system'], formula=c['formula'], T=c['T'], q_PF=c['q'], prefix=c['prefix'],
            physical_candidate_id=c['id'], axis=r['axis'], kind='MODEL_PREDICTION' if is_model else 'DIRECT_COMPILED_DETERMINISTIC',
            rz_count=r['value'], record_role=r['record_role'], source=deepcopy(r['source']),
            status=('MODEL_NOT_DIRECT_PRICE' if is_model else CONDITIONAL),
            reasons=(['AFFINE_MODEL_ONLY_QUALIFIED_AT_SAVED_Q4_DOMAIN'] if is_model else
                     ['SEPARATE_FIXED_R_PRICE_STUDY_CONFIRM_SAME_NATIVE_COMPOSE_AS_LONG_TIME_PATH'])))
    return dict(series='H6_DF19_PRICE_V2_FULL19_M5_BEST_DELTA1.6', provider=dict(release=h6.RELEASE, frozen=h6.FROZEN_SOURCE,
                v1=h6.V1, provider_SOURCE=h6.PROVIDER_SOURCE), compiler=h6.PROFILE['compiler'], rows=rows)


def coverage(k):
    rows = [classify(c, k) for c in full_candidates(k)]
    def count(field):
        return dict(Counter(r[field] if isinstance(r[field], str) else r[field]['status'] for r in rows))
    return dict(status='FULL_PF_SAVED_EVIDENCE_COVERAGE_METADATA_ONLY', carrier=CARRIER, carrier_sha=k['sha'],
        rows=rows, totals=dict(candidates=len(rows), groups=dict(Counter(r['group'] for r in rows)),
            signal=count('signal'), reference=count('reference'), X=count('X'), Y=count('Y')),
        checked_scope=['pinned long-time carriers at ' + CARRIER, 'pf_common source_input_identity',
                       'private H4/H6_context.json (read-only)' if k['context'] else 'private context NOT read'],
        rmse_policy_qualified=False, values_recomputed=False, new_compile=0, mandatory_stop=True)


def for_packet(packet, cov):
    """Join a D bind() packet to saved coverage by physical candidate ID; V2 IDs are copied, not created."""
    rows = {r['physical_candidate_id']: r for r in cov['rows']}
    out = []
    for mapping, phys in zip(packet['identity_map'], packet['physical']['rounds']):
        r = rows.get(phys['physical_candidate_id'])
        out.append(dict(m=phys['m'], physical_candidate_id=phys['physical_candidate_id'],
            V2_configuration_id=mapping['V2_configuration_id'], V2_candidate_id=mapping['V2_candidate_id'],
            V2_formula_id=mapping.get('V2_formula_id'), V2_resolution=mapping.get('V2_resolution'), b_commit=mapping.get('b_commit'),
            saved=None if r is None else dict(group=r['group'], signal=r['signal']['status'], reference=r['reference']['status'],
                                              X=r['X'], Y=r['Y']),
            group=r['group'] if r else 'OUTSIDE_SAVED_GRID_ALL_MISSING'))
    return out

"""Stage saved deterministic-PF round means of the price shortlist as B packet means (no science).

Six candidates (Full PF and Truncated B0I p7; S2 16h/7, Yoshida4 16h/3, M5 16h). For each,
the canonical configuration is regenerated with the published generators (T2 recipe) and
planned through the existing C baseline path (`I.plan` -> `C.prepare`). Saved current-SOURCE
direct means (A H_eff pilot/scan/follow-up, T2 two-time checks; class S2 in A's coverage
c212cee85) are wrapped in the baseline record shape and passed through the unchanged
`C.b_mean`; the provider binding is added exactly as `I.produce_signals` does. Every other
round stays `C.unavailable` (S2L: legacy mean awaiting qualification; S3: no saved mean).

Rules kept: raw = corrected = saved direct_PF_mean, B = 1, logB = 0, deterministic law,
reference = saved pair (bba15083) at that round, scalar rule bound (not recomputed), saved
values unchanged. No mean, spectrum, power, gate rerun, bias or allowance is computed; the
staged packet is incomplete and `require_complete` refuses it as a risk input.
"""
from copy import deepcopy
from fractions import Fraction
import json
import subprocess
import sys
from pathlib import Path

MODULE = 'src/trottertracks/saved_pf_mean_staging.py'
KIND = 'SAVED_PF_MEAN_STAGING_V1'
COVERAGE = dict(commit='c212cee857d639d2400819bbca3afa1dc052a9f4',
                path='docs/research/a_signal_candidate_identity_coverage_20261114/SAVED_ROUND_AUDIT.json')
MANIFEST = dict(commit='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',
                path='docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
BASE_SPEC = dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',
                 path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json', selector=['spec'])
CELL_PLAN = dict(commit='855573064b2c17cfe20eb94a01ece1a554141b0b',
                 path='docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json')
PHYSICS_REF = dict(commit='bba150834ca30bdb186fbfd58c55a8d51177b855',
                   path='docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json')
PREFIX7_CONTEXT = dict(path='/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/H4_context.json',
                       sha256='8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f',
                       selector=['prefixes', '7', 'scalar'], manifest_selector=['shared', 'prefix7_scalar'])
H = Fraction(.1)
CANDIDATES = (
    dict(label='FULL_PF_S2_16h_over_7', method='FULL', formula='2nd', width=16 * H / 7, coverage='FULL_PF S2_16h_over_7', id_prefix='11cf8ca6'),
    dict(label='FULL_PF_Y4_16h_over_3', method='FULL', formula='4th', width=16 * H / 3, coverage='FULL_PF Y4_16h_over_3', id_prefix='a799697f'),
    dict(label='FULL_PF_M5_16h', method='FULL', formula='4th(m5_best)', width=16 * H, coverage='FULL_PF M5_16h', id_prefix='70f7b317'),
    dict(label='B0I_P7_S2_16h_over_7', method='B0I', formula='2nd', width=16 * H / 7, coverage='TRUNCATED_B0I_P7 S2_16h_over_7', id_prefix='a6e55777'),
    dict(label='B0I_P7_Y4_16h_over_3', method='B0I', formula='4th', width=16 * H / 3, coverage='TRUNCATED_B0I_P7 Y4_16h_over_3', id_prefix='defc5e88'),
    dict(label='B0I_P7_M5_16h', method='B0I', formula='4th(m5_best)', width=16 * H, coverage='TRUNCATED_B0I_P7 M5_16h', id_prefix='424eecbf'),
)
# Current-SOURCE saved direct means (class S2). Track A and screen means are legacy (S2L).
CURRENT = {'FULL': ('A_FOLLOWUP_DIRECT_MEAN', 'A_HEFF_SCAN_DIRECT_MEAN', 'A_HEFF_PILOT_DIRECT_MEAN'),
           'B0I': ('T2_TWO_TIME_DIRECT_MEAN',)}
LEGACY = ('TRACK_A_LONG_TIME_MEAN', 'TRACK_A_PF_COMMON_T08_MEAN', 'SCREEN_RUN02_M6_RUN_LEVEL_STOP')
READY = 'SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED'
S2L = 'STAGED_S2L_LEGACY_MEAN_QUALIFICATION_PENDING'
S3 = 'STAGED_S3_NO_SAVED_MEAN_NEW_MEAN_REQUIRED'
SCALAR_RULE = {'FULL': dict(quantity='HAMILTONIAN_CONSTANT', application='ONCE_AT_TOTAL_T'),
               'B0I': dict(quantity='HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY_P7', application='ONCE_AT_TOTAL_T')}


def git_json(root, d):
    out = json.loads(subprocess.check_output(['git', 'show', d['commit'] + ':' + d['path']], cwd=root))
    for key in d.get('selector', ()):
        out = out[key]
    return out


def enc(v):
    return dict(numerator=v.numerator, denominator=v.denominator)


def configuration(root, cand, api, C):
    """T2 recipe (5fb2c32e scripts/run_h4_truncated_t2.py plan_cells) with the LD1 base spec."""
    spec = api.D.full_pf_spec(git_json(root, BASE_SPEC), formula=cand['formula'], delta_pf=enc(cand['width']), M=12, rank=10)
    if cand['method'] == 'B0I':
        spec['L_D'] = 7
        spec['split'].update(L_D=7, deterministic_scope='TRUNCATED_H', original_PF_method='B0I')
        spec['split']['retained_hamiltonian'].update(construction='B0I', prefix=7)
    config = C.configure(root, spec, api=api)
    planned = {x['configuration_id'] for x in git_json(root, CELL_PLAN)
               if x['route'] == ('FULL_PF' if cand['method'] == 'FULL' else 'TRUNCATED_PF') and x['formula'] == cand['formula']
               and Fraction(x['width_over_h']) == cand['width'] / H}
    if not config['id'].startswith(cand['id_prefix']) or planned != {config['id']}:
        raise ValueError('Regenerated configuration differs from the price shortlist: ' + cand['label'])
    return config


def references(root, config, binding):
    return [dict(m=r['m'], T=deepcopy(r['T']), binding=deepcopy(binding),
                 source=dict(PHYSICS_REF, selector=['rounds', r['m'], 'diagnostics', 'reference_pair']))
            for r in config['rounds']]


def coverage_rounds(root, cand):
    rows = [r for r in git_json(root, COVERAGE)['rounds'] if r['candidate'] == cand['coverage']]
    if sorted(r['m'] for r in rows) != list(range(13)):
        raise ValueError('Coverage rows incomplete for ' + cand['label'])
    return {r['m']: r for r in rows}


def scalar_binding(root, cand, saved_cell):
    """Bind the scalar rule and the saved component values; never add them here."""
    rule = deepcopy(SCALAR_RULE[cand['method']])
    rule['hamiltonian_constant'] = dict(value=None if saved_cell is None else saved_cell.get('scalar_constant_ha'),
                                        source='saved producer cell result.json' if saved_cell else 'binding constant (producer)')
    if cand['method'] == 'B0I':
        rule['tail_identity'] = dict(PREFIX7_CONTEXT, value=None, value_status='BOUND_BY_LOCATOR_AND_SHA_NOT_COPIED')
    return rule


def _source_of(hit):
    commit, path = hit['source'].split(':', 1)
    return dict(commit=commit, path=path)


def saved_round(root, hit):
    d = _source_of(hit)
    return d, git_json(root, d)


def stage_record(root, cand, _unused, pending_row, hit, pair, staging_source):
    """Wrap one saved current-SOURCE direct mean in the baseline record shape (no computation)."""
    locator, saved = saved_round(root, hit)
    if saved['physical_candidate'] != pending_row['physical_candidate']:
        raise ValueError('Saved mean physical candidate differs from the planned round')
    num = saved.get('numerical') or {}
    if not (saved.get('gate_pass') is True and num.get('gate_pass') is True and num.get('norm_gate_pass') is True
            and num.get('unitarity_gate_pass') is True and num.get('repaired') is False):
        raise ValueError('Saved mean lacks its passed numerical gates')
    scalar_flag = saved.get('scalar_once_at_total_T_in_direct_route', saved.get('scalars_once_at_total_T_in_direct_route'))
    if scalar_flag is not True or saved.get('B', 1.0) != 1.0 or saved.get('raw_equals_corrected_at_B1', True) is not True:
        raise ValueError('Saved mean is not a once-scalar B=1 deterministic direct mean')
    if not 0 <= pair['signal_difference'] <= 1e-10:
        raise ValueError('Saved reference pair gate (NUMERIC_POLICY 1e-10) not satisfied')
    cell_dir = locator['path'].rsplit('/', 1)[0]
    try:
        cell = git_json(root, dict(commit=locator['commit'], path=cell_dir + '/result.json')) if '/cell_' in cell_dir else None
    except subprocess.CalledProcessError:
        cell = None
    mean = deepcopy(saved['direct_PF_mean'])
    numerical = dict(deepcopy(num), gates_passed=True, reference_pair_difference=pair['signal_difference'],
                     backend='SAVED_DIRECT_PF_MEAN', origin_note=hit['note'], scalar_once=True,
                     staging='SAVED_VALUE_BOUND_NO_RECOMPUTATION', certified=False)
    return dict(domain=deepcopy(pending_row['domain']), source=deepcopy(staging_source), status='NUMERICAL_MEAN_AVAILABLE',
                signal=dict(raw=mean, corrected=deepcopy(mean), reference=deepcopy(pair['reference']), log_B=0.0, B=1.0,
                            attenuation=1.0, total_signed_bias=None),
                numerical=numerical,
                input_source=dict(saved_mean=locator, origin_note=hit['note'],
                                  origin_cell=None if cell is None else dict(label=cell.get('label'), source=cell.get('source'),
                                                                             status=cell.get('status')),
                                  origin_round_m=saved.get('m'), physical_round_shared_with_origin_cell=True,
                                  scalar=scalar_binding(root, cand, cell)))


def plan_request(root, cand, *, I, staging_source):
    """Existing C baseline plan (I.plan -> C.prepare) for the regenerated shortlist configuration."""
    api = I.providers(str(root))
    config = configuration(root, cand, api, I.C)
    manifest = git_json(root, MANIFEST)
    return I.plan(root, config, target=manifest['shared']['target'], signal_source=staging_source,
                  references=references(root, config, manifest['shared']['binding']))


def stage(root, cand, *, I, staging_source):
    """Staged (incomplete) B packet + per-round state for one shortlist candidate."""
    C = I.C; api = I.providers(str(root))
    manifest = git_json(root, MANIFEST)
    request = plan_request(root, cand, I=I, staging_source=staging_source)
    config = request['configuration']
    sys.path.insert(0, str(Path(root) / 'scripts'))
    import run_truncated_fullh_signals as TR   # existing saved reference-pair reader (5fb2c32e caller)
    reader = TR.reference_reader(root, manifest['shared'], api)
    cov = coverage_rounds(root, cand)
    means, states = [], []
    for req in request['pending']['requests']:
        m = req['m']; row = cov[m]
        current = [h for h in row['hits'] if h['note'] in CURRENT[cand['method']]]
        legacy = [h for h in row['hits'] if h['note'] in LEGACY]
        if row['status'].startswith('S2_') and current:
            hit = sorted(current, key=lambda h: CURRENT[cand['method']].index(h['note']))[0]
            pair = reader(deepcopy(req['domain']['reference']))
            record = stage_record(root, cand, None, req, hit, pair, staging_source)
            mean = C.b_mean(req, record)
            mean['provider_binding'] = dict(physical_context=I.price_context(request, m), numerical_source=deepcopy(record['source']),
                                            input_source=deepcopy(record['input_source']),
                                            reference_source=deepcopy(record['domain']['reference']))
            state = 'READY_SAVED_S2_MEAN_BOUND'
        elif row['status'].startswith('S2L_'):
            mean = C.unavailable(req, S2L); state = 'S2L_QUALIFICATION_PENDING'
        else:
            mean = C.unavailable(req, S3); state = 'S3_MISSING_NEW_MEAN_REQUIRED'
        means.append(mean)
        states.append(dict(m=m, state=state, physical_id=req['physical_candidate']['id'],
                           canonical_id=req['identity']['candidate']['id'], q_PF=req['q_PF'],
                           actual_delta_pf=deepcopy(config['rounds'][m]['actual_delta_pf']),
                           coverage_status=row['status'], saved_sources=[h['source'] + ' [' + h['note'] + ']' for h in row['hits']],
                           legacy_sources=[h['source'] for h in legacy]))
    packet = dict(configuration_id=config['id'], target=deepcopy(request['target']), source=deepcopy(staging_source), means=means)
    return dict(candidate=cand['label'], method=cand['method'], formula=cand['formula'], width_over_h=str(cand['width'] / H),
                configuration_id=config['id'], packet=packet, rounds=states,
                complete=all(s['state'] == 'READY_SAVED_S2_MEAN_BOUND' for s in states), risk_input_allowed=False)


def require_complete(packet):
    """Gate used before any risk/N hand-off: all 13 rounds must carry a bound available mean."""
    if [m['m'] for m in packet['means']] != list(range(13)) or any(
            m.get('raw') is None or m.get('status') != READY or not m.get('provider_binding') for m in packet['means']):
        raise ValueError('Incomplete staged signal packet: not a risk/N input until all 13 rounds are available')
    return packet

#!/usr/bin/env python3
"""PR-Pauli L_D candidates: saved-evidence table for the split comparison plan (metadata only).

Reads fixed Git objects and one saved input context (read-only, sha256-checked). No sampling,
signal generation, QPY read, build, compile, fit, price prediction, allocation or Q3.
Writes docs/research/a_pauli_ld_split_plan_20261116/CANDIDATES.{json,csv}.
"""
from fractions import Fraction
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/research/a_pauli_ld_split_plan_20261116'

SRC = dict(
    context_index=('7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
                   'artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json'),
    p0_summary=('4220f3006d0d8d413e195ed948e250578513b557',
                'docs/research/df_head_pauli_tail_p0_results_20261114/PREFIX_SUMMARY.csv'),
    signals=('714247619afb4714fd663eb15c0470ef32bdc7af',
             'docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json'),
    q3_vp=('50be68b803a90fd893a1499ebce74e033225144d', 'docs/research/pauli_q3_three_quarter_20261115/SUMMARY.json'),
    q3_v0=('caf1417ef20d0a5ab2030a2d957ee4f09443be43', 'docs/research/pauli_total_q3_accelerated_results_20261114/SUMMARY.json'),
    realloc=('738df226232e539cc084dfad2aed2feae82f3073', 'docs/research/pauli_ld2_reallocation_q3_results_20261115/SUMMARY.json'),
    tref=('a7eceecba5f1d2172682a4e5cc6f1573568debd0', 'docs/research/pauli_ld2_tref_20261115/RESULT.json'),
    ld1_b0=('00f74f86d16cc6878e1d4c5977eb9abd9ce9c076', 'docs/research/d_pauli_ld1_m0_pilot_results_20261114/run02/PRICES.json'),
    ld2_w=('6c4709f85d0f28ce7c6957e9a70c85e378d8cad0', 'docs/research/d_pauli_ld2_b1_w_results_20261115/run01/PRICES.json'),
    s0_edges=('8fbda2ef0f51ea7c31fce5b59f8f0f8fff891f0b', 'docs/research/a_pauli_ld2_step_audit_20261115/run02/EDGES.json'),
    s0_stats=('8fbda2ef0f51ea7c31fce5b59f8f0f8fff891f0b', 'docs/research/a_pauli_ld2_step_audit_20261115/run02/STEP_STATS.csv'),
    s0p_stats=('2d3aa21f58864f186236bac677a232e61e31b26c',
               'docs/research/a_pauli_ld2_step_audit_20261115/s0plus_m9/run03/STEP_STATS_M9.csv'),
    s0_blocks=('7be3bf24423295bb70422641247068c198e03c81',
               'docs/research/a_pauli_ld2_step_audit_20261115/run01/DIAGNOSTIC_P2.txt'),
    prdf_plan=('f42f6e9b9', 'docs/research/d_pr_df_ld_split_cost_plan_20261115/CANDIDATES.csv'),
)
CONTEXT = Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/H4_context.json')
FRAGMENTS = 10
VP = [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9]


def blob(key):
    commit, path = SRC[key]
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{path}'], check=True, capture_output=True).stdout


def loc(key, selector=None):
    commit, path = SRC[key]
    out = dict(commit=commit, path=path)
    if selector is not None:
        out['selector'] = selector
    return out


def show(x):
    if isinstance(x, dict) and 'numerator' in x:
        return '%.12e' % float(Fraction(x['numerator'], x['denominator']))
    return str(x)


def main():
    checks = []

    def check(name, ok, detail=None):
        checks.append(dict(check=name, status='PASS' if ok else 'FAIL', detail=detail))
        if not ok:
            raise SystemExit('FAIL ' + name + ' ' + json.dumps(detail, default=str))

    # saved DF tail descriptors (lambda_R, scalar) of the current H4 DF10 snapshot
    index = json.loads(blob('context_index'))
    want = index['T20_reference_files']['H4']
    raw = CONTEXT.read_bytes()
    check('context_sha256', hashlib.sha256(raw).hexdigest() == want['sha256'], want)
    context = json.loads(raw)
    df = {int(k): v for k, v in context['prefixes'].items()}
    p0 = {int(r['prefix']): r for r in csv.DictReader(io.StringIO(blob('p0_summary').decode()))}
    for p in (1, 2):
        check(f'p0_lambda_R_equals_context_LD{p}', float(p0[p]['saved_DF_lambda_R']) == df[p]['lambda_R'])
        check(f'p0_scalar_equals_context_LD{p}', float(p0[p]['saved_s_R']) == df[p]['scalar'])
    signals = json.loads(blob('signals'))
    packets = {p['prefix']: p for p in signals['packets']}
    vp = {r['label']: r for r in json.loads(blob('q3_vp'))['rows']}
    v0 = json.loads(blob('q3_v0'))
    realloc = {r['proposal']: r for r in json.loads(blob('realloc'))['rows']}
    tref = json.loads(blob('tref'))
    ld1_b0 = json.loads(blob('ld1_b0'))
    ld2_w = json.loads(blob('ld2_w'))
    edges = json.loads(blob('s0_edges'))
    s0_stats = list(csv.DictReader(io.StringIO(blob('s0_stats').decode())))
    s0p_stats = list(csv.DictReader(io.StringIO(blob('s0p_stats').decode())))
    blocks_line = next(l for l in blob('s0_blocks').decode().splitlines() if l.startswith('blocks per step'))
    prdf = list(csv.DictReader(io.StringIO(blob('prdf_plan').decode())))

    rounds = {}
    for p, pk in packets.items():
        cfg = pk['configuration']
        lam = cfg['spec']['split']['lambda_R']
        lam_p = Fraction(lam['numerator'], lam['denominator'])
        check(f'packet_lambda_P_equals_P0_LD{p}', float(lam_p) == float(p0[p]['lambda_P']))
        rs = []
        for r in pk['rounds']:
            c = r['identity']['candidate']
            rs.append(dict(m=r['m'], q=c['q'], R=c['R_outer'], events=c['event_count']))
        check(f'packet_13_rounds_LD{p}', [x['m'] for x in rs] == list(range(13)))
        rounds[p] = dict(configuration_id=cfg['id'], lambda_P=float(lam_p), rounds=rs,
                         kappa=cfg['spec']['kappa'], formula=cfg['spec']['formula'], delta=cfg['spec']['delta_pf'],
                         K=cfg['spec']['finite_K'], h=cfg['spec']['h'], M=cfg['spec']['M'])
    check('same_kappa_LD1_LD2', rounds[1]['kappa'] == rounds[2]['kappa'])
    ratio_events = [Fraction(a['events'], b['events']) for a, b in zip(rounds[1]['rounds'], rounds[2]['rounds'])]
    lam_ratio_sq = (rounds[1]['lambda_P'] / rounds[2]['lambda_P']) ** 2

    # allocation-weighted event positions (metadata; both axes)
    def weighted_events(p, N):
        return 2 * sum(n * r['events'] for n, r in zip(N, rounds[p]['rounds']))

    stat = {(r['R'], r['context']): r for r in s0_stats}
    m9 = next(r for r in s0p_stats if r['path_id'] == 'POOLED_2_PATHS')
    rows = []
    for L in range(FRAGMENTS + 1):
        row = dict(L_D=L, head=('one-body' + (f' + fragments 0..{L - 1}' if L else '')),
                   tail=(f'fragments {L}..{FRAGMENTS - 1}' if L < FRAGMENTS else 'empty'),
                   head_fragment_blocks=L, lambda_R_DF=df.get(L, {}).get('lambda_R'), scalar_s_R=df.get(L, {}).get('scalar'),
                   lambda_R_source=(dict(index=loc('context_index'), file=str(CONTEXT), sha256=want['sha256'],
                                         selector=['prefixes', str(L)]) if L in df else None),
                   lambda_P=None, dictionary_sha256=None, configuration_id=None, formula='S2', delta_pf=0.1, K=2,
                   kappa='original exact kappa (common)', schedule=None, R_m=None,
                   signals=None, q3=[], direct_prices=[], model_reference_price=None, status=None, missing=[])
        if L in (1, 2):
            rd = rounds[L]
            row.update(lambda_P=rd['lambda_P'], dictionary_sha256=p0[L]['dictionary_sha256'],
                       nonidentity_terms=int(p0[L]['nonidentity_terms']), configuration_id=rd['configuration_id'],
                       schedule=loc('signals', ['packets', L - 1, 'rounds']), R_m=[r['R'] for r in rd['rounds']],
                       events_m=[r['events'] for r in rd['rounds']],
                       weighted_event_positions_VP=weighted_events(L, VP),
                       signals=dict(rounds=13, locator=loc('signals', ['packets', L - 1]),
                                    status='S1_SAVED_13ROUND_PACKET_LOCAL_FINITE_MEANS'),
                       status='SAVED_CANDIDATE')
            c0 = next(c for c in v0['cases'] if c['label'] == 'LD%d' % L)
            row['q3'].append(dict(allocation='V0 [64,60,56,52,47,43,38,34,30,25,21,17,12]', status=c0['status'],
                                  MSE_upper=show(c0['MSE_upper']), locator=loc('q3_v0', ['cases', L - 1])))
            r = vp['LD%d' % L]
            row['q3'].append(dict(allocation='VP ' + json.dumps(VP), allocation_id=r['allocation_id'], status=r['status'],
                                  MSE_upper=r['MSE_upper'], locator=loc('q3_vp', ['rows', L - 1])))
        if L == 1:
            row['direct_prices'] = [dict(m=0, axes='X/Y', paths=1, rz=[int(x['metrics']['rz_count']) for x in ld1_b0],
                                         role='WIRING_SINGLE_PATH', locator=loc('ld1_b0'))]
            row['model_reference_price'] = None
            row['missing'] = ['LD1_OWN_26_AXIS_REFERENCE_PRICE_RULE (no LD1 calibration/model)',
                              'B allocation search + Q3 for LD1-own allocations (signals and VP Q3 are saved)']
        if L == 2:
            for name in ('FPN-B', 'OWN-B'):
                r = realloc[name]
                row['q3'].append(dict(allocation=name + ' ' + json.dumps(r['allocation']), status=r['status'],
                                      T_model=r['T_display'], locator=loc('realloc')))
            row['direct_prices'] = [
                dict(m=0, axes='X/Y', paths=1, rz=[int(x['metrics']['rz_count']) for x in ld2_w], role='WIRING_SINGLE_PATH',
                     locator=loc('ld2_w')),
                dict(cells='S66,S526,S4205,Q8R66,P2R66', compiles=18, role='CALIBRATION_NON_CONFIGURATION',
                     locator=dict(commit='5dac74959ed9ea49465d393fc792ef5b4727edba',
                                  path='docs/research/d_pauli_ld2_b1_cal_results_20261115/C_HANDOFF.json')),
                dict(m=[5, 6], paths=[2, 1], compiles=6, role='INDEPENDENT_HOLDOUT_DESCRIPTIVE',
                     locator=dict(commit='705f59ea863e1b9fe90e023f1cb1057d00366b16',
                                  path='docs/research/d_pauli_ld2_b1_holdout_results_20261115/COMPARISON.json')),
                dict(m=9, paths=2, compiles=4, role='INDEPENDENT_HOLDOUT_DESCRIPTIVE_C2',
                     locator=dict(commit='cb034e547c444ea67a4f4f90a8534c13f4db4451',
                                  path='docs/research/d_pauli_ld2_m9_longpoint_results_20261115/HANDOFF.md'))]
            row['model_reference_price'] = dict(
                T_ref_VP=tref.get('T_ref_display') or tref.get('T_ref'), locator=loc('tref'),
                frozen_model=dict(commit='b74e4e536392c2bbe94cd371436615fc56c0564f',
                                  path='docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json'),
                structure=dict(decomposition='RZ = 308 + 1900 q + sum(step bodies)', verified_q='1,2,8,32,64 (13 paths), 512 (m9 path0)',
                               locators=[loc('s0_edges'), dict(commit=SRC['s0p_stats'][0],
                                                              path='docs/research/a_pauli_ld2_step_audit_20261115/s0plus_m9/REPORT_S0PLUS.md')]),
                scope='Diagnostic model reference; m10-m12 (q>=1024) are extrapolations; not E[C]')
            row['missing'] = ['none for B allocation search (done: FPN-B fail, OWN-B pass at T=2.4428e9)',
                              'm10-m12 price basis beyond q=512 remains extrapolation']
        if L in (3, 5, 7, 9):
            row['status'] = 'CONSTRUCTIBLE_FROM_CURRENT_SOURCE_WITH_SAVED_DF_DESCRIPTOR'
            row['missing'] = ['P0 Pauli dictionary for this prefix (lambda_P, s_P check, operator/sector gates; code fixes prefixes (1,2))',
                              'configuration + common-kappa 13-round schedule (R_m depends on lambda_P)',
                              '13-round signals (provider/adapter fix prefixes (1,2)) and B total view',
                              'candidate-own 26-axis reference price rule', 'Q3 for VP and candidate-own allocations']
        if L in (4, 6, 8):
            row['status'] = 'DF_DESCRIPTOR_NOT_IN_SAVED_CONTEXT'
            row['missing'] = ['saved lambda_R/scalar descriptor with provenance for this prefix', 'then everything listed for L_D=3']
        if L == 0:
            row['status'] = 'ADAPTER_BOUNDARY_ONE_BODY_HEAD_ONLY'
            row['missing'] = ['current signal context requires 1 <= prefix < rank', 'DF descriptor for L_D=0', 'then everything listed for L_D=3']
        if L == FRAGMENTS:
            row['status'] = 'EMPTY_TAIL_ENDPOINT_NOT_PR'
            row['missing'] = ['PR runtime requires positive tail lambda; Full PF is the separate deterministic endpoint']
        rows.append(row)

    # structural quantities that bear on the comparison (saved values only; no prediction)
    lam = {L: df[L]['lambda_R'] for L in df}
    structure = dict(
        LD1_vs_LD2=dict(
            lambda_P_ratio_squared=lam_ratio_sq,
            event_ratio_per_round=[float(x) for x in ratio_events],
            weighted_event_positions_VP=dict(LD1=weighted_events(1, VP), LD2=weighted_events(2, VP)),
            q3_VP_MSE_upper=dict(LD1=vp['LD1']['MSE_upper'], LD2=vp['LD2']['MSE_upper']),
            q3_V0_MSE_upper={c['label']: show(c['MSE_upper']) for c in v0['cases']},
            m0_single_path_rz=dict(LD1=int(ld1_b0[0]['metrics']['rz_count']), LD2=int(ld2_w[0]['metrics']['rz_count']),
                                   note='q=1; LD1 R=3 (head one-body+f0), LD2 R=2 (head one-body+f0+f1); single paths')),
        lambda_P_over_lambda_R=dict(LD1=rounds[1]['lambda_P'] / lam[1], LD2=rounds[2]['lambda_P'] / lam[2]),
        lambda_R_ratio_to_LD2_squared={str(L): (lam[L] / lam[2]) ** 2 for L in sorted(lam)},
        LD2_step_parts=dict(deterministic_per_step=1900, intercept=308,
                            later_body_mean=dict(R33=float(stat[('33', 'LATER')]['mean_body_rz']),
                                                 R66=float(stat[('66', 'LATER')]['mean_body_rz']),
                                                 R526=float(m9['mean'])),
                            first_body_mean_R4205=float(stat[('4205', 'FIRST')]['mean_body_rz']),
                            D_region_pairs_per_block=blocks_line),
        PR_DF_split_plan=dict(locator=loc('prdf_plan'), LD3_status=next(r['definition_status'] for r in prdf if r['L_D'] == '3')))
    check('D_region_blocks_33_36_8', '[33, 36, 8]' in blocks_line, blocks_line)
    result = dict(schema='A_PAULI_LD_SPLIT_CANDIDATES_V1', status='METADATA_ONLY_NO_NEW_SCIENCE',
                  definition=dict(L_D='number of two-body DF fragments kept in the deterministic head, in saved fragment order; '
                                      'the one-body block is always in the head and is not counted',
                                  head='one-body + fragments 0..L_D-1 (native S2: U(1b) U(f0).. [tail] ..D(f0) D(1b))',
                                  tail='fragments L_D..9 with saved weighted-Hermitian completion, Jordan-Wigner, global-collected '
                                       'Pauli strings; identity extracted as s_R (scalar once at total T); RTE events from |c_j|/lambda_P, K=2',
                                  not_PR_DF='PR-DF uses the same L_D count and head, but its tail events are DF-fragment components '
                                            '(DF basis + controlled ZZ/Z), lambda = lambda_R(DF); R schedules and prices differ'),
                  sources={k: loc(k) for k in SRC}, rows=rows, structure=structure, checks=checks,
                  new_science=0, compiles=0, price_predictions=0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'CANDIDATES.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    fields = ['L_D', 'status', 'head', 'tail', 'lambda_R_DF', 'scalar_s_R', 'lambda_P', 'configuration_id', 'R_m12',
              'signals_13', 'q3', 'direct_prices', 'model_reference', 'missing']
    with open(OUT / 'CANDIDATES.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(dict(L_D=r['L_D'], status=r['status'], head=r['head'], tail=r['tail'], lambda_R_DF=r['lambda_R_DF'],
                            scalar_s_R=r['scalar_s_R'], lambda_P=r['lambda_P'], configuration_id=r['configuration_id'],
                            R_m12=(r['R_m'] or [None])[-1], signals_13=bool(r['signals']),
                            q3='; '.join(f"{q['allocation'].split(' ')[0]}:{q['status']}" for q in r['q3']),
                            direct_prices='; '.join(d.get('role', '') for d in r['direct_prices']),
                            model_reference=bool(r['model_reference_price']), missing=' | '.join(r['missing'])))
    print(json.dumps(dict(checks=len(checks), structure=structure), indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()

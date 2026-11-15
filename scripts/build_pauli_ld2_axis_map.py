#!/usr/bin/env python3
"""PR-Pauli LD2: 26-axis map from the Q3 candidate to saved price evidence (metadata only).

Planning aid for docs/research/a_pauli_ld2_expected_price_plan_20261115/. It reads fixed Git
objects only, checks physical identities and schedule facts, and tabulates the fixed-shot
weights of the Q3 allocation. It does not read private QPYs, compile, sample, fit, predict a
price, or compute E[C]/CI/G. Published observed RZ values are transcribed with their locators.
"""
from fractions import Fraction
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/research/a_pauli_ld2_expected_price_plan_20261115'

CONFIG = 'd7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3'
TOTAL_VIEW = '285bf4812e07a519f266ed88b83293bc4675493c5a9cc03985a6312363c6b8f0'
VECTOR = [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9]
STEP_TIME = Fraction(0.1)  # binary64 0.1 exactly
SRC = dict(
    alignment=('0b4b2d941cd669aaa45231e1236bc2d3158a9d12',
               'docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json'),
    q3_ld2=('50be68b803a90fd893a1499ebce74e033225144d',
            'docs/research/pauli_q3_three_quarter_20261115/run01/RESULT_LD2.json'),
    cal=('5dac74959ed9ea49465d393fc792ef5b4727edba',
         'docs/research/d_pauli_ld2_b1_cal_results_20261115/C_HANDOFF.json'),
    cal_obs=('5dac74959ed9ea49465d393fc792ef5b4727edba',
             'docs/research/d_pauli_ld2_b1_cal_results_20261115/OBSERVATIONS.csv'),
    frozen=('b74e4e536392c2bbe94cd371436615fc56c0564f',
            'docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json'),
    holdout=('705f59ea863e1b9fe90e023f1cb1057d00366b16',
             'docs/research/d_pauli_ld2_b1_holdout_results_20261115/RECEIPT_ADMISSION.json'),
    wiring=('6c4709f85d0f28ce7c6957e9a70c85e378d8cad0',
            'docs/research/d_pauli_ld2_b1_w_results_20261115/run01/PRICES.json'),
    m9=('3f6e6604a4877dc882fd1ee7538a854464160a5b',
        'docs/research/d_pauli_ld2_m9_longpoint_results_20261115/PREDICTIONS.json'),
)
# C2 m9 publication seen at the pre-publication recheck (commit subjects and file names only;
# A does not open the observed values and this plan does not use them).
M9_PUBLICATION = dict(raw_and_receipts='a7fdd200da8d499e93965d63b5b2e5b01e0b8940',
                      canonical_binding='c9986c82f866bfa57a7142124746f89fc42f9838',
                      directory='docs/research/d_pauli_ld2_m9_longpoint_results_20261115/run01/',
                      comparison_published=False)


def blob(key):
    commit, path = SRC[key]
    return subprocess.run(['git', '-C', str(ROOT), 'show', commit + ':' + path],
                          check=True, capture_output=True).stdout.decode()


def frac(x):
    return Fraction(x['numerator'], x['denominator']) if isinstance(x, dict) else Fraction(x)


def ratio(num, den):
    v = Fraction(num, den)
    return dict(numerator=v.numerator, denominator=v.denominator, display_percent='%.4f' % (100 * float(v)))


def locator(key, selector=None):
    commit, path = SRC[key]
    out = dict(commit=commit, path=path)
    if selector is not None:
        out['selector'] = selector
    return out


def main():
    checks = []

    def check(name, ok, detail=None):
        checks.append(dict(check=name, status='PASS' if ok else 'FAIL', detail=detail))
        if not ok:
            raise SystemExit('FAIL ' + name + ' ' + json.dumps(detail))

    alignment = json.loads(blob('alignment'))
    rows = sorted((r for r in alignment['rows'] if r['configuration_id'] == CONFIG),
                  key=lambda r: (r['m'], r['axis']))
    check('alignment_26_rows', len(rows) == 26, len(rows))

    q3 = json.loads(blob('q3_ld2'))['record']
    shots = {s['m']: s for s in q3['allocation']['shots']}
    check('q3_allocation_id', q3['allocation']['allocation_id'] == 'PAULI_LD2_USER_SCALE_3O4_01')
    check('q3_vector_matches_instruction',
          [shots[m]['N_X'] for m in range(13)] == VECTOR and [shots[m]['N_Y'] for m in range(13)] == VECTOR)
    check('q3_total_view_configuration', q3['configuration_id'] == TOTAL_VIEW)
    q3_rounds = {}
    for mean in q3['signal_input']['means']:
        origin = json.dumps(mean['canonical_origin'])
        cand = mean['identity']['candidate']
        q3_rounds[mean['m']] = dict(total_view_candidate=mean['law_binding']['candidate_id'],
                                    origin_text=origin, q=cand['q'], R=cand['R_outer'],
                                    events=cand['event_count'], T=frac(cand['T']), K=cand['K'],
                                    law=mean['law_binding']['definition']['model'])

    holdout = json.loads(blob('holdout'))
    held = {}
    for unit in holdout['units']:
        held[(unit['m'], unit['axis'])] = unit
    wiring = {(r['m'], r['axis']): r for r in json.loads(blob('wiring'))}
    m9 = json.loads(blob('m9'))
    cal = json.loads(blob('cal'))
    cal_obs = list(csv.DictReader(io.StringIO(blob('cal_obs'))))
    frozen = json.loads(blob('frozen'))

    # Step time is bitwise binary64(0.1) for every round and every saved cell (same head angles).
    for m, r in q3_rounds.items():
        check('round_step_time_m%d' % m, Fraction(float(r['T']) / r['q']) == STEP_TIME and r['T'] == STEP_TIME * r['q'])
    for c in cal['cells']:
        check('cal_step_time_' + c['cell_id'], Fraction(float(c['T']) / c['q']) == STEP_TIME)
    for p in frozen['predictions']:
        check('frozen_prediction_step_time_' + p['prediction_id'], Fraction(float(frac(p['T_exact'])) / p['q']) == STEP_TIME)

    # X/Y equality of every published saved path (transcription check, not a new statistic).
    xy = {}
    for o in cal_obs:
        xy.setdefault((o['cell_id'], o['index']), {})[o['axis']] = int(o['rz_count'])
    for unit in holdout['units']:
        for p in unit['paths']:
            xy.setdefault(('M%d_ROUND' % unit['m'], str(p['index'])), {})[p['axis']] = p['compiled_RZ']
    for (m, axis), r in wiring.items():
        xy.setdefault(('W_M0', '0'), {})[axis] = int(r['mean']['rz_count'])
    check('saved_paths_13', len(xy) == 13, sorted(xy))
    check('saved_paths_X_equals_Y', all(v['X'] == v['Y'] for v in xy.values()),
          {'%s/p%s' % k: v for k, v in sorted(xy.items())})

    cal_by_R = {}
    for c in cal['cells']:
        cal_by_R.setdefault(c['R'], []).append(c['cell_id'])

    observed_q = dict(cal=max(c['q'] for c in cal['cells']), holdout=64, m9_pending=512)
    cal_R = sorted({c['R'] for c in cal['cells']})

    sums = dict(shots=0, steps=0, events=0)
    per_round = []
    for m in range(13):
        r = q3_rounds[m]
        N = VECTOR[m]
        per_round.append(dict(m=m, N=N, q=r['q'], R=r['R'], events_per_path=r['events'],
                              shots_both_axes=2 * N, steps_both_axes=2 * N * r['q'],
                              events_both_axes=2 * N * r['events']))
        sums['shots'] += 2 * N
        sums['steps'] += 2 * N * r['q']
        sums['events'] += 2 * N * r['events']
    for pr in per_round:
        pr['step_share'] = ratio(pr['steps_both_axes'], sums['steps'])
        pr['event_share'] = ratio(pr['events_both_axes'], sums['events'])

    out_rows = []
    for row in rows:
        m, axis = row['m'], row['axis']
        r = q3_rounds[m]
        pc = row['physical_candidate']
        check('alignment_schedule_m%d%s' % (m, axis), (pc['q'], pc['R_outer'], pc['event_count']) == (r['q'], r['R'], r['events']))
        check('q3_origin_physical_id_m%d%s' % (m, axis), row['physical_id'] in r['origin_text'])
        check('q3_law_m%d' % m, r['law'] == 'ideal_finite_RTE_law' and r['K'] == 2)
        direct = None
        if (m, axis) in held:
            unit = held[(m, axis)]
            check('holdout_physical_id_m%d%s' % (m, axis), unit['canonical_physical_id'] == row['physical_id'])
            direct = dict(kind='INDEPENDENT_HOLDOUT_FRESH_PATHS', n_paths=len(unit['paths']),
                          cohort=unit['paths'][0]['cohort'],
                          rz_per_path_transcribed=[p['compiled_RZ'] for p in unit['paths']],
                          receipts=[p['receipt'] for p in unit['paths']],
                          comparison='DESCRIPTIVE_COMPLETE_NO_TOLERANCE',
                          comparison_source=dict(commit=SRC['holdout'][0],
                                                 path='docs/research/d_pauli_ld2_b1_holdout_results_20261115/COMPARISON.json'))
        elif (m, axis) in wiring:
            w = wiring[(m, axis)]
            check('wiring_physical_id_m0%s' % axis, w['identity']['candidate']['id'] == row['physical_id'])
            direct = dict(kind='WIRING_SINGLE_PATH', n_paths=w['n_paths'], cohort=w['cohort'],
                          rz_per_path_transcribed=[int(w['mean']['rz_count'])],
                          receipts=[locator('wiring', [0 if axis == 'X' else 1])],
                          comparison=None)
        elif m == 9:
            check('m9_binding_physical_id_%s' % axis, m9['binding']['request']['candidate_id'] == row['physical_id']
                  and m9['binding']['configuration_id'] == CONFIG)
            direct = dict(kind='C2_RAW_RECEIPTS_PUBLISHED_NOT_OPENED_BY_A', n_paths=0,
                          planned_paths=m9['binding']['n_paths'], cohort=m9['binding']['cohort'],
                          predictions_status=m9['status'], predictions_source=locator('m9'),
                          publication=M9_PUBLICATION, rz_per_path_transcribed=[])
        if r['R'] < cal_R[0]:
            r_domain = 'R_BELOW_CAL_RANGE'
        elif r['R'] in cal_by_R:
            r_domain = 'R_EQUALS_CAL_STEP_LAW'
        else:
            r_domain = 'R_INSIDE_CAL_RANGE_NO_SAME_R_CELL'
        if r['q'] <= observed_q['cal']:
            q_domain = 'Q_WITHIN_CAL'
        elif r['q'] <= observed_q['holdout']:
            q_domain = 'Q_WITHIN_HOLDOUT_OBSERVED'
        elif r['q'] < observed_q['m9_pending']:
            q_domain = 'Q_BETWEEN_HOLDOUT_AND_M9'
        elif r['q'] == observed_q['m9_pending']:
            q_domain = 'Q_AT_M9_C2'
        else:
            q_domain = 'Q_BEYOND_M9'
        frozen_pred = [p['prediction_id'] for p in frozen['predictions'] if p['m'] == m and p['axis'] == axis]
        if m == 9:
            frozen_pred = [p['prediction_id'] for p in m9['predictions'] if p['axis'] == axis]
        if direct is None:
            evidence = 'MODEL_APPLICATION_REQUIRED_NO_DIRECT_PATH'
        elif direct['kind'] == 'WIRING_SINGLE_PATH':
            evidence = 'DIRECT_SINGLE_WIRING_PATH_PLUS_MODEL_REQUIRED'
        elif direct['kind'].startswith('C2_'):
            evidence = 'C2_M9_PUBLISHED_UNUSED_HERE_PLUS_MODEL_REQUIRED'
        else:
            evidence = 'DIRECT_FINITE_HOLDOUT_PLUS_MODEL_REQUIRED'
        out_rows.append(dict(
            m=m, axis=axis, physical_id=row['physical_id'], total_view_candidate=r['total_view_candidate'],
            configuration_id=CONFIG, total_view_configuration=TOTAL_VIEW,
            q=r['q'], R=r['R'], events_per_path=r['events'],
            T=dict(numerator=r['T'].numerator, denominator=r['T'].denominator),
            step_time_binary64_0p1=True, N=VECTOR[m],
            same_step_law_cal_cells=cal_by_R.get(r['R'], []),
            R_domain=r_domain, q_domain=q_domain,
            direct_observation=direct, frozen_predictions_existing=frozen_pred,
            evidence_class=evidence,
            law_target='ideal_finite_RTE_law',
            law_expectation_available=False,
            target_law_expected_price_usable=False,
            price_value=None, E_C=None, CI=None, G=None,
            signal_source=row['signal']['packet_source']))

    segments = {}
    for c in cal['cells']:
        if c.get('paired_dependence'):
            continue
        segments.setdefault(c['R'], 0)
        segments[c['R']] += c['n_paths'] * c['q']
    segments[33] = segments.get(33, 0) + 2 * 32
    segments[66] = segments.get(66, 0) + 1 * 64
    segments[2] = segments.get(2, 0) + 1
    saved_step_samples = [dict(R=R, independent_steps_in_saved_paths=n,
                               same_R_rounds=[m for m in range(13) if q3_rounds[m]['R'] == R])
                          for R, n in sorted(segments.items())]

    result = dict(
        schema='A_PAULI_LD2_AXIS_MAP_V1',
        status='METADATA_ONLY_NO_PRICE_VALUES_COMPUTED',
        sources={k: locator(k) for k in SRC},
        candidate=dict(configuration_id=CONFIG, total_view_configuration=TOTAL_VIEW,
                       allocation_id=q3['allocation']['allocation_id'], vector_both_axes=VECTOR,
                       q3_status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', law='ideal_finite_RTE_law', K=2,
                       dictionary_sha256=frozen['scope']['dictionary_identity']['dictionary_sha256']),
        rows=out_rows,
        fixed_shot_weights=dict(per_round=per_round, totals_both_axes=sums,
                                meaning='Coefficient weights of the fixed-shot total for a model linear in [1,q,qR]; not a price'),
        cal_cells=[dict(cell_id=c['cell_id'], q=c['q'], R=c['R'], T=c['T'], n_paths=c['n_paths'],
                        paired=bool(c.get('paired_dependence'))) for c in cal['cells']],
        saved_step_samples=saved_step_samples,
        saved_paths_rz_transcribed={'%s/p%s' % k: v for k, v in sorted(xy.items())},
        checks=checks,
        price_values_computed=0, predictions_computed=0, compiles=0, sampling=0, private_reads=0,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'AXIS_MAP.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    fields = ['m', 'axis', 'physical_id', 'q', 'R', 'events_per_path', 'N', 'same_step_law_cal_cells',
              'R_domain', 'q_domain', 'direct_kind', 'direct_n_paths', 'rz_per_path_transcribed',
              'frozen_predictions_existing', 'evidence_class', 'target_law_expected_price_usable']
    with open(OUT / 'AXIS_MAP.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in out_rows:
            d = r['direct_observation'] or {}
            w.writerow(dict(m=r['m'], axis=r['axis'], physical_id=r['physical_id'], q=r['q'], R=r['R'],
                            events_per_path=r['events_per_path'], N=r['N'],
                            same_step_law_cal_cells='|'.join(r['same_step_law_cal_cells']),
                            R_domain=r['R_domain'], q_domain=r['q_domain'], direct_kind=d.get('kind', ''),
                            direct_n_paths=d.get('n_paths', 0),
                            rz_per_path_transcribed='|'.join(str(x) for x in d.get('rz_per_path_transcribed', [])),
                            frozen_predictions_existing='|'.join(r['frozen_predictions_existing']),
                            evidence_class=r['evidence_class'],
                            target_law_expected_price_usable=r['target_law_expected_price_usable']))
    print(json.dumps(dict(checks=len(checks), failed=sum(c['status'] != 'PASS' for c in checks),
                          totals=sums, segments=saved_step_samples), ensure_ascii=False))
    for pr in per_round:
        print(pr['m'], pr['N'], pr['q'], pr['R'], pr['steps_both_axes'], pr['events_both_axes'],
              pr['step_share']['display_percent'], pr['event_share']['display_percent'])


if __name__ == '__main__':
    main()

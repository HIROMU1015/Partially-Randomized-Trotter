"""Snapshot saved H8 exploration prices; never dispatch scientific workers."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import h8_long_time_adapter as a
from repair_h8_compiler_phase import completed_trials, TARGETS


def summarize(private):
    signal = a.read(private / 'stage_A_combined.json')
    trials = completed_trials(private)
    for cid in TARGETS:
        path = private / 'phase_repair01' / cid / 'result.json'
        if path.exists():
            value = a.read(path)
            key = cid, 0
            a.require(key not in trials and value['status'] == 'PASS'
                      and value['candidate_id'] == cid and value['index'] == 0, 'UNIQUE_REPAIRED_ORIGINAL')
            trials[key] = value
    endpoints = {r['endpoint']['candidate']['id']: r['endpoint'] for r in signal['results']}
    rows = []
    for batch in ('stage_B01', 'stage_B02'):
        scope = a.read(private / 'runs' / batch / 'scope.json')
        a.require(scope['binding'] == signal['binding'] and scope['cohort'] == 'EXPLORATION', 'FIXED_EXPLORATION')
        for candidate in scope['candidates']:
            cid = candidate['id']
            endpoint = endpoints[cid]
            values = sorted((v for (c, _), v in trials.items() if c == cid), key=lambda v: v['index'])
            n = scope['n'] if candidate['method'] == 'PR' else 1
            representative = next((v for v in values if v['index'] == 0), None)
            validated = representative is not None and len(representative['checks']) == 7 and all(
                c['phase_alignment'] is False and 0 <= c['error'] < c['tolerance'] for c in representative['checks'])
            indices = [v['index'] for v in values]
            a.require(set(indices) <= set(range(n)), 'ORIGINAL_INDICES_ONLY')
            stats = a.paired_G(values, endpoint, 'EXPLORATION') if values else None
            rows.append(dict(candidate=candidate, observed_n=len(values), target_n=n,
                missing_indices=sorted(set(range(n)) - set(indices)), representative_PASS=validated,
                complete=validated and indices == list(range(n)),
                status='COMPLETE_EXPLORATION' if validated and indices == list(range(n)) else 'PARTIAL_NOT_VALIDATED',
                N_cos=endpoint['N_cos'], N_sin=endpoint['N_sin'], log_B=endpoint['log_B'],
                mean_C_cos_exact=str(Fraction(sum(v['prices']['cosine']['rz_count'] for v in values), len(values))) if values else None,
                mean_C_sin_exact=str(Fraction(sum(v['prices']['sine']['rz_count'] for v in values), len(values))) if values else None,
                statistics=stats))
    a.require(len(rows) == 19 and sum(r['observed_n'] for r in rows) == len(trials), 'REGISTERED_PRICE_POOL')
    pf = min((r for r in rows if r['candidate']['method'] != 'PR' and r['complete']),
             key=lambda r: Fraction(r['statistics']['mean_G_exact']))
    for row in rows:
        row['observed_mean_over_best_measured_PF_exact'] = (
            str(Fraction(row['statistics']['mean_G_exact']) / Fraction(pf['statistics']['mean_G_exact']))
            if row['statistics'] else None)
    return dict(status='INTERIM_EXPLORATION_ONLY', captured_JST=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        binding=signal['binding'], T=1.6, epsilon=.0005, state_preparation_cost=0,
        cost='G=N_cos*C_cos+N_sin*C_sin; compiled RZ',
        signal_SOURCE=signal['SOURCE'], native_SOURCE=a.read(private / 'config_B.json')['SOURCE'],
        phase_supplement_SOURCE=a.read(private / 'phase_repair01/scope.json')['supplement']['SOURCE'],
        aggregation_script_sha256=a.sha(__file__),
        signal_result_sha256=a.sha(private / 'stage_A_combined.json'),
        stage_A=dict(candidates=len(signal['results']), statuses=dict(Counter(v['endpoint']['status'] for v in signal['results']))),
        stage_B=dict(saved_PASS_trials=len(trials), target_trials=51, complete_candidates=sum(r['complete'] for r in rows),
                     representative_PASS_candidates=sum(r['representative_PASS'] for r in rows), target_candidates=19),
        best_measured_complete_PF=pf['candidate']['id'], rows=rows,
        trial_observations=[dict(candidate_id=v['candidate_id'], index=v['index'],
            C_cos=v['prices']['cosine']['rz_count'], C_sin=v['prices']['sine']['rz_count'],
            original_result_digest=a.digest(v)) for _, v in sorted(trials.items())],
        confirmation='NOT_RUN', population_guarantee=False, global_optimum_claim=False,
        incomplete_mean_policy='DESCRIPTIVE_AVAILABLE_SAMPLES_ONLY_NOT_A_VALIDATED_COMPARISON', new_samples=0)


def write(private):
    result = summarize(private)
    stamp = datetime.fromisoformat(result['captured_JST']).strftime('%Y%m%d_%H%MJST')
    stem = 'interim_' + stamp
    output = a.ROOT / 'artifacts/resource_applicability/track_a_h8_long_time/2026-11-04'
    a.save(output / (stem + '.json'), result)
    fields = ['id', 'method', 'formula', 'prefix', 'q', 'observed_n', 'target_n', 'representative_PASS',
              'status', 'N_cos', 'N_sin', 'mean_C_cos_exact', 'mean_C_sin_exact', 'mean_G_exact',
              'sample_SE_G', 'observed_mean_over_best_measured_PF_exact']
    with (output / (stem + '.csv')).open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result['rows']:
            merged = dict(row, **row['candidate'], **(row['statistics'] or {}))
            writer.writerow({k: merged.get(k) for k in fields})
    best = next(r for r in result['rows'] if r['candidate']['id'] == result['best_measured_complete_PF'])
    selected = [best] + sorted((r for r in result['rows'] if r['candidate']['method'] == 'PR'),
                              key=lambda r: (r['candidate']['formula'], r['candidate']['prefix']))
    lines = ['# H8長時間・高次PR：中間集計', '', result['captured_JST'] + ' の保存済み結果。', '',
        'Linear H8、隣接距離1.00 Å、STO-3G、保存tol-only1e-8 DF28、元state・fragment順序0..27。',
        'T=1.6、ε=0.0005、P=0。PRはR_outer32/K4。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/topologyなし。',
        '費用G=N_cos C_cos+N_sin C_sin。measurement付きfull Hadamard wrapperのRZ数を固定Nで重み付けする。全RPE roundの最終総費用ではない。', '',
        f"Stage A：69候補（適格40、不適格29）。Stage B：保存PASS {result['stage_B']['saved_PASS_trials']}/51標本、代表検証完了 {result['stage_B']['representative_PASS_candidates']}/19候補。", '',
        '| 方式 | prefix | q | 保存標本/予定 | 平均G（RZ） | 最良の測定済みPF比 | 状態 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in selected:
        c = row['candidate']
        label = c['method'] + ' ' + c['formula']
        g = float(Fraction(row['statistics']['mean_G_exact']))
        ratio = float(Fraction(row['observed_mean_over_best_measured_PF_exact']))
        lines.append(f"| {label} | {c['prefix']} | {c['q']} | {row['observed_n']}/{row['target_n']} | {g:.7e} | {ratio:.5f} | {'探索群完了' if row['complete'] else '暫定・代表検証待ち'} |")
    lines += ['', 'PF15候補は全て代表検証済み。測定した範囲で最小はB0+I / m5_best / prefix15 / q1。',
        'PR prefix8の2群は元index0がHOLDとなり、同じ標本を位相補助SOURCEで再検証中。8標本だけの値は利用可能標本の記述値であり、9標本の平均・確定した比較・母集団での優位を意味しない。',
        'PR prefix7は各9標本と代表7検査が完了。m5はq1、S2はq32だが、qだけでなく固定shotと一体compiled価格を含むGで比較する。',
        'sample SEはpaired Gの軸間共分散を含む記述統計で、希少次数の母集団保証や確認cohortの精度を示さない。PFの1回価格には標本SEを作らない。', '',
        f"Signal SOURCE `{result['signal_SOURCE']}`、元native SOURCE `{result['native_SOURCE']}`、位相補助SOURCE `{result['phase_supplement_SOURCE']}`。",
        '元SOURCE・raw/QPY・失敗/HOLD・seed/event列は保全し、稼働中の科学コードは変更していない。新しい標本は作成していない。', '',
        f'[全19候補CSV](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-04/{stem}.csv)、',
        f'[固定N・paired統計・49標本の匿名価格metadata](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-04/{stem}.json)、',
        '[集計コード](../../../scripts/summarize_h8_long_time_progress.py)。', '',
        '次の担当：Codexが稼働中2代表の検査結果を確認し、全51標本が検査を満たした場合に完全な探索集計を行う。Stage Cは未実行で、別途条件と実行許可が必要。', '']
    report = a.ROOT / 'docs/research/h8_long_time_execution_20261103' / ('INTERIM_RESULTS_' + stamp + '.md')
    with report.open('x') as stream:
        stream.write('\n'.join(lines))
    print('\n'.join(lines))
    print('REPORT_PATH', report.relative_to(a.ROOT))
    print('ARTIFACT_STEM', stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    args = parser.parse_args()
    write(args.private.resolve())

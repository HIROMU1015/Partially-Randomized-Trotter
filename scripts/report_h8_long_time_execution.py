"""Collect completed H8 exploration outputs; never launch scientific workers."""
import argparse
import csv
from fractions import Fraction
import gzip
import json
from pathlib import Path

import h8_long_time_adapter as a


def collect(private, output):
    signal = a.read(private / 'stage_A_combined.json')
    endpoints = {v['endpoint']['candidate']['id']: v['endpoint'] for v in signal['results']}
    batches = [a.read(private / 'runs' / name / 'result.json') for name in ('stage_B01', 'stage_B02')]
    a.require(all(v['status'] == 'PASS' and v['binding'] == signal['binding'] for v in batches), 'COMPLETED_SAME_INPUT_ONLY')
    a.require(batches[0]['SOURCE'] == batches[1]['SOURCE'] and batches[0]['source_files'] == batches[1]['source_files'], 'SAME_NATIVE_SOURCE')
    a.require(all(v['signal_SOURCE'] == signal['SOURCE'] for v in batches), 'FIXED_SIGNAL_SOURCE')
    rows, trials, ids = [], [], set()
    for batch in batches:
        for cid, summary in batch['summaries'].items():
            a.require(cid not in ids, 'DISJOINT_PRICE_BATCHES')
            ids.add(cid)
            ep = endpoints[cid]
            samples = [v for v in batch['trials'] if v['candidate_id'] == cid]
            expected = 9 if ep['candidate']['method'] == 'PR' else 1
            a.require(len(samples) == summary['n'] == expected and {v['index'] for v in samples} == set(range(expected)), 'COMPLETE_EXPLORATION_N')
            representative = next(v for v in samples if v['index'] == 0)
            a.require(len(representative['checks']) == 7 and all(
                v['phase_alignment'] is False and 0 <= v['error'] < v['tolerance']
                for v in representative['checks']), 'REPRESENTATIVE_NATIVE_COMPILED_CHECKS')
            a.require(all(abs(v['log_B'] - ep['log_B']) <= ep['log_B_margin'] for v in samples), 'FIXED_FINITE_NORMALIZATION')
            # Reuse the tested exact weighted accounting, including covariance.
            checked = a.paired_G(samples, ep, 'EXPLORATION')
            a.require(all(checked[k] == summary[k] for k in checked), 'SAVED_PAIRED_G_MATCH')
            rows.append(dict(candidate=ep['candidate'], N_cos=ep['N_cos'], N_sin=ep['N_sin'],
                mean_C_cos_exact=str(Fraction(sum(v['prices']['cosine']['rz_count'] for v in samples), expected)),
                mean_C_sin_exact=str(Fraction(sum(v['prices']['sine']['rz_count'] for v in samples), expected)),
                **summary))
        trials.extend(batch['trials'])
    a.require(len(rows) == 19 and len(trials) == 51, 'FIXED_SELECTED_POOL_COMPLETENESS')
    a.require({v['candidate_id'] for v in trials} == ids, 'NO_UNREGISTERED_PRICE_TRIALS')
    pf = [v for v in rows if v['candidate']['method'] != 'PR']
    best_pf = min(pf, key=lambda v: Fraction(v['mean_G_exact']))
    comparisons = [dict(PR=v['candidate']['id'], PF=best_pf['candidate']['id'],
        PR_over_PF_exact=str(Fraction(v['mean_G_exact']) / Fraction(best_pf['mean_G_exact'])),
        PR_minus_PF_exact=str(Fraction(v['mean_G_exact']) - Fraction(best_pf['mean_G_exact'])),
        sample_SE_G=v['sample_SE_G'], exploration_only=True, population_guarantee=False)
        for v in rows if v['candidate']['method'] == 'PR']
    result = dict(status='COMPLETE_EXPLORATION_ONLY', signal_SOURCE=signal['SOURCE'],
        native_SOURCE=batches[0]['SOURCE'], source_files=batches[0]['source_files'], binding=signal['binding'],
        report_script_sha256=a.sha(__file__), rows=rows, trials=trials, comparisons=comparisons,
        measured_best_PF=best_pf['candidate']['id'], global_grid_optimum_claim=False,
        provenance={name: a.sha(private / 'runs' / name / 'result.json') for name in ('stage_A01', 'stage_A02', 'stage_B01', 'stage_B02')},
        confirmation=dict(status='NOT_RUN', n=None, precision_target=None, pooled=False),
        original_STOPs_preserved=True, speedup_benchmark=False)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'exploration.json.gz'
    with target.open('xb') as stream:
        stream.write(gzip.compress((json.dumps(result, sort_keys=True, allow_nan=False) + '\n').encode(), mtime=0))
    with (output / 'compiled_costs.csv').open('x') as stream:
        columns = ['id', 'method', 'formula', 'prefix', 'q', 'R_outer', 'K', 'n', 'N_cos', 'N_sin',
                   'mean_C_cos_exact', 'mean_C_sin_exact', 'mean_G_exact', 'sample_SE_G']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row['candidate'].get(k) if k in row['candidate'] else row[k] for k in columns})
    lines = [
        '# H8長時間・高次PR：Stage A/B実行結果', '',
        '2026-11-03。Stage AとStage Bの探索を完了。Stage Cは未実行。高速化率のbenchmarkは行っていない。', '',
        'Linear H8、隣接距離1.00 Å、STO-3G、保存tol-only1e-8 DF rank28、元state・fragment順序0..27。',
        'T=1.6、ε=0.0005、状態準備P=0。費用はG=N_cos C_cos+N_sin C_sin。', '',
        '## SOURCEと検査', '',
        f"Signal SOURCE：`{signal['SOURCE']}`。Native価格SOURCE：`{batches[0]['SOURCE']}`。",
        '価格経路では既存APIが要求する整数instruction boundへ修正し、元Stage AのSOURCEと入力・値を保全した。',
        '新T full-H sparse reference、signed primitive、finite mean/normalization、bias/headroom/shotを検査した。',
        'Stage Aは初期57＋境界補完12＝69候補。適格40、不適格29、判定保留0。',
        'Stage Bは選んだPR4候補（各n=9）とPF15候補（各1回）、計51 trajectory・102軸の価格を取得した。',
        '各群の代表例でlow/high control、algebraic event vector、cos/sin期待値、compiled vectorを検査した。',
        'CompilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/topology未指定。measurement付きfull Hadamard wrapperのRZを用いる。', '',
        '## 探索価格の比較', '',
        f"測定したPF15候補内の最小Gは {best_pf['candidate']['method']} / {best_pf['candidate']['formula']} / prefix{best_pf['candidate']['prefix']} / q{best_pf['candidate']['q']}、G={float(Fraction(best_pf['mean_G_exact'])):.6e}。", '',
        '| PR公式 | prefix | q | 平均G | Gのsample SE | PR/PF比 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for comp in comparisons:
        row = next(v for v in rows if v['candidate']['id'] == comp['PR'])
        c = row['candidate']
        lines.append(f"| {c['formula']} | {c['prefix']} | {c['q']} | {float(Fraction(row['mean_G_exact'])):.6e} | {row['sample_SE_G']:.4e} | {float(Fraction(comp['PR_over_PF_exact'])):.6f} |")
    lines += ['', '観測平均の比較であり、母集団での優位や全domainの最適性の保証ではない。',
        '同じtrajectoryのcos/sin価格を固定Nで重み付けし、軸間共分散を含むpaired Gの分散・SEを用いた。',
        '未測定候補の費用はunknown。未観測次数の条件付き費用もunknownとし、希少orderとpopulation tail bound未確立の限界を保持する。', '',
        '## 保存結果と次の段階', '',
        '[69候補のsignal/shot](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-03/stage_A_combined.json.gz)、',
        '[価格・会計・比較](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-03/exploration.json.gz)、',
        '[価格CSV](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-03/compiled_costs.csv)、',
        '[集計コード](../../../scripts/report_h8_long_time_execution.py)。', '',
        'seed実値、event列、入力raw、QPY、内部pathはprivateに保全。元SOURCE・結果・grant/ledger/marker・既存STOPは変更していない。',
        '主担当Codexによる検査であり、別担当者の独立科学レビューではない。', '',
        '次の担当：ユーザーが確認精度・n・候補・方向を判断しStage Cを許可する。Codexがweighted G分散と希少orderの限界を示し、許可後に別cohortで確認する。',
        '**Stage Cは未実行。自動top-up・候補差替え・旧rare-RTE run再開は行わない。**', '']
    (a.ROOT / 'docs/research/h8_long_time_execution_20261103/REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'], measured_candidates=len(rows), trajectories=len(trials),
                         best_measured_PF=best_pf['candidate'], comparisons=comparisons)))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    collect(args.private, args.output)

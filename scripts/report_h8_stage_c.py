"""Publish completed fixed Stage C metadata; never sample or build circuits."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from importlib.metadata import version
import json
import math
from pathlib import Path

import h8_long_time_adapter as a
import h8_stage_c as c

SOURCE = '82fdbdd098e9d3edae01a930437e3d76c1f8bf36'
RUN_TAG = 'H8_M5_P8_Q1_CONFIRMATION_01'
ARTIFACT = 'artifacts/resource_applicability/track_a_h8_long_time/2026-11-05'
REPORT = 'docs/research/h8_stage_c_execution_20261105/REPORT.md'
SCOPE_FIELDS = ('SOURCE', 'absolute_SE_G_target', 'automatic_top_up', 'binding',
                'candidates', 'cohort', 'comparison_direction', 'comparison_ids',
                'design_digest', 'evidence', 'n', 'native_SOURCE', 'phase_SOURCE',
                'pooled', 'preflight_scientific_calls', 'run_tag', 'seed_namespace',
                'signal_SOURCE', 'source_files', 'stage', 'workers')
TRIAL_FIELDS = ('candidate_id', 'checks', 'cohort', 'compiler_phase_accounting',
                'compiler_phase_source', 'conditional', 'endpoint_digest',
                'event_digest', 'index', 'laws', 'log_B', 'orders',
                'preparation_hash', 'prices', 'read_only_cache_reuse', 'status')


def collect(result_path, output, report):
    raw = a.read(result_path)
    scope, samples = raw['scope'], raw['trials']
    design = c.load(a.ROOT / 'configs/h8_long_time/stage_c_proposal_20261104.json')
    c.validate_design(design)
    a.require(raw['status'] == 'CONFIRMATION_COMPUTATION_COMPLETE'
              and scope['SOURCE'] == SOURCE and scope['run_tag'] == RUN_TAG
              and scope['stage'] == 'C' and scope['n'] == 9
              and scope['binding'] == design['binding']
              and scope['design_digest'] == a.digest(design)
              and scope['comparison_ids'] == design['comparison_ids']
              and scope['candidates'] == [design['comparison']['PR']['candidate']]
              and scope['cohort'] == 'CONFIRMATION'
              and scope['seed_namespace'] == c.NAMESPACE
              and scope['comparison_direction'] == 'PR_LOWER'
              and not scope['pooled'] and not scope['automatic_top_up'], 'FIXED_COMPLETED_SCOPE')
    for path, expected in scope['source_files'].items():
        a.require(a.sha(a.ROOT / path) == expected, 'EXECUTED_SOURCE_UNCHANGED:' + path)
    a.require(len(samples) == 9 and {v['index'] for v in samples} == set(range(9))
              and all(v['status'] == 'PASS' and v['cohort'] == 'CONFIRMATION'
                      and v['candidate_id'] == scope['comparison_ids']['PR']
                      and not v['conditional'] and set(v['prices']) == {'cosine', 'sine'}
                      for v in samples), 'NINE_FRESH_PAIRED_RESULTS')
    representative = next(v for v in samples if v['index'] == 0)
    names = {'LOW_CONTROL', 'HIGH_CONTROL_LEAKAGE', 'SAVED_BLOCK_ALGEBRAIC_EVENT_VECTOR',
             'HADAMARD_cosine', 'HADAMARD_sine', 'COMPILED_VECTOR_cosine', 'COMPILED_VECTOR_sine'}
    checks = representative['checks']
    a.require(len(checks) == 7 and {v['name'] for v in checks} == names
              and all(math.isfinite(v['error']) and 0 <= v['error'] < v['tolerance']
                      and v['tolerance'] == (1e-7 if v['name'].startswith('COMPILED_VECTOR') else 1e-8)
                      and v['phase_alignment'] is False for v in checks), 'SEVEN_ORIGINAL_CHECKS')
    a.require(len(representative['compiler_phase_accounting']) == 2, 'TWO_PHASE_ACCOUNTS')
    for phase in representative['compiler_phase_accounting'].values():
        a.require(phase['derived_from'] == 'CERTIFIED_SMALL_GATE_OPERATORS'
                  and phase['statevector_phase_alignment'] is False
                  and phase['gate_counts_unchanged'] is True
                  and phase['delta_phase_quarters'] == (-phase['xx_plus_yy_count']) % 4
                  and 0 <= phase['maximum_local_operator_error'] < 1e-10, 'FIXED_PHASE_ACCOUNTING')
    pr = raw['PR_summary']
    checked = a.paired_G(samples, design['comparison']['PR']['endpoint'], 'CONFIRMATION')
    a.require(all(checked[k] == pr[k] for k in checked), 'SAVED_PAIRED_ACCOUNTING')
    pf = design['comparison']['PF']['saved_aggregate']
    a.require(raw['fixed_PF'] == pf and raw['comparison'] == c.compare_fixed_PF(
        pr, pf, design['absolute_SE_G_target']), 'FIXED_PF_AND_SE_DECISIONS')
    a.require(pr['rarity']['observed_order_counts'] == {
        str(k): sum(v['orders'][str(k)] for v in samples) for k in (0, 2, 4)}, 'SAVED_ORDER_COUNTS')

    mean, baseline = Fraction(pr['mean_G_exact']), Fraction(pf['mean_G_exact'])
    b = design['comparison']['PR']['saved_aggregate']
    comparison = dict(raw['comparison'], PR_over_fixed_PF_exact=str(mean / baseline),
                      observed_reduction_fraction_exact=str(1 - mean / baseline))
    public = dict(schema='H8_STAGE_C_PUBLIC_RESULT_V1', status=raw['status'],
        scope={k: scope[k] for k in SCOPE_FIELDS},
        trials=[{k: v[k] for k in TRIAL_FIELDS} for v in samples],
        PR_summary=pr, fixed_PF=pf, comparison=comparison,
        exploration_comparison=dict(saved_Stage_B=b, confirmation_minus_exploration_mean_exact=str(
            mean - Fraction(b['mean_G_exact'])), pooled=False, no_significance_test=True),
        statistical_claim=raw['statistical_claim'], population_guarantee=False,
        representative_checks=dict(index=0, count=7, status='PASS'),
        provenance=dict(raw_result_sha256=a.sha(result_path),
                        raw_scope_sha256=a.sha(result_path.parent / 'scope.json'),
                        report_script_sha256=a.sha(__file__),
                        seed_values_sha256=a.sha(result_path.parent / 'seeds_PRIVATE.json')),
        environment=dict(python=__import__('platform').python_version(),
                         packages={k: version(k) for k in ('numpy', 'scipy', 'numba', 'qiskit')}),
        mandatory_STOP=True, next_scientific_run_started=False,
        privacy='No seed values, event sequences, input raw, QPY bytes or internal paths.')
    tz = timezone(timedelta(hours=9))
    stamp = lambda p: datetime.fromtimestamp(p.stat().st_mtime, tz).isoformat()
    public['execution_times_JST'] = dict(start=stamp(result_path.parent / 'scope.json'), end=stamp(result_path))
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'confirmation.json'
    with target.open('x') as stream:
        stream.write(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with (output / 'paired_prices.csv').open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['index', 'cohort', 'C_cos_RZ', 'C_sin_RZ', 'N_cos', 'N_sin', 'G_RZ', 'order0', 'order2', 'order4'])
        for v in sorted(samples, key=lambda x: x['index']):
            cc, cs = (v['prices'][axis]['rz_count'] for axis in ('cosine', 'sine'))
            nc, ns = pr['fixed_N_cos'], pr['fixed_N_sin']
            writer.writerow([v['index'], v['cohort'], cc, cs, nc, ns, nc*cc + ns*cs,
                             v['orders']['0'], v['orders']['2'], v['orders']['4']])
    lines = [
        '# H8長時間・高次PR：Stage C独立確認実験の結果', '',
        '状態：固定9標本・18価格と結果集計が完了。代表index0は元の7検査すべてPASS。追加標本・自動retry/resumeなし。', '',
        f"実行：{public['execution_times_JST']['start']} ～ {public['execution_times_JST']['end']}。9 workerで並列実行。",
        f'実行SOURCE：`{SOURCE}`。run_tag：`{RUN_TAG}`。namespace：`{c.NAMESPACE}`。', '',
        'Linear H8、隣接1.00 Å、STO-3G、保存tol-only1e-8 DF28、16 system qubit・spin sector4900、元state・fragment順序0..27。T=1.6、ε=.0005、状態準備P=0。',
        'PR：4th(m5_best)/prefix8/q1/R_outer32/K4、ID `7eb0a3d691b94089969b`、固定N_cos=84916951・N_sin=88339169。',
        '固定PF：B0I/4th(m5_best)/prefix15/q1、ID `1425ae6ed83fd9098b39`、保存G=89291219516568 RZ。PFは保存両軸価格613049 RZ、N_cos=71893270・N_sin=73757762を引用し、新sampling/build/compileなし。元cohort=EXPLORATIONを保持する。',
        'Qiskit1.3.0、rz/sx/x/cx、opt1、transpiler seed17、backend/target/topologyなし。費用はmeasurement付きfull Hadamard wrapperのcompiled RZによるG=N_cos C_cos+N_sin C_sin。', '',
        '## 確認標本だけの会計と判定', '',
        '| 項目 | 結果 |', '|---|---:|',
        f"| 平均G | {pr['mean_G_exact']} RZ |",
        f"| 固定PFとの比 | {float(mean / baseline):.12f} |",
        f"| 固定PFとの差 PR−PF | {comparison['PR_minus_fixed_PF_exact']} RZ |",
        f"| 観測平均の削減率 | {100 * float(1 - mean / baseline):.8f}% |",
        f"| paired G sample variance（n−1） | {pr['sample_variance_G_exact']} RZ² |",
        f"| paired G sample SE | {pr['sample_SE_G']} RZ |",
        f"| 固定h_G | {design['absolute_SE_G_target']} RZ |",
        f"| 比較方向 PR平均 < 固定PF | {'成立' if comparison['observed_PR_lower'] else '不成立'} |",
        f"| 記述的SE ≤ h_G | {'達成' if comparison['descriptive_precision_met'] else '未達'} |", '',
        '同一trajectoryのcos/sinを固定Nで重み付けしてから分散・SEを求めた。既存paired_Gを再利用し、軸間共分散を含む。探索標本とはpoolしない。95%信頼区間・新しい有意水準は採択していない。', '',
        '## 回路検査と位相会計', '',
        '| 元検査名 | error | tolerance |', '|---|---:|---:|']
    lines += [f"| {v['name']} | {v['error']:.8e} | {v['tolerance']:.0e} |" for v in checks]
    lines += ['', '全検査はstatevectorのglobal-phase alignmentなし。固定compiler位相補助は、実際の小gate operatorから導出した余分な位相をfull wrapperのglobal phaseへ会計した。RZ数・gate数・許容値は変更していない。',
        '今回の代表はXX+YY gate 52214個に対しΔφ=π（mod 2π）。局所operator最大誤差3.828435720135541e-14。両軸の証拠hashは公開JSONに保存し、private QPY・局所gate実値は公開しない。', '',
        '## 希少次数と結論の限界', '',
        '確認9標本のevent観測はorder0=288、order2=0、order4=0。未観測order2/4の条件付きcompiled費用はunknown/null、population tail cost boundもnull。0で補完していない。',
        '保存sampling lawはUNCHANGED_PCG64_NUMPY_CHOICE_FINITE_PAIRED_RTE。同lawで1 trajectoryにorder2以上が含まれる確率0.017503368117490408、order4以上は1.7365913367114017e-6。9標本で未観測となるlawモデルの確率は0.8530601056102244／0.9999843707865361。これは保存lawの診断で、PRNGの数学的独立性の証明ではない。',
        '独立seed namespaceの新cohortでも、固定PFに対する観測平均の低さと記述的SE目標を確認した。一方、未観測rare群の費用・分散と一様なpost-compile費用上界が未確立なので、母集団平均の優位・母集団精度を保証しない。全候補最適性、全RPE総費用、他分子・別compilerへの一般性も未検証。', '',
        '## Stage Bとの比較（別cohortのまま）', '',
        '| cohort | n | 平均G | sample SE |', '|---|---:|---:|---:|',
        f"| EXPLORATION（保存値） | {b['n']} | {b['mean_G_exact']} | {b['sample_SE_G']} |",
        f"| CONFIRMATION | {pr['n']} | {pr['mean_G_exact']} | {pr['sample_SE_G']} |", '',
        f"確認平均−探索平均は {mean - Fraction(b['mean_G_exact'])} RZ（探索平均比 {100 * float(mean / Fraction(b['mean_G_exact']) - 1):.8f}%）。これは記述比較で、両群を混合した推定や新しい有意差検定ではない。", '',
        '## 成果物・SOURCE・保全とSTOP', '',
        f'[確認JSON・provenance](../../../{ARTIFACT}/confirmation.json)、[9標本の価格CSV](../../../{ARTIFACT}/paired_prices.csv)、[報告用コード](../../../scripts/report_h8_stage_c.py)。',
        '[固定計画](../h8_stage_c_preparation_20261104/PLAN.md)、[固定scope案](../../../configs/h8_long_time/stage_c_proposal_20261104.json)、[Stage A/B保存結果](../h8_long_time_execution_20261103/EXPLORATION_RESULTS_20261104.md)。',
        f'Signal SOURCE `{c.SIGNAL_SOURCE}`、native SOURCE `{c.NATIVE_SOURCE}`、位相補助SOURCE `{c.PHASE_SOURCE}`。Stage A/B圧縮結果hash、入力binding、今回SOURCE hash、原結果hashはJSONから辿る。',
        '元Stage A/BのSOURCE・raw/QPY・結果・旧grant/ledger/marker・失敗/HOLD・STOPを保全した。新cohortのseed一意性と旧seedとの非衝突は実行時に確認済み。seed実値・event列・入力raw・QPY・内部pathはprivateに保持する。',
        'これは主担当Codexによる実行・技術確認であり、別担当者の独立科学レビューや外部再現ではない。',
        '起動記録は本REPORTの直前公開版commit `6439f59d8a24667dfaf207102d738dcff1729c98` に保存されている。', '',
        '**固定9標本の実行・集計を完了。結果公開とremote照合後はmandatory STOP。希少費用検証・母集団保証・別候補探索・総RPE評価・詳細科学レビューへ自動移行しない。**',
        '次の作業なし（今回の許可範囲を完了）。', '']
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines))
    print(json.dumps(dict(status=public['status'], n=len(samples), axes=18, checks=7,
                         comparison=comparison, artifact_sha256=a.sha(target)), sort_keys=True))
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    collect(args.result, a.ROOT / ARTIFACT, a.ROOT / REPORT)

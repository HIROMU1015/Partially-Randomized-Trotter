"""Audit completed saved trials and export compact evidence. Never run science."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import execute as x
from h246_saved_cost_evaluation import paired


def audit_import(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in ('numpy', 'scipy', 'qiskit', 'pyscf', 'openfermion'):
        raise RuntimeError('Saved-record finalization cannot import scientific runtimes')


def forbidden(*args, **kwargs):
    raise RuntimeError('Saved-record finalization cannot build, sample or compile')


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def export_csv(path, rows):
    x.report.write_csv(path, rows)


def write_report(plan, public, accounted, cells, groups, audit):
    """Render the fixed comparisons, keeping point rankings and diagnostics apart."""
    comparisons = read_csv(public / 'primary_comparisons.csv')
    contrasts = read_csv(public / 'R_unit_price_contrasts.csv')
    ec, cc = plan['cohorts']
    lookup = {(v['id'], v['cohort'], float(v['epsilon'])): v for v in accounted}
    lines = ['# H4/H6未取得7候補：実測費用取得結果', '',
        '2026-10-27 JST。固定7候補の探索1,856本と独立確認54,833本、計56,689 paired trajectoryを取得した。'
        '両cohortを分けて会計し、所定本数の取得・保存値監査後に**mandatory STOP**。追加候補・top-upは0。', '',
        '対象はlinear H4/H6、隣接1.00 Å、STO-3G、tol-only DF threshold 1e-8、保存rank H4=10/H6=19。'
        'H6保存weighted-Hermitian completion、元Hamiltonian/state/reference/scalar/fragment順を維持した。'
        '保存prefix、q、R_outer、Kのみを使い、今回のdelta=T/q窓は0.2〜2.0。'
        'signed S2/m5_best、ABS_TIME_PER_OUTER、finite-RTE/PCG64/NumPy、symmetric-directional control/actual adjoint、'
        'cosine/+Im sineの測定付きfull Hadamard wrapper、状態準備P=0を使用した。'
        'Qiskit1.3.0/NumPy1.26.4、basis rz/sx/x/cx、optimization_level=1、seed_transpiler=17。'
        '元のsignal適格性・allowance・N_cos/N_sinを再利用し、Hamiltonian/state/reference/signal/shotの新規計算は0。', '',
        '## 固定7候補の結果', '',
        'GはRZ単価×保存shot数、SEはcosine/sineのpaired covarianceを含む標本平均のSE。'
        '同じ物理候補を複数epsilonに会計しても新しい候補や独立試行とは数えない。', '',
        '|Candidate ID|系/T/epsilon|公式/prefix/q/R/K|探索n / 確認n|探索G|確認G ± SE|確認点Gの旧PR比|',
        '|---|---|---|---:|---:|---:|---:|']
    for item in plan['candidates']:
        r = item['candidate']; identity = r['id']; e = item['primary_epsilon']
        a, b = lookup[identity, ec, e], lookup[identity, cc, e]
        percent = (float(b['G']) / item['primary_PR']['G'] - 1) * 100
        lines.append(f"|{identity}|{r['system']}/{r['T']}/{e}|{r['formula']}/{r['prefix']}/{r['q']}/{r['R_outer']}/{r['K']}|{a['n']} / {b['n']}|{float(a['G']):.9g}|{float(b['G']):.9g} ± {float(b['G_SE']):.5g}|{percent:+.4f}%|")
    lines += ['', '## 最優先：H4・T=0.8・epsilon=0.05のPF/PR順位', '',
        '固定PF参照G=75,021,842（B0+I/prefix5/S2/q1）と、旧PR参照G=76,135,649.8889を使用する。'
        f"事前固定した正規/Bonferroni係数 z={plan['normal_bonferroni_z']:.11f}（11比較）による差の診断を示す。"
        '以下の近似区間は母平均・rare tailの厳密保証ではない。', '',
        '|Candidate ID|確認G − 固定PF|差の固定点近似区間|確認点がPF未満|', '|---|---:|---|---|']
    for identity in x.prepare.IDS[:2]:
        v = next(v for v in comparisons if v['id'] == identity and v['cohort'] == cc
                 and v['reference_id'] == plan['candidates'][list(x.prepare.IDS).index(identity)]['primary_PF']['id'])
        lines.append(f"|{identity}|{float(v['difference']):+.8g}|[{float(v['fixed_point_interval_lower']):+.8g}, {float(v['fixed_point_interval_upper']):+.8g}]|{'はい' if v['observed_below']=='True' else 'いいえ'}|")
    lines += ['', '## 旧PRとの比較と残る不確かさ', '',
        '旧PRの探索参照値自体にSEと選択の不確かさがある。候補側だけの対固定点区間と、'
        '新旧双方のSEを含む差の正規近似診断を分離する。旧PRの追加取得は今回の範囲外で行っていない。', '',
        '|Candidate ID|確認G − 旧PR|双方SEを含む差の近似区間|確認点が旧PR未満|', '|---|---:|---|---|']
    diagnostics = []
    for item in plan['candidates']:
        identity = item['candidate']['id']
        v = next(v for v in comparisons if v['id'] == identity and v['cohort'] == cc and v['reference_id'] == item['primary_PR']['id'])
        difference, halfwidth = float(v['difference']), float(v['two_mean_diagnostic_halfwidth'])
        low, high = difference - halfwidth, difference + halfwidth
        lines.append(f"|{identity}|{difference:+.8g}|[{low:+.8g}, {high:+.8g}]|{'はい' if difference<0 else 'いいえ'}|")
        diagnostics.append(dict(id=identity, reference_id=v['reference_id'], difference=difference,
            fixed_point_lower=float(v['fixed_point_interval_lower']), fixed_point_upper=float(v['fixed_point_interval_upper']),
            two_mean_normal_lower=low, two_mean_normal_upper=high,
            observed_lower=difference < 0, two_mean_normal_diagnostic_direction='LOWER' if high < 0 else 'HIGHER' if low > 0 else 'UNRESOLVED',
            population_guarantee=False))
    export_csv(public / 'decision_diagnostics.csv', diagnostics)
    lines += ['', '保存済み独立確認の旧PR価格が利用できる場合、そのG/SEと比較はprimary_comparisons.csvの'
        '`saved_independent_reference_*`欄に別途保持した。参照値の差し替えやcohortのpoolingは行っていない。', '',
        '## 40セルの拡張集合の点順位', '',
        f"旧取得集合のPR低費用セル数 **33/40** を変更せず保存する。今回の探索価格を追加した集合では "
        f"**{sum(v['extended_PR_below_PF'] for v in cells)}/40**。"
        '独立確認は点順位へpool/代入せず、各候補の別cohortの検証として扱う。'
        '全40セルはextended_cells.csvに保存した。', '',
        '|系/T/epsilon|探索追加後のPR最小値が変化|PF/PRの点方向が変化|採用Candidate ID|', '|---|---|---|---|']
    for v in cells:
        if v['changed_PR_minimum']:
            lines.append(f"|{v['system']}/{v['T']}/{v['epsilon']}|はい|{'はい' if v['changed_PF_PR_direction'] else 'いいえ'}|{v['extended_PR_id']}|")
    lines += ['', '## R_outer対照', '',
        '単価はcosine/sineの平均RZで、総Gとは異なる。H4の対照はRとKの同時変更なのでR単独の効果を同定しない。'
        'H6の対照は同prefix/q/KでRのみが異なるが、Nも変化するため単価差とG差を分ける。', '',
        '|cohort|左 → 右|差（右単価−左単価）|近似区間|', '|---|---|---:|---|']
    for v in contrasts:
        lines.append(f"|{'探索' if v['cohort']==ec else '確認'}|{v['left']} → {v['right']}|{float(v['difference_right_minus_left']):+.7g}|[{float(v['interval_lower']):+.7g}, {float(v['interval_upper']):+.7g}]|")
    lines += ['', '## 希少event・保証の限界', '',
        '|Candidate ID|cohort|n|観測order2|観測order4|観測order6|', '|---|---|---:|---:|---:|---:|']
    for v in groups:
        lines.append(f"|{v['id']}|{'探索' if v['cohort']==ec else '確認'}|{v['n']}|{v['N2']}|{v['N4']}|{v['N6']}|")
    lines += ['', '未観測orderの条件付き費用や母平均tail boundはnullとし、0を補入しない。'
        '事前nは既存近傍の最大CV²に分散倍率4を置いた経験的計画で、未知候補の分散上界ではない。'
        '有限PCG64列の独立性、無条件のPR母平均優位、全探索域の最適性、全trajectory/全状態でのcompiled誤差rhoを保証しない。'
        '本結果は固定T・保存endpointによる条件付きGであり、全RPE/energy評価の最終総費用ではない。', '',
        '## SOURCE・入力・取得と監査', '',
        f"実行SOURCE `{audit['execution_SOURCE']}`、PLAN SHA256 `{x.prepare.sha(HERE/'PLAN.json')}`。"
        '元研究SOURCE 155529e20a8fac39e00786c40b57830b71203c4f、選定SOURCE 2e518b86e7c052dafb20dece0de7ffa6c7c85f8f、'
        'provider c31ef424175c02379eafebe8fc6c7548c1d9aabeを参照する。'
        'H6は原snapshot/receiptのbyte一致、H4は保存済み2026-10-22再構築入力の7科学配列一致でありNPZ容器SHAは異なる。'
        '今回は入力の再生成をしていない。詳細はPLAN.jsonのsystemsと元準備commitのREPORTを参照する。', '',
        f"全{audit['trajectories']:,} trial・{audit['logical_axis_evaluations']:,}軸・{audit['RTE_events']:,} RTE eventを照合した。"
        f"実新規compile {audit['actual_new_compile_calls']:,}軸、compile cache再利用 {audit['compile_cache_hits']:,}軸。"
        f"native検査{audit['checked_native_gates']}件、compiled-vector検査{audit['checked_compiled_gates']}件。"
        f"最大native誤差{audit['maximum_native_error']:.5g}、最大compiled誤差{audit['maximum_compiled_error']:.5g}。"
        '14本の代表trialは固定nに含まれる。元のnative1e-9/compiled1e-7閾値とphase-alignment禁止を維持した。'
        '元cache仕様どおり、representativeのcompiled価格がcache hitした場合はそのtrialでのcompiled-vector検査を再実行しない。'
        '検査実施件数は上記の実記録で示し、全113,378軸に対するvector検査とは扱わない。', '',
        f"保存seed {audit['new_piece_seed_identifiers']:,}識別子の列digestは `{audit['seed_sequence_sha256']}` で事前計画と一致。"
        'known除外資料との照合も再実行し衝突0。全歴史的seedの回収保証はない。'
        '全trialのID/index/cohort、event数とorder、scalar/preparation/log_B、整数price、paired momentsと共分散、保存N/exact Gを監査した。'
        '21件の事前人工/mock検査とは区別し、今回の終了監査はローカル保存記録の照合であって外部再現ではない。', '',
        '元controllerのtool sessionは終了後に参照できず、OSの終了codeは取得不可（null）。'
        'COMPLETIONの全worker join記録、STOP不在、該当実行process不在を直接照合した。'
        '終了code=0を推測で補っていない。完成試行の再取得やcontrollerの再起動はしない。', '',
        '最大128 workerのpoolと資源に応じた投入数調整を使った。cohort別のcache件数・同時投入数は*_performance.json、'
        '候補別の実trial時間・RSSはcohort_diagnostics.csvに保存した。比較対象となる未最適化の実分子runは実施しておらず、'
        'pipeline全体の速度向上率は未測定。Phase Aのschedule単体4.06倍を全体の速度比として扱わない。', '',
        '## 成果物・終了', '',
        '- results/candidate_costs.csv：適格epsilon別のN、C、G、paired SE。',
        '- results/unit_cost_statistics.csv：両軸分散・共分散、RZ/CX/depth等の平均。',
        '- results/primary_comparisons.csv、decision_diagnostics.csv、R_unit_price_contrasts.csv：固定参照との差と診断。',
        '- results/cohort_diagnostics.csv、trajectory_costs.csv.gz：全56,689 trialのpaired費用・order数と取得範囲。',
        '- results/extended_cells.csv、extended_point_summary.json：旧33/40と拡張探索集合を分離した40セル。',
        '- results/execution_provenance.json、COMPLETION.json、END_AUDIT.json：実行SOURCE、固定範囲の完了、結果hash。', '',
        '元研究・元SOURCE/raw/grant/ledger/marker/STOPは変更していない。B3関連の操作・別研究worktreeの切替・'
        '共有環境の変更はしていない。private raw/event/cacheは専用run01に保全し、公開表はその派生コピーである。', '',
        '**固定取得と終了監査後のmandatory STOP。追加候補、標本追加、別研究、詳細科学レビューへ自動進行しない。**', '']
    (HERE / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-exit-code', type=int, default=None)
    args = parser.parse_args()
    assert args.controller_exit_code in (None, 0)
    sys.addaudithook(audit_import)
    for name in ('compose', 'sample_requests', 'compile_cached', 'cost_trial', 'native_context'):
        setattr(x.native, name, forbidden)
    plan = x.read_plan()
    x.verify_static(plan)
    private, public = Path(plan['private_output_root']), Path(plan['public_results_root'])
    running = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'analysis/h46_seven_cost_readiness_20261027/execute.py' in command and b'run' in command:
                running.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    assert not running, 'Execution still active; do not finalize'
    completion = x.study.read(public / 'COMPLETION.json')
    assert completion['status'] == 'FIXED_COUNTS_COMPLETE_MANDATORY_STOP'
    assert completion['all_workers_joined'] and not completion['automatic_next_stage']
    assert not (public / 'STOP.json').exists()
    provenance = x.study.read(public / 'execution_provenance.json')
    assert provenance['SOURCE'] == 'cab73373593ba7c113c1af018b1dc3077c8e78fa'
    assert provenance['plan_sha256'] == x.prepare.sha(HERE / 'PLAN.json')
    assert not any(private.glob('cost/*/*/*/STOP.json'))
    assert {p.name for p in (private / 'cost').iterdir()} == set(plan['cohorts'])
    all_groups = {}
    seed_digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    observed_seeds = set()
    unit_rows, group_rows, trial_rows = [], [], []
    total_trials = total_events = cache_hits = 0
    max_native_error = max_compiled_error = 0.
    compiled_checks = native_checks = 0
    started = datetime.fromisoformat(provenance['started_at'])
    for cohort in plan['cohorts']:
        saved = x.study.read(public / (cohort + '_costs.json.gz'))
        assert set(saved) == set(x.prepare.IDS)
        assert {p.name for p in (private / 'cost' / cohort).iterdir()} == set(saved)
        count_field = 'n_exploration' if cohort == plan['cohorts'][0] else 'n_confirmation'
        all_groups[cohort] = saved
        for item in plan['candidates']:
            row = item['candidate']; identity = row['id']; n = item[count_field]
            folder = private / 'cost' / cohort / identity
            assert {p.name for p in folder.iterdir()} == {str(i) for i in range(n)}
            schedule = x.prepare.core.schedule(row['formula'], row['q'], 'ABS_TIME_PER_OUTER', row['R_outer'], T=row['T'])
            prices, metrics, orders = [], {a: Counter() for a in ('cosine', 'sine')}, Counter()
            reference = None
            wall, peak = [], 0
            for index in range(n):
                trial = x.accepted_trial(row, cohort, index, private, item['saved_endpoints'][0]['log_B'])
                assert trial is not None
                binding = (trial['preparation_hash'], trial['scalar'])
                if reference is None:
                    reference = binding
                assert binding == reference
                event_path = folder / str(index) / 'events_PRIVATE.json'
                result_path = folder / str(index) / 'result.json'
                events = x.study.read(event_path)
                seeds = [x.native.trial_seed(row, cohort, index, s['outer'], s['piece']) for s in schedule]
                assert events['seeds'] == seeds and events['cohort'] == cohort and events['conditional'] is False
                assert len(events['events']) == len(schedule)
                assert all(len(batch) == s['rte_steps'] for batch, s in zip(events['events'], schedule))
                actual = Counter(str(e['taylor_order']) for batch in events['events'] for e in batch)
                assert all(actual[str(k)] == trial['orders'][str(k)] for k in range(0, row['K'] + 1, 2))
                assert sum(actual.values()) == row['q'] * row['R_outer']
                for seed in seeds:
                    assert seed not in observed_seeds
                    observed_seeds.add(seed); seed_digest.update(str(seed).encode() + b'\n')
                for path in (result_path, event_path):
                    raw_digest.update(str(path.relative_to(private)).encode() + b'\0')
                    raw_digest.update(bytes.fromhex(x.prepare.sha(path)))
                pair = tuple(trial['prices'][a]['rz_count'] for a in ('cosine', 'sine'))
                prices.append(pair); orders.update(actual)
                for axis in metrics:
                    metrics[axis].update(trial['prices'][axis])
                for check in trial['checks']:
                    if check['check'].startswith('compiled_'):
                        compiled_checks += 1; max_compiled_error = max(max_compiled_error, check['error'])
                    else:
                        native_checks += 1; max_native_error = max(max_native_error, check['error'])
                cache_hits += trial['cache_hits']; total_trials += 1; total_events += sum(actual.values())
                wall.append(trial['wall_seconds']); peak = max(peak, trial['peak_RSS_bytes'])
                trial_rows.append(dict(id=identity, cohort=cohort, index=index,
                    **{axis + '_' + metric: trial['prices'][axis][metric] for axis in metrics for metric in x.native.METRICS},
                    N0=actual['0'], N2=actual['2'], N4=actual['4'], N6=actual['6'],
                    cache_hits=trial['cache_hits'], seed_list_sha256=hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
                    scalar=trial['scalar'], log_B=trial['log_B'], preparation_hash=trial['preparation_hash'],
                    wall_seconds=trial['wall_seconds'], peak_RSS_bytes=trial['peak_RSS_bytes']))
            stats = paired(prices)
            group = saved[identity]
            assert group['n'] == n and group['cohort'] == cohort and group['candidate'] == row
            assert stats == group['unit_statistics']
            assert {a: {m: metrics[a][m] / n for m in x.native.METRICS} for a in metrics} == group['mean_metrics']
            assert {str(k): orders[str(k)] for k in range(0, row['K'] + 1, 2)} == group['rarity']['observed_order_counts']
            unit_rows.append(dict(row, cohort=cohort, n=n,
                C_cos=stats['cosine']['mean'], C_sin=stats['sine']['mean'],
                variance_cos=stats['cosine']['variance'], variance_sin=stats['sine']['variance'],
                covariance=stats['covariance'], covariance_exact=str(x.prepare.fraction(stats['covariance_exact'])),
                minimum_cos=stats['cosine']['minimum'], maximum_cos=stats['cosine']['maximum'],
                minimum_sin=stats['sine']['minimum'], maximum_sin=stats['sine']['maximum'],
                **{'mean_' + a + '_' + m: group['mean_metrics'][a][m] for a in metrics for m in x.native.METRICS}))
            group_rows.append(dict(id=identity, cohort=cohort, n=n, N0=orders['0'], N2=orders['2'], N4=orders['4'], N6=orders['6'],
                mean_trial_wall_seconds=math.fsum(wall) / n, maximum_trial_wall_seconds=max(wall), peak_RSS_bytes=peak,
                population_tail_cost_bound=None, unobserved_cost_imputed=False))
    assert seed_digest.hexdigest() == plan['seed_audit']['new_seed_sequence_sha256']
    assert len(observed_seeds) == plan['seed_audit']['new_piece_seed_identifiers']
    assert total_trials == completion['actual_trajectories'] == 56689
    assert completion['logical_axis_evaluations'] == 2 * total_trials
    assert total_events == plan['scientific_calls']['logical_RTE_events'] == 539044
    assert cache_hits == completion['recorded_cache_hits']
    assert 2 * total_trials - cache_hits == completion['recorded_new_compile_calls']
    accounted = read_csv(public / 'candidate_costs.csv')
    keys = [(v['id'], v['cohort'], float(v['epsilon'])) for v in accounted]
    expected_keys = {(v['candidate']['id'], co, e['epsilon']) for v in plan['candidates']
                     for co in plan['cohorts'] for e in v['saved_endpoints'] if e['status'] == 'ELIGIBLE'}
    assert len(keys) == len(set(keys)) and set(keys) == expected_keys
    for value in accounted:
        item = next(v for v in plan['candidates'] if v['candidate']['id'] == value['id'])
        ep = next(e for e in item['saved_endpoints'] if e['epsilon'] == float(value['epsilon']))
        assert ep['status'] == 'ELIGIBLE' and (ep['N_cos'], ep['N_sin']) == (int(value['N_cos']), int(value['N_sin']))
        expected = x.report.cost_for(ep, all_groups[value['cohort']][value['id']])
        assert Fraction(expected['G_exact']) == Fraction(value['G_exact'])
        assert expected['G_SE'] == float(value['G_SE'])
    comparisons = read_csv(public / 'primary_comparisons.csv')
    assert len(comparisons) == 18
    for v in comparisons:
        item = next(i for i in plan['candidates'] if i['candidate']['id'] == v['id'])
        reference = next(i for i in (item['primary_PF'], item['primary_PR']) if i['id'] == v['reference_id'])
        value = next(i for i in accounted if (i['id'], i['cohort'], float(i['epsilon'])) == (v['id'], v['cohort'], float(v['epsilon'])))
        assert Fraction(v['difference_exact']) == Fraction(value['G_exact']) - Fraction(reference['G_exact'])
    export_csv(public / 'unit_cost_statistics.csv', unit_rows)
    export_csv(public / 'cohort_diagnostics.csv', group_rows)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(trial_rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(trial_rows)
    (public / 'trajectory_costs.csv.gz').write_bytes(gzip.compress(buffer.getvalue().encode(), compresslevel=6, mtime=0))
    best = {(b['system'], b['T'], b['epsilon'], b['family']): b['best'] for b in x.study.read(x.study.PUBLIC / 'best_configurations.json') if b['formula_group'] == 'ALL'}
    cells = []
    for (s, t, e, family), old_pr in best.items():
        if family != 'PR':
            continue
        pf = best[s, t, e, 'PF']
        added = [v for v in accounted if (v['system'], float(v['T']), float(v['epsilon']), v['cohort']) == (s, t, e, plan['cohorts'][0])]
        chosen = min([old_pr, *added], key=lambda v: Fraction(v['G_exact']))
        old_below = Fraction(old_pr['G_exact']) < Fraction(pf['G_exact'])
        new_below = Fraction(chosen['G_exact']) < Fraction(pf['G_exact'])
        cells.append(dict(system=s, T=t, epsilon=e, old_PF_id=pf['id'], old_PR_id=old_pr['id'],
            old_PF_G=pf['G'], old_PR_G=old_pr['G'], old_PR_below_PF=old_below,
            added_distinct_PR=len(added), extended_PR_id=chosen['id'], extended_PR_G=float(chosen['G']),
            extended_PR_G_exact=chosen['G_exact'], extended_PR_below_PF=new_below,
            changed_PR_minimum=chosen['id'] != old_pr['id'], changed_PF_PR_direction=old_below != new_below,
            ranking_basis='EXPLORATION_POINT_ONLY; confirmation_not_pooled'))
    assert len(cells) == 40 and sum(v['old_PR_below_PF'] for v in cells) == 33
    extended = x.study.read(public / 'extended_point_summary.json')
    assert sum(v['extended_PR_below_PF'] for v in cells) == extended['extended_exploration_point_count']
    export_csv(public / 'extended_cells.csv', cells)
    audit = dict(status='PASS_FIXED_COUNTS_COMPLETE_MANDATORY_STOP', validation_scope='LOCAL_SAVED_RECORD_AUDIT; not external reproduction',
        execution_SOURCE=provenance['SOURCE'], finalizer_SOURCE_sha256=x.prepare.sha(Path(__file__)),
        controller_exit_code=args.controller_exit_code, exit_code_evidence='TOOL_SESSION_NO_LONGER_AVAILABLE' if args.controller_exit_code is None else 'REPORTED_BY_TOOL',
        all_workers_joined=True, observed_execution_processes=running, new_scientific_calls_in_finalizer=0,
        trajectories=total_trials, logical_axis_evaluations=2 * total_trials, RTE_events=total_events,
        actual_new_compile_calls=completion['recorded_new_compile_calls'], compile_cache_hits=cache_hits,
        checked_native_gates=native_checks, checked_compiled_gates=compiled_checks,
        maximum_native_error=max_native_error, maximum_compiled_error=max_compiled_error,
        new_piece_seed_identifiers=len(observed_seeds), seed_sequence_sha256=seed_digest.hexdigest(),
        result_event_file_digest=raw_digest.hexdigest(), source_hashes_checked=len(plan['source_hashes']),
        saved_data_hashes_checked=len(plan['saved_data_hashes']), runtime_seconds_to_audit=(datetime.now(timezone.utc)-started).total_seconds(),
        automatic_next_stage=False, additional_candidates=0, confirmation_topup=0,
        population_tail_bound=None, old_33_of_40_preserved=True)
    write_report(plan, public, accounted, cells, group_rows, audit)
    audit['sealed_public_results'] = {p.name: x.prepare.sha(p) for p in sorted(public.iterdir()) if p.is_file() and p.name != 'END_AUDIT.json'}
    audit['report_sha256'] = x.prepare.sha(HERE / 'REPORT.md')
    x.study.save(public / 'END_AUDIT.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

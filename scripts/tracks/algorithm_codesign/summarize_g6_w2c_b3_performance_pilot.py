#!/usr/bin/env python3
"""Summarize artificial pilot results only; never reads scientific raw values."""
from collections import defaultdict
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import statistics as st
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT/'.pilot-runs'
DOC = ROOT/'docs/research/track_b_g6_w2c_b3_performance_scaling_pilot_20261019'
BASE = '07f65831f045bf1fdbd4b370e44c7f59a9901ddd'
SUITES = ['scaling', 'optimizations', 'hotpaths', 'stop']
DATA = {s: [json.loads(x) for x in (RAW/s/'results.jsonl').read_text().splitlines()] for s in SUITES}


def median(rows, key):
    return st.median(r[key] for r in rows)


def group(rows, keys):
    result = defaultdict(list)
    for r in rows:
        result[tuple(r.get(k) for k in keys)].append(r)
    return result


def md_table(headers, rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join(
        '| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)


def write(name, text):
    (DOC/name).write_text(text)


def signatures(row):
    return {w['task']['key']: (w['result']['trace_sha256'], w['normalized_file_sha256'],
                              w['normalized_chain_sha256'], w['result']['axes']) for w in row['workers_results']}


def main():
    # Additional analysis-time equality checks for the reuse/pass experiments.
    for case in ['startup64_tiny', 'pass_overlap16']:
        rows = [r for r in DATA['optimizations'] if r['case'] == case]
        assert all(signatures(r) == signatures(rows[0]) for r in rows)
    for keys, rows in group(DATA['hotpaths'], ['kind', 'bits', 'size']).items():
        if keys[0] != 'io':
            assert len({r['checksum'] for r in rows}) == 1, keys
    for suite in ['scaling', 'optimizations']:
        assert all(w['status'] == 'OK' for r in DATA[suite] for w in r['workers_results'])
    measurements = DOC/'measurements'
    measurements.mkdir(exist_ok=True)
    for suite in SUITES:
        with gzip.GzipFile(filename=str(measurements/(suite+'.jsonl.gz')), mode='wb', mtime=0) as out:
            out.write((RAW/suite/'results.jsonl').read_bytes())
        for name in ['environment_start.json', 'environment_end.json', 'source_inventory.json', 'ARTIFICIAL_SUITE_STOP.json']:
            shutil.copyfile(RAW/suite/name, measurements/(suite+'_'+name))
    # Publish complete representative logs and all failure fixtures, retaining
    # the remaining complete raw files locally with their hashes in the summaries.
    archive_paths = [RAW/'scaling/pair8_mixed_r0_w1',
                     RAW/'optimizations/heavy_reference_b0_r0',
                     RAW/'optimizations/heavy_orjson_b0_r0', RAW/'stop']
    with (measurements/'representative_raw_jsonl.tar.gz').open('wb') as handle:
        with gzip.GzipFile(filename='', mode='wb', fileobj=handle, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as tar:
                for folder in archive_paths:
                    for path in sorted(folder.rglob('*.jsonl')):
                        tar.add(path, arcname=str(path.relative_to(RAW)))
    scaling = group(DATA['scaling'], ['case', 'workers'])
    metrics = []
    for (case, workers), rows in sorted(scaling.items()):
        base = scaling[(case, 1)]
        speedup = median(base, 'wall_s')/median(rows, 'wall_s')
        metrics.append({'case': case, 'workers': workers, 'repeats': len(rows),
                        'trials': rows[0]['completed'], 'wall_s': median(rows, 'wall_s'),
                        'wall_min_s': min(r['wall_s'] for r in rows), 'wall_max_s': max(r['wall_s'] for r in rows),
                        'total_cpu_s': median(rows, 'total_cpu_s'),
                        'worker_cpu_s': median(rows, 'worker_total_cpu_s'),
                        'parent_cpu_s': median(rows, 'parent_cpu_s'),
                        'throughput': median(rows, 'throughput'), 'speedup': speedup,
                        'parallel_efficiency': speedup/workers,
                        'peak_worker_hwm_mib': median(rows, 'peak_worker_hwm_bytes')/(1 << 20),
                        'sampled_parent_workers_peak_rss_mib': median(rows, 'sampled_parent_and_workers_rss_peak_bytes')/(1 << 20),
                        'output_mib': median(rows, 'bytes')/(1 << 20),
                        'logical_output_mib_s': st.median(r['bytes']/r['wall_s']/(1 << 20) for r in rows),
                        'sum_worker_setup_cpu_s': st.median(sum(w['setup_cpu_ns'] for w in r['workers_results'])/1e9 for r in rows)})
    with (measurements/'scaling_summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), lineterminator="\n");writer.writeheader();writer.writerows(metrics)
    write('parallel_scaling_results.md', '# Process scaling実測\n\n3反復の中央値。min/maxはCSV、worker個別CPU・setup・RSS・I/Oは圧縮JSONLに保存。speedupは同じ人工task集合の1 worker中央値との比。総CPUはworker＋parent。\n\n'+
          '\n'.join('## '+case+'\n\n'+md_table(
              ['Workers', 'Wall s', 'Total CPU s', 'Trials/s', 'Speedup', 'Efficiency', 'Worker HWM MiB', 'Sampled total RSS MiB', 'Output MiB/s'],
              [[m['workers'], f"{m['wall_s']:.4f}", f"{m['total_cpu_s']:.4f}", f"{m['throughput']:.0f}",
                f"{m['speedup']:.2f}×", f"{100*m['parallel_efficiency']:.1f}%", f"{m['peak_worker_hwm_mib']:.1f}",
                f"{m['sampled_parent_workers_peak_rss_mib']:.1f}", f"{m['logical_output_mib_s']:.1f}"]
               for m in metrics if m['case'] == case]) for case in sorted({m['case'] for m in metrics}))+
          '\n8-pair構造では8 workerでlight 6.79×、mixed 6.02×、heavy 3.95×。heavyではB側の長いpairがcritical pathとなり、worker追加だけで8倍にはならない。64-task測定は本番topologyの証明ではない。\n\n'+
          '64-taskの最高throughputは64 worker、並列効率が最も高い並列設定は2 worker。32 workerはCPU費用をほぼ維持して約23倍、64 workerは約37倍にwallを短縮した一方、総CPUは約8.6%増加した。目的がwall最小化かCPU効率かで選択が異なる。RSSは共有pageを含む和で、物理memoryの固有使用量ではない。出力はpage cacheへのlogical throughputで、fsyncを伴うdevice性能ではない。\n')
    optim = group(DATA['optimizations'], ['case', 'variant', 'buffer_bytes'])
    opt_rows = []
    for case in ['mixed', 'heavy']:
        ref = optim[(case, 'reference', 0)]
        for variant, buffer in [('reference', 0), ('recurrence', 0), ('gmp', 0), ('orjson', 0), ('orjson', 65536)]:
            rs = optim[(case, variant, buffer)]
            opt_rows.append([case, variant+(' +64KiB' if buffer else ''), f"{median(rs,'wall_s'):.4f}",
                             f"{median(rs,'total_cpu_s'):.4f}", f"{median(ref,'wall_s')/median(rs,'wall_s'):.3f}×",
                             f"{median(ref,'total_cpu_s')/median(rs,'total_cpu_s'):.3f}×",
                             f"{median(rs,'peak_worker_hwm_bytes')/(1<<20):.1f}"])
    hot = group(DATA['hotpaths'], ['kind', 'bits', 'size', 'variant'])
    hot_rows = []
    for bits in [32, 96, 256]:
        ref = hot[('rational', bits, None, 'reference')]
        for variant in ['reference', 'recurrence', 'gmp']:
            rs = hot[('rational', bits, None, variant)]
            hot_rows.append([f'Rational {bits} bit/J32', variant, f"{1e6*median(rs,'cpu_per_op_s'):.2f}",
                             f"{median(ref,'cpu_per_op_s')/median(rs,'cpu_per_op_s'):.3f}×"])
    for kind, variants in [('serialization', ['reference', 'fused', 'orjson']),
                           ('io', ['reference', 'fused', 'orjson', 'orjson_buffer']),
                           ('observer', ['none', 'stage', 'function', 'stage_function'])]:
        ref = hot[(kind, None, None, variants[0])]
        for variant in variants:
            rs = hot[(kind, None, None, variant)]
            hot_rows.append([kind, variant, f"{1e6*median(rs,'cpu_per_op_s'):.2f}",
                             f"{median(ref,'cpu_per_op_s')/median(rs,'cpu_per_op_s'):.3f}×"])
    for size in [512, 16384, 262144]:
        ref = hot[('hash', None, size, 'concat')]
        for variant in ['concat', 'incremental']:
            rs = hot[('hash', None, size, variant)]
            hot_rows.append([f'hash padding {size}B', variant, f"{1e6*median(rs,'cpu_per_op_s'):.2f}",
                             f"{median(ref,'cpu_per_op_s')/median(rs,'cpu_per_op_s'):.3f}×"])
    write('optimization_results.md', '# 高速化prototypeの実測\n\n全て非採択。既存科学SOURCEは変更していない。下表は8 worker、各3反復の中央値。\n\n'+
          md_table(['Workload', 'Variant', 'Wall s', 'Total CPU s', 'Wall speedup', 'CPU speedup', 'Worker HWM MiB'], opt_rows)+
          '\nrecurrenceは級数の冪の漸化式＋fused canonical、gmpはGMP級数＋fused canonical、orjsonはGMP級数＋native JSON。combinedの効果を単独変更の効果と混同しない。light/mixed系では初回importや残るgenerator/evidence処理が支配し、GMPのend-to-end利益は小さい。heavyではGMP候補でwall約1.34×、総CPU約1.30×。native JSON追加のend-to-end利益は小さく、GMPのみとの差を優位と断定しない。追加libraryによりworker HWMは約20.9→23.4 MiB。出力の本質内容と正規化bytes/hashは全variantで一致。\n\n## Hot-pathの分離測定\n\n'+
          md_table(['Synthetic operation', 'Variant', 'CPU µs/op', 'CPU speedup vs reference'], hot_rows)+
          '\n有理数は最終的なFractionへの変換・約分まで含む。GMPは32/96/256 bitで約1.33/1.18/1.43×。serializationはfusedで約1.34×、nativeで約3.04×、Stream全体では約1.58×。microの最大倍率を全体倍率として使わない。hashのincremental更新は入力UTF-8 bytesを事前生成する条件で、大payloadの一時文字列・再encodeを避ける利益がある。新しいchainは依存するため並列化していない。\n\n'+
          'stage/function observerは小さなhash kernelのCPUをそれぞれ約5倍/26倍、併用で約46倍にした。これは小kernelに対する比で、実G7全体の倍率ではない。別の人工paired passではinstrumented/baseline task CPU比を記録した。FunctionObserverを除けば細かいfunction費用が失われ、stageだけにしても旧R1の同じinstrumentationではなくなる。軽量化はB候補。\n\n'+
          '64 tiny tasksのpool再利用はfreshの約0.1086秒→0.0347秒（約3.13×）、CPU約0.647→0.223秒。full-Nでは起動費が償却されるためこの倍率を移さない。reuseはprocess/HWM/setup独立性を変えるB候補。\n\n'+
          '16 pass-overlap tasksは1/8/16 workerで約1.818/0.293/0.226秒、逐次比約6.20/8.03×。baselineとinstrumentedの資源競合・順序・STOP意味論を変更するためB候補。pair分割は今回は優先しなかった。8-pairのload imbalanceには有望だが、bit offset/state/setup/STOPの追加設計が必要。\n\n'+
          '64KiB bufferingはheavyでnative unbufferedとの差が約0.4%に留まった。accepted-write prefix保証は人工crashで変わるため、R1採用は推奨しない。正常終了時のbytes一致だけでは不十分。GPUは4台のA100を確認したが、この厳密・可変長処理を移植する有利な候補が得られず、実測していない。CPUより遅いと実証したわけではない。\n')
    failure_rows = [r for r in DATA['stop'] if r['case'] == 'injected_sample_failure']
    stop_table = []
    for (workers,), rs in sorted(group(failure_rows, ['workers']).items()):
        stop_table.append([workers, f"{min(r['starts_after_failure'] for r in rs)}–{max(r['starts_after_failure'] for r in rs)}",
                           f"{min(r['completed_after_failure'] for r in rs)}–{max(r['completed_after_failure'] for r in rs)}",
                           f"{1e3*median(rs,'stop_detection_to_last_worker_collection_s'):.2f}",
                           f"{min(len(r['not_launched']) for r in rs)}–{max(len(r['not_launched']) for r in rs)}"])
    write('contract_compatibility_and_stop.md', '# R1互換性・F01・STOP\n\nAは科学・保存・計測・逐次STOPの意味を維持し得る候補、Bは契約変更が必要な候補、Cはそのままでは不適格。Aでも現行SOURCE pinへ無断反映せず、別途採用判断・新SOURCE固定が必要。人工PASSは実F3全domainでの統合認定ではない。\n\n'+
          md_table(['Candidate', '分類', '根拠・残る条件'], [
              ['冪の漸化式', 'A候補', 'exact partial/remainder/端点一致、Jと丸め・guard・再queryは不変にする'],
              ['GMP rational adapter', 'A候補', 'Fractionへ戻す型境界を維持。実F3全体は未接続・未検証'],
              ['fused/native canonical', 'A候補', 'Unicode/surrogate/DEL fallback、元bytesとF01を人工確認'],
              ['incremental SHA更新', 'A候補', '同じUTF-8列とchain順序。転送・preencode条件を実装時に維持'],
              ['8 pair process並列', 'B', 'fixed order/比較競合/最初の失敗時の逐次prefixが変わる'],
              ['pass並列・worker reuse', 'B', 'pass順序、setup/HWM、CPU観測の比較条件が変わる'],
              ['buffering', 'B', 'emit完了とOS write受理が一致しなくなる'],
              ['function profiling省略', 'B', '研究で求めるfunction費用内訳が失われる'],
              ['float置換・余剰sampling無課金・failureをZERO化', 'C', '厳密性/予算/STOP違反。実装・採択していない']])+
          '\n## 人工failure injection\n\n全workerのfirst sampleをbarrierでそろえ、最初のpairのX trial 3で人工失敗。共有lock下のtrial開始gateが失敗時刻を記録し、その後の新規開始を拒否。未開始pairはdispatchしない。開始済みtrialは安全に完了・prefix保存し、次のgateで停止する。強制killや全process走査を使わない。\n\n'+
          md_table(['Workers', '通知後の新規開始', '通知後の完了trial', '通知→全join median ms', '未開始pair数'], stop_table)+
          '\n「開始」はlock下でtrialをadmitした時点。通知から全joinまではcooperative終了とIPC/cleanupを含み、物理的な即時停止ではない。失敗通知後0新規開始を保証しても、開始済みの複数trialは存在する。失敗を最初に観測・通知した時刻が基準であり、未検知の失敗発生を先取りできない。従って逐次R1と完全同義ではなく、成功時のtrace一致だけでAに分類しない。\n\n'+
          '実科学契約に採用するなら、すべての開始済みcall（失敗・取消・後で破棄するものも含む）を予算へ課金し、atomic reservation、最大in-flight、失敗の順序/採用prefix、既存grant/claimとのbindingを新契約で定義する必要がある。今回のControlは人工start gateのみで、実科学call budget reservationやadmissionを提供しない。\n\n'+
          'F01は失敗sampleのresult/eventを前trialから流用しない。completed prefixはemit正常return後だけ更新し、streamの完全行prefixとtrial completionを分離する。Python/native unbufferedそれぞれで元F01全26 testsを維持。partial write、file hash、flush、encoder失敗を含む。\n\n'+
          '人工os._exit(17)では、unbufferedはemit受理2行/496 bytesを再読でき、64KiB bufferedは受理2行を報告していても再読0行/0 bytesだった。電源断耐性の証明ではなく、bufferingによる保証差の具体例である。どちらもretry/resumeなし。同一run directory再使用は拒否。\n')
    projection = []
    for case in ['pair8_light', 'pair8_mixed', 'pair8_heavy']:
        for workers in [1, 8]:
            rs = scaling[(case, workers)]
            wall = median(rs, 'wall_s')*31710848/rs[0]['completed']
            cpu = median(rs, 'total_cpu_s')*31710848/rs[0]['completed']
            projection.append({'scenario': case, 'workers': workers, 'projected_wall_s': wall,
                               'projected_total_cpu_s': cpu, 'basis_trials': rs[0]['completed'],
                               'instrumented_treated_like_baseline': True, 'not_real_G7_prediction': True})
    (measurements/'conditional_projection.json').write_text(json.dumps(projection, indent=2))
    ratios = []
    for row in DATA['optimizations']:
        if row['case'] == 'pass_overlap16' and row['workers'] == 1:
            costs = {p: sum(w['cpu_ns'] for w in row['workers_results'] if w['task']['pass_name'] == p)
                     for p in ['baseline', 'instrumented']}
            ratios.append(costs['instrumented']/costs['baseline'])
    write('full_workload_runtime_projection.md', '# 31,710,848 callsへの条件付き外挿\n\n**実G7の所要時間は未測定・未確定。本表は人工workloadと同じ1 call費用、同じ混合率・pair負荷・出力密度を仮定した機械的な規模換算であり、B3の予報・上下界ではない。** baseline相当の人工callを31,710,848回処理した場合を示す。正式二passのinstrumentation差を省いているため、このまま本番時間として使えない。\n\n'+
          md_table(['人工scenario', 'Workers', '実測trials', '外挿Wall h', '外挿CPU h'],
                   [[r['scenario'], r['workers'], r['basis_trials'], f"{r['projected_wall_s']/3600:.3f}",
                     f"{r['projected_total_cpu_s']/3600:.3f}"] for r in projection])+
          '\n軽量/中間/重いscenarioは楽観・中間・保守的な人工仮定の比較としてのみ読める。実G7がこの範囲に入る証拠はない。64 workerの機械scaling倍率は8 pairへ適用しない。\n\n'+
          f'別のmixed人工pass比較では、instrumented/baselineのtask CPU比は中央値 **{st.median(ratios):.3f}×**。仮に同じ比が本番で成り立つ場合、二つのfull-N passを両方baseline扱いした表のCPU部分は約(1+{st.median(ratios):.3f})/2倍になる。ただしこの倍率自体が人工endpoint・対象Python関数の呼出頻度に依存し、実G7へ転用できない。\n\n'+
          '本来は `T = S + 256*c_correctness + 15,855,296*c_baseline + 15,855,296*c_instrumented + Ω` を実入力の各arm/layer別で積算する。並列wallはCPU合計をworker数で割るだけではなく、各pairの長さとpass barrierを考慮する。CPU-boundなら負荷偏り・共有競合、I/O-boundなら出力bytes/実効帯域の床が残る。\n\n'+
          '不足しているのは実G7のpath発生率、bit draw数、packet/conditional law費用、operand長、F3 source/state guard・angle/requery費用、全Nでのcompact row割合、fine profiler呼出頻度、実際のoutput帯域である。今回のuniform Nと本番のarm別保存Nも異なる。短いfixtureのsetup/full-prefix費用を線形に増やす影響もある。科学保存値の新規再解析でこれを補っていない。\n')
    stages = defaultdict(list)
    for case in ['mixed', 'heavy']:
        rs = optim[(case, 'reference', 0)]
        totals = defaultdict(list)
        for row in rs:
            values = defaultdict(int)
            for w in row['workers_results']:
                for name, cost in w['stage_cost'].items():
                    values[name] += cost['cpu_ns']/1e9
            for name, value in values.items():
                totals[name].append(value)
        stages[case] = sorted(((name, st.median(v)) for name, v in totals.items()), key=lambda x: -x[1])
    write('bottleneck_analysis.md', '# 静的構造と人工測定の対応\n\n固定F01 driverは各pair fresh process、保持stateをX/Yで共有。Fullはraw-validならcoin reject前にpacketを払い、子lawの構築/sort・有理root・weightを処理する。canonicalもconditional lawを再構築する。Bでは毎acceptedにF3/canonical audit・source/state fingerprint・元event再query・数学的angleを追加する。これらは静的な呼出順の確認であり、実費の順位ではない。\n\n'+
          '静的参照：`g7_generator.py` のDeterministicBits/_law/FullReturnGenerator.sample/packet、`return_aggregation.py` のdigital_parent/dyadic_distribution、`g6_f3_generator.py` の_guard/angle_for_event、`g6_f3_precision.py:145` の_atan_endpoint、F01 driverのMeter/FunctionObserver/Stream/run_pair。いずれもimport・実行せず読んだ。\n\n'+
          '\n'.join('## '+case+' 人工stage CPU中央値\n\n'+md_table(['Stage', 'CPU s'], [[name, f'{cost:.6f}'] for name, cost in stages[case][:10]]) for case in ['mixed', 'heavy'])+
          '\nheavyの人工角度級数が支配的で、GMP/漸化式の効果を確認できた。mixedではevidence/canonical化やgenerator処理も大きく、angleだけを改善しても全体利益は小さい。serializationのmicro改善がend-to-endで希釈されることも確認した。実G7の最大ボトルネックや4方式の科学的な優劣は決めていない。\n\n'+
          '新prototypeはguardやevent再queryを省いていない元run_pairの骨格を使うが、人工state hashは実F3の全SOURCEファイル照合ではない。人工packet/lawもReturnKernelと同じ計算ではなく、zero率・operand幅は設計値。実SOURCEのCPU profilerや数値再解析を実施していない。\n')
    write('handoff.md', '# 技術ハンドオフ・Q1–Q8\n\n**人工性能検証完了、全候補非採択。科学実行・科学SOURCE統合なし。公開確認はpublication_receipt.jsonを参照。**\n\n'+
          '1. **Q1 重い処理**：静的にはFull packet/子law、有理数、B guard/requery/angle、canonical/save/profilerが候補。人工heavyではangleが支配し、mixedではevidence/generator/saveも無視できなかった。実G7の順位は未確定。\n'+
          '2. **Q2 scaling**：8 pairのlight/mixed/heavyで8 workerは6.79/6.02/3.95×。64人工taskの2/4/8/16/32/64 workerは約1.95/3.81/7.26/13.53/23.41/36.78×。\n'+
          '3. **Q3 資源と効率**：64 taskのCPUは1 worker約11.643秒→64 worker約12.645秒、sampled aggregate RSS約53→1577 MiB。最高throughputは64 worker、並列効率重視なら2 worker。32 workerはCPU維持とwall短縮の折衷。\n'+
          '4. **Q4 非並列改善**：native canonical micro約3.04×、Stream全体約1.58×。exact GMP級数micro約1.18–1.43×。heavy end-to-endでGMP候補wall約1.34×/CPU約1.30×。漸化式は小さく安定した利益。mixedの全体改善は小さい。大payload hashはpreencode+incrementalが有効。\n'+
          '5. **Q5 本番時間**：人工scalingを31710848へ条件付き換算した表まで。実G7の数値予報や上下界は出せない。instrumented費用、実path/operand/guard、実出力が未測定。\n'+
          '6. **Q6 互換性**：fused/native canonical、冪漸化式、exact rational adapter、同じbytesのincremental hashはA候補。process/pass並列、pool、buffer、profiling省略はB。8 worker failureでは通知後の新規開始0でも、開始済みtrialが中央値7件完了した。逐次R1のSTOPを維持したと主張しない。\n'+
          '7. **Q7 実G7 pilot**：実用的な時間予測には実入力観測が必要。計画案は新しい専用one-shotでcorrectness16/cell=256、その後baseline64/cell=1024、instrumented同数1024、合計2304 callsを上限固定する小pilot。coverage不足でも追加しない。必要ならさらに縮小できるがprecision/setup以外の費用推定は弱くなる。F01を含む新しい公開SOURCE commit、R1 math-only τ=10^-6/Kφ=32/J=32、固定input/policy/N、正式pilot契約digest、fresh namespaceと実承認参照/atomic admission、retry0/resumeなしを先に固定し、別途ユーザーの科学実行承認を受ける。今回はSOURCE統合・contract採択・claim発行・pilotを実施しない。既存本番one-shotを消費しない。\n'+
          '8. **Q8 導入候補の優先順位**：下表。性能だけで採択せず、現在のR1を変更するものは別判断。\n\n'+
          md_table(['優先', '候補', '推奨理由・条件'], [
              [1, 'fused canonical＋必要ならnative backend', 'bytes/F01を人工検証。依存なし版から導入検討でき、nativeの単独micro効果も明確'],
              [2, '角度級数の冪漸化式', '厳密式を維持し外部依存なし。GMP追加より統合範囲が小さい'],
              [3, 'GMP exact adapter', 'heavy全体のCPU/Wall改善。型境界と全domain互換性・専用依存の固定が必要'],
              [4, '8 pair並列', 'wall利益は最大級だがB。失敗予算・prefixと費用比較条件を変更する判断が必要'],
              [5, '大payload incremental hash', '同じbytesを維持できる場合。小rowでは利益が小さく優先度を下げる'],
              [6, 'pass overlap/pool', 'B。full-Nへの利益は未確定、tiny task倍率は流用しない'],
              ['見送り', 'buffer・GPU・pair分割', 'bufferは小利益とprefix保証低下。GPU/pair分割は未測定、今回のCPU候補より優先しない']])+
          '\n次の担当（未着手）：**ユーザー**がA候補の統合技術準備を依頼するか、Bの契約変更検討や別途実入力pilotを認めるか判断する。依頼された**Codex**だけが、Codex Aの公開固定成果とF01のbindingを確認して新しい独立scopeで準備する。現行admission準備へ自動merge/rebaseしない。公開・保全確認後mandatory STOP。\n')
    write('README.md', '# B3 並列化・高速化の人工性能検証\n\n2026-10-19 JST。状態：**ARTIFICIAL_PERFORMANCE_COMPLETE_NOT_ADOPTED**。独立branch `track-b-g6-w2c-b3-performance-scaling-pilot-20261019`。\n\n'+
          '- 現行8 pair構造：8 workerでlight/mixed/heavyのwallを **6.79/6.02/3.95倍**短縮。\n'+
          '- 64人工task：64 workerで **36.78倍**、CPUは約11.643→12.645秒。これは8 pairの本番倍率ではない。\n'+
          '- heavyのGMP候補：8 worker wall **約1.34倍**、総CPU **約1.30倍**改善。mixedの利益は小さい。\n'+
          '- 全**172人工tests PASS**、全測定出力のchain/file readback、正規化したevent/trace/bit/bytes互換性確認。\n'+
          '- 並列化は通知後0新規開始でもin-flightが完了し、**現行逐次R1と同じSTOPではない**。buffer crashでは受理2行が残らない例を確認。\n\n'+
          '実G7 sampling・量子回路build/compile/synthesis・科学保存値再解析・実authorization/grant/claim発行消費は全て0。既存科学SOURCE、契約、raw結果、ledger/marker、他worktreeを変更していない。GPUは強制せず、CPUを優先して実測した。\n\n'+
          '環境：AMD EPYC 7532、64 logical CPU（lscpu上SMTなし）、RAM約503 GiB、起動時available約469 GiB、Python 3.12.3、専用venvのgmpy2 2.3.2/orjson 3.13.0。既存quotaを変更しない。共有OS cacheはuncontrolled。\n\n'+
          'SOURCE：F01基点 `'+BASE+'`。scaling SOURCE `734519386b38c999fa7f502d800e9e7e2df9402f`、その他suite SOURCE `2371261024176d9e343bc8c8109758d5571cbd59`。後者はnative encoderのDEL fallbackとbuffer crash probeを追加。scalingは旧revisionのreference経路だけを実行し、native encoderは使用していない。測定中のSOURCE書換えは行っていない。\n\n'+
          '- [測定方法・再現](methodology.md) / [SOURCE inventory](protected_source_inventory.json)\n'+
          '- [ボトルネック](bottleneck_analysis.md) / [scaling表](parallel_scaling_results.md)\n'+
          '- [高速化比較](optimization_results.md) / [条件付き外挿](full_workload_runtime_projection.md)\n'+
          '- [R1・F01・STOP](contract_compatibility_and_stop.md) / [Q1–Q8・優先順位・次の担当](handoff.md)\n'+
          '- [全172 testsログ](artificial_regression.log) / [機械可読CSV](measurements/scaling_summary.csv)\n'+
          '- measurements配下に全57 scaling/45 optimization/78 hot-path/14 STOP・crash測定の圧縮JSONL、環境、SOURCE hash、代表rawを公開。全rawは専用worktreeの `.pilot-runs/` に保持し、per-file SHA/bytesは各測定summaryに記録。代表raw以外の大量trial JSONLはGitへ重複格納していない。\n\n'+
          '正式採択・本番統合・実入力pilotは別途ユーザー判断。公開照合は [publication_receipt.json](publication_receipt.json)、保全は [verification.json](verification.json) を参照。確認後mandatory STOP。\n')
    # Update the count without overwriting historical exploratory checks.
    method = (DOC/'methodology.md').read_text().replace('全170人工回帰', '全172人工回帰').replace('新しい8件', '新しい10件')
    write('methodology.md', method)
    exploratory = (DOC/'exploratory_checks.md').read_text()
    if 'U+007F' not in exploratory:
        exploratory += '\n- Additional ASCII probe found a real native-encoder mismatch for U+007F (DEL). Fixed by reference fallback, added all-ASCII regression before native optimization measurements. Completed scaling reference-path data retain the earlier SOURCE inventory; no live measured source was edited.\n'
    write('exploratory_checks.md', exploratory)
    manifest = {'artificial_only': True, 'runs': {s: len(DATA[s]) for s in SUITES},
                'all_success_path_workers_OK': True, 'all_worker_output_chain_file_readbacks_pass': True,
                'all_cross_setting_equivalence_pass': True, 'all_non_IO_micro_checksums_match': True,
                'tests': {'passed': 172, 'failures': 0, 'errors': 0, 'skipped': 0},
                'quantum_science_sampling': 0, 'science_numerical_reanalysis': 0,
                'quantum_build_compile_synthesis': 0, 'real_authorization_claim_issuance_consumption': 0,
                'production_integration': False, 'scientific_contract_adoption': False,
                'payloads': {str(p.relative_to(DOC)): {'bytes': p.stat().st_size, 'sha256': sha256(p.read_bytes()).hexdigest()}
                             for p in sorted(measurements.iterdir())},
                'private_raw_retained_at': str(RAW)}
    write('verification.json', json.dumps(manifest, indent=2)+'\n')
    print('summarized', manifest['runs'], 'published measurement bytes', sum(p.stat().st_size for p in measurements.iterdir()))


if __name__ == '__main__':
    main()

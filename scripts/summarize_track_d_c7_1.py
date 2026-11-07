#!/usr/bin/env python3
"""Render stopped C7-1 saved scalar results; no circuits, fitting or statistics recomputation."""
from pathlib import Path
import csv
import json
from collect_track_d_c6b import read, write_new

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_d_c7_1_q16/2026-11-07/run01'
DOC=ROOT/'docs/research/track_d_c7_1_q16_20261107/RESULT.md'


def main():
    result=read(OUT/'result.json');stop=read(OUT/'STOP.json')
    if result['status']!='TECHNICAL_PASS_MANDATORY_STOP' or not stop['mandatory_stop']:
        raise ValueError('COMPLETE_TECHNICAL_RESULT_REQUIRED; preserve/report partial separately')
    if DOC.exists():raise ValueError('REPORT_ALREADY_EXISTS')
    verify=read(OUT/'verification_summary.json');stats=read(OUT/'paired_statistics.json')
    diag=read(OUT/'exploratory_diagnostics.json');cx=read(OUT/'cx_candidate_residuals.json')
    model=read(OUT/'frozen_model.json');source=read(OUT/'SOURCE.json')
    publication=read(OUT/'payload_publication.json')
    assert len(result['completed_ids'])==16 and not result['missing_or_failed_ids']
    assert result['source_input_preserved'] and verify['compile_calls']==64
    assert verify['raw_PASS']==16 and verify['compiled_postcheck_PASS']==64 and verify['paired_paths']==16
    conditions={}
    allowed=['q','h','time','cohort','endpoint','model','cut_after_complete_step','AB_order',
        'initial_state','state_preparation','scalar','extracted_identity','scalar_identity_occurrences',
        'per_step_scalar_phase','aggregated_scalar_phase','per_step_chi','phase_chi','lambda_R',
        'normalization_applied_to_unitary','attenuation_applied_to_unitary','reference_gadgets',
        'short_gadget_repetition','strong_symmetric','reference_optimization']
    for sid in result['planned_ids']:
        d=read(OUT/'local'/sid/'conditions.json')
        conditions[sid]={k:d[k] for k in allowed}
    write_new(OUT/'path_conditions_scalar.json',conditions)
    metrics=[r['metric'] for r in stats['rows'] if r['axis']=='X']
    table={(r['axis'],r['metric']):r for r in stats['rows']}
    with (OUT/'paired_differences.csv').open('x',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['sample_id','axis',*metrics])
        for i,sid in enumerate(result['planned_ids']):
            for axis in ['X','Y']:
                writer.writerow([sid,axis,*[table[axis,g]['paired_deltas_by_sample'][i] for g in metrics]])
    outside=[dict(axis=r['axis'],metric=r['metric']) for r in diag['rows']
             if r['population_line_diagnostic']=='EXPLORATORY_OUTSIDE_WIDTH']
    summary=dict(status=result['status'],execution_source=result['source_commit'],q=16,n_whole_paths=16,
        raw_PASS=verify['raw_PASS'],postcheck_PASS=verify['compiled_postcheck_PASS'],paired_paths=verify['paired_paths'],
        CX_candidate_status=cx['status'],primary_status=diag['status'],secondary_outside=outside,
        statistical_qualification=diag['statistical_qualification'],raw_payload_public=False,
        source_input_preserved=result['source_input_preserved'],next_stage_authorized=False,mandatory_stop=True)
    write_new(OUT/'report_summary.json',summary)
    def number(value):return format(value,'.12g') if isinstance(value,float) else str(value)
    def description(row):
        return f"{number(row['mean'])} ± {number(row['sample_SD'])} [{number(row['min'])}, {number(row['max'])}]"
    prices=['|axis|資源|strong symmetric|Reference|Reference−strong|','|---|---|---|---|---|']
    for r in stats['rows']:
        prices.append('|'+r['axis']+'|'+r['metric']+'|'+description(r['symmetric'])+'|'+description(r['reference'])+'|'+description(r['paired_delta'])+'|')
    residuals=['|path|X ΔCX|X residual|Y ΔCX|Y residual|','|---|---:|---:|---:|---:|']
    cr={r['axis']:{v['sample_id']:v for v in r['per_path']} for r in cx['rows']}
    for sid in result['planned_ids']:
        residuals.append('|'+sid+'|'+ '|'.join(str(cr[a][sid][k]) for a in ['X','Y'] for k in ['paired_delta','residual'])+'|')
    diagnoses=['|axis|資源|観測−予測|SE|τ|一次margin/判定|別population margin/判定|',
               '|---|---|---:|---:|---:|---|---|']
    for r in diag['rows']:
        diagnoses.append('|'+r['axis']+'|'+r['metric']+'|'+ '|'.join(number(r[k]) for k in ['prediction_error','holdout_mean_nominal_SE','tau'])+'|'+number(r['primary_margin'])+' / '+r['primary_status']+'|'+number(r['population_line_margin'])+' / '+r['population_line_diagnostic']+'|')
    predictions=['|axis|資源|strong予測|Reference予測|paired予測|paired観測|', '|---|---|---:|---:|---:|---:|']
    for r in diag['rows']:
        predictions.append('|'+r['axis']+'|'+r['metric']+'|'+ '|'.join(number(v) for v in [r['method_predictions']['symmetric_directional'],r['method_predictions']['reference'],r['predicted_paired_delta'],r['observed_paired_mean']])+'|')
    a=read(ROOT/'artifacts/track_d_c6a_paired_price/2026-11-06/run01/paired_statistics.json')
    b=read(ROOT/'artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/paired_statistics.json')
    old={q:{r['metric']:r['paired_delta']['mean'] for r in group['rows'] if r['axis']=='X'}
         for q,group in a['by_q'].items()}
    old['8']={r['metric']:r['paired_delta']['mean'] for r in b['rows'] if r['axis']=='X'}
    comparison=['|資源/X|q2 n16|q4 n16|q8 n32|q16 n16|','|---|---:|---:|---:|---:|']
    for g in metrics:
        comparison.append('|'+g+'|'+ '|'.join(number(old[q][g]) for q in ['2','4','8'])+'|'+number(table['X',g]['paired_delta']['mean'])+'|')
    pairs=verify['paired_checks']
    paired_max=max((check['max_entry'] for row in pairs for check in row['checks'].values() if isinstance(check,dict) and 'max_entry' in check),default=None)
    leakage_max=max((r['max_reference_leakage_L2'] for r in verify['reference_leakage'] if isinstance(r,dict) and 'max_reference_leakage_L2' in r),default=None)
    text=f'''# Track D C7-1：q16独立whole-path探索的paired価格検証

2026-11-07 JST。**全16 pathのraw correctness、全64 P0 wrapper postcheck、最終paired X/Y・vacuum χはPASS。CX候補の観測判定は `{cx['status']}`、凍結予測への一次診断は `{diag['status']}`。確認的統計保証、全path公式、q≥32・全13-round費用優位は未認定。**

[固定PLAN](PLAN.md)／[result][RESULT]／[STOP][STOP]／[検証集約][VERIFY]／[科学scalar要約][SUMMARY]。科学実行と許可範囲の整理を終了し、公開照合後mandatory STOPとする。

## 固定条件・SOURCE・入力・seed

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q16のみ、complete step8後cut。元Hamiltonian Hの厳密groundを仮定し、準備費除外。ground vectorを用いず状態非依存observableと全256 system入力列で検証した。新ground solve/DF fitなし。

実行SOURCE `{result['source_commit']}`。C7-0公開基点 `8b422c28c22666a32103fdae33a3c9b9512ef248` から専用branchへ分離し、sampling前にSOURCE tree `4b3dc546c4ba59b9ce8eb21e695d6eed4a9eeda2` と12公開ファイルのbytes一致を確認した。local/remote科学SOURCE commitは同一。元C SOURCEは `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。全SOURCE/input/package bindingは[SOURCE][SOURCE]にある。

packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、CDF endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、law記述 `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。signed atom/frame/sector/scalar/identity/normalization/attenuationと測定規約を保持し、weightをunitaryへ重複適用していない。

cohort `track_d_c7_1_q16_exploratory_paired_price_v1`、master2026110701、新domain `track_d_c7_1_q16_exploratory_paired_price_v1:independent_holdout:q16`。完全master tuple・seed keyは[事前seed表][SEED]、NumPy1.26.4 PCG64と既存trajectory_seed/canonical samplerを使用。q16_sample00..15、各occurrence0..15、各32 events。全16 record/256 occurrence/8192 eventsを一度だけ保存してhashを固定し、その後CAR/回路/価格を評価した。[event hash binding][EVENTS]

C6a96・C6b256既使用seedと非重複。旧q8 domainのunused文字列を未使用と扱わず、旧cohort巡回/prefix共有/短path反復なし。同pathを両方式・X/Yへ共有した。seed分離は数学的ideal joint-IIDの証明でなく、有限bit law距離・PRNG joint-lawは未資格。

C6b当初23725のsampling/raw/compileと承認済み44a58の残task引継ぎは過去証拠として保持し、今回へresume/承認を流用していない。旧SOURCE/result/raw/grant/ledger/marker/STOPを変更していない。

## 再利用・人工検証・実行件数

新q16専用request/seed/event接続・完全標本集計を追加した。旧C6a q2/4/n16と旧C6b q8/n32 guardは変更していない。旧入口が固定domainのため、新入口だけを最小追加し、統計/診断は同じC6b手順とC6a description helperを用いた。

C3 prepare_steps/physical_segment/mixed_head、B2/C6a CAR/head/signed atom cache、StrongBuilder、既存Reference actual inverse/χ、C5 wrapper_for/_checked_price、Stage A raw checker、C2 P0/postcheck、C3 pairedχを再利用した。新Gaussian合成器/sampler/compiler/numerical simulatorはない。新workerでC6b BatchedStateActionsを選び、同Aer double/native変換・copy削減・入力batch・宣言位相復元を用いた。

人工/回帰41件PASS（新7＋C6b9＋C6a7＋C5 8＋C3 4＋C2 2＋高速化4）。2-mode toy q16のAB/actual inverse/vacuumχ/X/Y、新seed/旧guard、mock canonical sampler、共分散伝播、CX反例の保存、raw PASS後4compileの投入を確認した。[人工記録][TEST]。人工PASSを分子性能や全law保証にしない。

|実処理|実施件数・結果|
|---|---|
|canonical occurrence sampling|256、追加標本0|
|whole-path raw correctness|16/16 PASS|
|P0 full-wrapper compile|64、各wrapper一回|
|compile後postcheck|64/64 PASS|
|最終paired X/Y・vacuum χ|16/16 PASS|
|新fit/追加calibration/N/Born shots/ground solve/DF fit|0|
|retry/resume・閾値緩和・完了task再実行|0|

独立pathとraw PASS後のcompileを並列pipeline化し、工程別完了memory資料と実空きCPU/RAMで投入した。旧worker数を恒常上限にせず、共有環境/他jobへの変更、正常job中断、実行SOURCE hot-editなし。速度最適性・実測比較なしのspeedupは主張しない。内部資源観測はローカル保全のみ。

## Whole-path・対照公平性・correctness

V=U15…U0=A B、B前8 step/A後8 step、gadget一つ。strongはuncontrolled Gaussian、directional中心、head relative fusion、tail同basis共有、15 step境界の合法half-sweep融合、scalar/identity位相集約。Referenceはactual A.inverse、analytic全pathχ、内部14境界融合、global cutの1境界融合損失、追加8 qubits/16 logical CSWAP/χ/readoutを含む。弱い対照、結果依存cut、短gadget反復、別回路の節約値加算なし。[pathごとのscalar条件][CONDITIONS]

rawでは独立CAR/whole積、AB=V、actual adjoint、unitarity、vacuum ray/χ、scalar/identity/sign、strong各branch、complex coherence、X/Y誘導observable・global phaseを照合。各path raw PASS後だけその4 wrapperへ進んだ。

P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation_degree1.0/qubits_initially_zero=false/topology-free/位相保持。component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fitなし。compile後CAR/raw/paired X/Y/χ/reference leakage・宣言位相を確認。17-qubit全Operatorを構成せず、全256 system入力・各131072出力振幅を扱った。

|保存検証の最大値|値|
|---|---:|
|raw component|{number(verify['raw_max_component'])}|
|compiled component|{number(verify['compiled_max_component'])}|
|compiled observable Frobenius|{number(verify['compiled_observable_max_frobenius'])}|
|最終paired/χ component（保存欄の集約）|{number(paired_max)}|
|Reference leakage（対応する保存欄がある場合）|{number(leakage_max)}|

個別検証は[全check集約][VERIFY]とpaths/に保存した。固定精度内のlocal binary64証拠であり、丸め誤差の厳密包絡・全support/全round実装誤差bound・immutable CI・独立詳細科学レビューではない。

## q16有限標本のfull-wrapper価格

以下は平均 ± 標本SD [観測min,max]、ddof1、**n=16 whole-path**。X/Yを32独立標本と数えない。[全価格][PRICES]／[paired統計・共分散][STATS]／[path別差CSV][CSV]。min/maxはlaw supportではない。

{chr(10).join(prices)}

追加register/CSWAP/actual inverse/χ/global cut損失/shellを含む実compile価格。qubitsは9/17の幅制約で、round/shotに加算しない。RZをTへ換算せず、filtered/total DAG depthを実機時間と同一視しない。routing/T合成/noise/reset/実機費は未評価であり0ではない。

## CX整数構造候補の残差

事前候補208−224qのq16値−3376に対し、残差=観測ΔCX+3376を整数として保存した。判定 `{cx['status']}`。[完全な残差・保存平均/SD/範囲][CX]

{chr(10).join(residuals)}

これは上の観測pathだけの一致/反例であり、全sampling lawの公式、母分散0、全q移送資格を認定しない。5%価格幅での診断と整数一致は別評価。残差が非零でもcorrectnessと契約が成立する場合は有効な結果として保持し、path除外・追加sampling・model変更をしない。

## 凍結モデルとの比較と不確かさ

C6a q2/q4保存係数・全48係数共分散をsampling前に凍結し、q16予測を保存した。q8をfit点へ追加せず、再fit0。以下は事前予測と今回の保存平均の対応。[凍結モデル][MODEL]

{chr(10).join(predictions)}

一次は凍結予測値に条件付け、e=観測−予測、SE=s16/√16、|e|+z SE≤τ。τ=1+.05×事前strong予測規模、family12/名目的alpha=.05、normal/Bonferroni。別population-line欄は|e|+z√(SE²+vcal)≤τ。q16予測の重み−6/7とC6a保存方式間共分散からvcalを伝播した。一次へ校正分散を重複加算せず、両estimandを区別する。[全診断][DIAG]

{chr(10).join(diagnoses)}

一次全体 `{diag['status']}`、別population-line欄の幅外は{len(outside)}量。これらは探索的・名目的余裕規則の判定であり、有効95%同時coverageや真の誤差がτ外という確認的結論ではない。全体STATISTICALLY_UNQUALIFIED。古典価格のalphaをRPE energy/failureへ混ぜない。

path間変動、係数校正不確かさ、model bias、q移送、finite-bit/ideal law、P0依存を分ける。小さなpaired分散は方式別絶対価格の予測精度を保証しない。共分散は方式/資源/axisの依存と特異性を保持し、共有係数を異なるroundで独立と扱わない。正常性/tail/母分散/ideal-IID/bias domainの保証は未取得。

## 保存q2/4/8との対応・q32以降の不足

次表は各既存JSONのpaired平均欄の転記、Xを表示する。既存価格の再compile・再fitはしていない。

{chr(10).join(comparison)}

qごとにpath長、標本、cut位置、境界数が変わるため、差をqだけの因果効果と断定しない。RZ/sizeの符号・観測範囲は上の保存表の範囲で述べ、qが大きければ必ず減るとはしない。小さなRZ差と絶対RZ価格の精度、depthのlane/critical-path残差を別に扱う。旧Stage A q1は保存cohort/event16 cut、C5 q2やC3 q4/q8は既知fixtureで、今回のfresh標本へ混ぜていない。

q16の有限技術証拠はq≥32を資格付けしない。次はユーザー／GPTが、今回のCX整数残差とRZ/size/depth残差を材料に、(i)SOURCE/P0の構造bridgeを限定して補う、(ii)未使用の別qを独立検証する、(iii)用途がpopulation校正線/絶対価格精度で校正不確かさが支配する場合だけ追加calibrationを設計する、のどれが必要か選ぶ。係数/特徴を変更するなら探索版として別保存し、新モデルの独立検証domainを別に定める。事後幅拡張・同標本への再fitによるPASS化・全q大量compileへの自動進行はしない。

B4-2の14738は元H厳密ground・ideal finite-RTE IID・exact native等の条件付き十分shot数として参照のみ。B5のfinite-bit law、shot/occurrence joint-law、全path/全round実装誤差・actual共通Nは未解決。本有限q16結果を価格×Nへ接続して全13-round費用を算定せず、N再算定/actual成功保証/B側SOURCE変更も行わない。

## 保全・公開・STOP

SOURCE/inputの実行前後照合は一致。元event/QPY/NPZ/reference配列はlocalに保持し、{publication['members']} memberの[hash索引][PAYLOAD]を公開する。raw payload追加公開0。[公開範囲][PUB]。内部CPU/RAM/runtime/thread、内部絶対path、資格情報、raw event内容を公開しない。collectorは保存check/価格統計を読み、回路検証・fit・sample統計を再実行していない。

**C7-1の許可済み実行・結果整理を終了。専用branchへ公開し、remote SHA/tree/公開bytes照合後mandatory STOP。** 次担当は**ユーザー／GPT**が次の目的・q-domain・構造モデル/追加検証/校正の必要性を判断する。**Codex**は新たな明示許可後だけ採択scopeを担当する。q≥32、新sampling/追加標本、再fit、確認的保証、全RPE費用、actual成功保証、詳細科学レビューへ自動進行しない。

'''
    links={'RESULT':'result.json','SOURCE':'SOURCE.json','STOP':'STOP.json','VERIFY':'verification_summary.json',
        'SUMMARY':'report_summary.json','SEED':'seed_plan.json','EVENTS':'event_bindings.json',
        'CONDITIONS':'path_conditions_scalar.json','PRICES':'prices.json','STATS':'paired_statistics.json',
        'CSV':'paired_differences.csv','CX':'cx_candidate_residuals.json','MODEL':'frozen_model.json',
        'DIAG':'exploratory_diagnostics.json','PAYLOAD':'payload_indices','PUB':'payload_publication.json'}
    for label,path in links.items():text+=f'[{label}]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/{path}\n'
    text+='[TEST]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/technical/synthetic_tests.json\n'
    DOC.write_text(text)
    print(json.dumps(summary))


if __name__=='__main__':main()

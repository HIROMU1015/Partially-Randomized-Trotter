#!/usr/bin/env python3
"""Render stopped C7-3 saved scalar results; no circuits, fitting or statistics recomputation."""
from pathlib import Path
import csv
import json
from collect_track_d_c6b import read, write_new

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_d_c7_3_q32/2026-11-07/run01_accelerated_completion'
DOC=ROOT/'docs/research/track_d_c7_3_q32_20261107/RESULT.md'


def main():
    result=read(OUT/'result.json');stop=read(OUT/'STOP.json')
    if result['status']!='TECHNICAL_PASS_MANDATORY_STOP' or not stop['mandatory_stop']:
        raise ValueError('COMPLETE_TECHNICAL_RESULT_REQUIRED; preserve/report partial separately')
    if DOC.exists():raise ValueError('REPORT_ALREADY_EXISTS')
    verify=read(OUT/'verification_summary.json');stats=read(OUT/'paired_statistics.json')
    diag=read(OUT/'exploratory_diagnostics.json');cx=read(OUT/'cx_candidate_residuals.json')
    model=read(OUT/'frozen_model.json');source=read(OUT/'SOURCE.json')
    publication=read(OUT/'payload_publication.json')
    assert len(result['completed_ids'])==8 and not result['missing_or_failed_ids']
    assert result['source_input_preserved'] and verify['compile_calls']==32
    assert verify['raw_PASS']==8 and verify['compiled_postcheck_PASS']==32 and verify['paired_paths']==8
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
    summary=dict(status=result['status'],execution_source=result['source_commit'],q=32,n_whole_paths=8,
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
    c=read(ROOT/'artifacts/track_d_c7_1_q16/2026-11-07/run01/paired_statistics.json')
    old['16']={r['metric']:r['paired_delta']['mean'] for r in c['rows'] if r['axis']=='X'}
    comparison=['|資源/X|q2 n16|q4 n16|q8 n32|q16 n16|q32 n8|','|---|---:|---:|---:|---:|---:|']
    for g in metrics:
        comparison.append('|'+g+'|'+ '|'.join(number(old[q][g]) for q in ['2','4','8','16'])+'|'+number(table['X',g]['paired_delta']['mean'])+'|')
    feature_rows=[]
    for sid in result['planned_ids']:
        raw_features=read(OUT/'paths'/sid/'structural_features_raw.json')
        compiled={m+'_'+a:read(OUT/'paths'/sid/(m+'_'+a+'_features.json')) for m in ['symmetric_directional','reference'] for a in ['X','Y']}
        feature_rows.append(dict(sample_id=sid,basis_runs=raw_features['basis_runs_within_tails'],
            different_basis_transitions=raw_features['different_basis_transitions_within_tails'],
            support_counts=raw_features['support_counts'],sign_counts=raw_features['coefficient_sign_counts'],
            zero_signed_angles=raw_features['zero_signed_angles'],phase_chi=raw_features['phase_chi'],
            strong_boundaries=raw_features['strong']['step_boundaries_fused'],
            reference_boundaries=raw_features['reference']['internal_step_boundaries_fused'],
            logical_CSWAP=raw_features['logical_CSWAP'],
            compiled_wire_finish={k:v['weighted_wire_finish'] for k,v in compiled.items()},
            compiled_total_critical_counts={k:v['total_DAG_longest_path_counts'] for k,v in compiled.items()}))
    write_new(OUT/'structural_feature_summary.json',dict(status='RECORDED_EXISTING_FULL_WRAPPERS',rows=feature_rows,
        extra_compiles=0,new_model_fits=0,filtered_semantics='zero-weight excluded operations still synchronize wires',
        qualification='descriptive features, not a compile-free predictor or causal attribution'))
    feature_table=['|path|tail basis runs|異basis遷移|零角|strong/ref境界|CSWAP|','|---|---:|---:|---:|---|---:|']
    for r in feature_rows:feature_table.append('|'+r['sample_id']+'|'+str(r['basis_runs'])+'|'+str(r['different_basis_transitions'])+'|'+str(r['zero_signed_angles'])+'|'+str(r['strong_boundaries'])+'/'+str(r['reference_boundaries'])+'|'+str(r['logical_CSWAP'])+'|')
    pairs=verify['paired_checks']
    paired_max=max((check['max_entry'] for row in pairs for check in row['checks'].values() if isinstance(check,dict) and 'max_entry' in check),default=None)
    leakage_max=max((r['max_reference_leakage_L2'] for r in verify['reference_leakage'] if isinstance(r,dict) and 'max_reference_leakage_L2' in r),default=None)
    text=f'''# Track D C7-3：q32独立whole-path探索的paired価格検証

2026-11-07 JST。**全8 pathのraw correctness、全32 P0 wrapper postcheck、最終paired X/Y・vacuum χはPASS。CX候補の観測判定は `{cx['status']}`、凍結予測への一次診断は `{diag['status']}`。確認的統計保証、全path公式、q≥64・全13-round費用優位は未認定。**

[固定PLAN](PLAN.md)／[result][RESULT]／[STOP][STOP]／[検証集約][VERIFY]／[科学scalar要約][SUMMARY]。科学実行と許可範囲の整理を終了し、公開照合後mandatory STOPとする。

## 固定条件・SOURCE・入力・seed

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q32のみ、complete step16後cut。元Hamiltonian Hの厳密groundを仮定し、準備費除外。ground vectorを用いず状態非依存observableを検証し、Referenceでは全256 system入力列を扱った。新ground solve/DF fitなし。

sampling/raw SOURCE `35bfd0423d08114b0a24107501ce545cdfb28f32`。C7-2公開基点 `709d6ed9c234bf6d5a8848d3a30be69d88fd6476` から専用branchへ分離し、sampling前にSOURCE tree `46af60bca001ba2748ac21cebad75e92a87443fa` と11公開ファイルのbytes一致を確認した。local/remote科学SOURCE commitは各公開時点で同一。元C SOURCEは `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。全SOURCE/input/package bindingは[SOURCE][SOURCE]にある。

利用者の追加許可に基づく[高速化・引継ぎPLAN](ACCELERATED_COMPLETION_PLAN.md)で、compile/post SOURCEを `{result['source_commit']}`（tree `a9cc9993638c39061be97feab008b310381c2f6a`、34公開ファイルのbytes照合済み）へ切り替えた。sampling/rawは旧35bfd、compile/postは新94e94に分け、旧event record/raw requestとそのfingerprint、compile側request/sourceを併記している。

旧controllerだけを調整停止し、投入済みraw worker8件はsignalを送らず完了まで待った。旧controllerのKeyboardInterrupt・STOP_FAILURE・集約raw_pass=0/source_input_preserved=falseは途中集約として保全し、修正していない。一方、実際に保存された全8 precompile_checksはPASSで、旧compile着手0を確認した。135 memberの固定継承hashを照合し、新側でraw8件を再利用、新raw0・新sampling0・未着手compile32件だけを各一回実行した。新側SOURCE/input前後照合は一致であり、旧集約の未完了フィールドと混同しない。[旧STOP][OLDSTOP]／[旧result][OLDRESULT]／[固定継承索引][INHERITANCE]。

packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、CDF endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、law記述 `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。signed atom/frame/sector/scalar/identity/normalization/attenuationと測定規約を保持し、weightをunitaryへ重複適用していない。

cohort `track_d_c7_3_q32_exploratory_paired_price_v1`、master2026110703、新domain `track_d_c7_3_q32_exploratory_paired_price_v1:independent_holdout:q32`。完全master tuple・seed keyは[事前seed表][SEED]、NumPy1.26.4 PCG64と既存trajectory_seed/canonical samplerを使用。q32_sample00..07、各occurrence0..31、各32 events。全8 record/256 occurrence/8192 eventsを一度だけ保存してhashを固定し、その後CAR/回路/価格を評価した。[event hash binding][EVENTS]

C6a96・C6b256・C7-1 q16の256既使用seed、合計608と非重複。旧q8 domainのunused文字列を未使用と扱わず、旧cohort巡回/prefix共有/短path反復なし。同pathを両方式・X/Yへ共有した。seed分離は数学的ideal joint-IIDの証明でなく、有限bit law距離・PRNG joint-lawは未資格。

C6b当初23725のsampling/raw/compileと承認済み44a58の残task引継ぎは過去証拠として保持し、今回へresume/承認を流用していない。旧SOURCE/result/raw/grant/ledger/marker/STOPを変更していない。

## 再利用・人工検証・実行件数

新Q32Request/seedとlaw-only入力接続を追加し、旧q16 helperの独立module instanceへdomainだけ束縛して統計/ingestion/凍結モデルの既存関数を再利用した。通常のq16 module・旧guard・SOURCEは変更していない。schedulerも独立module instanceのworker/progressを束縛し、旧mainを呼んでいない。law-only contextで全eventのhash固定前にhead回路をemitせず、元prepare_arm/configureを再利用した。C7-2のmax-plusと既存DAGを使うfeature取得は既存full-wrapperの読取だけで、単体A/B/shell compileを追加していない。

C3 prepare_steps/physical_segment/mixed_head、B2/C6a CAR/head/signed atom cache、StrongBuilder、既存Reference actual inverse/χ、C5 wrapper_for/_checked_price、Stage A raw checker、C2 P0/postcheck、C3 pairedχを再利用した。新Gaussian合成器/sampler/compiler/数値シミュレータの再実装はない。高速化workerでは標準gateの実class/control state/角度binary64 bytesをkeyとするexact局所cacheを選択し、cost回路を変更せず公式局所行列を再利用した。9qubit以下の全演算子評価は旧Qiskit Operatorから公式Aer double/unitaryへ変更し、全行列coverageと宣言global phase復元を維持した。17qubit Referenceは既存C6b BatchedStateActionsの公式Aer double/statevector・全256入力列を維持し、simulation変換だけをcache版に選択した。compile/postは既存compile_one/verify_oneを分離してdispatchした。P0は同じ1 thread、postは両方式ともnative thread budgetの対象であり、SOURCE/runtimeの差を隠さない。

人工/回帰50件PASS（新9＋q16 7＋C6b9＋C6a7＋C5 8＋C3 4＋C2 2＋高速化4）。2-mode toy q32のAB/actual inverse/vacuumχ/X/Y、新seed/旧guard、mock canonical sampler、共分散伝播、CX反例の保存、raw PASS後4compileの投入を確認した。[sampling前人工記録][TEST]。切替前には追加のcache/native/full-matrix・handoff・長い9qubit人工identity（196608命令）を含む人工/回帰61件PASS（新11＋既存50）を確認した。[切替前人工記録][FASTTEST]。人工PASSを分子性能や全law保証にしない。

|実処理|実施件数・結果|
|---|---|
|canonical occurrence sampling|256、追加標本0|
|whole-path raw correctness|旧8/8 PASSを再利用、新raw0|
|P0 full-wrapper compile|32、各wrapper一回|
|compile後postcheck|32/32 PASS|
|最終paired X/Y・vacuum χ|8/8 PASS|
|新fit/追加calibration/N/Born shots/ground solve/DF fit|0|
|科学retry/同task resume・閾値緩和・完了task再実行|0（許可済みSOURCE引継ぎ1回）|

独立pathとraw PASS後のcompileを並列pipeline化し、工程別完了memory資料と実空きCPU/RAMで投入した。旧worker数を恒常上限にせず、共有環境/他jobへの変更、他job中断、実行SOURCE hot-editなし。旧controllerの調整停止は上記の許可済み引継ぎのみ。人工9qubitでの速度比較を実H4全体/P0 compileの倍率へ転用せず、実H4全体のspeedupや速度最適性は未測定。内部資源観測はローカル保全のみ。

## Whole-path・対照公平性・correctness

V=U31…U0=A B、B前16 step/A後16 step、gadget一つ。strongはuncontrolled Gaussian、directional中心、head relative fusion、tail同basis共有、31 step境界の合法half-sweep融合、scalar/identity位相集約。Referenceはactual A.inverse、analytic全pathχ、内部30境界融合、global cutの1境界融合損失、追加8 qubits/16 logical CSWAP/χ/readoutを含む。弱い対照、結果依存cut、短gadget反復、別回路の節約値加算なし。[pathごとのscalar条件][CONDITIONS]

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

## q32有限標本のfull-wrapper価格

以下は平均 ± 標本SD [観測min,max]、ddof1、**n=8 whole-path**。X/Yを16独立標本と数えない。[全価格][PRICES]／[paired統計・共分散][STATS]／[path別差CSV][CSV]。min/maxはlaw supportではない。

{chr(10).join(prices)}

追加register/CSWAP/actual inverse/χ/global cut損失/shellを含む実compile価格。qubitsは9/17の幅制約で、round/shotに加算しない。RZをTへ換算せず、filtered/total DAG depthを実機時間と同一視しない。routing/T合成/noise/reset/実機費は未評価であり0ではない。

## CX整数構造候補の残差

事前候補208−224qのq32値−6960に対し、残差=観測ΔCX+6960を整数として保存した。判定 `{cx['status']}`。[完全な残差・保存平均/SD/範囲][CX]

{chr(10).join(residuals)}

これは上の観測pathだけの一致/反例であり、全sampling lawの公式、母分散0、全q移送資格を認定しない。5%価格幅での診断と整数一致は別評価。残差が非零でもcorrectnessと契約が成立する場合は有効な結果として保持し、path除外・追加sampling・model変更をしない。

## C7-2に基づく構造特徴

全8 pathの既存raw/compiled full-wrapperから、support/sign/zero-angle集計、tail内basis run、異basis遷移、head/tail/global cut、scalar/identity/χ、lane raw metrics、CSWAP/ancilla role依存、compiled wire別RZ/CX/total終端時刻とtotal-DAG critical pathの操作分類を保存した。[特徴要約][FEATURE]とpaths/の個別features.json。元event順序やraw配列は公開しない。

{chr(10).join(feature_table)}

C7-2 max-plus recurrenceは既存metricsと照合した。対象外gateもwireを重み0で同期する。total-DAG longest pathのRZ数/CX数と、filtered RZ/CX depthに対応するweighted critical pathは同じとは限らず、別featureとして保持する。A†/Bの別register中央演算、CSWAP前後のwire到着/出発依存、cutでの1境界融合損失を含む。単純shell＋max(lane)や局所CX和はC7-2で一般恒等式への反例があり、今回も価格代理へ登録していない。

特徴と価格残差の対応は記述的な原因候補であり、新しい係数fit/相関モデルによる因果効果・全path価格定理を認定しない。CX残差が非零なら該当support/run/端部/P0依存を候補に挙げ、追加compileなしで取得証拠の限界を報告する。残差0でもP0全体bridgeを得たことにはならない。

## 凍結モデルとの比較と不確かさ

C6a q2/q4保存係数・全48係数共分散をsampling前に凍結し、q32予測を保存した。q8/q16をfit点へ追加せず、再fit0。以下は事前予測と今回の保存平均の対応。[凍結モデル][MODEL]

{chr(10).join(predictions)}

一次は凍結予測値に条件付け、e=観測−予測、SE=s8/√8、|e|+z SE≤τ。τ=1+.05×事前strong予測規模、family12/名目的alpha=.05、normal/Bonferroni。別population-line欄は|e|+z√(SE²+vcal)≤τ。q32予測の重み−14/15（分散係数196/225）とC6a保存方式間共分散からvcalを伝播した。cross-q項0はC6a q2/q4集合の独立設計を条件とする仮定で、ideal joint-IIDや無条件の0を証明したものではない。一次へ校正分散を重複加算せず、両estimandを区別する。[全診断][DIAG]

{chr(10).join(diagnoses)}

一次全体 `{diag['status']}`、別population-line欄の幅外は{len(outside)}量。これらは探索的・名目的余裕規則の判定であり、有効95%同時coverageや真の誤差がτ外という確認的結論ではない。全体STATISTICALLY_UNQUALIFIED。古典価格のalphaをRPE energy/failureへ混ぜない。

path間変動、係数校正不確かさ、model bias、q移送、finite-bit/ideal law、P0依存を分ける。小さなpaired分散は方式別絶対価格の予測精度を保証しない。共分散は方式/資源/axisの依存と特異性を保持し、共有係数を異なるroundで独立と扱わない。正常性/tail/母分散/ideal-IID/bias domainの保証は未取得。

## 保存q2/4/8/16との対応・q64以降の不足

次表は各既存JSONのpaired平均欄の転記、Xを表示する。既存価格の再compile・再fitはしていない。

{chr(10).join(comparison)}

qごとにpath長、標本、cut位置、境界数が変わるため、差をqだけの因果効果と断定しない。RZ/sizeの符号・観測範囲は上の保存表の範囲で述べ、qが大きければ必ず減るとはしない。小さなRZ差と絶対RZ価格の精度、depthのlane/critical-path残差を別に扱う。旧Stage A q1は保存cohort/event16 cut、C5 q2やC3 q4/q8は既知fixtureで、今回のfresh標本へ混ぜていない。

q32の有限技術証拠はq≥64を資格付けしない。次はユーザー／GPTが、今回のCX整数残差とRZ/size/depth残差を材料に、(i)SOURCE/P0の構造bridgeを限定して補う、(ii)未使用の別qを独立検証する、(iii)用途がpopulation校正線/絶対価格精度で校正不確かさが支配する場合だけ追加calibrationを設計する、のどれが必要か選ぶ。係数/特徴を変更するなら探索版として別保存し、新モデルの独立検証domainを別に定める。事後幅拡張・同標本への再fitによるPASS化・全q大量compileへの自動進行はしない。

B4-2の14738は元H厳密ground・ideal finite-RTE IID・exact native等の条件付き十分shot数として参照のみ。B5のfinite-bit law、shot/occurrence joint-law、全path/全round実装誤差・actual共通Nは未解決。本有限q32結果を価格×Nへ接続して全13-round費用を算定せず、N再算定/actual成功保証/B側SOURCE変更も行わない。

## 保全・公開・STOP

旧sampling/rawと新compile/postのSOURCE/inputおよび継承hashの前後照合は一致。旧STOPは変更していない。元event/QPY/NPZ/reference配列はlocalに保持し、{publication['members']} memberの[hash索引][PAYLOAD]を公開する。raw payload追加公開0。[公開範囲][PUB]。内部CPU/RAM/runtime/thread、内部絶対path、資格情報、raw event内容を公開しない。collectorは保存check/価格統計を読み、回路検証・fit・sample統計を再実行していない。

**C7-3の許可済み実行・結果整理を終了。専用branchへ公開し、remote SHA/tree/公開bytes照合後mandatory STOP。** 次担当は**ユーザー／GPT**が次の目的・q-domain・構造モデル/追加検証/校正の必要性を判断する。**Codex**は新たな明示許可後だけ採択scopeを担当する。q≥64、新sampling/追加標本、再fit、確認的保証、全RPE費用、actual成功保証、詳細科学レビューへ自動進行しない。

'''
    links={'RESULT':'result.json','SOURCE':'SOURCE.json','STOP':'STOP.json','VERIFY':'verification_summary.json',
        'SUMMARY':'report_summary.json','SEED':'seed_plan.json','EVENTS':'event_bindings.json',
        'CONDITIONS':'path_conditions_scalar.json','PRICES':'prices.json','STATS':'paired_statistics.json',
        'CSV':'paired_differences.csv','CX':'cx_candidate_residuals.json','MODEL':'frozen_model.json',
        'FEATURE':'structural_feature_summary.json','DIAG':'exploratory_diagnostics.json','PAYLOAD':'payload_indices','PUB':'payload_publication.json'}
    for label,path in links.items():text+=f'[{label}]: ../../../artifacts/track_d_c7_3_q32/2026-11-07/run01_accelerated_completion/{path}\n'
    text+='[TEST]: ../../../artifacts/track_d_c7_3_q32/2026-11-07/technical/synthetic_tests.json\n'
    text+='[FASTTEST]: ../../../artifacts/track_d_c7_3_q32/2026-11-07/accelerated_technical/synthetic_tests.json\n'
    text+='[INHERITANCE]: ../../../artifacts/track_d_c7_3_q32/2026-11-07/accelerated_technical/inherited_evidence.json\n'
    text+='[OLDSTOP]: ../../../artifacts/track_d_c7_3_q32/2026-11-07/run01/STOP.json\n'
    text+='[OLDRESULT]: ../../../artifacts/track_d_c7_3_q32/2026-11-07/run01/result.json\n'
    DOC.write_text(text)
    print(json.dumps(summary))


if __name__=='__main__':main()

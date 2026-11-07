# Track D C7-1：q16独立whole-path探索的paired価格検証

2026-11-07 JST。**全16 pathのraw correctness、全64 P0 wrapper postcheck、最終paired X/Y・vacuum χはPASS。CX候補の観測判定は `CX_CANDIDATE_MATCHED_OBSERVED_PATHS`、凍結予測への一次診断は `EXPLORATORY_WITHIN_WIDTH`。確認的統計保証、全path公式、q≥32・全13-round費用優位は未認定。**

有限16本の両axisでCX・RZ/CX/total depth・RZ count・sizeはすべて減少した。各axisの平均差はCX−3376、total depth−25648.4375、RZ−488.75、size−3672.75。qubitsは9→17。一次12量は探索的幅内だが、校正不確かさを含む別population-line診断はRZ depth・total depthの各X/Y、計4/12で幅外。全law・長q・全RPEの保証へ拡張しない。

[固定PLAN](PLAN.md)／[result][RESULT]／[STOP][STOP]／[検証集約][VERIFY]／[科学scalar要約][SUMMARY]。科学実行と許可範囲の整理を終了し、公開照合後mandatory STOPとする。

## 固定条件・SOURCE・入力・seed

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q16のみ、complete step8後cut。元Hamiltonian Hの厳密groundを仮定し、準備費除外。ground vectorを用いず状態非依存observableと全256 system入力列で検証した。新ground solve/DF fitなし。

実行SOURCE `462f6bff66a1e3fd84a56e32aba83e9aed076257`。C7-0公開基点 `8b422c28c22666a32103fdae33a3c9b9512ef248` から専用branchへ分離し、sampling前にSOURCE tree `4b3dc546c4ba59b9ce8eb21e695d6eed4a9eeda2` と12公開ファイルのbytes一致を確認した。local/remote科学SOURCE commitは同一。元C SOURCEは `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。全SOURCE/input/package bindingは[SOURCE][SOURCE]にある。

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
|raw component|1.14130926935e-13|
|compiled component|1.86968987012e-11|
|compiled observable Frobenius|6.10437513362e-11|
|最終paired/χ component（保存欄の集約）|5.46351852648e-12|
|Reference leakage（対応する保存欄がある場合）|8.05775112685e-13|

個別検証は[全check集約][VERIFY]とpaths/に保存した。固定精度内のlocal binary64証拠であり、丸め誤差の厳密包絡・全support/全round実装誤差bound・immutable CI・独立詳細科学レビューではない。

## q16有限標本のfull-wrapper価格

以下は平均 ± 標本SD [観測min,max]、ddof1、**n=16 whole-path**。X/Yを32独立標本と数えない。[全価格][PRICES]／[paired統計・共分散][STATS]／[path別差CSV][CSV]。min/maxはlaw supportではない。

|axis|資源|strong symmetric|Reference|Reference−strong|
|---|---|---|---|---|
|X|rz_count|83987 ± 2385.07950951 [79442, 87544]|83498.25 ± 2388.2628694 [78947, 87059]|-488.75 ± 6.37181292883 [-502, -479]|
|X|cx_count|32145 ± 730.974874169 [30714, 33224]|28769 ± 730.974874169 [27338, 29848]|-3376 ± 0 [-3376, -3376]|
|X|rz_depth|20391.8125 ± 601.248835758 [19046, 21222]|9867.875 ± 348.83919409 [8990, 10283]|-10523.9375 ± 339.315284802 [-11118, -10056]|
|X|cx_depth|15733.6875 ± 333.472581332 [14985, 16204]|6059.75 ± 193.075978136 [5572, 6294]|-9673.9375 ± 185.38769062 [-10002, -9413]|
|X|total_depth|47514.625 ± 1315.87627965 [44582, 49319]|21866.1875 ± 762.501254097 [19965, 22784]|-25648.4375 ± 746.461695266 [-26944, -24617]|
|X|circuit_size|171584.5 ± 4789.19570144 [162349, 178709]|167911.75 ± 4792.32172682 [158670, 175040]|-3672.75 ± 6.37181292883 [-3686, -3663]|
|X|peak_qubits|9 ± 0 [9, 9]|17 ± 0 [17, 17]|8 ± 0 [8, 8]|
|Y|rz_count|83987 ± 2385.07950951 [79442, 87544]|83498.25 ± 2388.2628694 [78947, 87059]|-488.75 ± 6.37181292883 [-502, -479]|
|Y|cx_count|32145 ± 730.974874169 [30714, 33224]|28769 ± 730.974874169 [27338, 29848]|-3376 ± 0 [-3376, -3376]|
|Y|rz_depth|20391.8125 ± 601.248835758 [19046, 21222]|9867.875 ± 348.83919409 [8990, 10283]|-10523.9375 ± 339.315284802 [-11118, -10056]|
|Y|cx_depth|15733.6875 ± 333.472581332 [14985, 16204]|6059.75 ± 193.075978136 [5572, 6294]|-9673.9375 ± 185.38769062 [-10002, -9413]|
|Y|total_depth|47514.625 ± 1315.87627965 [44582, 49319]|21866.1875 ± 762.501254097 [19965, 22784]|-25648.4375 ± 746.461695266 [-26944, -24617]|
|Y|circuit_size|171584.5 ± 4789.19570144 [162349, 178709]|167911.75 ± 4792.32172682 [158670, 175040]|-3672.75 ± 6.37181292883 [-3686, -3663]|
|Y|peak_qubits|9 ± 0 [9, 9]|17 ± 0 [17, 17]|8 ± 0 [8, 8]|

追加register/CSWAP/actual inverse/χ/global cut損失/shellを含む実compile価格。qubitsは9/17の幅制約で、round/shotに加算しない。RZをTへ換算せず、filtered/total DAG depthを実機時間と同一視しない。routing/T合成/noise/reset/実機費は未評価であり0ではない。

## CX整数構造候補の残差

事前候補208−224qのq16値−3376に対し、残差=観測ΔCX+3376を整数として保存した。判定 `CX_CANDIDATE_MATCHED_OBSERVED_PATHS`。[完全な残差・保存平均/SD/範囲][CX]

|path|X ΔCX|X residual|Y ΔCX|Y residual|
|---|---:|---:|---:|---:|
|q16_sample00|-3376|0|-3376|0|
|q16_sample01|-3376|0|-3376|0|
|q16_sample02|-3376|0|-3376|0|
|q16_sample03|-3376|0|-3376|0|
|q16_sample04|-3376|0|-3376|0|
|q16_sample05|-3376|0|-3376|0|
|q16_sample06|-3376|0|-3376|0|
|q16_sample07|-3376|0|-3376|0|
|q16_sample08|-3376|0|-3376|0|
|q16_sample09|-3376|0|-3376|0|
|q16_sample10|-3376|0|-3376|0|
|q16_sample11|-3376|0|-3376|0|
|q16_sample12|-3376|0|-3376|0|
|q16_sample13|-3376|0|-3376|0|
|q16_sample14|-3376|0|-3376|0|
|q16_sample15|-3376|0|-3376|0|

これは上の観測pathだけの一致/反例であり、全sampling lawの公式、母分散0、全q移送資格を認定しない。5%価格幅での診断と整数一致は別評価。残差が非零でもcorrectnessと契約が成立する場合は有効な結果として保持し、path除外・追加sampling・model変更をしない。

## 凍結モデルとの比較と不確かさ

C6a q2/q4保存係数・全48係数共分散をsampling前に凍結し、q16予測を保存した。q8をfit点へ追加せず、再fit0。以下は事前予測と今回の保存平均の対応。[凍結モデル][MODEL]

|axis|資源|strong予測|Reference予測|paired予測|paired観測|
|---|---|---:|---:|---:|---:|
|X|rz_count|81224.75|80733.8125|-490.9375|-488.75|
|X|cx_count|31293.375|27917.375|-3376|-3376|
|X|rz_depth|19573.0625|9546.375|-10026.6875|-10523.9375|
|X|cx_depth|15322.6875|5891.25|-9431.4375|-9673.9375|
|X|total_depth|45686.5625|21139.625|-24546.9375|-25648.4375|
|X|circuit_size|165987.625|162312.6875|-3674.9375|-3672.75|
|Y|rz_count|81224.75|80733.8125|-490.9375|-488.75|
|Y|cx_count|31293.375|27917.375|-3376|-3376|
|Y|rz_depth|19573.0625|9546.375|-10026.6875|-10523.9375|
|Y|cx_depth|15322.6875|5891.25|-9431.4375|-9673.9375|
|Y|total_depth|45686.5625|21139.625|-24546.9375|-25648.4375|
|Y|circuit_size|165987.625|162312.6875|-3674.9375|-3672.75|

一次は凍結予測値に条件付け、e=観測−予測、SE=s16/√16、|e|+z SE≤τ。τ=1+.05×事前strong予測規模、family12/名目的alpha=.05、normal/Bonferroni。別population-line欄は|e|+z√(SE²+vcal)≤τ。q16予測の重み−6/7とC6a保存方式間共分散からvcalを伝播した。一次へ校正分散を重複加算せず、両estimandを区別する。[全診断][DIAG]

|axis|資源|観測−予測|SE|τ|一次margin/判定|別population margin/判定|
|---|---|---:|---:|---:|---|---|
|X|rz_count|2.1875|1.59295323221|4062.2375|6.75172555809 / EXPLORATORY_WITHIN_WIDTH|19.6734509625 / EXPLORATORY_WITHIN_WIDTH|
|X|cx_count|0|0|1565.66875|0 / EXPLORATORY_WITHIN_WIDTH|0 / EXPLORATORY_WITHIN_WIDTH|
|X|rz_depth|-497.25|84.8288212004|979.653125|740.306648467 / EXPLORATORY_WITHIN_WIDTH|1380.72634625 / EXPLORATORY_OUTSIDE_WIDTH|
|X|cx_depth|-242.5|46.3469226549|767.134375|375.295994662 / EXPLORATORY_WITHIN_WIDTH|721.04491172 / EXPLORATORY_WITHIN_WIDTH|
|X|total_depth|-1101.5|186.615423817|2285.328125|1636.20175376 / EXPLORATORY_WITHIN_WIDTH|3039.24113145 / EXPLORATORY_OUTSIDE_WIDTH|
|X|circuit_size|2.1875|1.59295323221|8300.38125|6.75172555809 / EXPLORATORY_WITHIN_WIDTH|19.6734509621 / EXPLORATORY_WITHIN_WIDTH|
|Y|rz_count|2.1875|1.59295323221|4062.2375|6.75172555809 / EXPLORATORY_WITHIN_WIDTH|19.6734509625 / EXPLORATORY_WITHIN_WIDTH|
|Y|cx_count|0|0|1565.66875|0 / EXPLORATORY_WITHIN_WIDTH|0 / EXPLORATORY_WITHIN_WIDTH|
|Y|rz_depth|-497.25|84.8288212004|979.653125|740.306648467 / EXPLORATORY_WITHIN_WIDTH|1380.72634625 / EXPLORATORY_OUTSIDE_WIDTH|
|Y|cx_depth|-242.5|46.3469226549|767.134375|375.295994662 / EXPLORATORY_WITHIN_WIDTH|721.04491172 / EXPLORATORY_WITHIN_WIDTH|
|Y|total_depth|-1101.5|186.615423817|2285.328125|1636.20175376 / EXPLORATORY_WITHIN_WIDTH|3039.24113145 / EXPLORATORY_OUTSIDE_WIDTH|
|Y|circuit_size|2.1875|1.59295323221|8300.38125|6.75172555809 / EXPLORATORY_WITHIN_WIDTH|19.6734509621 / EXPLORATORY_WITHIN_WIDTH|

一次全体 `EXPLORATORY_WITHIN_WIDTH`、別population-line欄の幅外は4量。これらは探索的・名目的余裕規則の判定であり、有効95%同時coverageや真の誤差がτ外という確認的結論ではない。全体STATISTICALLY_UNQUALIFIED。古典価格のalphaをRPE energy/failureへ混ぜない。

path間変動、係数校正不確かさ、model bias、q移送、finite-bit/ideal law、P0依存を分ける。小さなpaired分散は方式別絶対価格の予測精度を保証しない。共分散は方式/資源/axisの依存と特異性を保持し、共有係数を異なるroundで独立と扱わない。正常性/tail/母分散/ideal-IID/bias domainの保証は未取得。

## 保存q2/4/8との対応・q32以降の不足

次表は各既存JSONのpaired平均欄の転記、Xを表示する。既存価格の再compile・再fitはしていない。

|資源/X|q2 n16|q4 n16|q8 n32|q16 n16|
|---|---:|---:|---:|---:|
|rz_count|338.125|219.6875|-16.90625|-488.75|
|cx_count|-240|-688|-1584|-3376|
|rz_depth|-1310.375|-2555.5625|-5232.0625|-10523.9375|
|cx_depth|-1160.9375|-2342.4375|-4791.875|-9673.9375|
|total_depth|-3177.25|-6230.0625|-12747.9375|-25648.4375|
|circuit_size|290.125|-276.3125|-1408.90625|-3672.75|
|peak_qubits|8|8|8|8|

qごとにpath長、標本、cut位置、境界数が変わるため、差をqだけの因果効果と断定しない。RZ/sizeの符号・観測範囲は上の保存表の範囲で述べ、qが大きければ必ず減るとはしない。小さなRZ差と絶対RZ価格の精度、depthのlane/critical-path残差を別に扱う。旧Stage A q1は保存cohort/event16 cut、C5 q2やC3 q4/q8は既知fixtureで、今回のfresh標本へ混ぜていない。

q16の有限技術証拠はq≥32を資格付けしない。次はユーザー／GPTが、今回のCX整数残差とRZ/size/depth残差を材料に、(i)SOURCE/P0の構造bridgeを限定して補う、(ii)未使用の別qを独立検証する、(iii)用途がpopulation校正線/絶対価格精度で校正不確かさが支配する場合だけ追加calibrationを設計する、のどれが必要か選ぶ。係数/特徴を変更するなら探索版として別保存し、新モデルの独立検証domainを別に定める。事後幅拡張・同標本への再fitによるPASS化・全q大量compileへの自動進行はしない。

B4-2の14738は元H厳密ground・ideal finite-RTE IID・exact native等の条件付き十分shot数として参照のみ。B5のfinite-bit law、shot/occurrence joint-law、全path/全round実装誤差・actual共通Nは未解決。本有限q16結果を価格×Nへ接続して全13-round費用を算定せず、N再算定/actual成功保証/B側SOURCE変更も行わない。

## 保全・公開・STOP

SOURCE/inputの実行前後照合は一致。元event/QPY/NPZ/reference配列はlocalに保持し、320 memberの[hash索引][PAYLOAD]を公開する。raw payload追加公開0。[公開範囲][PUB]。内部CPU/RAM/runtime/thread、内部絶対path、資格情報、raw event内容を公開しない。collectorは保存check/価格統計を読み、回路検証・fit・sample統計を再実行していない。

**C7-1の許可済み実行・結果整理を終了。専用branchへ公開し、remote SHA/tree/公開bytes照合後mandatory STOP。** 次担当は**ユーザー／GPT**が次の目的・q-domain・構造モデル/追加検証/校正の必要性を判断する。**Codex**は新たな明示許可後だけ採択scopeを担当する。q≥32、新sampling/追加標本、再fit、確認的保証、全RPE費用、actual成功保証、詳細科学レビューへ自動進行しない。

[RESULT]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/result.json
[SOURCE]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/SOURCE.json
[STOP]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/STOP.json
[VERIFY]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/verification_summary.json
[SUMMARY]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/report_summary.json
[SEED]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/seed_plan.json
[EVENTS]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/event_bindings.json
[CONDITIONS]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/path_conditions_scalar.json
[PRICES]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/prices.json
[STATS]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/paired_statistics.json
[CSV]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/paired_differences.csv
[CX]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/cx_candidate_residuals.json
[MODEL]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/frozen_model.json
[DIAG]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/exploratory_diagnostics.json
[PAYLOAD]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/payload_indices
[PUB]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/run01/payload_publication.json
[TEST]: ../../../artifacts/track_d_c7_1_q16/2026-11-07/technical/synthetic_tests.json

# Track D C6b：独立q8 whole-pathの探索的paired価格検証

2026-11-07 JST。**予定32本のraw、128 wrapperのcompile後検証、最終paired X/Y・vacuum χはすべてPASS。凍結予測値に条件付けた一次12量は探索的価格幅内だった。一方、校正不確かさを加える別のpopulation-line診断はRZ depth・total depthのX/Y、計4量で幅外となった。確認的モデルPASS、厳密95%保証、全law・長q・全RPE費用優位は未認定。**

[機械可読結果][RESULT]／[検証集約][VERIFY]／[報告用scalar要約][SUMMARY]／[STOP][STOP]。元の[固定PLAN](PLAN.md)、[高速化の検討](ACCELERATION.md)、利用者が明示許可した[限定SOURCE引継ぎPLAN](ACCELERATED_COMPLETION_PLAN.md)を区別する。source/inputと旧証拠は前後照合で一致した。科学計算は終了済みで、残る公開照合後mandatory STOPとする。

## 固定条件と独立holdout

linear H4/隣接1.00 Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q8。元Hamiltonian Hの厳密groundを仮定し、状態準備費を除外する。ground vector・ground solve・DF fitを使わず、reference/ancilla=0の全256 system入力列で検証した。

元H packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、CDF endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。元science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` の入力・SOURCEと、既存Track Dのprecision・baseline・位相規約を保持した。[SOURCE][SOURCE]

q8_sample00..31、各8 occurrence、各step32 events、計256 canonical occurrence/8192 events。C6a既存 `configure_rte → sample_occurrence → iter_sample_rte_events` を使用した。master2026110606、予約済みholdout domain `track_d_c6a_q2_q4_paired_price_pilot_v1:independent_holdout:q8:unused`、完全seed key・sample/occurrence順・NumPy1.26.4 PCG64は[事前seed表][SEED]に固定した。文字列中unusedは予約identityを維持したもので、今回初めて使用した。C6a q2/q4 calibration seedとの重複なし。C3既知q8 fixtureや旧cohort巡回を流用していない。

32本すべてのeventを価格評価前に一回生成・保存し、[record/event hash][EVENTS]を固定した。同じpathを両方式・X/Yで共有し、価格による選択/除外/差替えをしない。X/Yは64独立標本ではなく32 whole-path。PCG64 seed分離は数学的IID証明ではなく、ideal lawとの距離・joint-lawは未資格である。

## SOURCE引継ぎと実際の実行件数

|役割|固定SOURCE/保存結果|
|---|---|
|校正|C6a結果 `e9d4e2f602d5f9e014507ca7e391719bab017b88`、科学SOURCE `6b6667026ec8b670f610e41141d195883a9eb513`。q2/q4各16、係数/共分散を凍結|
|C6b当初実行|`23725fc6f6d6d0c2bec93e31e32074711804e250`。sampling256・raw16・compile/postcheck48を実施|
|高速化部品|`75f43855b02a49e710d7a2ba3ee82eda3de1631e`。人工検証だけ。稼働中旧SOURCEへのhot-editなし|
|承認済み残task引継ぎ|`44a58b9fba5d3aba7a31d0a6b12e3aa25be9def4`。旧証拠を再利用し、未着手raw16・未compile80を実施|

利用者は、実行中taskの完了を待つ調整停止と新SOURCEへの残task引継ぎを明示許可した。旧controllerだけの新規投入を止め、workerにsignalを送らず投入済み処理を完了させた。旧[partial結果][OLDRESULT]／[KeyboardInterrupt記録][OLDFAIL]／[STOP][OLDSTOP]をそのまま保持する。これは承認済みの運用移管であり、correctness不一致によるSTOPではない。

割込み時点のcontroller集約にはpostcheck32件だけが記録されたが、その後drain中に完成した16件を含め、保存済みPASS checkは48件ある。新SOURCEはその48件を[456 memberのhash付き引継ぎ記録][INHERIT]で結び、数を上書きせず両記録を保持した。raw/QPY/NPZは別inodeのbyte一致コピーで再利用し、旧原本へ書き込まない。

|処理|旧SOURCEで完了・再利用|新SOURCEで初回実行|合計|
|---|---:|---:|---:|
|canonical occurrence sampling|256|0|256|
|whole-path raw correctness|16|16|32/32 PASS|
|P0 full-wrapper compile|48|80|128、各wrapper一回|
|compile後correctness|48|80|128/128 PASS|
|最終paired X/Y・vacuum χ|0|32 path|32/32 PASS|

全32 IDが完了し、欠損/失敗/除外は0。完了済み科学taskの再実行0、新sampling/ground solve/DF refit/N評価/Born shots/model refit/追加calibrationは0。旧新合算上限128を守った。旧SOURCEと新SOURCEでどのraw/compile/postcheckが生成されたかは[価格record][PRICES]のprovenanceとpayload memberごとのgenerated_by_sourceを参照する。価格のexecution_sourceは新しい評価SOURCEであり、旧回路を新SOURCEで再生成した意味ではない。

## 再利用・高速化・比較の公平性

C6aのrequest/event ingestion、CAR/head/atom cache、C3/C5 `prepare_steps`/StrongBuilder/Reference gadget/価格資格、Stage A checker、C2のP0・state-action、C3 pairedχ、既存paired統計/診断を再利用した。新sampler/compiler/numerical simulatorは作っていない。[実行runner](../../../scripts/run_track_d_c6b_accelerated.py)

高速化は新しい自分のworkerだけで既存state-action参照をopt-in `BatchedStateActions` へ選択する。元SOURCEファイルと旧workerは変えない。同じofficial local-matrix conversion/Aer native/double precisionを用い、操作deep copy削減、全入力の分割処理、実行thread内のOpenMP配分と工程別memory見積りを適用した。cost回路/P0・入力列・全出力振幅・declared phase復元は同じ。41件の人工/回帰（引継ぎ5＋高速化4＋既存32）はPASS。[人工記録][TEST]。人工速度比較は分子性能・最適性の証拠ではない。完了済み分子検証の再実行を速度比較のために加えていない。

whole path V=U7…U0=A B、B=前4 step、A=後4 step、cutはcomplete step4後。strong symmetricはuncontrolled Gaussian、directional control、head relative fusion、tail同basis共有、7つの合法half-sweep境界融合、scalar/identity位相集約を保持する。Referenceはwhole Vにgadget一つ、同じAのactual inverse、analytic全pathχ、内部6境界融合、global cutでの1境界融合損失、追加8 qubits・16 CSWAP・χ補正を含む。弱いbaseline・短gadget反復・事後cut変更・節約値の加算なし。[pathごとのscalar条件][CONDITIONS]

## Correctnessとcompile

rawは独立CAR、AB=V、actual A†、unitarity、vacuum ray、χ/scalar/identity/sign、strong direct controlled-V両branch、complex coherence、X/Y observable・位相、χ省略negative control、最終paired照合を確認した。各pathのraw PASS後だけ、そのpathのcompileへ進んだ。

P0はQiskit1.3.0/optimization_level1/seed17/rz,sx,x,cx/approximation_degree1.0/qubits_initially_zero=false/topology-free/位相保持。component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fitなしを維持。Referenceの17-qubit全Operatorは構成せず、全256入力列・各131072出力振幅を検査した。compile後CAR/raw/paired X/Y/χ/leakageを同精度で照合した。[全check集約][VERIFY]と`paths/<ID>/`に保存する。

|保存された最大差・量|値|
|---|---:|
|raw component|7.460698725e-14|
|compiled component|1.027580392e-11|
|compiled observable Frobenius|3.656744835e-11|
|Reference leakage L2|6.983525935e-13|
|最終paired/vacuum χ component|2.848767249e-12|

すべて既存thresholdを満たし、閾値緩和・phase fitting・不利path除外なし。SOURCE/input・旧456 memberの前後照合も一致。local binary64の有限検証であり、丸め誤差の厳密包絡、全support/全round実装誤差bound、immutable CIや独立科学レビューではない。

## q8有限cohortの絶対価格・paired差

各欄は**平均 ± 標本SD [観測min,max]**。SDはddof1、n=32 whole-path。全標本でX/Yの全7価格が一致したため下表はXを代表表示し、family12を事後に縮めない。両axisの[全32 paired差CSV][CSV]、[全128価格record][PRICES]、[統計/共分散][STATS]を保存する。

|資源|strong symmetric|Reference|Reference−symmetric|
|---|---|---|---|
|rz_count|42628.65625 ± 1586.74029 [39134, 45584]|42611.75 ± 1587.700549 [39117, 45570]|-16.90625 ± 3.871811904 [-25, -11]|
|cx_count|16263.9375 ± 485.9413171 [15188, 17184]|14679.9375 ± 485.9413171 [13604, 15600]|-1584 ± 0 [-1584, -1584]|
|rz_depth|10351.21875 ± 423.8943671 [9435, 11144]|5119.15625 ± 234.7916179 [4633, 5586]|-5232.0625 ± 270.6933675 [-5767, -4559]|
|cx_depth|7944.75 ± 217.1681376 [7476, 8354]|3152.875 ± 118.7395496 [2908, 3398]|-4791.875 ± 142.5327919 [-5068, -4438]|
|total_depth|24096.0625 ± 916.2559488 [22109, 25820]|11348.125 ± 504.6274736 [10296, 12359]|-12747.9375 ± 589.811831 [-13923, -11278]|
|circuit_size|87049.03125 ± 3192.84699 [79985, 93025]|85640.125 ± 3193.786435 [78576, 91619]|-1408.90625 ± 3.871811904 [-1417, -1403]|
|peak_qubits|9 ± 0 [9, 9]|17 ± 0 [17, 17]|8 ± 0 [8, 8]|

全32本で6資源の差は負、qubitsは常に9→17だった。q2/q4で増えていたRZは今回q8では小幅減少を観測した。小さなRZ差をT数・実用的節約へ読み替えない。CX差−1584/標本SD0も有限32本の観測であり、全lawでの定数差・分散0・必要n=0という保証ではない。min/maxもlaw supportではない。

28列（方式×axis×7資源）のsample covarianceと12列paired covarianceをddof1で保存した。方式間/資源間/axis間の依存と固定幅・X/Y同価格による特異性を保持し、可逆性を仮定しない。sample_covarianceの一般的rank上限31という保存条件を、実測rankや正規性証明としない。

Reference register/CSWAP/χ/actual inverse/cut損失とreadoutを含むfull-wrapperの価格である。RZ→T換算、routing/noise、有限精度T合成は未評価。DAG depthを実機時間と同一視せず、qubitsをroundへ加算しない。

## 凍結C6aモデルとの比較

[凍結済み係数/共分散/予測/幅][MODEL]はholdout生成前に固定した。C6aモデルを再fitせず、q8を校正に混ぜていない。次表の予測は既存affineモデルのq8値、観測は32本の保存平均。X/Yは同じ値である。絶対価格は記述対象で、一次精度PASS familyではない。

|資源|S予測|R予測|paired予測|paired観測|観測−予測|
|---|---:|---:|---:|---:|---:|
|rz_count|41341|41323.8125|-17.1875|-16.90625|0.28125|
|cx_count|15866.375|14282.375|-1584|-1584|0|
|rz_depth|9975.5625|4929.625|-5045.9375|-5232.0625|-186.125|
|cx_depth|7757.1875|3051.75|-4705.4375|-4791.875|-86.4375|
|total_depth|23255.0625|10919.375|-12335.6875|-12747.9375|-412.25|
|circuit_size|84439.875|83030.6875|-1409.1875|-1408.90625|0.28125|

### 一次frozen条件付き診断と別population-line診断

一次は保存予測値そのものを固定し、e=paired観測平均−凍結予測、SE8=s8/√32、`|e|+z SE8≤tau`。価格幅tau=1+.05×事前symmetric予測規模、alpha=.05/family12の二側normal z=2.865260239。差を分母にした相対誤差は使わない。一次へcalibration varianceを二重加算しない。

別欄はpopulation校正線の診断 `|e|+z√(SE8²+saved vcal)`。両者はestimand/conditioningが異なり、別欄を第二の確認的familyとして追加認定しない。すべて事前固定規則による探索的・名目的診断である。[全12量の診断][DIAG]

|資源|tau|一次名目halfwidth|一次margin|一次|別population margin|別診断|
|---|---:|---:|---:|---|---:|---|
|rz_count|2068.05|1.961116234|2.242366234|幅内|7.473874716|幅内|
|cx_count|794.31875|0|0|幅内|0|幅内|
|rz_depth|499.778125|137.1092322|323.2342322|幅内|565.984122|幅外|
|cx_depth|388.859375|72.19446063|158.6319606|幅内|291.2009064|幅内|
|total_depth|1163.753125|298.7463196|710.9963196|幅内|1244.288129|幅外|
|circuit_size|4222.99375|1.961116234|2.242366234|幅内|7.473874716|幅内|

**一次12/12はEXPLORATORY_WITHIN_WIDTH。別診断はRZ depth・total depthの各X/Y、計4/12でEXPLORATORY_OUTSIDE_WIDTH。** 別診断の幅外はその名目的余裕規則の未達で、真の誤差がtau外という確認的棄却ではない。一次幅内は有効な95%同時coverageやmodel bias boundを認定しない。

正規近似/母分散/tail/support、ideal law対finite-bit implemented lawとPRNG joint-law、affine bias/domainが未資格なので、全体は**STATISTICALLY_UNQUALIFIED**。古典価格alphaをRPE energy/failureと混ぜない。モデル精度幅内と、資源差の符号・最小実用節約の確認的資格も別である。

## C6a・長q・Stage Bとの接続と不足

C6a q2/q4各16のpaired平均CXは−240/−688、total depthは−3177.25/−6230.0625、RZは+338.125/+219.6875だった。今回q8はCX−1584、total depth−12747.9375、RZ−16.90625という別の有限cohort観測を加える。path長・cut・境界数と標本が変わるため、qだけの因果効果とはしない。旧C3固定q8やStage A保存cohortは今回holdoutへ混合しない。

q8での一次幅内だけではq>8・全13-roundへ移送できない。長qでは構造/boundary/laneモデルのdomainと残差、全supportでのcompiler rewrite、共有係数covariance、独立な未観測qでの証拠が必要。CX定数差の構造候補を、今回のrandom32本から新定理に格上げしない。絶対期待価格や長q分散/tailも未資格である。

B4-2 `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` の14738は、元H厳密ground・ideal finite-RTE IID・exact native等を条件とした13-round十分shot数の保存結果として参照するだけ。再算定・価格との乗算はしない。B5 `0a8b953aea1be20b110784960763d7f36ebccac8` が示したfinite-bit law/shot-occurrence joint-law/全path全round実装誤差の不足や、actual条件で両方式が共通Nを使える資格は、本32本で解決していない。B6へ介入せず、全RPE総費用・actual成功保証を未評価のまま保持する。

## 保存・公開範囲・STOPと次の担当

原event/QPY/NPZは許可されたローカル領域に保持する。32 path×20 member=640の[member hash索引][PAYLOAD]を作成し、各memberの旧新生成SOURCEを区別した。archive/回路/配列本体はGitHubへ追加公開しない。[公開範囲][PUB]。公開するのはSOURCE/PLAN、科学scalar checks/χ条件、価格/統計/共分散/診断、hash索引、旧新STOP。内部CPU/RAM/runtime/thread、絶対path、資格情報・非公開rawは除く。raw外部公開が必要なら具体的な別許可が必要。

最小の次作業は、**ユーザー／GPTが今回の探索的q8証拠を採用する範囲と、確認的価格精度・長q移送のどちらを次の目的にするか判断すること**。確認的ならlaw/IID/tailと多重性・精度用途の根拠を先に固定し、必要な限定統計設計や追加校正を別scopeで許可する。長qなら構造モデル/誤差domainと未使用qの検証計画を選ぶ。holdoutに合わせた再fit・幅変更・追加標本を自動で行わない。Codexは次の明示許可後だけ採択scopeを実行する。

**C6b科学実行・許可された結果整理を終了。専用branchへcommit/pushしremote SHA/tree/公開bytesを照合後mandatory STOP。** q>8、追加calibration、新sampling、長q資格化、N/全RPE費用、actual成功保証、詳細科学レビューへ進まない。

[RESULT]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/result.json
[VERIFY]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/verification_summary.json
[SUMMARY]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/report_summary.json
[STOP]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/STOP.json
[SOURCE]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/SOURCE.json
[OLDRESULT]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01/result.json
[OLDFAIL]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01/failure.json
[OLDSTOP]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01/STOP.json
[INHERIT]: ../../../artifacts/track_d_c6b_q8/2026-11-07/accelerated_technical/inherited_evidence.json
[SEED]: ../../../artifacts/track_d_c6b_q8/2026-11-07/technical/seed_plan.json
[EVENTS]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/event_bindings.json
[TEST]: ../../../artifacts/track_d_c6b_q8/2026-11-07/accelerated_technical/synthetic_tests.json
[CONDITIONS]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/path_conditions_scalar.json
[CSV]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/paired_differences.csv
[PRICES]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/prices.json
[STATS]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/paired_statistics.json
[MODEL]: ../../../artifacts/track_d_c6b_q8/2026-11-07/technical/frozen_model.json
[DIAG]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/exploratory_diagnostics.json
[PAYLOAD]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/payload_indices
[PUB]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/payload_publication.json

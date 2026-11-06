# Codex C：B2 P1長round価格方式の人工技術検証

2026-11-07 JST / SERVER 2 / Codex主エージェント。

**元のgate列を保つrequest/tail streamingは、人工9ケース・X/Y計18回路で全6価格指標がEXACT_MATCH。**
第一候補とする。一方、既存boundary fusionは14/18回路で価格が変わり、分割compileも18/18で不一致。
streamingは中間保持を減らすが最終flat回路/DAGを残すため、**m17・全18round MCの実用的成立性は未確立**。
実H4の新規build/compile・期待価格推定・価格MC・総RPE費用は実行していない。公開後**mandatory STOP**。

[人工結果JSON][RESULT] / [分類・検証summary][SUMMARY] / [MC統計設計と不足][STATS] / [実行前PLAN](PLAN.md)。

## 固定SOURCE・条件・前回証拠

人工検証SOURCEは公開commit **[`c897389643a8d3b81ff63f9a0d71b250b17a565e`][SOURCE]**。
local準備commit `665c7176…`と同一tree `fa58dc1f4bb7fc938bfb0940fd299a67fde82712`を公開し、取得・一致確認後に
実行HEADを公開SOURCEへ合わせた。旧commitもlocal履歴branchへ保全。実行後のSOURCE hash不変を照合した。

前回基点は[`2d21c238…`][PREVIOUS]。前回の実H4 m0価格、元SOURCE `5aa5d061…`、公開SOURCE `b443f25…`、
元結果/公開snapshot/bundleの対応を変更していない。前回資料はこのbranchの祖先から参照できる。
今回もGitHub接続で変更分だけを公開し、共有Git設定やCodex A/Bのworktree・成果物を変更しない。

将来のB2対象条件はLinear H4/1.00Å/STO-3G、cutoff前H_phys、保存DF12、generation-prefix LD3、
partial-S2/r4/K2、delta=.02、m0..17/Q=2^m、元H基底state仮定・prep費用除外のまま。
今回は既存の人工**二mode・rank4** fixtureを拡張し、LD2/3、Q1/2/4/8を検査した。
LD2は異なるRTE basis境界、LD3はprefix長を検査する人工条件であり、H4の科学条件の変更ではない。
order0/2・正負係数・complex basis・scalar/identityを含む固定event patternと反復patternを使用。
これらはcanonical support内の明示ケースで、確率平均を推定するIID標本ではない。分子入力・stateは未使用。

compilerは**Qiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/targetなし、all-to-all**。
hash **`a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`**、backend fingerprint **`no_backend`**。
全候補で`single_hadamard_interrogation_without_state_preparation`を維持し、X/Y wrapper・ancilla測定を各1回含む。

## 再利用と実装差分

- [stream adapter][ADAPTER]：既存`B2RPERound.explicit_request`をQ1のrequest検証に使い、
  canonical eventsを**r個ずつ**消費する。既存native `_append_block`、RTE `build_sequence`、`_finish`を再利用。
  forward→RTE→reverse、referenceと同じ`T=delta*Q; dt=T/Q`、最後のscalar/identity相対位相一回を保ち、
  Q個のrequest tuple・controlled-tail listを作らない。最終回路は元と同じgate列/fingerprint。
- compiler・sampler・価格集計器は不変更。既存`build_native_hadamard_wrapper`、`_compile_pair`、
  `transpile_and_measure_cost`、`RPE_COST_METRICS`を再利用。分割出力の価格だけは、再compileせずに
  既存gate count/filtered-depth primitivesとQiskit depth/sizeで接続回路を計測する薄いhelperを追加した。
- boundary候補は既存`QiskitDFPartialS2RepeatedCircuitBuilder(...boundary_optimized)`そのもの。
  cacheは既存`TranspiledCircuitCostCache`へのcompile関数注入だけ。
  actual circuit・compiler・backend identityによる既存keyを使い、drawを除去・再重み付けしない。
- 並列処理は`1b64e46c…`の[`o1_m1_parallel_price.py`][PARALLEL]を**原文のまま**再利用。
  matched専用MC builderは使わず、その既存schedulerへ明示的な人工B2 taskを注入した。
  新schedulerやsamplerは実装していない。これは本格B2 MCの並列入口の完成を意味しない。
- 人工fixture/独立CAR oracleも既存testの関数を再利用。
  [runner][RUNNER]は人工入力だけを生成し、`np.load`と分子solverをprocess内で禁止する。
  [人工回帰test][TEST]と前回adapter testは**21/21 PASS、9.77秒**。

既存full-wrapper streaming/cache/online統計の`df_rpe_hadamard_compiled_cost`も調査した。
plannerはQ1/2/4限定で、元のままでは長roundには**NOT_APPLICABLE**。
境界融合builder自体も全step結果を作るので、そのままの利用はrequest memory問題を解決しない。

## whole compileとの比較と意味検査

9ケース＝Q1/2/4/8×LD2/3のmixed pattern 8件＋Q4/LD2のrepeat pattern 1件。全ケースで両軸を測定した。

| 方式 | 全6指標EXACT_MATCH | APPROXIMATE | 意味の検査 |
|---|---:|---:|---|
| 同一gate列streaming | **18/18** | 0 | PASS |
| 既存boundary fusion | 4/18（Q1） | **14/18** | PASS |
| 分割compile後の接続回路 | 0/18 | **18/18** | PASS |
| 同一回路のcache再利用 | 全hitで一致 | 0 | compilerを再実行せず同じ価格を返す |

各軸のreference/candidate/符号付き差/絶対差/相対差は結果JSONに保存した。
以下は人工取得集合内の最大絶対差で、各列の最大値は同じケースとは限らない。

| 指標 | streaming | boundary fusion | 分割後接続 |
|---|---:|---:|---:|
| CX count | 0 | 28 | 8 |
| RZ count | 0 | 35 | 44 |
| CX depth | 0 | 28 | 8 |
| RZ depth | 0 | 21 | 20 |
| total depth | 0 | 49 | 42 |
| circuit size | 0 | 63 | 76 |

例：人工Q8/LD3/XのreferenceはCX1380/RZ2191/depth3131/size4386。
fusionは1352/2156/3082/4323、分割接続は1388/2235/3173/4462。
fusionの最大相対差はCX count/depthの**2.602%**、分割接続はRZ countの**2.310%**。
これらはH4の誤差保証・価格の補完値ではない。

独立CARでcontrol-0 identity・control-1 evolution、signed event、forward/reverse順序、scalar/identity相対位相を
**位相fitなし**で確認した。3方式の最大operator差9.43e-15、X=Re/Y=Imのwrapper差3.27e-15。
ancillaはsystemの後、systemはQiskit LSB配置。測定は1 clbitへ1回で、state preparationなし。
compile後も全参照/候補のancilla測定効果が許容1e-10内（最大約3.15e-14）。
whole wrapperの無害なglobal phaseはchannel/測定効果で扱い、controlled内部の相対位相を合わせ直していない。
whole referenceのchannel差も最大3.50e-14。この意味のPASSと価格一致の分類は別fieldで保存した。

差の原因：fusionは隣接outer step間のblock0 half sweepsを一つにし、scalarを集約する別のgate列。
今回Q≥2ではreferenceより境界ごとCX4/RZ5/depth7少なく、元compilerの価格を再現する方式ではなかった。
分割は各部分でcompilerを完結させるため、境界をまたぐ1q合成・inverse cancellationが失われる。
固定Qiskit1.3のlevel1 SOURCEにある`Optimize1qGatesDecomposition`、`InverseCancellation`、depth/size固定点loopと整合する。
追加の人工回帰例でも、分割境界をまたぐCXの取消が価格差を作ることを確認した。
測定位置・wrapper境界も元の順序のまま分割し、最終測定を二重計上していない。

分割候補のdepthは**接続したcompiled DAG**から測った。ブロックdepthの単純和はRZ/total depthで最大5過大だった。
一定幅の境界だけを見れば常にwhole compilerを再現できるという証拠はない。
正しい代替回路であることと、固定whole-compile価格の同値性は区別する。

## 性能・メモリの実測とm17への限界

人工検証全体221.451秒。build計測はtracemalloc込み、whole compile＋価格計測は各軸.101～1.499秒。
compiled DAG operation数は490～4386。Q8/LD3のnativeは両方式745 instruction。
このケースのbuildはreference3.911秒/stream4.141秒、Python allocation peakは64,185,034/62,704,690 bytes。
小さい削減は観測したが、build高速化や大幅なメモリ削減は確認できていない。
process RSS high-waterはmainで1,183,200 KiBであり、import等を含む累積値。方式間の増分RSSとは扱わない。

人工Q64の**requestだけ**の診断はmaterialized182,856 bytes/.0723秒、stream191,960 bytes/.0809秒。
この小fixtureではpeak低下を観測していない。O(1)個のrequest/tail保持という処理構造と、
fixture生成・検証・allocatorを含む実測peakを混同しない。Q64回路のbuild/compileはしていない。

cacheは36 misses/36 hits、**36回の重複compileを回避**した。persistent cache復元、実回路identity変更時のmiss、
重複観測を3件とも既存accumulatorへ数えることも人工testで確認した。
warm lookupにも再帰的fingerprint作成の費用が残る。JSONの`compile_cold_seconds`は最初のcache request全体の時間で、
純粋なtranspiler時間ではない。同じshared cacheの先行候補でhitする可能性もあり、計測名から速度優位を推定しない。

同じ11 task（重複2件を保持）のserial23.108秒、parallel13.027秒、今回のspeedup **1.774**。
最初のtaskをprobe後、既存schedulerが空きCPU/RAMから残りに10 workersを選んだ。probe RSS472,456 KiB、
観測空きRAM516,667,012 KiB。worker内数値thread1、submission-order reduction、入力/軸別価格/fingerprint全一致。
workerのcache/tempは専用`.venv/cache`下。共有設定・他process操作なし。将来の長回路で同じ並列度を固定しない。
これは小さい人工taskの一回の観測で、H4やm17のspeedup/worker RAMの見積りではない。

m17の固定構造はQ131072、524288 events、1048576 deterministic block appends/trajectoryのまま。
今回のstream adapterはQ個のrequest/tail listを作らないが、最終回路、wrapper copy、再帰的fingerprint payload、
transpiler DAG/最適化一時領域はO(回路長)で残る。さらに既存MC providerはQ*r event tuple、既存parallel MCは全task列を保持する。
**m17の実instruction/RAM/compile時間/価格はnull、INCONCLUSIVE。** 人工価格やRSSをQ倍していない。
独立trajectoryの並列化は1本の巨大DAGのメモリを解決しない。

## MCへ進む条件と次の担当

既存samplerはPCG64のorder→component選択を逐次生成し、providerはfresh Q*r blockごとに両軸を共有する。
人工seed771の回帰でevent streamの一致を確認しただけで、本格MCのseedや標本数を採択していない。
既存`CompiledMetricAccumulator`のmean・不偏分散・SEは再利用でき、**CIとpaired covarianceは未実装**。
将来の固定shot数N_X/N_Yについて、paired W_i=N_X*C_X_i+N_Y*C_Y_iを既存accumulatorへ渡せばcovarianceを含むSEを得られる。
これは方式の提案で、今回G_mや総費用を計算していない。量子shotの失敗確率と古典価格MC誤差は別に管理する。
CI候補・rare-event/optional-stoppingの条件、seed分離、独立holdoutの必要性は[統計設計JSON][STATS]に整理した。

| 証拠段階 | 対象 |
|---|---|
| 人工で検証済み | 同一gate列streaming、既存cache、B2 taskへの既存scheduler注入、fusion/分割の観測残差 |
| 追加実H4検証が必要 | streamingのnative fingerprint/全6価格の一致、代表サイズでの実memory/時間、MCへの接続 |
| 提案段階 | bounded task submission、event列全保持の解消、fingerprintの省メモリ化、compilerのincremental化、独立holdout付きproxy |

**次の実H4限定検証の提案**：まず前回m0の同じ保存trajectory・同じcompilerでstreaming X/Yを確認し、保存済みnative/軸別fingerprintと
価格に照合する。outer-step境界の実H4確認には、その後の短いQ2等の一つの明示trajectoryについてreference/stream比較を別途固定する。
いずれも今回未実行・未承認で、追加trajectory探索や期待価格推定へ拡張しない。

全18round MCには、上記実H4資格、最大roundを処理できる方式と実資源証拠、bounded submission、round/sample count/seed、
paired6指標、SE/CI・精度目標・停止規則・rare-event処理、再現性、proxyなら独立holdoutとmodel-error扱いの固定が残る。
同一gate列streaming＋cacheを優先し、fusion/分割価格を元のwhole価格として採用しない。

**ユーザー**：次の実H4限定検証を行うか、その固定SOURCE・対象・trajectory・compiler・STOP範囲を判断して明示承認する。
**Codex**：SOURCE・人工結果・本REPORTを専用branchで公開し、remote SHA/tree/全変更内容を照合して**mandatory STOP**。
実H4追加compile・価格MC・強PF価格・信号/branch/shot・総費用・GPT詳細科学レビューへ自動進行しない。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/c897389643a8d3b81ff63f9a0d71b250b17a565e
[PREVIOUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d21c23879aaae063a796fa95c1433fb3d0b5738/docs/research/track_a_b2_p1_compiled_cost_20261107/REPORT.md
[ADAPTER]: ../../../src/trotterlib/b2_streamed_native_cost.py
[PARALLEL]: ../../../src/trottertracks/o1_m1_parallel_price.py
[RUNNER]: ../../../scripts/run_b2_cost_method_validation.py
[TEST]: ../../../tests/test_b2_streamed_native_cost.py
[RESULT]: ../../../artifacts/track_a_b2_p1_long_cost_method/2026-11-07/run01/result.json
[SUMMARY]: ../../../artifacts/track_a_b2_p1_long_cost_method/2026-11-07/summary.json
[STATS]: ../../../artifacts/track_a_b2_p1_long_cost_method/2026-11-07/statistics_design.json

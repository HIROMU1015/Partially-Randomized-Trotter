# B2長round価格pipelineの省メモリ化と人工検証（2026-11-07）

人工回路だけで実装・compile・性能検証を行った。実H4追加計算、価格MC、強PF価格、総RPE費用算出は実施していない。
実H4 m17の実行可能性・価格は未確定である。既存の科学条件・compiler profile・whole compileの定義を維持した。

## 実装と再利用

基点は`c6f3f07914feca4f1233fd54951212aeffee001d`。
専用branchは`codex/track-a-b2-memory-pipeline-20261107`。主測定SOURCEは`0c5ef07a7e23fad88fe151ebf50c9a3c58cdb013`。
GC修正と最終補足のSOURCEは`3763b4b9966dbd039ed492791a183a371c00822b`。
旧SOURCE・raw・grant・ledger・marker・STOPは元commitと既存pathに保全した。
主測定後のGCなしweakref検査で、参照削除だけではQiskitの循環参照が残ることを検出した。
省メモリpair自身が各軸の終了時に回収する修正を加え、observerのstage前GCを無効にした最大条件で補足測定した。
初回task参照の解放、未消費eventのcompile前判定、旧duck provider互換も追加検証した。
その後、価格関数が所有するnative回路の返却後の循環参照も検出し、返却前の解放・回収を追加して人工回帰で確認した。
pair補足はnative入力を保持した状態であり、この返却後処理の時間/RSSは測定対象に含まない。最終コードのhashはtests.jsonに記録した。
flat生成・canonical bytes・数値metrics・compiler profile・RSSの境界観測方法は維持した。

- `MatchedNoPrepCostProvider(..., memory_efficient=True)`を追加。Xの価格を取り出した後、wrapperとcompiled circuitへの参照を解放してYへ進む。
  cache本体に加え、miss返却値もmetric-onlyにし、循環参照は軸境界で明示的にGCする。既定の従来経路は維持した。
- `TranspiledCircuitCostCache`に同一canonical policyのhasherとmetric helperを注入できるようにした。
  cache lookupのたびに実回路を走査する。caller labelでidentityを代用せず、compiler/backend/targetのkey・unsupported時のbypass・persistent schemaを維持する。
  `cached_compiler`の重複hashを除き、省メモリproviderではwrapperの事前hashもcache側の実hashへまとめた。
- 検証済みstreamed encoderを再利用。CX/H/SX/Sdg/Xの5 singletonだけは**そのhash呼出内**でencoded operationを再利用する。
  hash間に値を保持しないため、definitionの改変も次のhashで検出する。mutable/custom gateはmemo対象にしない。
- compiled metricsはflat rz/sx/x/cx/measure/barrierに限って1回の走査で取得する。
  Qiskit1.3.0と同様、filterから除いたgateもwireの同期点として扱う。条件付きgate・変数・その他のgateは従来helperへ戻る。追加の診断DAGを作らない。
- `price_streamed_trajectory`を既存`stream_native`・step request iterator・native wrapper・pair compilerへ接続した。
  event/requestはr個ずつ消費し、出力は既定でevent数と旧JSON-listと同じordered digestだけを残す。
  必要なら`retain_trajectory=True`で旧形式の全eventを返せる。caller自身が全eventを保持する場合、そのメモリは残る。代表trajectoryを期待価格へ格上げしない。
- B2の既存price入口も省メモリprovider選択時にはevent iteratorをstream builderへ渡す。
  このMC入口はdispatchを置換した人工テストで配線だけを確認し、sampler・価格MC・統計集計は実行していない。
- 既存`ParallelMatchedNoPrepCostProvider`のspawn initializer・memory probe・submission順reductionを再利用した。
  `_iter_map`はtaskを遅延生成し、pending futureをworker数以内に保つ。MC task生成側もlistからiteratorへ変更したがMC実行はしていない。
  互換用`_map`は従来どおり結果listを返す。stream利用側には結果全件の保持を要求しない。

使用入口は次のとおり。`round_`と`events`は呼出側で対象を固定した既存オブジェクトである。

```python
provider = MatchedNoPrepCostProvider(compiler, memory_efficient=True)
result = price_streamed_trajectory(round_, provider, events)
```

## correctnessと回帰

人工two-mode B2 rank4 fixture、既存CAR oracle・channel/measurement-effect比較、既存fingerprint/cacheテストを再利用した。
全6価格指標は`rz_count, rz_depth, cx_count, cx_depth, total_depth, circuit_size`。
native/入力fingerprint、位相、測定bitの±1意味、ordinary-controlledの相対位相を維持し、測定は各軸1回でstate prepなし。
全3規模でnative fingerprintと全6価格が保存値に直接一致。wrapper入力fingerprint・compiled gate列/位相は全規模のtrace on/off間と、小条件の従来pairとの直接比較で一致した。
wire同期・classical wire・barrier・fallbackのdepth、入力/definition改変、compiler seed分離、persistent hit、backend/target/bypassを確認した。
weakrefでXのwrapper/compiled circuitがYの生成前に、関数所有のnativeが価格返却前に解放されることを、observer GCなしで確認した。
task generatorはcompletion順が異なってもsubmission順で返り、pending数がworker数以内となる。
これらは主agentの人工実装検証であり、独立科学レビュー・実H4の性能証拠ではない。

## 測定方法と比較の範囲

[runner](../../../scripts/benchmark_b2_memory_pipeline.py)は既存147 basis instructionの人工unitと、修正済みの外部RSS observer/StageMeterを再利用する。
主測定は9 qubit上で4,116 / 65,562 / 262,248 instructionsへ増やし、tracemallocなし/ありを別workerで測った。
主測定のobserverは各stage前にGCを行うため、実運用でのobject回収の証明には単独で使わない。最終補足は最大条件・tracemallocなし・observer GCなしである。
unitのcontrol-0恒等性は既存テストで確認済み。人工block数28/446/1784はB2 outer stepやQを表さない。
compilerはQiskit1.3.0、rz/sx/x/cx、optimization1、seed7、logical all-to-all、backend/target/couplingなし。

小条件では従来pairと改善pairを同じrunnerで比較する。大条件のlegacy hash・旧compiler/metric stageは
[既存raw](../../../artifacts/large_artificial_memory/2026-11-07/summary.json)を再利用し、同じnative fingerprintと全6価格値で対応を確認する。
旧測定はcopy/DAG診断・object保持順が異なるため、**大条件の全pipeline時間を直接比較した実験ではない**。
新しいwrapper stageにはnative改変検査も含む。旧wrapper-copy単独stageとの時間比は出さない。
pair wallにはobserver handshakeとcompiled gate列digest診断を含む。digest時間をproduction price時間へ含めない。

RSSはparentが自workerの`/proc`を公称5ms間隔で観測し、stageで新たに増加したkernel HWMも使った。
実際の間隔とbegin/end境界時刻を保存し、全stageで時間整合を確認する。短い未観測peakに対する厳密上限ではない。
Python値はstage中のtraced allocation peakで、既存のPython総使用量ではない。RSS−PythonをRust/Cの厳密量にはしない。
tracemallocありの時間/RSSにはtracer自身の負担がある。各条件1観測であり、通常実行の上限・速度保証・統計的な差の証明にはしない。

## 実測結果と成果物

[集約JSON](../../../artifacts/b2_memory_pipeline/2026-11-07/summary.json)・[主測定CSV](../../../artifacts/b2_memory_pipeline/2026-11-07/run01/stages.csv)・[補足CSV](../../../artifacts/b2_memory_pipeline/2026-11-07/run02/stages.csv)・[比較CSV](../../../artifacts/b2_memory_pipeline/2026-11-07/comparison.csv)・[主raw](../../../artifacts/b2_memory_pipeline/2026-11-07/run01/result.json)・[補足raw](../../../artifacts/b2_memory_pipeline/2026-11-07/run02/result.json)・[テスト結果](../../../artifacts/b2_memory_pipeline/2026-11-07/tests.json)。

人工回帰80項目PASS（cohort間の重複を含むPASS実行112件）。13 worker・121 stage、whole compile18回＋unit setup compile2回。
request測定のfull native build/compileは0回。軸境界1件・native所有権2件の追加テストで発見したGC依存は、原因・修正・最終PASSをtests.jsonに記録した。

小条件4,116 instructionsの同時pair比較（run01、両方式ともobserverのstage前GCあり）：

| tracing | 方式 | pair wall 秒 | 全stage観測peak RSS MiB | 最大stage Python peak MiB | 保持compiled数 |
|---|---|---:|---:|---:|---:|
| off | reference | 5.499 | 332.21 | 未測定 | 2 |
| off | efficient | 3.781 | 292.22 | 未測定 | 0 |
| on | reference | 18.411 | 378.59 | 46.283 | 2 |
| on | efficient | 10.320 | 302.99 | 6.961 | 0 |

最大262,248 instructions：afterは最終production GC経路のrun02（tracemallocなし、observer stage GCなし）。
beforeは保存済みstageであり、全pipeline同時比較ではない。

| component | before 秒 | after 秒 | before peak RSS MiB | after peak RSS MiB |
|---|---:|---:|---:|---:|
| flat_generation | 0.265 | 0.280 | 291.88 | 298.32 |
| native_fingerprint | 50.015 | 37.649 | 432.48 | 311.82 |
| cosine_whole_compile | 1.088 | 1.425 | 451.32 | 458.45 |
| sine_whole_compile | 1.049 | 1.161 | 502.95 | 479.56 |
| cosine_compiled_metrics | 3.233 | 1.584 | 387.32 | 395.61 |
| sine_compiled_metrics | 3.319 | 1.597 | 437.32 | 414.60 |

最終補足のworker全体観測peakは479.56 MiB、pair wallは161.930秒、native保持中のpair終端RSSは414.60 MiB、保持compiled数0。
hash・metricsの数値処理は同じだが、observerが各stageで回収する初期pilotと実運用GCの履歴を区別する。
compiler自体の時間/RSSは一律には減っていない。主な改善は全payload hashの回避、同時保持の削減、metrics走査の削減である。

人工requestだけの比較（r=4、tracemallocあり、sampler/full native build/compileなし）：

| Q | 方式 | 秒 | peak RSS MiB | Python peak MiB |
|---:|---|---:|---:|---:|
| 1,024 | materialized | 1.135 | 289.40 | 0.788 |
| 1,024 | stream | 1.251 | 287.27 | 0.672 |
| 131,072 | materialized | 148.733 | 409.94 | 58.204 |
| 131,072 | stream | 154.353 | 288.26 | 0.739 |

CPU affinity 64、観測最低空き473.56 GiB。資源不足STOPなし。
全121 stageの終了境界を確認し、最大sample gapは18.424ms。
時間比較の自worker間競合を避け、測定条件は順次実行した。CPU/RAMのcapは加えず、共有設定・他processは変更していない。

## 残る障害と実H4 m17へ必要な検証

最終flat native、少なくとも1軸のwrapper、Qiskit compiler内DAG/workspaceは依然として長さに依存する。
object解放とプロセスRSSの即時低下は同義ではない。allocatorやnative定義の保持を含み、長期workerの異なるtrajectory連続処理は未測定。
copyをなくすためのoperation共有は新しいalias/mutation問題を生むため採用していない。分割compileもwhole価格の代用にしていない。
native hash1回、各軸のnative改変検査・wrapper実hashの計5回のcanonical走査が残り、長回路ではCPU時間の主因になる。
今回のoperation memoは小さい定義だけに限定した。巨大custom gateの単一定義展開に関する既存のメモリ限界は残る。
parallel dispatcherはbatch全体を保持しなくなったが、従来MC workerの1 taskには依然として1 trajectoryのevent tupleがある。
同時worker数の選択は実測peakと実際の空き容量に依存し、人工回路のpeakを実H4へ流用できない。

実H4へ進むには対象を別途固定し、次の検証が必要である。今回これらは実行していない。

1. 保存DF/source、P1のm/r/K/delta・event列・basis policy・compilerを固定した少数の実H4条件で、旧経路との全6価格・native/入力identity・測定意味を照合する。
   最小の追加検証案は[保存Q2検証](../b2_q2_fingerprint_h4_20261107/REPORT.md)と同じH4/1.00Å/STO-3G/DF12、LD3、delta=.02、r4/K2、m1/Q2の固定8event
   （identity `0ecf28e7a5d2d367c6c609a8d8a4da544c2d03b955aef86b0656f0bf107b7ae8`）に、新経路でnative1回・X/Y各1回compileするもの。
   保存referenceを再build/compileせず照合し、m17はそのroundの固定P1条件を別に指定する。今回この追加検証は行っていない。
2. 長roundの代表event列でnative規模、wrapper/hash/whole-compile workspace、wall time、実際の空き容量を測り、段階的な実行可否を判断する。
   人工Q=131072のrequest保持比較は、この実H4 build/compileの代わりにはならない。
3. 価格MCを行う段階では、fresh IIDの同一乱数列、paired axes、反復drawの非dedup、SE・reduction順序を新接続で確認する。
   同一trajectoryのcache hitを新しいsample数や独立標本の追加と解釈しない。総RPE費用や強PF比較は別段階である。

次段階で可能な技術作業は、実回路の対象条件を具体化すること、compiler workspaceとhash走査の更なるprofile、parallel workerへのevent stream配送の設計である。
実H4 m17の価格、必要RAM、所要時間、MC精度、総費用は未確定。

**成果物のcommit/pushとremote SHA・tree・公開内容照合後mandatory STOP。実H4追加検証や価格MCへ自動進行しない。**
次の担当：ユーザーが追加検証の対象・条件・範囲を判断し、Codexが具体化した範囲を次の指示に従って実施する。

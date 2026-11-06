# Codex C：B2 P1 full no-prep価格pipelineの接続

2026-11-07 JST / SERVER 2（GPUA100-gleap02）/ Codex主エージェント。

**m0/Q1の既存固定trajectory一つについて、native→X/Y測定付きfull wrapper→既存compilerの接続がPASS。**
X/Yは各CX **3,262**、RZ **6,025**、depth **5,504**。
これは指定trajectory・m0・指定compilerのcompiled costであり、期待価格・価格MC・総RPE費用ではない。
指定した2 compileを完了し、結果公開後**mandatory STOP**。他round・強PFへ進まない。

機械可読成果：[実行result.json][RESULT]、[成立性feasibility.json][FEASIBILITY]、
[人工検証summary][VALIDATION]。取得していない価格・長回路実測はnull。

## SOURCE・科学条件・compiler

- 実行SOURCE：[`5aa5d061600382abe438707c14eaf665466ae383`][SOURCE]。
  shot成果`0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`から専用worktree/branchを作成。
  SOURCEは実行前にcommit固定し、消費するcoreの作業ファイル一致、終了時の不変を確認した。
  通常のHTTPS pushは認証情報未設定で失敗したため、接続GitHub APIで**同一tree**を公開した。
  このSOURCEリンクは公開snapshot `b443f25…`へ向く。元commitと公開commitのmetadata差、
  SOURCE/result両treeの一致は[publication.json][PUBLICATION]に記録。
  元SOURCE/結果commitは[標準Git bundle][BUNDLE]から復元でき、科学SOURCE/raw値は不変更。
- Linear H4/1.00Å/STO-3G、energy target=cutoff前H_phys、保存DF12、generation-prefix LD3、
  partial-S2/r4/K2、delta=.02、m0/Q1、元H保存基底stateの仮定、prep費用除外。
  核scalar 2.29310124732、tail identity .09261516589219072、signed factorsとbit-orderを保持。
- DF入力は`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`の既存snapshot。
  SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
  canonical DF hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。
  既存loaderで照合し、そのsnapshotのDF stateは破棄。新DF/SCF/state solveなし。
- 元H/state/sector/orderは`89165cf…`のinput specと`ea2c091…`の保存結果に結合した。
  private state/provenance/HDF5はfile hashのみ確認し、物理Hやstateを再生成・再解析していない。
  private payload・積分・provenance原本は公開していない。
- compilerの根拠は**実価格pilot** [`0daf01d…`][PRICE] の保存resultと実行SOURCE `1b64e46…`。
  **Qiskit1.3.0、basis rz/sx/x/cx、optimization1、transpiler seed7**。
  backend/target/coupling map/layout/routing指定なし、論理all-to-all。
  保存X/Y価格のcompiler・scope/backend、今回の実runtime、compiler wrapper SOURCEの同一性を照合。
  P1信号metadataのseed17を実価格profileとして採用していない。
- compiler hash **`a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`**、
  backend fingerprint **`no_backend`**。これは意図的なbackendなしの識別子で、欠落値ではない。
  Python3.12.3/NumPy1.26.4/SciPy1.14.1/OpenFermion1.6.1/Qiskit Nature0.7.2。

## 再利用・測定scope・correctness

`B2RPERound.explicit_request`→`B2RPERound.native`→既存`partial_native_from_step_requests`、
`build_native_hadamard_wrapper`、`MatchedNoPrepCostProvider`の`_compile_pair`、
`transpile_and_measure_cost`をそのまま使用。新規は薄い[adapter][ADAPTER]、[runner][RUNNER]、
[人工test][TEST]、[PLAN](PLAN.md)。既存sampler・native builder・compiler・集計器を再実装していない。

共通scopeは`single_hadamard_interrogation_without_state_preparation`。
全Q stepのordinary controlled決定論sweep・finite-RTE・basis transition・scalar/identity相対位相、
H→evolution→H、YのみSdg→H、ancilla測定を含む9 qubit/1 clbit回路全体をcompileした。
Full S2/Yoshida4も既存`price_strong_baseline`/providerのdeterministic経路でこのscopeを使用できるが、今回はbuild/compileしていない。

既存[`ea2c091…`][CORRECT]の**16/16 PASS**を再利用：control-0 identity、control-1の位相合わせなし対照、
核scalar/tail identity一回計上、signed event、forward/reverse half sweep、X=Re/Y=Im。
今回消費する12 coreファイルは、その実行SOURCE `5ef55f9…`とbyte一致。
この証拠は代表primitive/complete Q1 step/理想平均lawに限られ、全support・全trajectory・長回路compiler保証ではない。

人工検査は新adapter **8件PASS**、既存B2接続 **47件PASS**。既存47件は同じ実装で一度、新8件は修正後4.29秒で確認。
初回の新testはwhole wrapperのunitary行列そのものの一致を要求して1件失敗した。
測定前回路全体のglobal phase差を切り分け、**全量子channelとancilla測定効果を位相fittingなしで比較**する適切な検査へ修正した。
差は最大約4.12e-15、許容1e-10内。controlled evolution内の相対位相を合わせ直したものではない。
人工PASSは分子の科学的性能や全長compiled correctnessの証明ではない。

## 限定実回路の価格

既存controlled correctnessの**outer case1**を再利用し、保存代表event indices **[0,4,4,0]**を選んだ。
first→first negative order2→last→firstという既存規則を実行前PLANで固定。
orderは[0,2,2,0]で、すべて既存canonical `_make_event`と`explicit_request`で照合した。
RNG・探索・価格による選択なし。非常に小さい確率の代表成分も含むため、典型trajectoryとも扱わない。

trajectory identity：`c3ea48da7dcd082f948b6df9879b3105fc44145c438342396a9777d0b409fbc5`。
完全なevent law/probability/sign/basis identityと順序は[trajectory.json][TRAJECTORY]。

| 指標（compile後） | X/cosine | Y/sine |
|---|---:|---:|
| CX count | 3262 | 3262 |
| RZ count | 6025 | 6025 |
| CX depth | 2658 | 2658 |
| RZ depth | 2526 | 2526 |
| total depth | 5504 | 5504 |
| circuit size | 11494 | 11494 |
| measure count | 1 | 1 |

`RPE_COST_METRICS`の6指標を使用。countはcompile後の当該gate数、filtered depthは当該gateだけを
対象にしたQiskit `depth(filter_function=...)`、total depthは測定を含む全回路depth、sizeはQiskit `size()`。
総sizeにはSX/X/測定も含み、CX+RZだけの和ではない。
native evolutionは812 instruction、X wrapper815、Y816、保守的request bound1316。
X/Y同価格はこの2回路の観測に限る。

両軸のSOURCE・round binding・scope・compiler/backend・compile前後fingerprint・gate countsを
[X][X]と[Y][Y]へ別々に保存した。工程全体24.615秒、各compile＋計測は6.170秒/6.107秒。
process peak RSSは1,183,200 KiB（約1.13GiB）で、import/準備/計測を含むhigh-water値。
compileだけの増分メモリではない。任意のCPU/RAM cap・共有設定変更・他プロセス操作なし。
価格未取得・期待値・sample SE/CI・総費用は**null**で保持し、0で埋めていない。

## 長round・価格MCの成立性と残る条件

- m17はQ=131072、**524288 events/trajectory、1048576 deterministic block appends**。
  これは固定r/LDとbuilder構造からの計数で、長回路を展開・compileした実測ではない。
  実際の長回路instruction数・RAM・compile時間は未取得。m0の価格やRSSをQ倍していない。
- B2の`explicit_request`はQ個のoccurrence/seed、`iter_step_requests`はtupleを作る。
  native loweringも全stepのcontrolled tail回路をlistに保持し、最終flat circuitへcomposeする。
  上限整数を大きくするだけでstreamingにはならない。compiler DAG・gate定義の展開・最適化・
  再帰的fingerprint生成の一時メモリも必要。**現実装による全18round価格MCの実用的成立性は未確立**。
  長回路実測なしに不可能とも認定しない。
- RTE builderの既存adjacent-basis cancellation、scalar集約は利用済み。
  旧repeated builderにはboundary fusionがあるが、B2 native経路へ無条件移送していない。
  compilerの境界をまたぐ融合・取消があるため、step/event価格の和や短回路反復は実compiled価格ではない。
- 旧`df_rpe_hadamard_compiled_cost`にはfull-wrapper stream/cache/online統計があるが、
  現行入口はQ=1/2/4を要求し、B2 nativeの長roundへそのまま接続できない。
  cacheはactual circuit/compiler/backend identityで再利用できても、drawの重複を除いてはいけない。
- `B2RPERound.price`→`monte_carlo_with_builder`→既存canonical samplerのfresh Q*r eventsを
  draw間で独立に生成し、同じdrawをX/Yで共有する経路は既存・人工検証済み。
  `CompiledMetricAccumulator`は不偏標本分散とsample SEを出すが**CIは出さない**。
  量子shot数/失敗確率とは別の古典価格誤差であり、軸を合わせる費用のSEにはpaired covarianceが必要。
- 候補は、既存並列providerのworker/メモリ観測・ordered reductionをB2 builderへ薄く接続する方法、
  exactな反復/境界の簡約、streaming/分割処理、独立holdout付きcluster/短block proxy。
  旧parallel providerはmatched builder固定かつ全task/event列を保持するため、そのままB2用にはならない。
  独立compileの並列化は1本の長いDAGのメモリを解決しない。
  分割compileやproxyはwhole compileとの一致/残差を別途検証する必要があり、今回採択・実行していない。

次のfull no-prep比較には、B2と強Full S2/Yoshida4を同じcompiler/backend/scopeで固定し、
baselineのsubsteps/basis/fusion条件、各roundの回路適格性、長回路処理方式を確認する必要がある。
次回価格MCではSOURCE/input/law、round範囲、標本数、seed/stream分離、paired X/Y、
全6指標、SE/CIの方法・目標精度・停止規則、rare-event処理、cacheと並列実行の同値性、
proxyを使うなら較正/独立holdout/誤差の扱いを別途固定・承認する。今回は新しい標本数・seed・round範囲を採択していない。

**ユーザー**：上記の長回路処理候補と価格不確かさの目的を判断し、必要なら次回の固定範囲を承認する。
**Codex**：結果を専用branchへcommit/pushし、remote SHA/tree/全変更内容を照合してmandatory STOP。
新信号/shot/branch、追加trajectory、他m/強PF compile、価格MC/外挿/総RPE費用、GPT詳細科学レビューは実行していない。
旧SOURCE/input/raw/result/grant/ledger/marker/STOPとCodex A/Bのworktree・成果物は不変更。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b443f25b30160917c5b4446380456c49e746fe82
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0daf01d51c022977a1c0a7d29dde34a1c4c82d7c/artifacts/track_a_o1_m1_r8_price_pilot_20261106/run01/result.json
[CORRECT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5/artifacts/track_a_b2_p1_two_round/2026-11-07/run01/result.json
[ADAPTER]: ../../../src/trotterlib/b2_p1_compiled_cost.py
[RUNNER]: ../../../scripts/run_b2_p1_compiled_cost.py
[TEST]: ../../../tests/test_b2_p1_compiled_cost.py
[TRAJECTORY]: ../../../artifacts/track_a_b2_p1_compiled_cost/2026-11-07/run01/trajectory.json
[X]: ../../../artifacts/track_a_b2_p1_compiled_cost/2026-11-07/run01/cosine.json
[Y]: ../../../artifacts/track_a_b2_p1_compiled_cost/2026-11-07/run01/sine.json
[RESULT]: ../../../artifacts/track_a_b2_p1_compiled_cost/2026-11-07/run01/result.json
[FEASIBILITY]: ../../../artifacts/track_a_b2_p1_compiled_cost/2026-11-07/feasibility.json
[VALIDATION]: ../../../artifacts/track_a_b2_p1_compiled_cost/2026-11-07/synthetic_validation.json
[PUBLICATION]: ../../../artifacts/track_a_b2_p1_compiled_cost/2026-11-07/publication.json
[BUNDLE]: ../../../artifacts/track_a_b2_p1_compiled_cost/2026-11-07/execution_commits.bundle

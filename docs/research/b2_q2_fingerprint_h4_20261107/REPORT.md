# B2実H4 Q2：opt-in fingerprint限定検証

2026-11-07 JST / SERVER 2 / Codex主エージェント。

**限定PASS。** 固定1 trajectoryの実H4 Q2で、新方式のnative行digest・fingerprint、X/Y wrapper、
compiled fingerprintと全6価格指標が保存Q2 referenceに一致した。
今回の科学実行は**stream native1回、X/Y各1回compile＝計2回**。旧referenceの再build/compileは0回。
SOURCE・入力の終了時不変確認もPASS。公開照合後**mandatory STOP**。

[RESULT][RESULT] / [summary][SUMMARY] / [完全trajectory][TRAJECTORY] / [native・wrapper比較][NATIVE] /
[X][X]・[Y][Y] / [人工test][TESTRESULT] / [PLAN](PLAN.md)。

## 固定SOURCE・入力・参照

実行SOURCEは[`90425140694cfb3fbe53ea949fa59fbde651ff83`][SOURCE]。
前段の人工pipeline成果`c71bb8227000bd544aa0c4f946c0e6cf28a80bee`を基点に、
新runner・人工test・PLANだけを追加し、同一treeのlocal準備commitを保全した。
公開SHA/treeを取得・照合したSOURCEで1回実行し、消費する20ファイルとprivate入力のhash不変を終了時に確認した。
hash/native/wrapper/pair compilerは前段の実装から不変更。既定経路への一括切替は行っていない。

- Linear H4/1.00Å/STO-3G、保存DF12、cutoff前H_phys target、generation LD3、partial-S2/r4/K2、
  delta=.02、m1/Q2/T=.04、ordinary control、元H基底state仮定・状態準備費用除外。
  signed factors、one-body、核scalar、tail identity、spin/JW・LSB bit order、ancilla最後を維持。
- DFは既存`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`の保存snapshot。
  SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、
  DF Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`を照合。
  既存loaderが返すDF stateは破棄し、元H/state/sector/order metadataとprivate 3原本のfile hashだけを既存証拠へ照合。
  SCF/DF生成/state solve、private原本の公開、物理H再構築なし。
- Step1 `[0,4,4,0]`、Step2 `[4,0,0,4]`。完全event identityは
  `0ecf28e7a5d2d367c6c609a8d8a4da544c2d03b955aef86b0656f0bf107b7ae8`。
  既存helperで復元した全8 eventの値・符号・確率・basisを保存payloadと比較し、canonical lawを検証してbuild前に保存。
  新SOURCEに伴うround bindingは別記。価格を見た選択、RNG、代替trajectoryなし。
- referenceは[`ce2e6736ca1178a4adafc4af344852d958951ea6`の保存Q2成果][REFERENCE]。
  m0の倍算や新しい従来nativeの測定ではなく、その実測済みQ2をそのまま参照した。
  旧m0/Q2・人工検証のSOURCE/raw/STOP、grant/ledger/marker、他worktreeは変更していない。

compilerはQiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/targetなし・論理all-to-all。
hash `a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`、backend `no_backend`。
scopeは測定を含む`single_hadamard_interrogation_without_state_preparation`。

## 再利用・identity・意味の確認

[runner][RUNNER]は旧Q2/m0のinput/law/serialized event helper、Q2 event順序、native行digest、軸別6価格比較を再利用。
前段の段階timerとopt-in stream native/wrapper/pair compiler、既存`transpile_and_measure_cost`を使用した。
builder・sampler・wrapper・compiler・価格集計の再実装、fallback、boundary fusion、phase fittingは行っていない。

**人工12/12 PASS、54.04秒**（新adapter/STOP 5件＋既存pipeline 7件）。
人工の保存referenceで1-build/2-compileの再現、native/両wrapper不一致でcompile前STOP、
価格不一致/例外で最初のcompile後にSTOPすることを確認した。元のcontrolled CAR oracleとX/Y規約も再利用。
保存済み実H4の代表controlled correctness **16/16 PASS**は同じ証拠範囲で再利用した。
今回、新しい独立H4 operator/statevector/signal simulationは行っていない。
主エージェントのlocal技術検証で、独立reviewerやCIによる再現ではない。

nativeは1,623 instructions、system8＋ancilla1、clbit0、global phase=0。
行digestは`2a4bc5aea5492e021a8778f91048e51cb175dca6760b0032ff786e3956450630`、
recursive fingerprintは`eca7ca67f91f3b422e014a0b7399d23fdc8a9af71a8776d3f4a3b180c4b43e48`で保存値と一致。
parameter・gate順・qubit/clbitの配置を行digest、gate定義とglobal/controlled相対位相を同一policyのfingerprintへ結び付けた。
X/Y wrapperは1,626/1,627 instructions、各9 qubit/1 clbit/測定1回で、保存fingerprint・gate countsも一致した。

| 保存referenceと一致するfingerprint | X | Y |
|---|---|---|
| compile前 | dda45b81af904e7fed1be73adf71e15f0dc9ae30bfe2a12aa44f2d9cc7b45edb | 983f69abbe34e43b8c2ba5499f2c7a5cf42af6cf56895467803395e967324277 |
| compile後 | cbfd2b0f05723fd97608d4a6f745b2e025ba7fb425d62adc6df622b6be1e1bdd | 524effa0e62f5a6e2572cf7e41a9d64689c67c21c007d0646fed6ad6beabe02c |

nativeと両wrapperの照合後にX→Yの順でwhole compileし、各軸で不一致なら次へ進まない。
2本ともcompiled global phase・fingerprint、compiler/backend/scope、測定数と全6指標が一致した。
compiled DAGは各22,458 operations。全support・全trajectory・長roundの科学的保証への拡張はしない。

| 指標 | 新X | 新Y | 保存Q2との差（各軸） |
|---|---:|---:|---:|
| CX count | 6444 | 6444 | 0 |
| RZ count | 11761 | 11761 | 0 |
| CX depth | 5295 | 5295 | 0 |
| RZ depth | 4999 | 4999 | 0 |
| total depth | 10904 | 10904 | 0 |
| circuit size | 22458 | 22458 | 0 |

## 時間・メモリ

全elapsedは198.250秒（runner開始後のpreflight・段階計測・同一回路のhash技術比較を含む、import時間は外）。
各stageはtracemalloc付きで、前から保持する入力全体を含まず、そのstageの新規Python割当を計測した。
native emissionと最終hash、wrapper改変チェックとcompose、pair入力hash、whole compile＋6指標、compiled hash/DAGを分けた。
両wrapperをcompile前に照合したうえで、pair compiler入口でも現在のfingerprintを再検証している。
同じ保存値を返すだけのcacheで改変検出を省略していない。

| 新pipelineのstage | 秒 | Python traced peak MB |
|---|---:|---:|
| flat native emission | 1.1524 | 3.4812 |
| 初回native hash | 16.7300 | 43.6910 |
| X native再検証hash | 14.4121 | 13.8093 |
| X wrapper compose＋測定 | 0.4373 | 4.3639 |
| X compile前照合hash | 18.1229 | 42.3061 |
| Y native再検証hash | 14.2052 | 14.9532 |
| Y wrapper compose＋測定 | 0.4795 | 4.2513 |
| Y compile前照合hash | 18.0920 | 43.7747 |
| X pair入力hash | 18.2003 | 45.8600 |
| X whole compile＋6指標 | 3.0895 | 5.3045 |
| X compiled hash | 20.4592 | 25.1882 |
| X compiled DAG | 0.0172 | 0.0046 |
| Y pair入力hash | 17.0405 | 43.7747 |
| Y whole compile＋6指標 | 2.5527 | 4.5354 |
| Y compiled hash | 17.7893 | 24.8617 |
| Y compiled DAG | 0.0167 | 0.0046 |

本体終了時にnative・両wrapper・両compiled回路を保持した現在RSSは495,016 KiB、
その時点までの累積process peakは684,860 KiB。import/preflight/lazy定義/allocatorの履歴を含む。
診断用compiled DAGは測定後に解放した。stage個別のRSS peakは未測定/nullで、Rust/C側割当もPython peakだけでは判断できない。

**同じ完成native回路のhash技術比較**（追加build/compileなし）：
本体終了後、warmなnative定義を使い旧→新を各1回測定した。両方のSHAは保存nativeと一致。

| 同一nativeのwarm hash | 秒（tracemalloc付き） | Python traced peak MB |
|---|---:|---:|
| 旧 | 14.6065 | 255.9409 |
| 新 | 13.1096 | 14.6455 |

この後の累積process peakは870,080 KiBで、旧hash診断の負担を含む。本体終了時のpeakと混同しない。
新nativeの初回hash peak 43.691MBと、後段warm比較の14.645MBも別の観測である。
旧Q2は2 native保持・4 compile・異なるtimer/cache順序を含むため、旧全体elapsed/RSSとの削減率は取らない。
CPU64、開始時load約52.86/51.88/40.25、MemAvailable約461GiB。
共有CPU負荷と順序/cacheの影響がある単回測定で、一般の速度優位やm17のRAM/時間を主張しない。
独自の資源cap、共有設定変更、他process操作なし。

## 残条件・次の候補・STOP

今回確認したのは、指定実H4 Q2での同一identity/価格と、hash新規割当peakの低下。
flat circuit、wrapper copy、whole compiler/DAG、個別instructionの再帰定義、event tupleは依然として長さ依存で残る。
hash/改変検証のCPU時間も残り、期待価格MC・強PF価格・全18round価格・総RPE費用は未評価。
m17の実build/compile実行可能性は未確定/nullで、Q2価格や時間から外挿していない。

**次の候補（未着手）**：より大きい人工flat回路でwrapper/whole DAGの保持とRust/C側割当を調べ、
現在RSSと、自分の計測processを対象とする定期観測で得たstage RSS最大値を分け、観測間隔も明記する。
科学的identity・改変検出を保ち、実H4 m2以降や価格MCを自動開始しない。

**ユーザー**：長roundへ向けた次の人工技術検証の範囲を判断する。
**Codex**：本結果を公開しremote SHA/tree/全変更ファイルを照合後、**mandatory STOP**。
今回の実行枠で追加の実H4 build/compileは行わない。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/90425140694cfb3fbe53ea949fa59fbde651ff83
[REFERENCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce2e6736ca1178a4adafc4af344852d958951ea6/docs/research/track_a_b2_p1_q2_streaming_h4_20261107/REPORT.md
[RESULT]: ../../../artifacts/b2_q2_fingerprint_h4/2026-11-07/run01/result.json
[SUMMARY]: ../../../artifacts/b2_q2_fingerprint_h4/2026-11-07/summary.json
[TRAJECTORY]: ../../../artifacts/b2_q2_fingerprint_h4/2026-11-07/run01/trajectory.json
[NATIVE]: ../../../artifacts/b2_q2_fingerprint_h4/2026-11-07/run01/native_and_wrappers.json
[X]: ../../../artifacts/b2_q2_fingerprint_h4/2026-11-07/run01/cosine.json
[Y]: ../../../artifacts/b2_q2_fingerprint_h4/2026-11-07/run01/sine.json
[TESTRESULT]: ../../../artifacts/b2_q2_fingerprint_h4/2026-11-07/tests.json
[RUNNER]: ../../../scripts/run_b2_q2_fingerprint_h4.py

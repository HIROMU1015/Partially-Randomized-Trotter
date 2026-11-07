# Codex C：B2 P1 m1/Q2実H4 streaming限定検証

2026-11-07 JST / SERVER 2 / Codex主エージェント。

**固定1 trajectoryのQ2で限定PASS。** outer-step境界を含む従来nativeとstreamingのgate列・global phase・native fingerprint、
X/Y wrapper fingerprint、compiled fingerprintと全6価格指標が一致した。
新規compileは**従来X/Y各1回＋streaming X/Y各1回＝4回**。Q2の従来方式を今回新たに測定してreferenceとした。
m0の再build/compile、価格の倍算、sampling・期待価格MC・長round展開はしていない。公開後**mandatory STOP**。

[実行結果][RESULT] / [完全trajectory][TRAJECTORY] / [native・wrapper比較][NATIVE] / [summary][SUMMARY] / [PLAN](PLAN.md)。

## 固定SOURCE・入力・trajectory

実行SOURCEは公開commit **[`e09fd13c5a2357dd001d9cfbe86d4e99c3a5620d`][SOURCE]**。
local準備commit `61e6722e…`と同一tree `e62796099ec9096d78dc4f66f354904addba0cce`を公開・取得・照合後、
実行HEADを公開SOURCEへ合わせて1回実行した。実行後も消費するSOURCEとprivate入力のhash不変を確認した。

- 基点は[m0成果 `f6eea2f2…`][M0]。元SOURCE/公開snapshot/bundleの対応、m0のrun01 STOPとrun02結果、
  人工技術検証 `1bed45be…`、旧SOURCE/input/raw/result/grant/ledger/marker/STOPは不変更。
  Codex A/Bのworktree・成果物・processへ干渉していない。
- Linear H4/1.00Å/STO-3G、cutoff前H_phys target、保存DF12、generation-prefix LD3、partial-S2/r4/K2、
  delta=.02、**m1/Q2/T=.04**、ordinary controlled、元H保存基底state仮定・prep費用除外を維持。
  signed factors/one-body/核scalar/tail identity、spin/JW/Qiskit LSBとancilla最後の配置を保持した。
- DFは既存`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`の保存snapshot。
  SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、
  Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`を照合。
  既存loaderでDF12/generation order/sectorを確認し、保存DF stateは破棄。元H/state/sector/order metadataと
  private state/HDF5/provenanceのfile hashも保存証拠に一致。新state solve/SCF/DF/物理H再構築なし。
- Step1 **[0,4,4,0]**、Step2 **[4,0,0,4]**、ordersは**[0,2,2,0,2,0,0,2]**。
  既存m0のevent metadataから全8 eventの値・符号・確率・basis・順序を固定し、既存canonical validatorで検証。
  完全trajectory identityは実行前に
  **`0ecf28e7a5d2d367c6c609a8d8a4da544c2d03b955aef86b0656f0bf107b7ae8`**と確定し、
  全event payloadをnative build前に保存した。RNG/探索/価格による選択・代替trajectoryへの変更なし。
  極小確率の代表成分を含む明示ケースであり、典型trajectoryや期待価格の標本ではない。

compilerは**Qiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/targetなし・論理all-to-all**。
hash **`a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`**、backend fingerprint **`no_backend`**。
scopeは`single_hadamard_interrogation_without_state_preparation`。
Python3.12.3/NumPy1.26.4/SciPy1.14.1/OpenFermion1.6.1/Qiskit Nature0.7.2。

## 再利用・native比較・意味の確認

新規は[Q2接続runner][RUNNER]、[人工test][TEST]、PLAN。
m0 runnerの入力/JSON表現比較・native/compiled比較・計測helper、既存`explicit_request`/`native`、
`iter_b2_step_requests`/`stream_native`、wrapper、`MatchedNoPrepCostProvider`のpair compiler、
`transpile_and_measure_cost`と価格指標を再利用した。既存科学SOURCE・builder/sampler/compilerは変更していない。

**人工34/34 PASS、16.42秒**。人工二modeのQ2 CAR oracle、control-0/1、signed event、scalar/identity、
forward/reverse順序、X=Re/Y=Im、Step2切替、実compiler4 call経路と失敗時停止を確認した。
人工PASSと今回の実H4 identity再現は区別する。

実H4のnativeは各方式1回構築し、**1,623 instructions**、system8＋ancilla1、clbit0。
gate名・対象qubit・parameter・順序・global phaseと再帰的fingerprintが一致した。
native fingerprintは両方式とも
`eca7ca67f91f3b422e014a0b7399d23fdc8a9af71a8776d3f4a3b180c4b43e48`。
2個のreference RTE requestに異なる4 event列が入ったことも保存した。
reverse→次のforwardの接続、Step2 request切替、T=.04のscalar/identity集約は、同一gate列として対応した。
phase fittingなし。X/Y wrapperはそれぞれ1,626/1,627 instructions、各9 qubit/1 clbit/measure1回で、両方式のidentityが一致。

過去の代表controlled correctness **16/16 PASS**を、同一のcore SOURCEと証拠範囲のまま再利用。
今回の人工Q2 oracleと実H4でのgate列一致を合わせた**実装上の限定的な確認**であり、
新しい独立H4 statevector/signal simulationや全support・全trajectory・長回路の厳密誤差保証ではない。
この実H4ケースのRTE eventは保存済み代表basisを使うため、すべてのbasis切替を走査したものでもない。

## 新規whole compileの価格比較

native/wrapper identity成立後、reference X→Y、stream X→Yの順に固定compilerで各1回compile。
cache省略、boundary fusion、別compilerへの変更や再compileはなし。
全Q stepのcontrolled evolution、basis transition、scalar/identity、Hadamard wrapperと測定を含むwhole回路を計測した。

| 指標 | 従来X | 従来Y | stream X | stream Y | 軸別差 |
|---|---:|---:|---:|---:|---:|
| CX count | 6444 | 6444 | 6444 | 6444 | 0 |
| RZ count | 11761 | 11761 | 11761 | 11761 | 0 |
| CX depth | 5295 | 5295 | 5295 | 5295 | 0 |
| RZ depth | 4999 | 4999 | 4999 | 4999 | 0 |
| total depth | 10904 | 10904 | 10904 | 10904 | 0 |
| circuit size | 22458 | 22458 | 22458 | 22458 | 0 |

各measure1回、compiled DAG operation数22,458。filtered gate depthと測定込みwhole depthを既存定義どおり使用。
入力/compiled fingerprint、compiler/backend/scope、qubit数、prep除外も各軸で一致した。
X/Y同価格はこの指定ケースの観測であり、一般の同価格性を主張しない。

| 両方式で一致するidentity | X | Y |
|---|---|---|
| compile前 | `dda45b81af904e7fed1be73adf71e15f0dc9ae30bfe2a12aa44f2d9cc7b45edb` | `983f69abbe34e43b8c2ba5499f2c7a5cf42af6cf56895467803395e967324277` |
| compile後 | `cbfd2b0f05723fd97608d4a6f745b2e025ba7fb425d62adc6df622b6be1e1bdd` | `524effa0e62f5a6e2572cf7e41a9d64689c67c21c007d0646fed6ad6beabe02c` |

4本の軸別JSONと全6指標の符号付き差/絶対差は[結果ディレクトリ][RAW]に保存。
referenceは今回のQ2実測であり、m0価格の2倍から計算していない。

## 時間・メモリの観測

全体125.686秒。入力/preparationを共通に用意し、同一processでreference→streamを各1回buildした。
build timerとPython traced peakにはrequest生成とnative fingerprintが含まれる。
reference保持・cacheの温まり・実行順序・共有CPU負荷の影響があり、単回の速度/削減率を一般化しない。

| Build観測 | 従来native | streaming |
|---|---:|---:|
| 時間（秒） | 18.062 | 17.320 |
| Python traced peak（bytes） | 295471190 | 290923059 |
| build後process peak RSS（KiB） | 913048 | 955024 |

Python traced peakの差は約4.55MB。m0の保存観測でも数MB差であり、今回Q2でも大幅な削減は示していない。
従来経路は2 request/2 controlled-tail回路をまとめて保持し、streamはrequest/tailをO(1)個ずつ扱う。
ただし今回は固定8 eventの入力tuple自体を両経路で保持しており、最終flat回路も残る。

| Compile＋価格計測（秒） | X | Y |
|---|---:|---:|
| 従来native | .755 | .736 |
| streaming | .725 | .731 |

このtimerはinput fingerprint計算後から、既存`transpile_and_measure_cost`が返るまで。
compiled fingerprint/DAG生成・wrapper生成の時間は外にある。全体時間や過去の異なるtimerと速度比を取らない。
各compile時のprocess peak RSSは955,024 KiBで、import/build履歴を含むhigh-water値。
現在RSSの前後値もJSONに保存し、compile単独の増分peakはnullとしている。
実行前の資源観測は空きRAM約471GiB、実行JSONのloadは約10.86/18.43/25.80、利用可能CPU64。
独自のCPU/RAM/worker cap、共有環境変更、他process操作なし。不適格時に後続callを止めるため順次実行した。

## 長roundへの残条件・推奨する次段階

Q2は境界を含む指定trajectoryで資格を確認できたが、m17のRAM/時間/価格は**未確定/null**。
固定構造はQ131072、524288 RTE events、1048576 deterministic block appends/trajectoryのまま。
Q2価格・時間・RSSからm17へ外挿していない。

| 論点 | 今回の状態・残る問題 |
|---|---|
| request/tail逐次化 | 実H4 Q2で同じgate列を確認。中間保持削減と最終回路の保持は別問題 |
| flat回路・wrapper copy | 依然として全長を保持。wrapper生成時のcopyも必要 |
| fingerprint | 既存policyは全instruction・再帰的gate定義のpayloadとJSONを作る。build peakにはこの処理も含まれる |
| compiler DAG・最適化 | whole回路のメモリが残る。分割価格をwhole価格に代用できる証拠はない |
| canonical event stream | samplerはstreaming可能だが既存MC入口はQ*r tupleを保持。分布/順序を変えない接続が必要 |
| cache・parallel | 同一identityのcompile再利用と独立taskの高速化は可能。1本の巨大DAGのメモリは解消しない |
| MC統計 | 既存online mean/variance/SEを再利用可能。CI、paired covariance/集約SE、rare event、停止規則の固定が残る |

**次の優先候補は、canonical fingerprint生成を分離計測し、同じSHA/policyを保つ省メモリ実装を人工回路で検証すること。**
1,623 native instructionsでもbuild/fingerprintを含むPython peakが約291～295MBだったことと、既存SOURCEの全payload保持構造が根拠。
今回のtimerだけではfingerprint単独の寄与を分離していないので、まず成分別の計測を行う。
その後、flat circuit/wrapper/whole DAGの保持を扱う方式と、boundedなevent/task投入を検討する。
compilerやhash policyを勝手に変えず、fusion/分割価格は独立holdoutなしに採用しない。

**ユーザー**：次の人工技術改善を指示するか判断する。新しい実H4 build/compileへ進む場合は、
SOURCE/input/round/明示trajectory/compiler/回数/STOPを別途固定して明示承認する。
本格MCにはさらにsample count/seed/round範囲、paired X/Y・6指標、SE/CI・精度目標・停止規則・rare-event扱いの固定が必要。
**Codex**：結果・REPORTを専用branchへ公開し、remote SHA/tree/公開内容を照合して**mandatory STOP**。
実H4 m0・m2以降、追加trajectory、sampling/期待価格MC、強PF、信号/branch/shot、総費用、GPT詳細科学レビューは未実行。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e09fd13c5a2357dd001d9cfbe86d4e99c3a5620d
[M0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6eea2f28be1f15b6ff5e2954b9c8fb48f3c5a0f/docs/research/track_a_b2_p1_m0_streaming_h4_20261107/REPORT.md
[RUNNER]: ../../../scripts/run_b2_q2_streaming_h4.py
[TEST]: ../../../tests/test_b2_q2_streaming_h4.py
[RESULT]: ../../../artifacts/track_a_b2_p1_q2_streaming_h4/2026-11-07/run01/result.json
[TRAJECTORY]: ../../../artifacts/track_a_b2_p1_q2_streaming_h4/2026-11-07/run01/trajectory.json
[NATIVE]: ../../../artifacts/track_a_b2_p1_q2_streaming_h4/2026-11-07/run01/native_and_wrappers.json
[RAW]: ../../../artifacts/track_a_b2_p1_q2_streaming_h4/2026-11-07/run01/
[SUMMARY]: ../../../artifacts/track_a_b2_p1_q2_streaming_h4/2026-11-07/summary.json

# Codex C：B2 P1 m0実H4 streaming限定検証

2026-11-07 JST / SERVER 2 / Codex主エージェント。

**同一固定trajectoryのm0/Q1でPASS。** 従来nativeとstreamingのgate列・global phase・native fingerprint、
X/Y wrapper fingerprint、保存済みcompiled fingerprintと全6価格指標が一致した。
新規compileは**streaming X/Y各1回、合計2回**。従来回路は再compileしていない。
指定trajectoryの再現検証であり、期待価格・全trajectory・長roundの保証ではない。公開後**mandatory STOP**。

[実行result.json][RESULT] / [native・wrapper比較][NATIVE] / [X結果][X] / [Y結果][Y] / [summary][SUMMARY]。

## SOURCE・入力・trajectory・compiler

- 実行SOURCE **[`db7f0daeffc9345b69fd8695283a20fea2f8e764`][SOURCE]** を実行前に公開し、
  local tree/contentとの一致を確認して固定した。[runner][RUNNER]と比較用人工test/PLANだけを追加。
  既存streaming `c897389…`、従来native/compiler core、sampler、wrapperは不変更。
- 基点は人工技術検証`1bed45be…`。価格referenceは[`2d21c238…`の前回結果][PREVIOUS]。
  前回の元SOURCE `5aa5d061…`／公開snapshot `b443f25…`／bundle対応、旧raw/result/STOPを保全。
- Linear H4/1.00Å/STO-3G、cutoff前H_phys target、保存DF12、generation-prefix LD3、
  partial-S2/r4/K2、delta=.02、**m0/Q1のみ**。元H保存基底stateを仮定し、prep費用を除外。
  signed factors、one-body、核scalar 2.29310124732、tail identity .09261516589219072、spin/JW/LSBを維持。
- DF snapshotは`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`の既存NPZ。
  SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、
  Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`を照合。
  既存`load_snapshot`を使い、保存DF stateは破棄。元H/state/sector/orderは保存input specと既存証拠に一致。
  private state/HDF5/provenanceはhash確認のみで原本不変更・非公開。新state solve/SCF/DF/物理H再構築なし。
- trajectoryは前回と同じindices **[0,4,4,0]**、orders **[0,2,2,0]**。
  既存`fixed_events`で復元し、保存eventの**全field・係数符号・確率・順序・basis情報**を比較した。
  identity **`c3ea48da7dcd082f948b6df9879b3105fc44145c438342396a9777d0b409fbc5`** に一致。
  新sampling・探索・選択なし。極小確率の代表成分を含むため、典型trajectoryとは扱わない。
- Qiskit **1.3.0**、basis **rz/sx/x/cx**、optimization **1**、transpiler seed **7**、
  backend/targetなし・論理all-to-all。compiler hash
  **`a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`**、backend fingerprint **`no_backend`**。
  scopeは`single_hadamard_interrogation_without_state_preparation`。

## preflightの技術修正と保全

初回SOURCE `a0befce2…`は、event比較でnative build前に停止した。[run01のINCONCLUSIVE記録][STOP]を保全。
この試行の新規native build/compileはいずれも0回。
原因は`RTEEvent.to_dict()`（`dataclasses.asdict`）が保持するtupleを、保存JSONの配列と直接比較した実装不具合だった。
JSON表現へそろえたうえで全fieldと元のcanonical digestを厳密比較する修正を行い、
値・符号・確率・順序・identity変更を拒否する人工回帰5件を追加した。数値や科学条件を変えていない。

既存人工21件＋新比較test15件の計36種類がPASS。初回は31件11.16秒、修正後の新15件は4.16秒。
修正SOURCEを固定し、同じ承認範囲で未実施だったbuild/compileをrun02へ保存した。
初回STOPの書換え、異なるtrajectory/条件への救済、compile失敗後のやり直しではない。

## native gate列とcontrolled correctness

同じpreparation/eventsから`B2RPERound.native`と`stream_native`を**各1回**構築した。
system8＋ancilla1、classical bits0、ordinary control、instruction順序/名前/対象qubit/parameter、
global phaseを比較し、再帰的fingerprintも一致。**812 instructions**、phase fittingなし。

native fingerprintは保存結果・従来・streamの三者で
`78dd305c8c9e31b4c97942c76152ea79e0c2beffc1cf0e94c373e747186ab0de`。
X wrapperは815、Yは816 instructions、各9 qubit/1 clbit/measure1回で、両経路と前回fingerprintが一致した。

過去の代表controlled correctness **16/16 PASS**を再利用。
control-0 identity/control-1 evolution、核scalar/tail identity一回計上、signed RTE、forward/reverse順序、
X=Re/Y=Imの既存証拠に、今回の**同じcontrolled gate列・wrapper identity**で対応付けた。
controlled内部の相対位相を事後fitせず、whole wrapperのglobal phaseを混同していない。
新しいH4 operator/statevector/RPE信号simulationはしておらず、全support・全長compiled correctness認証へ拡張しない。

## X/Y full no-prep compiled cost

既存`_compile_pair`→`transpile_and_measure_cost`を使い、streaming X/Yを各1回compile。
全Q stepのcontrolled evolution、basis transition、ancilla操作、Hadamard wrapper、測定を含む。
state preparationなし、cacheによるcompile省略やcompiler/backend変更なし。

| 指標 | 保存X/Y各軸 | 今回X | 今回Y | 差 |
|---|---:|---:|---:|---:|
| CX count | 3262 | 3262 | 3262 | 0 |
| RZ count | 6025 | 6025 | 6025 | 0 |
| CX depth | 2658 | 2658 | 2658 | 0 |
| RZ depth | 2526 | 2526 | 2526 | 0 |
| total depth | 5504 | 5504 | 5504 | 0 |
| circuit size | 11494 | 11494 | 11494 | 0 |

X/Y各measure count1、compiled DAG operation数11494。
指標は既存`RPE_COST_METRICS`と同じで、filtered depthと測定込みwhole depthを区別する。

| Identity | X | Y |
|---|---|---|
| compile前 | `732d8feae6af04838c687b390e6d257e19edb138c3fdc91c204cb0b17add043f` | `c4dea7a8b62faa17a8dc07328787ec56d38b8a765b35d7bb49302b863a975e5f` |
| compile後 | `45706a6e8f066499e2206691041deffda6406ded4d69e51c7e4f9b2a4d87df32` | `5d556741ef8a2d2500ea22f71ee706e62479041e7f4ae4692c72442b908c5cca` |

両軸とも前後fingerprint、compiler/backend、scope・qubit数・prep除外が保存referenceに一致。
同じ価格の別回路を再現成功として扱っていない。

## 時間・メモリの観測と限界

実行全体50.877秒。共通の入力load/preparation後、同一processでreference→streamを各1回計測した。
buildはtracemallocとnative fingerprint生成を含む。2回目は定義cacheが温まり、reference回路も保持されている。

| Build観測 | 従来native | streaming |
|---|---:|---:|
| 時間（秒） | 9.413 | 8.784 |
| Python traced peak（bytes） | 151081307 | 146245003 |
| build後のprocess peak RSS（KiB） | 597796 | 617044 |

約4.84MBのPython traced peak差は観測したが、単回・固定順序・cache条件の影響を含む。
process peak RSSは累積値で、streamだけの増分や公平な独立process比較ではない。
m0では旧経路もrequest/tailが一つなので、長roundでの保持削減をこの差から実証したとは扱わない。

compile＋既存価格計測はX **.334秒**、Y **.316秒**。compile前入力fingerprintは既に渡されており、
compile後fingerprint/DAG生成はこのtimerの外。前回REPORTの約6秒はcompile後fingerprint等を含むため、速度比を取らない。
両軸のprocess peak RSSは617044 KiB、現在RSSの前後値もJSONに保存。
import/build履歴を含むhigh-water値で、compile単独の増分peakは**null**。
独自のCPU/RAM cap、共有設定変更、他processの操作なし。順次照合し、最初の不適格で次のcompileを止める構成。

m17の固定規模はQ131072、524288 RTE events、1048576 deterministic block appends/trajectoryのまま。
request/tail中間listの逐次化と、次の未解決事項を区別する。

- 最終flat circuitとwrapper copy、再帰的fingerprint payloadは回路長に応じたメモリを使う。
- whole compilerのDAG・最適化一時領域をstream adapterだけでは削減できない。
- 既存MC入口はQ*r event tuple、既存parallel MCは全task列を保持する。
- cache hitにもactual circuitのfingerprint作成が必要。並列worker数は長回路の実RSSを測って決める必要がある。

m17のRAM・時間・compiled価格、期待価格・CI・総RPE費用は未取得/null。m0値をQ倍していない。

## 次のQ2検証と終了

**次の段階としてQ2の限定確認が必要**と提案する。m0にはouter-step間の境界がないため、
実H4でのreverse/forward接続、scalar集約、stream request切替をまだ実測していない。
候補範囲は、同じ保存H4入力・同じcompilerで、固定した8 canonical eventsの一つの明示trajectoryを用い、
従来/stream nativeとX/Y wrapperのidentityを先に確認する限定検証。
必要なcompile対象/回数も次の指示で固定する。Q2の保存価格は今回作っておらず、m0価格の2倍をreferenceにしない。
期待価格MC・長roundの資源資格とは別の段階である。

その後はevent/taskの保持上限を科学的入力数とは独立に扱う実装、fingerprintの省メモリ化、whole DAG処理方式、
代表長での実資源計測が残る。fusion/分割価格をwhole価格へ無検証で代用しない。

**ユーザー**：次のQ2限定検証を行うか、対象SOURCE・明示event列・compile対象/回数・STOPを判断し明示承認する。
**Codex**：本結果とREPORTを専用branchへ公開、remote SHA/tree/公開内容を照合して**mandatory STOP**。
今回Q2以降build/compile、追加trajectory、sampling/期待価格MC、強PF、信号/branch/shot、総費用、科学レビューは未実行。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/db7f0daeffc9345b69fd8695283a20fea2f8e764
[PREVIOUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d21c23879aaae063a796fa95c1433fb3d0b5738/docs/research/track_a_b2_p1_compiled_cost_20261107/REPORT.md
[RUNNER]: ../../../scripts/run_b2_m0_streaming_h4.py
[RESULT]: ../../../artifacts/track_a_b2_p1_m0_streaming_h4/2026-11-07/run02/result.json
[NATIVE]: ../../../artifacts/track_a_b2_p1_m0_streaming_h4/2026-11-07/run02/native_and_wrappers.json
[X]: ../../../artifacts/track_a_b2_p1_m0_streaming_h4/2026-11-07/run02/cosine.json
[Y]: ../../../artifacts/track_a_b2_p1_m0_streaming_h4/2026-11-07/run02/sine.json
[STOP]: ../../../artifacts/track_a_b2_p1_m0_streaming_h4/2026-11-07/run01/result.json
[SUMMARY]: ../../../artifacts/track_a_b2_p1_m0_streaming_h4/2026-11-07/summary.json

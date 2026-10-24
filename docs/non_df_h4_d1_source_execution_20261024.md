# H4 D1：SOURCE基準Bによるone-shot完了・9/9計測

2026-10-24 JST。**保存full4の旧RZ価格をすべて再現し、retained/event/emptyを含む9/9計測が完了した。**
PR index0のretained-only/event-onlyはPauliが6,524/3,110、DFが13,483/5,773 RZ。
符号付き非加法性はPauli +18、DF +3,765 RZだった。
この選択代表では、PRのPauli低RZは保持部・event部の両方の低RZと、DFの大きい正の非加法性に対応する。
一意の因果寄与率、256本の母平均、表現一般の優位性やN/Gの結果ではない。

Pauli-PRの旧hash FAIL、原因UNKNOWN、旧native UNKNOWNは維持する。
採用SOURCE基準での対応と旧価格再現が成立しても、失われた旧recursive/compiled programを回収したことにはならない。
新engineと18 taskは全exit0・回収済み。旧740file・封印code42/input71と終了時runtime6,999 SHAを保全した。
**今回の有限D1はmandatory STOP。追加計算へは移行しない。**

[全結果JSON](../artifacts/non_df_h4_d1_source_execution/2026-10-24/result_summary.json)、
[9構成CSV](../artifacts/non_df_h4_d1_source_execution/2026-10-24/metrics_table.csv)、
[非加法性CSV](../artifacts/non_df_h4_d1_source_execution/2026-10-24/interaction_table.csv)、
[終了・保存整合性](../artifacts/non_df_h4_d1_source_execution/2026-10-24/verification.json)。

## 固定条件・承認・SOURCE

Linear H4、隣接1.00 Å、STO-3G、system8 qubit＋ancilla8。
Pauli184非identity項・PR prefix27（Z/ZZのみ）、DF tol-only1e-8/rank10・PR prefix1＋one-body。
S2、T=.8、q8、r4、K4、cosine、元singleton/trajectory_index0。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。
Full Hadamard wrapper、状態準備費用除外、DF raw_concatenation・occurrence内equal-basis reuseを維持する。
同じprefix番号を同じ物理的分割とは扱わない。

ユーザーの「次の作業に進んで大丈夫」は、直前に具体化した基準B採用と保存full4＋隔離5・最大9compileへの承認。
[発言・文脈・許可範囲](../artifacts/non_df_h4_d1_source_execution/2026-10-24/approval.json)を新契約へSHAで結合した。
科学条件、phase所有、符号付きcount式は前段設計から変更していない。
別研究H2/Yoshida4やD2/D3と統合しない。

| 役割 | commit |
|---|---|
| 基点（GitHub取得確認） | `f173b76c7aba56012cbf1d51cb8f27dd062b3bb2` |
| **今回の新科学SOURCE** | **`9843b803878db6c3aee7ab5e7c6bb4818572c944`** |
| 保存full QPYを作ったD1 SOURCE | `48d4d27a05a529e05b11599f5f3c671302826617` |
| 元H4科学SOURCE | `0c329b30a81accb35c212be8defa66fee0dd0f12` |

新[controller](../src/trotterlib/h4_d1_source_control.py)／[runner](../scripts/run_h4_d1_source.py)は
元adapter、Pauli/DF primitive、RTE event、wrapper/compiler、recursive fingerprint、
CallBudget、durable JSON save、native subreaper witnessを再利用する。
旧controllerは旧hash全一致を要求するので、新namespaceの接続を用意し、旧入口を変更しなかった。
量子回路を組む元library SOURCEは変更していない。

新controller人工10＋既存17＝27件PASS。主agent自身の検査で、独立レビューではない。
無grant/未採用拒否、full build0、予算非返還、QPY mismatch証拠保存、旧hashFAIL保持、
Stage A失敗でcompile0、最初の価格不一致で後続0、full4後だけ隔離5、phase/control/order/angle等を確認した。
[人工結果](../artifacts/non_df_h4_d1_source_execution/2026-10-24/artificial)、
[技術確認](../artifacts/non_df_h4_d1_source_execution/2026-10-24/author_check.json)。

SOURCEを先にcommitし、[seal](../artifacts/non_df_h4_d1_source_execution/2026-10-24/seal.json)へ
code42/input71、元runtime6,999、旧証拠map、承認を固定した。
[active contract](../artifacts/non_df_h4_d1_source_execution/2026-10-24/active_contract.json)と
新grant `1429f252f92deb51253f5b8ad3ddac7f48708a61a778d3fc23cec54e65ee8fe5` を一度だけ使用した。
旧grantの未使用callを転用していない。

## Stage A：旧hashとSOURCE対応を別に記録

4つの元full QPYをbyteコピーし、current recursive hashを保存D1 hashと照合した。
全4一致。hashに必要なlazy local definition復元は行うが、full emitter/controlの再構成は0。
前段の12辺・保存命令token/位相更新規則と固定SOURCE/runtimeの対応を再利用し、
現在のQPY自己整合性だけを旧回路対応の根拠にはしない。

| 代表 | A：元のhistorical hash | 現在hashと保存D1 hash | B：採用SOURCE対応 | 旧RZ再現 |
|---|---|---|---|---|
| Pauli-PF singleton | PASS | PASS | PASS（採用基準の範囲） | PASS |
| Pauli-PR index0 | **FAILを維持** | PASS | PASS（採用基準の範囲） | PASS |
| DF-PF singleton | PASS | PASS | PASS（採用基準の範囲） | PASS |
| DF-PR index0 | PASS | PASS | PASS（採用基準の範囲） | PASS |

Pauli-PR旧hash `8a8adeded83edfddead525494906ccbd2a3796b11ed038f1d53f113b61468986`、
保存D1/今回hash `b677002ac37cd37a3f8a0ec84320963f7e44d3dcd29c271c369b1e6ab49fcd21`。
名前だけの差という認定、名前除去hash、counter操作、名前総当たりは行っていない。
旧payload欠測と排他的原因UNKNOWNは[前段調査](non_df_h4_d1_reproducibility_design_20261024.md)のまま。
同RZ数もunitary/program同一性やcompiler名非依存性の証明にはしない。

[stage_A.json](../artifacts/non_df_h4_d1_source_execution/2026-10-24/run01/stage_A.json)がBの受理statusを記録する。
各input.jsonの`input_fingerprint_matches`は旧hashの判定を残す。
`current_D1_fingerprint_matches`は今回の自己整合性で、親の採用SOURCE gateとは別のsnapshotである。
input.jsonの親gate待ちの文字列を、完了後のstage_Aの不合格と読み替えない。

## Stage B/C：保存した9構成の実測

full4を元順序で各一度compileし、旧RZ価格とすべてexact一致した後だけ残り5へ進んだ。
以下は各PRの元index0の値で、PR256本平均ではない。

| 回路 | RZ | CX | total depth | 旧full RZ一致 |
|---|---:|---:|---:|---|
| Pauli-PF full | 255500 | 164096 | 358563 | PASS |
| Pauli-PR full | 9652 | 6872 | 12973 | PASS |
| DF-PF full | 135355 | 78628 | 134134 | PASS |
| DF-PR full | 23021 | 11134 | 18306 | PASS |
| Pauli-PR retained-only | 6524 | 4896 | 8803 | 対象外 |
| Pauli-PR event-only | 3110 | 1976 | 4200 | 対象外 |
| DF-PR retained-only | 13483 | 8100 | 13990 | 対象外 |
| DF-PR event-only | 5773 | 1932 | 2825 | 対象外 |
| Empty wrapper | **0** | **0** | 1 | 対象外 |

Emptyの0は実測値で、欠測nullの補完ではない。
input/output QPY、recursive fingerprint、pre/post gate histogram、RZ/CX depth、
sidecarのrole/basis/controlタグ・DF basis/reuse counterを[variants](../artifacts/non_df_h4_d1_source_execution/2026-10-24/run01/variants)へ保存した。
Pauliのbefore-control histogramと、DFのvirtual/central-only descriptorの区別も維持する。
sidecarをcompiler-visibleなbarrier/labelへ変更せず、nested basis/diagonal/controlの費用を重複加算しない。

元のcount式は `isolated_prediction = C_R + C_E - C_0`、
`Delta_interaction = C_F - C_R - C_E + C_0`。
符号を保持し、clipping・fit・寄与率化を行わない。

| 表現・指標 | full | retained-only | event-only | empty | isolated prediction | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Pauli RZ | 9652 | 6524 | 3110 | 0 | 9634 | **+18** |
| DF RZ | 23021 | 13483 | 5773 | 0 | 19256 | **+3765** |
| Pauli CX | 6872 | 4896 | 1976 | 0 | 6872 | **0** |
| DF CX | 11134 | 8100 | 1932 | 0 | 10032 | **+1102** |

このDeltaはfull構築・control・compilerの文脈を含む操作的な非加法性。
単独のcompiler pass、basis、境界、位相のどれかだけへ帰属させる値ではない。
Depthは各回路の値を示し、加算的分解はしていない。
元SOURCEのcount式の保存出力を引用し、別の科学再集計を追加していない。

## D1から答えられたことと残る範囲

| 問い | 今回の回答と限界 |
|---|---|
| Q1 保持部のRZ差 | Pauli6524、DF13483。Pauli保持27がZ/ZZというSOURCE構造と整合する。ただし保持内容・分割・controlも異なり、Z/ZZだけの因果効果ではない |
| Q2 event列の費用差 | Pauli3110、DF5773 RZ。CXはPauli1976、DF1932で逆方向なので、event部の全指標をPauliが下回るとはいえない。index0の2列の比較に限る |
| Q3 DF basisの役割 | 保持/event両方にbasis変換が存在するSOURCE証拠とsidecarを保持。basis単独のcompiled価格・一意の寄与率は未識別 |
| Q4 非加法性 | Pauli RZ+18/CX0、DF RZ+3765/CX+1102。隔離費用の単純加算ではDF fullを再現しない |
| Q5 逆転の候補 | PF fullではDFが低RZ、PRでは保持・event双方のRZと正の非加法性の違いがPauli低RZに対応。sourceの保持内容/event粒度/control連動を支持する選択代表の証拠 |
| Q6 残る交絡 | generic whole-body対native central-only control、異なる保持集合/law/angle、basisとdiagonalとcompilerの結合。旧payload、256本全体・母平均への外挿も未解決 |
| Q7 次の診断 | Pauli control方式の寄与を知るならD2の要否を次に判断できる。今回D2/D3は未承認・未実行 |

retainedはconstant/extracted identity/deterministic primitive phaseを所有し、
eventはorder/negative-product/identity-rotation phaseを所有する。
fullは元のfloat更新順序・raw concatenationを保ち、emptyはH/H/measureのみ。
隔離回路は新しい有効な近似simulationや精度保証を主張するものではない。
全RPE総費用、無条件PR優位性、N/G更新、正式certificate受理は行っていない。

## 終了監査・保全・公開

| call | 上限 | 消費 |
|---|---:|---:|
| full QPY validation | 4 | 4 |
| 新規full input build | 0 | **0** |
| retained/event/empty build | 5 | 5 |
| 費用top-level transpile | 9 | 9 |

[ledger](../artifacts/non_df_h4_d1_source_execution/2026-10-24/run01/call_ledger.json)、
[CONSUMED](../artifacts/non_df_h4_d1_source_execution/2026-10-24/run01/CONSUMED.json)、
[最終STOP](../artifacts/non_df_h4_d1_source_execution/2026-10-24/run01/STOP.json)。
全callを起動前に非返還予約し、retry/resume・旧grant再使用は0。
Generic control内部のunroll等と、価格計測用top-level compile9回を区別する。

engine native exit0、直接18 taskも全exit0・native wait/reap済み。
[engine native receipt](../artifacts/non_df_h4_d1_source_execution/2026-10-24/run01/native/worker_native_exit.json)、
[task receipts](../artifacts/non_df_h4_d1_source_execution/2026-10-24/run01/native_tasks)。
新runは終了時seal検証PASS。旧740fileとrun出力177fileのSHA inventoryを照合し、
全9 metricsのhistogram/count/profile/input hash、full4のbyte再利用、ledger/claim/receiptも整合した。
新native証拠で元H4のnative UNKNOWNを補完していない。

CPU affinity64を観測し、独立4 validationと後半5対象を並列実行した。
full4だけは最初の価格不一致で後続を止める契約のため逐次とした。
CPU/RAM/workerの人工的な低い上限、共有設定・他ユーザー・他worktreeへの変更は0。
旧N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCE、元native UNKNOWNを保持する。
SCF/FCI/分子DF/state/signal・sampling/RNG replay・N/G・D2/D3・正式certificate受理は0。

専用branch：`codex/non-df-h4-d1-source-execution-20261024`。
[commit/pushとremote状態](../artifacts/non_df_h4_d1_source_execution/2026-10-24/publication_status.json)。
実行SOURCE9843b8038からcodeを変更せず、結果・資料・終了監査を別commitで保存する。
新SOURCEのHTTPS pushは認証不足で失敗した。最終公開状態は上の公開記録を正本とする。

**mandatory STOP。** 次の担当は**ユーザー**：このD1結果を採用し、control方式の寄与を分けるD2を準備するか判断する。
**Codex**：明示指示後だけD2の固定条件・最小範囲・非実行契約を用意する。今回のgrantで追加実行しない。

# Track A AX-2B：H4-P読込gate v2・保存receipt再監査

2026-10-10 JST。利用者の「作業を進めて」を、直前に提示した**読込gateの保全修正・合成検査・保存receipt再監査**へ適用する。
[H4-P原実行](track_a_ax2b_h4_native_receipt_execution_v1.md)は親STOP・worker COMPLETEのまま保持する。
元のnative準備・signal・sampling・wrapper build・compileを再実行せず、科学比較・PF/control・target/state・metricを変更しない。

## 新しい経路と固定上限

[v2 gate](../../src/trottertracks/resource_applicability/ax2b_h4_saved_receipt_gate_v2.py)と
[保存専用runner](../../scripts/resource_applicability/audit_track_a_ax2b_h4_native_receipt_v2.py)を追加する。
凍結v1 gate/runner・原terminal・原freezeは編集しない。新runnerは`--execute`/`--worker`を持たず、approved grantも生成しない。
stdlibだけで保存JSON、Git blob、file hash、input header/Unicode metadataを確認し、数値ライブラリのimportを入口で禁止する。
NPZ ndarray payloadのdecode、native準備、作用、回路構築はない。

| gate | v2の動作 |
|---|---|
| 原16MiB aggregate budget | 固定16 file全体をstatで先に検査。超過・欠落・余分なfile・directory・symlinkで読込前に拒否 |
| JSON読込 | 各fileを**残りaggregate budget**までread。16MiBを一fileずつ独立に使える設計にしない |
| 小record | terminal/grant/claim/bindingは引き続き8KiB以内 |
| data | actual bytes、file identity/size/mtimeを照合。NaN/Infinity/overflow、duplicate key、非object・欠損JSONを拒否 |
| integrity | 原publication commitのfile bytes/hash、receipt/cell/spec digest、全primitive-time集合、basis metadata、元の構造式を照合 |
| source | 実行時commitのtreeから旧closureを復元しexact local/Git bytesを検査。新audit sourceを別commit/closureとして固定 |
| 完了flags | 原親STOP、workerの1 load/8 preparationと禁止call0、N/G=null・u未認定・UNDETERMINED・mandatory STOPを要求 |

4MiBを超えるB3 fileは、既存の16MiB output budget内で扱える。原runのwall/AS/output上限を増やしたことではない。
新module追加により現在のlibrary closureは拡張するが、実行時sourceを現在のsourceへ読み替えない。
旧closure170件は実行時commit `7877131e764f5ce8b296cbbfa9ff3e859d04a8f2`のtreeと照合する。
現在のworktreeに旧fileと違うbytesがあれば拒否する。future moduleを許すために旧fileのhash gateを弱める設計ではない。
監査source commit・source hashesは新audit receiptで別fieldに保存する。

## 合成検査

[専用tests](../../tests/tracks/resource_applicability/test_ax2b_h4_saved_receipt_gate_v2.py)は36件。
[JUnit](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_reaudit/2026-10-10/synthetic_tests_v2.junit.xml)へlocal engineering evidenceを保存する。
実分子NPZとnumerical backendを使わず、paddingで4MiB超のJSONを作り、cap前gate・aggregate超過・不正/欠落・bytes/digest・終了flag・historical sourceを検査する。
v1の48 testsや過去の科学検証は再実行しない。CI、独立分子検証、精度認定の証拠ではない。

## 保存再監査の来歴とscope

正本はresult保存commit `1173dd3342e88239458bcce17ae6a4047f8f1fef`。
linear H4 1.00 Å、STO-3G、legacy DF rank12、generation-prefix L_D=0/6/12、8 qubits、Nα=Nβ=2・sector36、T=0.8、q1/q4・δ0.8/0.2の登録8 cell。
B2/B3はR8・r2・K2/4/6、B1はglobal S2/S4。保存binary64 Hamiltonianと指定保存stateの数学的正規化というtargetに変更なし。
今回読むものは**native準備のmetadata/receipt**で、state/vector作用を再評価しない。

| 一次記録 | 役割 |
|---|---|
| [原親terminal](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/terminal_status.json) | `H4_NATIVE_RECEIPT_STOP` / `TERMINAL_INVALID:JSON_INPUT_SIZE`。書き換え不可 |
| [原native receipt](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/native_receipt.json) | 8 cellの構造上界・coverage。SHA-256 `1a399c93f0aef35994ca26c4f62f61595fd020bda0153fdf6bf1455082a654d9` |
| [原file registry](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/execution_inventory_v1.json) | 原保存file hashの正本。新metadataで上書きしない |
| [新再監査receipt](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_reaudit/2026-10-10/reaudit_receipt_v2.json) | 原16 fileのbytes、実行時source170件、新audit source、構造式・coverageの対応 |
| [新audit freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_reaudit/2026-10-10/audit_source_freeze_v2.json) | 旧準備sourceを継承し、新gate/runner/testsを追加。原execution freezeを置換しない |
| [新inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_reaudit/2026-10-10/reaudit_inventory_v2.json) | 新source/tests/receipt/docsの発見性とhash対応。科学manifestとは別 |

再監査PASSは**保存receiptの整合性を新read gateで確認したこと**を表す。
原H4-P実行の成功、科学的correctness、precision certificate、総u、shot/total cost/winner、H6 GOを表さない。
179 primitive-ID/time組・537 probeは後続科学検証の予定coverage。新probe作用0、native準備0、分子計算0。
原構造上界は実compile/expanded gate数/物理メモリの上界ではない。
各passで16MiBまでのbounded readを行い、終了時の第二passで原output bytesが不変なことを確認する。

## この段階の終了と次の依存関係

原source/freezes/STOP/結果と既存dirty・未追跡、Track Bを保存し、今回のsource/合成検査/保存監査だけを既存branchへ公開する。
再監査後にmandatory STOP。**H4 science manifestの組込み・source更新・sealはこの作業で行わない**。
receiptはfuture manifest reviewの入力として使用できるが、H4限定科学検証launchには別認可が必要。
`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、N/G=null、accuracy UNDETERMINED、総allowance未認定を維持する。

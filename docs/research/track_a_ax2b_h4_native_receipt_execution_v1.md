# Track A AX-2B：H4-P一回実行・親監査STOP

2026-10-10 JST。直前に提示したH4-P限定scopeに対する利用者の「作業を進めて」を認可として記録し、
[固定準備manifest](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/native_preparation_manifest_v1.json)を一回だけ実行した。
**原実行の最終statusは `H4_NATIVE_RECEIPT_STOP`**。workerはload1・native準備8・receipt保存まで完了したが、親のJSON読込gateでSTOPした。
worker COMPLETEを全実行の成功や科学的GOへ読み替えない。再実行・resume・source修正・science sealは行っていない。

## 来歴・一次記録

| 項目 | 正本 |
|---|---|
| preparation source commit | `7877131e764f5ce8b296cbbfa9ff3e859d04a8f2`。runner closure170件、準備freeze184件のexact bytesを再照合 |
| execution base / branch | `2f72584db87ea9458f2550a43499abd30f8a147f` / `track-a-ax2b-h4-post-review-20261010` |
| manifest file SHA-256 | `a51d5f3f364c7588541bb278c094720a2c7e4039f029937ed618b8a01336bdb1` |
| [実行認可](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/authorization_v1.json) | H4_NATIVE_RECEIPTだけ。user messageと直前scope、manifest、CPU3、exclusive outputを結合。H4科学検証/H6等は対象外 |
| [起動command](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/launch_command_v1.json)・[prelaunch監査](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/prelaunch_audit_v1.json) | 別認可と両file SHA-256を専用runnerへ渡した。output不存在・source/input/env/CPU一致を起動前に確認 |
| [親terminal](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/terminal_status.json) | STOP、`TERMINAL_INVALID:JSON_INPUT_SIZE`、worker exit0。原bytesを保存 |
| [worker terminal](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/worker_terminal.json) | COMPLETE、snapshot_loads1 / native_preparation_calls8、completed8、禁止call0。親のSTOPとは別記録 |
| [native receipt](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/native_receipt.json) | SHA-256 `1a399c93f0aef35994ca26c4f62f61595fd020bda0153fdf6bf1455082a654d9`。8 cellのdigestと構造上界を保存 |
| [静的保存監査](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/saved_evidence_audit_v1.json)・[監査script](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/audit_saved_receipts_v1.py) | stdlib-onlyで保存JSONを照合。原STOPの修正・再実行・新数値検証ではない |
| [保全監査](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/preservation_audit_v1.json)・[実行inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/execution_inventory_v1.json) | 旧科学manifestとは別。source/旧結果/既存dirtyとTrack Bを保全し、今回の証拠だけを公開 |

保存監査helperの初回はraw cellの`order`をcanonical cellの`formula`と取り違えて出力前にKeyErrorとなった。
[開発記録](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_execution/2026-10-10/saved_auditor_development_record_v1.json)に初回source hashと原因を残し、helperのみを修正した。
H4-P runner/source/結果は変更せず、分子処理の再実行はしていない。

## 実行scopeと観測

入力はlinear H4 1.00 Å、STO-3G、legacy DF rank12、generation-prefix L_D=0/6/12、8 qubits、Nα=Nβ=2・sector36、T=0.8。
q1/q4、δ=0.8/0.2、global S2/S4およびB0/B2/B3の登録8 cell。B2/B3はR8、r2、K2/4/6。
targetは保存binary64 Hamiltonianと保存vectorを数学的に正規化した指定stateで変更なし。
**今回はnative準備と上界取得だけ**。PF/signal/MP/reference/probe作用を評価していない。

固定上限はCPU3、worker/BLAS1、startup/imports込み900秒、AS8GiB、output16MiB、log64KiB、load1/prepare8、retry/resumeなし。
親のwall観測は2.658937864936888秒、log345 bytes。ASは設定上限で、RSS peakは保存していない。
原output全体は9,821,513 bytes（約9.37 MiB）で16MiB以内。file・log・call上限超過ではない。
`worker_claim.json`と固定sourceのlimit-before-import経路を保存している。独立したCPU/ASのruntime telemetryは追加取得していない。
禁止call0は登録controller/source経路のcounterであり、外部system-call traceによる測定ではない。

| cell / 一次native記録 | deterministic blocks / tail specs | ordinary構造上界 | symmetric-directional構造上界 |
|---|---:|---:|---:|
| [B1 S2 q1](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/H4_B1_S2_q1_native.json) | 13 / 0 | 2,563 | 4,102 |
| [B1 S2 q4](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/H4_B1_S2_q4_native.json) | 13 / 0 | 10,237 | 16,393 |
| [B0 q4](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/H4_B0_q4_native.json) | 7 / 0 | 5,389 | 8,497 |
| [B2 K2](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/H4_B2_K2_native.json) | 7 / 216 | 7,161 | 10,269 |
| [B2 K4](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/H4_B2_K4_native.json) | 7 / 216 | 8,345 | 11,453 |
| [B3 K6](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/H4_B3_K6_native.json) | 1 / 432 | 4,665 | 4,725 |
| [B1 S4 q1](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/H4_B1_S4_q1_native.json) | 13 / 0 | 7,679 | 12,296 |
| [B1 S4 q4](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1/H4_B1_S4_q4_native.json) | 13 / 0 | 30,701 | 49,169 |

表は既存`actual_bounds`・block/tail構造式が生成した**untranspiled instruction構造上界**で、実compile/expanded gate数/量子資源測定や方式の優劣ではない。
最大49,169は登録1,000,000 instructions上限以内。179 primitive-ID/time組・537 probeは後続H4検証の予定coverageで、今回はprobe0。
保存metadataに対する既存構造式の計数と、全time集合・basis metadata同一性・component spec digest・cell digestは補助監査で一致した。
分子回路の正しさ・作用精度・独立numerical reference・総uを検証したわけではない。

## STOP原因と後続依存関係

親の凍結`bounded_json`は各cell JSONをdefault **4 MiB（4,194,304 bytes）**以内に制限している。
B3 K6の完全component spec記録は **4,443,419 bytes**で、この読込gateだけを超えた。
writerの16MiB aggregate budgetとの整合が準備時の合成testsでは未検出だった。準備完了をこの容量条件の分子検証と見なしてはいけない。
親はread gateで例外を捕捉してSTOPしたため、terminal内のworker_terminalはnullのまま。worker terminalは別fileに存在する。

補助監査は原16MiB budget内の**保存済みfileだけ**を読み、全8 cell/receipt/source/input/envを静的照合した。
補助監査の読込許容量を凍結launcherの実行時上限が変更されたことへ読み替えない。原STOP・grant・結果のbytesは全て保持している。
今回の一回grantは消費済み。output再使用・retry/resume・cap自動増量はしない。

次の候補は、**読込gateをwriter budgetと整合させる保全的source修正・合成検査・保存receiptの再監査**である。
この修正はPF/control/target/metric変更ではなく、GPT独立レビュー§21の通常schema修正範囲に当たる。今回は着手していない。
H4-Pを再計算せずに保存receiptを利用できるかは、その修正後の監査で確認する。旧terminalを成功へ書き換えることはしない。
その後にscience manifestへのreceipt組込み・source binding更新・sealと、**別のH4限定科学検証認可**が必要。

H4 science manifestは未seal・未認可、`H4_LIMITED_NOT_AUTHORIZED`。
N/G=null、accuracy UNDETERMINED、総数値allowance未認定、`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
sampling、wrapper build、transpile/compile、solver、Hamiltonian/state生成、H6/H8/GPU実行は行っていない。

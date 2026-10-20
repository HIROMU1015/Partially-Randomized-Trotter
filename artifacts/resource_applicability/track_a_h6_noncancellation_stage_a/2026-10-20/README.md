# H6 noncancellation段階A：保存値再解析・終了監査

基点6cfc4579、SOURCE85e25030。適格34/不適格86/未確定0、closure12/12 PASS。

- [報告](../../../../docs/research/track_a_h6_noncancellation_stage_a_20261020.md)
- [全120 endpoint表](endpoints120.md)、`endpoints.json/csv`、`eligibility12.csv`、`components.json`。
- `sealed_contract.json` / `grant.json` / `user_approval_receipt.txt`：今回だけの承認とSOURCE/入力/runtime/call予算。
- `artificial_tests.json`：人工14＋既存shot-only9、全23PASS。実保存値の事前数値解析0。
- `one_shot.marker` / `launch.marker` / `ledger.json` / `progress.jsonl`：一回の11構成＋1copy、12closure/envelope、120会計、12＋120数値監査。
- `numeric_validation.json`：one-shot子内の数値整合性132記録。独立担当のreviewではない。
- `numeric_completion.json`：数値完了時点のsnapshot。native audit pendingの表記はその時点の状態で、後続completion_audit/STOPで終了確認済み。
- `native_started.json` / `native_exit_evidence.json` / `native_stdout.txt` / `native_stderr.txt`：exit0、同じ親wait4/reaped、残存0。
- `grant_consumption_receipt.json`：消費済み、追加実行なし。
- `completion_audit.json`：120行・call予算・SHA・旧証拠・native終了PASS。primary authorによる検査。
- `stage_B_handoff.json`：別承認が必要な後続事項。費用・選択・比・順位計算0。
- `packet_manifest.json` / `publication_scope.json`：新しい成果物のSHAと未承認公開範囲。
- `STOP.json`：mandatory STOP、stageB未承認、review/freeze未開始。

新規科学SOURCEはcommit済みのまま保全。旧raw/SOURCE/grant/ledger/marker/STOP・70件・論文主表/Figure2を変更しない。結果の再生成・one-shot retryは行わない。
post-runの構造照合入口は`python -B scripts/check_h6_noncancellation_stage_a.py`（sealed契約の元Pythonを使用）。数値解析をやり直さず保存receiptを照合する。

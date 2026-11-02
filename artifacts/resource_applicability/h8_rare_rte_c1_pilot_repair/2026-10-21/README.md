# H8 C-1 repaired pilot evidence

2026-10-21。両条件43/43本・計86 trajectory/172 full-wrapper compileを完了、数値照合・終了監査を保存しmandatory STOP。S2_SINGLEのみ43本×2、multi/S4未知・S0旧P48は別anchor。mandatory STOP、push未実行。

- [結果・条件・技術修正・限界](../../../../docs/research/h8_rare_rte_c1_pilot_repair_20261021.md)
- [SOURCEとSHA](SOURCE_manifest.json)・[実行前preflight](preflight.json)・[技術修正証拠](technical_repair_evidence.json)・[runtime照合](runtime_repair.json)
- [coarse契約](contract_H8_0.05.json)・[tight契約](contract_H8_0.0005.json)・[coarse新承認receipt](execution_approval_H8_0.05.json)・[tight新承認receipt](execution_approval_H8_0.0005.json)・[seed除外](seed_receipt.json)
- [条件別結果と予定/予約/実使用](condition_results.json)・[新grant/marker/非返還会計](grant_ledger_audit.json)
- [coarse43本全費用JSON](trajectory_costs_H8_0.05.json)・[coarse両軸CSV](paired_costs_H8_0.05.csv)・[coarse統計](pilot_summary_H8_0.05.json)・[coarse数値検査](independent_numeric_audit_H8_0.05.json)
- [tight43本全費用JSON](trajectory_costs_H8_0.0005.json)・[tight両軸CSV](paired_costs_H8_0.0005.csv)・[tight統計](pilot_summary_H8_0.0005.json)・[tight数値検査](independent_numeric_audit_H8_0.0005.json)
- [coarse law](law_gate_H8_0.05.json)・[tight law](law_gate_H8_0.0005.json)・[coarse native回収](native_receipts_H8_0.05.json)・[tight native回収](native_receipts_H8_0.0005.json)
- [coarse負荷](resource_H8_0.05.json)・[tight負荷](resource_H8_0.0005.json)・[private raw保全SHA](private_raw_receipt.json)
- [人工検証](synthetic_validation.json)・[新12 tests](driver_tests.txt)・[既存55 tests](c05_tests.txt)
- [終了監査](completion_audit.json)・[最終STOP](STOP.json)・[全成果物SHA](SHA256_manifest.json)

source/contract/receiptは新規、旧SOURCE/STOP/grant/raw不変。数値law整合はengineering gate、mean/variance/SEは条件付き標本推定。all-support bound・母平均certificate・PR/PF順位の追加判断を含まない。全event/seed/内部pathと資源全観測はprivate保全。

次担当ユーザー：後続科学範囲と公開範囲の判断。Codexは停止し、production・追加層・review/freezeへ自動進行しない。

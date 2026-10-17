# H8 conditional s2 run01 STOP bundle

[結果・原因・限定修正・次の判断](../../../../docs/research/track_a_h8_conditional_s2_pilot_STOP_20261018.md)。実行SOURCE 3db51df7…、修正SOURCE 7522d74b…（科学未実行）。実行試行1、sampling/compile0、q8未起動、driver/8 worker全回収。消費one-shotは保持しretry0。

- `result_STOP.json`：取得数0・全統計null・次stage未認可。
- `failed_terminal.json` / `failed_native_exit.json` / `failed_final_gate.json`：原本byte copy。
- `failed_parallel_receipt_sanitized.json` / `worker_STOPs.json`：全workerのnative終了・first STOPと副次error。
- `execution_binding.json`：実行時SOURCE/plan/config/grant digestのみ。有効grant原本は含めない。
- `fresh_gate_sanitized.json` / `post_STOP_preservation.json`：fresh gateと停止後の不変性・process不在。
- `failed_SOURCE_closure.json` / `fixed_SOURCE_closure.json` / `source_change_audit.json`：旧新SOURCEと全8 blob/hash。
- `receipt_schema_provenance.json`：元receipt形式とpreparation hash内partition binding。
- `nonrefundable_reservations.json` / `ledger_audit.json`：実到達0と予約保持を区別。
- `metadata_regression20.txt` / `metadata_test_receipt.json` / `invalid_grant_rejection.txt`：限定人工回帰と拒否gate。
- `prospective_plan_UNAPPROVED.json` / `prospective_grant_INVALID.json` / `prospective_config_UNSEALED.json`：将来用draft。有効な実行許可ではない。
- `private_evidence_hashes.json` / `private_run_file_inventory.json`：private原本のbytes/SHAのみ。NPZ・使用済みgrant/lock・raw cacheは非公開。
- `publication_inventory.json`：このbundleの軽量公開fileのbytes/SHA（inventory自身は除く）。

旧準備・人工22検査は[先行bundle](../../track_a_h8_conditional_s2_pilot_readiness/2026-10-17/README.md)で保持する。主担当自身の検査であり、独立agent reviewは行っていない。mandatory STOP、新しい実行承認が必要。

# H4 DF-native L one-shot局所診断

2026-10-23。SOURCE eb195bf2d、固定128 bit、148 forward/148 inverse/11 basis/324 supportを一回実行し局所範囲でPASS。C/L結合BLOCKED、旧compiler MISSING_EVIDENCE、正式受理/N_G0、mandatory STOP。

- [結果・限界・終了監査](../../../docs/non_df_h4_local_diagnostic_20261023.md)
- [結果要約](result_summary.json)、[検証](verification.json)、[契約](L_execution_contract.json)、[ユーザー承認](USER_AUTHORIZATION.md)
- [生のone-shot結果・grant・call ledger・native・STOP](run01/)
- [局所296行](audit02/local_gate_bounds.csv)、[基底11行](audit02/basis_bounds.csv)、[support324行](audit02/component_support.csv)、[一様伝播](audit02/uniform_basis_transfer.csv)
- [C/L重複ledger](coverage_ledger.json)、[人工回帰148PASS](regression_tests.json)、[保全2486](preservation_after.json)
- [初回auditorのmtime検査失敗](audit01/failure.json)。科学再実行なし、source/raw/grant不変。
- [新run649 file SHA](raw_result_inventory.json)、[公開記録](publication_status.json)、[引継ぎ](CONTINUATION.md)

局所norm区間は[0, exact upper]。表示decimalは上向き丸め、exact rationalを正本とする。LとC totalの加算・相殺はしない。現authorityはrun01/STOPで失効済み。旧H4のnative証拠不足へ新L receiptを代用しない。

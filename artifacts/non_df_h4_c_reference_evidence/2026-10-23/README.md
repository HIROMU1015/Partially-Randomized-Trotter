# H4 C修正版：新grantによる固定40軸実行

[実行・結果・残る制約](../../../docs/non_df_h4_c_reference_evidence_20261023.md)。SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`、branch `codex/non-df-h4-c-reference-evidence-20261023`。18:53 JSTに新grantを一回消費し起動した。

- `launch01/approval.json` / `active_contract.json` / `contract_delta.json`：利用者の承認・新契約・科学条件不変。
- `launch01/preflight.json` / `preflight_artificial_reference/`：source/runtime/inputのSHAと人工14件PASS。
- `launch01/dispatcher.json` / `DISPATCH_REQUEST.json`：一回だけの起動・outer native witness。
- `run01/`：今回の新grant・CONSUMED・claim・ledger・model・tier/result・native receipt・STOP。
- `launch01/observation*.json`：既知runと所有processだけの読取観測。予約数は完成call数ではない。
- `saved_result_audit.py`：終了後の保存構造・SHA・native監査のみ。科学値を再計算しない。

最終結果はまだ確定していない。Pauli20軸TARGET_METを保存、DFを実行中。現在の状態はrun01とwitnessのSTOPを確認する。旧結果・旧N/G未確定・元native欠落は保全する。

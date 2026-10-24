# H4 Pauli central-control Arb包含評価の準備

[技術報告書](../../../docs/non_df_h4_central_arb_preparation_20261024.md)。
新bridge人工10＋QPY人工6＋既存sine人工9 PASS。3qubit・49,000gate人工列は全精度で既知値を包含。
実H4のArb評価・状態作用・新科学grant・compile/sampling・N/Gは0。

- `PF_cosine_program_NO_EXECUTION.json`：保存QPYの構造decodeのみ、exact binary64 native列。
- `PF_C_target_interval_QUOTED.json`、`state_identity_QUOTED.json`：R0保存値の引用。
- `pilot_contract_DRAFT_DISABLED.json`：PF cosine 1対象、64→96→128bit最大3状態作用の**無効契約**。
- `artificial_tests.json`、各log、`artificial_long_sequence.json`：人工検証のみ。
- `reuse_matrix.json`：既存資産の再利用・新実装の理由。
- `inputs.json`、`preservation_before.json`、`prior_manifest_rows.json`：旧証拠の保全参照。
- `verify_preparation.py`：metadata/SHA検証のみ、数値runnerをimportしない。
- `publication_status.json`：commit/push/remoteの時点記録。
- `STOP.json`：今回の技術準備のみmandatory STOP。元のnative証拠を代替しない。

実H4 pilotにはユーザーの別途明示承認とproduction SOURCE/controller/grantの新規固定が必要。
局所ρ・PR全law・tail・C/L・旧N/Gの未解決状態を保持する。

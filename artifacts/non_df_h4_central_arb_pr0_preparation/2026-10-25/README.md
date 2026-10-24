# PR0 cosine Arb包含評価の準備

[技術報告書](../../../docs/non_df_h4_central_arb_pr0_preparation_20261025.md)。
個別C-targetとcompiled bridgeの接続を実装し、新13＋再利用25＝38人工テストPASS。
実H4は4保存QPY・32order0 eventの静的検査まで。Arb状態作用/新科学grant/compile/sampling/N_Gは0。

- `PR0_selected_C_plan_NO_EXECUTION.json`：Cのexact operandsと選択event順序。信号区間ではない。
- `PR0_compiled_program_NO_EXECUTION.json`：保存native QPYのliteral descriptor。
- `pilot_contract_DRAFT_DISABLED.json`：PR0 cosine、64→96→128bit・各C-target1＋compiled1・最大6状態作用の**無効契約**。
- `arb_artificial_tests.json`、`qpy_artificial_tests.json`と各log：人工検証のみ。
- `structural_inspection.json`：4 QPYとSOURCE/control/phase/state/join照合。数値区間はnull。
- `reuse_and_source_checks.json`：既存C/R0/RTE/compiled bridgeの再利用と版照合。
- `preservation_before.json`、`inputs.json`、`prior_manifest_rows.json`：旧証拠481・実入力11・旧manifest187行。
- `verification.json`：metadata/Fraction/SHAだけの検証。Arb/QPY/科学runnerをimportしない。
- `publication_status.json`：技術commit・公開状態。
- `STOP.json`：今回の技術準備だけmandatory STOP。

実H4評価には別途ユーザー明示承認、新production SOURCE/入力/runtime/契約/未使用grantと終了監査が必要。
PR1/sine/R2/N_G/正式受理には拡張しない。旧PF結果・未解決状態を保全する。

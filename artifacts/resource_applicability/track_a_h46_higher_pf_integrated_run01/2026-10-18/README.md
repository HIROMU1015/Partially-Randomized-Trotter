# H4/H6 integrated high-order PR comparison run01

[結果と独立確認案](../../../../docs/research/track_a_h46_higher_pf_integrated_run01_results_20261018.md)。

- `result.json`：固定kernelの未加工結果。24条件・18適格費用群・6層の点q選択。
- `comparison_table.csv` / `selected_point_comparisons.json`：bias、normalization、shot、費用/SE、点比。
- `selection.json`：結果SHAと選択候補digest。母平均winner認定なし。
- `confirmation_design_UNAPPROVED.json`：未採用MCSE/検出差案。adopted_n=null、C未実行。
- `audit.json` / `native_exit.json` / `postprocess_exit.json` / `fresh_gate.json`：再現・終了・起動証拠。
- `execution_plan_projection.json` / `source_input_seal.json` / `postprocess_registered.py`：private planの公開projectionと起動前seal補助script。
- `grant_consumption_receipt.json`：消費済み証拠のみ。有効grant非公開。
- `artificial_tests.json` / `preflight_technical_notes.json`：作者自身の42人工/mockと科学起動前の修正記録。

新signal/sampling/compile0。独立確認・厳密bias・希少層を含む母平均優位は未確立。mandatory STOP。

# H8 Phase B1 run01 / 2026-10-17

全4 S2層PASS。B0独立holdout両軸0%、B0Iはtraining anchor残差全0%。新compile0。S4と全gridの誤差は未保証、PR総費用は未再会計。

- `frozen_models_v1.json`: 4独立fit、TRAINとHOLDOUTの保存診断、S4未検証の区分。
- `B2_predictions_UNAPPROVED_v1.json`: model推定4行。実測・compile承認ではない。
- `input_anchor_registry_v1.json`: 新B0 8軸＋既存B0I16軸。旧B0未結合軸は含めない。
- `execution_terminal_v1.json` / `phase_terminal_v1.json`: driverのfit完了と、終了後gateまで含むB1完了を区別。
- `resource_and_cleanup_v1.json` / `launcher_native_exit_v1.json` / `self_resource_observations_v1.json`: native exit0、全回収、資源観測。
- `execution_contract_v1.json` / `execution_binding_v1.json` / `fresh_gate_projection_v1.json` / `CPU_fresh_v1.json`: 一回のB1だけを承認したscopeとSOURCE/input/profile。
- `technical_tests_v1.json`: 実データfit前の35人工tests。独立レビュー/CIではない。
- `preservation_v1.json`: 旧681保護files/119worktree不変。元B0取得契約STOPは保持。
- `original_execution_file_hashes_v1.json`: private原本の相対論理名/bytes/SHA。公開byte inventoryとは別。
- `publication_files_v1.json`: 自己参照を除く公開bytes/SHA。

`python3 -B verify_saved_result.py`は保存bytes/metadataだけを照合し、fitやcompileを再実行しない。B2/Cは別承認、mandatory STOP。

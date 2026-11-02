# H8 Phase B2 run01 / 2026-10-17

H8/1Å/STO-3G/DF28/T=.8、S4の固定4軸を直列・一回実行。実compile entry3、保存判定2軸（PASS2）。SOURCE244c45e7…、model8420b9f2…を固定。STOP理由は結果報告と原terminalに保存（MEMORY_PRESSURE）。B0I cosineは中断、sine未起動、未settle予約1を保持し、後続compile・retry/refit0。driver native exit−9、launcher報告exit1（外側native wait欠測）、全owned process回収。旧684保護files/123worktree・B0原53/B1原34 rawを保持。新fit/signal/再会計0、元B0 STOP/V2 NO_GOを維持。C未承認、公開後mandatory STOP。以下は各時点の履歴。

- `axis_measurements_v1.json`: 各軸の実compiled RZと凍結model誤差。総shot費用の再会計なし。
- `execution_contract_reference_v1.json`: 公開済み契約と今回承認の参照。実行可能なgrantではない。
- `source_profile_input_binding_v1.json`: SOURCE・runtime・入力・modelの固定hash。有効grant本体は非公開。
- `execution_terminal_v1.json` / `launch_admission_terminal_v1.json` / `final_gate_v1.json`: 原terminalのpath-redacted projection。成功/STOPを区別。
- `serial_worker_receipts_v1.json` / `native_control_preparations_v1.json`: 固定順序、native exit/reap、full-wrapper/compiler/control証拠。
- `call_ledger_projection_v1.json`: 非返却予約・settlement・実compile entry marker数。STOP時の旧globによるraw報告数とは別に監査。
- `fresh_gates_v1.json`: 固定3窓CPU提案と初回/driver再照合後gate。自動再割当なし。
- `resource_and_cleanup_v1.json` / `resource_observation_summary_v1.json` / `launcher_native_exit_v1.json`: 実native wait、監視、全回収。
- `preservation_receipt_v1.json`: 旧worktree/ファイル・元B0 STOP/B1原本・今回config/grantを保全。
- `original_execution_file_hashes_v1.json`: private原本のSHA inventory。scratch/cache除外、原本内容と公開redactionを区別。
- `publication_files_v1.json`: 公開bytesのSHA inventory（自身を除く）。

`python3 -B verify_saved_result.py` は保存bytesとmetadataだけを確認する。fit、compile、signal、費用再会計を実行しない。Phase C未承認、mandatory STOP。

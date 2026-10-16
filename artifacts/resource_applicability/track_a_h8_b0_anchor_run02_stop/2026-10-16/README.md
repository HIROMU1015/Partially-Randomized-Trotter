# H8 B0 run02：8軸保存後のSTOP

取得契約statusはSTOP。8つのcompile workerはexit0/reapで測定を保存したが、最終worktree確認中にPROCESS_THREADSを検知した。保存値を正常完了・B1受入へ自動昇格させない。

- `driver_STOP_v1.json` / `resource_and_cleanup_STOP_v1.json` / `launcher_admission_STOP_v1.json`: 元STOPを維持。
- `launcher_native_exit_v1.json`: 外側launcherの実wait exit1。driver exit−9と区別。
- `train_measurements.json` / `holdout_measurements.json` / `new_B0_saved_measured_axes_v1.csv`: 保存8軸の実測。fit/残差/総費用を含まない。
- `worker_exit_reap_receipts_v1.json` / `preparation_control_receipts_v1.json`: 全8 workerと4 control/preparation。
- `STOP_assessment_v1.json` / `driver_stderr_projection.txt`: 停止箇所と未記録証拠。違反PID・thread数を推測で補わない。
- `post_STOP_file_integrity_v1.json` / `preservation_receipt_v1.json`: 事後の読取専用照合。最終gateの代替ではない。
- `original_execution_file_hashes_v1.json`: private originalの相対論理名/bytes/SHA。原本は変更しない。
- `publication_files_v1.json`: 公開byte inventory。自己参照を避けて本ファイル自身を除外。

保存metadata/bytesだけの検証：`python3 -B verify_saved_stop.py`。科学環境・実gate・compile・fitは起動しない。新科学計算は承認待ち、mandatory STOP。

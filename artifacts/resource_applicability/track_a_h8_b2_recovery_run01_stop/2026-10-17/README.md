# H8 B2 recovery run01: pre-science CPU gate STOP

初回fresh CPU gateでCPU6 raw82%>5%を検出。科学driver/worker/control/QPY/compile0、CPU再割当・retry0。launcherの実native exit1はdetached witnessのwaitで保存した。

- `stop_summary_v1.json`: 固定対象と科学起動前STOP。
- `launch_admission_terminal_v1.json` / `CPU_initial_v1.json`: 失敗counterと未作成path。
- `CPU_proposal_selected_v1.json`: freshとは別の固定3窓。選択coreだけを転記。
- `native_launcher_exit_v1.json` / `dispatcher_native_exit_v1.json`: 実native wait。dispatcher0とlauncher1を区別。
- `witness_and_observer_start_v1.json`: ownership/session証拠のpath-redacted projection。
- `source_input_binding_v1.json`: 31 source/265 provider/34,273 runtime/16 input、私有grant/configのSHA。
- `preservation_receipt_v1.json`: 旧688保護・125worktree・B2原45files不変。
- `original_execution_file_hashes_v1.json`: private原本のbytes/SHA inventory。
- `publication_files_v1.json`: 公開bytes/SHA inventory（自身を除く）。

`python3 -B verify_saved_stop.py` は保存bytesとmetadataだけを確認する。resource gate・科学計算・compileは起動しない。mandatory STOP。

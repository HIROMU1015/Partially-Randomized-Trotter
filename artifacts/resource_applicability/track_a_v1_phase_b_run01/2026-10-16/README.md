# V1 Phase B run01：全4軸PASS・終了STOP

[結果・範囲・限界](../../../../docs/research/track_a_v1_phase_b_run01_results_20261016.md)。wall3037.384秒、native exit0、all_reaped=true。登録4軸のみ検証し、再fit・追加compile・5精度費用再評価・V2実行は0。

- [4軸全表CSV](axis_validation.csv)、[同JSON・検証判定](validation_complete.json)。
- [driver終了](execution_terminal.json)、[owned launcher終了・回収](launcher_terminal.json)。
- [SOURCE/input/runtime/650旧ファイル保全・終了監査](completion_audit.json)。
- [保存JSON/CSVだけによる再照合](verify_public_results.py)。分子入力・compile・再会計を呼ばない。

以下は起動時点の履歴。

[実行状況・制約](../../../../docs/research/track_a_v1_phase_b_run01_started_20261016.md)、[固定済みモデルと契約](../../track_a_v1_phase_b_readiness/2026-10-16/README.md)、[時刻付き起動snapshot](start_receipt.json)。

このsnapshot時点で最初のS4 compileが実際に開始。全4軸PASSや実行完了を示すものではない。以後は各軸2%以上・資源違反で即STOP、完了時mandatory STOP。5精度再会計とV2への自動移行はない。

# H8保存値統合：成果物入口

科学SOURCE `1e1e1c82a09ab656d9dc6ca71ef0b818e04fae87`。2条件の固定保存値解析を1回完了、mandatory STOP。

[結果・主張境界](../../../../docs/research/h8_rare_rte_saved_integration_results_20261021.md)、[契約](../../../../docs/research/h8_rare_rte_saved_integration_contract_20261021.md)。

|内容|正本|
|---|---|
|固定計算・数値規約|[analysis_definition.json](analysis_definition.json)、[contract.json](contract.json)|
|SOURCE・入力SHA|[SOURCE_manifest.json](SOURCE_manifest.json)、[input_acquisition.json](input_acquisition.json)|
|保存値runner・主経路・別経路|[runner](../../../../scripts/h8_rare_saved_run.py)、[core](../../../../scripts/h8_rare_saved_core.py)、[verifier](../../../../scripts/h8_rare_saved_verify.py)|
|人工test・文書/構造checker|[tests](../../../../tests/test_h8_rare_saved_integration.py)、[document checker](../../../../scripts/document_h8_rare_saved_integration.py)|
|現在の承認・新grant|[execution_approval_receipt.json](execution_approval_receipt.json)、[grant_issue_receipt.json](grant_issue_receipt.json)、[grant_consumption_receipt.json](grant_consumption_receipt.json)|
|人工・実行前gate|[synthetic_validation.json](synthetic_validation.json)、[preflight.json](preflight.json)、[technical_preflight_log.json](technical_preflight_log.json)|
|全数値|[result.json](result.json)|
|既知層4行・条件概要2行|[known_layers.csv](known_layers.csv)、[condition_summary.csv](condition_summary.csv)|
|2層境界10行・SE感度8行|[two_layer_boundary.csv](two_layer_boundary.csv)、[known_error_sensitivity.csv](known_error_sensitivity.csv)|
|別算術検証・非返還会計|[independent_numeric_audit.json](independent_numeric_audit.json)、[calls.json](calls.json)|
|native/保全/構造/最終SHA|[native_exit.json](native_exit.json)、[completion_audit.json](completion_audit.json)、[documentation_audit.json](documentation_audit.json)、[SHA256_manifest.json](SHA256_manifest.json)|
|状態・停止|[status_manifest.json](status_manifest.json)、[STOP.json](STOP.json)|

未知費用/分散はnull。軸切片の0は仮想座標で、unknown zero-fillではない。known SEはiid・独立性モデル下の標本推定で、全層保証ではない。法則差1e-12 gateを期待費用biasの保証へ格上げしない。有限標本・条件付き感度の報告に追加compileは必須でなく、母平均順位は未保証。

非公開grant原本・marker・予約ledger・入力cache・保全locatorは新worktree内のignored private directoryへ保存し、公開候補はSHA/receiptのみ。次担当ユーザー：主張/公開範囲判断。Codexは停止。

`partial_results.json`は実行途中の保存snapshotで、最終判定には`result.json`・`worker_completion.json`・native終了記録を使用する。

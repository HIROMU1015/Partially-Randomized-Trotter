# H8 native B0+I 実装・実行準備の入口

[準備報告・対象範囲・未承認資源案・次の承認](../../../../docs/research/track_a_h8_native_b0i_readiness_20261015.md)。
実入力C/signal/shot、native build/probe/compile、実worker起動は0。これは実行認可ではない。

- [実行契約draft：false / CPU[] / 未seal / command null](execution_contract_draft_v1.json)
- [固定SOURCE closure](source_closure_v1.json)
- [provider265/input7のbyte・保存metadata監査](input_and_provider_audit_v1.json)
- [事前固定のpreparation recipe12件](preparation_recipes_v1.json)
- [人工45件・旧履歴と未承認CLI拒否](artificial_tests_v1.json)
- [read-only host snapshot：将来の起動gateではない](read_only_resource_snapshot_v1.json)
- [主担当による限定整合確認](primary_review_v1.json)
- [bundle fileのbytes/SHA](review_bundle_file_receipts_v1.json)

実NPZ・private input mapping・runtime/ledger/checkpoint・venv実体・credential・内部SSH情報は収録しない。旧資料と証拠は元の場所に保全する。
新SOURCEで科学provider/compilerを勝手に変更せず、実行認可・fresh bindingが揃うまで起動しない。subagents0。

4 workerへの変更は[four_worker_update_v1.json](four_worker_update_v1.json)を参照。2構成を保持し、全正しさ検証後に最大4軸をcompile、最後の1構成は2軸。旧read-only snapshotは旧2 worker案採取時の履歴であり、freshな4 worker起動可否を証明しない。

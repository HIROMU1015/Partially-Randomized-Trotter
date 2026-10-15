# H6 B0+I blueprint fix — evidence index

SOURCE `6b4d3026bf6c30bea3d74e7ccaeb06bda0e9baaf`。単独担当、修正82人工tests PASS、実pilot再起動0。

- [原因・修正・未検証事項・次の条件](../../../../docs/research/track_a_h6_b0i_blueprint_fix_20261015.md)
- [SOURCEに固定した現profile identity／旧hash保持](current_profile_preparation_identity_v1.json)
- [修正契約案（未認可・未seal）](h6_pilot_repaired_contract_draft_v1.json) / [binding状態](binding_status_v1.json)
- [runtime code11＋metadata1 closure](source_closure_v1.json)
- [限定診断計画](diagnostic_plan_v1.json) / [thread1再現](blueprint_reproduction_v1.json)
- [旧thread4比較計画](diagnostic_thread4_plan_v1.json) / [比較結果](blueprint_thread4_reproduction_v1.json)
- [科学処理前に停止した診断bootstrap試行](diagnostic_attempt01_v1.json) / [診断会計](diagnostic_accounting_v1.json)
- [人工計画](artificial_test_plan_v1.json) / [82件PASS](regression_result_v1.json) / [人工log](artificial_tests_v1.txt)
- [同一担当の差分review](author_review_v1.md) / [profile・旧run証跡保全](preservation_receipt_v1.json)
- [公開状態](publication_receipt_v1.json) / [最終整合性](final_consistency_v1.json) / [資料manifest](artifact_manifest_v1.json)

現profileの準備hashへ明示bindingし、runtimeで期待値を再学習しない。旧prepared bit同一性を認定せず、将来のnative correctness/empirical u gateを保つ。
サブエージェント・新grant・本計算再起動・signal/overlay/native build/probe/compile0。限定診断で保存snapshot2回、B0準備6回だけを実行し費用を保持する。

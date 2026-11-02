# H6精度一致資源比較 v2 実行準備とsynthetic検証

入口は[実行準備契約](../../../../docs/research/track_a_h6_matched_accuracy_execution_contract_v2.md)。[sealed manifest](sealed_manifest_v2.json)のsource54986a9と旧v1科学planを使用する。科学runは未認可、新grant未発行、別出力は未使用。

- [準備監査](preparation_audit_v2.json)、[科学条件同一性](science_plan_identity_v2.json)、[保全監査](preservation_audit_v2.json)。
- [176 local tests](synthetic_test_report_v2.json)、[JUnit](synthetic_tests_v2.xml)、[検査履歴](test_history_v2.json)。
- [synthetic実子process receipts](synthetic_process_receipts/)はdummy metadata/sourceだけの検査。実H6結果や実行認可を表さない。
- [host資源](host_inventory_v2.json)、[旧timingと静的性能確認](performance_review_v2.json)。新benchmarkや科学評価ではない。
- [未承認template](authorization_template_v2.json)、[具体的な認可対象](launch_request_v2.json)。templateはapproved_by_user=false。

coordinatorはAS soft2/hard8 GiB、workerはsoft/hard8 GiB。softで元の割当予算を強制し、hardは継承ceilingである。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / mandatory STOP / next_stage_authorized=false。

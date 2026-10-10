# H4 host PSI限定猶予：未承認修正案

[変更の根拠と範囲](../../../../docs/research/track_a_h4_host_pressure_grace_fix_20261010.md)、[条件案](pressure_amendment_draft_v1.json)、[SOURCE69](source_freeze_v1.json)、[14回帰結果](limited_test_result_v1.json)、[事前test予算](limited_test_plan_v1.json)、[run08停止監査](run08_stop_summary_v1.json)、[限定差分review](independent_pressure_grace_delta_review_v1.json)、[commit一覧](commit_inventory_v1.json)。
旧production67byte不変、新inactive module/test2件だけ。host1〜5%は全cgroupPSI0/OOM基準一致/available152.25GiB/fresh5秒なら30秒まで、>=5%や30秒継続・nonroot/OOM/欠測は即STOPする案。
approved/runtime/wiring=false、allowed[]、未seal、commandnull。本計算再起動0、科学array/compile/worker0。従来1%即STOPを適用中。

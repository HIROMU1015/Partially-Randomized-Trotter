# H6 v0.2：主張と証拠、限界、固定方法

2026-10-13 JST。新しいGPT詳細レビューではなく、原稿表示値の技術的整合性と証拠scopeの対応表である。

| 原稿の主張 | 根拠 | 許されない拡張 |
|---|---|---|
| 探索592 group完了 | [探索summary](../../artifacts/resource_applicability/track_a_h6_exploration_summary_v1/2026-10-12/result_summary_v1.json) | 592独立PR trajectoryではない、全精度全候補適格ではない |
| 事前8構成×独立32 | [確認summary](../../artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/result_summary_v1.json) | 探索8とpoolしない、残64の独立確認を主張しない |
| PR約54–63% RZ点削減 | [表示CSV](../../artifacts/resource_applicability/track_a_h6_manuscript_h8_plan_v1/2026-10-13/h6_precision_results.csv) | population upper bound・energy/chemical accuracy・他metricへ同じ削減率を適用しない |
| 事前K4でも効果が残る | [8構成のG/SE](../../artifacts/resource_applicability/track_a_h6_manuscript_h8_plan_v1/2026-10-13/h6_fixed_panel.csv) | postconfirmation選択を完全独立winnerと呼ばない |
| s2補助平均更新+0.00734% | [pilot寄与](../../artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/cost_contribution_analysis_v1.json) | これは最大誤差でない、他候補のrare cost未測定 |
| 複数order2・希少component未解決 | [保存RTE/観測](../../artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13/selected_trajectory_cost_evidence_v1.json) | 発生確率やinstruction guardをcompiled RZ上界にしない |
| H4は別case study | [PM-2](../pr2_pm2_precision_resource_result_validation.md)・[比較案](../research/track_a_h4_h6_common_comparison_plan_v1.md) | 65%とH6比をsize scalingで結ばない |
| H8にはmemory/reference設計が必要 | [H8案](../research/track_a_h8_resource_comparison_plan_v1.md) | 4900sector動作・32GiB完走・8worker速度は未検証 |

uのcertificate、母平均信頼上界、state preparation、最終energy/RPE資源、general asymptotic scalingは未解決。共通Pは仮想感度、旧host compiler完全同一性は未確認。古典wall/RSSと量子資源を別会計とする。

本稿用再生成sourceはartifactのbuild_evidence.py/render_figures.py/verify_evidence.pyであり、科学sourceを変更しない。metadata入力byte SHAは[receipt](../../artifacts/resource_applicability/track_a_h6_manuscript_h8_plan_v1/2026-10-13/input_metadata_receipt.json)、公開identityは[publication receipt](../../artifacts/resource_applicability/track_a_h6_manuscript_h8_plan_v1/2026-10-13/publication_receipt.json)へ記録する。remote取得とlocal scalar検査は実science再実行やCIを意味しない。

# H2/H4/H6/H8：固定P条件・費用・欠測対応表

2026-10-21 JST。保存値の転記のみ。全行linear 1 Å/STO-3G/tol-only1e-8/T=.8/P=0/通常signed/factor1。DF10とlegacy DF12を区別する。`conditions.json/csv`と`literal_extracts.json`が機械可読正本である。

## 固定条件と保存費用

平均は保存G_RZ exact文字列、SEは保存paired SE。式を再評価していない。

|系・DF|ε|PR|q/r/K・δ|n|G_RZ exact|paired SE|保存固定PF費用|
|---|---|---|---|---:|---|---|---|
|H2 DF3|0.05|B2_p1_q1_2nd_r2_K4|1/2/4; .8|80|384135011/80|15992.018778605378|B1_p3_q1_2nd: 5533984|
|H2 DF3|0.0005|B2_p1_q4_2nd_r2_K4|4/2/4; .2|12|998482376201/4|276760628.92856276|B1_p3_q4_2nd: 305695076994|
|H4 DF10|0.05|B2_p3_q1_2nd_r4_K2|1/4/2; .8|1108|39405103093/554|174917.73703479147|B0I_p5_q1_2nd: 70108214|
|H4 DF10|0.0005|B2_p2_q8_2nd_r2_K4|8/2/4; .1|78|193598605843204/39|14140828222.905321|B0_p7_q8_2nd: 7856558002614|
|H6 DF19|0.05|B2_p4_q2_2nd_r4_K2|2/4/2; .4|48|10144814685/16|4054009.1059934883|B0I_p7_q1_2nd: 356097424|
|H6 DF19|0.0005|B2_p4_q8_2nd_r4_K2|8/4/2; .1|104|1709532575849751/52|65539320043.303085|B0I_p11_q4_4th: 36384720355183|
|H8 DF28|0.05|B2_p8_q2_2nd_r4_K4|2/4/4; .4|48|9840910801/8|3175066.353534034|B0I_p11_q1_2nd: 1267321444|
|H8 DF28|0.0005|B2_p7_q16_2nd_r4_K4|16/4/4; .05|48|546493203592571/6|144211830223.71762|B0I_p15_q4_4th: 94193016652860|

SOURCE：H2 [H2_SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce1f511c4f7429190adcee58cde86be8a87a2fb1/scripts/h2_p_science.py)、H4 [H4_SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3c664ca4a9f003880b9f6d56f2a0e55ed5b56603/scripts/h4_tol_p_science.py)、H6 [H6_SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf0c26dae4477de5fe7264a7c91e077661ef9b38/scripts/h6_pr_confirmation.py)、H8 [H8_SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/h8_pr_confirmation.py)。
結果：[H2_confirmation_result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2d1a86c431f835db700f348ce618d8604e7ee23/artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/confirmation_result.json)、[H4_result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/result.json)、[H6_result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cccfcb2051dacf29e91473e2e1152acef1574bd/artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json)、[H8_comparisons](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv)。

## 次数・層・compiled指標

|系/ε|保存event件数：0/2/4|保存trajectory層|未観測・少数観測と支持外|
|---|---|---|---|
|H2 .05|条件別総数未引用；2条件合計256|S0_ONLY79、S2_SINGLE1|single少数、multi/S4未観測。K4のS4は支持内|
|H2 .0005|同上|S0_ONLY12|single/multi/S4未観測。law数値は公開projectionでnull|
|H4 .05|4395/37/0|S0_ONLY1071、S2_SINGLE37、multi0、S4 0|multi未観測、S4はK2支持外|
|H4 .0005|1241/7/0|S0_ONLY71、S2_SINGLE7、multi0、S4 0|single少数、multi/S4未観測|
|H6 .05|374/10；RTE_event384|公開aggregateは次数件数、single/multi trajectory別集計なし|nonzero観測あり、multi coverageは未判定のまま。S4支持外|
|H6 .0005|3327/1；RTE_event3328|同上|order2一回、multi未被覆、component列不足。S4支持外|
|H8 .05|384/0/0|報告上全event0、層の新集計なし|single/multi/S4未観測|
|H8 .0005|3072/0/0|同上|single/multi/S4未観測|

H2は[H2_trajectory_receipts_projection](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2d1a86c431f835db700f348ce618d8604e7ee23/artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/trajectory_receipts_projection.json)、H4は[H4_event_label_audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/event_label_audit.json)と[H4_result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/result.json)、H6は[H6_result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cccfcb2051dacf29e91473e2e1152acef1574bd/artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json)、H8は[H8_REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md)から転記。H2の欠けたevent histogramやH6のtrajectory histogramをSOURCEから新生成していない。

全行full measured Hadamard wrapper。H2の`paired_costs_projection.json`、H4のgzip projection、H6の`axis_costs.csv`、H8の`paired_metrics.csv`にRZ/CX、各depth、total depth、sizeが保存される。fingerprint/compiler/event digestは各projectionのfieldとH8 packet task-axis対応まで確認する。回路本体があることの証明にはしない。

例示する保存replica0 cosine指標は、H2 q1 RZ301/CX148、H4 q1 RZ3433/CX1500、H6 q2 RZ21787/CX8992、H8 p8q2 RZ75764/CX30936。これは標本の一行であり平均・上限ではない。[H2_paired_costs_projection](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2d1a86c431f835db700f348ce618d8604e7ee23/artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/paired_costs_projection.json)、[H4_paired_costs_projection](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/paired_costs_projection.json.gz)、[H6_axis_costs](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cccfcb2051dacf29e91473e2e1152acef1574bd/artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/axis_costs.csv)、[H8_paired_metrics](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/paired_metrics.csv)。

## A/B/C/D：条件ごとの次の不足

A＝既存保存証拠・既証明関係で扱える範囲。B＝保存データの新しい数値解析。C＝新しい回路生成・compile等の候補。D＝評価法・必要仮定が未確定。区分は作業要素別であり、各行を単一文字へ潰さない。Cは新取得必須という科学判断ではなく、B/静的確認後も必要情報が不足する場合の候補である。

|固定条件|A：今扱えること|B：不足情報・最小解析候補|C：既存費用が不足する場合|D：現時点で未確定|
|---|---|---|---|---|
|H2 p1q1 r2 K4|P n80の平均/SE、single1・S0 79、full wrapper metrics|原traceのτ/λ_R/B_Kとcanonical law、各層質量、保存層費用の結合。公開確率nullを新解析で埋める場合は別承認|同一p1q1のmulti/S4または未被覆component列の条件付きwrapper取得を限定設計|P固有alphabet/compiler/library閉包、全support費用/推論証明。小さいH2でも全列挙を当然に容易とはしない|
|H2 p1q4 r2 K4|P n12の保存点、全S0|上と同じq4 lawと層質量/費用対応。q1の単発1本を転用しない|p1q4のsingle/multi/S4で必要な層だけ取得案|q1と同じ証明前提に加えq4の境界処理|
|H4 DF10 p3q1 r4 K2|P n1108、single37、既存同cohort MC/層平均/経験min-max|元lawとmulti質量、既存MC不足層の感度を保存値で評価する案。既存bootstrapを重複実行しない|DF10 p3q1 K2 multiの必要なconditional取得。legacy p3と混合しない|K2はS4支持外だがmulti/component tail上界は未証明|
|H4 DF10 p2q8 r2 K4|P n78、single7、既存MC/層平均|p2固有law/normalization、multi/S4と保存層費用の結合|p2q8のmulti/S4。legacy DF12 p3q8 pilotは代用しない|DF10/p2 preparationのalphabetとtranslator closure、条件付き小標本からの推論|
|H6 p4q2 r4 K2|P n48、order2=10、保存全paired費用|原P traceのsingle/multi対応、P-specific law、未被覆histogram/component列の確認・感度|p4q2に必要なmulti等。旧p5/q8/r2の費用移送はしない|元L1/L2の条件を今回P input/runtimeへ結ぶ証明不足|
|H6 p4q8 r4 K2|P n104、order2=1、保存paired SE|r4のlaw・層分割と少数観測寄与。旧p5/r2の既存π/費用は転用しない|p4q8 r4のsingle/multiで必要範囲。既存p5 s2/multi pilotは再実行しない|one rare eventだけで母分散・母平均上限を認定する既存証拠なし。compiler閉包も未確定|
|H8 p8q2 r4 K4|P n48、次数0のみ、redacted packet・profile補足|p8固有lawと未観測層mass、保存S0点/shotを結合した限定感度案。既存p7q4/8境界は別条件|p8q2のsingle/multi/S4。原回路0/192取得という欠測を解消する再生成も別科学承認|compiled body不在下のfingerprint再導出、全alphabet/translator/runtime閉包・推論|
|H8 p7q16 r4 K4|P n48、次数0のみ、profile/runtime mapの保存対応|q16 law・境界を旧p7q4/8のlawと区別して結合。旧conditional費用は参考だけ|同p7でもq16 whole-wrapperの必要層取得。qを掛けるだけの転用はしない|cross-event最適化とcompiler上界。古いprofile未取得statusは現在の不足へ再掲しない|

参照可能なSOURCEは上の4科学SOURCE、共通sampler [RTE_SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7/src/trotterlib/rte.py)、[NATIVE_COST](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/pr_cost_native.py)、既存law/感度 [RARE_SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3c0c9546701fd953de3b299be561344375c1a78b/src/trotterlib/h8_rare_tail_selection.py)、理論DP [NATIVE_MODEL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/src/trotterlib/native_df_gate_model.py)。SOURCEの存在は実Pへの新評価済みという意味ではない。

## Cohortを混合しない対応

H2 FP n1/A n8/P n80,12、H4 DF10 FP n32/A n8/P n1108,78、H6/H8 A16探索n8/旧確認union8または6 n32/新Pを分離する。H6旧p5確認n32・H4 legacy DF12のP・H8条件付きpilot n8は今回Pへ追加しない。H8主表と既存Figure2Bの同じn48を新独立証拠と二重計上しない。100scope統合も旧union由来である。[PROTOCOL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_pr_resource_evaluation_protocol_v2.md)、[Q100](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_pf_q_model_evidence_integration_20261020.md)。

保存paired SEは上記Gの標本平均MCSE。CI/p値/実power/選択後coverage、未観測component列を含む母分散/母平均上界は付与しない。層n1のSEを0にしない。H4の保存group min/maxも各観測層の実現値の範囲である。

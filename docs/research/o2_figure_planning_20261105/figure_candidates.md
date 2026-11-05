# 図表候補の役割・根拠・制約

図表の仮候補を列挙した技術準備。主図/補足/枚数/配置/claimの正式採択ではない。
priorityは研究上の役割・保存値の直接性・条件の明確さ・重複からの定性的判断。新しい数値スコアはない。
各IDの厳密な固定commit/path/blob/selectorと全matching条件はfigure_data_plan.csvへ進む。
保存値は条件付き有限標本/元政策の記述で、観測と元reportの機構解釈を区別する。

## DF01：旧H4 development・固定transferの精度依存

- 役割・問い：既存有限候補の精度適格性と費用方向は要求精度でどう変わるか
- 研究：pr2_matched_accuracy_m1_b1_execution, pr2_matched_accuracy_m2_transfer_execution, pr2_pm2_precision_analysis
- 保存比較ID：M1_registered_domain, M2_fixed5_transfer, PM2_posthoc_precision
- データ/軸候補：ADD_OLD_PRECISION, ADD_OLD_TRANSFER。x=epsilon_signal [dimensionless]、y=saved G [RZ×shot], N_real/N_imag [shots]。
- 固定条件：H4 legacyDF12;1.00Å development /1.30Å fixed5 transfer;T=.8/S2;原302点POSTHOC政策;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：旧有限集合で保存された精度依存・固定5構成の移送を表示できる。
- 制約：DF10・後続B0I/高次PFへ接続しない。transfer再最適化をした図にしない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。旧研究の出発点と条件依存を説明する。後続強対照図とdomainを区別。
- 重複候補：DF02, DF03。同じ価格/cohortを別実験と数えない。

## DF02：H4 B0I追加後のgeometry×精度比較

- 役割・問い：対照familyの拡張で保存された資源方向がどう変わるか
- 研究：track_a_h4_b0i_map_results
- 保存比較ID：H4_B0I_geometry_map
- データ/軸候補：PRI_H4B0ICSV。x=geometry [Å] またはepsilon_signal、y=saved B0I_work/B2_work [RZ×shot]。
- 固定条件：H4 legacyDF12;元6geometry;T=.8/S2;代表18保存行;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：代表18点のB0I/B2費用・原winnerを表示できる。
- 制約：1812点全地図の再構築ではない。397/1415を新成功率へ変換しない。geometryごと別H/state。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。強いdiscard対照の役割を示す。DF01と重複する精度説明は統合候補。
- 重複候補：DF01。同じ価格/cohortを別実験と数えない。

## DF03：H2/H4/H6共通PF候補域と保存PR

- 役割・問い：強いPF対照を含めた取得候補内の精度別方向を分子別にどう示せるか
- 研究：track_a_h246_pf_common_final, track_a_h246_reconstructed_confirmation
- 保存比較ID：H246_common588, H246_fixed_high_order_confirmation
- データ/軸候補：PRI_PF60。x=epsilon_signal;system H2/H4/H6 [categorical]、y=PF_total_RZ/PR_mean_RZ [saved G,RZ×shot];PR_SE [same unit]。
- 固定条件：H2/H4/H6 tol-onlyDF3/10/19（H6 weighted completion）;T=.8;S2/Y4/m5元588域;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：元60比較行の有限候補・元方向と独立固定高次確認の意味を示せる。
- 制約：systemをscaling fitに使わない。PF_total_RZ等は原shot加重総費用の列名;PR探索と118価格固定確認を混ぜない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。強PFを含めた条件依存を直接示せ、O2のDF側の基礎比較となる。
- 重複候補：DF04, DF10。同じ価格/cohortを別実験と数えない。

## DF04：H4/H6長時間の保存資源比較

- 役割・問い：時間Tと要求精度に対する方式別の保存費用・確認方向は何か
- 研究：track_a_h46_long_time
- 保存比較ID：H46_original_independent_comparisons
- データ/軸候補：PRI_LONGBEST, DF_LONG_COMPARE。x=T [元研究の時間単位]、y=saved G [RZ×shot];saved difference_SE [RZ×shot]。
- 固定条件：H4DF10/H6DF19 weighted completion;T=.8/1.2/1.6/2;元5精度;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：元探索nomineeと別価格の方向確認を分けて表示できる。
- 制約：旧33/40・追加7後33/40・境界Rは別集合。新成功率/勝者/母平均CIを作らない。Tに伴いq/prefix等も変わる。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。時間・精度依存と探索/確認の分離を同時に示せる。
- 重複候補：DF03, ME01, ME03, ME04。同じ価格/cohortを別実験と数えない。

## DF05：同q/prefix/R32/K4でのPF次数対照

- 役割・問い：同じラベル条件でも公式の違いでNと保存費用はどう異なるか
- 研究：track_a_h46_long_time
- 保存比較ID：same_q_prefix_R32_K4.csv
- データ/軸候補：MATCHED。x=formula [S2/Y4/m5等の原ラベル]、y=N_cos/N_sin [shots];saved G/G_SE [RZ×shot];log_B [dimensionless]。
- 固定条件：H4DF10/H6DF19;元same_q_prefix_R32_K4集合;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：N・log_B・取得済み価格の次数依存を示せる。
- 制約：signed piecesとr配分は公式ごと異なる。NOT_ACQUIREDのGは欠測表示。純粋次数単独の因果効果ではない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。DF03の次数選択を固定ラベル対照で補う。
- 重複候補：ME01, ME02。同じ価格/cohortを別実験と数えない。

## DF06：固定sliceでの決定論prefixとN/C/G

- 役割・問い：保存prefix候補の適格性と取得価格をどこまで表示できるか
- 研究：track_a_h46_long_time
- 保存比較ID：H46_original_independent_comparisons
- データ/軸候補：DF_LONG_CAND。x=prefix L_D [DF fragments]、y=N [shots], C [RZ/shot], G [RZ×shot]を別軸候補。
- 固定条件：H4DF10;T=.8/epsilon=.0005/S2/q8;同R/Kで分離する保存slice;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：選択済みslice内のprefix候補と欠測をそのまま示せる。
- 制約：sourceの取得価格範囲だけ。prefix以外のR/Kやlawが違う点を一本の因果曲線にしない。新min/補間なし。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。prefix選択と価格欠測を可視化する補助候補。
- 重複候補：ME02。同じ価格/cohortを別実験と数えない。

## DF07：DF B3の保存負結果

- 役割・問い：one-body保持B3は取得域でB2/決定論envelopeとどう比較されたか
- 研究：track_a_h46_long_time_b3
- 保存比較ID：B3_fixed_negative
- データ/軸候補：PRI_B3CSV。x=T/epsilon_signal [元条件]、y=saved B3_G/baseline_G [RZ×shot]。
- 固定条件：H4DF10/H6DF19 weighted completion;元T・精度・固定nominee域;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：取得40条件低G0/40という原負結果を保存比較のまま示せる。
- 制約：全純RTE不可能性ではない。B3はone-body保持。公開主要結果一致と全private rawidentityは別。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。有利例だけに偏らず、保持境界の負結果を示せる。
- 重複候補：XP04。同じ価格/cohortを別実験と数えない。

## DF08：旧H8 T=.8固定確認の保存主表

- 役割・問い：旧固定PR/PFの2精度・価格cohortの結果をどう示せるか
- 研究：track_a_h8_new_pr_confirmation_results
- 保存比較ID：H8_confirm_T08
- データ/軸候補：H8_OLD_COMPARISONS。x=epsilon_signal .05/.0005、y=saved mean_exact/fixed_PF_G_exact [G,RZ×shot]。
- 固定条件：H8DF28/T=.8;元fixed PR/PF;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：元POINT_REPRODUCEDと固定費用値を別cohortとして参照できる。
- 制約：Main/Mechanism/Figure2Bは同じn48。p8旧1+新47。export-match falseを科学不一致/照合完了へ書き換えない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。旧H8確認履歴を補うがDF09のT1.6とは別条件。
- 重複候補：DF09, DF10。同じ価格/cohortを別実験と数えない。

## DF09：H8 Stage C固定PR/PF比較

- 役割・問い：T1.6固定候補の確認平均と保存PF費用はどう対応するか
- 研究：track_a_h8_long_time
- 保存比較ID：H8_M5_P8_Q1_CONFIRMATION_01
- データ/軸候補：H8_C_RAW。x=fixed candidate PR/PF [categorical]、y=saved G and sample_SE_G [RZ×shot];N_cos/N_sin [shots]。
- 固定条件：H8DF28/T1.6/epsilon=.0005;PR m5/p8/q1/R32/K4 ID7eb0a3d691b94089969b;PF B0I m5/p15/q1 ID1425ae6ed83fd9098b39;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：保存固定比較の観測PR低Gと記述的sample SEを示せる。
- 制約：探索PR n9をpoolしない。PR平均Cはaggregate未保存。PF SE null・rare order条件費用nullを保つ。母平均上界ではない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。公開SOURCE/inputと固定候補の確認aggregateが明確なH8例。
- 重複候補：DF08, ME10。同じ価格/cohortを別実験と数えない。

## DF10：H6/H8強化決定論対照の版別比較

- 役割・問い：対照を強めると保存方向はどう変わったか
- 研究：track_a_v1_p7_measured_reaccount_run01, track_a_h8_six_axis_reaccount_run01
- 保存比較ID：H6_p7_measured, H8_strengthened184
- データ/軸候補：PRI_H6STRONGCSV, PRI_H8STRONGCSV。x=epsilon_signal/allowance/bias/scope原ラベル、y=H6 deterministic_G_exact/saved_PR_G_exact・H8 G_before_exact/G_after_exact [RZ×shot]。
- 固定条件：H6DF19/H8DF28/T=.8;H6 epsilon=.05主条件;H8元signed/allowance/scope;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：元の強化対照・direction列を版別に示せる。
- 制約：DIRECT/MODEL/外挿を区別。H8旧固定n48主表・H8Cとpoolしない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。DF03/04の対照強度を理解する補足。
- 重複候補：DF03, DF08。同じ価格/cohortを別実験と数えない。

## PA01：H2最新純RTE対Hybridの精度依存

- 役割・問い：保存最新選択と旧Hybridの条件付きGはどう異なるか
- 研究：h6_hybrid_fairness_extension, h2_pauli_rte_validation, non_df_h4_pauli_pr_search, pauli_rte_event_extension
- 保存比較ID：Pauli_latest14of15
- データ/軸候補：PA_BASE, PRI_PA-E4。x=epsilon_signal、y=saved G [RZ×shot];N [shots];C [RZ/shot]。
- 固定条件：H2 Pauli14/raw state/full-H reference;T=.8/S2/central/P0;hypothetical rho=0
- 読める内容：元候補ID・精度対応を保持してH2の条件付き競合を示せる。
- 制約：候補2本×全精度を新winner探索に使わない。元精度別対応に従う。逐次確認後minの選抜効果がある。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。Pauli側の小系比較とN/C交換を保存値から示せる。
- 重複候補：PA05。同じ価格/cohortを別実験と数えない。

## PA02：H4最新Hybrid対純RTEと粗精度例外

- 役割・問い：精度による保存方向の例外をどう示せるか
- 研究：h6_hybrid_fairness_extension, h2_pauli_rte_validation, non_df_h4_pauli_pr_search, pauli_rte_event_extension
- 保存比較ID：Pauli_latest14of15
- データ/軸候補：PRI_PA-E5。x=epsilon_signal、y=saved G [RZ×shot];N/Cは別軸候補。
- 固定条件：H4 Pauli184;保存raw ray/full-H reference;T=.8/S2/central/P0;hypothetical rho=0
- 読める内容：H4粗精度Hybrid低G、他4精度純RTE低Gという保存方向を示せる。
- 制約：E3 primary nominee .05 K2/.01 K4と現在直接比較の.05 HybridK4/.01 RTEK2を混同しない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。14/15の例外を明示し、保持が常に不利という解釈を防げる。
- 重複候補：PA04, PA05。同じ価格/cohortを別実験と数えない。

## PA03：H6 Hybrid拡張後の最新固定比較

- 役割・問い：後続r32 Hybrid追加後も保存方向は何か
- 研究：h6_hybrid_fairness_extension, h2_pauli_rte_validation, non_df_h4_pauli_pr_search, pauli_rte_event_extension
- 保存比較ID：Pauli_latest14of15
- データ/軸候補：PRI_PA-E6compare。x=epsilon_signal、y=saved new_Hybrid_G_exact/RTE_G_exact [RZ×shot]。
- 固定条件：H6 Pauli918/full-H saved sector state;T=.8/S2/central/P0;Hybrid p78/q8/r32/K2対保存RTE r64;hypothetical rho=0
- 読める内容：最新H6結果版の条件付き方向を既存比較行で示せる。
- 制約：旧H6最良family・旧水平線を新r32値へ再ラベルしない。RTE保存cohortを再取得した扱いにしない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。O2のPauli主比較に使う結果版が明確で、旧H6選択の混入を避ける。
- 重複候補：PA06, ME05, ME06。同じ価格/cohortを別実験と数えない。

## PA04：Pauli Full PF/B0/Hybridの保存対照

- 役割・問い：強い決定論対照を含めた分子別の保存値は何か
- 研究：non_df_h4_pauli_pr_search, h6_hybrid_fairness_extension, h2_pauli_rte_validation, pauli_rte_event_extension
- 保存比較ID：Pauli_H4_PR_vs_deterministic, Pauli_latest14of15
- データ/軸候補：PA_BASE, PRI_PA-E3。x=epsilon_signal;system [categorical]、y=saved G [RZ×shot]。
- 固定条件：H2/H4/H6 Pauli14/184/918;T=.8/S2;各保存H/state/reference;rho=0仮想
- 読める内容：元決定論対照と保存PRの費用を歴史版として示せる。
- 制約：PA_BASEの旧H6 Hybridを最新r32へ置換しない。H4 E3とPA02の選択policyが違う。B0/B0Iは同じPauli実装で独立方式でない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。四方式比較の背景だが最新主比較との重複・版違いが大きい。
- 重複候補：PA01, PA02, PA03。同じ価格/cohortを別実験と数えない。

## PA05：Pauli event数s・r/Kの保存感度

- 役割・問い：event増加でB/N/C/Gの保存観測はどう競合するか
- 研究：h6_hybrid_fairness_extension, h2_pauli_rte_validation, non_df_h4_pauli_pr_search, pauli_rte_event_extension
- 保存比較ID：Pauli_latest14of15
- データ/軸候補：ADD_PA_EVENT, ADD_H2_EVENT。x=saved s [events], r/K [counts]、y=saved B [dimensionless];N [shots];C [RZ/shot];G [RZ×shot]を別軸候補。
- 固定条件：Pauli H2/H4/H6別保存H/state/reference;T=.8/S2;sourceのevent extension域;rho=0仮想
- 読める内容：元離散event候補の非単調な保存費用と欠測を示せる。
- 制約：sの未保存点の補間最適化なし。旧H6 Hybrid水平線は歴史anchorで最新PA03ではない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。機構説明に有用だがPA01/02/03と同じ価格を再引用する。
- 重複候補：PA01, PA02, PA03, PA06。同じ価格/cohortを別実験と数えない。

## PA06：H6 prefix78でのr16/r32/r64感度

- 役割・問い：保持prefix固定の追加Hybrid候補のN/C/Gをどう示せるか
- 研究：h6_hybrid_fairness_extension, h2_pauli_rte_validation, non_df_h4_pauli_pr_search, pauli_rte_event_extension
- 保存比較ID：Pauli_latest14of15
- データ/軸候補：PA_H6_CAND。x=r [events per outer]、y=N [shots];C [RZ/shot];G [RZ×shot]。
- 固定条件：H6 Pauli918/prefix78/q8/K2/T=.8/S2;保存SOURCE887b69e4;rho=0仮想
- 読める内容：元固定prefix/r候補と探索・確認の保存値を表示できる。
- 制約：保持部分・lawをDFと一致とみなさない。確認後minの新選抜やprefix間因果評価なし。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。PA03の選択文脈とP1/P2の対象候補の橋渡し。
- 重複候補：PA03, PA05, ME05, ME06。同じ価格/cohortを別実験と数えない。

## XP01：旧H4共通入力の固定実装比較

- 役割・問い：共通MO入力/raw rayでの既存固定実装費用はどう違うか
- 研究：non_df_h4_comparison_synthesis
- 保存比較ID：H4_generic_central_DF
- データ/軸候補：H4_CROSS_PRICES, H4_CROSS_ACCOUNT。x=case/implementation またはepsilon_signal、y=saved C [RZ/shot];conditional G [RZ×shot]。
- 固定条件：旧H4/共通MO入力・raw ray/T.8/S2/q8/r4/K4;Pauli184/p27 vs DF10/p1+one-body
- 読める内容：固定実装/cohortの保存価格とrho0条件付き会計の差を示せる。
- 制約：H一致は厳密DF近似同一性の新認定でない。decomposition/split/law/basis/controlが同時に違う。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `CONDITIONAL_MATCHED`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。表現横断の既存case studyだが交絡の説明が必要。
- 重複候補：XP02, XP03。同じ価格/cohortを別実験と数えない。

## XP02：H4 P3共通Gaussian保持の条件付き比較

- 役割・問い：matched partitionで保存Nとcompiled費用の競合をどう示せるか
- 研究：df_pauli_matched_partition_validation
- 保存比較ID：P3_Stage4
- データ/軸候補：P3_ACCOUNT, P3_COMPARE。x=epsilon_signal、y=saved N [shots];G_RZ/G_SE [RZ×shot];saved ratio [dimensionless]。
- 固定条件：H4/tol-onlyDF10/Pauli184/DF rotated-Z/ZZ360・10basis;共通raw ray・Gaussian T_eff保持/V_square残差;T=.8/S2/q8/r4/K4;Stage3run04各arm32価格;finite-bit gap=0・compiled rho=0・B engineering coverage仮定
- 読める内容：共通保持case studyでDFが少NでもPauliが低Gという保存条件付き交換を示せる。
- 制約：表現単独の効果でない。finite-bit追補で仮定を解除せず、実装accuracy保証やG順位更新をしない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `CONDITIONAL_MATCHED`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。共通保持を明示した最も直接的な既存横断case study。
- 重複候補：ME07, ME08, XP01。同じ価格/cohortを別実験と数えない。

## XP03：同一Pauli programのgeneric/central制御価格

- 役割・問い：同一保存Pauli programでcontrol方式の価格差をどう示せるか
- 研究：non_df_h4_comparison_synthesis
- 保存比較ID：H4_generic_central_DF
- データ/軸候補：H4_CROSS_PRICES。x=control implementation [categorical]、y=saved compiled C/SE [RZ/shot]。
- 固定条件：H4 Pauli184/p27;同保存program・同case/q8/r4/K4/T.8;generic/central別保存価格
- 読める内容：同保存Pauli programの操作的control価格対照を示せる。
- 制約：MATCHEDはcontrol対照の範囲。DF–Pauli表現単独効果を検証したことにはならない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_FROM_SAVED_DATA` / `MATCHED`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。一致条件が明確な単回価格の対照。
- 重複候補：XP01, ME06。同じ価格/cohortを別実験と数えない。

## XP04：DF B3とPauli純RTEの異なる観測

- 役割・問い：異なるランダム化境界の保存観測をどの限定で並置できるか
- 研究：track_a_h46_long_time_b3, h6_hybrid_fairness_extension, h2_pauli_rte_validation, non_df_h4_pauli_pr_search, pauli_rte_event_extension
- 保存比較ID：B3_fixed_negative, Pauli_latest14of15
- データ/軸候補：SRC_B3, PAULISYNTH。x=algorithm/context [categorical]、y=元報告の方向・定義・適用条件の表（共通数値軸なし）。
- 固定条件：DF10/19長時間one-body保持 vs Pauli14/184/918 T.8全非identityランダム化;H/state/law/cohortは別
- 読める内容：保存された異なる定義と観測をcontext表として対比できる。
- 制約：共通G軸の直接順位・DF/Pauliの表現因果比較には使えない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `CONTEXTUAL`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。O2の統合で必要な方式定義の相違を簡潔に示せる。
- 重複候補：DF07, PA03。同じ価格/cohortを別実験と数えない。

## XP05：最新family最良候補同士の直接matched図

- 役割・問い：最新DF/Pauliの選択を共通条件として結合できるか
- 研究：track_a_h246_pf_common_final, pauli_h2_h6_comparison
- 保存比較ID：既存matchingの未確認/不適切比較の記録
- データ/軸候補：SRC_PF588, PAULISYNTH。x=共通条件の候補対応（未確定）、y=matched G（未確認）。
- 固定条件：同一H/state/reference、時間、精度政策、候補集合、価格cohortのbinding未確認
- 読める内容：既存台帳で何が未確認かを一覧化するまで。
- 制約：latestというラベル・分子名・q/r/Kの一致だけでmatchedを成立させない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`UNCONFIRMED` / `UNCONFIRMED`。共通H/state/reference・候補集合・cohortの固定対応。まず既存bindingの確認範囲をユーザーが指定。
- 優先：DEFER。既存資料の対応が未確認で、直接比較図を先に決められない。
- 重複候補：なし（本計画内）。同じ価格/cohortを別実験と数えない。

## XP06：legacyDF12とP3 DF10の同条件結合図

- 役割・問い：異なるDF近似を同一Hのprecision系列として結合できるか
- 研究：pr2_matched_accuracy_m1_b1_execution, df_pauli_matched_partition_validation
- 保存比較ID：M1_registered_domain, P3_Stage4
- データ/軸候補：SRC_M1, P3_STAGE4。x=epsilon_signal（結合禁止）、y=共通G（不適切）。
- 固定条件：H4 legacyDF12 とtol-onlyDF10;異なるH/分割/条件
- 読める内容：条件の相違の表だけに使える。
- 制約：同条件の一本の曲線/順位に結合しない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`INSUFFICIENT_COMPARABILITY` / `NOT_COMPARABLE`。意図する同一H比較のbindingがない。両結果は別条件表示として維持。
- 優先：DEFER。異なるHamiltonian近似を混合する意図が不適切。
- 重複候補：なし（本計画内）。同じ価格/cohortを別実験と数えない。

## XP07：H8 StageCとPauli別分子の直接費用図

- 役割・問い：異なる分子・時間の費用を同条件の表現比較にできるか
- 研究：track_a_h8_long_time, h6_hybrid_fairness_extension, h2_pauli_rte_validation, non_df_h4_pauli_pr_search, pauli_rte_event_extension
- 保存比較ID：H8_M5_P8_Q1_CONFIRMATION_01, Pauli_latest14of15
- データ/軸候補：H8_C, PAULISYNTH。x=representation（共通条件なし）、y=common G ranking（不適切）。
- 固定条件：分子/H/state/T/cohort全て別
- 読める内容：各研究の条件一覧を別欄で示すまで。
- 制約：scaling/表現効果/横断winnerを読み取らない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`INSUFFICIENT_COMPARABILITY` / `NOT_COMPARABLE`。同分子・同H/state/Tの対応がない。
- 優先：DEFER。分子と時間を揃えた既存対照ではない。
- 重複候補：なし（本計画内）。同じ価格/cohortを別実験と数えない。

## ME01：保存C/N/G分解とPF次数

- 役割・問い：高次公式のshot節約と単回費用増をどの保存比で説明できるか
- 研究：track_a_h46_long_time
- 保存比較ID：C_N_G_decomposition.csv
- データ/軸候補：CNG。x=higher_formula/T/epsilon [original labels]、y=saved higher_over_S2_N/C_eff/G [dimensionless]。
- 固定条件：H4DF10/H6DF19;元長時間探索選択のhigher対S2;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：元保存比からN・C_eff・Gの競合を説明できる。
- 制約：C_effは元shot加重効率指標でC_cos/C_sinと同一ではない。q/prefixも変わるので因果寄与率でない。新比を作らない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。総費用だけでは見えない負担交換を保存比のまま示せる。
- 重複候補：DF04, DF05。同じ価格/cohortを別実験と数えない。

## ME02：R_outer/K・B/bias/headroomの保存感度

- 役割・問い：有限RTE条件とN/C/G取得状態はどう対応するか
- 研究：track_a_h46_long_time
- 保存比較ID：r_K_sensitivity.csv
- データ/軸候補：RK。x=R_outer/K [original allocation]、y=saved log_B/bias/headroom [dimensionless];N/C/Gは別軸。
- 固定条件：DF10/19;元長時間保存R/K域;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：適格signalと価格取得を分け、保存値の競合を示せる。
- 制約：ELIGIBLEでも未価格Gは0でない。log_BからBを復元しない。R_outerと各signed occurrenceのrは違う。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。広い感度を示せるが欠測が多く、固定境界例ME03/04と目的が重なる。
- 重複候補：DF06, ME03, ME04。同じ価格/cohortを別実験と数えない。

## ME03：H4境界Rの固定保存比較

- 役割・問い：R変更による保存N/C/Gと元診断の関係は何か
- 研究：pr_boundary_execution_20261027
- 保存比較ID：H4_R16_R128_vs_R64
- データ/軸候補：BOUNDARY_RESULTS。x=R_outer [元ABS_TIME_PER_OUTER配分]、y=N [shots];C [RZ/shot];G/SE [RZ×shot]。
- 固定条件：H4 DF10/19;T2/q1/m5_best;H4 L_D3/K2又はH6 L_D7/K6;epsilon=.0005;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：保存固定比較の負担交換と元HIGHER/INCONCLUSIVEを表示できる。
- 制約：R16/R128は保存HIGHER。R32探索n9は確認へ入れない。R64全域最適でない。 ±2 combined-SEは元normal近似感度で厳密CIでない。完全SOURCEはprivate保全、公開summary IDのみ。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。Rを増減すれば一律低Gという解釈を防ぐ具体的固定例。
- 重複候補：ME02, DF04。同じ価格/cohortを別実験と数えない。

## ME04：H6境界Rの固定保存比較

- 役割・問い：R変更による保存N/C/Gと元診断の関係は何か
- 研究：pr_boundary_execution_20261027
- 保存比較ID：H6_R11_vs_R16
- データ/軸候補：BOUNDARY_RESULTS。x=R_outer [元ABS_TIME_PER_OUTER配分]、y=N [shots];C [RZ/shot];G/SE [RZ×shot]。
- 固定条件：H6 DF10/19;T2/q1/m5_best;H4 L_D3/K2又はH6 L_D7/K6;epsilon=.0005;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）
- 読める内容：保存固定比較の負担交換と元HIGHER/INCONCLUSIVEを表示できる。
- 制約：R11はpoint_lower=Trueでもdiagnostic=INCONCLUSIVE。確定改善へ変えない。 ±2 combined-SEは元normal近似感度で厳密CIでない。完全SOURCEはprivate保全、公開summary IDのみ。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。Rを増減すれば一律低Gという解釈を防ぐ具体的固定例。
- 重複候補：ME02, DF04。同じ価格/cohortを別実験と数えない。

## ME05：H6 P1の保存signal差分

- 役割・問い：outer split/可換保持/finite-RTE/gridの元診断は何か
- 研究：pauli_pure_rte_mechanism_validation/bias
- 保存比較ID：P1_bias
- データ/軸候補：P1_DIFFERENCES。x=candidate H16/H32/H64/R64 × saved difference component、y=saved real/imag/absolute signal differences and diagnostic resolution [dimensionless]。
- 固定条件：H6Pauli918/T.8/S2/q8/K2;H prefix78 r16/32/64 vs pureRTE r64;保存full-H ray/reference;engineering floor1e-10
- 読める内容：解像されたouter差と未解像finite/grid差を原診断のまま表示できる。
- 制約：hold構造ゼロと浮動差を分ける。UNRESOLVEDを0/同等としない。N/Gなし。DF比較へ移送しない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。観測差と機構解釈の境界が明記された限定機構例。
- 重複候補：PA06, ME06。同じ価格/cohortを別実験と数えない。

## ME06：H6 P2のFull/R/E/Emptyと非加法性

- 役割・問い：同固定programの部分価格・contextual incrementはどこまで保存されているか
- 研究：pauli_pure_rte_mechanism_validation/cost
- 保存比較ID：P2_parts
- データ/軸候補：P2_INTERACTIONS。x=fixed program/axis/component [categorical]、y=saved full/retained/event/empty/Delta_interaction [RZ/shot]。
- 固定条件：H6Pauli918;保存confirmation/index0000;T.8/S2/q8/K2;Full6既存/部品12元compile
- 読める内容：保存部分価格と元非加法性をそのまま示せる。
- 制約：母平均N/Gの分解でない。FullをR+Eへ加算して再構成しない。因果寄与率/一般compiled節約率ではない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_FROM_SAVED_DATA` / `WITHIN_SAVED_STUDY`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：PRIORITY。固定programの単回費用を直接説明でき、新集計が不要。
- 重複候補：PA06, ME05, XP03。同じ価格/cohortを別実験と数えない。

## ME07：P3のB・bias/headroom・N・Gの競合

- 役割・問い：共通保持の条件付き会計で何がNとGの交換を説明するか
- 研究：df_pauli_matched_partition_validation
- 保存比較ID：P3_Stage4
- データ/軸候補：P3_STAGE4, P3_AXES, P3_ACCOUNT。x=arm P/DF × epsilon/axis、y=保存B/bias/headroom [dimensionless];N [shots];G [RZ×shot]。
- 固定条件：H4/tol-onlyDF10/Pauli184/DF rotated-Z/ZZ360・10basis;共通raw ray・Gaussian T_eff保持/V_square残差;T=.8/S2/q8/r4/K4;Stage3run04各arm32価格;finite-bit gap=0・compiled rho=0・B engineering coverage仮定
- 読める内容：元reportにあるB候補、headroom、N、Gを併記して条件付き競合を説明できる。
- 制約：B値はreport表示又は元operandを使用。ここから新N/Gや寄与率を計算しない。平均CはG/N逆算しない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `CONDITIONAL_MATCHED`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。XP02と同じデータなので独立図として重複させる必要は未確定。
- 重複候補：XP02, ME08。同じ価格/cohortを別実験と数えない。

## ME08：P3 finite-bit engineering headroom

- 役割・問い：保存候補headroomと未保証のactual欄をどう併記できるか
- 研究：df_pauli_matched_partition_validation
- 保存比較ID：P3_finite_bit
- データ/軸候補：P3_BIT_HEADROOM, P3_BIT。x=epsilon_signal/arm/axis、y=saved engineering headroom [dimensionless];actual=null/unknownを別欄。
- 固定条件：H4/tol-onlyDF10/Pauli184/DF rotated-Z/ZZ360・10basis;共通raw ray・Gaussian T_eff保持/V_square残差;T=.8/S2/q8/r4/K4;Stage3run04各arm32価格;finite-bit gap=0・compiled rho=0・B engineering coverage仮定;finite-bit追補の保存engineering候補
- 読める内容：仮定下の正headroomと実装保証の未成立を区別した表にできる。
- 制約：Stage4のN/Gを更新しない。certified_gap_upper=null/implemented_accuracy_guaranteed=false/actual headroom nullを保つ。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`READY_WITH_CONDITIONS` / `CONDITIONAL_MATCHED`。保存値の選択・表示設計のみ。新集計・比・順位を作らない。図表生成は今回未実施。
- 優先：SUPPORTING。XP02の限定の説明に有用で、別の資源順位図ではない。
- 重複候補：XP02, ME07。同じ価格/cohortを別実験と数えない。

## ME09：表現・保持・law・controlの因果寄与率

- 役割・問い：総費用差を個別要因の割合へ分解できるか
- 研究：pauli_pure_rte_mechanism_validation/bias, pauli_pure_rte_mechanism_validation/cost, df_pauli_matched_partition_validation
- 保存比較ID：P1_bias, P2_parts, P3_Stage4
- データ/軸候補：H6_P1, H6_P2, P3_STAGE4。x=hypothetical factor [unidentified]、y=causal contribution [%]（未保存）。
- 固定条件：既存P1/P2/P3の限定診断のみ;一因子介入対照は未成立
- 読める内容：現在は因果割合が未同定であるという限界表まで。
- 制約：元非加法性・signal差分を総費用因果配分へ変換しない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`NEEDS_NEW_ANALYSIS` / `WITHIN_SAVED_STUDY`。識別対象・因果対照の定義と、その定義に対応する保存/新規データ。
- 優先：DEFER。目的の寄与率が保存されておらず、元研究でも未分離。
- 重複候補：なし（本計画内）。同じ価格/cohortを別実験と数えない。

## ME10：H8 StageCの平均CとGの三量表示

- 役割・問い：未保存PR平均Cを含むN/C/G全量図をそのまま作れるか
- 研究：track_a_h8_long_time
- 保存比較ID：H8_M5_P8_Q1_CONFIRMATION_01
- データ/軸候補：H8_C_RAW, H8_C_PRICES。x=fixed PR/PF candidate、y=N/C_mean/G（PR C_meanはaggregate未保存）。
- 固定条件：DF09と同じ固定H8DF28/T1.6/epsilon.0005/cohort
- 読める内容：N/G図DF09と個別trajectory Cの保存値表示までは可能。
- 制約：G/Nから平均Cを逆算しない。PR_mean_Cの新平均をこの作業で求めない。 取得済み候補・元allowance/shot政策・compiler・有限価格標本に限定。母平均順位・全域最適・実機/実装全体保証は示さない。
- 状態：`NEEDS_NEW_ANALYSIS` / `WITHIN_SAVED_STUDY`。PR confirmation両軸の平均C（必要なら対応SE）の保存aggregate。
- 優先：DEFER。完全な三量図だけが追加集計を要し、DF09で主比較は既存値から示せる。
- 重複候補：DF09。同じ価格/cohortを別実験と数えない。

## 表現横断の読み方

XP01は同じMO/raw rayを使った固定実装の条件付きcase study、XP02は共通Gaussian保持の条件付きcase study。
どちらもdecomposition/law/B/basis/control等が同時に変わり、表現のみの因果効果はNOT_ESTABLISHED。
XP03は同保存Pauli programのcontrol対照であり、MATCHEDという語をDF–Pauli全体へ広げない。
XP04は異なるアルゴリズム境界の説明表。XP05–07を新matched資源図として成立済みにしない。
ME05の観測差、ME06の固定programの非加法性、ME07のB/shot/価格の会計説明はそれぞれ別の範囲。
これらを加算して原因別寄与率ME09を新作成しない。

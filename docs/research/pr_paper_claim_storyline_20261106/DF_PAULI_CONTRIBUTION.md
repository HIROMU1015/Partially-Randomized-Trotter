# DF・Pauliが中心課題Aに答える範囲

2026-11-06 JST。中心課題Aに沿う研究構成メモ。既存証拠の引用・対応付けのみ。新しい数値解析、詳細科学レビュー、原稿本文・Abstract・題名の執筆、Main Claim/構成の正式採択は行わない。

## 1. 共通の比較課題と方式の意味

暫定課題A「強い決定論的手法と比較して、どの条件で、どの程度の資源優位があるか」を、保存状態の有限時間複素signal推定、元の精度/失敗確率政策、P=0、測定付きfull Hadamard wrapperのcompiled RZ×解析的十分shotという既存estimandに限定して検討する。B1のCXは元N・候補を固定した感度である。量子shot実行、状態準備込み費用、最終energy/RPE、FT T数や実機runtimeまで答えた研究とはしない。[SYN][CAT][REAN]

方式を二つの意味に分ける。非空prefixを保持するHybridの利益と、純RTE端点を含むPR familyの利益は同義ではない。DF B3はone-body保持・two-body prefix0、Pauli純RTEはconstant以外の全非identity項のランダム化であり、同じ「L_D=0」だけでは同方式にならない。Full PFに加えB0/B0Iと高次PFを含む元envelopeを対照とし、未実装の全決定論手法に勝つという意味の「最強」は使わない。[MATCH][COMPS]

## 2. DF：Aへの直接的な証拠

**直接性**：A10.BASEの共通PF対照、A11.BASEの精度/時間と確認、A12.BASEの負結果が、同じ資源課題に答える。A06/A07はB0+Iによるbaselineの強化、A09は次数、A14.H8_Cは別条件での固定確認を支える。証拠の直接性と新規性は別に評価する。[OLDMAP]

**保存された量的な例**：Linear H2/H4/H6、隣接1Å、STO-3G、tol-only1e-8のDF3/10/19（H6 weighted-Hermitian completion）、T=.8、q≤8、S2/Y4/m5、P0・symmetric-directional・Qiskit1.3/opt1/seed17の共通PF588比較。H6のε=.0005では、m5 PR prefix5/q1/R32/K4（保存EXPLORATION）対m5 B0I prefix11/q1で保存削減率 **+40.9592%**。同じ共通比較のε=.05では、S2 PR prefix5/q1/R32/K4対S2 B0I prefix7/q1の保存削減率 **-53.8932%**。いずれも元reportの引用で、新しい比・差ではない。δは元q1行の.8。候補域が非対称な有限比較であり、厳精度なら普遍的に勝つ閾値を示したものではない。[SRC_PF588]

**精度・時間**：長時間研究はLinear H4/H6・1Å/STO-3G・DF10/19、T=.8/1.2/1.6/2、元5精度、Full/B0/B0I・S2/Y4/m5を含む取得域で、元RZ点費用は33/40セルでPR低G。強化した候補域と時間を明示して読む。共通PF588の「粗3精度PF・厳2精度PR」と長時間の精度依存は異なる候補集合の結果であり、矛盾や単一の閾値曲線へ処理しない。厳精度側ε=.01/.005/.001/.0005と粗精度ε=.05の保存比較を区別する。旧642、追加7、別価格確認、境界Rを合算しない。[SRC_LONG][SRC_PF588][G01][G02]

**負結果と境界**：共通PF比較ではH2の取得5精度すべてでPF低G。B3は元RZではB2/Full PF/envelopeに対して低Gでなく、B1のCXでもB2/envelopeに対する負結果が残る。ただしFull PF単独にはCX例外がある。粗精度・保持identity・全二体ランダム化・元候補集合が、単なる「PR成功数」以上の比較条件を与える。[SRC_B3][REAN]

**高次・H8**：固定S2に対する高次PR確認と、最良PF envelopeへの勝利は別。高次の探索方向が確認で反転した例も残す。H8 Stage CはDF28/T1.6/ε=.0005、m5 PR prefix8/q1/R32/K4（確認n9）対m5 B0I prefix15/q1（Stage B価格n1）で保存削減24.58909439%。保存PR ID `7eb0a3d691b94089969b` / PF ID `1425ae6ed83fd9098b39`、同じLinear1Å/STO-3G/P0/compiler条件。H2/H4/H6とのscaling fitにせず、rare order費用UNKNOWN・母平均保証なしを付す。[H8_C][SYN]

**RZ/CX**：B1は同じ保存RZ nomineeの方向を39/40で維持した。DFのCX SEは未取得で点比較。これはmetric別の再最適化や全指標保証ではない。旧H4 legacy DF12とtol-only DF10、旧H6 DF11とDF19をpoolしない。[REAN]

## 3. Pauli：Aへの直接証拠と、別の補助問い

**Aへ直接答える既存結果**：P03.BASE / `Pauli_H4_PR_vs_deterministic`。Linear H4・1Å/STO-3G、Pauli-JW184＋constant、保存raw ray/full-H reference、T=.8/S2、central-control・P0/full wrapper・Qiskit1.3/opt1/seed17。141 PR候補から確認前固定nomineeを用い、106決定論候補の保存B0/B0I/Full PFと比較した元RZ主表は5要求精度でPR family低Gである。[PRI_PA-E3][PAULISYNTH]

その主表のε=.05はHybrid `p036_q2_r8_K2`（確認n48）対B0/B0I prefix132/q1で、元reportの表示削減 **48.42%**。ε=.0005の表示 **95.27%** は `p000_q8_r8_K4`（確認n384）対B0/B0I prefix180/q8であり、**純RTE端点**の結果。大きい後者を「非空保持の利益」と言い換えない。これらの表示は固定H4 reportの転記で、新しい削減率ではない。[H4PR_REPORT]

**最新の内部問い**：P01/P02/P04とV05は、Linear H2/H4/H6・1Å/STO-3G、Pauli14/184/918、T=.8/S2/P0・元H/state/reference・仮想compiledρ0・central/compiler条件下のHybrid対純RTE。元RZでは14/15で純RTE低G、B1の固定CXでは15条件すべて純RTE低点費用。これは強い決定論PFへの追加CX優位を検証した表ではない。H4旧primary（粗K2/.01純RTE K4）と最新内部比較（粗Hybrid K4/.01純RTE K2）を混ぜない。H6の最新Hybridはp78/q8/r32/K2、RTEはq8/r64/K2であり、旧r8を再ラベルしない。[COMPS][PAULISYNTH][REAN]

**機構材料**：P1はH6 Pauli918/T.8/S2/q8/K2のHybrid p78とpure RTEで、外側splitと可換保持内の理想PF差を区別した。H32/H64 finite項・全grid差UNRESOLVEDを残す。P2は同条件の固定confirmation/index0000でFull/R/E/Emptyの非加法性を示す。単一program価格、一般の母平均C/N/G因果分解ではない。[H6_P1][H6_P2]

**証拠と新規性**：強い決定論対照へのH4固定比較はAの直接証拠であり、純RTE対Hybridだけを理由にPauliを弱いと扱わない。一方、分割・randomized tail・既知費用則・測定会計そのものは先行研究にある。T=.8/S2、非対称な探索、確認後min/逐次延長と共有価格、仮想ρ0という範囲が残る。独立した主論文の増分性は大きな削減率だけでは決まらない。[PRIOR][STATUS]

## 4. 両者を統合して得られるもの／得られないもの

1. **共通の問いを別条件で検証**：Aに対する正負の結果・強baseline・保持の価値を、それぞれのMethodsと比較集合を伴って提示できる。
2. **既知理論に整合すること**：normalization/shot/回路価格の競合という解釈枠を共有できる。新しい統一設計理論の証明にはしない。
3. **両方を並べる編集上の意義**：Hybrid保持・randomization端点の価値が取得実装に依存する、という一般化の限界を具体的に示せる。ただし同時掲載で初めて確定した新しい因果知見があるとは主張しない。
4. **直接比較不可のもの**：DF B3とPauli純RTEの統一winner、H8対別分子、legacy DF12とP3 DF10、最新family bestの完全matched表は成立していない。
5. **交絡と限定対照**：P3は共通Gaussian保持の条件付きcase study、D2は同Pauli programのcontrol対照。H/state、split、law、basis、control、compiler、候補/cohortを固定した表現単独比較ではない。[MATCH]

S2の意味は独立な実証系列を共通の問いで読むことにある。主張が「どちらの表現が優れるか」だけなら現証拠では統合の意義が乏しく、S1またはS3の単一系列構成が明確になる。この判断はDF/Pauliのどちらにも対称に適用する。

## 5. 何を追加せず整理できるか

現在の限定結果・負結果・図表・先行差分・選抜政策は追加科学計算なしで構成できる。DF CX母平均優位、表現単独因果、普遍的selector、Pauliの時間/高次則を主張する場合に必要証拠は変わる。今回はどれも正式採択せず、その検証を自動開始しない。RPE/Track BはAの直接成功例へ混入させず、条件付き接続・別系列として扱う。`CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT`と既存STOPを保持する。

[H4PR_REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/docs/non_df_h4_pauli_pr_search_20261025.md>

[SYN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md>
[OLDMAP]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/CLAIM_EVIDENCE_MAP.csv>
[STATUS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/STATUS_AND_OPEN_QUESTIONS.md>
[CAT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2779710a86d380bb484accc51d3dccba183370b8/docs/research/o2_data_catalog_20261105/README.md>
[COMPS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2779710a86d380bb484accc51d3dccba183370b8/docs/research/o2_data_catalog_20261105/comparison_catalog.csv>
[MATCH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2779710a86d380bb484accc51d3dccba183370b8/docs/research/o2_data_catalog_20261105/df_pauli_matching.csv>
[FIG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/README.md>
[FM]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/FIGURE_MANIFEST.csv>
[DISPLAY]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/DISPLAY_DATA.csv>
[PRIOR]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/788ebad04e620055a80c75b1b3bef4a15712d4c2/docs/research/o2_claude_review_followup_20261106/PRIOR_ART_MATRIX.md>
[TRACE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/788ebad04e620055a80c75b1b3bef4a15712d4c2/docs/research/o2_claude_review_followup_20261106/SOURCE_FORMULA_TRACE.md>
[REAN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/REPORT.md>
[B1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/B1_CX_SENSITIVITY.csv>
[B2A]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/B2A_NORMALIZATION_RESULTS.csv>
[FIXED]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/B1_FIXED_COMPARISONS.json>
[SRC_LONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md>
[SRC_PF588]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md>
[SRC_B3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md>
[PAULISYNTH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md>
[PAULI_ADDENDUM]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6eb34dad982f8f753a618c609b528fb3c20d142/docs/research/pauli_h2_h4_h6_paper_evidence_addendum_20261102.md>
[PRI_PA-E3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/artifacts/non_df_h4_pauli_pr_search/2026-10-25/best_by_precision.csv>
[H6_P1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/REPORT.md>
[H6_P2]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/REPORT.md>
[H6_P1_BINDINGS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/source_and_inputs.json>
[H6_P2_BINDINGS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/source_and_inputs.json>
[P3_STAGE4]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a1d3f32aba6416fd1e200cd66a97d1a825b5dd73/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/REPORT.md>
[H8_C]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/docs/research/h8_stage_c_execution_20261105/REPORT.md>
[G01]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G01_CANDIDATE_BINDINGS.csv>
[G02]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G02_COHORT_BINDINGS.csv>
[CNG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/C_N_G_decomposition.csv>

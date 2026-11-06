# 中心課題Aの先行研究と実証差分

2026-11-06 JST。中心課題Aに沿う研究構成メモ。既存証拠の引用・対応付けのみ。新しい数値解析、詳細科学レビュー、原稿本文・Abstract・題名の執筆、Main Claim/構成の正式採択は行わない。

## 位置付けの原則

「中心的な証拠として強い」「先行との差が大きい」「論文として重要」は別の評価である。既存[PRIOR_ART_MATRIX][PRIOR]を基準にし、今回の限定的な一次資料確認は書誌・該当箇所の読取りまでとした。新たな網羅的review、全証明の再査読、優先権/世界初/科学的新規性の正式認定は行わない。

## 既知として置く部分

Hamiltonianのdeterministic/randomized tail分割、PRの基本構成、qDRIFT/RTEの資源則、composite高次構成、保持範囲の最適化、回路価格とshot負担の組合せ自体を新貢献にしない。DFへのPR適用、truncationとの比較も既存研究にある。full PFだけより強いbaselineを使う本研究の具体的比較と、「強baseline比較が一般に初めて」を区別する。[PRIOR]

## 文献ごとの重複と残る差分候補

**Günther et al., arXiv:2503.05647v2**：App.A.3 (A38)–(A40)はsigned stageのRTE合成、絶対時間に応じた配分、二乗依存と整数ceil項を直接含む。§VII.BにはDFを含むfactorized表現の議論がある。したがって高次×randomization、絶対時間二乗則、DF適用を新原理としない。文献の無限RTE normalization上界と、本研究の固定finite-Kのexact β_K・最低1 step整数配分・特定compiled候補の検証は分ける。[固定v2 PDF App.A.3/§VII.B](https://arxiv.org/pdf/2503.05647v2)

**Hagan–Wiebe, arXiv:2206.06409v3**：§5のhigher-order compositeとTrotter/qDRIFT分割・query費用上界が既知であることを確認した。本研究のfinite-RTE信号・十分shot込みcompiled費用と同じestimandではない。広い「高次と分割の共同選択」を独自性にしない。[固定v3 §5](https://arxiv.org/html/2206.06409v3#S5)

**Pocrnic et al., arXiv:2306.16572v1**：§II.2のexponential-query費用、composite分割と§Vの数値比較が先行する。分割と次数・誤差指標による競合の一般的発想が既知である一方、本研究のfinite-K/Hadamard測定/compiled価格/cohort設計は別の実証層である。文献内に見つけていない特定式を「存在しない」としない。[固定v1](https://arxiv.org/html/2306.16572v1)

**Kanasugi et al., arXiv:2603.22778v1**：§II.2.4のPR-RPE costと§Vの物理資源推定は、分割・測定・価格の共同会計が既知であることを示す。本研究の全RZ/CX countを同文献のSTAR rotation/物理資源へ換算しない。小分子finite-timeの保存比較とend-to-end energy資源推定を別にする。[固定v1 §II.2.4/§V](https://arxiv.org/html/2603.22778v1)

その他の関連文献は既存35行mapと前回調査の範囲に保持し、今回新たな広域検索・全参考文献reviewへ拡張していない。現在の論文化差分は引用した既知法の上で定義する。

## 本研究の実証差分候補

- **強い対照を含む有限比較**：Full/B0/B0I・保存高次PFを含むenvelope、精度・時間・入力/候補域による利益の変化と負結果。一般PRの成立証明とは違う。
- **finite-Kを含む実装資源会計**：bias、normalization、headroom、十分N、測定込みfull-wrapper compiled価格を元条件で結ぶ。測定会計そのものの新規性は主張しない。
- **正負と選抜過程の記録**：別価格cohort、探索nomineeと確認後min、共有価格・希少order・モデル価格の区別。証拠の信頼性を支えるが、これだけで高い科学的新規性とは言わない。
- **資源指標感度**：保存RZ選択をCXでも評価した既存B1の維持・反転。あらゆる実装・全指標への頑健性は言わない。
- **normalizationの技術整合**：B2-aの432/432再現は既存理論とSOURCEの具体的接続を支持する。二乗則の発見でもGの因果機構の完成でもない。[REAN][TRACE]

DFとPauliで、以上のどの実証差分がどの既存対照で支えられるかを[貢献整理](DF_PAULI_CONTRIBUTION.md)と[claim対応](CLAIM_EVIDENCE_LIMITATION.csv)へ結んだ。S3を大きな削減率だけで高新規性と評価せず、S1も強い証拠だけで新原理と評価しない。

## 原reviewとの版差・未確認

サーバーで参照できたのは `(1)` を含まない旧 `Claude_O2_DF_Pauli_独立科学レビュー_20261106.md` の関連節である。指定された `(1)` 版とのbyte同一性、`Claude_O2_限定追補科学レビュー_20261106(1).md` の本文は未確認。指示書には追補が案Xを撤回しO1′を推奨したとあるが、本作業はその本文を読了済みとせず、推薦の権威を構成の根拠にしない。

旧reviewが二乗構造を独自性の強い候補とした箇所は、固定Günther A.3・前回技術調査・B2-aを優先して位置付け直す。案Xの費用因果モデルを正式採択したり、B2-b未実行の結果を補ったりしない。原本は変更しない。

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

# 論文化範囲の候補

2026-11-03 JST。以下は既存証拠から作る編集・科学的メッセージの候補であり、正式採択、独立した新規性レビュー、投稿可能性の認定ではない。主張IDは[対応CSV](CLAIM_EVIDENCE_MAP.csv)。数値は元資料の保存値を引用し、今回の再解析・再ランキングはない。

## 1. 推奨候補：DFの資源・適用条件を中心にする

仮題：**有限時間信号推定におけるDF部分ランダム化の資源競争と適用条件**。

中心の問いは「partial randomizationが常に勝つか」ではなく、同じ精度会計で、保持・discard・有限RTE・強いPFのどの交換が有利になるか。DFの固定入力・候補集合・保存価格標本に限定した実装資源研究とする。

本案を推す根拠は、肯定点だけでなく、B0I追加、強い高次PF、精度・時間・geometry、別価格cohort、DF B3の負結果が、同じ問いの比較条件を具体化するためである。新しいpartial-randomization一般原理や最終energy優位を必要条件にせず、既存の完成証拠を最も多く一つの論理へまとめられる。[全体地図][LANDSCAPE]、[強いPF][SRC_PF588]、[長時間][SRC_LONG]、[B3][SRC_B3]

| 範囲案 | 主張と主要証拠 | 利点 | 制限・現証拠では書かないこと |
|---|---|---|---|
| **O1 DF中心・推奨候補** | A01–A13/F01。旧H4 development/transferからB0I、H6/H8、共通PF、長時間B2/B3へ。別価格確認とG01/G02を対応付ける | 正負を一緒に扱う適用条件の論旨。旧稿Methods・図表を活用できる | legacy DF12とtol-only DF10/19/28、状態・law・候補域を統一したscaling実験ではない。全母平均winner・全PFへの優位は主張しない |
| O0 H4固定事例に絞る | A04/A05中心、旧稿C1–C6、218候補と固定5構成移送 | 最も狭く条件を説明できる。既存v0.1の文章・図・付録をほぼそのdomainで使える | 後続強対照の存在をDiscussionへ明記する。約43.07%を研究全体の代表利得にはしない。独立resource paperかtechnical noteか未判断 |
| O2 DF＋Pauliの二つの適用地図 | O1＋P01–P05。比較taskの共通会計と、分割・law・controlの差を二部構成にする | DF B3とPauli純RTEの対照的な観測を一つの研究目的へ結べる | 方式とcompiler等が交絡する。Pauli対DFの因果説明・同一algorithm比較にはしない。全表の条件説明で本文が広くなる |
| O3 条件付きRPE接続のresource/model note | R01/R02/V1–R1。共通E0、18round分枝・成功、直接価格域と外挿・不確かさ | signal研究がenergy推定のどこまで接続したかを具体化できる | 成功は条件付き、費用優位はUNDETERMINED。無条件のenergy優位を題名・結論にできない。DF主論文の必要条件ではない |
| O4 Track Bの構成・会計note | B04/B05/B06/B11。return/local/finite-bitの構成、matched CTS、D3、分子C1 | 形式的構成とnative会計を適切な単位で扱える | 構成の独立新規性は未確立。native T/CXとDF compiled RZを合算しない。分子全task優位を主張しない |

O0は小さな資料で閉じる代替案だが、後続の負結果を伏せて古い比較集合を一般化する案ではない。O2は研究目的の幅を説明する代替案、O3/O4は別系列としての選択肢であり、同時に全稿を作る提案ではない。

## 2. O1の主張の強さ

本文で最も強く書けるのは、固定保存比較において**強い決定論対照を含めるとPRの費用方向が条件依存になる**こと。H6粗精度p7の実測、H8強化域の正負、共通PF比較、長時間でのB2とB3の差を具体例として示す。候補集合が違う結果は別panel／表へ置き、単一の「成功率」へまとめない。[H6][SRC_H6STRONG]、[H8][SRC_H8STRONG]、[長時間][SRC_LONG]

次に、finite-RTE bias・normalization・shot headroomとwhole compiled costの競合を、既存の同条件説明例で示せる。ただし未保存のdeterministic/random/basis費用分解や、qだけによる因果効果を新たに推定しない。[旧稿C4/C5][OLDAUDIT]

別価格cohortの方向確認は補強として重要だが、固定nominee・保存監査範囲に限定する。確認後の最小選抜、探索価格との共有、H8のalias、同seed継続を明記する。「すべて独立」「外部再現」「厳密な母平均最適」は使わない。[G01/G02][GREPORT]

## 3. Pauliをどこへ置くか

| 配置 | 科学的な役割 | 必要な限定と編集判断 |
|---|---|---|
| O1の補足・短い関連結果 | 保持する項とlawが異なると、同じ費用会計でも観測方向が違うことを示す | 14/15は保存有限比較・P0・仮想ρ0、H4粗精度の反例を一緒に記す。DFの効果を説明する因果対照には使わない |
| O2の独立Results節 | H2/H4/H6の内部Hybrid対純RTE、最新H6追加、N/C競合を十分に説明 | DFと別Methods・候補／選抜表・価格版を必要とする。旧H6 Hybrid水平線を最新結果へラベル変更しない |
| **別稿候補** | Pauli内部の強い比較・選抜・仮想残差条件を中心に展開 | 研究対象を明確にできるため本統合では有力な配置。ただし独立論文としての十分性・主張採用はユーザー判断 |

DF B3はone-bodyを保持し、Pauli純RTEは全非identity項をランダム化する。Pauliでの純RTEの保存低Gと、DF B3の負結果は矛盾しない。Pauli–DFの既存共通H/ray・操作的D1/D2はDiscussionの境界説明に使え、最新最良候補の完全matched表や一因子の原因証明には使えない。[Pauli統合][PAULISYNTH]、[主張図表対応][PAULICLAIMS]、[DF–Pauli][SRC_PX]

## 4. RPEをどこへ置くか

| 配置 | 利点 | 今回の証拠で守る条件 |
|---|---|---|
| 本文の最終application節 | V2の条件付き成功と、費用モデルの未確定まで含む到達点を示せる | 最終資源優位をabstractの結論にしない。V1の状態・半径、IID・alias、長q外挿、共分散UNKNOWNを省かない |
| **O1の補足・推奨候補** | finite-time研究の先に何が検証され、何が残るかを短く分離できる | V2 CONDITIONAL_SUCCESSとR1 UNDETERMINEDを併記。旧Phase3の「未実施」を最新成功検証の否定へ使わない |
| 別resource/model note | 共有較正誤差、直接価格のtuple、長q判別力を中心にできる | 現時点の点選好約4.858%は頑健な優位ではない。R2/R3は順位決定目的で現在非推奨。追加計算を自動要件にしない |

V1/V2は成功条件への接続、V3/R1は価格・不確かさを扱う。長qの物理信号評価は長qのcompile価格保証を与えない。今回これらの平均・SE・費用・順位を再計算していない。[V1][V1]、[V2][V2]、[V3][V3]、[R1][R1]、[次段判断][R1NEXT]

## 5. 新規性と旧資産の扱い

一般的partial randomization、DFの部分保持、discard比較、single-ancilla phase estimation、cost×shotsは先行研究にある。O1の寄与候補は、強い対照と実装価格を持つ条件付き適用地図、正負の境界、cohortと価格移送を明示した証拠整理に置く。「初」「一般最適」「保証資源優位」は採らない。[旧先行研究対応][PMMAP]、[Güntherら固定v2](https://arxiv.org/abs/2503.05647v2)

旧v0.1のC1–C6は元domainで保持する。新しいAbstractへ広い成果を入れる場合は、Methods、cohort、比較集合、Resultsもそれに対応する新構成へ改める。旧図や元結果を上書きせず、再利用版・新しい表の設計を分ける。[構成案](PAPER_STORYLINE_OUTLINE.md)、[旧本文][OLDMAIN]、[旧補足][OLDSUPP]

## 6. ユーザーへ戻す判断

現段階で必要なのは、O1/O0/O2の中心範囲、Pauliの配置、RPEの配置、Track Bを別系列へ分けるか、現在の有限・条件付きclaimを論文のどこまで正式採用するか、という判断である。投稿先・独立paperかnoteか・厳密保証へ強めるかも未決定。

Codexは本資料を公開・照合してSTOPする。選択後の完成原稿執筆、独立詳細レビュー、必要な場合の限定資料回収・科学検証は、それぞれ選んだ主張に合わせた次の指示で扱う。現在の限定claimのために全SERVER再棚卸し、RPE R2/R3、DF B3、H8残余samplingを一律に要求しない。

<!-- fixed-reference-definitions -->
[GREPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/REPORT.md>
[LANDSCAPE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md>
[OLDAUDIT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/manuscripts/track_a_resource_study_claim_audit_v0_1.md>
[OLDMAIN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/manuscripts/track_a_resource_study_v0_1.md>
[OLDSUPP]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/manuscripts/track_a_resource_study_supplement_v0_1.md>
[PAULICLAIMS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_claim_figure_map_20261101.md>
[PAULISYNTH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md>
[PMMAP]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_post_pm2_claim_evidence_map.md>
[R1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fde197c64da671f1d428f0d3e8fdb762dc21930/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md>
[R1NEXT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fde197c64da671f1d428f0d3e8fdb762dc21930/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/MINIMAL_NEXT_ACTION.md>
[SRC_B3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md>
[SRC_H6STRONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md>
[SRC_H8STRONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md>
[SRC_LONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md>
[SRC_PF588]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md>
[SRC_PX]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_comparison_synthesis_20261025.md>
[V1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md>
[V2]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md>
[V3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc60beb51930d4311d2ded84440682e583ccafc7/docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md>

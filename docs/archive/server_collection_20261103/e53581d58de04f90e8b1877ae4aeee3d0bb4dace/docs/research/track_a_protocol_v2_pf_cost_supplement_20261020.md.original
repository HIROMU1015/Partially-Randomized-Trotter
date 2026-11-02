# 共通実験プロトコルv2：PF単価置換の補足証拠

2026-10-20。補足対象は既存結果commit `85433c48fd4c37d4cb4594391da422558ac3ce7f`、科学SOURCE `19f2c5a9409db69fd8bd6500eb4228fefa00145c`。
[採用済みプロトコルv2](track_a_pr_resource_evaluation_protocol_v2.md)の§1/3/4/6/7/8に従い、条件付きのPF単価感度診断として整理する別資料である。プロトコルの比較条件・科学契約・掲載主表の変更、新規科学実行、最終freezeではない。

## 参照資料と費用指標

[元報告](track_a_pf_selection_cost_comparison_20261020.md)、[全70件の保存表](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/conditions70.md)、[保存summary](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/summary.json)を参照する。
Linear H6/H8、隣接1 Å、STO-3G、tol-only DF19/28、T=.8、P=0。状態準備を除くfull cosine/sine Hadamard wrapperのcompiled RZ×保存analytic shotを両軸で足す指標である。PRは決定論的prefix、確率的RTE tail、full wrapperを含む旧確認cohortの保存平均・SE。H6 union8/H8 union6各n32であり、独立確認Pのn48/n104や探索n8へ移送しない。
基準側は過去のモデル推定と既存実測が混在した政策。450新実測軸＋過去公開8軸へ単価を置換し、元から実測102軸・shot・適格性・候補集合・PR選択/費用/SE/cohortを保持した。純モデル反実仮想との比較ではない。

## 保存結果から引用できること

| 保存項目 | 結果 | 読み方 |
| --- | ---: | --- |
| 条件数 | 70 scope / unique30 endpoint | H6 signed30、H8両bias40、scopeと倍率は独立反復ではない |
| 適格PF費用の基準再現 | 2,594件完全一致 | 元保存結果の再現gateであり、母平均保証ではない |
| 最良PF候補変更 | 2/70 | H8非相殺ε=.0005、strengthened184、倍率1/10、B0I p16→p15、q4/S4保持 |
| q / PF次数変更 | 0 / 0 | 対象条件内の保存点選択に限定 |
| 最良PF総費用変更 | 30/70 | 小さい単価差と候補順位差を区別する |
| PR/PF方向逆転 | 0/70 | PR低費用60/PF低費用10、保存n32の点比較 |
| 候補pairの厳密逆転 / tie変化 | 146 / 8 | scope・倍率に重複があり、独立現象の件数ではない |

H6 ε=.005のB0_p11_q4_2ndは最良選択を維持した。これらは保存summary/報告の引用であり、本資料作成時に費用・差・比・順位を再計算していない。

## 図と掲載用caption

[PF費用変化 PNG](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/PF_cost_change.png)、[SVG](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/PF_cost_change.svg)、[PDF](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/PF_cost_change.pdf)。

**Supplementary figure caption — PF selection under saved unit-price replacement.** Minimum PF costs for 70 overlapping scopes of linear H6/H8 at 1 Å in STO-3G with tol-only DF ranks 19/28, T=0.8 and zero charged state-preparation cost. Costs sum saved cosine/sine analytic shot counts multiplied by compiled full-Hadamard-wrapper RZ counts. The historical mixed measured/model price policy is compared with saved direct-price replacement, with eligibility and candidate sets fixed. The selected PF changes in two H8 noncancellation scopes; q and PF order do not change. Profile/runtime differences, the S4 composition hypothesis and missing initial-run final gate remain qualifications.

[PR/PF比 PNG](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/PR_PF_ratios.png)、[SVG](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/PR_PF_ratios.svg)、[PDF](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/PR_PF_ratios.pdf)。

**Supplementary figure caption — Conditional stability of saved PR/PF point-cost directions.** Ratios use unchanged saved PR confirmation costs from H6 union8 or H8 union6, each with 32 paired trajectories, divided by the selected PF cost before and after unit-price replacement. The direction does not reverse in the 70 evaluated scopes, representing 30 unique system/bias/accuracy/allowance endpoints. This sensitivity diagnostic does not establish a population-mean ordering, robustness outside these scopes or equivalence to the independent P cohorts. The 30 unresolved H6 noncancellation scopes are excluded and remain unresolved.

## 主張境界

H6 profile差、H8/model runtime同一性未証明、S4合成仮説、初回572軸の最終gate欠測（使用新450軸中428軸）、旧profile待ち48軸の条件を維持する。モデル固有誤差と環境差を分離できず、2%単価検証から全順位不変を導かない。
PR有限標本の平均/SEを母平均・未観測希少event保証へ変えない。shot込み平均費用とn本を実行する古典MC取得合計費用は別である。新PR平均/SE、信頼区間、追加標本設計を作っていない。
H6 noncancellation30件は[原因調査](track_a_h6_noncancellation_investigation_20261020.md)のとおり未解決。後続Dのunresolved空配列は解決証拠として使わない。
既存主表の独立P6条件・Figure2・Methods/CEM/evidence ledgerは不変。今回の診断は追加の独立sampling証拠ではなく、同じ保存資料に条件付けた補足である。COMMON_CORE_READY、詳細科学レビュー、最終evidence freezeを認定しない。

機械可読対応は[補足manifest](../../artifacts/resource_applicability/track_a_h6_noncancellation_investigation/2026-10-20/protocol_v2_supplement_manifest.json)と[claim表](../../artifacts/resource_applicability/track_a_h6_noncancellation_investigation/2026-10-20/paper_claims.csv)。数値参照先のSOURCE/結果SHAは元packetを継承し、今回新しい科学結果を生成していない。

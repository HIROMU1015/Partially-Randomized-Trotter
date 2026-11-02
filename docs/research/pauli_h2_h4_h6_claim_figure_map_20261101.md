# Pauli-JW H2・H4・H6：主張候補・一次証拠・図表の対応

2026-11-01。[論文用統合資料](pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md)の編集用対応表である。主張、タイトル、図番号、配置、captionは候補であり、中心claim・新規性・投稿方針の正式採択ではない。Pauli-onlyの既存証拠を使用し、新科学計算・数値再解析・作図データ生成は行わない。

## 1. 主張候補と直接証拠

| ID・主張候補 | 直接支持する固定証拠・保存行 | 本文／補足で対応する資料 | 必須の限定・追加検証なしでは言えないこと |
|---|---|---|---|
| C1：有限保存比較で純RTEは14/15精度条件においてHybrid PRより低G | H2 [E4][E4]・[比較CSV][h2comparison]、H4 [E5][E5]、H6 [E6比較][E6compare]の5精度ごとの既存判定 | 本文Table 1、Supplementary Table S1 | 仮想ρ=0・有限候補・確認集合min・有限価格標本。15独立実験、母平均winner、普遍的方式優位とはしない |
| C2：H4の.05はHybrid低Gという例外 | E5の.05：p036_q2_r8_K4対p000_q8_r8_K2 | 本文Table 1、Fig. 1候補のH4図 | 旧探索nominee K2と確認後min K4を区別。微小なK差の母平均順位を確定しない |
| C3：Full PFだけでなくB0/B0Iを含めた決定論的対照が重要 | [E1][E1]・[E2][E2]。H2の保存代表はFull PF、H4/H6はB0。H2高精度B0は適格登録候補なし | 本文Table 1、統合§5の保存PR/D表 | B0/B0Iを独立方式として二重計上しない。Full PF比とfamily最良比を区別。最新候補の未保存倍率は新計算しない |
| C4：H4のPR探索主表は全5精度で保存決定論的familyより低G | [E3][E3]のselected_PR、PR_over_best_deterministic_exact。保存削減率48.42–95.27% | 本文Results§9.2、旧倍率のSupplementary表 | 高精度4条件のPRは純RTE境界。Hybrid自体の優位と呼ばない。E3 nomineeと現在の事後minを混同しない |
| C5：H2の更新は既存s4候補の追加独立確認による | [H2報告][H2report]・[coverage][h2coverage]・[E4][E4]。Phase Aは保存target/N再利用 | 本文Results§9.1–2、Fig. S1 | 大event探索による成功ではない。確認後選抜・逐次延長を補正した母平均保証ではない |
| C6：H4ではs128/256のC増加がshot節約を上回った | [event CSV][eventcsv]のq8/K2/K4、[H4図][H4g] | 本文Fig. 1、補足Fig. S2 | s256は探索価格。s64の確認集合minとK2系列の代表表を区別。連続域最適性は未証明 |
| C7：H6純RTEはs512で低G、s1024で再上昇 | 同CSVのH6/q8/K2、[H6拡大図][H6g]、s512確認n48 | 本文Fig. 2、Table 1 | 隣接s128/256/1024は探索n8。最小点・単峰性・母平均最適性の保証ではない |
| C8：H6 Hybridのevent拡張後も純RTE低G | [E6][E6]・[E6compare][E6compare]。prefix78/q8/K2 r32がnominee/確認後min、r64でG上昇 | 本文Table 2（統合§6.2）とTable 1 | 旧H6図のHybrid線は拡張前。r32/n64の最新比較は表で示す。全prefix/q/r/Kの対称探索ではない |
| C9：費用差はBだけでなくbias/headroom、N、Cの競合に対応 | [原因調査E7][E7]§2–3、統合§6の原文転記 | 本文Fig. 3（H6 B/N/C）、Table 3（代表内訳）、Supplementary bias/h表 | H6厳の小bias→広いh→小Nは会計上の説明。prefix/tail/law/compilerの独立因果寄与率ではない |
| C10：control/compileは回路価格を変え得る | [Pauli D2対照][D2]・[比較CSV][D2csv]のPauli PF/PR index0 | 必要時のみ補足Table S3。本文機構説明はE7参照 | 同一bodyのgeneric/central対照。最新純RTE対Hybridの直接原因ではない。本統合ではDF数値を再掲しない |

## 2. 本文に置く図表の候補とcaption草稿

既存の図はPNGを参照し、同じ固定directoryのPDF/SVGを原本として利用できる。新しい数値図を必要条件にしない。以下の図番号は暫定である。統合資料は原本を変更せず固定リンクを指す。

### Table 1：四方式の最新保存G

**役割**：H2/H4/H6×5精度にFull PF・B0/B0I・Hybrid PR・純RTEを併記し、Hybrid/RTEの既存判定を示す。統合§4のTable 1を使用する。候補ID・q/r/K・cohort/n・G SE・結果commit/科学SOURCEはTable S1と統合§3へ委ねる。

**Caption草稿**：線形H2、H4およびH6に対する四方式の保存条件付き資源G。共通条件は隣接1.00 Å、STO-3G、中性singlet、S2、T=.8、central-rotation制御、Qiskit1.3.0/opt1/seed17、状態準備除外、仮想ρ=0である。決定論的二方式は既存の有限family選択値、Hybrid PRと純RTEは各段階の保存確認集合選択値を引用した。H6 Hybridは10/27の追加4候補を含む。H2のB0欠測は適格登録候補なしを示し、0/∞で補わない。各分子の5精度は同じ価格標本を共有し、15行を15独立実験と解釈しない。四方式の新しい再ランキングではない。

### Fig. 1：H4のG対event数

原図：[H4_G_vs_s.png][H4g]。関連データ：[event_cost_trends.csv][eventcsv]。既存5精度panel・K2/K4・探索/確認の描き分けと保存2SE表示をそのまま用いる。

**Caption草稿**：H4純RTEの保存条件付きGと総event数sの関係。q8を固定し、K2/K4の探索価格と独立確認価格を別系列として表示した。点線は保存Hybridの確認値であり、.05のHybrid低Gと他4精度の純RTE低Gを示す。s128/256でBとNが低下しても単回C増加が上回る。点を結ぶ線は視覚的な補助で、連続域の単峰性や大域最適性の証明ではない。価格標本の2SE表示は希少費用tailや選抜後同時coverageの保証ではない。

### Fig. 2：H6純RTEの大event領域

原図：[H6_G_vs_s_extension_zoom.png][H6g]。関連データは同じevent CSV。**図のHybrid点線は10/26の旧n48参照であり、最新p78/q8/r32/K2・n64を表していない。** 最新Hybridとの比較はTable 1/2を必ず併記する。既存画像の線を最新値と読み替えない。

**Caption草稿**：H6純RTE、q8/K2のs64–1024における条件付きG。s512は別seedの確認n48を持ち、s128/256/1024は探索n8である。図中のHybrid水平線はevent拡張時点の旧保存確認値で、後続Hybrid r32の値はTable 1/2に示した。純RTEはこのgridでs512の低Gを示し、s1024で再上昇した。仮想ρ=0・有限価格標本の結果であり、母平均tail・全パラメータ最適性は未保証である。

### Fig. 3：H6のB/N/Cトレードオフ

原図：[H6_B_N_C_vs_s.png][H6bnc]。横軸s、左B、中央単回RZ、右N_cos+N_sinの既存panelを再利用する。**biasを描いた図ではない**ので、H6高精度のheadroom説明にはTable 3と統合§6.3の保存bias/hを併記する。

**Caption草稿**：H6純RTEのevent数に伴う保存normalization B、単回価格Cおよび条件付き必要shot数Nの変化。event数増加によりB/Nは低下しCは増加する。Gの最小点はこれらの競合と候補固有biasを含む会計に依存する。有限grid上の探索と確認を区別し、Bの低下だけを総費用差の因果説明とはしない。右panelの5精度は共通価格標本を共有する。

### Table 2 / Table 3：追加Hybridと代表費用内訳

Table 2は統合§6.2のp78/q8/K2、r8/16/32/64表、Table 3は統合§6.3の代表N/C表を使用する。新しいHybrid plotは作成しない。

**Table 2 Caption草稿**：H6 Hybridの固定prefix78/q8/K2系列における最厳精度.0005の保存費用。r8は旧確認n48、r16/32/64は新確認n64であり、r32の探索nomineeと確認集合minが一致した。rの変更はevent law・角度・B・bias・compiler入力も変更するため、保持部分だけの因果効果を測る比較ではない。全5精度の最新Hybrid/RTE比較はTable 1に示す。

**Table 3 Caption草稿**：H2/H4/H6の代表精度におけるB、軸別N、CおよびGの保存内訳。H2粗/H4厳では純RTEの大きいshot負担を低いCが補い、H4粗ではHybridのN節約が勝る。H6厳では純RTEの小biasと広いheadroomにより、大BでもNが小さい。headroomの保存値は補足表に示す。B、bias、law、回路実装の各因果寄与率を分離したものではない。

## 3. Supplementaryに適する既存資料

| 補足候補 | 使用する既存図表・固定証拠 | 読者へ示す内容／適用範囲 |
|---|---|---|
| Table S1：全5精度のcandidate/cohort/n/G/SE | 統合§4の60行の追跡表、[E1][E1]・[E2][E2]・[E4][E4]・[E5][E5]・[E5b][E5b]・[E6][E6] | 保存family選択と未適格セルを追跡。新しい母平均順位なし |
| Fig. S1：H2の確認拡充と大s探索 | [H2 G図][H2g]・[H2 B/N/C図][H2bnc]、[coverage][h2coverage] | s4の既存候補確認と、改善しなかったs64/128探索を分ける。同じsでもq/rの違いを保持 |
| Fig. S2：H4 B/N/C | [H4_B_N_C_vs_s][H4bnc] | B/N減少とC増加。厳精度の確認集合minはs64/K4であり、K2代表表と区別 |
| Fig. S3：H6全s図 | [H6_G_vs_s][H6gfull] | 旧小s領域からの変化。Hybrid参照線は拡張前のため最新比較はTable 1/2へ誘導 |
| Table S2：過去PR対決定論的倍率・選抜規則 | [既存統合CSV][oldratio]、[E3][E3]、統合§5 | H4 nominee対確認後min、H2/H6追加前後の版差。旧比を最新候補へ流用しない |
| Table S3：Pauli control実装の補助証拠 | [D2比較CSV][D2csv]のPauli PF/PR index0 | generic→centralでPF255500→15028、PR0 9652→1145 RZ。最新family差を説明する寄与率ではない |
| Table S4：bias/u/headroom・保証範囲 | [E7][E7]§2、[H2 verification][H2verify]・[event verification][extverify]・[H6 verification][H6verify] | 理想law包含と実ρ、標本SEと母平均tailを分離。過去のPASSを今回の検証実施として表示しない |

PNG/PDF/SVGの存在と今回目視したG図を確認した。既存のH2/H4/H6図でevent依存と探索/確認の区別は十分に提示できる。一方、**最新四方式を一枚に描いた新しいplotや新しいHybrid拡張plotを今回作ったとは表示しない**。Table 1/2で必要な最新値を提示できるため、それらの新plotや再解析は完成条件ではない。

## 4. 既存資料からの再利用・今回補完した部分

[旧H4 paper plan][oldplan]から、条件付きGと単回Cを分ける構成、Methods/Results/captionの分担、本文とSupplementaryの分け方を再利用した。旧Methodsの記述枠組みは[保存草稿][oldmethods]を参照した。旧資料のPauli–DF比較・6系列の数値・旧固定prefix27の性能主張は、本Pauli-only統合へ持ち込んでいない。

[旧H2/H4/H6統合][oldintegrated]と[10/26統合][E8]の保存表・SOURCE索引を再利用し、今回必要なH6追加Hybrid、H2追加確認、H4例外・選抜差、11/1原因調査の説明段階を接続した。過去版は上書きせず固定commitへ参照する。旧D1停止と後続D1/D2の証拠範囲も原因調査E7の区別を保持する。

新しい管理manifest、科学CSV、検証frameworkは追加していない。文書専用 `one`・`source_rows`・`check_links` を再利用し、数値tokenを文字列のまま転記した。今回の検査は一次資料23ファイルのGitHub/local blob一致、表の候補・値・cohortの原文一致、固定参照・図fileの存在、内部絶対path/秘密情報、変更範囲、公開本文/SHAの技術照合である。過去の科学/人工テストを再実行せず、独立科学レビューとも呼ばない。

## 5. 未採択の論旨・タイトル候補と次の判断

作業題候補は「Pauli-JW時間発展における保持・切り捨て・有限ランダム化の条件付き資源比較：H2・H4・H6」または「線形水素分子のPauli-JW時間発展における回路価格とshot負担の競合」である。いずれも未採択で、新規性・先行研究・投稿先の評価は実施していない。

現在の証拠は、有限候補の条件付き費用比較と、分子・精度ごとに異なるN/C競合を記述する論旨を支える。prefix除去のみを単一原因とする論旨、全分子のサイズ則、母平均・全law実精度・FT-QPE/RPE end-to-end優位は追加検証なしでは主張できない。追加科学計算を行わず限定したResults/Methods草稿を構成できることと、中心claimを採択する判断を区別する。

次の担当は**ユーザー/GPT**：本文で採用する主張候補・図表配置、中心claimや詳細レビューの要否を判断する。**Codexの次の実行作業なし。公開・本文/remote SHA照合後mandatory STOP**とし、新科学計算、原因検証、DF研究横断統合、既存論文本体の変更に自動移行しない。

[E1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/artifacts/pauli_h2_h6_comparison/2026-10-25/H2_H6_best_comparison.csv
[E2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbfbfd21a252a4a32f52e3b40e50acd91b3c626a/artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/best_by_precision.csv
[E3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/artifacts/non_df_h4_pauli_pr_search/2026-10-25/best_by_precision.csv
[E4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/candidate_precision_table.csv
[E5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/selected_cost_breakdown.csv
[E5b]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3/artifacts/pauli_hybrid_rte_comparison/2026-10-25/cost_breakdown.csv
[E6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CANDIDATE_RESULTS.csv
[E6compare]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/COMPARISON_RESULTS.csv
[E7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5432857b94c3714ea9f4d6031e493e8dfc308a91/docs/research/pauli_pure_rte_cost_mechanism_inventory_20261101.md
[E8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md
[oldintegrated]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/pauli_h2_h4_h6_integrated_comparison_20261025.md
[oldratio]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/H2_H4_H6_optimized_PR_comparison.csv
[oldplan]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/non_df_h4_paper_plan_20261025.md
[oldmethods]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/non_df_h4_paper_methods_DRAFT_20261025.md
[H2report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/docs/h2_pauli_rte_validation_20261026.md
[H2verify]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/verification.json
[extverify]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/verification.json
[H6verify]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/verification.json
[H2g]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/figures/H2_G_vs_s.png
[H2bnc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/figures/H2_B_N_C_vs_s.png
[H4g]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H4_G_vs_s.png
[H4bnc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H4_B_N_C_vs_s.png
[H6g]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H6_G_vs_s_extension_zoom.png
[H6gfull]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H6_G_vs_s.png
[H6bnc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H6_B_N_C_vs_s.png
[eventcsv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/event_cost_trends.csv
[h2comparison]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/comparison_by_precision.csv
[h2coverage]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/old_candidate_confirmation_map.csv
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/non_df_h4_d2_execution_20261024.md
[D2csv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/artifacts/non_df_h4_d2_execution/2026-10-24/comparison_table.csv

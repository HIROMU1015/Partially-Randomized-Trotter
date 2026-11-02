# 図表配置案と保存データの対応

2026-10-26。配置・caption案のみ。新規図、再plot、集計、ranking、fit、SEや比の計算は行っていない。既存図の数値・label・旧注記は原本のまま保持した。Main/Supplementaryは推薦で、正式採択ではない。

本表の番号は提案論文内の番号であり、原資料のFigure2や既存原稿の図番号を変更しない。科学的scopeは[証拠対応表](EVIDENCE_MATRIX.csv)、節の狙いは[論文構成](PAPER_OUTLINE.md)に対応する。

## 1. 主図候補

### Figure 1 — 研究対象と評価の流れ

**Section：** Introduction／Methods。**目的：** 4証拠ブロックを同じGの問いで接続しつつ、異なる入力・law・controlの実験を混同させない。

**内容案：** 保存H/state/reference→表現（DF／Pauli）→family（Full／B0／B0+I／Hybrid／純RTE）→PF公式・q/δ・prefix・RTEパラメータ→signal bias/B/u/ρとcompiled価格C→shot N→条件付きG。探索→固定nominee→独立確認と、確認集合内minを別枝で描く。DF長時間のL_D=0は未評価と表示する。

**元資料：** [DFデータ体系][INDEX]、[Pauli総合資料][PS]、[長時間REPORT][LT]、[固定H4比較][FIX]。入力図の科学SOURCEは各studyのものとし、この概念図に一つの実行SOURCEを割り当てない。

**Caption案：** 「各枝は保存された別studyの評価経路を表す。全経路を同一のHamiltonian、random law、cohortで実施した意味ではない。Nはanalytic shot会計、nは古典的価格標本数である。」

**推奨：** Main。既存の一枚で全要素は表現できないため、今回は構成要素だけ記述。概念図の将来作成に数値再解析は不要。今回新図生成なし。

### Figure 2 — DF-nativeの対照強化と費用分解

**Section：** 3。**目的：** 強いPF対照、shotと単価の釣合いを示す。

**既存図：** [原2A PNG][F2A]／[SVG][F2AS]、[原2B PNG][F2B]。結果snapshot `eb33de67475e7d33eed56801b74a2a277147e4be`、render SOURCE `3de17acd`。原34ファイルと数値116点は保全。[更新案][F2PLAN]のscopeを使用する。

**データ：** Aの固定6比較は[main_table_proposed.json][MAIN]と同MarkdownをTable2へ置く。新しいH4 DF10/H6 Pを原2Aへ貼り替えない。既存CEM v3／ledger v5のFigure2原本参照も[DF索引][INDEX]から追える。

**伝える結果：** 2Aはlegacy H4 DF12を含む探索・旧確認の18比較、2BはH8独立n48の2比較である。A82説明因子の後続完成と、原図描画時の未完了注記は別の時点である。

**Caption案：** 「(A) legacy H4 DF12/H6 DF19/H8 DF28の保存探索・旧確認。H4 tol-only DF10の新独立Pではない。PFの塗りつぶしは直接実測、白抜きはmodel推定であり、全点実測とはしない。(B) H8独立n48、同じcohortに保存された分解。analytic shots×compiled RZ、P=0。母平均優位やenergy/RPE全体を保証しない。H8詳細監査は公開receiptから再検証できていない。原図内のA82説明未回収の文言は描画時点の履歴であり、後続の因子保存完了と区別する。」

**推奨：** Main候補。旧図だけでこの限定scopeは表現可能。新Pへの数値置換には別の保存軸集計・因子計算が必要だが、今回必須ではなく実施しない。原図の古い注記を消す編集も未実施。

### Figure 3 — 長時間・高次PF/PRの費用と選択刻み幅

**Section：** 4。**目的：** Tの増大と精度によって、少ないq／大きいδを使う高次方式が低い取得域Gを得る条件を示す。

**既存図：** [cost_vs_time][LTCOST]、[selected_delta_vs_time][LTDELTA]、[selected_formula_grid][LTGRID]。各PNGに同名SVG/PDFがある。結果commit `155529e20a8fac39e00786c40b57830b71203c4f`。signal SOURCE `e417ffc9f`、native SOURCE `27ebabdd2`、q32拡張 `206936049`、後続会計の実ファイルhashは[completion_provenance.json][LTPROV]。

**既存データ：** [best_configurations.csv][LTBEST]、同JSON、signal_feasibility.csv。価格の探索点最良とsignal適格性の最小qを別にする。

**伝える結果：** 厳精度・大Tでのm5 q1、緩精度のS2。r/Kはgridの保存labelで、prefixはCSV／Table3から参照する。

**Caption案：** 「H4/H6 linear1Å/STO-3G、tol-only DF10/19、T=.8～2、P=0。点と1SEは価格取得済み適格subsetの探索値。11,632 signal構成の全域価格最適ではない。q32はS2のT1.6/2だけ追加。δはouterのT/qで、signed高次pieceの大きさとは別。独立確認は別表、例外1件を含む。」

**推奨：** Main。3図をそのまま別panelまたは組図として配置可能。重ね描き・再集計は不要。紙幅に応じてδ図をSupplementへ移す案もある。新図合成やcaption焼き込みは今回未実施。

### Figure 4 — Pauli純RTEのevent数と測定負担

**Section：** 5。**目的：** H6の旧Hybrid優位からの逆転を、B/N/C/Gの競合と比較集合付きで示す。

**既存図：** [H6_G_vs_s_extension_zoom.png][H6G]、[H6_B_N_C_vs_s.png][H6BNC]、必要に応じて同directoryの`H6_G_vs_s.png`。結果 `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`、科学SOURCE `e755ecf581600192fd2745d7066e410cebd933d3`。全図にSVG/PDFあり。

**既存データ：** [event_cost_trends.csv][EVENTCSV]、[best_by_precision.csv][EVENTBEST]。全分子・全精度の最新比較は[latest15.csv][LATEST]をTable4へ置き、図に無理に全15値を重ねない。

**伝える結果：** H6 s64→512でshot節約が単価増を上回り、s1024では探索Gが増す。H2の変化はs拡張ではなく有望候補の確認coverage、H4はs128/256で非改善。これらを一つの普遍的s最適則にしない。

**Caption案：** 「固定Pauli-JW H6、T=.8、S2、central control、q8、L_D=0。主Gは仮想ρ=0。n8探索とn48確認、保存2SEのerror barを区別する。線は視認用の連結でfitではない。s512の独立価格でも保存Hybridより低い点Gだったが、Hybridは旧保存familyを固定し、新r>8のmatched Hybridは未取得である。母平均winner・全域最適性・実compiled精度を保証しない。」

**推奨：** Main。H6の完成2図だけで目的を表現でき、追加数値再解析は不要。H2/H4図はSupplement、latest15はMain表。最新3分子を一枚に再plotする案は今回は採用せず、実行もしない。

### Figure 5 — 同じH4 bodyにおけるcontrol実装差

**Section：** 6。**目的：** Pauli／DFの差を表現だけに帰属できないことを、同body対照で示す。

**既存図：** [同一Pauli trajectoryのgeneric/central価格図][CONTROLFIG]と[両軸paired価格図][PAIREDFIG]を優先。元A_COHORT_COS SOURCE `34ef0a7ca2412a61f5215a4ce1192e99f3226b2a`、A_AXES `a37232f85812688c3b9ab957c5073c5ce143b17e`。固定結果引用snapshotは `03efd057a7e77c40ae7e90b9c77885d6606e793b`。

**既存表：** [D2 comparison_table.csv][D2]（SOURCE `74515d8eb4f073c40fc783681a3775dbca89e9b1`）、[6系列単価CSV][COST6]。D2 PF255500→15028、PR0 9652→1145と、n256平均を別欄に置く。

**Caption案：** 「fixed H4、同じ保存Pauli bodyをgeneric／central controlでcompileした対照。control構成と固定compiler応答を合わせた効果であり、特定passだけの寄与率ではない。n256は同じPauli trajectory IDのpaired対照。DF価格は異なる分割・lawの別cohortで、Pauliとのpaired差を計算しない。」

**推奨：** Main。既存control図＋D2小表で示せる。6系列を一枚にした新図は未生成だが、既存単価表で十分であり、追加集計は不要。generic系列の図をcentral全結果の図として再ラベルしない。

## 2. 主表候補

|表|Section・科学目的|使用する保存資料／結果commit|伝える内容・caption条件|Main/Supplement・追加再解析|
|---|---|---|---|---|
|Table1 条件対応|Methods。入力と比較可能性の境界|本EVIDENCE_MATRIX、[DF索引][INDEX] `731b3d6f`、[Pauli総合][PS] `b6c89969`、[固定H4][FIX] `03efd057`、[長時間][LT] `155529e2`|分子・geometry・DF12/10/19/28・M・state/reference・T・law/control・費用政策。unknownを推測しない。|Main。文書転記のみ。|
|Table2 固定独立確認|§3。正式採用された点方向と負の結果|[6行主表JSON][MAIN]、H4 `821cc0f6`／H6 `3cccfcb2`／H8 `e6eea538`。SOURCEはCSVに記載|保存exactG/SE/n/PF ID。H4粗精度非再現、H6粗精度PF低費用再現。H8 aliasと公開監査制約。|Main。既存表のみ。新削減率・CI不要。|
|Table3 長時間の構成と独立確認|§4。探索点最良と確認結果の区別|[best CSV][LTBEST]と[independent_comparisons.csv][LTIND] `155529e2`|T2厳精度のq/δ/prefix/r/K、39候補1,295標本、59比較中58方向再現、H4 T1.6 ε.005例外。|Main抜粋＋全表Supplement。新minなし。|
|Table4 Pauli最新15条件|§5。Hybridと純RTEを直接比較|[latest15][LATEST]は`b6c89969`の引用表。H2原結果`4aea054b`、H4/H6`10cb3e5c`|exactG、n、SE、候補ID、保存判定。14/15は共有価格を使う条件行、確認後集合minを含む。PF/B0は過去対照のまま。|Mainまたは5精度をまとめたMain＋exact全表Supplement。再ランキングなし。|
|Table5 固定H4の単価と条件付き会計|§6。control差とG条件の区別|[6価格][COST6]、[30会計引用][CNG30] `03efd057`、D2|単価n256とsingleton、ρ0のN/G、guardと実ρの違い。一般的Pauli優位を主張しない。|単価はMain、G/guard全行はSupplement。追加再解析なし。|

## 3. 補足図表候補

|補足|対応section・目的|保存資産・固定結果commit|必須注意点|既存資産だけで可能か／追加数値作業|
|---|---|---|---|---|
|S1 B0/B0+I geometry|§3。強いbaselineが点winnerを変える|`288d2206`内`artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/figures/figure_2_measured_cost_ratio.png`。科学取得`98cdb35c`、SOURCE `924b779c`|legacyDF12・6geometry・ordinary control。DF10へ移送しない。|既存PNG/SVG使用可。元の302精度点を新格子へ投影しない。|
|S2 共通588候補|§3。全公式vs同公式、強い分母|`90f1cd55`内`artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/comparisons_60.csv`と`precision_rankings_15.json`|旧PR探索、有限grid、決定論価格。独立Pと呼ばない。|既存CSV/JSONの表で可能。新rank作成なし。|
|S3 標準Y4固定独立比較|§3/4。異q比較と同q対照|`4d117041`内`artifacts/resource_applicability/track_a_h46_standard_y4_confirmation_recovery/2026-10-24/results.json`、採択`09e71910`|ε.001、S2q8/Y4q4、sameqでFull/PR逆転なし。次数だけの因果効果ではない。|既存2比較の表。新図なし。|
|S4 同q/prefix・r/K|§4。探索条件と費用coverage|`155529e2`内長時間`same_q_prefix_R32_K4.csv`、`r_K_sensitivity.csv`|未計測Gは空欄。signal全域の情報を費用全域最適性と混同しない。|表の再利用可。新r/K集計・再plotは今回なし。|
|S5 固定δのT依存|§4。刻み幅固定時の負担|[fixed_delta_time_dependence.png][LTFIX]、同SVG/PDF、`fixed_delta_comparisons.csv`、`155529e2`|δ.2、PR H4prefix3/H6prefix5。最良調整域の図ではない。|既存4panelをそのまま使用。追加計算不要。|
|S6 C/N/G分解|§4。shot節約が単価増を上回る条件|長時間`C_N_G_decomposition.csv` `155529e2`、DF索引のCNG22/SHOT88|保存比と絶対単価を区別。異studyの倍率は積算しない。|保存表のみ。新しい共通分母は作らない。|
|S7 独立確認と不確かさ|§4/5。nomineeと確認後min|長時間59行、H2`old_candidate_confirmation_map.csv`／`comparison_details.json` `4aea054b`、event拡張`comparison_details.json` `10cb3e5c`|逐次延長・価格標本共有・近似区間・非再現1件。全行を独立実験数としない。|既存表・判断記録。新CI/p値なし。|
|S8 H2/H4 event非改善|§5。一般的な大s優位ではない対照|[H2_G_vs_s.png][H2G]／[H2_B_N_C_vs_s.png][H2BNC] `4aea054b`、[H4_G_vs_s.png][H4G] `10cb3e5c`|H2s64/128は探索n8、H4旧s64と新拡張のcohort差。|完成PNG/SVG/PDF。新数値作業不要。|
|S9 rare-event・law|Methods/Discussion。未観測を0にしない|長時間`confirmation_costs.json.gz`のrarity `155529e2`、固定H4`rare_cost_sensitivity.json` `03efd057`|K2のorder4はsupport外。K4/K6未観測費用は未知。全law母平均Gを保証しない。|保存欄・既存記述で可能。新stratum推定はしない。|
|S10 局所ρとguard|§6。保証scopeの階段|固定H4`conditional_resources_QUOTED.csv`、`six_local_results.json`、R2-C報告 `03efd057`|選択6件と全law、CとG、guard予算と実残差を分ける。|保存表のみ。局所値を新候補へ転送しない。|
|S11 条件付き母平均C|§6。限定した十分条件の保存証拠|[scalar結果][MEANC] `07d4f6ff`、SOURCE `fa3337b5`、`scalar_result.json`|named implementations、IID/union bound、SOURCE count on normal return、totality/wheel未証明。N/G/ρは対象外。|保存上界・下界包含を引用するだけ。scalar batch再実行不要。|
|S12 SOURCE／条件表|Methods/Data availability。再現性と採択の追跡|本CSVと既存CEMv3・ledgerv5・study_index、各SOURCE/STOP|文書索引は独立の科学証拠ではない。H8公開監査不足とレビュー原本未取得。|固定リンクの静的表で可能。新管理基盤不要。|

## 4. 今回作らないもの

Figure2を新H4/H6 P cohortの機構分解へ置換する、旧図の数値を新しい共通allowanceで再集計する、3分子の改善率を平均する、latest15から新winnerを選ぶ、保存CSVから新しいSE・fit・scalingを計算する作業は実施しない。必要な場合も、限定論文化に不可欠なのか、強い主張を追加したいだけなのかを先に区別する。

既存画像の誌面向け並べ替え・文字修正は将来の編集案であり、今回原図へ書き戻していない。作図案の採否はユーザー判断、Codexは文書公開後mandatory STOP。

[INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/731b3d6f3572c6aa7950a579b68251fec9a5218b/docs/data_collection/2026-10-24_track_a_research_data/README.md
[PS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md
[LT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[FIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03efd057a7e77c40ae7e90b9c77885d6606e793b/docs/non_df_h4_comparison_synthesis_20261025.md
[F2A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb33de67475e7d33eed56801b74a2a277147e4be/artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/figure2A.png
[F2AS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb33de67475e7d33eed56801b74a2a277147e4be/artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/figure2A.svg
[F2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb33de67475e7d33eed56801b74a2a277147e4be/artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/figure2B.png
[F2PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/figure2_update_proposal.md
[MAIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json
[LTCOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/cost_vs_time.png
[LTDELTA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/selected_delta_vs_time.png
[LTGRID]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/selected_formula_grid.png
[LTFIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/fixed_delta_time_dependence.png
[LTPROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/completion_provenance.json
[LTBEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/best_configurations.csv
[LTIND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/independent_comparisons.csv
[H6G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H6_G_vs_s_extension_zoom.png
[H6BNC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H6_B_N_C_vs_s.png
[EVENTCSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/event_cost_trends.csv
[EVENTBEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/best_by_precision.csv
[LATEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/data/pauli_jw_summary/latest15.csv
[H2G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/figures/H2_G_vs_s.png
[H2BNC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/figures/H2_B_N_C_vs_s.png
[H4G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H4_G_vs_s.png
[CONTROLFIG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03efd057a7e77c40ae7e90b9c77885d6606e793b/artifacts/non_df_h4_a_cohort_cos/2026-10-24/run01/cost_comparison.png
[PAIREDFIG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03efd057a7e77c40ae7e90b9c77885d6606e793b/artifacts/non_df_h4_a_axes/2026-10-24/run01/paired_costs.png
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03efd057a7e77c40ae7e90b9c77885d6606e793b/artifacts/non_df_h4_d2_execution/2026-10-24/comparison_table.csv
[COST6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03efd057a7e77c40ae7e90b9c77885d6606e793b/artifacts/non_df_h4_comparison_synthesis/2026-10-25/circuit_costs_QUOTED.csv
[CNG30]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03efd057a7e77c40ae7e90b9c77885d6606e793b/artifacts/non_df_h4_comparison_synthesis/2026-10-25/conditional_resources_QUOTED.csv
[MEANC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d4f6ff7f1df085f292d36c36e460354df8a740/docs/non_df_h4_pr_mean_cost_scalar_execution_20261025.md

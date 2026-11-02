# PR/PF：解析的評価と実回路評価のC・N・G比較

**CとN固定のGは20/20条件を作図した。独立したN_circuitは0/20条件で、N比・G10・G11・N変更寄与はnull。**

同じNを使ったG00→G01の差は、会計上すべてCの置換に対応する。独立にNの変化が0と確認できたわけではない。PRのC差には法則期待値と有限MC平均の差も含まれ、全量をcompile最適化だけに帰属しない。

## 図への入口

各図のPNG/SVG/PDFを保存。Cは両測定軸別のcompiled/theory単価比、Nは取得済みの十分shot数（独立比較値は未取得）、GはG00基準のG01。主図の横軸はH2/H4/H6/H8。空抜きのH6/H8は条件付き参考値。

|ε|4系・条件付き参考を含む|H2/H4の実施済み検査PASS範囲|
|---|---|---|
|0.05|C: [PNG](all_conditional/C_eps_0p05.png)・[SVG](all_conditional/C_eps_0p05.svg)・[PDF](all_conditional/C_eps_0p05.pdf) / N: [PNG](all_conditional/N_eps_0p05.png)・[SVG](all_conditional/N_eps_0p05.svg)・[PDF](all_conditional/N_eps_0p05.pdf) / G: [PNG](all_conditional/G_eps_0p05.png)・[SVG](all_conditional/G_eps_0p05.svg)・[PDF](all_conditional/G_eps_0p05.pdf)|C: [PNG](confirmed_H2_H4/C_eps_0p05.png)・[SVG](confirmed_H2_H4/C_eps_0p05.svg)・[PDF](confirmed_H2_H4/C_eps_0p05.pdf) / N: [PNG](confirmed_H2_H4/N_eps_0p05.png)・[SVG](confirmed_H2_H4/N_eps_0p05.svg)・[PDF](confirmed_H2_H4/N_eps_0p05.pdf) / G: [PNG](confirmed_H2_H4/G_eps_0p05.png)・[SVG](confirmed_H2_H4/G_eps_0p05.svg)・[PDF](confirmed_H2_H4/G_eps_0p05.pdf)|
|0.01|C: [PNG](all_conditional/C_eps_0p01.png)・[SVG](all_conditional/C_eps_0p01.svg)・[PDF](all_conditional/C_eps_0p01.pdf) / N: [PNG](all_conditional/N_eps_0p01.png)・[SVG](all_conditional/N_eps_0p01.svg)・[PDF](all_conditional/N_eps_0p01.pdf) / G: [PNG](all_conditional/G_eps_0p01.png)・[SVG](all_conditional/G_eps_0p01.svg)・[PDF](all_conditional/G_eps_0p01.pdf)|C: [PNG](confirmed_H2_H4/C_eps_0p01.png)・[SVG](confirmed_H2_H4/C_eps_0p01.svg)・[PDF](confirmed_H2_H4/C_eps_0p01.pdf) / N: [PNG](confirmed_H2_H4/N_eps_0p01.png)・[SVG](confirmed_H2_H4/N_eps_0p01.svg)・[PDF](confirmed_H2_H4/N_eps_0p01.pdf) / G: [PNG](confirmed_H2_H4/G_eps_0p01.png)・[SVG](confirmed_H2_H4/G_eps_0p01.svg)・[PDF](confirmed_H2_H4/G_eps_0p01.pdf)|
|0.005|C: [PNG](all_conditional/C_eps_0p005.png)・[SVG](all_conditional/C_eps_0p005.svg)・[PDF](all_conditional/C_eps_0p005.pdf) / N: [PNG](all_conditional/N_eps_0p005.png)・[SVG](all_conditional/N_eps_0p005.svg)・[PDF](all_conditional/N_eps_0p005.pdf) / G: [PNG](all_conditional/G_eps_0p005.png)・[SVG](all_conditional/G_eps_0p005.svg)・[PDF](all_conditional/G_eps_0p005.pdf)|C: [PNG](confirmed_H2_H4/C_eps_0p005.png)・[SVG](confirmed_H2_H4/C_eps_0p005.svg)・[PDF](confirmed_H2_H4/C_eps_0p005.pdf) / N: [PNG](confirmed_H2_H4/N_eps_0p005.png)・[SVG](confirmed_H2_H4/N_eps_0p005.svg)・[PDF](confirmed_H2_H4/N_eps_0p005.pdf) / G: [PNG](confirmed_H2_H4/G_eps_0p005.png)・[SVG](confirmed_H2_H4/G_eps_0p005.svg)・[PDF](confirmed_H2_H4/G_eps_0p005.pdf)|
|0.001|C: [PNG](all_conditional/C_eps_0p001.png)・[SVG](all_conditional/C_eps_0p001.svg)・[PDF](all_conditional/C_eps_0p001.pdf) / N: [PNG](all_conditional/N_eps_0p001.png)・[SVG](all_conditional/N_eps_0p001.svg)・[PDF](all_conditional/N_eps_0p001.pdf) / G: [PNG](all_conditional/G_eps_0p001.png)・[SVG](all_conditional/G_eps_0p001.svg)・[PDF](all_conditional/G_eps_0p001.pdf)|C: [PNG](confirmed_H2_H4/C_eps_0p001.png)・[SVG](confirmed_H2_H4/C_eps_0p001.svg)・[PDF](confirmed_H2_H4/C_eps_0p001.pdf) / N: [PNG](confirmed_H2_H4/N_eps_0p001.png)・[SVG](confirmed_H2_H4/N_eps_0p001.svg)・[PDF](confirmed_H2_H4/N_eps_0p001.pdf) / G: [PNG](confirmed_H2_H4/G_eps_0p001.png)・[SVG](confirmed_H2_H4/G_eps_0p001.svg)・[PDF](confirmed_H2_H4/G_eps_0p001.pdf)|
|0.0005|C: [PNG](all_conditional/C_eps_0p0005.png)・[SVG](all_conditional/C_eps_0p0005.svg)・[PDF](all_conditional/C_eps_0p0005.pdf) / N: [PNG](all_conditional/N_eps_0p0005.png)・[SVG](all_conditional/N_eps_0p0005.svg)・[PDF](all_conditional/N_eps_0p0005.pdf) / G: [PNG](all_conditional/G_eps_0p0005.png)・[SVG](all_conditional/G_eps_0p0005.svg)・[PDF](all_conditional/G_eps_0p0005.pdf)|C: [PNG](confirmed_H2_H4/C_eps_0p0005.png)・[SVG](confirmed_H2_H4/C_eps_0p0005.svg)・[PDF](confirmed_H2_H4/C_eps_0p0005.pdf) / N: [PNG](confirmed_H2_H4/N_eps_0p0005.png)・[SVG](confirmed_H2_H4/N_eps_0p0005.svg)・[PDF](confirmed_H2_H4/N_eps_0p0005.pdf) / G: [PNG](confirmed_H2_H4/G_eps_0p0005.png)・[SVG](confirmed_H2_H4/G_eps_0p0005.svg)・[PDF](confirmed_H2_H4/G_eps_0p0005.pdf)|

計30図・90ファイル。既存スライド6・詳細A.9と元図は変更していない。スライド6への採用は未判断。

## 定義とN監査

|量|解析側|回路側|今回の扱い|
|---|---|---|---|
|C_a|NATIVE_MIXED / explicit_rz_sx_x_cx_v1の法則期待RZ数|PRの保存compiled平均／PFの決定論計数|cos/sin別に比較。理論lower/upperを保持|
|N_a|保存bias・normalization・経験的allowanceをHoeffding十分条件へ代入|同じ推定器・精度・失敗率のもとでcompiled信号から独立評価する量|未取得null。実機の実測必要shot数ではない|
|G00|Σ N_theory,a C_theory,a|—|native費用モデル＋数値信号入力によるhybrid会計|
|G01|—|Σ N_theory,a C_circuit,a|**N固定・Cのみ置換した総費用比較**|
|G10 / G11|Σ N_circuit,a C_theory,a|Σ N_circuit,a C_circuit,a|独立N未取得のため両方null|

N_theoryは純粋に解析式だけから得た値ではない。corrected平均信号とfull-H参照との差を古典数値評価し、経験的u、finite-RTE正規化、α_cos=α_sin=.025、各軸ε/√2を使う。Nは必要最小回数でなく十分条件に基づく配分。補正付き独立有界±1測定平均にHoeffdingを適用する。PRでは量子shotごとにfresh IID RTE抽選を仮定する。

すべての抽選回路で同じ理想測定法則を実装し、推定器・bias・u・B・αを同じにすれば、同じ式は同じ十分Nを返す。これは条件付きの数学的帰結で、保存compiled費用だけからNを独立取得した証拠ではない。代表状態ベクトルの1e-7検査だけを、全抽選列のbias・正規化・数値誤差を保証する証明へ格上げしない。独立Nを得るにはcompiled平均信号または全法則に対する同値性／誤差移送と、同じ保証強度の数値誤差予算が必要。分散適応や別の統計保証を採用する計算は行っていない。

[N比較可能性監査](N_comparability_audit.json)に6項目のpreflightと不足情報を記録した。[追加比較の契約案](N_comparison_contract_DRAFT.md)は未採用・未実行。新推定器の科学的採用は未承認。今回のN図は取得済みshotの情報図であり、独立比較図ではない。

## 固定条件・cohort・留保

linear H2/H4/H6/H8、隣接1.00 Å、STO-3G、signed tol-only DF1e-8、rank3/10/19/28、generation順、T=.8、通常signed bias、allowance倍率1、P=0。full cos/sin Hadamard wrapper、native DF、symmetric-directional、original_plus_tail_once、reference_rotated=false。Qiskit1.3.0・Qiskit Nature0.7.2、rz/sx/x/cx、opt1、seed17、topologyなし。外側S2・旧compiled最良候補を固定し、理論側の再選択なし。各L_D/q/r/K・δ=.8/qと入力hashは[データ](figure_data.json)・[coverage](coverage_matrix.md)・[provenance](provenance.json)に対応する。

H2は元cohort n8の回収5条件。H4/H6/H8は新cohort15条件で、H2とpoolしない。H6/H8の10条件はunweighted-G補助診断不合格の留保付きで、固定1e-9閾値を変更せず、検証済み主結果へ格上げしない。[元の物理検証](../pr_pf_s2_recomputed_20261022/physical_validation_audit.json)を参照。失敗科学単位・grant・STOPは再利用／retryなし。

PRの誤差棒は保存標本から得る平均費用の名目Student-t 95%区間。Gの両軸はpairedで共分散を含める。Nの量子測定統計誤差とは別。元H2 n8に新1%目標を適用しない。新本評価11候補の設計目標達成を継承するが、5つの未観測positive-order層のcompiled母平均寄与は不明で、通常区間は厳密保証ではない。[MC記録](MC_uncertainty.json)・[元の希少event診断](../pr_pf_s2_recomputed_20261022/rare_event_summary.json)。

原PR論文のToffoli資源式の直接再現検証ではなく、native DFのRZ費用モデルとcompiled回路の比較。FT T、実機noise、全RPE最終総費用、一般的優位性、外部再現、immutable CIを主張しない。

## 総費用とPR/PFの傾向

表のR=100(1−G01/G00)はNを固定した低下率。正のRはcompiled費用が低い。R_PF−R_PR>0はPFの相対低下が大きいという点比較。N寄与は未取得で、0と代入しない。

|系|ε|L_D PR/PF|δ PR/PF|R_PR [%]|R_PF [%]|R_PF−R_PR [pp]|
|---|---|---|---|---:|---:|---:|
|H2|0.05|1/3|0.8/0.8|44.601|54.314|+9.713|
|H2|0.01|1/3|0.8/0.8|44.600|54.313|+9.713|
|H2|0.005|1/3|0.8/0.4|44.597|56.337|+11.741|
|H2|0.001|1/3|0.2/0.2|48.908|57.354|+8.447|
|H2|0.0005|1/3|0.2/0.2|48.907|57.354|+8.447|
|H4|0.05|3/5|0.8/0.8|42.991|46.006|+3.015|
|H4|0.01|3/7|0.4/0.4|45.083|47.286|+2.202|
|H4|0.005|2/7|0.2/0.2|48.572|48.020|-0.551|
|H4|0.001|2/7|0.1/0.1|49.482|48.388|-1.094|
|H4|0.0005|2/7|0.1/0.1|49.482|48.388|-1.094|
|H6*|0.05|4/7|0.4/0.8|43.175|42.546|-0.629|
|H6*|0.01|4/11|0.2/0.4|44.369|42.683|-1.687|
|H6*|0.005|4/11|0.2/0.2|44.369|43.202|-1.167|
|H6*|0.001|4/11|0.1/0.1|44.544|43.462|-1.082|
|H6*|0.0005|4/11|0.1/0.1|44.544|43.462|-1.082|
|H8*|0.05|8/11|0.4/0.8|40.401|39.577|-0.824|
|H8*|0.01|8/15|0.2/0.2|41.170|40.381|-0.788|
|H8*|0.005|8/15|0.2/0.2|41.170|40.381|-0.788|
|H8*|0.001|6/15|0.1/0.1|41.972|40.593|-1.379|
|H8*|0.0005|7/15|0.05/0.1|41.492|40.593|-0.899|

元H2の5条件ではPFの低下率がPRより大きい。H4は粗い2精度でPF、残る3精度でPRの低下率が大きい。H6/H8の条件付き点比較では全精度でPRの低下率がわずかに大きい。普遍的な方式差や有意差、compilerだけの因果効果とはしない。

[C/N/G分解の検算表](CNG_decomposition_verification.json)を保存した。集約C比×集約N比をG比として使っていない。G01−G00=ΣN_theory,a(C_circuit,a−C_theory,a)を有理数で検算した。G01→G11の寄与と置換順序依存性はN未取得のため評価不能。

## 保存・検証・停止

[数値と形式の検証](verification.json)、[C/N/G CSV](figure_data.csv)、[実行receipt](../../../../../artifacts/resource_applicability/pr_pf_analytic_vs_circuit_CNG/2026-10-22/execution_receipt.json)、[SOURCE](../../../../../artifacts/resource_applicability/pr_pf_analytic_vs_circuit_CNG/2026-10-22/SOURCE.json)。新しい保存値会計は40単位をone-shot実行し、新signal・shot評価・compile・samplingは0。既存SOURCE/raw/grant/ledger/marker/STOP/図/スライドを保全した。

branch `codex/pr-pf-analytic-vs-circuit-cng-20261022`。SOURCE `bb5251fc9780ac7ac3b6c3c540aa3c865879eb4f`。完成物はローカルcommitへ保存し、push・公開・merge・PRは実行しない。既存研究原本の保全指示を優先し、研究概要・目的・スライドは変更せず、検証文書・新日付ノート・索引・証拠台帳へ追加した。主担当自身の技術検証で、GPT独立詳細科学レビューは未開始。**mandatory STOP**。

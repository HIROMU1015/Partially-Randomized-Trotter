# H4/H6：理論総費用 vs compiled総費用

S4修正後RESULT `6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1` の保存データを表示。
[説明・固定候補・採否・限界](../../../../docs/research/track_a_h46_theory_compiled_total_plots_20261019.md)。

|系|allowance倍率|絶対RZ総費用|PR/PF費用比|
|---|---:|---|---|
|H4|1|[PNG](figures/h4_absolute_rz_factor_1.png) · [SVG](figures/h4_absolute_rz_factor_1.svg) · [PDF](figures/h4_absolute_rz_factor_1.pdf)|[PNG](figures/h4_ratio_rz_factor_1.png) · [SVG](figures/h4_ratio_rz_factor_1.svg) · [PDF](figures/h4_ratio_rz_factor_1.pdf)|
|H6|1|[PNG](figures/h6_absolute_rz_factor_1.png) · [SVG](figures/h6_absolute_rz_factor_1.svg) · [PDF](figures/h6_absolute_rz_factor_1.pdf)|[PNG](figures/h6_ratio_rz_factor_1.png) · [SVG](figures/h6_ratio_rz_factor_1.svg) · [PDF](figures/h6_ratio_rz_factor_1.pdf)|
|H4|10|[PNG](figures/h4_absolute_rz_factor_10.png) · [SVG](figures/h4_absolute_rz_factor_10.svg) · [PDF](figures/h4_absolute_rz_factor_10.pdf)|[PNG](figures/h4_ratio_rz_factor_10.png) · [SVG](figures/h4_ratio_rz_factor_10.svg) · [PDF](figures/h4_ratio_rz_factor_10.pdf)|
|H6|10|[PNG](figures/h6_absolute_rz_factor_10.png) · [SVG](figures/h6_absolute_rz_factor_10.svg) · [PDF](figures/h6_absolute_rz_factor_10.pdf)|[PNG](figures/h6_ratio_rz_factor_10.png) · [SVG](figures/h6_ratio_rz_factor_10.svg) · [PDF](figures/h6_ratio_rz_factor_10.pdf)|

主表示はNATIVE_MIXED/RZ、5精度。H4はlegacy DF12/PR anchor n32、H6はtol-only DF19/PR探索n8。PFは固定回路。
理論期待費用とcompiled保存標本費用を区別し、理論lower/upperを中点やCIにしない。不適格は空白・灰色×。
今回のshot込みRZ定義だけで、状態準備・FTQC全体・RPE全roundの費用ではない。

- [絶対費用exact表](data/absolute_totals.csv)：70条件、50適格、20不適格。必要shot/n/cohort/DF/L_D/δ/入力hash/出典を保持。
- [PR/PF比exact表](data/pf_pr_ratios.csv)：60条件、32共通適格、28共通適格性なし。保存ratioと保存方向判定を転記。
- [閲覧用表示表](data/display_tables.md)、[全描画座標](data/plot_coordinates.csv)、[採用元record](data/source_records.json)、[入力SHA](data/input_sources.json)。
- [caption案](captions.md)、[manifest](manifest.json)、[構造検査](verification.json)、[PNG目視確認](visual_review.json)。
- [描画環境のpackage版](requirements-plot.txt)。

```bash
# worktree root、Matplotlibが入った専用環境で実行
python3 scripts/plot_h46_theory_compiled_totals.py

# 保存値・出力・保全をstdlibだけで読取照合（科学runnerは動かさない）
python3 scripts/plot_h46_theory_compiled_totals.py --check
```

描画には基点commitのGit objectが必要。今回の環境は前回のプロジェクト専用`.plot-venv`（Matplotlib3.9.4）を再利用。
PNGは250dpi・幅3300px、SVG/PDFはvector。既存Figure2・主表・結果を変更しない。新しい科学解析0。

今回の成果物を専用branchにcommit/pushした後、次の作業なし。**mandatory STOP**。

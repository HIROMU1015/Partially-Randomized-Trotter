# PR/PF絶対コスト比較：v2・保存済み5精度

[採用・除外・注意点の説明](../../../../docs/research/track_a_pr_pf_absolute_cost_v2_20261019.md)。作図基準はv2 snapshot c0808b25。既存の26比較・52点だけを使用し、新しい科学計算・会計・順位計算なし。

|要求複素信号精度ε|PNG|SVG|PDF|表示範囲|
|---|---|---|---|---|
|0.05|[PNG](figures/absolute_rz_eps_0p05.png)|[SVG](figures/absolute_rz_eps_0p05.svg)|[PDF](figures/absolute_rz_eps_0p05.pdf)|独立確認6点＋探索8点、別パネル|
|0.01|[PNG](figures/absolute_rz_eps_0p01.png)|[SVG](figures/absolute_rz_eps_0p01.svg)|[PDF](figures/absolute_rz_eps_0p01.pdf)|探索8点|
|0.005|[PNG](figures/absolute_rz_eps_0p005.png)|[SVG](figures/absolute_rz_eps_0p005.svg)|[PDF](figures/absolute_rz_eps_0p005.pdf)|探索8点|
|0.001|[PNG](figures/absolute_rz_eps_0p001.png)|[SVG](figures/absolute_rz_eps_0p001.svg)|[PDF](figures/absolute_rz_eps_0p001.pdf)|探索8点|
|0.0005|[PNG](figures/absolute_rz_eps_0p0005.png)|[SVG](figures/absolute_rz_eps_0p0005.svg)|[PDF](figures/absolute_rz_eps_0p0005.pdf)|独立確認6点＋探索8点、別パネル|

- [比較データCSV](data/comparisons.csv)：26比較、exact総費用、保存paired SE、候補・n・cohort・DF・L_D/δ・PF集合/実測/model・出典。
- [閲覧用データ表](data/comparisons_display.md)：精度・証拠段階別。表示用の数値丸めのみ。
- [実描画点CSV](data/plot_points.csv)：PR/PF各点のexact値・float座標・ラベル・元record。
- [採否表](data/selection_decisions.csv)、[保存元record](data/source_records.json)、[入力Git objectのSHA](data/input_sources.json)。
- [スライド掲載用caption案](captions.md)。
- [manifest](manifest.json)、[検査結果](verification.json)、[描画環境の固定package](requirements-plot.txt)。

独立確認はH4/H6/H8の両端2精度のみ。探索はH2/H4/H6/H8、全点新n8。H2 P未完了、H4 legacy DF12/旧n32/FP標本は不採用。H6中間3精度のPFは検証済みmodel（白抜き）。選択点が実測でもH6/H8の登録PF集合にはmodel依存が残る。H8詳細公開PR監査/原profile bytesは未照合。未観測rare費用と母平均順位は未保証。既存論文主表/CEM/原図は不変。

保存値を表示するため、以下のコマンドは科学moduleを実行しない。描画には記載された固定commitのGit objectが必要。

```bash
# worktree rootで、今回作成した専用環境を使う
.plot-venv/bin/python3 scripts/plot_pr_pf_absolute_cost_v2.py

# 標準ライブラリだけで、既存出力と固定保存値を読み取り照合
python3 scripts/plot_pr_pf_absolute_cost_v2.py --check
```

環境を再作成する場合は専用venvへ`requirements-plot.txt`を導入する。共有環境を変更しない。`.plot-venv`と`.plot-cache`はGit対象外。PNGは3600×2100、300dpi。SVGはtextを保持、PDFはvector出力。各精度の縦軸範囲は異なり、両端の2パネル内では共通。

local commitのみ。公開pushなし。次の作業なし。**mandatory STOP**。

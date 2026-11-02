# H4 B0+I全面実測の原稿v0.4・図表・科学review資料

**科学的採否前のdraft。新しい科学計算0、資料整備後mandatory STOP。**

一つの資料入口は[技術統合報告](../../../../docs/research/track_a_h4_b0i_manuscript_integration_20261014.md)。
[通し原稿v0.4](../../../../docs/manuscripts/track_a_resource_study_geometry_native_v0_4.md)、[GPT handoff](../../../../docs/manuscripts/track_a_h4_b0i_v0_4_scientific_review_handoff.md)、[主張と証拠](../../../../docs/manuscripts/track_a_h4_b0i_v0_4_claim_evidence.md)、[v0.3からの変更](../../../../docs/manuscripts/track_a_h4_b0i_v0_4_changes.md)、[H4/H6/H8条件表](../../../../docs/research/track_a_resource_comparison_conditions_20261014_v0_4.md)へ進める。

固定結果commit `98cdb35cd08469714f20d6cc3b982d9c80908f46`、実行SOURCE `924b779c2bc6f6e844d6283c355f91c73aed6445`、data identity SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`。
元v0.3、旧4図、元結果・SOURCE・profile・入力・raw/ledger/cacheは変更しない。

## 保存表と図

|File|内容|
|---|---|
|`cost_factors_1812.csv`|双方のfamily点最小のbias/B/headroom/shot/軸C/C_eff/G、shot比×C_eff比=費用比|
|`uncertainty_comparison_1812.csv`|paired SE、点差、374/23と1392/23の記述的分類。正式confidence intervalではない|
|`method_grid_regions.csv`|12離散method領域、最初/最後のεと隣接bracket|
|`configuration_grid_regions.csv`|拡張集合の点最良template19領域、隣接bracket|
|`matched_B0_B0I_120.csv`|同一geometry/prefix/q、ε=.05での旧B0とB0+Iのbias/headroom/shot比較|
|`fixed_configuration_factors.csv`|既存B0+I6/B2 6の固定display panel×6距離×3精度=216行。新独立確認ではない|
|`compiled_metric_deltas_1440.csv`|240 wrappers×6 metricsの旧B0との差と新numeric FP。全delta0|
|`selected_factors18.csv`|ε=.005/.05/.1の代表18点|
|`analysis_summary.json`|397/1415、374/23、図表用要約とSTOP/scope|
|[review tables](tables_for_review.md)|代表比較・切替bracket・因子・SE分類のMarkdown表|
|[figure captions](figure_captions.md)|4新主図＋S1の定義・元CSV・限界|
|`figures/`|5図のPNG/SVG/PDF、15 assets|
|`provenance.json`|固定Git inputs/SOURCE59/profile/input、旧12図asset保全、H6/H8固定参照、直前レビューSHA|
|`references/` / `review_context/`|既存固定参照と利用者共有の直前科学レビューをbyte-identicalに収録|
|`verification.json` / `artifact_manifest.json`|限定8検査・再算術・リンク・bytes/SHA・doc/figure対応|

H4 linear/STO-3G/DF12、T=.8、6距離、prefix3/4/5/6/9、q1/2/4/8、δ=.1〜.8、S2/ordinary/structured/fixed compiler/P=0。εはcomplex signal、全1812点は同じ保存値のprecision投影で独立実験ではない。
同じ32 paired trajectoriesでB2を選抜・推定し、希少event母平均未保証を保持する。

## 再生成・照合

Repository rootから、stdlib-onlyの保存表再生成と照合:

```bash
python3 -P -B scripts/resource_applicability/integrate_h4_b0i_manuscript.py
python3 -P -B scripts/resource_applicability/integrate_h4_b0i_manuscript.py --verify
python3 -P -B tests/tracks/resource_applicability/test_h4_b0i_manuscript_integration.py -v
python3 -P -B scripts/resource_applicability/verify_h4_b0i_manuscript_bundle.py
```

push後のremote HEAD・全公開doc/CSV/figure bytes/SHA再取得照合は、最後のcommandに`--remote`を付ける。GitHubへ読取アクセスし、remote branch SHAとlocal HEADが一致するときだけ固定commitの全assetsを照合する。認証できなければ設定を変えず未完了と報告する。

図は既存Matplotlibを使う。`MPLCONFIGDIR`、`TMPDIR`、`XDG_CACHE_HOME`を既に存在するhome専用directoryに設定し、内部threadを1に固定して、[plot script](../../../../scripts/resource_applicability/plot_h4_b0i_manuscript.py)の`--output`に**新しいexclusive directory**を指定する。既存`figures/`を上書きしない。
PNG/SVG/PDFは保存CSVから生成され、量子/科学library・compilerを呼ばない。生成capは[plan](analysis_plan.json)、実使用PythonとMatplotlib/NumPyのversionはverificationのplot profileを参照。
図の二次再生成で描画library versionが変わる場合、バイナリ同一性を保証しない。byte manifestは今回固定したassetsを識別する。

原稿はGPT採否待ち。H4独立確認・希少次数診断、H8 core最終map、他compiler/energy/QPE/RPE/fault-tolerant/state-preparationを本taskで補完しない。
公開状況は公開監査と最終チャットを参照する。credential設定を変更しない。

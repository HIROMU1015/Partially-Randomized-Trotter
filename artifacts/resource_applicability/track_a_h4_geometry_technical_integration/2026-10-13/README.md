# H4 geometry：保存値技術統合bundle（2026-10-13）

[全体の資料入口](../../../../docs/research/track_a_h4_geometry_technical_integration_20261013.md)／
[通し原稿v0.3](../../../../docs/manuscripts/track_a_resource_study_geometry_native_v0_3.md)／
[比較条件](../../../../docs/research/track_a_resource_comparison_conditions_20261013.md)／
[GPTへの判断事項](SCIENTIFIC_DECISIONS.md)。

元登録集合の結果を変更しない追加技術解析。B0+Iは未測定・未登録の感度であり、正式baselineではない。
new science/signal/eigh/trajectory/build/compile/transpile/GPU=0、H8/共有環境/venv変更=0。
同じ保存cost32本を302 precisionへ投影しており、精度点を独立実験として数えない。

## ファイル対応

|ファイル|内容|
|---|---|
|`precision_grid_1812.csv`|全6×302：method適格数/最小費用、winner構成、axis bias/B/headroom/shots/RZ、G/SE/covariance、deterministic比・因子|
|`configuration_switches.csv`|登録winner/B0最小の連続grid領域と境界bracket|
|`nearby_candidates.csv`|±2SEが重なる4803候補×precision行、paired SEと費用差；正式CIではない|
|`scalar_classification_1812.csv`|全1812点の補助B0+I Δ=0 envelope、B2構成、gap、axis shot、固定候補/全補助候補閾値|
|`scalar_reversal_groups.csv`|397点をgeometry・補助構成・元B2構成で分けた8群|
|`scalar_comparison_switches.csv`|補助zero-Δ構成と比較順位のgrid切替|
|`scalar_threshold_extremes.csv`|固定zero-Δ候補の逆転閾値が最小/最大の各12点|
|`delta_sensitivity.csv`|共通非負Δ仮定：補助候補再選択と固定候補を分けた点数|
|`fixed_configuration_factors.csv`|posthoc明示6構成×3精度×6距離、shots/effective-RZ/Gの比とlog分解|
|`rare_order_law_diagnostics.csv`|1164 stochastic候補：保存Bから法則逆算、確率、保存cost、NOT_SAVED/OUTSIDE_CUTOFF；B=1の172件は未解決|
|`analysis_summary_v1.json`|集計・提案6/10/240 calls・14人工検査結果、実行wall（集計自体の時間）|
|`event_metadata_audit_v1.json`|streaming file-family、最大24×4KiBのrecord schema読取、source-defined payloadによる未保存区分|
|`comparison_conditions.csv`|H4 geometry/anchor/H6/H8の条件対照（文書表の機械可読転記）|
|`provenance_v1.json`|public解析入力、元source19のうち静的参照subset、固定比較文書・native codeのGit commit/blob/bytes/SHA|
|`verification_v1.json`|別経路Fraction arithmetic・paired SE・補助envelope・law、公開入力/Git参照byte照合のlocal結果|
|`artifact_manifest_v1.json`|新bundle/図/原稿/索引のbytes/SHA（manifest自身は除く）|
|`analysis_plan_v1.json`|process/thread/AS/CPU/wall/output/read上限、科学実行0の範囲|
|`figures/`|4図をPNG/SVG/PDFで収録。captionは原稿とfigure_captions.md|
|`SCIENTIFIC_DECISIONS.md`|GPTが採否を判断する論点と、未実行6/10/240 compile proposal|

旧6入力とtail scalarのprovenanceは隣接
`track_a_h4_geometry_scalar_diagnostic/2026-10-13/source_input_provenance_v1.json` を固定参照。
旧停止proof/元full-record監査は再実行せず保存結果を参照した。今回NPZアクセス0。

## 再生成と照合

以下はrepository rootで実行する例。解析script/validatorはstdlibのみ。
出力directoryは新規exclusiveで、既存bundleを直接上書きしない。
`--repo` はGit worktree root。single process、OMP/BLAS等thread1、Python -P -Bを使う。

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -P -B artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/analyze_saved_geometry.py \
  --repo "$PWD" --output /home/AbeHiromu/h4-analysis-reproduction-new

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -P -B artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/verify_publication.py \
  --repo "$PWD" --bundle artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13 \
  --verify-manifest
```

公開だけの検証にraw/NPZは不要。固定比較source/documentのbyte確認にはprovenanceに記したGit objectsが必要。
shallow cloneや未取得の比較branchでは、そのobjectを取得してから照合する。欠落をPASSへ置換しない。
`analyze_saved_geometry.py` 自体は同branch内のpublic projectionだけで動き、比較Git objectsを必要としない。
summaryのwall秒は再実行で変わる。CSVは同じ固定保存入力と演算順から再生成できる。

図は既存のMatplotlibが入ったPythonを使う（installは不要）。MPLCONFIGDIR/TMPDIRを自分のhome内に設定する。
今回使った既存private venvは変更していない。

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/home/AbeHiromu/h4-plot-config TMPDIR=/home/AbeHiromu/h4-plot-temp \
  /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python \
  -P -B artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/build_figures.py \
  --bundle artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13 \
  --output /home/AbeHiromu/h4-figures-reproduction-new
```

cache/temp directoriesは事前にhome配下で作成する。PDF日時など描画metadataの差により図byte SHAが変わる場合がある。
図はデータとscriptを固定し、公開したassetそのもののSHAはmanifestで確認する。

`audit_saved_event_schema.py` は任意のprivate raw rootに対する読み取り専用補助。最大24 recordの4KiBのみ、
残りは名前だけをstreaming集計する。public算術再現に必須ではなく、seed replayを行わない。
event auditはsource payload＋以前の全record監査を併用するため、sample24だけで全recordの同一性を推定しない。

## 公開・検証の限界

local技術検証、14限定人工算術PASS、全grid/nearby costsのFraction照合PASS。CIや独立科学reviewではない。
新branchの公開はauthenticated non-force push後にremote commitとasset bytes/SHAを再確認する。
認証できない場合はGit/credential設定を変えず、最終報告の手動push commandを使い、remote公開未確認を維持する。
B0+I採否・実compile・追加samplingを自動的に進めない。最終報告後mandatory STOP。

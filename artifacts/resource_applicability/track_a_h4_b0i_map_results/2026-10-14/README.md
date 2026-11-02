# H4 B0+I measured map240 results

[日本語結果報告・比較表・条件・限界](../../../../docs/research/track_a_h4_b0i_map_results_20261014.md)。SOURCE `924b779c2bc6f6e844d6283c355f91c73aed6445`、run `h4-b0i-map240-parallel-run02-20261014` は全240件完了後STOP。234新規compile＋6件の検証済みpilot再利用。新しい科学計算や次stageは実行しない。

- `compiled_wrapper_metrics.json`: source/input/numeric FP付きの240軸の軽量projection。
- `measured_candidates.json`: 120構成の保存bias、normalization、実測両軸RZ。
- `precision_grid_1812.csv`: 各geometry/precisionで20 B0+I構成を比較し、旧登録最小B2との点差・paired SEを保存。
- `selected_precision18.csv`: ε=.005/.05/.1の18比較。
- `summary.json`: 全地図の点最小件数、6指標の旧B0との差、対象domainと解釈限界。
- `native_resource_receipt.json`: COMPLETE/cleanup/監視/実資源/累積会計と原本bytes/SHAの監査。runtimeやNPZ本体を含めない。
- `artifact_manifest.json`: 公開fileと依存資料のbytes/SHA。
- `verification.json`: 公開保存値だけで算術を再現したlocal PASS。独立科学reviewではない。

Repository rootから、stdlib-onlyの読取再現:

```bash
python3 -P -B artifacts/resource_applicability/track_a_h4_b0i_map_results/2026-10-14/analyze_saved_results.py
```

このcommandはNPZ/private runtimeへアクセスせず、compiler/libraryをimportせず、新しいbuild/signal/trajectory/compileを行わない。公開projectionの算術再現であり、量子計算や母平均費用の独立再現ではない。B2の同じ32 paired trajectoriesによる選抜と推定、多重比較・希少eventの未確定を保持する。

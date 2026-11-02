# H4 geometry map 完了結果（2026-10-13）

[結果の説明・距離別比較表](../../../../docs/research/track_a_h4_geometry_results_20261013.md)が研究結果の入口です。
H4 linear/STO-3G/DF12、T=0.8、二次DF-prefix PF/canonical finite RTE、6距離、全1,308候補・74,784 wrappersがMAP_COMPLETE_STOPで完了しました。

- [完了・coverage・18比較点・予算・限界の要約](summary.json)
- [ε=0.005/0.05/0.1の距離別比較CSV](precision_comparison.csv)
- [全1,308候補の軽量保存値projection](saved_candidates_v1.json)
- [公開保存値だけを使うstdlib照合script](verify_saved_result_projection.py)
- [公開bundleのbytes/SHAと元集計のprovenance](artifact_manifest_v1.json)

ε=0.05・状態準備P=0ではB2が各距離の点推定最小で、最小deterministic候補より27.9〜42.7%低コストでした。
保存済み302精度点×6距離でも点最小methodは全てB2でした。近接候補の±2SE区間は重なり、厳密winnerや一般最適性は認定しません。
εは複素signalの誤差で、Haや最終RPEのエネルギー精度ではありません。

公開projectionはraw runtime/checkpoint/cacheのfile copyではなく、保存済みsignalから必要fieldを取り出した軽量結果です。
NPZ、全74,784回路record、ledger/journal、observer log、native proof、one-shot/control、認証情報、内部SSH情報は収録しません。
元runtimeとの全record照合はlocal保存値監査であり、公開projectionだけでnative停止・physics correctness・実compileを外部再現したと主張しません。

branch `track-a-h4-geometry-results-20261013`。
実行SOURCE `d0726e946f89f8c30a08d383fb8ff48b363629ec`、dataset/seed SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`、実行時REVIEW `81c78b366d2cf2c699bb31977f72ae9deab4f1d4`。
原認可資料は[resume02 launch bundle](../../track_a_h4_production_resume02/2026-10-11/README.md)に保持します。
次stageは未認可、追加science/compile/transpileは0です。

照合はリポジトリrootから以下で実行できます。Python標準ライブラリだけを使い、install・runtime/NPZ読込・GPU・worker・共有設定変更は不要です。
process AS256MiB/CPU45秒/wall60秒、file write0です。

```bash
python3 -P -B artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/verify_saved_result_projection.py
```

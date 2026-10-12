# N3幅比較 v2 準備：2026-10-12レビュー系列

完了済みintegration科学レビュー第14節を受領し、弱い旧族分離から、同等/強い参照に対する有限費用の証拠取得へ比較実装を具体化した。
同等SDP、固定5 Gram blocksのnumber付きSOS、full one-body lower、独立sector residual checkerを実装。
同modeとH単位を分離し、全主要法に同じcheap ORを与える。densityを残した縮約Hamiltonianと元H fallbackを返す。
kernel/cold×3 timing repetition、2workers/thread1、call/bit/output/log/memory capを固定。wall/CPU capなし。
32 synthetic PASS、CAR/reference、最大dimension、認可拒否、改変拒否、cold cache/証拠継承、parallel証拠一致。
未使用6stress Hの係数・seed・candidate・εを保存。結果で筛選していないがIID分布性能とは呼ばない。数値適用0。
元source/S1/primary/2-base/STOP・N1停止/N2保留を維持。中心テーマ未採択。
source/input/plan公開後も新scientific pilotの明示実行認可を待つ。新sourceだけを理由に同内容のGPTレビューを繰り返さない。
入口：[準備](../n3_width_comparison_v2_preparation_20261012.md)。Mandatory STOP。

## 最終source・plan固定の追記

source `3eba0814e690e95993db60809c4c3c96784b07c0`、13 blob一致。selector/serializationの実測をtask費用へ加えた。
32 synthetic PASSと最終sourceの保存90証拠検査を記録。未使用6入力はidentity/schemaのpreflightだけで数値適用0。
[固定plan](../n3_width_comparison_v2_execution_plan_20261012.json) SHA `dc43a8d781e76ba4c3fc92f528c973f225861bf47160c5866bff2bebd9eb9d53`。
旧manifest104件、root差分を保全。公開・再取得確認まで進め、明示科学実行認可前で停止する。

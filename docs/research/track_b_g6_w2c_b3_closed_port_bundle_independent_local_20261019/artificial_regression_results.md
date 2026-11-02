# 実行した回帰と件数

直接基点 `5e862a70f2d74defe72dd39205d38c2a4f8ebeaf`。統合公開HEAD `62967269e611b61e73b70abe20e998d9c7eaaf50`、人工測定SOURCE `deade78a19f5b23859d16816d3a7bfb37765dd80`。実施2026-10-19〜20 JST。ARTIFICIAL_ONLY・未採択・実科学入口閉鎖。

|集合|今回の結果|重複の扱い|
|---|---:|---|
|既存原版10suite|623PASS|今回再実行|
|既存独立Bundle|165PASS|今回再実行|
|既存独立Storage V2|57PASS|今回再実行|
|既存独立API|130PASS|今回再実行|
|今回独立統合|63PASS|14修復済み偽dispatch、26契約、12component、11保存証拠|
|独立unique計|**1,038 PASS**|975＋63、FAIL/ERROR/SKIP0|
|Producer新統合module|134PASS|別集合、最終独立判定に使わない|

53 cases＋45 mutations＋追加3＝101件は保存raw単位。test数へ再加算しない。Port固有10 mutationsは全拒否。新14偽dispatchとmissing4は独立63のsubset。別scopeの新規人工3件もtest数へ足さない。rerun attemptはunique PASSへ重複加算しない。

最終配布helperで101件、63 testsを再確認。SHA/case/process並列と負荷確認は環境資料参照。Producer134は固定test SOURCEのまま別processで実行。基点内旧test/Source/pendingを新しい期待値へ変更していない。

[集計](regression_summary.json)、[新63 logs](regressions/new63/summary.json)、[Producer134](regressions/producer134.json)、[新規3case](fresh_artificial_verification.json)、[local archive](local_artificial_raw_manifest.json)。

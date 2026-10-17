# Track A targeted parallel validation: metadata preparation only

2026-10-18 JST。計画SOURCE `daf701d8e4fb25a22e5c383202f535f9db7a6e25`。
科学実行0、有効grant0、production adapter未seal。mandatory STOP。

- `campaign_plan_UNAPPROVED_v1.json`: 6系列/15個別承認単位の固定案。実行許可なし。
- `scheduling_validation_v1.json`: 計画量と依存関係。現在のfrontierは空。
- `grants_INVALID_v1.json`: 全scope未承認、SOURCE/command/grant_idはnull。
- `source_closure_v1.json`: 計画コード3 fileのSOURCE/SHA256。
- `reuse_audit_v1.json`: 18既知fileの固定Git blob/AST監査。科学再実行なし。
- `tests_receipt_v1.json`: author自身の人工テスト。実データ計算・性能試験ではない。
- `preservation_receipt_v1.json`: 起点の結果cloneのHEAD/status確認。private全件の新規byte監査とはしない。
- `inventory_v1.json`: 公開bundleのhash/size（自己fileは除外）。

[具体的な範囲・依存・高速化・未完成部分](../../../../docs/research/track_a_targeted_parallel_validation_plan_20261018.md)。
compile上限422軸は将来の全系列を合わせた提案で、control probeは別・未確定。
実測費用、モデル推定、有限標本、経験的allowanceを区別し、厳密bias/母平均/最適性は保証しない。

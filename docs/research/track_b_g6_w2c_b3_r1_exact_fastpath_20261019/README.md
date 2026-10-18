# R1 exact fastpath：非採択の技術候補

2026-10-19 JST。**PROTOTYPE_AND_ARTIFICIAL_VALIDATION_COMPLETE_NOT_ADOPTED**。
Python fused canonicalをdriverへ最小差分で接続し、冪漸化式は固定F3を変更しないartificial-only moduleへ実装。
既存275＋新64＝**339 PASS、FAIL/ERROR/SKIP各0**。実G7 sampling/correctness、実F3 session、実authority/claimは0。

直接基点 `564409487a35e15d73f1d0f0c2611824b400ab02`、専用branch `track-b-g6-w2c-b3-r1-exact-fastpath-integration-20261019`。
サーバー公開 `75ba5b944c56669a2136393924a8bb8d22e40fd1` のbundle SHA/verify後に測定revisionを一時Gitへ回収し、正しいPython候補を参考にした。
他branchのmerge/rebaseなし。R1採択記録・旧成果/契約/結果/STOPは不変。正式science release・最終contract digestは作成していない。

- [SOURCE差分と接続](source_diff_and_integration.md)
- [canonical exact互換性](canonical_optimization.md)
- [angle漸化式・本番接続境界](angle_recurrence.md)
- [339件の人工回帰](artificial_regression_results.md)／[machine結果](artificial_test_results.json)
- [ローカル性能比較](performance_comparison.md)／[結果](performance_results.json)／[全測定metadata](benchmark_results.json.gz)
- [A/B/C分類・採択](r1_compatibility_and_adoption.md)／[JSON](r1_compatibility.json)
- [Q1–Q8・次の担当](handoff.md)
- [SOURCE inventory](source_inventory.json)／[保全](verification.json)／[公開receipt](publication_receipt.json)
- [読み取り検証器](verify_fastpath.py)／[bundle検証log](bundle_verify.log)

研究対象の固定条件はG7 P5_general_order、p=(1/5,3/10,1/2)、x5/7、m5、σ+1、4方式×A/B×X/Y、A6 math-only。
分子/geometry/basis/DF rank/split L_D/δ窓は非該当。既存seed、N、confidence、31,710,848将来call予算、保存/二pass/one-shot/STOPは変更していない。
測定は人工比率/endpointだけ。原F3の純粋なprecision/root/Evidence関数は人工入力でのみ技術検証した。
科学的全domain同値性・実CPU優位・本番時間の保証ではない。公開・保全後 **mandatory STOP**。

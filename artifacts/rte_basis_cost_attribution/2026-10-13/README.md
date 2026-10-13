# 保存H4 RTEの基底費用診断

[結果と注意点](../../../docs/diagnostics/rte_basis_cost_attribution_20261013.md)／[最終summary](saved_audit_v1/summary.json)／[CSV](saved_audit_v1/resource_rows.csv)／[evidence manifest](evidence_manifest.json)。

- `run_plan.json`：結果前の8列選択、compiler、資源上限。
- `result_v1/`：原sourceの24 compile、全raw／native IR、入力identity、strict phase検査の停止。元のclassificationは不変。
- `saved_audit_v1/`：追加compileなしの費用再計数と全演算子の共通位相照合。最適化後の厳密費用内訳・母平均ではない。
- `focused_tests*.xml`、`saved_phase_tests.xml`：限定テストの最終4＋3 PASSと初回fixture失敗記録。
- process receipt／log：最初のexit 1とsaved auditのexit 0を区別する。

基底割合は加法的RZ84.8266%／CX68.9655%。新T合成、分子生成、新seed、新手法採択なし。旧source／result／marker／STOP保持。ここでSTOP。

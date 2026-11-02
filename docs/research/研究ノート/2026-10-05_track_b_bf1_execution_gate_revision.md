# Track B BF-1 execution gate revision — 2026-10-05

利用者はcommit `2d0ddafa95aefb272a36a13f259ee0b45bcc80ef`をreviewし、
`PROCEED_BF1_AFTER_MINIMAL_EXECUTION_GATE_REVISION`とした。科学実行承認ではない。
修正はpost-search cross-objective採点とauthorization publicationの二点に限定する。

[v2 amendment](../../tracks/algorithm_codesign/bf1_execution_gate_revision_v2.md)を規範とする。
cross-scoreは全O/L/F/fixed係数のprimary finite再採点後、cache-onlyで三objectiveの値・順位・
選択parameterを保存する。探索とprimary分類へ戻さず、追加係数・signal・cellを取得しない。
F winnerがLでも好まれる場合は、探索到達・budget差の説明を除外できないと記録する。

authorizationはB案を採用し、修正版source commitを唯一の親とするauthorization-only commitを検査する。
working/committed auth JSON一致と、二つの固定authorization pathの追加だけを許す。
HEAD==source_commitの旧guardを修正し、source SHAとauthorization SHAを別記録する。

旧準備artifactを上書きせずv2 packetを作る。domainは既存v1 JSONとそのhashを参照する。
必要なsource/numerical review、別実行承認、一回実行後のmandatory STOPは維持する。
研究Bの全面再設計、別geometry/family/precision、science、NPZ操作、trajectory/compile/GPUは行わない。

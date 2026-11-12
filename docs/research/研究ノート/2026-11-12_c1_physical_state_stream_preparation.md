# C1の物理状態閉包とStreaming Sの準備（2026-11-12）

前段のMeanKernelは固定した有限状態の人工modelに対する期待値代数だった。今回はC1実SOURCEを調べ、固定数のregisterと小さな状態閉包を区別した。basis/非ゼロflagだけの縮約、周期角の同一視、途中flushは一般に元のliteral Sを保たない人工例を確認した。実H4/H6の合法supportの到達性・有効な別の縮約は未確認であり、全event生成なしの実平均Sは未接続である。

元Accountingとstructural loopを保持し、tailを順に渡すconsumerだけを追加した。新9件と既存3＋sampler toy1件がPASS。これは一path Sのメモリ保持構造の技術確認で、物理平均や実compiler精度の確認ではない。既存C1係数・資格、前段kernel/providerを変更せず、A/Bの作業領域へ介入していない。

次の最小科学案は保存4path index0のX/Y計8件を新streamで評価し保存Sと照合するS0a。新sampling/compile/fit/N/Gは0の案だが、reader/validator SOURCEの確定と別途の科学実行許可が必要。今回は未実行。旧32path S0/40・16wrapper C2等へ自動拡張しない。

[REPORT](../c1_physical_state_stream_preparation_20261112/REPORT.md)／[HANDOFF](../c1_physical_state_stream_preparation_20261112/HANDOFF.md)／[科学案](../c1_physical_state_stream_preparation_20261112/SCIENCE_PLAN.json)。commit/push・remote照合後mandatory STOP。次はユーザーが科学範囲を判断し、Cが必要leaf/SOURCE、Bが最終統合を担当する。

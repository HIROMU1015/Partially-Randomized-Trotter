# 2026-10-10 Track B G10：RSS原因のread-only技術調査

固定結果R `e429c99d77b3222c5cca62750b2d111f87e4cb50` から独立branch `track-b-g10-rss-cause-audit-20261010` を作成し、保存記録と固定sourceを静的に照合した。
共通RSS guardの明示例外が直接停止理由。全17 lower/protected_after保存からlate completionまで到達。
exact throw/allocation位置は未確定だが、serial全体copyとCPython JSON chunk list/joinが最有力の増幅経路。

同じ保存JSONだけの3独立I/O検算（census/materialized/stream）で、materialized peak508.90625 MiB、
stream peak253.25 MiB、双方原bytes/hash一致。元live heapの再現・G10科学成功の証拠ではない。
Fraction/tuple人工fixtureと非string-key差分も確認。typed streaming/lifetime cleanupと
guarded finalization/bounded failure receiptを次のsource preparation候補へ整理した。

[技術報告](../../tracks/algorithm_codesign/g10_rss_failure_technical_investigation_20261010.md)、
[technical findings](../../../artifacts/track_b_g10_rss_cause_audit/2026-10-10/v1/technical_findings_v1.json)。
source/caps/科学条件/原結果・監査・marker・STOPは不変。science/synthesis/matrix/sampling/LP/runner再実行0。
production修正と次の実行scopeは未認可、GPT判断へ返しmandatory STOP。
